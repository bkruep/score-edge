"""Rate-limit-aware, disk-persisted market fetching across many leagues.

The free tier allows only ~12 requests/minute, so the rule is simple: never
issue a request unless it's actually needed, and never let a throttled league
blank a board that already has good data cached.

Two stores back the system:
  * odds.py owns the per-league slate/prop cache (short TTL, whole-board)
  * this module owns the price-history series (long TTL, per-quote) that makes
    line-movement forensics possible across days

Leagues are refreshed in a staggered rotation rather than all at once, so a
board refresh costs at most a couple of requests and the five-league set
self-heals over a few minutes instead of tripping the limiter.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import edge
import odds

# Leagues we actively keep hot. College boards are intentionally absent: they're
# kept alive by their existing cache but never enter the refresh rotation.
PRIORITY_LEAGUES = ("nfl", "mlb", "nba", "nhl", "mls")

HISTORY_PATH = Path(__file__).with_name("price_history.json")
HISTORY_TTL = 6 * 3600          # hours to keep a series before trimming
HISTORY_SAVE_EVERY = 300        # seconds between disk writes
HISTORY_MIN_POINTS = 2          # a series needs 2+ points to show any move

_lock = threading.Lock()
_history: edge.PriceHistory | None = None
_last_save: float = 0.0
_last_error: dict[str, str] = {}


# --------------------------------------------------------------- persistence


def _load_history() -> edge.PriceHistory:
    """Read the persisted series off disk once, tolerating a corrupt file."""
    global _history
    if _history is not None:
        return _history
    hist = edge.PriceHistory()
    try:
        if HISTORY_PATH.exists():
            raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            now = time.time()
            for key, points in (raw.get("series") or {}).items():
                clean = [(float(p[0]), int(p[1])) for p in points
                         if isinstance(p, (list, tuple)) and len(p) == 2 and now - float(p[0]) < HISTORY_TTL]
                if clean:
                    hist.series[key] = [edge.PricePoint(ts, price) for ts, price in clean]
    except (OSError, ValueError, TypeError, IndexError):
        hist = edge.PriceHistory()
    _history = hist
    return _history


def save_history(force: bool = False) -> bool:
    """Persist the series, rate-limited so refreshes don't thrash the disk."""
    global _last_save
    hist = _load_history()
    now = time.time()
    if not force and now - _last_save < HISTORY_SAVE_EVERY:
        return False
    payload = {
        "saved_at": now,
        "series": {k: [[round(p.ts, 3), p.price] for p in v] for k, v in hist.series.items() if v},
    }
    try:
        tmp = HISTORY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(HISTORY_PATH)   # atomic swap so a crash can't truncate the file
        _last_save = now
        return True
    except OSError:
        return False


def history_stats() -> dict:
    """Coverage stats for the movement layer — shown in the UI."""
    hist = _load_history()
    with_series = sum(1 for v in hist.series.values() if len(v) >= HISTORY_MIN_POINTS)
    return {
        "tracked": len(hist.series),
        "with_movement": with_series,
        "points": sum(len(v) for v in hist.series.values()),
        "saved_at": _last_save,
    }


# ------------------------------------------------------------------ recording


def record_quotes(quotes) -> int:
    """Fold a slate's quotes into the price series. Returns points added."""
    hist = _load_history()
    now = time.time()
    added = 0
    with _lock:
        for q in quotes or []:
            if not getattr(q, "best_odds", None):
                continue
            key = edge.quote_key(q.event_id, q.market, q.selection, q.line, q.player_name)
            before = len(hist.get(key))
            hist.record(key, int(q.best_odds), now)
            if len(hist.get(key)) > before:
                added += 1
        hist.trim()
    return added


def movement_for(quote) -> edge.Movement | None:
    """Read the current move for one quote from the series."""
    hist = _load_history()
    key = edge.quote_key(quote.event_id, quote.market, quote.selection, quote.line, quote.player_name)
    return hist.movement(key, int(quote.best_odds) if quote.best_odds else None)


def movers(min_change: int = 15, limit: int = 20, league: str = "") -> list[dict]:
    """Biggest movers, optionally filtered to one league's quote keys."""
    hist = _load_history()
    out = []
    for key, mv in hist.movers(min_change=min_change, limit=limit * 4):
        if league:
            tag = f"{league}:"
            if not (key.startswith(tag) or f"|{league}|" in key):
                continue
        out.append({
            "key": key, "movement": mv,
            "label": _label_from_key(key),
            "game": _game_from_key(key),
        })
        if len(out) >= limit:
            break
    return out


def _parts(key: str) -> list[str]:
    return key.split("|")


def _label_from_key(key: str) -> str:
    parts = _parts(key)
    if len(parts) < 5:
        return key
    _, market, player, side, line = parts[0], parts[1], parts[2], parts[3], parts[4]
    who = player.title() if player else side.title()
    return f"{who} {side.title()}" + (f" {line}" if line else "")


def _game_from_key(key: str) -> str:
    parts = _parts(key)
    return parts[5] if len(parts) > 5 else ""


# ------------------------------------------------------- derived signal layer


def enrich(quotes, league: str = "") -> dict:
    """Run the full analytics pass over one league's quotes.

    Returns a dict the templates render directly: per-quote signal lookups plus
    market-level structure, sharp positioning, arbitrage, and a sized
    portfolio. Nothing here mutates the quotes themselves.
    """
    quotes = list(quotes or [])
    if not quotes:
        return _empty_signals()

    signals: dict[str, dict] = {}
    structures: list[dict] = []
    sharp_rows: list[dict] = []

    # edge.group_markets keys on (event, market, player, line) and
    # edge.collapse_sides reduces each market to its distinct outcomes — both
    # shared with the engine so the UI's numbers always match the engine's.
    by_market = edge.group_markets(quotes)

    for (event_id, market, player, line), group in by_market.items():
        sides = edge.collapse_sides(group)
        side_prices = [int(s.best_odds) for s in sides.values() if s.best_odds]
        fair_by_side: dict[str, float] = {}
        # A lone side has no sibling to devig against, so it carries no fair
        # probability — only vig and residual signals are available.
        vig_ok = False
        if len(side_prices) >= 2:
            struct = edge.analyze_market(side_prices)
            fair_by_side = dict(zip(list(sides.keys()), struct.fair))
            vig_ok = edge.MIN_PLAUSIBLE_VIG_PCT <= struct.vig_pct <= edge.MAX_PLAUSIBLE_VIG_PCT
            structures.append({
                "event": group[0].game,
                "market": market,
                "player": group[0].player_name or "",
                "line": line,
                "n_outcomes": struct.n_outcomes,
                "n_rows": len(group),
                "sides": list(sides.keys()),
                "prices": side_prices,
                "fair": [round(f, 4) for f in struct.fair],
                "vig_pct": round(struct.vig_pct, 2),
                "hold_pct": round(struct.hold_pct, 4),
                "method": struct.method,
                # When the book is incoherent the fair numbers are meaningless,
                # so the UI must not offer them as a bet.
                "usable": vig_ok,
            })

        for q in group:
            key = edge.quote_key(q.event_id, q.market, q.selection, q.line, q.player_name)
            books = q.book_odds or {}
            sig = edge.sharp_signal(books) if len(books) >= 2 else None
            residuals = edge.book_residuals(books) if len(books) >= 3 else []
            mv = movement_for(q)
            fair = fair_by_side.get(edge.side_key(q)) if vig_ok else None

            signals[key] = {
                "sharp_label": sig.label if sig else "unclear",
                "sharp_conf": sig.confidence if sig else 0.0,
                "sharp_split": round(sig.split, 4) if sig else 0.0,
                "leader_book": (sig.leader.book if sig and sig.leader else ""),
                "leader_price": (sig.leader.price if sig and sig.leader else None),
                "n_outliers": sig.n_outliers if sig else 0,
                "top_residuals": [
                    {"book": r.book, "tier": r.tier, "price": r.price,
                     "residual_pts": round(r.residual * 100, 2)}
                    for r in residuals[:3]
                ],
                "fair_prob": round(fair, 4) if fair is not None else None,
                "ev_pct": round(edge.expected_value(q.best_odds, fair) * 100, 2) if fair is not None else None,
                "kelly_pct": round(edge.kelly_fraction(q.best_odds, fair) * 100, 2) if fair is not None else None,
                "movement": mv,
                "n_books": len(books),
            }

    arbs = edge.find_arbitrage(edge.complement_pairs(quotes))
    issues = edge.internal_consistency(quotes)

    positions = []
    for q in quotes:
        sig = signals.get(edge.quote_key(q.event_id, q.market, q.selection, q.line, q.player_name)) or {}
        pos = edge.position_from_quote(q, fair_prob=sig.get("fair_prob"))
        if pos and pos.edge > 0:
            positions.append(pos)
    portfolio = edge.build_portfolio(positions, bankroll=100.0)

    for q in quotes:
        sig = signals.get(edge.quote_key(q.event_id, q.market, q.selection, q.line, q.player_name)) or {}
        for r in (sig.get("top_residuals") or []):
            sharp_rows.append({**r, "game": q.game, "market": q.market})
    sharp_rows.sort(key=lambda r: r.get("residual_pts", 0), reverse=True)

    return {
        "signals": signals,
        "structures": structures,
        "sharp_books": sharp_rows[:12],
        # Only real, stakeable arbitrage. Implausible "profits" are feed defects
        # and belong in issues, where they get read as bugs rather than money.
        "arbs": [a for a in arbs if not a.suspicious][:8],
        "suspect_arbs": [a for a in arbs if a.suspicious][:8],
        "issues": issues,
        "portfolio": portfolio,
        "portfolio_summary": edge.portfolio_summary(portfolio),
        "movers": movers(min_change=15, limit=12, league=league),
        "n_quotes": len(quotes),
        # Average only over coherent markets; letting rejected ones in would report a
        # headline hold that no bettor ever saw.
        "avg_vig": (round(sum(s["vig_pct"] for s in structures if s["usable"])
                          / max(1, sum(1 for s in structures if s["usable"])), 2)
                    if structures else 0.0),
        "n_markets": len(structures),
        "n_usable": sum(1 for s in structures if s["usable"]),
        "n_issues": len(issues),
    }


def _empty_signals() -> dict:
    return {
        "signals": {}, "structures": [], "sharp_books": [], "arbs": [],
        "suspect_arbs": [], "issues": [],
        "portfolio": {"legs": [], "staked": 0.0, "to_win": 0.0, "expected_profit": 0.0,
                      "roi": 0.0, "groups": 0, "note": "no positive-EV candidates"},
        "portfolio_summary": edge.portfolio_summary({}),
        "movers": [], "n_quotes": 0, "avg_vig": 0.0, "n_issues": 0,
        "n_markets": 0, "n_usable": 0,
    }


# --------------------------------------------------- staggered league refresh


def refresh_league(tag: str, force: bool = False) -> dict:
    """Refresh one league's slate + props and record prices into the series.

    Never raises: a rate-limited or failing league returns the last good board
    so the page still renders. Records the error for the UI to show.
    """
    if tag not in odds.SPORTS:
        return {"ok": False, "error": f"unknown league {tag}"}
    try:
        slate, err = (odds.fetch_sport_slate(tag), "")
    except odds.OddsError as e:
        _last_error[tag] = str(e)
        stale = _stale_slate(tag)
        if stale:
            return {"ok": False, "error": str(e), "stale": True}
        return {"ok": False, "error": str(e)}
    if err:
        _last_error[tag] = err

    quotes = _slate_quotes(slate)
    added = record_quotes(quotes)
    save_history()
    return {"ok": True, "quotes": len(quotes), "recorded": added,
            "errors": _last_error.get(tag, "")}


def _slate_quotes(slate: dict) -> list:
    """Every side the feed posted, not the per-game best rows the board shows.

    Collapsed boards (MLS three-way, college) hand us a single side per game,
    which has no sibling to devig against. `all_sides` is the pre-collapse
    snapshot; falling back to the visible buckets keeps older cached slates
    working.
    """
    out = []
    for bucket in ("all_sides", "moneylines", "spreads", "totals", "value_spots"):
        out.extend(slate.get(bucket) or [])
    # value_spots is a subset of the other buckets; de-dupe by identity so a
    # quote is not counted twice in vig, overround, and portfolio sizing.
    seen: set[int] = set()
    uniq = []
    for q in out:
        if id(q) in seen:
            continue
        seen.add(id(q))
        uniq.append(q)
    return uniq


def _stale_slate(tag: str) -> dict | None:
    """Last known good board for a tag, regardless of cache age."""
    with odds._disk_lock:
        disk = odds._load_disk_cache()
    _, val = _slate_cache_age(disk, tag)
    if not isinstance(val, dict):
        return None
    if val.get("moneylines") or val.get("spreads") or val.get("totals"):
        return val
    return None


_cache_lock = threading.Lock()


def league_status() -> dict:
    """Per-league health for the UI: cache age, coverage, last error."""
    with odds._disk_lock if hasattr(odds, "_disk_lock") else _null_lock():
        disk = odds._load_disk_cache()
    out = {}
    now = time.time()
    for tag in PRIORITY_LEAGUES:
        age, cached = _slate_cache_age(disk, tag)
        out[tag] = {
            "name": odds.SPORTS.get(tag, {}).get("name", tag.upper()),
            "age_seconds": age,
            "fresh": age is not None and age < odds.CACHE_TTL,
            "error": _last_error.get(tag, ""),
            "quotes": _count_quotes(cached),
        }
    return out


def _slate_cache_age(disk: dict, tag: str) -> tuple[int | None, dict | None]:
    """Freshest cached slate for a tag, matching the dated-key scheme fetch_sport_slate uses."""
    best_age = None
    best_val = None
    for key, entry in (disk or {}).items():
        if not key.startswith(f"sport_slate_{tag}"):
            continue
        try:
            ts, val = entry
        except (TypeError, ValueError):
            continue
        if not isinstance(val, dict):
            continue
        age = time.time() - ts
        if best_age is None or age < best_age:
            best_age, best_val = int(age), val
    return best_age, best_val


def _count_quotes(slate: dict | None) -> int:
    if not isinstance(slate, dict):
        return 0
    return sum(len(slate.get(b) or []) for b in ("moneylines", "spreads", "totals", "value_spots"))


class _null_lock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

API_KEY = os.getenv("ODDS_API_KEY", "").strip()

SHARPAPI_BASE_URL = "https://api.sharpapi.io/api/v1"
SHARPAPI_EVENTS_URL = f"{SHARPAPI_BASE_URL}/events"
SHARPAPI_BEST_ODDS_URL = f"{SHARPAPI_BASE_URL}/odds/best"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
}

NFL_GAME_MARKETS = ["moneyline", "point_spread", "total_points"]

NFL_PROP_MARKETS = ["player_touchdowns", "player_rushing_yards", "player_rushing_attempts",
                    "player_pass_yds", "player_pass_tds", "player_pass_attempts",
                    "player_pass_completions", "player_pass_interceptions",
                    "player_receiving_yards", "player_receptions", "player_receiving_tds"]

MARKET_LABELS = {
    "moneyline": "Moneyline",
    "point_spread": "Spread",
    "total_points": "Total",
}

PROP_LABELS = {
    "player_touchdowns": "Touchdowns",
    "player_anytime_touchdown": "Anytime TD",
    "player_rushing_yards": "Rushing Yards",
    "player_rushing_attempts": "Rushing Attempts",
    "player_pass_yds": "Passing Yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_attempts": "Pass Attempts",
    "player_pass_completions": "Pass Completions",
    "player_pass_interceptions": "Interceptions",
    "player_receiving_yards": "Receiving Yards",
    "player_receptions": "Receptions",
    "player_receiving_tds": "Receiving TDs",
}

NFL_TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 900
MAX_LINE_HISTORY = 24  # rolling per-market sample cap for the line-move charts

import pickle
import threading

_DISK_CACHE_FILE = os.path.join(os.path.dirname(__file__), "odds_cache.pkl")
_DISK_KEYS = ("slate_", "bets_active_", "props_", "line_", "bets_slate_")
_disk_lock = threading.Lock()


def _load_disk_cache() -> dict:
    try:
        with open(_DISK_CACHE_FILE, "rb") as fh:
            raw = pickle.load(fh)
        if isinstance(raw, dict):
            return raw
    except (OSError, pickle.PickleError, EOFError):
        pass
    return {}


def _save_disk_cache(snapshot: dict) -> None:
    try:
        with open(_DISK_CACHE_FILE, "wb") as fh:
            pickle.dump(snapshot, fh)
    except (OSError, pickle.PickleError):
        pass


def _cached(key: str):
    if key in _CACHE:
        ts, val = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return val
    if any(key.startswith(p) for p in _DISK_KEYS):
        with _disk_lock:
            disk = _load_disk_cache()
        entry = disk.get(key)
        if entry is not None:
            ts, val = entry
            if time.time() - ts < CACHE_TTL:
                _CACHE[key] = entry
                return val
    return None


def _cache_set(key: str, val) -> None:
    now = time.time()
    _CACHE[key] = (now, val)
    if any(key.startswith(p) for p in _DISK_KEYS):
        with _disk_lock:
            disk = _load_disk_cache()
            disk[key] = (now, val)
            _save_disk_cache(disk)


class OddsError(RuntimeError):
    """Raised when SharpAPI odds cannot be fetched."""


@dataclass(frozen=True, slots=True)
class NflEvent:
    event_id: str
    away_team: str
    home_team: str
    start_time: str

    def label(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


@dataclass(frozen=True, slots=True)
class BetQuote:
    event_id: str
    game: str
    selection: str
    market: str
    best_odds: int
    consensus_odds: int
    best_book: str
    edge_pct: float
    line: float | None = None
    player_name: str | None = None
    book_odds: dict[str, int] | None = None
    book_edges: dict[str, float] | None = None

    def book_ladder(self, top: int = 4) -> list[dict]:
        """Top books by price for this quote: [{book, am, edge}] best first."""
        if not self.book_odds:
            return []
        items = sorted(self.book_odds.items(), key=lambda kv: (kv[1], ), reverse=True)
        out = []
        for book, am in items[:top]:
            e = (self.book_edges or {}).get(book)
            out.append({"book": book, "am": am, "edge": round(e, 1) if e is not None else None})
        return out

    @property
    def is_value(self) -> bool:
        """Best price meaningfully beats consensus (real value spot)."""
        return self.edge_pct >= 1.5

    def slip_payload(self) -> dict:
        """Compact payload the tracked-slip drawer locks: m=market tag,
        g=game, s=selection, l=line, a=american price, b=book, e=edge %,
        p=player (props only)."""
        m = {"moneyline": "ML", "point_spread": "SPR", "total_points": "O/U"}\
            .get(self.market, "PROP" if self.player_name else self.market)
        return {
            "m": m, "g": self.game, "s": self.selection,
            "l": self.line, "a": self.best_odds, "b": self.best_book,
            "e": round(self.edge_pct, 2) if self.edge_pct is not None else None,
            "p": self.player_name or "",
        }


@dataclass(frozen=True, slots=True)
class PlayerProp:
    event_id: str
    game: str
    player_name: str
    market: str
    label: str
    stat_category: str
    side: str
    line: float | None
    best_odds: int
    consensus_odds: int
    best_book: str
    edge_pct: float
    book_odds: dict[str, int] | None = None

    @property
    def is_value(self) -> bool:
        return self.edge_pct >= 1.5

    def book_ladder(self, top: int = 4) -> list[dict]:
        """Top books by price for this prop: [{book, am, edge}] best first."""
        if not self.book_odds:
            return []
        items = sorted(self.book_odds.items(), key=lambda kv: (kv[1], ), reverse=True)
        out = []
        for book, am in items[:top]:
            out.append({"book": book, "am": am, "edge": None})
        return out

    def slip_payload(self) -> dict:
        """Compact tracked-slip payload (m=PROP). See BetQuote.slip_payload."""
        return {
            "m": "PROP", "g": self.game, "s": self.player_name or self.side,
            "l": self.line, "a": self.best_odds, "b": self.best_book,
            "e": round(self.edge_pct, 2) if self.edge_pct is not None else None,
            "p": self.player_name or "",
        }


def american_to_decimal(american: int) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / float(-american)


def decimal_to_american(decimal: float) -> int:
    if decimal <= 1.0:
        return 500
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def odds_html(american: int) -> str:
    if american > 0:
        return f"+{american}"
    return str(american)


def matches_team(selection: str, team_name: str) -> bool:
    if not selection or not team_name:
        return False
    selection = selection.strip()
    if selection.casefold() == team_name.casefold():
        return True
    abbreviation = NFL_TEAM_ABBREVIATIONS.get(team_name)
    if not abbreviation:
        return False
    tokens = [t.upper() for t in selection.split()]
    return any(t == abbreviation or abbreviation.startswith(t) for t in tokens)


def _canonical_team(selection: str) -> str | None:
    """Map a provider team label (abbr+mascot, full name) to the canonical full
    name. Returns None when the label can't be resolved to a real NFL team."""
    sel = (selection or "").strip()
    if not sel:
        return None
    lower = sel.casefold()
    if lower in NFL_TEAM_ABBREVIATIONS:
        return sel
    tokens = {t.upper() for t in sel.split() if t}
    for team, abbr in NFL_TEAM_ABBREVIATIONS.items():
        if abbr in tokens:
            return team
        mascot = team.split()[-1].upper()
        if mascot in tokens and mascot not in _MASCOT_AMBIGUOUS:
            return team
        team_tokens = {t.upper() for t in team.split()}
        if tokens == team_tokens:
            return team
    return None


_MASCOT_AMBIGUOUS = {"NEW", "LOS", "LAS"}


def _request_json(session: requests.Session, url: str, params: dict[str, Any], timeout: int) -> Any:
    headers = {**HEADERS, "X-API-Key": API_KEY}
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, timeout=timeout, headers=headers, params=params)
        except requests.RequestException as error:
            raise OddsError(f"Could not reach SharpAPI: {error}") from error

        if response.status_code == 429:
            if attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise OddsError("SharpAPI rate limit hit. Try again shortly.")
        if response.status_code == 401:
            raise OddsError("SharpAPI rejected the API key. Check ODDS_API_KEY.")
        if response.status_code == 403:
            raise OddsError("SharpAPI tier restriction — upgrade to unlock this data.")
        if response.status_code != 200:
            raise OddsError(f"SharpAPI returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError as error:
            raise OddsError("SharpAPI response was not valid JSON.") from error


def _parse_start_time(raw: Any) -> str:
    try:
        return (
            datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .strftime("%b %d, %I:%M %p ET")
        )
    except ValueError:
        return str(raw)


def fetch_nfl_events(date: str) -> list[NflEvent]:
    cache_key = f"events_{date}"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    payload = _request_json(session, SHARPAPI_EVENTS_URL, {
        "sport": "football", "league": "nfl", "date": date[:10], "limit": 200,
    }, timeout=20)

    events: list[NflEvent] = []
    seen_matchups: set[tuple[str, str]] = set()
    for entry in payload.get("data") or []:
        if entry.get("is_live"):
            continue
        event_id = str(entry.get("id", "")).strip()
        away = str(entry.get("away_team", "")).strip()
        home = str(entry.get("home_team", "")).strip()
        # SharpAPI's feed mixes real games with pseudo-events (alt spreads,
        # "Team goes to @ ...", "Over X @ Under Y", winning-margin markets).
        # Keep only events where both sides are real NFL teams.
        if not event_id or away not in NFL_TEAM_ABBREVIATIONS or home not in NFL_TEAM_ABBREVIATIONS:
            continue
        matchup = (home, away)
        if matchup in seen_matchups:
            continue        # dedupe mirrored/duplicate entries for the same game
        seen_matchups.add(matchup)
        events.append(NflEvent(event_id, away, home, _parse_start_time(entry.get("start_time", ""))))

    _cache_set(cache_key, events)
    return events


def _parse_best_odds_rows(payload: Any, labels: dict[str, str]) -> list[BetQuote]:
    quotes: list[BetQuote] = []
    for entry in payload.get("data") or []:
        best = entry.get("best_odds") or {}
        consensus = entry.get("consensus_odds") or {}
        if not best or not consensus:
            continue
        try:
            best_am = int(best.get("american"))
            cons_am = int(consensus.get("american"))
        except (TypeError, ValueError):
            continue
        if not best_am or not cons_am:
            continue

        market = str(entry.get("market_type", "")).strip().casefold()
        line = None
        try:
            if entry.get("line") is not None:
                line = float(entry.get("line"))
        except (TypeError, ValueError):
            line = None

        book_odds: dict[str, int] | None = None
        book_edges: dict[str, float] | None = None
        all_books = entry.get("all_books") or []
        if all_books:
            book_odds, book_edges = {}, {}
            for b in all_books:
                bname = str(b.get("book") or b.get("sportsbook") or "").strip()
                bo = b.get("odds") or {}
                try:
                    bam = int(bo.get("american"))
                except (TypeError, ValueError):
                    continue
                if not bname or not bam:
                    continue
                book_odds[bname] = bam
                try:
                    if b.get("edge") is not None:
                        book_edges[bname] = float(b.get("edge"))
                except (TypeError, ValueError):
                    pass
            if not book_odds:
                book_odds = book_edges = None

        quotes.append(BetQuote(
            event_id=str(entry.get("event_id", "")).strip(),
            game=labels.get(str(entry.get("event_id", "")).strip(),
                            str(entry.get("event_name", "")).strip() or ""),
            selection=str(entry.get("selection", "")).strip(),
            market=market,
            best_odds=best_am,
            consensus_odds=cons_am,
            best_book=str(entry.get("best_book", "")).strip(),
            edge_pct=(american_to_decimal(best_am) / american_to_decimal(cons_am) - 1.0) * 100.0,
            line=line,
            player_name=str(entry.get("player_name") or "").strip() or None,
            book_odds=book_odds,
            book_edges=book_edges,
        ))
    return quotes


def fetch_best_odds(event_ids: list[str], markets: str, labels: dict[str, str],
                    max_rows: int = 500, request_delay: float = 0.2) -> list[BetQuote]:
    if not event_ids:
        return []
    cache_key = f"odds_{','.join(event_ids[:5])}_{markets}"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    quotes: list[BetQuote] = []
    offset = 0
    while True:
        payload = _request_json(session, SHARPAPI_BEST_ODDS_URL, {
            "sport": "football", "league": "nfl",
            "event_id": ",".join(event_ids), "market": markets,
            "limit": 200, "offset": offset,
        }, timeout=20)
        page = _parse_best_odds_rows(payload, labels)
        quotes.extend(page)
        pagination = payload.get("pagination") or {}
        next_offset = pagination.get("next_offset")
        if not pagination.get("has_more") or len(quotes) >= max_rows or not next_offset:
            break
        offset = int(next_offset)
        if request_delay > 0:
            time.sleep(request_delay)

    quotes = quotes[:max_rows]
    _cache_set(cache_key, quotes)
    return quotes


def _dedupe_side_quotes(quotes: list[BetQuote], *, side_by_team: bool) -> list[BetQuote]:
    """Collapse per-team/per-side duplicates into one best row per game.

    SharpAPI posts one row per team (or per over/under side) using both the full
    team name and an abbreviated alias, and occasionally posts moved lines too.
    For moneyline/spreads (side_by_team=True) keep one row per canonical team;
    for totals keep one row per side (Over/Under). Only the main line for the
    game is kept, preferring the line closest to consensus over a stale moved one.
    """
    from collections import defaultdict

    # (event, canonical side) -> list of quotes; the main line for the game is the
    # one with the smallest |edge| (closest to consensus market).
    by_side: dict[tuple[str, str], list[BetQuote]] = defaultdict(list)
    for q in quotes:
        team = _canonical_team(q.selection) if side_by_team else (q.selection or "").strip().casefold()
        if side_by_team and not team:
            continue
        by_side[(q.event_id, team)].append(q)

    result: list[BetQuote] = []
    for (evt, side), group in by_side.items():
        group.sort(key=lambda x: (abs(x.edge_pct), x.best_odds), reverse=False)
        chosen = group[0]  # main line
        # Surface the canonical team name instead of the provider alias.
        canonical = _canonical_team(side) if side_by_team else side
        if side_by_team and canonical and canonical.casefold() != chosen.selection.casefold():
            chosen = BetQuote(
                chosen.event_id, chosen.game, canonical, chosen.market,
                chosen.best_odds, chosen.consensus_odds, chosen.best_book,
                chosen.edge_pct, chosen.line, chosen.player_name,
                chosen.book_odds, chosen.book_edges,
            )
        result.append(chosen)
    return result


def _last_good_slate(cache_key: str) -> dict | None:
    """Best cached slate for a date regardless of age.

    Used as a fallback when SharpAPI is rate-limited or returns no main market
    lines for a day that previously carried them. Events can exist while the
    best-odds fetch is throttled (silently returning no quotes); preferring the
    last known lines keeps the board populated instead of blanking that day.
    """
    with _disk_lock:
        disk = _load_disk_cache()
    entry = disk.get(cache_key)
    if entry is None:
        return None
    _, val = entry
    if not isinstance(val, dict):
        return None
    if val.get("moneylines") or val.get("spreads") or val.get("totals"):
        return val
    return None


def fetch_edge_slate(date: str) -> dict:
    """Fetch best moneyline/spread/total odds for a date across books."""
    cache_key = f"slate_{date}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        events = fetch_nfl_events(date)
        labels = {e.event_id: e.label() for e in events}
        event_ids = [e.event_id for e in events]
        quotes = fetch_best_odds(event_ids, ",".join(NFL_GAME_MARKETS), labels) if event_ids else []

        props = fetch_player_props(labels)
    except OddsError:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            _cache_set(cache_key, stale)
            return stale
        raise

    moneylines = _dedupe_side_quotes([q for q in quotes if q.market == "moneyline"], side_by_team=True)
    spreads = _dedupe_side_quotes([q for q in quotes if q.market == "point_spread"], side_by_team=True)
    totals = _dedupe_side_quotes([q for q in quotes if q.market == "total_points"], side_by_team=False)

    # Totals: show only the best side per game (the stronger edge), never both
    # Over and Under — betting both sides of one line is a coin flip, not a bet.
    totals_by_game: dict[str, BetQuote] = {}
    for q in totals:
        cur = totals_by_game.get(q.event_id)
        if cur is None or q.edge_pct > cur.edge_pct:
            totals_by_game[q.event_id] = q
    totals = list(totals_by_game.values())

    value_spots = build_value_spots(moneylines, spreads, totals)

    slate = {
        "events": events,
        "moneylines": moneylines,
        "spreads": spreads,
        "totals": totals,
        "props": props,
        "value_spots": value_spots,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    # Only cache slates that actually carried lines. Events can exist even if
    # the best-odds fetch was rate-limited (silently returning no quotes), and
    # caching that would poison the hot path for the TTL window.
    if moneylines or spreads or totals:
        _cache_set(cache_key, slate)
    else:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            return stale
    record_line_baselines(quotes)
    record_line_history(moneylines + spreads + totals)
    log_slate_snapshot(slate, "fetch")
    try:
        import journal
        journal.record_slate(slate, date)
    except Exception:
        pass
    return slate


def build_value_spots(moneylines: list[BetQuote], spreads: list[BetQuote],
                      totals: list[BetQuote]) -> list[BetQuote]:
    """One value row per market per game.

    Never show both sides of the same bet (e.g. Over AND Under on one total) —
    that's a coin flip, not a value signal. For totals keep only the stronger
    side; for moneylines and spreads each side is its own fair price, but only
    if it independently beats consensus.
    """
    def _best_side(quotes: list[BetQuote]) -> list[BetQuote]:
        by_key: dict[tuple, BetQuote] = {}
        for q in quotes:
            key = (q.event_id, q.market, q.selection.casefold())
            cur = by_key.get(key)
            if cur is None or abs(q.edge_pct) > abs(cur.edge_pct):
                by_key[key] = q
        return list(by_key.values())

    total_side_keys: dict[tuple, BetQuote] = {}
    for q in totals:
        line_key = (q.event_id, q.line)
        cur = total_side_keys.get(line_key)
        if not q.is_value:
            continue
        if cur is None or q.edge_pct > cur.edge_pct:
            total_side_keys[line_key] = q

    game_quotes = _best_side(moneylines) + _best_side(spreads) + list(total_side_keys.values())
    return sorted(
        [q for q in game_quotes if q.is_value],
        key=lambda q: q.edge_pct, reverse=True,
    )


def _line_steam_key(q) -> tuple:
    return (q.event_id, q.market, (q.selection or "").casefold())


def _load_line_hist() -> dict:
    """Main-line movement series, read from disk directly (ignores the TTL —
    history should survive idle gaps, unlike the ephemeral line snapshot)."""
    with _disk_lock:
        disk = _load_disk_cache()
    entry = disk.get("line_hist")
    if entry is None:
        return {}
    _, val = entry
    return val if isinstance(val, dict) else {}


def record_line_history(quotes: list[BetQuote]) -> None:
    """Append a snapshot of each main spread/total line to a rolling series.

    SharpAPI serves a snapshot, not an opening-line endpoint, so the closest
    honest "line move" chart samples the main line on every fresh fetch and
    diffs against the first-seen value (the baseline). Unchanged samples are
    skipped to avoid bloat and the series is capped per market; the disk cache
    persists it across restarts so overnight/weekly moves are still charted.
    """
    hist = _load_line_hist()
    changed = False
    for q in quotes:
        if q.line is None or q.market not in ("point_spread", "total_points"):
            continue
        key = _line_steam_key(q)
        rec = hist.get(key)
        if rec is None:
            hist[key] = {
                "game": q.game or "",
                "market": q.market,
                "side": (q.selection or ""),
                "pts": [{"t": time.time(), "line": q.line}],
            }
            changed = True
            continue
        if rec["pts"][-1]["line"] == q.line:
            continue
        rec["pts"].append({"t": time.time(), "line": q.line})
        rec["pts"] = rec["pts"][-MAX_LINE_HISTORY:]
        hist[key] = rec
        changed = True
    if changed:
        _cache_set("line_hist", hist)


def line_series() -> list[dict]:
    """Main-line movement series for the steam charts, movers first."""
    hist = _load_line_hist()
    out = []
    for key, rec in hist.items():
        pts = rec.get("pts") or []
        if not pts:
            continue
        out.append({
            "key": key,
            "game": rec.get("game", ""),
            "market": rec.get("market", ""),
            "side": rec.get("side", ""),
            "base": pts[0]["line"],
            "pts": [{"t": p["t"], "line": p["line"]} for p in pts],
        })
    out.sort(
        key=lambda r: (len(r["pts"]) > 1, abs(r["pts"][-1]["line"] - r["base"])),
        reverse=True,
    )
    return out


def record_line_baselines(quotes: list[BetQuote]) -> None:
    """Persist the first-seen main line per (event, market, side) as a baseline.

    SharpAPI only serves a current snapshot — there is no opening line — so the
    first fetch seeds the reference and later fetches reveal movement against it.
    Baselines survive restarts via the disk cache, so a line that moved overnight
    (or week-to-week on a Thursday/Sunday slate) is still detected.
    """
    snap = _cached("line_snap") or {}
    changed = False
    for q in quotes:
        if q.line is None or q.market not in ("point_spread", "total_points"):
            continue
        k = _line_steam_key(q)
        if k not in snap:
            snap[k] = {"line": q.line, "ts": time.time()}
            changed = True
    if changed:
        _cache_set("line_snap", snap)


def line_moves(quotes: list[BetQuote], min_delta: float = 1.5) -> dict[tuple, dict]:
    """Movement vs each line's first-seen baseline (steam-move signal)."""
    snap = _cached("line_snap") or {}
    out: dict[tuple, dict] = {}
    for q in quotes:
        if q.line is None or q.market not in ("point_spread", "total_points"):
            continue
        k = _line_steam_key(q)
        base = snap.get(k)
        if not base:
            continue
        delta = abs(q.line - base["line"])
        if delta >= min_delta:
            out[k] = {"from": base["line"], "to": q.line, "delta": round(delta, 1)}
    return out


def steam(q, moves: dict[tuple, dict]):
    """Steam-move info for a quote against the moves map, or None."""
    m = moves.get(_line_steam_key(q))
    if not m:
        return None
    return {"from": m["from"], "to": m["to"], "delta": m["delta"],
            "up": m["to"] > m["from"]}


# ---------------------------------------------------------------------------
# Snapshot history — a compact KPI time series so the dashboard can draw real
# trend sparklines (edge counts, avg edge, board size) as snapshots accrue.
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_SNAPSHOT_DIR, exist_ok=True)
_SNAPSHOT_FILE = os.path.join(_SNAPSHOT_DIR, "nfl_odds_history.jsonl")
_snapshot_lock = threading.Lock()
_last_snapshot = {"ts": 0.0, "sig": ""}


def _quote_list(slate: dict) -> list:
    out: list = []
    for key in ("moneylines", "spreads", "totals", "props"):
        out.extend(slate.get(key) or [])
    return out


def log_slate_snapshot(slate: dict, source: str = "fetch") -> None:
    """Append one compact record per observed slate so trends build over time.

    Cheap (single json line) and idempotent — identical observations within 60s
    are skipped, and any I/O failure is swallowed so logging never breaks the
    page that triggered it.
    """
    global _last_snapshot
    if not isinstance(slate, dict):
        return
    try:
        quotes = _quote_list(slate)
        value = [q for q in quotes if (getattr(q, "edge_pct", 0) or 0) >= 1.5]
        top = 0.0
        if value:
            top = round(max(q.edge_pct for q in value), 1)
        rec = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).strftime("%H:%M"),
            "games": len(slate.get("events") or []),
            "pieces": len(quotes),
            "value": len(value),
            "avg_edge": round(sum(q.edge_pct for q in value) / len(value), 1) if value else 0.0,
            "top_edge": top,
            "source": source,
        }
        sig = f"{rec['games']}:{rec['value']}:{rec['top_edge']}:{rec['avg_edge']}"
        now = rec["ts"]
        if sig == _last_snapshot["sig"] and now - _last_snapshot["ts"] < 60:
            return
        _last_snapshot = {"ts": now, "sig": sig}
        with _snapshot_lock:
            with open(_SNAPSHOT_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def snapshot_history(max_age_s: float = 36 * 3600) -> list[dict]:
    """Read back snapshot records (oldest→newest) within max_age_s seconds."""
    try:
        out: list[dict] = []
        with open(_SNAPSHOT_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if time.time() - rec.get("ts", 0) <= max_age_s:
                        out.append(rec)
                except (ValueError, TypeError):
                    continue
        out.sort(key=lambda r: r.get("ts", 0))
        return out
    except Exception:
        return []


_TEAM_SLUGS: dict[str, str] = {}


def _team_from_slug(slug: str) -> str | None:
    """Map a lowercase event-id token (e.g. '49ers', 'rams') to the canonical
    full team name."""
    if not _TEAM_SLUGS:
        for name in NFL_TEAM_ABBREVIATIONS:
            mascot = name.split()[-1].casefold().replace(" ", "")
            _TEAM_SLUGS[mascot] = name
            _TEAM_SLUGS[name.casefold().replace(" ", "")] = name
    key = slug.casefold().replace(" ", "")
    return _TEAM_SLUGS.get(key)


def _game_label_from_event_id(event_id: str) -> str:
    """Resolve a game label from a SharpAPI event id ('nfl_49ers_rams_...').

    The props feed carries events across the week (TNF/SNF), so the slate's label
    map misses them. Parse the two team slug tokens out of the event id instead.
    """
    if not event_id:
        return ""
    parts = event_id.split("_")
    if len(parts) < 4 or not parts[0] == "nfl":
        return ""
    away = _team_from_slug(parts[1])
    home = _team_from_slug(parts[2])
    if away and home:
        return f"{away} @ {home}"
    return ""


def _human_prop_label(stat_category: str, market: str) -> str:
    if market == "player_touchdowns":
        return "Anytime Touchdown"
    if market in PROP_LABELS:
        return PROP_LABELS[market]
    if stat_category:
        return stat_category.replace("_", " ").title()
    return market.replace("player_", "").replace("_", " ").title()


def _group_props(raw_rows: list[dict], labels: dict[str, str]) -> list[PlayerProp]:
    """Aggregate raw per-book prop odds into best-price rows with edge, keyed by
    (player, market_type, selection_type, line) per SharpAPI canonical spec."""
    groups: dict[tuple, dict] = {}
    for row in raw_rows:
        player = str(row.get("player_name") or "").strip()
        market = str(row.get("market_type") or "").strip().casefold()
        if not player or not market:
            continue
        try:
            american = int(row.get("odds_american"))
        except (TypeError, ValueError):
            continue
        if not american:
            continue
        decimal = american_to_decimal(american)
        line = None
        try:
            if row.get("line") is not None and row.get("line") != "":
                line = float(row.get("line"))
        except (TypeError, ValueError):
            line = None
        if line is None:
            # Skip degenerate bins (yes/no combo rows, "2+ rec" longshots) that
            # produce meaningless edges — the canonical prop has a threshold line.
            continue
        side = str(row.get("selection_type") or row.get("selection") or "").strip()
        key = (player.casefold(), market, str(side).casefold(), line)
        group = groups.setdefault(key, {
            "event_id": str(row.get("event_id") or "").strip(),
            "game": "",
            "player": player,
            "market": market,
            "label": _human_prop_label(str(row.get("stat_category") or ""), market),
            "stat_category": str(row.get("stat_category") or "").strip(),
            "side": side,
            "line": line,
            "best_decimal": 0.0,
            "best_american": 0,
            "best_book": "",
            "decimals": [],
            "books": [],
            "book_odds": {},
            "count": 0,
        })
        decimal_sum = group["best_decimal"]
        if decimal > group["best_decimal"]:
            group["best_decimal"] = decimal
            group["best_american"] = american
            group["best_book"] = str(row.get("sportsbook") or "").strip()
        book_name = str(row.get("sportsbook") or "").strip()
        if book_name:
            group["book_odds"][book_name] = american
        if not group["game"]:
            group["game"] = (
                labels.get(group["event_id"])
                or _game_label_from_event_id(group["event_id"])
                or str(row.get("event_name") or "").strip()
            )
        group["decimals"].append(decimal)
        if row.get("sportsbook"):
            group["books"].append(str(row.get("sportsbook")))
        group["count"] += 1

    props: list[PlayerProp] = []
    for g in groups.values():
        if not g["decimals"]:
            continue
        consensus = sum(g["decimals"]) / len(g["decimals"]) if g["decimals"] else 0.0
        edge = (g["best_decimal"] / consensus - 1.0) * 100.0 if consensus else 0.0
        cons_am = decimal_to_american(consensus) if consensus else g["best_american"]
        props.append(PlayerProp(
            event_id=g["event_id"], game=g["game"], player_name=g["player"],
            market=g["market"], label=g["label"], stat_category=g["stat_category"],
            side=g["side"], line=g["line"],
            best_odds=g["best_american"], consensus_odds=cons_am,
            best_book=g["best_book"], edge_pct=round(edge, 2),
            book_odds=(g["book_odds"] or None),
        ))

    # Collapse over/under duplicates: keep a single row per (player, market,
    # line), preferring the higher-edge side so the board isn't full of mirrors.
    by_line: dict[tuple, PlayerProp] = {}
    for p in props:
        key = (p.player_name.casefold(), p.market.casefold(),
               None if p.line is None else round(float(p.line), 4))
        cur = by_line.get(key)
        if cur is None or p.edge_pct > cur.edge_pct:
            by_line[key] = p
    props = sorted(by_line.values(), key=lambda p: p.edge_pct, reverse=True)
    return props


def fetch_player_props(labels: dict[str, str], max_rows: int = 800,
                       request_delay: float = 0.25) -> list[PlayerProp]:
    """Fetch NFL player props across books and aggregate best-price + edge.

    Uses the raw /odds endpoint (which carries player identity) filtered to the
    `props` alias. Props are sparse until books release the full weekly slate
    (typically Tue-Wed before games); returns [] when none are posted yet."""
    cache_key = "props_all"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    raw_rows: list[dict] = []
    offset = 0
    page_error = False
    while True:
        try:
            payload = _request_json(session, f"{SHARPAPI_BASE_URL}/odds", {
                "sport": "football", "league": "nfl", "market": "props",
                "is_main_line": "true", "limit": 200, "offset": offset,
            }, timeout=25)
        except OddsError:
            # Don't cache a partial board: surface no-props so the next request
            # retries the full pagination instead of freezing a truncated set.
            if raw_rows:
                page_error = True
            break
        raw_rows.extend(payload.get("data") or [])
        pagination = payload.get("pagination") or {}
        next_offset = pagination.get("next_offset")
        if not pagination.get("has_more") or len(raw_rows) >= max_rows or not next_offset:
            break
        offset = int(next_offset)
        if request_delay > 0:
            time.sleep(request_delay)

    props = _group_props(raw_rows[:max_rows], labels)
    if raw_rows and not page_error:
        _cache_set(cache_key, props)
    return props

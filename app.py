from __future__ import annotations
import hashlib, os, re, math
import concurrent.futures as _cf
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import data
from odds import BetQuote
import odds
import edge
import marketfeed
import draft
import dfs
import td_promo
import journal
import toolbox
import golf
import hockey
import baseball
import racing

app = FastAPI(title="ScoreEdge")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def _sigkey(value) -> str:
    """Map a BetQuote to its analytics-signal key (mirrors marketfeed.enrich)."""
    try:
        return edge.quote_key(value.event_id, value.market,
                              value.selection or "", value.line, value.player_name)
    except Exception:
        return ""


def _qtype(q) -> str:
    """Categorize a quote into the dashboard's market filter buckets."""
    m = (getattr(q, "market", "") or "").lower()
    if getattr(q, "player_name", None):
        return "prop"
    if m == "moneyline":
        return "ml"
    if "spread" in m or m in ("puck_line", "spread_total"):
        return "spr"
    if "total" in m or m in ("ou_points", "over_under", "o_u"):
        return "tot"
    if m.startswith("player"):
        return "prop"
    return "ml"


templates.env.filters["sigkey"] = _sigkey
templates.env.filters["qtype"] = _qtype

CACHE_BUST = hashlib.md5(str(datetime.now().hour).encode()).hexdigest()[:8]
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF", "DST"}

SPORTS = {
    "nfl": {"name": "NFL", "href": "/", "icon": "🏈"},
    "nba": {"name": "NBA", "href": "/nba", "icon": "🏀"},
    "cfb": {"name": "CFB", "href": "/cfb", "icon": "🏈"},
    "cbb": {"name": "CBB", "href": "/cbb", "icon": "🏀"},
    "mlb": {"name": "MLB", "href": "/mlb", "icon": "⚾"},
    "nhl": {"name": "NHL", "href": "/nhl", "icon": "🏒"},
    "mls": {"name": "MLS", "href": "/mls", "icon": "⚽"},
}

app.state.cache_bust = CACHE_BUST

@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif not path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

_data_loaded = False

def _get_season() -> int:
    # NFL season year = the calendar year the season begins. A fantasy/NFL
    # season spans Aug-Dec (current year) through Jan-Feb (of the next year).
    # e.g. today 2026-08-31 -> the 2026-27 season -> returns 2026.
    now = datetime.now()
    if now.month >= 8:
        return now.year
    # Jan-Jul: season began in the previous calendar year.
    return now.year - 1

def _season_display(season: int) -> str:
    # The 2026-27 NFL season is commonly referred to as the "2026 season".
    return f"{season}"

_in_season_memo = {"at": None, "value": False}


def _is_nfl_regular_season() -> bool:
    now = datetime.now()
    last = _in_season_memo.get("at")
    if last is not None and (now - last).total_seconds() < 3600:
        return _in_season_memo["value"]
    # Draft mode stays active until the regular-season opener has actually been
    # played (the Thursday night Week 1 game). Anchor on the real schedule when
    # it is available instead of a calendar heuristic.
    try:
        sched = _bounded(data.load_schedule, 6.0, _get_season())
        if not sched.empty and "gameday" in sched.columns:
            gamedays = pd.to_datetime(sched["gameday"], errors="coerce").dropna()
            if not gamedays.empty:
                first_game = gamedays.min().to_pydatetime().replace(tzinfo=None)
                _in_season_memo.update(at=now, value=now >= first_game)
                return _in_season_memo["value"]
    except Exception:
        pass
    # Fallback (no schedule data): historical calendar heuristic, but treat the
    # entire month of September as draft season since the opener lands ~Sep 10.
    month, day = now.month, now.day
    in_season = bool(month in (10, 11, 12) or month == 1 or (month == 2 and day <= 14))
    _in_season_memo.update(at=now, value=in_season)
    return in_season

def double_value_players(slate: dict) -> set:
    """Return player names that are both DFS value plays and betting prop edges.

    Only meaningful when live DraftKings DFS data is available (DFS_DATA_SOURCE=dk).
    When DFS is disabled this returns an empty set so the public site shows
    nothing, while personal/local deployments get the cross-tab flag.
    """
    try:
        if dfs.data_source() != "dk":
            return set()
        dk = dfs.fetch_slate()
    except Exception:
        return set()
    dk_players = dk.get("players", [])
    if not dk_players:
        return set()
    # DFS value plays: top pts-per-$1k players (salary-aware bargains).
    ranked = sorted(dk_players, key=lambda p: p.get("value_per_1k", 0.0), reverse=True)
    top_n = ranked[:40]
    dfs_names = {str(p.get("name", "")).strip().casefold() for p in top_n if p.get("name")}

    # Betting prop edges reference a player name (non-None) with real edge.
    prop_names: set[str] = set()
    for p in slate.get("props", []) or []:
        nm = getattr(p, "player_name", None) or ""
        edge = getattr(p, "edge_pct", 0.0) or 0.0
        if nm and edge >= 1.5:
            prop_names.add(str(nm).strip().casefold())

    return {str(p).strip() for p in top_n
            if str(p.get("name", "")).strip().casefold() in prop_names}


def _prop_to_quote(p) -> BetQuote:
    """Promote a PlayerProp to a BetQuote so props and main-market bets share
    one row shape on the dashboard."""
    side = (p.side or "").strip().capitalize()
    sel = f"{side} {p.line} {p.label}".strip() if p.line is not None else f"{p.label}"
    return BetQuote(
        event_id=p.event_id, game=p.game, selection=sel, market=p.market,
        best_odds=p.best_odds, consensus_odds=p.consensus_odds,
        best_book=p.best_book, edge_pct=p.edge_pct, player_name=p.player_name,
        line=getattr(p, "line", None),
        book_odds=getattr(p, "book_odds", None),
    )


def _best_bets(slate: dict, limit: int = 30) -> list:
    """Top +EV plays across markets (value spots + value props), round-robined
    so one hot game can't monopolize the board — every game that has a real
    edge contributes (max 2 seats each)."""
    value_spots = slate.get("value_spots") or []
    props = slate.get("props") or []
    best_all = sorted(
        list(value_spots) + [_prop_to_quote(p) for p in props if p.is_value],
        key=lambda q: q.edge_pct, reverse=True,
    )
    seats: dict[str, int] = {}
    best: list = []
    for q in best_all:
        g = q.game or "?"
        if seats.get(g, 0) >= 2:
            continue
        best.append(q)
        seats[g] = seats.get(g, 0) + 1
        if len(best) >= limit:
            break
    return best


def _resolve_prop_stat(stat_category: str) -> tuple[str, str | None, str]:
    """Map a provider prop market key to (bet group, stats column, unit).

    Keyed on the words inside the market name instead of exact keys, because
    the odds provider mixes canonical names (player_passing_yards) with the
    internal canonical list (player_pass_yds) in the same week's board.
    """
    c = (stat_category or "").lower()
    if not c:
        return "OTHER", None, ""
    if "interception" in c:
        return "PASS", "interceptions", "int"
    if "completion" in c:
        return "PASS", "pass_cmp", "cmp"
    if "attempt" in c:
        if "pass" in c or "passing" in c:
            return "PASS", "pass_att", "att"
        return "RUSH", "carries", "att"
    if "touchdown" in c or "_tds" in c or c.endswith("_td"):
        if "pass" in c or "passing" in c:
            return "PASS", "passing_tds", "td"
        if "rush" in c or "rushing" in c:
            return "RUSH", "rushing_tds", "td"
        if "receiv" in c or "rec" in c or c == "player_touchdowns":
            return ("TD" if "anytime" in c or c == "player_touchdowns" else "REC"), "__tot_tds", "td"
        return "TD", "__tot_tds", "td"
    if "yards" in c or "yds" in c:
        if "rushing" in c or "rush" in c:
            return "RUSH", "rushing_yards", "yds"
        if "receiving" in c or "rec" in c:
            return "REC", "receiving_yards", "yds"
        return "PASS", "passing_yards", "yds"
    if "reception" in c:
        return "REC", "receptions", "rec"
    if "pass" in c or "passing" in c:
        return "PASS", "passing_yards", "yds"
    return "OTHER", None, ""


def _prop_group(stat_category: str) -> str:
    g, _, _ = _resolve_prop_stat(stat_category)
    return g


def _prop_intel(slate_props, stats_df) -> list:
    """Attach a player's seasonal per-game production to every prop line.

    Each row blends the market (best price / consensus / edge) with the model
    (season avg vs line, lean direction, and a confirm/fade verdict when the
    price edge agrees with — or fights — the player's own numbers). Rows are
    JSON-safe so the bets page can filter/sort them entirely client-side.
    """
    if stats_df is None or getattr(stats_df, "empty", True):
        stats_df = {}
    by_name: dict = {}
    for _, _s in getattr(stats_df, "iterrows", lambda: [])():
        # Stats carry an ESPN short name (J.Love) plus the full display name
        # (Jordan Love); books use the full name, so index on both.
        for _c in ("player_display_name", "player_name"):
            nm = str(_s.get(_c, "") or "").strip().casefold()
            if nm:
                by_name.setdefault(nm, _s)
        pid = str(_s.get("player_id", "") or "").strip().casefold()
        if pid:
            by_name.setdefault(pid, _s)

    out: list = []
    for p in slate_props or []:
        row = by_name.get((p.player_name or "").strip().casefold())
        g, stat, unit = _resolve_prop_stat(p.stat_category)
        if row is None and g in ("PASS", "RUSH"):
            g_hint = {"PASS": "QB", "RUSH": "RB"}.get(g)
            out.append(_prop_intel_row(p, None, (stat, unit), group=g, pos_hint=g_hint))
            continue
        if row is None:
            out.append(_prop_intel_row(p, None, (stat, unit), group=g))
            continue
        out.append(_prop_intel_row(p, row, (stat, unit), group=g))
    return out


def _prop_intel_row(p, row, col_spec, group: str | None = None, pos_hint: str | None = None) -> dict:
    stat, unit = (col_spec or (None, ""))
    avg = None
    games = 0
    if row is not None:
        try:
            games = int(float(row.get("games") or 0))
        except (TypeError, ValueError):
            games = 0
        if stat == "__tot_tds":
            a = float(row.get("rushing_tds") or 0) + float(row.get("receiving_tds") or 0)
            avg = a / games if games else None
        elif stat and stat in getattr(row, "index", []):
            avg = float(row[stat]) / games if games else None
    gap = None
    lean = ""
    confirm = False
    if avg is not None and p.line:
        try:
            gap = round(avg - float(p.line), 1)
        except (TypeError, ValueError):
            gap = None
        if gap is not None:
            over = str(p.side or "").strip().casefold().startswith("over")
            model_over = gap > 0
            lean = "over" if model_over else "under"
            confirm = p.is_value and (over == model_over)
    return {
        "key": f"{p.event_id}-{p.market}-{p.player_name}-{p.side}-{p.line}",
        "game": p.game,
        "player": p.player_name,
        "market": p.market,
        "grp": group or _prop_group(p.stat_category),
        "side": p.side,
        "line": p.line,
        "label": p.label,
        "best_odds": odds.odds_html(p.best_odds),
        "consensus_odds": odds.odds_html(p.consensus_odds) if p.consensus_odds else "—",
        "book": p.best_book,
        "edge_pct": round(p.edge_pct, 1),
        "is_value": p.is_value,
        "two_x": False,
        "pos": str(row.get("position", pos_hint or "?")) if row is not None else (pos_hint or "?"),
        "team": str(row.get("recent_team") or row.get("team") or "") if row is not None else "",
        "games": games,
        "avg": (round(avg, 1) if avg is not None else None),
        "unit": unit,
        "gap": gap,
        "lean": lean,
        "confirm": confirm,
        "ladder": p.book_ladder(3) if hasattr(p, "book_ladder") else [],
        "slip": p.slip_payload() if hasattr(p, "slip_payload") else {},
    }


def _game_cards(slate: dict) -> list:
    """Per-game market cards for the dashboard: best side of each main market,
    prop count, value heat, and steam flags — richest games first."""
    from collections import defaultdict
    if not isinstance(slate, dict):
        return []
    rows_all = []
    for key in ("moneylines", "spreads", "totals", "props"):
        rows_all.extend(slate.get(key) or [])
    if not rows_all:
        return []
    try:
        moves = odds.line_moves(rows_all, min_delta=0.5)
    except Exception:
        moves = {}
    by_game: dict[str, list] = defaultdict(list)
    for q in rows_all:
        by_game[q.game or "?"].append(q)

    def _best(rows, market):
        sel = [q for q in rows if getattr(q, "market", "") == market]
        if not sel:
            return None
        return max(sel, key=lambda q: (getattr(q, "edge_pct", 0) or 0))

    cards = []
    for game, rows in by_game.items():
        if not game or game == "?":
            continue
        parts = game.split(" @ ")
        away, home = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (game, "")
        value = [q for q in rows if (getattr(q, "edge_pct", 0) or 0) >= 1.5]
        n_steam = 0
        for q in rows:
            if getattr(q, "market", "") in ("point_spread", "total_points"):
                try:
                    n_steam += 1 if odds.steam(q, moves) else 0
                except Exception:
                    pass
        cards.append({
            "game": game, "away": away, "home": home,
            "ml": _best(rows, "moneyline"), "spr": _best(rows, "point_spread"),
            "tot": _best(rows, "total_points"),
            "n_props": len([q for q in rows if q.market not in ("moneyline", "point_spread", "total_points")]),
            "value": value, "heat": len(value),
            "n_steam": n_steam, "n_lines": len(rows),
        })
    cards.sort(key=lambda c: c["heat"], reverse=True)
    return cards[:8]


def _team_abbr(team_name: str) -> str:
    name = (team_name or "").strip()
    abbr = odds.NFL_TEAM_ABBREVIATIONS.get(name)
    if abbr:
        return abbr
    tokens = name.split()
    return (tokens[-1][:3].upper() if tokens else "?")


def _radar_geom(values: list[float], labels: list[str], size: float = 118,
                cx: float = 128, cy: float = 128) -> dict:
    """Precompute an animated-friendly pentagon radar from normalized values.
    All geometry is computed here so the page renders zero-JS graphics."""
    n = len(values)
    mx = max(values) or 1.0

    def pt(i, r):
        ang = math.radians(-90 + 360 * i / n)
        return (cx + r * math.cos(ang), cy + r * math.sin(ang))

    grids, dots, spokes, labels_out = [], [], [], []
    for ring in (0.34, 0.68, 1.0):
        grids.append(" ".join(
            f"{x:.1f},{y:.1f}" for x, y in (pt(i, size * ring) for i in range(n))))
    for i in range(n):
        sx, sy = pt(i, 26)
        ex, ey = pt(i, size)
        spokes.append(f"{sx:.1f},{sy:.1f} {ex:.1f},{ey:.1f}")
        lx, ly = pt(i, size + 34)
        labels_out.append({"x": round(lx, 1), "y": round(ly, 1), "text": labels[i],
                           "above": ly <= cy})
    for i, v in enumerate(values):
        x, y = pt(i, 26 + (size - 26) * (v / mx))
        dots.append({"x": round(x, 1), "y": round(y, 1), "value": v})
    poly = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (pt(i, 26 + (size - 26) * (v / mx)) for i, v in enumerate(values)))
    return {"grids": grids, "spokes": spokes, "poly": poly, "dots": dots,
            "labels": labels_out, "cx": cx, "cy": cy}


def _radar_payload(cards: list) -> dict:
    """Platform data for the dashboard's signature Market Radar: top hot games
    with heat (count of value plays) and edge (avg edge %) series."""
    top = [c for c in cards if c["heat"] > 0][:5]
    if not top:
        return {"games": [], "heat": [], "edge": [],
                "heat_geom": None, "edge_geom": None, "hub": {"games": 0, "value": 0, "top": 0.0}}
    labels, heat, edge = [], [], []
    for c in top:
        labels.append(f"{_team_abbr(c['away'])}@{_team_abbr(c['home'])}")
        heat.append(float(c["heat"]))
        vals = [q.edge_pct for q in c["value"]]
        edge.append(round(sum(vals) / len(vals), 2) if vals else 0.0)
    hg = _radar_geom(heat, labels)
    eg = _radar_geom(edge, labels)
    return {
        "games": labels, "heat": heat, "edge": edge,
        "heat_geom": hg, "edge_geom": eg,
        "hub": {"games": len(top), "value": sum(int(h) for h in heat),
                "top": max(edge) if edge else 0.0},
    }


def _series_chart(rec: dict) -> dict:
    """Turn a line_hist series into chart geometry the template just drops in.

    X is sample index (SharpAPI has no clock openings — every fresh fetch is a
    sample), Y is the line scaled to a fixed 30px track with padding. A single
    sample is drawn as a flat bar so the chart reads even before movement.
    """
    pts = rec["pts"]
    if len(pts) == 1:
        pts = [pts[0], {"t": pts[0]["t"] + 1, "line": pts[0]["line"]}]
    lines = [p["line"] for p in pts]
    lo, hi = min(lines), max(lines)
    rng = (hi - lo) or 1.0
    pad = max(rng * 0.15, 0.25)
    lo, hi = lo - pad, hi + pad
    height = 30
    step = 26
    n = len(pts)
    width = max(90, n * step)

    def _y(v: float) -> float:
        return round((height - 4) - (v - lo) / (hi - lo) * (height - 4) + 2, 1)

    first, last = pts[0], pts[-1]
    delta = round(last["line"] - first["line"], 1)
    return {
        "game": rec.get("game", ""),
        "market": "SPR" if rec.get("market") == "point_spread" else "O/U",
        "side": rec.get("side", ""),
        "base": first["line"],
        "line": last["line"],
        "delta": delta,
        "up": delta > 0,
        "down": delta < 0,
        "n": n,
        "width": width,
        "height": height,
        "poly": " ".join(f"{i * step},{_y(p['line'])}" for i, p in enumerate(pts)),
        "base_y": _y(first["line"]),
        "cur_x": (n - 1) * step,
        "cur_y": _y(last["line"]),
    }


def _dashboard_analytics(slate: dict) -> dict:
    """Chart-ready aggregates for the home dashboard: edge histogram, per-game
    market heat, best-book share, and live line movements. Cheap, guarded, and
    never raises — a thin slate just yields empty series."""
    from collections import Counter
    out: dict = {"hist": [], "heat": [], "books": [], "moves": [], "n_moves": 0,
                 "series": [], "n_series": 0}
    if not isinstance(slate, dict):
        return out
    try:
        out["series"] = [_series_chart(r) for r in odds.line_series()][:8]
    except Exception:
        out["series"] = []
    out["n_series"] = len(out["series"])
    quotes = []
    for key in ("moneylines", "spreads", "totals", "props"):
        quotes.extend(slate.get(key) or [])
    val = [q for q in quotes if (getattr(q, "edge_pct", 0) or 0) >= 1.5]
    if not val:
        return out

    # Edge distribution — how big are the edges out there right now?
    buckets = [("1.5-3%", 0), ("3-5%", 0), ("5-8%", 0), ("8-12%", 0), ("12%+", 0)]
    for q in val:
        e = getattr(q, "edge_pct", 0) or 0
        if e < 3: buckets[0] = (buckets[0][0], buckets[0][1] + 1)
        elif e < 5: buckets[1] = (buckets[1][0], buckets[1][1] + 1)
        elif e < 8: buckets[2] = (buckets[2][0], buckets[2][1] + 1)
        elif e < 12: buckets[3] = (buckets[3][0], buckets[3][1] + 1)
        else: buckets[4] = (buckets[4][0], buckets[4][1] + 1)
    bmax = max(b[1] for b in buckets) or 1
    out["hist"] = [{"label": lab, "count": c, "pct": round(c / bmax * 100)} for lab, c in buckets]

    # Market heat — which games carry the most value right now?
    heat = Counter(q.game for q in val if getattr(q, "game", None))
    top_games = heat.most_common(6)
    hmax = max(c for _, c in top_games) or 1
    out["heat"] = [{"game": g, "count": c, "pct": round(c / hmax * 100)} for g, c in top_games]

    # Best-book share — who's offering the value?
    bk = Counter(q.best_book for q in val if getattr(q, "best_book", None))
    out["books"] = [{"book": b, "count": c, "pct": round(c / len(val) * 100)}
                    for b, c in bk.most_common(5)]

    # Line movements — steam across spreads + totals vs first-seen baselines.
    try:
        moves = odds.line_moves(quotes, min_delta=0.5)
    except Exception:
        moves = {}
    seen: set = set()
    rows: list = []
    for q in quotes:
        m = getattr(q, "market", "")
        if m not in ("point_spread", "total_points"):
            continue
        key = (q.event_id, m, (q.selection or "").casefold())
        if key in seen:
            continue
        seen.add(key)
        try:
            s = odds.steam(q, moves)
        except Exception:
            s = None
        if not s:
            continue
        rows.append({
            "game": q.game or "",
            "market": {"point_spread": "SPR", "total_points": "O/U"}.get(m, m),
            "side": (q.selection or ""),
            "line": q.line,
            "from": s["from"], "to": s["to"], "delta": s["delta"], "up": s["up"],
        })
    out["moves"] = rows[:8]
    out["n_moves"] = len(rows)
    return out


def _game_short(game: str) -> str:
    parts = (game or "").split(" @ ")
    if len(parts) == 2:
        return f"{_team_abbr(parts[0])}@{_team_abbr(parts[1])}"
    return game or "?"


def _market_map(slate: dict) -> dict:
    """Per-game × per-market edge matrix for the interactive Market Map.

    Full-board map: every game that appears in any quote stream gets a row and
    every cell (ML / SPR / O/U / PROP) carries the strongest edge found, so
    games and main markets show up even below the 1.5% value bar. Plain dicts,
    JSON-safe; cells that clear 1.5% are flagged `isv`.
    """
    from collections import defaultdict
    out = {"rows": [], "slots": [], "max": 0.0, "n_games": 0}
    if not isinstance(slate, dict):
        return out

    _MAIN = {"moneyline": "ml", "point_spread": "spr", "total_points": "tot"}
    _MTKS = ("ml", "spr", "tot", "prop")
    games: set[str] = set()
    best: dict[tuple, object] = {}
    nplays: dict[tuple, int] = defaultdict(int)

    def _scan(qs) -> None:
        for q in qs or []:
            g = getattr(q, "game", "") or ""
            if g and g != "?":
                games.add(g)

    _scan(slate.get("moneylines"))
    _scan(slate.get("spreads"))
    _scan(slate.get("totals"))
    _scan(slate.get("props"))

    def _consider(q, mkt) -> None:
        g = getattr(q, "game", "") or ""
        if not g or g == "?":
            return
        key = (g, mkt)
        if getattr(q, "is_value", False):
            nplays[key] += 1
        cur = best.get(key)
        if cur is None or (getattr(q, "edge_pct", 0) or 0) > (getattr(cur, "edge_pct", 0) or 0):
            best[key] = q

    for q in slate.get("moneylines") or []:
        mkt = _MAIN.get(getattr(q, "market", ""))
        if mkt:
            _consider(q, mkt)
    for q in slate.get("spreads") or []:
        mkt = _MAIN.get(getattr(q, "market", ""))
        if mkt:
            _consider(q, mkt)
    for q in slate.get("totals") or []:
        mkt = _MAIN.get(getattr(q, "market", ""))
        if mkt:
            _consider(q, mkt)
    for p in slate.get("props") or []:
        _consider(p, "prop")

    def _cell(q, mkt, min_edge: float | None = None) -> dict | None:
        if q is None:
            return None
        e = getattr(q, "edge_pct", 0) or 0
        if min_edge is not None and e < min_edge:
            return None
        if mkt == "prop":
            cls = ("bg-prop/80 text-black" if e >= 4.5
                   else "bg-prop/60 text-black" if e >= 3.5
                   else "bg-prop/40 text-white" if e >= 2.5
                   else "bg-prop/25 text-prop-light" if e >= 1.5
                   else "bg-prop/15 text-prop-light" if e >= 1.0
                   else "bg-prop/10 text-prop-light" if e >= 0.5
                   else "bg-prop/5 text-prop-light/70")
            label = ""
        else:
            cls = ("bg-emerald-400 text-black" if e >= 4.5
                   else "bg-emerald-400/80 text-black" if e >= 3.5
                   else "bg-emerald-400/50 text-emerald-100" if e >= 2.5
                   else "bg-emerald-400/30 text-emerald-200" if e >= 1.5
                   else "bg-emerald-400/20 text-emerald-300" if e >= 1.0
                   else "bg-emerald-400/10 text-emerald-300" if e >= 0.5
                   else "bg-emerald-500/10 text-emerald-400/80" if e >= 0
                   else "bg-surface-3/50 text-slate-400")
            label = odds.odds_html(getattr(q, "best_odds", 0))
        return {
            "edge": round(e, 1),
            "best": odds.odds_html(getattr(q, "best_odds", 0)),
            "sel": getattr(q, "selection", "") or getattr(q, "player_name", "") or "",
            "cls": cls,
            "label": label,
            "n": nplays.get((getattr(q, "game", ""), mkt), 0),
            "isv": e >= 1.5,
        }

    for game in sorted(games, key=lambda s: s.casefold()):
        row_cells = {
            "ml": _cell(best.get((game, "ml")), "ml"),
            "spr": _cell(best.get((game, "spr")), "spr"),
            "tot": _cell(best.get((game, "tot")), "tot"),
            "prop": _cell(best.get((game, "prop")), "prop", min_edge=0.0),
        }
        for mkt in _MTKS:
            c = row_cells[mkt]
            if c and c["edge"] > out["max"]:
                out["max"] = c["edge"]
        row = {"game": game, "short": _game_short(game), "cells": row_cells}
        out["rows"].append(row)
        out["slots"].append({"game": game, "short": row["short"], "m": "game"})
        for mkt in _MTKS:
            out["slots"].append({"game": game, "short": row["short"], "m": mkt,
                                 "cell": row_cells[mkt]})
    out["rows"].sort(key=lambda r: max(
        (c or {}).get("edge", 0.0) for c in r["cells"].values()), reverse=True)
    out["n_games"] = len(out["rows"])
    return out


def _deck_rows(slate: dict) -> list:
    """JSON-safe playbook rows: the full edge pool (value main-market plays +
    every prop that carries any edge). Sorted by edge so the client can render
    and filter a deep, sortable table without any page reload."""
    rows: list = []
    for q in slate.get("value_spots") or []:
        if not (q.edge_pct > 0):
            continue
        rows.append({
            "type": {"moneyline": "ML", "point_spread": "SPR", "total_points": "O/U"}
                    .get(getattr(q, "market", ""), "O/U"),
            "game": q.game, "short": _game_short(q.game),
            "sel": getattr(q, "selection", "") or "",
            "line": getattr(q, "line", None),
            "best": odds.odds_html(getattr(q, "best_odds", 0)),
            "cons": odds.odds_html(getattr(q, "consensus_odds", 0)) if getattr(q, "consensus_odds", None) else "—",
            "book": getattr(q, "best_book", "") or "—",
            "edge": round(q.edge_pct, 1), "isv": True, "prop": False,
            "books": q.book_ladder(3) if hasattr(q, "book_ladder") else [],
            "slip": _slip_tag("ML" if getattr(q, "market", "") == "moneyline"
                              else "SPR" if getattr(q, "market", "") == "point_spread"
                              else "TOT",
                              q.game, getattr(q, "selection", ""), getattr(q, "line", None),
                              getattr(q, "best_odds", 0), getattr(q, "best_book", "") or "",
                              round(q.edge_pct, 2)),
        })
    for p in slate.get("props") or []:
        if not (p.edge_pct > 0):
            continue
        side = (p.side or "").strip().capitalize()
        rows.append({
            "type": "PROP",
            "game": p.game, "short": _game_short(p.game),
            "sel": f"{side} {p.label}".strip(),
            "player": p.player_name, "line": p.line,
            "best": odds.odds_html(p.best_odds),
            "cons": odds.odds_html(p.consensus_odds) if p.consensus_odds else "—",
            "book": p.best_book or "—",
            "edge": round(p.edge_pct, 1), "isv": p.is_value, "prop": True,
            "cat": _prop_category(p.label),
            "books": [{"book": b, "am": a} for b, a in (getattr(p, "book_odds", None) or {}).items()][:3],
            "slip": _slip_tag("PROP", p.game, f"{side} {p.line} {p.label}".strip(),
                              p.line, p.best_odds, p.best_book or "",
                              round(p.edge_pct, 2), player=p.player_name),
        })
    rows.sort(key=lambda r: r["edge"], reverse=True)
    return rows[:120]


def _slip_tag(market: str, game: str, sel, line, american, book, edge, player: str = "") -> dict:
    """Compact JSON-safe slip payload embedded on every bet row so the global
    LEDGER drawer can pick picks up with one click (pure client-side storage)."""
    return {
        "m": market, "g": game or "", "s": sel or "", "l": line,
        "a": int(american or 0), "b": book or "", "e": round(float(edge or 0), 2),
        "p": player or "",
    }


def _prop_category(label: str) -> str:
    """Map a prop's label onto a compact stat-family key for the radar:
    passing / rushing / receiving / receptions / targets. Unknown stats
    collapse to their raw label so nothing is dropped."""
    l = (label or "").lower()
    if "passing" in l and ("yard" in l or "yd" in l):
        return "Passing Yards"
    if "rushing" in l and ("yard" in l or "yd" in l):
        return "Rushing Yards"
    if "receiving" in l and ("yard" in l or "yd" in l):
        return "Receiving Yards"
    if "reception" in l or "catch" in l:
        return "Receptions"
    if "target" in l:
        return "Targets"
    return label or "Other"


_RADAR_CATS = ["Passing Yards", "Rushing Yards", "Receiving Yards", "Receptions", "Targets"]


def _radar_payload(slate: dict) -> dict:
    """Category radar data: how deep + how hot each stat family is on the
    board. Axes are Passing/Rushing/Receiving Yds, Receptions and Targets.
    Each axis carries a share 0..1 (share of all value props) and a heat
    0..1 (its best edge / the slate max edge) for the SVG polygon."""
    from collections import defaultdict
    if not isinstance(slate, dict):
        return {"axes": [], "share_max": 1, "edge_max": 1}
    by_cat: dict[str, list] = defaultdict(list)
    for p in slate.get("props") or []:
        if not (p.edge_pct > 0):
            continue
        by_cat[_prop_category(p.label)].append(p)
    axes = []
    for cat in _RADAR_CATS:
        ps = by_cat[cat]
        if not ps:
            axes.append({"cat": cat, "n": 0, "best": 0.0, "avg": 0.0})
            continue
        best = max(p.edge_pct for p in ps)
        axes.append({"cat": cat, "n": len(ps),
                     "best": round(best, 1),
                     "avg": round(sum(p.edge_pct for p in ps) / len(ps), 1)})
    all_n = sum(a["n"] for a in axes)
    all_best = max((a["best"] for a in axes), default=0.01) or 0.01
    for a in axes:
        a["share"] = round(a["n"] / all_n, 3) if all_n else 0.0
        a["heat"] = round(a["best"] / (all_best or 0.01), 3)
    return {"axes": axes, "share_max": 1.0, "edge_max": round(all_best, 1)}


def _deck_payload(slate: dict | None) -> dict:
    """A single JSON blob powering the whole interactive dashboard deck:
    playbook rows, the market map, the edge distribution (for the dial) and
    book share. Served to `/api/slate` so the deck can refresh in place."""
    from collections import Counter
    import time as _t
    slate = slate or {}
    rows = _deck_rows(slate)
    m = _market_map(slate)
    value = [r for r in rows if r["isv"]]
    top = value[0]["edge"] if value else 0.0
    books = Counter(r["book"] for r in value if r["book"] and r["book"] != "—")
    return {
        "updated": (slate.get("fetched_at") or "") if isinstance(slate, dict) else "",
        "updated_ts": _t.time(),
        "ok": bool(rows),
        "max": m["max"],
        "rows": rows,
        "edges": sorted(round(r["edge"], 1) for r in rows),
        "map": m,
        "radar": _radar_payload(slate),
        "counts": {
            "games": len(slate.get("events") or []) if isinstance(slate, dict) else 0,
            "games_value": m["n_games"],
            "plays": len(rows),
            "value": len(value),
            "top": top,
            "lines": (len(slate.get("moneylines") or []) + len(slate.get("spreads") or [])
                      + len(slate.get("totals") or []) + len(slate.get("props") or []))
                     if isinstance(slate, dict) else 0,
        },
        "books": [{"book": b, "n": c, "pct": round(c / len(value) * 100) if value else 0}
                  for b, c in books.most_common(6)],
    }


def _current_slate() -> dict:
    """Resolve the same fullest board the bets tab shows (auto-advanced week
    board), falling back to building the full merged week board. Shared by the
    home page and the `/api/slate` refresh endpoint so both always agree."""
    import datetime as _bdt
    if not odds.API_KEY:
        return {}
    _today = _bdt.date.today().isoformat()
    _board = odds._cached(f"bets_slate_{_today}")
    if _board is not None:
        return _board
    _adv = odds._cached(f"bets_active_{_today}")
    if isinstance(_adv, dict) and _adv.get("active_date"):
        _cached = odds._cached(f"slate_{_adv.get('active_date')}")
        if _cached is not None:
            return _cached
    try:
        _full, _active = _full_slate_board(_today)
        return _full if isinstance(_full, dict) else {}
    except Exception:
        return {}


def ensure_data():
    global _data_loaded
    if _data_loaded:
        return
    import logging, time
    log = logging.getLogger(__name__)
    t0 = time.time()
    season = _get_season()
    try:
        data.load_player_stats(_get_season())
        data.load_schedule(season)
        journal.load_journal()
        journal.prime_from_cache()
        _data_loaded = True
        log.warning(f"ScoreEdge data loaded in {time.time()-t0:.1f}s (season {season})")
    except Exception as e:
        log.error(f"ScoreEdge data load FAILED: {e}")

def _filter_fantasy(df: pd.DataFrame) -> pd.DataFrame:
    if "position" in df.columns:
        return df[df["position"].isin(FANTASY_POSITIONS)]
    return df

def _normalize_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renames = {
        "player_display_name": "player_name",
        "passing_interceptions": "interceptions",
        "fantasy_points_ppr": "pts_ppr",
        "fantasy_points": "pts_std",
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    num_cols = ["passing_yards", "rushing_yards", "receiving_yards",
                "passing_tds", "rushing_tds", "receiving_tds", "receptions",
                "interceptions", "carries", "targets",
                "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
                "rushing_fumbles_lost", "receiving_fumbles_lost",
                "special_teams_tds", "games", "pass_att", "pass_cmp",
                "pts_ppr", "pts_std",
                "pos_rank_ppr", "pos_rank_half_ppr", "pos_rank_std"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "player_name" not in df.columns:
        if "player_id" in df.columns:
            df["player_name"] = df["player_id"]
        else:
            df["player_name"] = "Unknown"

    if "pts_half_ppr" not in df.columns:
        if "pts_ppr" in df.columns and "pts_std" in df.columns:
            df["pts_half_ppr"] = ((df["pts_ppr"] + df["pts_std"]) / 2).round(2)
        elif "pts_ppr" in df.columns:
            df["pts_half_ppr"] = df["pts_ppr"]
        else:
            df["pts_half_ppr"] = 0

    return df

def _tier_meta(records: list[dict], stats_df: pd.DataFrame,
               scoring: str = "ppr") -> dict[str, dict]:
    """Confidence tiers from recent-season per-game fantasy points.

    Median is the player's own per-game fantasy average (PPR/g) where a recent
    season is available; otherwise it is estimated from their model value score
    against the position's points relationship. The band (Ceiling/Floor) is the
    position's game-to-game point scatter, widened when the player missed games,
    so short/risky samples read as wide, volatile bands instead of fake-tight
    ones.
    """
    import statistics as _sm
    from collections import defaultdict as _dd
    pts_col = {"ppr": "pts_ppr", "half_ppr": "pts_half_ppr", "standard": "pts_std"}.get(scoring, "pts_ppr")

    pos_vals = _dd(list)
    for _, r in stats_df.iterrows():
        g = int(r.get("games") or 0)
        if g >= 1 and pts_col in r.index:
            pos_vals[str(r.get("position", ""))].append(float(r[pts_col]) / g)
    pos_sum: dict[str, dict] = {}
    for p, v in pos_vals.items():
        if len(v) < 6:
            continue
        m = _sm.mean(v)
        sd = _sm.pstdev(v)
        pos_sum[p] = {"mean": m, "median": _sm.median(v), "cv": (sd / m if m else 0.35)}

    by_id: dict[str, object] = {}
    by_name: dict[str, object] = {}
    for _, r in stats_df.iterrows():
        pid = str(r.get("player_id", ""))
        if pid:
            by_id[pid] = r
        nm = str(r.get("player_display_name") or r.get("player_name") or "").casefold().strip()
        if nm:
            by_name[nm] = r

    def _stat_row(rec):
        pid = str(rec.get("player_id") or "")
        if pid and pid in by_id:
            return by_id[pid]
        nm = str(rec.get("name") or rec.get("player_name") or "").casefold().strip()
        return by_name.get(nm)

    out: dict[str, dict] = {}
    for rec in records:
        p = str(rec.get("position") or "")
        ps = pos_sum.get(p, {"mean": 30.0, "median": 0.0, "cv": 0.35})
        row = _stat_row(rec)
        games = int(row.get("games") or 0) if row is not None else 0
        if row is not None and games >= 1 and pts_col in row.index:
            med = float(row[pts_col]) / games
            proj = False
        else:
            vs = float(rec.get("value_score") or rec.get("fantasy_pts") or 0) or 0
            base = ps["mean"] or 30.0
            med = base * (0.45 + 0.55 * min(max(vs, 0.0), 100.0) / 100.0)
            proj = True
        band = max(0.18, min(0.6, (ps["cv"] or 0.35) * (1 + max(0, 14 - games) / 14.0 * 0.6)))
        ceil = round(med * (1 + band), 1)
        floor = round(med * (1 - band), 1)
        med_r = round(med, 1)
        pmed = ps["median"] or 0
        if floor >= max(pmed * 0.85, 1.0):
            tier = "safe"
        elif floor <= max(pmed * 0.5, 0.5):
            tier = "risky"
        else:
            tier = "volatile"
        key = str(rec.get("player_id") or rec.get("name") or "")
        out[key] = {"med": med_r, "ceil": ceil, "floor": floor, "tier": tier, "proj": proj}
    return out


def _ctx(request: Request, sport: str = "nfl", **extra):
    season = _get_season()
    base = {"request": request, "active_sport": sport, "cache_bust": CACHE_BUST, "current_year": _season_display(season)}
    base.update(extra)
    return base


def _bounded(fn, timeout: float, *args, **kwargs):
    """Run fn with a wall-clock budget; return its result if it finishes within
    timeout, else a safe empty default (empty DataFrame for pandas callers)."""
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, *args, **kwargs)
            return fut.result(timeout=timeout)
    except _cf.TimeoutError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# Cross-league board: aggregate every live league's best-bets tape on one page.
# Cached in-process for OVERVIEW_TTL so rapid reloads don't re-hit the feed per
# league; the league slates already carry their own short-TTL caches.
OVERVIEW_TTL = 120.0


def _overview_board() -> dict:
    import time as _t
    _now = _t.time()
    if (_now - getattr(app.state, "_overview_ts", 0.0) < OVERVIEW_TTL
            and getattr(app.state, "_overview", None)):
        return app.state._overview

    _leagues = [
        ("nfl", "NFL", "🏈", "/nfl", "edge", 6),
        ("mlb", "MLB", "⚾", "/mlb", "hub", 0),
        ("nba", "NBA", "🏀", "/nba", "hub", 1),
        ("nhl", "NHL", "🏒", "/nhl", "hub", 2),
        ("mls", "MLS", "⚽", "/mls", "hub", 3),
        ("cfb", "CFB", "🏈", "/cfb", "hub", 4),
        ("cbb", "CBB", "🏀", "/cbb", "hub", 5),
    ]

    plays: list = []
    league_rows: list = []
    for tag, name, icon, href, source, order in _leagues:
        try:
            if source == "edge":
                slate = _current_slate() or {}
                props = list(slate.get("props") or [])
                # NFL props are PlayerProp objects (no is_playable); use the
                # NFL-specific tape helper for that board.
                best = _best_bets(slate, limit=40)
            else:
                slate, _e = _sport_slate(tag)
                props, _e2 = _sport_props_hub(tag)
                slate = slate or {}
                if not isinstance(slate, dict):
                    slate = {}
                best = _sport_best_bets(slate, props)
            if best is None:
                best = []
            _bk = {"ml": 0, "spr": 0, "tot": 0, "prop": 0}
            for _q in best:
                _bk[_qtype(_q)] = _bk.get(_qtype(_q), 0) + 1
            league_rows.append({
                "tag": tag, "name": name, "icon": icon, "href": href,
                "order": order,
                "plays": len(best), "games": len(slate.get("events") or []),
                "top_edge": round(max((q.edge_pts() for q in best), default=0.0), 2),
                "avg_edge": round((sum(q.edge_pts() for q in best) / len(best)) if best else 0.0, 1),
                "mkt": _bk,
                "quotes": best[:30],
                "tile": best[:4],
            })
            for q in best[:8]:
                plays.append({"tag": tag, "name": name, "icon": icon, "href": href, "q": q})
        except Exception:
            continue

    plays.sort(key=lambda p: p["q"].edge_pts(), reverse=True)
    top_plays = plays[:24]
    n = len(top_plays)
    avg = (sum(p["q"].edge_pts() for p in top_plays) / n) if n else 0.0
    board = {
        "plays": top_plays,
        "leagues": league_rows,
        "summary": {
            "plays": n,
            "leagues": len(league_rows),
            "avg_edge": round(avg, 1),
            "top_edge": round(top_plays[0]["q"].edge_pts(), 2) if top_plays else 0.0,
            "top_play": top_plays[0] if top_plays else None,
        },
    }
    app.state._overview = board
    app.state._overview_ts = _now
    return board


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        board = _overview_board()
    except Exception:
        board = {"plays": [], "leagues": [], "summary": {
            "plays": 0, "leagues": 0, "avg_edge": 0.0, "top_edge": 0.0, "top_play": None}}
    return templates.TemplateResponse(request, "overview.html", {
        **_ctx(request, "all", active_page="home"),
        "odds": odds,
        "board": board,
    })


@app.get("/nfl/dashboard", response_class=HTMLResponse)
def nfl_board(request: Request):
    ensure_data()
    season = _get_season()
    # schedule comes from nfldata.org which is flaky (drops TLS connections).
    # Bound it so a down API never hangs the page — result falls back to empty.
    schedule = _bounded(data.load_schedule, 8.0, season)
    odds_mod = odds
    slate = {}

    upcoming = []
    if not schedule.empty and "gameday" in schedule.columns:
        schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
        future = schedule[schedule["gameday"] >= pd.Timestamp.now()].head(10)
        for _, row in future.iterrows():
            game = {"week": row.get("week", ""), "date": str(row.get("gameday", ""))[:10]}
            for col in ["home_team", "away_team", "home_score", "away_score"]:
                if col in row.index:
                    game[col] = row[col]
            upcoming.append(game)

    # Betting-first dashboard: the board, the edge, and the schedule drive the
    # page. Fantasy surfaces (ADP, rankings, DFS) are deliberately absent —
    # those tools stay deep-linkable but are not part of this view.

    # Live betting "edge" feed (best value across books). Bounded + guarded so a
    # slow/down odds API never hangs or breaks the homepage.
    edge_spots = []
    fetched_at = ""
    edge_error = ""
    if odds_mod.API_KEY:
        import datetime as _dt
        try:
            slate = _bounded(
                lambda d=_dt.date.today().isoformat(): odds_mod.fetch_edge_slate(d),
                10.0,
            )
            if hasattr(slate, "get"):
                edge_spots = [q for q in slate.get("value_spots") or [] if getattr(q, "edge_pct", 0)]
                edge_spots = sorted(edge_spots, key=lambda q: q.edge_pct, reverse=True)[:8]
                # Props fallback: when no undecided main market lines exist today
                # (e.g. a Monday right after a game or a short week), surface the
                # strongest player-prop edges instead so the feed stays live.
                if not edge_spots:
                    edge_spots = [q for q in slate.get("props") or []
                                  if getattr(q, "edge_pct", 0)]
                    edge_spots = sorted(edge_spots, key=lambda q: q.edge_pct, reverse=True)[:8]
                for q in edge_spots:
                    try:
                        q.formatted = odds_mod.odds_html(q.best_odds)
                    except Exception:
                        q.formatted = str(q.best_odds)
                fetched_at = slate.get("fetched_at", "")
        except Exception as _edge_exc:
            edge_spots = []
            edge_error = str(getattr(_edge_exc, "message", None) or _edge_exc)[:140] or "odds provider rejected the request"

    # Betting KPI strip (games / lines / value spots) for the dashboard. Prefer
    # the slate a prior /bets visit auto-advanced to, so the numbers reflect the
    # same fullest board the bets tab shows; fall back to today's fetched slate.
    import datetime as _bdt
    bet_counts = {"games": 0, "lines": 0, "value": 0, "updated": ""}
    _src_slate = slate if isinstance(slate, dict) else {}
    if odds_mod.API_KEY:
        # Prefer the same fullest merged week board the bets tab renders (its
        # KPI strip and market map must agree with the /bets numbers). Falls
        # back to building it fresh if no cached board exists yet.
        _full_board = _current_slate()
        if _full_board:
            _src_slate = _full_board
    if isinstance(_src_slate, dict):
        bet_counts["games"] = len(_src_slate.get("events") or [])
        bet_counts["lines"] = (len(_src_slate.get("moneylines") or [])
                               + len(_src_slate.get("spreads") or [])
                               + len(_src_slate.get("totals") or [])
                               + len(_src_slate.get("props") or []))
        bet_counts["value"] = (len(_src_slate.get("value_spots") or [])
                               + len(_src_slate.get("props") or []))
        bet_counts["updated"] = _src_slate.get("fetched_at", "") or fetched_at
        # Sandbox: if a slate carries lines but its events list is empty (a
        # wrinkle in some cached slates), count distinct games from the market
        # quotes themselves so the KPI never shows "0 games on the board".
        if not bet_counts["games"]:
            _glabels = {
                q.game
                for _lst in ("moneylines", "spreads", "totals", "props")
                for q in (_src_slate.get(_lst) or [])
                if getattr(q, "game", None)
            }
            bet_counts["games"] = len(_glabels)

    # Top +EV plays from the same full board the KPI strip counts (aligns home
    # with the /bets tab — same slate, same numbers). Guarded so a thin/empty
    # day simply renders an empty list instead of breaking the page.
    best_bets = []
    if isinstance(_src_slate, dict):
        try:
            best_bets = _best_bets(_src_slate, limit=12)
        except Exception:
            best_bets = []

    # Touchdown-hunt promos live on the /bets tab only; the dashboard keeps its
    # own slate for the market map and playbook. (context keys kept for /bets)
    td_anytime, td_longest, td_value, td_warn = [], [], [], ""

    # Trend sparklines + chart aggregates for the dashboard. Every view (and
    # every fresh fetch) appends a compact snapshot, so the time series grows.
    try:
        odds_mod.log_slate_snapshot(_src_slate, "view")
    except Exception:
        pass
    try:
        analytics = _dashboard_analytics(_src_slate)
    except Exception:
        analytics = {"hist": [], "heat": [], "books": [], "moves": [], "n_moves": 0}
    try:
        game_cards = _game_cards(_src_slate)
    except Exception:
        game_cards = []
    try:
        history = odds_mod.snapshot_history(36 * 3600)
    except Exception:
        history = []

    # PROOF band + Edge Gems: the settled edge record (real outcomes) and the
    # five strongest plays as hero cards. Both guarded — a thin slate renders
    # empty rather than breaking the dashboard.
    try:
        proof = journal.proof(season)
    except Exception:
        proof = {"logged": 0, "settled": 0, "wins": 0, "losses": 0, "pushes": 0,
                 "roi_pct": 0.0, "hit_pct": 0.0, "avg_edge": 0.0, "top_edge": 0.0,
                 "by_bucket": [], "today_n": 0, "pending": 0}
    try:
        gems = []
        for _q in _best_bets(_src_slate, limit=5):
            _is_prop = bool(getattr(_q, "player_name", None))
            _mkt = getattr(_q, "market", "")
            _stype = ("PROP" if _is_prop
                      else "ML" if _mkt == "moneyline"
                      else "SPR" if _mkt == "point_spread" else "O/U")
            gems.append({
                "type": _stype,
                "sel": getattr(_q, "selection", "") or getattr(_q, "player_name", "") or "",
                "player": getattr(_q, "player_name", "") or "",
                "line": getattr(_q, "line", None),
                "game": getattr(_q, "game", "") or "",
                "best": odds.odds_html(getattr(_q, "best_odds", 0)),
                "book": getattr(_q, "best_book", "") or "—",
                "edge": round(getattr(_q, "edge_pct", 0) or 0, 2),
                "books": _q.book_ladder(4) if hasattr(_q, "book_ladder") else [],
                "slip": _slip_tag(_stype,
                                  getattr(_q, "game", ""),
                                  getattr(_q, "selection", "") or getattr(_q, "player_name", "") or "",
                                  getattr(_q, "line", None), getattr(_q, "best_odds", 0),
                                  getattr(_q, "best_book", "") or "", round(getattr(_q, "edge_pct", 0) or 0, 2),
                                  player=getattr(_q, "player_name", "") or ""),
            })
    except Exception:
        gems = []

    # Interactive deck payload: the raw material for the Market Map, Edge Dial,
    # live Playbook and Book Share modules. One JSON blob — also served by
    # /api/slate so the dashboard can refresh itself in place (auto/manual).
    try:
        deck = _deck_payload(_src_slate)
    except Exception:
        deck = {"updated": "", "updated_ts": 0, "ok": False, "max": 0.0, "rows": [], "edges": [],
                "map": {"rows": [], "slots": [], "max": 0.0, "n_games": 0},
                "radar": {"axes": [], "share_max": 1.0, "edge_max": 0.0},
                "counts": {"games": 0, "games_value": 0, "plays": 0, "value": 0, "top": 0.0, "lines": 0},
                "books": []}
    deck["fetched_at"] = bet_counts["updated"]

    # Prop intel for the dashboard: the strongest value props with their player's
    # seasonal production attached, plus confirm/fade verdicts when the price edge
    # agrees with (or fights) the player's own numbers.
    home_prop_rows: list = []
    try:
        _hp_stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        _hp_all = [r for r in _prop_intel((_src_slate.get("props") or []), _hp_stats) if r["is_value"]]
        _hp_all.sort(key=lambda r: r["edge_pct"], reverse=True)
        home_prop_rows = _hp_all[:6]
        home_prop_counts = {
            "value": len(_hp_all),
            "confirm": sum(1 for r in _hp_all if r["confirm"]),
            "fade": sum(1 for r in _hp_all if not r["confirm"]),
        }
    except Exception:
        home_prop_counts = {"value": 0, "confirm": 0, "fade": 0}

    return templates.TemplateResponse(request, "home.html", {
        **_ctx(request, "nfl", active_page="nfl"),
        "odds": odds_mod,
        "upcoming": upcoming,
        "edge_spots": edge_spots,
        "edge_fetched_at": fetched_at,
        "edge_error": edge_error,
        "bet_counts": bet_counts,
        "best_bets": best_bets,
        "td_anytime": td_anytime,
        "td_longest": td_longest,
        "td_value": td_value,
        "td_warn": td_warn,
        "analytics": analytics,
        "game_cards": game_cards,
        "history": history,
        "deck": deck,
        "home_prop_rows": home_prop_rows,
        "home_prop_counts": home_prop_counts,
        "proof": proof,
        "gems": gems,
    })

@app.get("/players", response_class=HTMLResponse)
def players_page(request: Request, q: str = "", pos: str = "", sort: str = "fantasy_pts"):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()
    board = draft.board_with_value(scoring="ppr", roster=None, top_n=9999)
    adp = pd.DataFrame(board)

    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if stats.empty:
            return templates.TemplateResponse(request, "players.html", {
                **_ctx(request, "nfl", active_page="players"),
                "players": [], "positions": [], "q": q, "pos": pos, "sort": sort, "in_season": True,
            })
        sort_map = {"fantasy_pts": "pts_ppr", "passing_yards": "passing_yards",
                    "rushing_yards": "rushing_yards", "receiving_yards": "receiving_yards"}
        sort_col = sort_map.get(sort, "pts_ppr")
        if q:
            stats = stats[stats["player_name"].str.contains(q, case=False, na=False)]
        if pos and pos != "ALL" and "position" in stats.columns:
            stats = stats[stats["position"] == pos]
        positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
        stats = stats.sort_values(sort_col, ascending=False).head(200)
        return templates.TemplateResponse(request, "players.html", {
            **_ctx(request, "nfl", active_page="players"),
            "players": stats.to_dict("records"), "positions": positions, "q": q, "pos": pos, "sort": sort, "in_season": True,
        })
    else:
        adp = pd.DataFrame(draft.board_with_value(scoring="ppr", roster=None, top_n=9999))
        if adp.empty:
            # No ADP for the upcoming draft board in September yet, so fall
            # back to the live-fantasy stats board (same chain the TD board
            # uses) instead of showing nothing on this tab.
            full = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
            positions = sorted(full["position"].dropna().unique().tolist()) if not full.empty and "position" in full.columns else []
            df = full.copy()
            if q and not df.empty and "name" in df.columns:
                df = df[df["name"].str.contains(q, case=False, na=False)]
            if pos and pos != "ALL" and not df.empty and "position" in df.columns:
                df = df[df["position"] == pos]
            df = df.sort_values("pts_ppr", ascending=False).head(200).reset_index(drop=True) if not df.empty and "pts_ppr" in df.columns else df
            df = df.assign(adp=None, adp_formatted="-", is_adp=False, high=None, low=None, high_formatted="-", low_formatted="-")
        return templates.TemplateResponse(request, "players.html", {
            **_ctx(request, "nfl", active_page="players"),
            "players": df.to_dict("records") if not df.empty else [], "positions": positions,
            "q": q, "pos": pos, "sort": "adp", "in_season": False,
        })

@app.get("/players/{player_id}", response_class=HTMLResponse)
def player_profile(request: Request, player_id: str):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()

    if in_season:
        stats = _normalize_stats(data.load_player_stats(_get_season()))
        player = {}
        perf = {}
        if not stats.empty and "player_id" in stats.columns:
            p = stats[stats["player_id"] == player_id]
            if not p.empty:
                player = p.iloc[0].to_dict()
            # Performance vs. position baseline (per-game averages) from the
            # full season board — reliable season totals only (no fabricated
            # weekly series). Helps gauge whether a player is above/below the
            # typical starter at their position.
            try:
                pos = str(player.get("position", "")).upper()
                if pos in ("QB", "RB", "WR", "TE"):
                    sub = stats[stats["position"] == pos].copy()
                    g = sub["games"].replace(0, pd.NA).astype(float)
                    base = {
                        "position": pos,
                        "count": int(len(sub)),
                        "avg_ppr": float((sub["pts_ppr"] / g).mean()),
                        "avg_half": float((sub["pts_half_ppr"] / g).mean()),
                        "avg_std": float((sub["pts_std"] / g).mean()),
                    }
                    gme = player.get("games") or 0
                    base["player_ppr"] = (player.get("pts_ppr") or 0) / gme if gme else 0.0
                    base["player_half"] = (player.get("pts_half_ppr") or 0) / gme if gme else 0.0
                    base["player_std"] = (player.get("pts_std") or 0) / gme if gme else 0.0
                    base["pct_ppr"] = (base["player_ppr"] / base["avg_ppr"] - 1.0) * 100 if base["avg_ppr"] else 0.0
                    perf = base
            except Exception:
                perf = {}
        return templates.TemplateResponse(request, "player.html", {
            **_ctx(request, "nfl", active_page="players"),
            "player": player, "player_id": player_id, "in_season": True,
            "perf": perf,
        })
    else:
        adp = pd.DataFrame(draft.board_with_value(scoring="ppr", roster=None, top_n=9999))
        player = {}
        if not adp.empty:
            p = adp[adp["player_id"].astype(str) == str(player_id)]
            if not p.empty:
                player = p.iloc[0].to_dict()
        return templates.TemplateResponse(request, "player.html", {
            **_ctx(request, "nfl", active_page="players"),
            "player": player, "player_id": player_id, "in_season": False,
        })

def _full_slate_board(date: str, explicit: bool = False) -> tuple[dict, str]:
    """Fetch the fullest betting board for `date` and merge the whole NFL game
    week (Thu..Tue) around it -- exactly what the /bets tab renders. Returns
    (slate, active_date). Daily slates are cached, so repeat calls are cheap."""
    import datetime as _dt
    today_iso = _dt.date.today().isoformat()

    def _slate_size(s) -> int:
        return (len(s.get("moneylines") or []) + len(s.get("spreads") or [])
                + len(s.get("totals") or []))

    slate = {"events": [], "moneylines": [], "spreads": [], "totals": [], "props": [],
             "value_spots": [], "best_bets": [], "fetched_at": ""}
    active_date = date
    if not odds.API_KEY:
        return slate, active_date

    # Auto-advance to the richest upcoming gameday when the requested day is thin.
    if not explicit:
        _dec_key = f"bets_active_{today_iso}"
        _hot_active = odds._cached(_dec_key)
        if isinstance(_hot_active, dict):
            _adv_date = _hot_active.get("active_date")
            if _adv_date:
                _cached_slate = odds._cached(f"slate_{_adv_date}")
                if (_cached_slate is not None and _slate_size(_cached_slate) >= 9
                        and len(_cached_slate.get("moneylines") or []) >= 1):
                    active_date = _adv_date
                    slate = _cached_slate
                else:
                    _hot_active = None
        if not _hot_active:
            _requested = odds.fetch_edge_slate(date)
            slate = _requested
            if _slate_size(slate) < 24:
                best: tuple[int, str, dict] = (_slate_size(slate), active_date, slate)
                candidate_days: list[tuple[int, str]] = []
                probe = _dt.date.fromisoformat(date)
                for _ in range(7):
                    probe += _dt.timedelta(days=1)
                    if probe.weekday() not in (3, 4, 5, 6):  # Mon..Wed non-slate
                        continue
                    probe_iso = probe.isoformat()
                    probe_count = 0
                    try:
                        ev = odds.fetch_nfl_events(probe_iso)
                        if ev:
                            probe_count = len(ev)
                    except (odds.OddsError, ValueError):
                        ev = None
                    # Even when the live events probe is throttled, a day with a
                    # cached slate is still a candidate so we can rank it.
                    if not probe_count and odds._last_good_slate(f"slate_{probe_iso}") is None:
                        continue
                    candidate_days.append((probe_count, probe_iso))
                candidate_days.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for _count, candidate_date in candidate_days[:2]:
                    try:
                        candidate = odds.fetch_edge_slate(candidate_date)
                    except (odds.OddsError, ValueError):
                        continue
                    if not (candidate.get("moneylines") or []):
                        continue
                    size = _slate_size(candidate)
                    if size > best[0]:
                        best = (size, candidate_date, candidate)
                    if size >= 24:
                        break
                if best[0] > _slate_size(slate):
                    slate = best[2]
                    active_date = best[1]
        # Persist the auto-advance decision so the next default load is instant
        # (only when we landed on a real, cached slate).
        if active_date and not explicit:
            _cached = odds._cached(f"slate_{active_date}")
            if (_cached is not None and _slate_size(_cached) >= 9
                    and len(_cached.get("moneylines") or []) >= 1):
                odds._cache_set(_dec_key, {"active_date": active_date})
    else:
        slate = odds.fetch_edge_slate(date)
        active_date = date

    # Merge game-market lines across the whole NFL game week (Thu..Tue) around
    # the active date. Thursday/Monday games belong to other days' event lists --
    # without this those games would show only props and no ML/spread/total.
    def _gameweek_span(d: _dt.date) -> list[_dt.date]:
        back = (d.weekday() - 3) % 7  # days back to the week's Thursday
        start = d - _dt.timedelta(days=back)
        return [start + _dt.timedelta(days=i) for i in range(6)]  # Thu..Tue

    _events: dict[str, object] = {e.event_id: e for e in (slate.get("events") or [])}
    _ml = list(slate.get("moneylines") or [])
    _spr = list(slate.get("spreads") or [])
    _tot = list(slate.get("totals") or [])
    _props = list(slate.get("props") or [])
    from concurrent.futures import ThreadPoolExecutor
    _merge_days: list[str] = []
    for _day in _gameweek_span(_dt.date.fromisoformat(active_date)):
        _iso = _day.isoformat()
        if _iso == active_date:
            continue
        _have_games = False
        try:
            _have_games = bool(odds.fetch_nfl_events(_iso))
        except (odds.OddsError, ValueError):
            _have_games = False
        if _have_games or odds._last_good_slate(f"slate_{_iso}") is not None:
            _merge_days.append(_iso)
    _others: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as _pool:
        _futures = [(_iso, _pool.submit(odds.fetch_edge_slate, _iso)) for _iso in _merge_days]
        for _iso, _fut in _futures:
            try:
                _others.append(_fut.result())
            except (odds.OddsError, ValueError):
                continue
    for _other in _others:
        for _e in (_other.get("events") or []):
            _events.setdefault(_e.event_id, _e)
        _ml += _other.get("moneylines") or []
        _spr += _other.get("spreads") or []
        _tot += _other.get("totals") or []
        _props += _other.get("props") or []

    # Keep the freshest quote for a duplicate (event, market, side, line).
    def _last_unique(quotes) -> list:
        kept = {}
        for q in quotes:
            kept[(q.event_id, q.market, (q.selection or "").casefold(), q.line)] = q
        return list(kept.values())

    slate["events"] = list(_events.values())
    slate["moneylines"] = _last_unique(_ml)
    slate["spreads"] = _last_unique(_spr)
    slate["totals"] = _last_unique(_tot)
    _props_seen: dict[tuple, object] = {}
    _props_dedup: list = []
    for _p in _props:
        _pk = (_p.event_id, _p.market, (_p.player_name or "").casefold(), _p.line)
        if _pk not in _props_seen:
            _props_seen[_pk] = _p
            _props_dedup.append(_p)
    slate["props"] = _props_dedup

    # Canonicalize game labels so the same matchup arriving from two day-fetches
    # (two event ids, home/away swapped) collapses to one deterministic label.
    _canon_labels: dict[tuple, str] = {}
    _label_by_evt: dict[str, str] = {}
    for _e in slate["events"]:
        _mk = (str(_e.home_team).casefold(), str(_e.away_team).casefold())
        _slot = tuple(sorted(_mk))
        if _slot not in _canon_labels:
            _canon_labels[_slot] = f"{_e.away_team} @ {_e.home_team}"
        _label_by_evt[_e.event_id] = _canon_labels[_slot]

    def _canon_game_label(q) -> str:
        lab = _label_by_evt.get(q.event_id)
        if lab:
            return lab
        g = getattr(q, "game", "") or ""
        if " @" in g:
            _a, _h = (x.strip() for x in g.split(" @ ", 1))
            _slot = tuple(sorted((_a.casefold(), _h.casefold())))
            return _canon_labels.setdefault(_slot, f"{_a} @ {_h}")
        return g

    def _relabel_quote(q):
        lab = _canon_game_label(q)
        if lab == q.game:
            return q
        return odds.BetQuote(q.event_id, lab, q.selection, q.market,
                             q.best_odds, q.consensus_odds, q.best_book,
                             q.edge_pct, q.line, q.player_name,
                             q.book_odds, q.book_edges)

    def _relabel_prop(p):
        lab = _canon_game_label(p)
        if lab == p.game:
            return p
        return odds.PlayerProp(p.event_id, lab, p.player_name, p.market, p.label,
                               p.stat_category, p.side, p.line, p.best_odds,
                               p.consensus_odds, p.best_book, p.edge_pct,
                               getattr(p, "book_odds", None))

    slate["moneylines"] = [_relabel_quote(q) for q in slate["moneylines"]]
    slate["spreads"] = [_relabel_quote(q) for q in slate["spreads"]]
    slate["totals"] = [_relabel_quote(q) for q in slate["totals"]]
    slate["props"] = [_relabel_prop(p) for p in slate["props"]]
    slate["value_spots"] = odds.build_value_spots(
        slate["moneylines"], slate["spreads"], slate["totals"])
    return slate, active_date

def _td_bundle(slate: dict) -> tuple:
    """Full touchdown-hunt bundle: anytime/longest/value boards, a warning
    string, and the bar-chart-race geometry. Used by the dedicated /touchdown
    page so the race + boards stay in sync on one canvas."""
    anytime, longest, value, warn = [], [], [], "TD promo unavailable"
    if isinstance(slate, dict) and (slate.get("events") or slate.get("props")):
        try:
            anytime, longest, value, warn = td_promo.build(slate)
        except Exception:
            anytime, longest, value, warn = [], [], [], "TD promo unavailable"
    race: list[dict] = []
    top = [r for r in anytime if r.get("scope") == "all"]
    if top:
        mx = max((r.get("p_pct") or 0) for r in top) or 1.0
        for i, r in enumerate(top[:10], 1):
            race.append({
                "name": r.get("name", ""),
                "team_name": r.get("team_name", ""),
                "pos": r.get("pos", ""),
                "p_pct": r.get("p_pct", 0.0),
                "booked": bool(r.get("booked")),
                "rank": i,
                "leader": i == 1,
                "pct_w": max(round((r.get("p_pct") or 0) / mx * 100.0, 1), 3.0),
            })
    return anytime, longest, value, warn, race


def _td_games(anytime: list, longest: list) -> list:
    """Game chips for the touchdown board (distinct games, stable order)."""
    seen: list = []
    for r in anytime + longest:
        g = r.get("game") or ""
        if g and g not in seen:
            seen.append(g)
    return seen


@app.get("/bets", response_class=HTMLResponse)
def bets_page(request: Request, date: str = "", focus: str = ""):
    ensure_data()
    season = _get_season()

    import datetime as _dt
    today_iso = _dt.date.today().isoformat()
    explicit = bool(date and date.strip())
    if not explicit:
        date = today_iso

    slate = {"events": [], "moneylines": [], "spreads": [], "totals": [], "props": [], "value_spots": [], "best_bets": [], "fetched_at": ""}
    error = ""
    active_date = date

    def _slate_size(s) -> int:
        return len(s.get("moneylines") or []) + len(s.get("spreads") or []) + len(s.get("totals") or [])

    if odds.API_KEY:
        try:
            slate, active_date = _full_slate_board(date, explicit)
        except odds.OddsError as e:
            error = str(e)
    else:
        error = "No API key set. Add ODDS_API_KEY to your environment."

    # Player props moved to the dedicated /props tab (full board, no truncation).
    # The Live Lines board keeps the raw prop feed only so best bets and the game
    # filter can still surface prop edges inline.
    double_value = double_value_players(slate)

    # Round-robin per game (max 2 seats each) so one hot game can't monopolize
    # the best-bets board — every game that has a real edge contributes.
    slate["best_bets"] = _best_bets(slate)

    # Steam-move detection: seed first-seen main lines, then diff current lines
    # against that baseline. SharpAPI offers no opening line, so a book-rolling
    # reference is the closest honest signal and reveals moves once they happen.
    _game_quotes = slate["moneylines"] + slate["spreads"] + slate["totals"]
    odds.record_line_baselines(_game_quotes)
    odds.record_line_history(_game_quotes)
    steam_map = odds.line_moves(_game_quotes)

    # Unique game labels for the in-page game filter. Include both the games with
    # ML/spread/total rows AND games that only have props (early-week games) so
    # every chip filters something real instead of an empty table.
    _richness: dict[str, int] = {}
    for _q in slate["moneylines"] + slate["spreads"] + slate["totals"] + slate["props"]:
        if _q.game:
            _richness[_q.game] = _richness.get(_q.game, 0) + 1
    _games = sorted(
        _richness,
        key=lambda g: (-_richness[g], g.casefold()),
    )

    weather_data = []
    schedule = data.load_schedule(season)
    if not schedule.empty and "gameday" in schedule.columns:
        schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        upcoming = schedule[(schedule["gameday"] >= today) & (schedule["gameday"] <= today + pd.Timedelta(days=7))]
        open_venues: list[tuple] = []
        for _, row in upcoming.head(10).iterrows():
            home = str(row.get("home_team", ""))
            stadium = data.NFL_STADIUMS.get(home, {})
            if stadium and stadium.get("roof") == "open":
                open_venues.append((row, stadium))
        if open_venues:
            def _weather_fetch(_row: object, _stadium: dict):
                try:
                    return _row, data.fetch_weather(_stadium["lat"], _stadium["lon"])
                except Exception:
                    return _row, {}
            with _cf.ThreadPoolExecutor(max_workers=min(8, len(open_venues))) as ex:
                _weather_futs = {ex.submit(_weather_fetch, r, st): (r, st) for r, st in open_venues}
                for _weather_fut in _weather_futs:
                    _row, _stadium = _weather_futs[_weather_fut]
                    try:
                        _w = _weather_fut.result(timeout=11)[1]
                    except Exception:
                        _w = {}
                    daily = _w.get("daily", {})
                    weather_data.append({
                        "home": _row.get("home_team", ""), "away": _row.get("away_team", ""),
                        "stadium": _stadium.get("name", ""),
                        "temp_max": daily.get("temperature_2m_max", [None])[0],
                        "temp_min": daily.get("temperature_2m_min", [None])[0],
                        "precip": daily.get("precipitation_sum", [None])[0],
                        "wind": daily.get("wind_speed_10m_max", [None])[0],
                    })

    def _weather_flag(w: dict) -> str | None:
        """Return a short weather advisory label for a weather dict, or None."""
        wind = w.get("wind") or 0
        precip = w.get("precip") or 0
        if wind and wind > 18:
            return "WINDY"
        if wind and wind > 12:
            return "BREEZY"
        if precip and precip > 0.3:
            return "RAIN"
        return None

    weather_by_game: dict[str, dict] = {}
    for _w in weather_data:
        _label = f"{_w['away']} @ {_w['home']}"
        weather_by_game[_label] = dict(_w)
        weather_by_game[_label]["flag"] = _weather_flag(_w)

    double_value = double_value_players(slate)

    # Mirror point the merged board for the dashboard KPI strip so the homepage
    # surface counts exactly what this tab shows (same events/lines/value).
    if odds.API_KEY and not error:
        odds._cache_set(f"bets_slate_{date}", slate)

    return templates.TemplateResponse(request, "bets.html", {
        **_ctx(request, "nfl", active_page="bets"),
        "odds": odds,
        "slate": slate,
        "weather_data": weather_data,
        "weather_by_game": weather_by_game,
        "double_value": double_value,
        "dfs_enabled": dfs.data_source() == "dk",
        "error": error,
        "api_key_set": bool(odds.API_KEY),
        "games": _games,
        "active_date": active_date,
        "focus": focus,
        "steam_map": steam_map,
    })


@app.get("/props", response_class=HTMLResponse)
def props_page(request: Request, date: str = ""):
    """Dedicated Player Props tab: the full weekly prop board vs each player's
    season numbers. No top-10 truncation like the Live Lines board — every
    posted line stands, sorted/filtered client-side."""
    ensure_data()
    season = _get_season()

    import datetime as _dt
    today_iso = _dt.date.today().isoformat()
    explicit = bool(date and date.strip())
    if not explicit:
        date = today_iso

    slate = {"events": [], "moneylines": [], "spreads": [], "totals": [], "props": [],
             "value_spots": [], "best_bets": [], "fetched_at": ""}
    error = ""
    active_date = date

    if odds.API_KEY:
        try:
            slate, active_date = _full_slate_board(date, explicit)
        except odds.OddsError as e:
            error = str(e)
    else:
        error = "No API key set. Add ODDS_API_KEY to your environment."

    try:
        _prop_stats = _filter_fantasy(_normalize_stats(data.load_player_stats(season)))
    except Exception:
        _prop_stats = pd.DataFrame()
    prop_rows = _prop_intel(slate.get("props") or [], _prop_stats)
    double_value = double_value_players(slate)
    for _r in prop_rows:
        if _r["player"] and _r["player"].casefold() in {str(d).casefold() for d in double_value}:
            _r["two_x"] = True

    _games = sorted({p.game for p in (slate.get("props") or []) if p.game},
                    key=lambda g: g.casefold())

    return templates.TemplateResponse(request, "props.html", {
        **_ctx(request, "nfl", active_page="props"),
        "odds": odds,
        "prop_rows": prop_rows,
        "double_value": double_value,
        "games": _games,
        "active_date": active_date,
        "error": error,
        "api_key_set": bool(odds.API_KEY),
    })


@app.get("/touchdown", response_class=HTMLResponse)
def touchdown_page(request: Request):
    """Dedicated Touchdown Hunt tab: P(TD) race + full most-likely/longest/value
    boards. Own slate (the fullest board), own game filter, no anchor-revert."""
    ensure_data()
    slate = _current_slate() if odds.API_KEY else {}
    td_anytime, td_longest, td_value, td_warn, td_race = _td_bundle(slate)
    games = _td_games(td_anytime, td_longest)
    return templates.TemplateResponse(request, "touchdown.html", {
        **_ctx(request, "nfl", active_page="td"),
        "td_anytime": td_anytime,
        "td_longest": td_longest,
        "td_value": td_value,
        "td_warn": td_warn,
        "td_race": td_race,
        "games": games,
        "api_key_set": bool(odds.API_KEY),
    })

def _team_color(name: str) -> str:
    """Deterministic per-team hue for the grouped-circle palettes."""
    h = (sum((ord(ch) * 31) for ch in (name or "?"))) % 360
    return f"hsl({h} 72% 62%)"


def _scatter_pt(q, mkt: str) -> dict:
    """One scatter point: edge on X, decimal payout on Y (american best odds
    converted), plus tooltip fields for the Value Lab quadrant chart."""

    def _dec(am):
        try:
            am = int(am)
        except (TypeError, ValueError):
            return 2.0
        return ((100.0 / abs(am)) + 1.0) if am < 0 else (am / 100.0 + 1.0)

    am = getattr(q, "best_odds", None)
    color = {"ML": "#34d399", "SPR": "#a78bfa", "O/U": "#60a5fa", "PROP": "#fbbf24"}.get(mkt, "#94a3b8")
    sel = getattr(q, "player_name", "") or getattr(q, "selection", "") or ""
    side = str(getattr(q, "side", "") or "").capitalize()
    line = getattr(q, "line", None)
    lab = getattr(q, "label", "") or ""
    if getattr(q, "player_name", ""):
        play = f"{side} {line}{(' ' + lab) if lab else ''} — {sel}" if line is not None else sel
    else:
        play = f"{sel}{(' ' + str(line)) if line is not None else ''}"
    return {
        "x": round(getattr(q, "edge_pct", 0) or 0, 1),
        "y": round(_dec(am), 2),
        "edge": round(getattr(q, "edge_pct", 0) or 0, 1),
        "best": am,
        "sel": sel,
        "play": play,
        "game": getattr(q, "game", "") or "",
        "book": getattr(q, "best_book", "") or "—",
        "type": mkt,
        "color": color,
        "ladder": q.book_ladder(3) if hasattr(q, "book_ladder") else [],
        "slip": q.slip_payload() if hasattr(q, "slip_payload") else {},
    }


@app.get("/edge-deck", response_class=HTMLResponse)
def edge_deck_page(request: Request):
    """Full-slate hexagon choropleth: every game × main market one hexagon,
    edge drives the fill (538-style), plus a book-share bubble strip. The
    dashboard's market map, supersized."""
    import math
    ensure_data()
    slate = _current_slate() if odds.API_KEY else {}
    deck = _deck_payload(slate)
    m = deck.get("map") or {}
    rows = (m.get("rows") or [])[:16]
    _MKTN = {"ml": "ML", "spr": "SPR", "tot": "O/U", "prop": "PROP"}

    def _hexfill(e: float) -> str:
        if e >= 4.5: return "#34d399"
        if e >= 3.0: return "#10b981"
        if e >= 2.0: return "#059669"
        if e >= 1.5: return "#047857"
        if e >= 1.0: return "#0f766e"
        if e >= 0.5: return "#115e59"
        if e > 0: return "#1d4a44"
        return "#334155"

    r = 19
    hexes: list[dict] = []
    for i, row in enumerate(rows):
        for j, mkt in enumerate(("ml", "spr", "tot", "prop")):
            cell = (row.get("cells") or {}).get(mkt)
            e = float((cell or {}).get("edge") or 0.0)
            cx = 190 + j * 52
            cy = 30 + i * 46 + (j % 2) * 18
            pts = " ".join(
                f"{cx + r * math.cos(math.radians(30 + 60 * k)):.1f},"
                f"{cy + r * math.sin(math.radians(30 + 60 * k)):.1f}"
                for k in range(6))
            hexes.append({
                "pts": pts, "fill": _hexfill(e),
                "cxp": cx, "cyp": cy,
                "edge": e, "isv": bool(cell and cell.get("isv")),
                "game": row.get("short", ""), "full": row.get("game", ""),
                "mkt": _MKTN[mkt],
                "sel": (cell or {}).get("sel", ""),
                "best": (cell or {}).get("best", ""),
                "n": (cell or {}).get("n", 0),
                "lab": f"{e:.1f}" if e >= 0.5 else "",
            })
    return templates.TemplateResponse(request, "edge_deck.html", {
        **_ctx(request, "nfl", active_page="edge_deck"),
        "hexes": hexes, "games": len(rows),
        "books": deck.get("books") or [], "counts": deck.get("counts") or {},
        "max_e": round(float((m or {}).get("max") or 0), 1),
        "W": 380,
        "H": round(30 + len(rows) * 46 + 24),
    })


@app.get("/line-lab", response_class=HTMLResponse)
def line_lab_page(request: Request):
    """Movement console: top-mover strip + the full spreads and totals
    sparkline wall — every main line sampled this week in one place."""
    ensure_data()
    slate = _current_slate() if odds.API_KEY else {}
    series = []
    try:
        series = [_series_chart(r) for r in odds.line_series()]
    except Exception:
        series = []
    spr = [s for s in series if s["market"] == "SPR"]
    ou = [s for s in series if s["market"] == "O/U"]
    movers = sorted([s for s in series if s.get("delta")],
                    key=lambda s: abs(s["delta"]), reverse=True)[:8]
    return templates.TemplateResponse(request, "line_lab.html", {
        **_ctx(request, "nfl", active_page="line_lab"),
        "spr": spr, "ou": ou, "movers": movers, "n_spr": len(spr), "n_ou": len(ou),
    })


@app.get("/value-lab", response_class=HTMLResponse)
def value_lab_page(request: Request):
    """Edge vs price laboratory: quadrant scatter of every play with any edge
    (value bar at 1.5%, evens line), packed-bubble EV, and the top plays."""
    ensure_data()
    slate = _current_slate() if odds.API_KEY else {}
    points: list[dict] = []
    for q in slate.get("value_spots") or []:
        if (getattr(q, "edge_pct", 0) or 0) > 0:
            mkt = {"moneyline": "ML", "point_spread": "SPR", "total_points": "O/U"}\
                .get(getattr(q, "market", ""), "O/U")
            points.append(_scatter_pt(q, mkt))
    for p in slate.get("props") or []:
        if (getattr(p, "edge_pct", 0) or 0) > 0:
            points.append(_scatter_pt(p, "PROP"))
    points.sort(key=lambda t: t["x"], reverse=True)
    if not points:
        return templates.TemplateResponse(request, "value_lab.html", {
            **_ctx(request, "nfl", active_page="value_lab"),
            "points": [], "top": [], "bubbles": [], "W": 680, "H": 430,
        })
    max_x = max((t["x"] for t in points), default=1.0)
    max_y = max((t["y"] for t in points), default=2.0)
    max_x = max(max_x, 2.2)
    max_y = max(max_y, 2.3)
    W, H, mx, my = 680, 430, 36, 34

    def _px(e: float) -> float:
        return round(mx + (e / max_x) * (W - mx - 64), 1)

    def _py(d: float) -> float:
        yy = min(max(d, 1.0), max_y)
        return round(H - my - ((yy - 1.0) / (max_y - 1.0)) * (H - my - 34), 1)

    for t in points:
        t["px"] = _px(t["x"])
        t["py"] = _py(t["y"])
    gl_y = [{"y": _py(v), "v": v} for v in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
            if 1.0 <= v <= max_y and _py(v) > 20]
    gl_x = [{"x": _px(e), "e": e} for e in (0.0, 1.0, 1.5) if e <= max_x]
    bubbles = []
    for t in points[:60]:
        ev = max(t["x"] * max(t["y"] - 1.0, 0.02), 0.05)
        bubbles.append({"r": round(min(26.0 + (ev ** 0.5) * 8.0, 90), 1),
                        "color": t["color"], "play": t["play"], "sel": t["sel"],
                        "edge": t["x"], "book": t["book"], "best": t["best"],
                        "type": t["type"], "game": t["game"], "y": t["y"],
                        "slip": t.get("slip") or {}, "ladder": t.get("ladder") or []})
    return templates.TemplateResponse(request, "value_lab.html", {
        **_ctx(request, "nfl", active_page="value_lab"),
        "points": points, "top": points[:12], "bubbles": bubbles,
        "max_x": max_x, "max_y": max_y, "W": W, "H": H,
        "vx": _px(1.5), "ey": _py(2.0), "gl_y": gl_y, "gl_x": gl_x,
        "mx": mx, "my": my,
    })


@app.get("/prop-radar", response_class=HTMLResponse)
def prop_radar_page(request: Request):
    """Grouped-circle prop mass: every player carrying any prop edge, clustered
    by position — circle size is the edge, tint is the team (grouped circles)."""
    from collections import defaultdict
    ensure_data()
    slate = _current_slate() if odds.API_KEY else {}
    season = _get_season()
    rows = []
    try:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(season)))
        rows = _prop_intel(slate.get("props") or [], stats)
    except Exception:
        rows = []
    by_pos: dict[str, list] = defaultdict(list)
    for r in rows:
        if (r.get("edge_pct") or 0) > 0:
            by_pos[(r.get("pos") or "?").upper()].append(r)
    groups = []
    for pos in ("QB", "RB", "WR", "TE"):
        rs = by_pos.get(pos) or []
        if not rs:
            continue
        rs.sort(key=lambda r: r.get("edge_pct") or 0, reverse=True)
        players = []
        for r in rs[:14]:
            players.append({
                "name": r.get("player", ""),
                "last": (r.get("player", "") or "").strip().split()[-1] if (r.get("player") or "").strip() else "?",
                "team": r.get("team", "") or "?",
                "edge": round(r.get("edge_pct") or 0, 1),
                "line": r.get("line", ""),
                "side": r.get("side", ""),
                "label": r.get("label", ""),
                "book": r.get("book", ""),
                "color": _team_color(r.get("team", "")),
                "confirm": bool(r.get("confirm")),
            })
        groups.append({"pos": pos, "n": len(rs),
                       "max": round(rs[0].get("edge_pct") or 0, 1),
                       "players": players})
    return templates.TemplateResponse(request, "prop_radar.html", {
        **_ctx(request, "nfl", active_page="prop_radar"),
        "groups": groups,
    })


# ---------------------------------------------------------------------------
# Toolbox — the secondary tools: parlay fair odds, Best Bets Today, the two
# player hunts (NHL goals / MLB home runs), golf top-20 and horse racing.
# Each is a thin route over existing data; all math lives in the tool modules.
# ---------------------------------------------------------------------------

# Every tool page offers the same sport selector: the NFL board plus each
# registry league with a live fetch path.
TOOL_SPORTS = ("nfl", "nba", "mlb", "nhl", "cfb", "cbb", "mls")


def _tool_slate(sport: str = "nfl") -> dict:
    """Never throw for a tool page: return an empty-but-typed slate on failure.

    The NFL path uses the full edge slate (value spots + PlayerProps); every
    other sport uses the registry fetcher with that league's prop markets.
    """
    if sport not in TOOL_SPORTS:
        sport = "nfl"
    try:
        if not odds.API_KEY:
            return {}
        if sport == "nfl":
            return _current_slate()
        slate = dict(odds.fetch_sport_slate(sport) or {})
        try:
            slate["props"] = odds.fetch_sport_props(sport) or []
        except Exception:
            slate["props"] = []
        slate["sport"] = sport
        return slate
    except Exception:
        return {}


@app.get("/parlay", response_class=HTMLResponse)
def parlay_page(request: Request, game: str = "", sport: str = "nfl", n: int = 10):
    """Same-game parlay fair-odds checker: ONE best bet per player (highest-edge
    market) with its de-vigged fair price, capped at `n` players per game; the
    browser builder compounds fair vs book odds and flags correlated legs.
    Kickers and novelty "longest" markets are excluded; sports with no prop
    markets (CFB/CBB) fall back to game-line legs."""
    ensure_data()
    slate = _tool_slate(sport)
    per_game = max(1, min(int(n or 10), 25))
    legs = toolbox.parlay_legs(slate, game, per_game=per_game)
    games = sorted({r["game"] for r in legs}, key=str.casefold)
    return templates.TemplateResponse(request, "parlay.html", {
        **_ctx(request, sport, active_page="parlay"),
        "legs": legs,
        "games": games,
        "active_game": game,
        "tool_sports": TOOL_SPORTS,
        "per_game": per_game,
        "slate_size": len(slate.get("props") or []),
        "api_key_set": bool(odds.API_KEY),
    })


@app.get("/api/slate", response_class=JSONResponse)
def api_slate(request: Request, refresh: int = 0, force: int = 0):
    """JSON payload for the dashboard deck; lets the home page auto-refresh its
    Market Map / Edge Dial / Playbook / Book Share without a full reload.

    `refresh=1` re-reads the cached odds (same source the bets tab mirrors) and
    recomputes the deck; `refresh=1&force=1` additionally re-fetches from the
    provider before recomputing.
    """
    import datetime as _bdt
    _src = {} if force else _current_slate()
    if not _src and odds.API_KEY:
        try:
            if force:
                for _k in list(odds._CACHE):
                    if _k.startswith("slate_") or _k.startswith("bets_slate_"):
                        odds._CACHE.pop(_k, None)
            _src = _bounded(
                lambda d=_bdt.date.today().isoformat(): odds.fetch_edge_slate(d),
                10.0,
            )
            if not isinstance(_src, dict):
                _src = {}
            else:
                odds._cache_set(f"bets_slate_{_bdt.date.today().isoformat()}", _src)
        except Exception:
            _src = {}
    try:
        deck = _deck_payload(_src)
    except Exception:
        deck = {"updated": "", "updated_ts": 0, "ok": False, "max": 0.0, "rows": [], "edges": [],
                "map": {"rows": [], "max": 0.0, "n_games": 0},
                "counts": {"games": 0, "games_value": 0, "plays": 0, "value": 0, "top": 0.0, "lines": 0},
                "books": []}
    deck["fetched_at"] = (_src.get("fetched_at", "") if isinstance(_src, dict) else "") or ""
    return JSONResponse(deck)


@app.get("/api/proof", response_class=JSONResponse)
def api_proof():
    """The PROOF band (edge journal settled against real scores) as JSON."""
    try:
        return JSONResponse(journal.proof(_get_season()))
    except Exception:
        return JSONResponse({"logged": 0, "settled": 0, "wins": 0, "losses": 0,
                             "pushes": 0, "roi_pct": 0.0, "hit_pct": 0.0,
                             "avg_edge": 0.0, "top_edge": 0.0, "by_bucket": [],
                             "today_n": 0, "pending": 0})


@app.post("/api/slip-resolve")
async def api_slip_resolve(request: Request):
    """Settle tracked-slip entries against real outcomes. Main markets resolve
    from final scores; prop legs report 'undecided' (weekly stat detail posts
    later). All state lives on the client — nothing is stored server-side."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    entries = body.get("entries") or []
    try:
        results = [journal.settle_entry(e) for e in entries if isinstance(e, dict)]
    except Exception:
        results = []
    return JSONResponse({"results": results})

@app.get("/rankings", response_class=HTMLResponse)
def rankings_page(request: Request, scoring: str = "ppr", position: str = "ALL"):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()

    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if stats.empty:
            return templates.TemplateResponse(request, "rankings.html", {
                **_ctx(request, "nfl", active_page="rankings"),
                "rankings": [], "scoring": scoring, "position": position, "positions": [], "source": "stats",
            })
        pts_col = {"ppr": "pts_ppr", "standard": "pts_std", "half_ppr": "pts_half_ppr"}.get(scoring, "pts_ppr")
        stats["fantasy_pts"] = stats[pts_col]
        if position and position != "ALL" and "position" in stats.columns:
            stats = stats[stats["position"] == position]
        stats = stats.sort_values("fantasy_pts", ascending=False).head(200).reset_index(drop=True)
        stats["rank"] = range(1, len(stats) + 1)
        positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
        _records = []
        for _row in stats.to_dict("records"):
            _row["tiers"] = {"med": None, "ceil": None, "floor": None, "tier": "volatile", "proj": False}
            _records.append(_row)
        tiers = _tier_meta(_records, stats, scoring)
        for _row in _records:
            _k = str(_row.get("player_id") or _row.get("player_name") or "")
            _row["tiers"] = tiers.get(_k, _row["tiers"])
        return templates.TemplateResponse(request, "rankings.html", {
            **_ctx(request, "nfl", active_page="rankings"),
            "rankings": _records, "scoring": scoring, "position": position,
            "positions": positions, "source": "stats",
        })
    else:
        full = draft.load_board(scoring)
        if full.empty:
            # No ADP for the upcoming draft board in September yet, so fall
            # back to the live-fantasy stats chain (same board the TD board
            # uses) instead of showing nothing on this tab.
            stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
            if stats.empty:
                return templates.TemplateResponse(request, "rankings.html", {
                    **_ctx(request, "nfl", active_page="rankings"),
                    "rankings": [], "scoring": scoring, "position": position, "positions": [], "source": "stats",
                })
            pts_col = {"ppr": "pts_ppr", "standard": "pts_std", "half_ppr": "pts_half_ppr"}.get(scoring, "pts_ppr")
            stats["fantasy_pts"] = stats[pts_col]
            if position and position != "ALL" and "position" in stats.columns:
                stats = stats[stats["position"] == position]
            stats = stats.sort_values("fantasy_pts", ascending=False).head(200).reset_index(drop=True)
            stats["rank"] = range(1, len(stats) + 1)
            positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
            _records = []
            for _row in stats.to_dict("records"):
                _row["tiers"] = {"med": None, "ceil": None, "floor": None, "tier": "volatile", "proj": False}
                _records.append(_row)
            return templates.TemplateResponse(request, "rankings.html", {
                **_ctx(request, "nfl", active_page="rankings"),
                "rankings": _records, "scoring": scoring, "position": position,
                "positions": positions, "source": "stats",
            })
        if not full.empty:
            positions = sorted(full["position"].dropna().unique().tolist()) if "position" in full.columns else []
            df = full.copy()
            if position and position != "ALL":
                df = df[df["position"] == position]
            df = df.head(200).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
            _records = df.to_dict("records")
            try:
                prior = _filter_fantasy(_normalize_stats(data.load_player_stats(min(_get_season() - 1, 2025))))
            except Exception:
                prior = pd.DataFrame()
            tiers = _tier_meta(_records, prior, scoring) if not prior.empty else {}
            for _row in _records:
                _row["tiers"] = tiers.get(str(_row.get("player_id") or ""),
                                          {"med": None, "ceil": None, "floor": None,
                                           "tier": "volatile", "proj": False})
            return templates.TemplateResponse(request, "rankings.html", {
                **_ctx(request, "nfl", active_page="rankings"),
                "rankings": _records, "scoring": scoring, "position": position,
                "positions": positions, "source": "model",
            })
        return templates.TemplateResponse(request, "rankings.html", {
            **_ctx(request, "nfl", active_page="rankings"),
            "rankings": [], "scoring": scoring, "position": position, "positions": [], "source": "model",
        })

@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request, players: str = ""):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()

    # Optional prefill from the Players tab "Compare" flow: ?players=idA,idB,...
    # The first two players land on Side A, the next two on Side B (a 2-for-2),
    # so the compare button connects the two tabs with one click.
    pre_a: list[dict] = []
    pre_b: list[dict] = []
    ids = [x for x in players.split(",") if x][:4]
    if ids:
        lookup: dict[str, str] = {}
        if in_season:
            stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
            if "player_id" in stats.columns and "player_name" in stats.columns:
                lookup = {str(r["player_id"]): str(r["player_name"])
                          for r in stats.to_dict("records")}
        else:
            adp = pd.DataFrame(draft.board_with_value(scoring="ppr", roster=None, top_n=9999))
            if not adp.empty and "player_id" in adp.columns:
                lookup = {str(r["player_id"]): str(r.get("name", ""))
                          for r in adp.to_dict("records")}
        named = [{"id": i, "name": lookup.get(i)} for i in ids if lookup.get(i)]
        pre_a = named[:2]
        pre_b = named[2:4]

    return templates.TemplateResponse(request, "trades.html", {
        **_ctx(request, "nfl", active_page="trades"),
        "in_season": in_season,
        "pre_a": pre_a,
        "pre_b": pre_b,
    })

@app.get("/draft-assistant", response_class=HTMLResponse)
def draft_assistant_page(request: Request, scoring: str = "half-ppr"):
    ensure_data()
    if scoring not in draft.SCORINGS:
        scoring = "half-ppr"

    roster = {k: int(v) for k, v in request.query_params.items() if k in draft.DEFAULT_ROSTER}
    roster = draft.normalize_roster(roster if roster else None)
    board = draft.board_with_value(scoring=scoring, roster=roster)
    # No ADP board in September (2026 draft is months away): fall back to the
    # same live-fantasy board the rankings/players tabs use so this is not empty.
    if not board:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if not stats.empty:
            board = (stats.sort_values("pts_ppr", ascending=False).head(100)
                     .assign(is_board=True).to_dict("records"))
    return templates.TemplateResponse(request, "draft_assistant.html", {
        **_ctx(request, "nfl", active_page="draft"),
        "board": board,
        "board_json": __import__("json").dumps(board),
        "scoring": scoring,
        "scoring_labels": draft.SCORING_LABELS,
        "scorings": draft.SCORINGS,
        "roster": roster,
        "roster_slots": draft.ROSTER_SLOTS,
    })

@app.get("/waivers", response_class=HTMLResponse)
def waivers_page(request: Request):
    ensure_data()
    sleepers = draft.waiver_sleepers()
    if not sleepers:
        # No ADP for the upcoming draft board in September yet, so fall back
        # to the same live-fantasy chain the TD/players boards use and show
        # "waiver-grade" names (decent per-game output) here instead of an
        # empty watch list.
        last_season = min(_get_season() - 1, 2025)
        live = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if not live.empty and "player_id" in live.columns:
            live = live.sort_values("pts_ppr", ascending=False).head(60).reset_index(drop=True)
            sleepers = []
            for _, _row in live.iterrows():
                _g = int(_row.get("games") or 0)
                _ppg = (float(_row.get("pts_ppr") or 0) / _g) if _g else None
                sleepers.append({
                    "player_id": str(_row.get("player_id") or ""),
                    "name": str(_row.get("player_name") or ""),
                    "pos": str(_row.get("position") or ""),
                    "team": str(_row.get("team_name") or ""),
                    "note": "live-fantasy waiver watch — solid per-game output while the 2026 ADP board fills",
                    "confidence": "watch",
                    "adp": None,
                    "ppg_last": round(_ppg, 1) if _ppg is not None else None,
                    "games_last": _g,
                    "last_season": last_season,
                })
    # Attach last season's real PPR output (per-game + games) to each sleeper so
    # this tab goes beyond a watch list and shows why the name is interesting.
    last_season = min(_get_season() - 1, 2025)
    prev = _filter_fantasy(_normalize_stats(data.load_player_stats(last_season)))
    by_name: dict[str, dict] = {}
    if not prev.empty and "player_name" in prev.columns:
        for _, r in prev.iterrows():
            key = str(r.get("player_display_name") or r["player_name"] or "").casefold().strip()
            if key:
                by_name.setdefault(key, r)
            bare = re.sub(r" jr\.?$| sr\.?$| iii$| ii$", "", key)
            if bare:
                by_name.setdefault(bare, r)
    for s in sleepers:
        r = by_name.get(str(s.get("name") or "").casefold().strip())
        if r is not None:
            games = int(r.get("games") or 0)
            ppg = (float(r.get("pts_ppr") or 0) / games) if games else None
            s["ppg_last"] = round(ppg, 1) if ppg is not None else None
            s["games_last"] = games
            s["last_season"] = last_season
    return templates.TemplateResponse(request, "waivers.html", {
        **_ctx(request, "nfl", active_page="waivers"),
        "sleepers": sleepers,
        "in_season": _is_nfl_regular_season(),
    })

@app.get("/dfs", response_class=HTMLResponse)
def dfs_page(request: Request, sport: str = "nfl", n: str = "1",
             stack: str = "", back: str = "",
             ds: str = "", gs: str = "", fade: str = "", cap: str = "",
             style: str = ""):
    sport = (sport or "").strip().lower()
    if sport not in dfs.ROSTERS:
        sport = "nfl"
    try:
        n = int(n or 1)
    except (TypeError, ValueError):
        n = 1
    n = max(1, min(n, dfs.MAX_LINEUPS))
    # Research construction rules (Milly Maker 2021-2025 winner study).
    # Defaults come from dfs.ROSTERS[sport]['rules']; the query params only
    # override when explicitly set to 0/1.
    rules = dict(dfs.ROSTERS[sport].get("rules") or {})
    if str(stack).strip() in ("0", "1"):
        rules["qb_stack"] = str(stack).strip() == "1"
    if str(back).strip() in ("0", "1"):
        rules["bring_back"] = str(back).strip() == "1"
    if str(ds).strip() in ("0", "1"):
        rules["double_stack"] = str(ds).strip() == "1"
    if str(gs).strip() in ("0", "1"):
        rules["game_stack"] = str(gs).strip() == "1"
    if str(fade).strip() in ("0", "1"):
        rules["fade_chalk"] = str(fade).strip() == "1"
    cap_i = int(cap) if str(cap).strip().isdigit() else 0
    rules["chalk_cap"] = cap_i if 0 <= cap_i <= 5 else 0
    style_s = str(style).strip().lower()
    rules["style"] = style_s if style_s in ("stars", "balanced") else "balanced"
    cfg = dfs.ROSTERS[sport]
    label = cfg["label"]
    error = ""
    note = ""
    slate = {"players": [], "slate_name": "", "start_time": "", "fetched_at": ""}
    lineups = []
    lineup = None
    res_min_diff = 0
    res_rules = {}
    res_relaxed = ""
    src = dfs.data_source()

    def _build(players):
        # Multi-lineup solve: n lineups under the pairwise overlap rule plus
        # whichever research construction rules are enabled.
        nonlocal lineups, lineup, res_min_diff, res_rules, res_relaxed
        res = dfs.build_lineups(players, sport, n, rules)
        lineups = res.get("lineups") or []
        lineup = lineups[0] if lineups else None
        res_min_diff = int(res.get("min_diff") or 0)
        res_rules = res.get("rules_applied") or {}
        res_relaxed = res.get("relaxed") or ""
        if res.get("error") and not lineups:
            return res["error"]
        return res.get("error") or ""

    if src == "dk":
        # PREFER live DraftKings pricing ONLY when it genuinely works (paid
        # provider reachable + licensed account-scoped). Any failure — and any
        # env that names dk but 403s — falls straight through to our clearly
        # labeled projection pricing rather than showing an empty error card.
        try:
            slate = dfs.fetch_slate(sport)
            build_err = _build(slate.get("players") or [])
            if build_err:
                note = build_err
                if not lineups:
                    slate = {"players": [], "slate_name": "", "start_time": "",
                             "fetched_at": ""}
        except Exception as e:
            detail = str(e).strip() or e.__class__.__name__
            note = detail if sport != "nfl" else (
                f"{detail} — falling back to our own projection pricing.")
            slate = {"players": [], "slate_name": "", "start_time": "", "fetched_at": ""}
            lineups, lineup = [], None
    # Projection pricing is the honest default: price our own slate off the
    # same live-fantasy chain every other tab already uses (clearly labeled
    # projection pricing; NOT official DraftKings pricing). Only NFL has that
    # chain wired — the other sports need the DK feed.
    if sport == "nfl" and not slate.get("players"):
        try:
            live = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
            sl, pnote = dfs.projection_slate(live.to_dict("records"), None)
            if sl.get("players"):
                slate = sl
                build_err = _build(slate.get("players") or [])
                if build_err and not lineups:
                    error = build_err
                note = f"{note} {pnote}".strip() if note else pnote
            else:
                error = pnote or f"Could not build a projection-priced {label} slate"
        except Exception as e:
            error = f"Could not build projection slate: {e}"
    if not error and not (slate.get("players") and lineup):
        if src != "dk":
            error = (f"DFS live data is off — set DFS_DATA_SOURCE=dk to pull the "
                     f"{label} slate from DraftKings.")
        else:
            error = (f"Could not load the {label} main slate from DraftKings"
                     + (f": {note}" if note else "."))
    q_swaps = (dfs.suggest_q_swaps(lineups, slate.get("players") or [], sport, rules)
               if lineups else {})
    contest = dfs.fetch_contest_info(sport) if sport == "nfl" else {}
    weather = (dfs.weather_notes(
                    sorted({r.get("game_id") or "" for lu in lineups
                            for r in lu.get("lineup") or []}),
                    {str(p.get("game_id") or ""): str(p.get("game") or "")
                     for p in (slate.get("players") or [])},
                    slate.get("weather_flags") or {})
               if (lineups and sport == "nfl") else [])
    portfolio = dfs.portfolio_view(lineups) if len(lineups) > 1 else {}
    return templates.TemplateResponse(request, "dfs.html", {
        **_ctx(request, sport, active_page="dfs"),
        "slate": slate,
        "lineup": lineup,
        "lineups": lineups,
        "error": error,
        "note": note,
        "dfs_sport": sport,
        "dfs_n": n,
        "dfs_n_options": [k for k in (1, 2, 3, 5, 10) if k <= dfs.MAX_LINEUPS],
        "dfs_min_diff": res_min_diff,
        "dfs_rules": res_rules,
        "dfs_rules_wanted": rules,
        "dfs_relaxed": res_relaxed,
        "dfs_stack": "1" if rules.get("qb_stack") else "0",
        "dfs_back": "1" if rules.get("bring_back") else "0",
        "dfs_ds": "1" if rules.get("double_stack") else "0",
        "dfs_gs": "1" if rules.get("game_stack") else "0",
        "dfs_fade": "1" if rules.get("fade_chalk") else "0",
        "dfs_cap": str(int(rules.get("chalk_cap") or 0)),
        "dfs_style": rules.get("style") or "balanced",
        "dfs_has_rules": bool(dfs.ROSTERS[sport].get("rules")),
        "dfs_q_swaps": q_swaps,
        "dfs_contest": contest,
        "dfs_weather": weather,
        "dfs_portfolio": portfolio,
        "dfs_excluded": slate.get("excluded") or {},
        "dfs_excluded_note": slate.get("excluded_note") or "",
        "dfs_sports": [{"key": k, "label": dfs.ROSTERS[k]["label"],
                        "icon": dfs.ROSTERS[k]["icon"]} for k in dfs.DFS_SPORTS],
        "roster": cfg,
    })

@app.get("/api/player-search", response_class=HTMLResponse)
def player_search_api(request: Request, q: str = ""):
    ensure_data()
    if len(q) < 2:
        return HTMLResponse("")
    in_season = _is_nfl_regular_season()
    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if stats.empty or "player_name" not in stats.columns:
            return HTMLResponse("")
        matches = stats[stats["player_name"].str.contains(q, case=False, na=False)].head(10)
        if matches.empty:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">No players found</div>')
        rows = []
        for _, r in matches.iterrows():
            name = r.get("player_name", "")
            pid = r.get("player_id", "")
            pos = r.get("position", "")
            team = r.get("recent_team", r.get("team", ""))
            rows.append(f'<a href="/players/{pid}" class="flex items-center gap-3 px-4 py-2.5 hover:bg-blue-500/10 transition-colors border-b border-white/5 last:border-0"><div class="flex-1"><div class="text-sm font-medium text-white">{name}</div><div class="text-xs text-slate-500">{team} · {pos}</div></div><svg class="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a>')
        return HTMLResponse(content=''.join(rows))
    else:
        adp = pd.DataFrame(draft.board_with_value(scoring="ppr", roster=None, top_n=9999))
        if adp.empty or "name" not in adp.columns:
            return HTMLResponse("")
        matches = adp[adp["name"].str.contains(q, case=False, na=False)].head(10)
        if matches.empty:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">No players found</div>')
        rows = []
        for _, r in matches.iterrows():
            name = r.get("name", "")
            pid = r.get("player_id", "")
            pos = r.get("position", "")
            team = r.get("team", "")
            adp_val = r.get("adp_formatted", "")
            rows.append(f'<a href="/players/{pid}" class="flex items-center gap-3 px-4 py-2.5 hover:bg-blue-500/10 transition-colors border-b border-white/5 last:border-0"><div class="flex-1"><div class="text-sm font-medium text-white">{name}</div><div class="text-xs text-slate-500">{team} · {pos} · ADP {adp_val}</div></div><svg class="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a>')
        return HTMLResponse(content=''.join(rows))

@app.get("/api/trade-compare", response_class=HTMLResponse)
def trade_compare_api(request: Request, a: str = "", b: str = ""):
    ensure_data()
    if not a or not b:
        return HTMLResponse("")
    a_ids = [x for x in a.split(",") if x]
    b_ids = [x for x in b.split(",") if x]
    if not a_ids or not b_ids:
        return HTMLResponse("")
    in_season = _is_nfl_regular_season()

    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if stats.empty or "player_id" not in stats.columns:
            return HTMLResponse("")
        players = {}
        for row in stats.to_dict("records"):
            players.setdefault(str(row.get("player_id")), row)
        def _build(ids):
            out = []
            for pid in ids:
                row = players.get(str(pid))
                if row is None:
                    continue
                out.append({"name": row.get("player_name", ""), "pos": row.get("position", ""),
                            "team": row.get("recent_team", ""), "pts": float(row.get("pts_ppr", 0)),
                            "games": float(row.get("games", 0) or 0), "row": row})
            return out
        sa, sb = _build(a_ids), _build(b_ids)
        if not sa or not sb:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">Player not found</div>')
        html = _trade_package_html(sa, sb, stat_mode=True)
        return HTMLResponse(content=html)
    else:
        adp = pd.DataFrame(draft.board_with_value(scoring="ppr", roster=None, top_n=9999))
        if adp.empty:
            return HTMLResponse("")
        adp_index = {str(r["player_id"]): r for r in adp.to_dict("records")}
        def _build_adp(ids):
            out = []
            for pid in ids:
                row = adp_index.get(str(pid))
                if row is None:
                    continue
                out.append({"name": row.get("name", ""), "pos": row.get("position", ""),
                            "team": row.get("team", ""), "adp": float(row.get("adp", 999)),
                            "adp_formatted": row.get("adp_formatted", ""),
                            "value": float(row.get("value_score", 0) or 0),
                            "bye": row.get("bye", "")})
            return out
        sa, sb = _build_adp(a_ids), _build_adp(b_ids)
        if not sa or not sb:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">Player not found</div>')
        html = _trade_package_html(sa, sb, stat_mode=False)
        return HTMLResponse(content=html)

def _trade_package_html(a: list[dict], b: list[dict], stat_mode: bool) -> str:
    """Render a 1v1 or package (2-for-2) trade with roster-impact totals."""
    def _total(side):
        return sum(p["pts"] for p in side) if stat_mode else sum(p["value"] for p in side)
    def _slots_used(side):
        return max(1, len(side))
    ta, tb = _total(a), _total(b)
    diff = ta - tb
    adv = "Side A" if diff > 0 else ("Side B" if diff < 0 else "Even")
    lib = "text-brand-light" if diff > 0 else "text-accent"

    if stat_mode:
        for p in a:
            p["lookup"] = "pts"
        metric_card = lambda p: f'{p["pts"]:.0f}'
        metric_sub = f'<span class="text-xs text-slate-500">PPR pts</span>'
        roto = (f'{max(abs(diff), 0):.0f}'
                f'<div class="text-xs text-slate-500">Total PPR pts on the move</div>')
    else:
        metric_card = lambda p: f'{p["value"]:.0f}'
        metric_sub = '<span class="text-xs text-slate-500">Model value (0-100)</span>'
        roto = (f'{max(abs(diff), 0):.0f}'
                f'<div class="text-xs text-slate-500">Total model value on the move</div>')

    def _side_html(p, accent):
        names = " & ".join(q["name"] for q in p)
        pills = "".join(f'<span class="pill bg-{"blue" if q["pos"]=="RB" else "emerald" if q["pos"]=="WR" else "purple" if q["pos"]=="QB" else "amber" if q["pos"]=="TE" else "slate"}-500/20 text-{"blue" if q["pos"]=="RB" else "emerald" if q["pos"]=="WR" else "purple" if q["pos"]=="QB" else "amber" if q["pos"]=="TE" else "slate"}-400">{q["pos"]}</span>' for q in p)
        team_lines = "<br>".join(f'<div class="text-xs text-slate-500">{q["team"] or "—"}</div>' for q in p)
        return f'''
    <div class="text-center border-r border-white/5">
      <div class="text-lg font-bold text-white">{names}</div>
      <div class="mt-1">{pills}</div>
      <div class="text-2xl font-display font-bold {accent} mt-2">{metric_card(p[0])}{f' + {metric_card(p[1])}' if len(p) > 1 else ''}</div>
      {metric_sub}
      <div class="mt-1">{team_lines}</div>
    </div>'''

    side_a = _side_html(a, "text-brand-light")
    side_b = _side_html(b, "text-accent")

    rows = ""
    for i in range(max(len(a), len(b))):
        pa = a[i] if i < len(a) else None
        pb = b[i] if i < len(b) else None
        na = pa["name"] if pa else "—"
        nb = pb["name"] if pb else "—"
        va = metric_card(pa) if pa else "—"
        vb = metric_card(pb) if pb else "—"
        row = (f'<tr class="border-b border-white/5"><td class="px-3 py-2 text-xs text-slate-500">{i+1}</td>'
               f'<td class="px-3 py-2 text-sm text-white">{na}</td>'
               f'<td class="px-3 py-2 text-right font-mono text-brand-light">{va}</td>'
               f'<td class="px-3 py-2 text-sm text-white">{nb}</td>'
               f'<td class="px-3 py-2 text-right font-mono text-accent">{vb}</td></tr>')
        rows += row

    pos_a = " / ".join(sorted(p["pos"] for p in a))
    pos_b = " / ".join(sorted(p["pos"] for p in b))
    slot_a = _slots_used(a)
    slot_b = _slots_used(b)
    eff_a = ta / slot_a
    eff_b = tb / slot_b
    edge_txt = ""
    if abs(eff_a - eff_b) > 0.01:
        winner = "Side A" if eff_a > eff_b else "Side B"
        edge_txt = (f'Per used slot, <span class="text-white">{winner}</span> is more efficient '
                    f'({eff_a:.1f} vs {eff_b:.1f}) — better roster value per active spot.')
    else:
        edge_txt = "Both sides are roughly equally efficient per used slot."

    return f'''<div class="glass-card rounded-xl p-5">
<div class="grid grid-cols-2 gap-4 mb-4">
  {side_a}
  {side_b}
</div>

<table class="w-full text-sm mb-4">
<thead><tr class="border-b border-white/5">
  <th class="px-3 py-2 text-left text-xs text-slate-400">#</th>
  <th class="px-3 py-2 text-left text-xs text-slate-400">Side A</th>
  <th class="px-3 py-2 text-right text-xs text-slate-400">{'PPR pts' if stat_mode else 'Value'}</th>
  <th class="px-3 py-2 text-left text-xs text-slate-400">Side B</th>
  <th class="px-3 py-2 text-right text-xs text-slate-400">{'PPR pts' if stat_mode else 'Value'}</th>
</tr></thead>
<tbody class="divide-y divide-white/5">
  {rows}
  <tr class="font-semibold">
    <td class="px-3 py-2"></td>
    <td class="px-3 py-2 text-slate-300">Combined</td>
    <td class="px-3 py-2 text-right font-mono text-brand-light">{ta if stat_mode else ta}</td>
    <td class="px-3 py-2 text-slate-300">Combined</td>
    <td class="px-3 py-2 text-right font-mono text-accent">{tb if stat_mode else tb}</td>
  </tr>
</tbody>
</table>

<div class="rounded-lg bg-surface-2/60 border border-white/5 p-4">
  <div class="flex items-center justify-between mb-2">
    <span class="text-xs uppercase tracking-wider text-slate-400">Roster impact</span>
    <span class="pill bg-brand/15 text-brand-light">Advantage: {adv}</span>
  </div>
  <p class="text-sm text-slate-300 mb-2">Moving <span class="text-white">{pos_a}</span> for <span class="text-white">{pos_b}</span> changes your projected output by <span class="{lib} font-semibold">{roto}</span>.</p>
  <p class="text-sm text-slate-300">{edge_txt}</p>
</div>
</div>'''

def _trade_html(a: dict, b: dict, stat_mode: bool) -> str:
    return _trade_package_html([a], [b], stat_mode)

def _sport_slate(tag: str, include_all: bool = False) -> tuple[dict, str]:
    slate = {"events": [], "moneylines": [], "spreads": [], "totals": [], "props": [], "value_spots": [], "conferences": {}, "scoped": False, "fetched_at": ""}
    if not odds.API_KEY:
        return slate, "No API key set. Add ODDS_API_KEY to your environment."
    try:
        return odds.fetch_sport_slate(tag, include_all=include_all), ""
    except odds.OddsError as e:
        return slate, str(e)


def _sport_props_hub(tag: str) -> tuple[list, str]:
    if not odds.API_KEY:
        return [], "No API key set. Add ODDS_API_KEY to your environment."
    try:
        return odds.fetch_sport_props(tag), ""
    except odds.OddsError as e:
        return [], str(e)


def _sport_best_bets(slate: dict, props: list) -> list:
    """Cross-market +EV tape ranked by true edge. Slate value spots carry the
    main-market plays; playable props join the tape so one view reads the whole
    day. Exact duplicate (event, market, side, line) rows collapse."""
    rows = list(slate.get("value_spots") or [])
    rows += [q for q in (props or []) if q.edge_pct and q.is_playable]
    seen: set[tuple] = set()
    out: list = []
    for q in sorted(rows, key=lambda x: x.edge_pts(), reverse=True):
        key = (q.event_id, q.market, (q.selection or "").casefold(), q.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out[:40]


def _sport_matchups(slate: dict, props: list) -> list:
    """Per-game command cards: every market's rows plus the game's best play."""
    by_game: dict[str, dict] = {}
    for q in slate.get("moneylines") or []:
        by_game.setdefault(q.game, {}).setdefault("ml", []).append(q)
    for q in slate.get("spreads") or []:
        by_game.setdefault(q.game, {}).setdefault("spr", []).append(q)
    for q in slate.get("totals") or []:
        by_game.setdefault(q.game, {}).setdefault("tot", []).append(q)
    for q in props or []:
        by_game.setdefault(q.game, {}).setdefault("prop", []).append(q)
    out = []
    for game, buckets in by_game.items():
        rows = buckets.get("ml", []) + buckets.get("spr", []) + buckets.get("tot", [])
        best = None
        playable = [q for q in rows if q.is_playable]
        if playable:
            best = max(playable, key=lambda q: q.edge_pts())
        covered = [k for k in ("ml", "spr", "tot") if buckets.get(k)]
        props = sorted((q for q in buckets.get("prop", []) if q.edge_pts() > 0),
                       key=lambda q: q.edge_pts(), reverse=True)
        top_plays = sorted((q for q in (rows + (buckets.get("prop") or [])) if q.edge_pts() > 0),
                           key=lambda q: q.edge_pts(), reverse=True)[:5]
        out.append({
            "game": game,
            "ml": buckets.get("ml", []),
            "spr": buckets.get("spr", []),
            "tot": buckets.get("tot", []),
            "props": buckets.get("prop", []),
            "top_props": props[:3],
            "top_plays": top_plays,
            "best": best,
            "n_markets": len(covered),
        })
    out.sort(key=lambda m: ((m["game"] or "").casefold()))
    return out


SPORT_TABS = [("lines", "Lines"), ("best", "Best Bets"), ("props", "Props"), ("games", "Matchups"), ("edge", "Deep Edge")]


def _sport_ctx(request: Request, tag: str, tab: str = "lines", include_all: bool = False) -> dict:
    cfg = odds.SPORTS[tag]
    if tab not in [k for k, _ in SPORT_TABS]:
        tab = "lines"
    slate, error = _sport_slate(tag, include_all)
    props, props_error = _sport_props_hub(tag)
    best_bets = _sport_best_bets(slate, props)
    matchups = _sport_matchups(slate, props)
    analytics = marketfeed.enrich(marketfeed._slate_quotes(slate) + list(props), tag)
    total_lines = len(slate.get("moneylines") or []) + len(slate.get("spreads") or []) + len(slate.get("totals") or [])
    market_names = dict(odds.DEFAULT_MARKET_NAMES)
    market_names.update(cfg.get("market_names") or {})
    avg_edge = round(sum(q.edge_pts() for q in best_bets) / len(best_bets), 1) if best_bets else 0.0
    top_edge = round(max((q.edge_pts() for q in best_bets), default=0.0), 2)
    prop_market_keys = list(cfg.get("prop_markets") or [])
    prop_groups = sorted({q.market for q in props},
                         key=lambda mk: prop_market_keys.index(mk) if mk in prop_market_keys else 99)
    prop_groups_labeled = [
        (mk, odds.PROP_LABELS.get(mk, mk.replace("player_", "").replace("_", " ").title()))
        for mk in prop_groups]
    prop_groups_group = {mk: sorted((q for q in props if q.market == mk),
                                 key=lambda q: q.edge_pts(), reverse=True) for mk in prop_groups}
    prop_group_top = {mk: round(max((q.edge_pts() for q in rows), default=0.0), 2)
                      for mk, rows in prop_groups_group.items()}
    return {
        **_ctx(request, tag),
        "odds": odds,
        "sport_cfg": cfg,
        "slate": slate,
        "odds_error": error,
        "props": props,
        "props_error": props_error,
        "best_bets": best_bets,
        "matchups": matchups,
        "total_lines": total_lines,
        "market_names": market_names,
        "prop_labels": odds.PROP_LABELS,
        "tabs": SPORT_TABS,
        "active_tab": tab,
        "scope_all": include_all,
        "no_props": bool(not cfg.get("prop_markets")),
        "avg_edge": avg_edge,
        "top_edge": top_edge,
        "best_markets": len({q.market for q in best_bets}),
        "props_players": len({(q.player_name or "").casefold() for q in props}),
        "props_markets": len({q.market for q in props}),
        "top_prop_edge": round(max((q.edge_pts() for q in props), default=0.0), 2),
        "prop_groups": prop_groups_labeled,
        "prop_groups_group": prop_groups_group,
        "prop_group_top": prop_group_top,
        "playable_games": sum(1 for m in matchups if m.get("best")),
        "analytics": analytics,
        "edge_nodes": _edge_nodes(slate, props, market_names, cfg.get("icon", "◆")),
    }


def _sport_hub(request: Request, tag: str, tab: str = "lines", include_all: bool = False):
    return templates.TemplateResponse(request, "sport_hub.html",
                                      _sport_ctx(request, tag, tab, include_all))


def _edge_nodes(slate: dict, props: list, market_names: dict, icon: str = "◆") -> list[dict]:
    """Node list for the deck's edge-landscape sim: one node per market family
    (main lines + each prop market), sized by quote count, edge = best edge."""
    def fam(rows, name, tag):
        tops = [q.edge_pts() for q in rows]
        return {"tag": tag, "name": name, "icon": icon, "plays": len(rows),
                "edge": round(max(tops, default=0.0), 1)}
    groups = []
    for key, name, tag in (("moneylines", market_names.get("moneyline", "Moneyline"), "lines"),
                           ("spreads", market_names.get("point_spread", "Spread"), "lines"),
                           ("totals", market_names.get("total_points", "Total"), "lines")):
        rows = slate.get(key) or []
        if rows:
            groups.append(fam(rows, name, tag))
    if props:
        by_mk: dict[str, list] = {}
        for q in props:
            by_mk.setdefault(q.market, []).append(q)
        for mk, rows in by_mk.items():
            label = odds.PROP_LABELS.get(mk, mk.replace("player_", "").replace("_", " ").title())
            groups.append(fam(rows, label, "props"))
    if not groups:
        groups = [{"tag": "lines", "name": "Moneyline", "icon": icon, "plays": 0, "edge": 0.0}]
    groups.sort(key=lambda g: g["plays"] * 3 + g["edge"], reverse=True)
    return groups[:10]


def _sport_deck(request: Request, tag: str, signature_tpl: str | None = None, include_all: bool = False):
    """Per-sport command deck. A shared shell (edge landscape, best-bets tape,
    props, matchups, deep edge) plus an optional signature module per sport."""
    ctx = _sport_ctx(request, tag, include_all=include_all)
    return templates.TemplateResponse(request, signature_tpl or "_sport_deck.html", {
        **ctx,
        "deck": {"edge_nodes": ctx["edge_nodes"]},
    })


NFL_DECK_CFG = {
    "sport": "football", "league": "nfl", "name": "NFL", "icon": "🏈",
    "markets": ["moneyline", "point_spread", "total_points"],
    "market_names": {"moneyline": "Moneyline", "point_spread": "Spread", "total_points": "Total"},
}


def _nfl_deck_ctx(request: Request) -> dict:
    """Command-deck context for the NFL board. NFL is not in odds.SPORTS (it has
    its own feed + week-board pipeline), so instead of _sport_ctx we assemble the
    same shell keys from the fullest merged week board and promote NFL PlayerProp
    objects to BetQuote so the shared deck engine (edge_pts/is_playable/markups)
    applies unchanged."""
    cfg = NFL_DECK_CFG
    slate = _current_slate() or {}
    error = "" if odds.API_KEY else "No API key set. Add ODDS_API_KEY to your environment."
    raw_props = list(slate.get("props") or [])
    props = [_prop_to_quote(p) for p in raw_props if getattr(p, "edge_pct", 0)]
    best_bets = _sport_best_bets(slate, props)
    matchups = _sport_matchups(slate, props)
    try:
        analytics = marketfeed.enrich(marketfeed._slate_quotes(slate) + props, "nfl")
    except Exception:
        analytics = {"n_quotes": 0, "n_usable": 0, "n_markets": 0, "avg_vig": 0.0,
                     "portfolio": {"expected_profit": 0.0, "roi": 0.0, "groups": 0, "legs": 0},
                     "portfolio_summary": "No market data on this slate.",
                     "arbs": [], "suspect_arbs": [], "n_issues": 0,
                     "sharp_books": [], "movers": [], "signals": []}
    total_lines = (len(slate.get("moneylines") or []) + len(slate.get("spreads") or [])
                   + len(slate.get("totals") or []))
    market_names = dict(odds.DEFAULT_MARKET_NAMES)
    market_names.update(cfg.get("market_names") or {})
    avg_edge = round(sum(q.edge_pts() for q in best_bets) / len(best_bets), 1) if best_bets else 0.0
    top_edge = round(max((q.edge_pts() for q in best_bets), default=0.0), 2)
    prop_groups = sorted({q.market for q in props})
    prop_groups_labeled = [
        (mk, odds.PROP_LABELS.get(mk, mk.replace("player_", "").replace("_", " ").title()))
        for mk in prop_groups]
    prop_groups_group = {mk: sorted((q for q in props if q.market == mk),
                                    key=lambda q: q.edge_pts(), reverse=True) for mk in prop_groups}
    prop_group_top = {mk: round(max((q.edge_pts() for q in rows), default=0.0), 2)
                      for mk, rows in prop_groups_group.items()}
    return {
        **_ctx(request, "nfl", active_page="nfl"),
        "odds": odds,
        "sport_cfg": cfg,
        "slate": slate,
        "odds_error": error,
        "props": props,
        "props_error": "",
        "best_bets": best_bets,
        "matchups": matchups,
        "total_lines": total_lines,
        "market_names": market_names,
        "prop_labels": odds.PROP_LABELS,
        "no_props": False,
        "scope_all": False,
        "avg_edge": avg_edge,
        "top_edge": top_edge,
        "props_players": len({(q.player_name or "").casefold() for q in props}),
        "props_markets": len(prop_groups),
        "top_prop_edge": round(max((q.edge_pts() for q in props), default=0.0), 2),
        "prop_groups": prop_groups_labeled,
        "prop_groups_group": prop_groups_group,
        "prop_group_top": prop_group_top,
        "playable_games": sum(1 for m in matchups if m.get("best")),
        "analytics": analytics,
        "edge_nodes": _edge_nodes(slate, props, market_names, cfg["icon"]),
        "deck": {"edge_nodes": _edge_nodes(slate, props, market_names, cfg["icon"])},
    }


@app.get("/nfl", response_class=HTMLResponse)
def nfl_page(request: Request):
    return templates.TemplateResponse(request, "_nfl_deck.html", _nfl_deck_ctx(request))

@app.get("/nba", response_class=HTMLResponse)
def nba_page(request: Request):
    return _sport_deck(request, "nba", signature_tpl="_nba_deck.html")

@app.get("/nba/best", response_class=HTMLResponse)
def nba_best(request: Request):
    return _sport_hub(request, "nba", "best")

@app.get("/nba/props", response_class=HTMLResponse)
def nba_props(request: Request):
    return _sport_hub(request, "nba", "props")

@app.get("/nba/games", response_class=HTMLResponse)
def nba_games(request: Request):
    return _sport_hub(request, "nba", "games")

@app.get("/nba/edge", response_class=HTMLResponse)
def nba_edge(request: Request):
    return _sport_hub(request, "nba", "edge")

@app.get("/cfb", response_class=HTMLResponse)
def cfb_page(request: Request, all: str = ""):
    return _sport_deck(request, "cfb", include_all=all.strip().casefold() in ("1", "true", "yes", "on"))

@app.get("/cfb/best", response_class=HTMLResponse)
def cfb_best(request: Request):
    return _sport_hub(request, "cfb", "best")

@app.get("/cfb/props", response_class=HTMLResponse)
def cfb_props(request: Request):
    return _sport_hub(request, "cfb", "props")

@app.get("/cfb/games", response_class=HTMLResponse)
def cfb_games(request: Request):
    return _sport_hub(request, "cfb", "games")

@app.get("/cfb/edge", response_class=HTMLResponse)
def cfb_edge(request: Request):
    return _sport_hub(request, "cfb", "edge")

@app.get("/cbb", response_class=HTMLResponse)
def cbb_page(request: Request):
    return _sport_deck(request, "cbb")

@app.get("/cbb/best", response_class=HTMLResponse)
def cbb_best(request: Request):
    return _sport_hub(request, "cbb", "best")

@app.get("/cbb/props", response_class=HTMLResponse)
def cbb_props(request: Request):
    return _sport_hub(request, "cbb", "props")

@app.get("/cbb/games", response_class=HTMLResponse)
def cbb_games(request: Request):
    return _sport_hub(request, "cbb", "games")

@app.get("/cbb/edge", response_class=HTMLResponse)
def cbb_edge(request: Request):
    return _sport_hub(request, "cbb", "edge")

@app.get("/mlb", response_class=HTMLResponse)
def mlb_page(request: Request):
    return _sport_deck(request, "mlb", signature_tpl="_mlb_deck.html")

@app.get("/mlb/best", response_class=HTMLResponse)
def mlb_best(request: Request):
    return _sport_hub(request, "mlb", "best")

@app.get("/mlb/props", response_class=HTMLResponse)
def mlb_props(request: Request):
    return _sport_hub(request, "mlb", "props")

@app.get("/mlb/games", response_class=HTMLResponse)
def mlb_games(request: Request):
    return _sport_hub(request, "mlb", "games")

@app.get("/mlb/edge", response_class=HTMLResponse)
def mlb_edge(request: Request):
    return _sport_hub(request, "mlb", "edge")

@app.get("/nhl", response_class=HTMLResponse)
def nhl_page(request: Request):
    return _sport_deck(request, "nhl")

@app.get("/nhl/best", response_class=HTMLResponse)
def nhl_best(request: Request):
    return _sport_hub(request, "nhl", "best")

@app.get("/nhl/props", response_class=HTMLResponse)
def nhl_props(request: Request):
    return _sport_hub(request, "nhl", "props")

@app.get("/nhl/games", response_class=HTMLResponse)
def nhl_games(request: Request):
    return _sport_hub(request, "nhl", "games")

@app.get("/nhl/edge", response_class=HTMLResponse)
def nhl_edge(request: Request):
    return _sport_hub(request, "nhl", "edge")

@app.get("/mls", response_class=HTMLResponse)
def mls_page(request: Request):
    return _sport_deck(request, "mls")

@app.get("/mls/best", response_class=HTMLResponse)
def mls_best(request: Request):
    return _sport_hub(request, "mls", "best")

@app.get("/mls/props", response_class=HTMLResponse)
def mls_props(request: Request):
    return _sport_hub(request, "mls", "props")

@app.get("/mls/games", response_class=HTMLResponse)
def mls_games(request: Request):
    return _sport_hub(request, "mls", "games")

@app.get("/mls/edge", response_class=HTMLResponse)
def mls_edge(request: Request):
    return _sport_hub(request, "mls", "edge")

@app.get("/golf", response_class=HTMLResponse)
def golf_page(request: Request, tour: str = "pga", event: str = "", force: int = 0):
    """Golf Top-20 best bets: DataGolf's model P(top 20) ranked against the
    Shin de-vigged consensus of books' own top-20 prices for the tournament."""
    ensure_data()
    data = golf.fetch_top20(tour, event, force=force)
    return templates.TemplateResponse(request, "golf.html", {
        **_ctx(request, "golf", active_page="golf"),
        "golf": data,
        "tours": golf.TOURS,
        "active_tour": tour,
        "active_event": event,
        "key_set": bool(golf.get_key()),
    })


@app.get("/api/golf", response_class=JSONResponse)
def api_golf(tour: str = "pga", event: str = "", force: int = 0):
    """Poll endpoint for the golf page (10-minute server cache, force=1 busts)."""
    return JSONResponse(golf.fetch_top20(tour, event, force=force))


@app.post("/golf/key", response_class=JSONResponse)
async def golf_key(payload: dict):
    """Save the DataGolf API key locally (server-side settings file only)."""
    saved = golf.set_key(str(payload.get("key") or ""))
    return JSONResponse({"ok": True, "set": bool(saved)})


@app.get("/goal-hunt", response_class=HTMLResponse)
def goal_hunt_page(request: Request, date: str = "", game: str = "",
                   top: int = 10, force: int = 0):
    """NHL Goal Hunt: the ten most-likely goal scorers on tonight's slate."""
    ensure_data()
    data = hockey.build_board(date, game, top=top, force=force)
    return templates.TemplateResponse(request, "goal_hunt.html", {
        **_ctx(request, "nhl", active_page="goal_hunt"),
        "hunt": data,
        "today": hockey._today(),
    })


@app.get("/api/goal-hunt", response_class=JSONResponse)
def api_goal_hunt(date: str = "", game: str = "", top: int = 10, force: int = 0):
    return JSONResponse(hockey.build_board(date, game, top=top, force=force))


@app.get("/homerun-hunt", response_class=HTMLResponse)
def homerun_hunt_page(request: Request, date: str = "", game: str = "",
                      top: int = 10, force: int = 0):
    """MLB Home Run Hunt: the ten most-likely home-run hitters on today's slate."""
    ensure_data()
    data = baseball.build_board(date, game, top=top, force=force)
    return templates.TemplateResponse(request, "homerun_hunt.html", {
        **_ctx(request, "mlb", active_page="homerun_hunt"),
        "hunt": data,
        "today": baseball._today(),
    })


@app.get("/api/homerun-hunt", response_class=JSONResponse)
def api_homerun_hunt(date: str = "", game: str = "", top: int = 10, force: int = 0):
    return JSONResponse(baseball.build_board(date, game, top=top, force=force))


@app.get("/racing", response_class=HTMLResponse)
def racing_page(request: Request, date: str = "", race_code: str = "gallops",
                track: str = "", timezone: str = "", force: int = 0):
    """Horse Racing Value Finder: FormFav form, model and book prices."""
    ensure_data()
    data = racing.build_board(date, race_code, track, timezone, force=force)
    return templates.TemplateResponse(request, "racing.html", {
        **_ctx(request, "nfl", active_page="racing"),
        "board": data,
        "race_codes": racing.RACE_CODES,
        "date": data.get("date") or racing._today(),
        "race_code": data.get("race_code") or "gallops",
        "today": racing._today(),
        "key_set": bool(data.get("key_set")),
    })


@app.get("/api/racing", response_class=JSONResponse)
def api_racing(date: str = "", race_code: str = "gallops", track: str = "",
               timezone: str = "", force: int = 0):
    return JSONResponse(racing.build_board(date, race_code, track, timezone, force=force))


@app.post("/racing/key", response_class=JSONResponse)
async def racing_key(payload: dict):
    """Save the FormFav API key locally (server-side settings file only)."""
    saved = racing.set_key(str(payload.get("key") or ""))
    return JSONResponse({"ok": True, "set": bool(saved)})


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse(request, "about.html",
                                      {**_ctx(request, "nfl"), "active_page": "about"})


@app.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse(request, "contact.html",
                                      {**_ctx(request, "nfl"), "active_page": "contact"})


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request):
    return templates.TemplateResponse(request, "privacy.html",
                                      {**_ctx(request, "nfl"), "active_page": "privacy"})


@app.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request):
    return templates.TemplateResponse(request, "terms.html",
                                      {**_ctx(request, "nfl"), "active_page": "terms"})


@app.get("/odds", response_class=HTMLResponse)
def odds_redirect(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/bets", status_code=301)

@app.get("/robots.txt")
def robots_txt():
    return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /static/\n", media_type="text/plain")

@app.get("/sitemap.xml")
def sitemap_xml():
    base = "https://scoreedge.onrender.com"
    pages = ["", "/bets", "/nba", "/cfb", "/cbb", "/mlb", "/nhl", "/mls", "/golf",
             "/goal-hunt", "/homerun-hunt", "/racing",
             "/about", "/contact", "/privacy", "/terms"]
    urls = "\n".join(f'  <url><loc>{base}{p}</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>' for p in pages)
    xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}\n</urlset>'
    return PlainTextResponse(xml, media_type="application/xml")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(request, "404.html", {
        **_ctx(request, "nfl"), "active_page": "404"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8512)





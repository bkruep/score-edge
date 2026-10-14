"""Auto-settlement of NFL player props from real weekly production.

The aggregated season stats feed drops the per-week detail a prop needs, so
props were previously left 'pending' forever. Sleeper's weekly feed carries the
same nflverse game logs, so we settle a prop by resolving the player (normalized
name + team, so same-name players never cross-settle), finding the game's week
from the schedule, reading the relevant stat, and comparing to the line.

Everything fails soft: no feed and no cache -> 'pending', never a wrong result.
Legacy prop records written before the Over/Under side was stored also stay
'pending' (we refuse to guess a side).
"""
from __future__ import annotations

import os
import re
import json
import time
import threading
import datetime as _dt

import requests

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_CACHE = os.path.join(_DIR, "nfl_week_stats.json")

_SLEEPER = "https://api.sleeper.app/v1"
_TTL = 6 * 3600  # refetch a cached week at most every 6h while the season runs
_lock = threading.Lock()
_mem: dict[str, dict] = {}

# Sleeper stat keys we keep (2026 uses *_yd; older seasons used *_yds — we read
# whichever is present). ``sack`` variants cover the rare defensive prop.
_TRACK = {
    "pass_yd", "pass_yds", "pass_td", "pass_int", "pass_att", "pass_cmp",
    "rush_yd", "rush_yds", "rush_td", "rush_att", "rec", "rec_yd", "rec_yds",
    "rec_td", "rec_tgt", "sack", "idp_sack",
}

# token (from the feed market after "player_") -> candidate Sleeper stat keys,
# first present wins. Unknown tokens settle to 'pending' (never a wrong grade).
_STAT_ALIASES: dict[str, list[str]] = {
    "receptions": ["rec"],
    "receiving_yards": ["rec_yd", "rec_yds"],
    "rushing_yards": ["rush_yd", "rush_yds"],
    "passing_yards": ["pass_yd", "pass_yds"],
    "passing_attempts": ["pass_att"],
    "passing_completions": ["pass_cmp"],
    "completions": ["pass_cmp"],
    "attempts": ["pass_att"],
    "passing_touchdowns": ["pass_td"],
    "passing_tds": ["pass_td"],
    "passing_interceptions": ["pass_int"],
    "interceptions": ["pass_int"],
    "rushing_attempts": ["rush_att"],
    "carries": ["rush_att"],
    "rushing_touchdowns": ["rush_td"],
    "rushing_tds": ["rush_td"],
    "receiving_touchdowns": ["rec_td"],
    "receiving_tds": ["rec_td"],
    "targets": ["rec_tgt"],
    "sacks": ["sack", "idp_sack"],
}
_COMBINED = {"rushing_+_receiving_yards", "rushing_receiving_yards",
             "rush_rec_yards", "rushing_and_receiving_yards"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm(name: str) -> str:
    """Name key: lowercase, punctuation dropped, hyphens to spaces, suffixes cut."""
    s = (name or "").lower().replace(".", "").replace("'", "")
    s = s.replace("-", " ").replace("`", "")
    parts = [p for p in re.split(r"\s+", s) if p]
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _abbr(team: str) -> str:
    import odds
    return (odds.NFL_TEAM_ABBREVIATIONS.get((team or "").strip()) or "").upper()


def _read_cache() -> dict:
    try:
        with open(_CACHE, "r", encoding="utf-8") as fh:
            v = json.load(fh)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(disk: dict) -> None:
    try:
        with open(_CACHE, "w", encoding="utf-8") as fh:
            json.dump(disk, fh)
    except OSError:
        pass


def _fetch_week(season: int, week: int) -> dict | None:
    """{normname: [ {team, pos, **stats}, ... ]} for one (season, week), or None
    when the feed is unreachable (caller may fall back to a stale cache)."""
    try:
        r = requests.get(f"{_SLEEPER}/stats/nfl/regular/{season}/{week}", timeout=30)
        r.raise_for_status()
        raw = r.json() or {}
    except Exception:
        return None
    import data
    roster = data.load_rosters() or {}
    out: dict[str, list] = {}
    for pid, st in raw.items():
        if not isinstance(st, dict):
            continue
        meta = roster.get(str(pid)) or {}
        nm = str(meta.get("full_name") or "").strip()
        if not nm:
            continue
        key = _norm(nm)
        if not key:
            continue
        row = {"team": str(meta.get("team") or "").upper(),
               "pos": str(meta.get("position") or "")}
        for k in _TRACK:
            if st.get(k) is not None:
                row[k] = st[k]
        out.setdefault(key, []).append(row)
    return out


def _week_rows(season: int, week: int) -> dict:
    key = f"{season}:{week}"
    if key in _mem:
        return _mem[key]
    disk = _read_cache()
    ent = disk.get(key)
    if ent and time.time() - ent.get("ts", 0) < _TTL:
        _mem[key] = ent.get("rows") or {}
        return _mem[key]
    rows = _fetch_week(season, week)
    if rows is None:
        if ent:  # feed down: serve stale rather than nothing
            _mem[key] = ent.get("rows") or {}
        else:
            _mem[key] = {}
        return _mem[key]
    _mem[key] = rows
    with _lock:
        disk = _read_cache()
        disk[key] = {"ts": time.time(), "rows": rows}
        _write_cache(disk)
    return rows


def _week_for(season: int, away_ab: str, home_ab: str, day: str) -> int | None:
    """The schedule week a matchup belongs to, using gameday proximity to pick
    the right one when a pair meets twice in a season."""
    if not away_ab or not home_ab:
        return None
    import data
    sched = data.load_schedule(season)
    if sched is None or getattr(sched, "empty", True):
        return None
    try:
        rows = sched[(sched["home_team"] == home_ab) & (sched["away_team"] == away_ab)]
    except (KeyError, TypeError):
        return None
    if rows.empty:
        return None
    d = None
    try:
        d = _dt.date.fromisoformat(str(day))
    except (TypeError, ValueError):
        d = None
    best: tuple[int, int] | None = None
    for _, r in rows.iterrows():
        try:
            wk = int(r["week"])
        except (TypeError, ValueError):
            continue
        if d is None:
            cand = (0, wk)
        else:
            try:
                gd = _dt.date.fromisoformat(str(r["gameday"]))
            except (TypeError, ValueError):
                continue
            diff = abs((gd - d).days)
            if diff > 3:
                continue
            cand = (diff, wk)
        if best is None or cand < best:
            best = cand
    return best[1] if best else None


def _week1_sunday(season: int) -> _dt.date:
    """Sunday of NFL week 1 for a season (first Sunday after Sep 7)."""
    sep7 = _dt.date(season, 9, 7)
    return sep7 + _dt.timedelta(days=(6 - sep7.weekday()) % 7)


def _weeks_near(season: int, day: str) -> list[int]:
    """Candidate weeks around a date, closest first (date-anchored fallback for
    when the schedule cache is incomplete — nfldata drops weeks under load)."""
    try:
        d = _dt.date.fromisoformat(str(day))
    except (TypeError, ValueError):
        return []
    base = _week1_sunday(season)
    approx = 1 + (d - base).days // 7
    cand = sorted({w for w in range(approx - 2, approx + 3) if 1 <= w <= 22},
                  key=lambda w: abs((base + _dt.timedelta(days=(w - 1) * 7)) - d))
    return cand


def _resolve_value(player: str, token: str, teams: tuple[str, str],
                   season: int, day: str, away_ab: str, home_ab: str):
    """(value, week) using the schedule-precise week when available, else the
    nearest week whose stats contain the player on one of the matchup teams."""
    week = _week_for(season, away_ab, home_ab, day)
    if week:
        v = _stat_value(player, token, teams, season, week)
        if v is not None:
            return v, week
    for wk in _weeks_near(season, day):
        rows = _week_rows(season, wk).get(_norm(player))
        if not rows:
            continue
        match = ([r for r in rows if (r.get("team") or "") in teams]
                 if any(teams) else rows)
        if not match:
            continue
        return _value_from_row(match[0], token), wk
    return None, None


def _pick(row: dict, aliases: list[str]) -> float | None:
    for a in aliases:
        v = row.get(a)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _value_from_row(row: dict, token: str) -> float | None:
    if token in _COMBINED:
        a = _pick(row, ["rush_yd", "rush_yds"])
        b = _pick(row, ["rec_yd", "rec_yds"])
        if a is None and b is None:
            return None
        return (a or 0.0) + (b or 0.0)
    aliases = _STAT_ALIASES.get(token)
    if not aliases:
        return None
    return _pick(row, aliases)


def _stat_token(rec: dict) -> str:
    for k in ("market_raw", "mkt", "stat", "market"):
        v = str(rec.get(k) or "")
        if v.startswith("player_"):
            return v[len("player_"):].strip().casefold()
    return ""


def _stat_value(player: str, token: str, teams: tuple[str, str],
                season: int, week: int) -> float | None:
    if not player or not token:
        return None
    rows = _week_rows(season, week).get(_norm(player))
    if not rows:
        return None
    if any(teams):
        on_team = [r for r in rows if (r.get("team") or "") in teams]
        if on_team:
            rows = on_team
    return _value_from_row(rows[0], token)


def settle_prop(rec: dict) -> dict:
    """Resolve one prop record to won/lost/push, else pending with a reason.

    Needs the stat line, the Over/Under side, and the player. Records lacking a
    side (legacy shape) are left pending rather than guessed."""
    line = rec.get("line")
    ou = str(rec.get("ou") or "").strip().casefold()
    player = str(rec.get("player") or "").strip()
    if not player:  # legacy: the PlayerProp path parked the player in 'side'
        cand = str(rec.get("side") or "").strip()
        if cand and cand.casefold() not in ("over", "under"):
            player = cand
    if not player or line is None:
        return {"status": "pending", "note": "missing player/line"}
    if ou not in ("over", "under"):
        return {"status": "pending", "note": "side not stored"}
    try:
        line = float(line)
    except (TypeError, ValueError):
        return {"status": "pending", "note": "bad line"}
    token = _stat_token(rec)
    if not token:
        return {"status": "pending", "note": "unknown market"}
    game = str(rec.get("game") or "")
    day = str(rec.get("day") or "")
    try:
        season = int(str(rec.get("year") or (day[:4] if day else "") or 0))
    except (TypeError, ValueError):
        season = 0
    if not season:
        return {"status": "pending", "note": "no season"}
    parts = game.split(" @ ")
    if len(parts) != 2:
        return {"status": "pending", "note": "bad game label"}
    away_ab, home_ab = _abbr(parts[0]), _abbr(parts[1])
    val, week = _resolve_value(player, token, (away_ab, home_ab),
                               season, day, away_ab, home_ab)
    if val is None:
        return {"status": "pending", "note": "no stat row"}
    if val > line:
        got = "over"
    elif val < line:
        got = "under"
    else:
        return {"status": "push", "value": val, "stat": token, "week": week}
    return {"status": "won" if got == ou else "lost",
            "value": val, "stat": token, "week": week}

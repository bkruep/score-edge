"""Home Run Hunt — MLB home-run projections (stats-based).

Our odds feed does not carry MLB player props, so this board is an honest
*stats projection*, not a market edge: for every hitter in a game we estimate a
per-game home-run rate from the player's own HR-per-plate-appearance pace and a
projected plate-appearance count, then a Poisson "hits at least one" chance.

Data comes from the league's public stats API (statsapi.mlb.com):
  * schedule    -> /api/v1/schedule?sportId=1&date=...&hydrate=probablePitcher,team
  * team hit.   -> /api/v1/stats?stats=season&group=hitting&season=YYYY&sportId=1&teamId=..

Honest labels: the probability is an estimate built from this hitter's HR rate
and expected trips to the plate. It does not know about the opposing pitcher,
park, weather, or lineup slot, and it is not a price.
"""
from __future__ import annotations

import math
import time
from datetime import date as _date
from typing import Any

import requests

import toolbox

_BASE = "https://statsapi.mlb.com/api/v1"
_CACHE = "_hr_hunt.json"
_CACHE_TTL = 900.0


def _today() -> str:
    return _date.today().isoformat()


def _get(path: str, params: dict) -> Any:
    r = requests.get(f"{_BASE}/{path}", params=params, timeout=25,
                     headers={"User-Agent": "ScoreEdge/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_schedule(date_str: str) -> list[dict]:
    d = _get("schedule", {"sportId": 1, "date": date_str,
                          "hydrate": "probablePitcher,team"})
    games: list[dict] = []
    for day in (d.get("dates") or []):
        for g in (day.get("games") or []):
            t = g.get("teams") or {}
            away = (t.get("away") or {}).get("team") or {}
            home = (t.get("home") or {}).get("team") or {}
            ap = ((t.get("away") or {}).get("probablePitcher") or {}).get("fullName", "")
            hp = ((t.get("home") or {}).get("probablePitcher") or {}).get("fullName", "")
            games.append({
                "gamePk": g.get("gamePk"),
                "away_id": away.get("id"), "away": away.get("abbreviation") or away.get("name", ""),
                "away_name": away.get("name", ""),
                "home_id": home.get("id"), "home": home.get("abbreviation") or home.get("name", ""),
                "home_name": home.get("name", ""),
                "away_pitcher": ap or "TBD", "home_pitcher": hp or "TBD",
                "status": (g.get("status") or {}).get("detailedState", ""),
                "start": (g.get("gameDate") or "")[11:16],
            })
    return games


def _team_hitters(team_id: int, season: int) -> list[dict]:
    d = _get("stats", {"stats": "season", "group": "hitting", "season": season,
                       "sportId": 1, "teamId": team_id, "playerPool": "ALL", "limit": 200})
    stats = (d.get("stats") or [{}])
    splits = (stats[0].get("splits") if stats else []) or []
    out: list[dict] = []
    for s in splits:
        st = s.get("stat") or {}
        gp = int(st.get("gamesPlayed") or 0)
        pa = int(st.get("plateAppearances") or 0)
        if gp <= 0 or pa <= 0:
            continue
        name = (s.get("player") or {}).get("fullName", "")
        if not name:
            continue
        out.append({
            "name": name,
            "pos": (s.get("position") or {}).get("abbreviation", ""),
            "gp": gp,
            "hr": int(st.get("homeRuns") or 0),
            "pa": pa,
            "avg": st.get("avg", ""),
            "slg": st.get("slg", ""),
            "pa_per_game": pa / gp,
            "hr_per_pa": int(st.get("homeRuns") or 0) / pa,
        })
    return out


def _project(row: dict) -> tuple[float, float]:
    """(expected HR this game, P(at least one HR)) — Poisson."""
    pa = min(max(row["pa_per_game"], 3.4), 4.8)
    rate = max(0.0, row["hr_per_pa"] * pa)
    return rate, 1.0 - math.exp(-rate)


def build_board(date_str: str = "", per_team: int = 5, force: int = 0) -> dict:
    date_str = date_str or _today()
    if not force:
        cached = toolbox._load(_CACHE, {})
        if (cached.get("date") == date_str and cached.get("result")
                and time.time() - float(cached.get("ts") or 0) < _CACHE_TTL):
            return cached["result"]

    season = int(date_str[:4])
    try:
        games = fetch_schedule(date_str)
    except Exception as exc:
        return {"ok": False, "error": f"schedule failed: {exc}", "date": date_str,
                "season": season, "games": [], "games_n": 0, "players_n": 0,
                "fetched_at": ""}

    if not games:
        result = {"ok": False, "error": "no_games", "date": date_str, "season": season,
                  "games": [], "games_n": 0, "players_n": 0,
                  "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
        toolbox._save(_CACHE, {"ts": time.time(), "date": date_str, "result": result})
        return result

    hit_cache: dict[int, list[dict]] = {}

    def hitters(team_id: int) -> list[dict]:
        if team_id not in hit_cache:
            try:
                hit_cache[team_id] = _team_hitters(team_id, season)
            except Exception:
                hit_cache[team_id] = []
        return hit_cache[team_id]

    out_games: list[dict] = []
    players_n = 0
    for g in games:
        gd: dict[str, Any] = {
            "matchup": f"{g['away']} @ {g['home']}",
            "away": g["away"], "home": g["home"],
            "away_name": g["away_name"], "home_name": g["home_name"],
            "away_pitcher": g["away_pitcher"], "home_pitcher": g["home_pitcher"],
            "start": g["start"], "status": g["status"], "teams": [],
        }
        for side_abbr, side_name, team_id, opp_pitcher in (
                (g["away"], g["away_name"], g["away_id"], g["home_pitcher"]),
                (g["home"], g["home_name"], g["home_id"], g["away_pitcher"])):
            rows = []
            for h in hitters(team_id):
                rate, p = _project(h)
                rows.append({**h, "team": side_abbr, "opp_pitcher": opp_pitcher or "TBD",
                             "proj_hr": round(rate, 3),
                             "prob": round(p * 100.0, 1)})
            rows.sort(key=lambda r: (r["prob"], r["hr"]), reverse=True)
            rows = rows[:per_team]
            players_n += len(rows)
            gd["teams"].append({"abbr": side_abbr, "name": side_name, "players": rows})
        out_games.append(gd)

    result = {
        "ok": True, "error": "", "date": date_str, "season": season,
        "games": out_games, "games_n": len(out_games), "players_n": players_n,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    toolbox._save(_CACHE, {"ts": time.time(), "date": date_str, "result": result})
    return result

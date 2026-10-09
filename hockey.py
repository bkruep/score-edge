"""Goal Hunt — NHL anytime-goal projections (stats-based).

Our odds feed does not carry NHL player props, so this board is an honest
*stats projection*, not a market edge: for every skater in a game we estimate a
per-game goal rate from the player's own scoring pace and shot volume, then a
Poisson "scores at least one" probability, and surface the single most-likely
ten across the whole slate (filterable to one game).

Data comes from the league's public stats API (api-web.nhle.com):
  * schedule      -> https://api-web.nhle.com/v1/schedule/{date}
  * team skaters  -> https://api-web.nhle.com/v1/club-stats/{ABBR}/{season}/{gameType}

Honest labels: the probability is an estimate built from this player's rate and
shot volume, regressed toward a league-average shooting percentage. It does not
know about injuries, lineup changes, power-play time, or tonight's goalie.
"""
from __future__ import annotations

import math
import time
from datetime import date as _date
from typing import Any

import requests

import toolbox

_BASE = "https://api-web.nhle.com/v1"
_CACHE = "_goal_hunt.json"
_CACHE_TTL = 900.0
_LEAGUE_SHG = 0.105  # league-average goals per shot on goal


def _today() -> str:
    return _date.today().isoformat()


def _season_for(date_str: str) -> str:
    y, m = int(date_str[:4]), int(date_str[5:7])
    return f"{y}{y + 1}" if m >= 9 else f"{y - 1}{y}"


def _prev_season(season: str) -> str:
    a, b = int(season[:4]), int(season[4:])
    return f"{a - 1}{b - 1}"


def _get(url: str) -> Any:
    r = requests.get(url, timeout=25, headers={"User-Agent": "ScoreEdge/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_schedule(date_str: str) -> list[dict]:
    d = _get(f"{_BASE}/schedule/{date_str}")
    games: list[dict] = []
    for day in (d.get("gameWeek") or []):
        if day.get("date") != date_str:
            continue
        for g in (day.get("games") or []):
            away = g.get("awayTeam") or {}
            home = g.get("homeTeam") or {}
            games.append({
                "id": g.get("id"),
                "away": away.get("abbrev") or "",
                "away_name": (away.get("placeName") or {}).get("default", ""),
                "home": home.get("abbrev") or "",
                "home_name": (home.get("placeName") or {}).get("default", ""),
                "start": (g.get("startTimeUTC") or "")[11:16],
                "state": g.get("gameState") or "",
            })
    return games


def _team_skaters(abbr: str, season: str) -> list[dict]:
    d = _get(f"{_BASE}/club-stats/{abbr}/{season}/2")
    out: list[dict] = []
    for s in (d.get("skaters") or []):
        gp = int(s.get("gamesPlayed") or 0)
        if gp <= 0:
            continue
        name = f"{(s.get('firstName') or {}).get('default', '')} " \
               f"{(s.get('lastName') or {}).get('default', '')}".strip()
        shots = int(s.get("shots") or 0)
        out.append({
            "name": name or "—",
            "pos": s.get("positionCode") or "",
            "gp": gp,
            "goals": int(s.get("goals") or 0),
            "shots": shots,
            "spg": shots / gp,
            "gpg": int(s.get("goals") or 0) / gp,
        })
    return out


def _project(row: dict) -> tuple[float, float]:
    """(expected goals this game, P(at least one goal)) — Poisson."""
    xg = row["spg"] * _LEAGUE_SHG
    rate = max(0.0, 0.65 * row["gpg"] + 0.35 * xg)
    return rate, 1.0 - math.exp(-rate)


def build_board(date_str: str = "", game: str = "", top: int = 10,
                force: int = 0) -> dict:
    """Top-`top` most-likely goal scorers across the slate (or one game)."""
    date_str = date_str or _today()
    top = max(1, min(int(top or 10), 25))
    ck = f"{date_str}|{game}|{top}"
    if not force:
        cached = toolbox._load(_CACHE, {})
        if (cached.get("key") == ck and cached.get("result")
                and time.time() - float(cached.get("ts") or 0) < _CACHE_TTL):
            return cached["result"]

    season = _season_for(date_str)
    prev = _prev_season(season)
    try:
        games = fetch_schedule(date_str)
    except Exception as exc:
        return {"ok": False, "error": f"schedule failed: {exc}", "date": date_str,
                "season": season, "games": [], "active_game": game, "rows": [],
                "count": 0, "fetched_at": ""}

    if not games:
        result = {"ok": False, "error": "no_games", "date": date_str, "season": season,
                  "games": [], "active_game": game, "rows": [], "count": 0,
                  "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
        toolbox._save(_CACHE, {"ts": time.time(), "key": ck, "result": result})
        return result

    sk_cache: dict[str, list[dict]] = {}

    def skaters(abbr: str) -> list[dict]:
        if abbr not in sk_cache:
            try:
                sk = _team_skaters(abbr, season)
            except Exception:
                sk = []
            if sum(x["gp"] for x in sk) < 30:
                try:
                    prior = _team_skaters(abbr, prev)
                except Exception:
                    prior = []
                if prior:
                    sk = prior
            sk_cache[abbr] = sk
        return sk_cache[abbr]

    game_names: list[str] = []
    rows: list[dict] = []
    for g in games:
        matchup = f"{g['away']} @ {g['home']}"
        game_names.append(matchup)
        for side_abbr, side_name, opp in ((g["away"], g["away_name"], g["home_name"]),
                                          (g["home"], g["home_name"], g["away_name"])):
            for sk in skaters(side_abbr):
                rate, p = _project(sk)
                rows.append({
                    "player": sk["name"], "team": side_abbr, "team_name": side_name,
                    "opp": opp, "pos": sk["pos"], "game": matchup,
                    "gp": sk["gp"], "goals": sk["goals"],
                    "spg": round(sk["spg"], 1), "gpg": round(sk["gpg"], 2),
                    "proj_goals": round(rate, 3), "prob": round(p * 100.0, 1),
                })

    if game:
        rows = [r for r in rows if r["game"] == game]
    rows.sort(key=lambda r: (r["prob"], r["gpg"]), reverse=True)
    rows = rows[:top]

    result = {
        "ok": True, "error": "", "date": date_str, "season": season,
        "games": game_names, "active_game": game, "rows": rows, "count": len(rows),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    toolbox._save(_CACHE, {"ts": time.time(), "key": ck, "result": result})
    return result

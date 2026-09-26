from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import requests

DK_LOBBY_URL = "https://www.draftkings.com/lobby/getcontests"
DK_DRAFTABLES_URL = "https://api.draftkings.com/draftgroups/v1/draftgroups/{draftgroup_id}/draftables"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
}

SALARY_CAP = 50000
MAIN_POINTS_STAT_ID = 90

_CACHE: dict[str, tuple[float, object]] = {}
TTL = 600


def _cached(k):
    if k in _CACHE:
        ts, v = _CACHE[k]
        if datetime.now().timestamp() - ts < TTL:
            return v
    return None


def _cache(k, v):
    _CACHE[k] = (datetime.now().timestamp(), v)


def _parse_dk_time(raw) -> str:
    m = re.search(r"-?\d+", str(raw))
    if not m:
        return ""
    try:
        return datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc).strftime("%b %d, %I:%M %p ET")
    except Exception:
        return ""


def projection_slate(stats: "list[dict]", games: "list[dict]" | None = None,
                     slate_label: str = "projection-priced") -> tuple[dict, str]:
    """Build a DFS DFS slate from our own live-fantasy projections.

    Used when DFS_DATA_SOURCE != dk (no licensed DK pricing feed): we price
    each player off our per-game PPR projection with a position-aware salary
    curve, then let dfs.build_lineup solve inside the $50K cap. Honest label:
    this is NOT official DraftKings pricing, it is our own projection-priced
    slate so the board stays useful without a paid provider.

    Returns (slate, note). slate has the same shape dfs.fetch_slate produces
    ({draftgroup_id/slate_name/start_time/players[...]/fetched_at}) so the
    template renders identically.
    """
    # Per-position salary curve (salary = base + slope * proj) chosen so the
    # $50K cap stays binding and every slot resolves to something real.
    curve = {
        "QB": (4000, 46.0), "RB": (3400, 42.0), "WR": (3400, 44.0),
        "TE": (3200, 52.0), "FLEX": (3200, 44.0),
    }
    max_sal = {"QB": 9700, "RB": 8800, "WR": 8800, "TE": 7600, "FLEX": 8000}
    rows: list[dict] = []
    for s in stats:
        proj = float(s.get("pts_ppr") or s.get("fantasy_points_ppr") or s.get("proj") or 0.0)
        pos = str(s.get("position") or "").strip().upper()
        slot = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}.get(pos, "FLEX")
        base, slope = curve.get(slot, curve["FLEX"])
        sal = int(round(base + slope * proj))
        sal = max(2900, min(sal, max_sal.get(slot, 8000)))
        rows.append({
            "name": str(s.get("player_name") or s.get("name") or ""),
            "position": pos,
            "team": str(s.get("recent_team") or s.get("team_name") or s.get("team") or ""),
            "salary": sal,
            "avg_points": round(proj, 2),
            "value_per_1k": round(proj / sal * 1000.0, 2) if sal else 0.0,
            "_valid": True,
            "_slot": slot,
            "_proj": proj,
            "_salary": sal,
        })

    # A neutral DST row per team: our chain tracks skill players, not defenses,
    # so give every team's DST the league-mean projection (labeled as such) so
    # the solver still has a DST pool to fill the DST slot.
    if games is not None:
        teams = sorted({str(g.get("team") or g.get("team_name") or "").strip().upper()
                        for g in games if (g.get("team") or g.get("team_name"))})
    else:
        teams = sorted({str(r.get("team") or "").strip().upper() for r in rows if r.get("team")})
    if not teams:
        teams = ["QB " + str(r.get("team") or "") for r in rows]  # placeholder
    for t in teams:
        rows.append({
            "name": f"{t} DEFENSE",
            "position": "DST",
            "team": t,
            "salary": 2500,
            "avg_points": 8.0,
            "value_per_1k": round(8.0 / 2500 * 1000.0, 2),
            "_valid": True,
            "_slot": "DST",
            "_proj": 8.0,
            "_salary": 2500,
        })

    if not rows:
        return {}, "Not enough projection data yet to price a slate."

    slate = {
        "draftgroup_id": "projection",
        "slate_name": "Projection-Priced Slate",
        "start_time": "by projection",
        "players": rows,
        "fetched_at": "local projections",
    }
    return slate, ("Projection-priced slate — salaries derived from our own live "
                   "fantasy projections, NOT official DraftKings pricing.")

def _find_main_slate() -> tuple[int, str, str]:
    session = requests.Session()
    r = session.get(DK_LOBBY_URL + "?sport=NFL&sortBy=StartDate", headers=HEADERS, timeout=20)
    r.raise_for_status()
    contests = r.json().get("Contests") or []
    classic = [c for c in contests if str(c.get("gameType", "")).casefold() == "classic"]
    counts = Counter(c.get("dg") for c in classic)
    if not counts:
        raise RuntimeError("No classic NFL contests found")
    dg, _ = counts.most_common(1)[0]
    sample = next(c for c in classic if c.get("dg") == dg)
    return int(dg), str(sample.get("n", "")).strip(), _parse_dk_time(sample.get("sd"))


def data_source() -> str:
    """Which DFS data source is enabled.

    Values (read from env DFS_DATA_SOURCE):
      - 'dk'    : pull live salaries/projections from DraftKings' public API.
                 PERSONAL USE ONLY - not safe for a public/paid product.
      - 'none'  : disable live pulls (default in production). DFS page shows a
                 required-provider message instead of scraping.
    """
    import os
    return os.getenv("DFS_DATA_SOURCE", "none").strip().lower()


def fetch_slate() -> dict:
    """Fetch the current NFL main-slate players with DraftKings salaries."""
    if data_source() != "dk":
        raise RuntimeError(
            "DFS live data is disabled. Set DFS_DATA_SOURCE=dk for local "
            "personal use only (DraftKings' public API is not licensed for a "
            "public/paid product)."
        )
    cache_key = "dk_slate"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    dg, name, start = _find_main_slate()
    session = requests.Session()
    r = session.get(DK_DRAFTABLES_URL.format(draftgroup_id=dg), headers=HEADERS, timeout=20)
    r.raise_for_status()
    payload = r.json()

    players: dict[str, dict] = {}
    for entry in payload.get("draftables") or []:
        if entry.get("isDisabled"):
            continue
        name_ = str(entry.get("displayName", "")).strip()
        pos = str(entry.get("position", "")).strip().upper()
        if not name_ or not pos:
            continue
        try:
            salary = int(entry.get("salary"))
        except (TypeError, ValueError):
            continue
        if salary <= 0:
            continue
        team = str(entry.get("teamAbbreviation", "")).strip().upper()
        pid = str(entry.get("playerDkId", "")).strip()
        if not pid:
            continue
        avg = 0.0
        for attr in entry.get("draftStatAttributes") or []:
            if attr.get("id") == MAIN_POINTS_STAT_ID:
                try:
                    avg = float(attr.get("value"))
                except (TypeError, ValueError):
                    avg = 0.0
        players[f"{pid}-{pos}"] = {
            "player_id": pid, "name": name_, "position": pos, "team": team,
            "salary": salary, "avg_points": avg,
            "value_per_1k": (avg / salary * 1000) if salary else 0.0,
        }

    result = {
        "draftgroup_id": dg, "slate_name": name, "start_time": start,
        "players": list(players.values()),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    _cache(cache_key, result)
    return result


FLEX_POS = {"RB", "WR", "TE"}

# Roster structure for DraftKings classic: position -> how many must start.
ROSTER_REQ = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1, "DEF": 1}
FLEX_SLOT = 1  # one additional RB/WR/TE

# Each roster position can draw from these real player positions.
POSITION_POOLS = {
    "QB": ["QB"],
    "RB": ["RB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "DST": ["DST", "DEF"],
    "FLEX": ["RB", "WR", "TE"],
}


def _norm_players(players: list[dict]) -> list[dict]:
    out = []
    for i, p in enumerate(players):
        pos = p.get("position", "").upper()
        slot = "DST" if pos in ("DST", "DEF") else pos
        rec = dict(p)
        rec["_i"] = i
        rec["_slot"] = slot
        rec["_proj"] = float(p.get("avg_points") or 0.0)
        rec["_salary"] = int(p.get("salary") or 0)
        rec["_valid"] = slot in {"QB", "RB", "WR", "TE", "DST"} and rec["_salary"] > 0
        out.append(rec)
    return out


def build_lineup(players: list[dict], extra_players: list[str] | None = None) -> dict:
    """
    Optimize a DraftKings classic lineup via integer programming.

    Roster: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST (9 players).

    Two-stage MILP:
      1. Maximize projected points subject to the $50K salary cap and roster rules.
      2. Holding points at the maximum, maximize salary used (push toward the cap).

    This fixes the old greedy-VORP bug that left large cap unused: the solver
    picks the highest-scoring combination that fits, then fills unused salary.
    """
    try:
        from pulp import (
            LpProblem, LpVariable, LpMaximize, LpBinary, LpStatusOptimal,
            PULP_CBC_CMD, value as lp_value,
        )
    except Exception:
        return {"error": "pulp (MILP solver) is not installed", "lineup": [], "count": 0}

    pool = _norm_players(players)
    eligible = [p for p in pool if p["_valid"]]
    if len(eligible) < 9:
        return {"error": "Not enough valid player data to build a lineup", "lineup": [], "count": 0}

    ids = [p["_i"] for p in eligible]
    slots = ["QB", "RB", "WR", "TE", "FLEX", "DST"]

    # Assignment variables y[player_i, slot]
    y: dict = {}
    y_by_player: dict[int, list] = {i: [] for i in ids}
    for p in eligible:
        for s in [sl for sl in slots if p["_slot"] in POSITION_POOLS[sl]]:
            v = LpVariable(f"y_{s}_{p['_i']}", cat=LpBinary)
            y[(p["_i"], s)] = v
            y_by_player[p["_i"]].append((s, v))

    def selected(p, slot=None):
        if slot is not None:
            return y[(p["_i"], slot)]
        return sum(v for (_, v) in y_by_player[p["_i"]])

    def add_roster_constraints(prob, pts_bound=None):
        for i in ids:
            prob += sum(v for (_, v) in y_by_player[i]) <= 1
        prob += sum(y[(p["_i"], "QB")] for p in eligible if p["_slot"] == "QB") == 1
        prob += sum(y[(p["_i"], "RB")] for p in eligible if p["_slot"] == "RB") == 2
        prob += sum(y[(p["_i"], "WR")] for p in eligible if p["_slot"] == "WR") == 3
        prob += sum(y[(p["_i"], "TE")] for p in eligible if p["_slot"] == "TE") == 1
        prob += sum(y[(p["_i"], "FLEX")] for p in eligible if p["_slot"] in FLEX_POS) == 1
        prob += sum(y[(p["_i"], "DST")] for p in eligible if p["_slot"] == "DST") == 1
        prob += sum(p["_salary"] * selected(p) for p in eligible) <= SALARY_CAP
        if pts_bound is not None:
            prob += sum(p["_proj"] * selected(p) for p in eligible) >= pts_bound

    # Stage 1: maximize points
    prob1 = LpProblem("dfs_points", LpMaximize)
    add_roster_constraints(prob1)
    prob1 += sum(p["_proj"] * selected(p) for p in eligible)
    st1 = prob1.solve(PULP_CBC_CMD(msg=False, timeLimit=25))
    if st1 != LpStatusOptimal or prob1.status != LpStatusOptimal:
        return {"error": "No feasible lineup could be solved", "lineup": [], "count": 0}
    max_pts = round(lp_value(prob1.objective), 3)

    # Stage 2: keep max points, maximize salary used (near-cap)
    prob2 = LpProblem("dfs_capfill", LpMaximize)
    add_roster_constraints(prob2, pts_bound=max_pts - 0.001)
    prob2 += sum(p["_salary"] * selected(p) for p in eligible)
    st2 = prob2.solve(PULP_CBC_CMD(msg=False, timeLimit=25))
    if st2 != LpStatusOptimal or prob2.status != LpStatusOptimal:
        return {"error": "No feasible lineup could be solved", "lineup": [], "count": 0}

    salary_used = int(round(lp_value(prob2.objective)))
    lineup = [p for p in eligible if any(v.value() and v.value() > 0.5 for (_, v) in y_by_player[p["_i"]])]

    def slot_of(o):
        return next((s for s, v in y_by_player[o["_i"]] if v.value() and v.value() > 0.5), None) or o["_slot"]

    slot_order = ["QB", "RB", "WR", "FLEX", "TE", "DST"]
    lineup.sort(key=lambda o: slot_order.index(slot_of(o)) if slot_of(o) in slot_order else 99)

    rows = []
    for o in lineup:
        rows.append({
            "player_id": o.get("player_id"), "name": o.get("name"),
            "position": o.get("position"), "team": o.get("team"),
            "salary": o["_salary"], "avg_points": round(o["_proj"], 1),
            "value_per_1k": round(o["_proj"] / o["_salary"] * 1000, 1) if o["_salary"] else 0.0,
            "slot": slot_of(o),
        })

    return {
        "lineup": rows,
        "salary_used": salary_used,
        "cap": SALARY_CAP,
        "salary_remaining": SALARY_CAP - salary_used,
        "projected": round(float(max_pts), 1),
        "count": len(rows),
        "optimal_points": max_pts,
    }

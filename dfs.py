from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import requests

DK_LOBBY_URL = "https://www.draftkings.com/lobby/getcontests"
DK_DRAFTABLES_URL = "https://api.draftkings.com/draftgroups/v1/draftgroups/{draftgroup_id}/draftables"

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.draftkings.com/lobby",
    "Origin": "https://www.draftkings.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Chromium";v="126", "Not)A;Brand";v="24"',
    "sec-fetch-site": "same-site",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
}

SALARY_CAP = 50000

# DraftKings classic roster per sport: the exact list of starting slots plus
# which player positions may fill each one. DK reports multi-eligibility as a
# combined position ("PG/SG"), which counts as eligible for either side. The
# slot list (not a per-position count) is what the MILP constrains, so one
# solver covers all three shapes.
ROSTERS: dict[str, dict] = {
    "nfl": {
        "label": "NFL", "dk_sport": "NFL", "icon": "\U0001f3c8",
        "proj_stat_id": 90,
        "summary": "1 QB · 2 RB · 3 WR · 1 TE · 1 FLEX · 1 DST",
        "filter_positions": ["QB", "RB", "WR", "TE", "DST"],
        "slots": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
        "pools": {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
                  "FLEX": {"RB", "WR", "TE"}, "DST": {"DST", "DEF"}},
        # Construction rules from the five-season Milly Maker winner study
        # (2021-2025, 89 Sunday winners). Both are toggles, not truth:
        #   qb_stack   85.4% of Sunday winners paired the QB with >=1 same-team
        #              WR/TE (RBs excluded from the stack).
        #   bring_back 42.7% used an opposing offensive player in the QB's game
        #              — explicitly optional in the research, so off by default.
        "rules": {"qb_stack": True, "bring_back": False},
    },
    "nba": {
        "label": "NBA", "dk_sport": "NBA", "icon": "\U0001f3c0",
        "proj_stat_id": 219,
        "summary": "PG · SG · SF · PF · C · G · F · UTIL",
        "filter_positions": ["PG", "SG", "SF", "PF", "C"],
        "slots": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        "pools": {"PG": {"PG"}, "SG": {"SG"}, "SF": {"SF"}, "PF": {"PF"},
                  "C": {"C"}, "G": {"PG", "SG"}, "F": {"SF", "PF"},
                  "UTIL": {"PG", "SG", "SF", "PF", "C"}},
    },
    "golf": {
        "label": "Golf", "dk_sport": "GOLF", "icon": "⛳",
        "proj_stat_id": 795,
        "summary": "6 golfers",
        "filter_positions": ["G"],
        "slots": ["G", "G", "G", "G", "G", "G"],
        "pools": {"G": {"G", "GOLF"}},
    },
}
DFS_SPORTS = ["nfl", "nba", "golf"]


def roster(sport: str = "nfl") -> dict:
    return ROSTERS.get(sport, ROSTERS["nfl"])

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
        return datetime.fromtimestamp(int(m.group(0)) / 1000, tz=timezone.utc).strftime("%b %d, %I:%M %p ET")
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

def _find_main_slate(sport: str = "nfl") -> tuple[int, str, str]:
    cfg = roster(sport)
    session = requests.Session()
    r = session.get(DK_LOBBY_URL + f"?sport={cfg['dk_sport']}&sortBy=StartDate",
                    headers=HEADERS, timeout=20)
    r.raise_for_status()
    contests = r.json().get("Contests") or []
    classic = [c for c in contests if str(c.get("gameType", "")).casefold() == "classic"]
    counts = Counter(c.get("dg") for c in classic)
    if not counts:
        kinds = sorted({str(c.get("gameType", "")).strip() for c in contests if c.get("gameType")})
        if kinds:
            raise RuntimeError(
                f"No open {cfg['label']} Classic slate right now — DraftKings is "
                f"only showing {', '.join(kinds[:3])} (the main slate has locked)"
            )
        raise RuntimeError(f"No classic {cfg['label']} contests found")
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


# DraftKings' own availability flag on this slate ("status" on each draftable).
# IR/OUT/D (doubtful) are treated as "will not play" and dropped from the pool.
# Q (questionable) normally suits up, so those players are kept but flagged in
# the UI so you can see them before you submit.
INELIGIBLE_STATUSES = {"IR", "OUT", "D"}


def _est_ownership(rec: dict) -> float:
    """Heuristic projected-ownership proxy (1-45%). Cheap + high-FPPG -> chalk;
    starting QBs soak up the most GPP ownership. Pure estimate."""
    sal = float(rec.get("salary") or 0)
    v = float(rec.get("value_per_1k") or 0)
    pos = str(rec.get("position") or "")
    cheap = max(0.0, 1.0 - max(0, sal - 3000) / 6500.0)
    base = 2.0 + 1.05 * v + 7.0 * cheap
    if pos == "QB":
        base *= 1.9
    elif pos == "DST":
        base *= 0.7
    if rec.get("dk_status") == "Q":
        base *= 0.7
    return round(min(45.0, max(1.0, base)), 1)


def fetch_slate(sport: str = "nfl") -> dict:
    """Fetch the current main-slate players with DraftKings salaries."""
    if data_source() != "dk":
        raise RuntimeError(
            "DFS live data is disabled. Set DFS_DATA_SOURCE=dk for local "
            "personal use only (DraftKings' public API is not licensed for a "
            "public/paid product)."
        )
    cfg = roster(sport)
    cache_key = f"dk_slate_{sport}_v3"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    dg, name, start = _find_main_slate(sport)
    session = requests.Session()
    r = session.get(DK_DRAFTABLES_URL.format(draftgroup_id=dg), headers=HEADERS, timeout=20)
    r.raise_for_status()
    payload = r.json()

    # team -> opposing team, from the slate's competition list ("CIN @ MIA").
    # Needed for the research's optional bring-back rule.
    opp: dict[str, str] = {}
    game_for_team: dict[str, str] = {}
    home_of_game: dict[str, str] = {}
    for comp in payload.get("competitions") or []:
        home = str((comp.get("homeTeam") or {}).get("abbreviation", "")).strip().upper()
        away = str((comp.get("awayTeam") or {}).get("abbreviation", "")).strip().upper()
        if home and away:
            opp[home] = away
            opp[away] = home
            gid = str(comp.get("competitionId", ""))
            if gid:
                game_for_team[home] = game_for_team[away] = gid
                home_of_game[gid] = home

    players: dict[str, dict] = {}
    excluded: dict[str, tuple[str, str]] = {}
    for entry in payload.get("draftables") or []:
        if entry.get("isDisabled"):
            continue
        name_ = str(entry.get("displayName", "")).strip()
        pos = str(entry.get("position", "")).strip().upper()
        if not name_ or not pos:
            continue
        raw_status = entry.get("status")
        status = str(raw_status).strip().upper() if raw_status else ""
        if status in INELIGIBLE_STATUSES:
            excluded[str(entry.get("playerDkId", ""))] = (name_, status)
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
            if attr.get("id") == cfg["proj_stat_id"]:
                try:
                    avg = float(attr.get("value"))
                except (TypeError, ValueError):
                    avg = 0.0
        comp = entry.get("competition") or {}
        if not comp:
            comp = (entry.get("competitions") or [{}])[0]
        gid = str(comp.get("competitionId", "")).strip()
        gname = str(comp.get("name", "")).strip() or gid
        raw_news = entry.get("newsStatus")
        news = str(raw_news).strip() if raw_news else ""
        players[f"{pid}-{pos}"] = {
            "player_id": pid, "name": name_, "position": pos, "team": team,
            "opponent": opp.get(team, ""),
            "game_id": gid, "game": re.sub(r"\s+", "", gname),
            "news": news, "dk_status": status,
            "salary": salary, "avg_points": avg,
            "value_per_1k": (avg / salary * 1000) if salary else 0.0,
        }

    # Starting-QB guard: DK prices a healthy team's #1 QB higher than the
    # backup, so we keep only the max-salary healthy QB per team and drop the
    # rest. This stops a cheap backup (Drew Lock, 17.1 FPPG at $4,000) from
    # masquerading as a value play when the starter (Sam Darnold) is back.
    # When a starter is IR/OUT/D he is already dropped above, so the next
    # max-salary QB becomes the de facto starter and stays eligible.
    qb_tops: dict[str, dict] = {}
    for rec in players.values():
        if rec["position"] != "QB":
            continue
        cur = qb_tops.get(rec["team"])
        if cur is None or rec["salary"] > cur["salary"]:
            qb_tops[rec["team"]] = rec
    dropped_qbs = {}
    for rec in players.values():
        if rec["position"] == "QB" and qb_tops[rec["team"]]["player_id"] != rec["player_id"]:
            dropped_qbs[rec["player_id"]] = rec["name"]
    if dropped_qbs:
        players = {k: v for k, v in players.items() if v["player_id"] not in dropped_qbs}

    # Chalk ownership ESTIMATE (heuristic, not projected-ownership data). Cheap
    # high-FPPG players — especially cheap starting QBs — draw outsized GPP
    # ownership. Used only by the optional fade/chalk-cap controls, both off by
    # default. Never a true ownership feed.
    for rec in players.values():
        rec["est_own"] = _est_ownership(rec)

    # Recent-form blend: mix DK season FPPG with the last-3-week Sleeper average
    # (0.6/0.4) so returning starters, hot/hurt streaks, and roster moves show
    # in the projection. Purely additive — falls back to FPPG when unavailable.
    form_note = ""
    if sport == "nfl":
        recent = {}
        try:
            import data
            recent = data.load_sleeper_recent(3) or {}
        except Exception:
            recent = {}
        if recent:
            form_note = ("Projections blend DraftKings season FPPG with the "
                         "last-3-week Sleeper average (0.6 / 0.4) where a "
                         "player has recent games.")
        for rec in players.values():
            ra = recent.get(str(rec["name"]).strip().lower(), 0.0)
            rec["recent_avg"] = round(ra, 2)
            if ra and ra > 0 and rec["avg_points"] > 0:
                rec["proj"] = round(0.6 * rec["avg_points"] + 0.4 * ra, 3)
                rec["value_per_1k"] = round(rec["proj"] / rec["salary"] * 1000, 2)
            else:
                rec["proj"] = rec["avg_points"]

    # Weather adjustment (NFL only — outdoor stadiums). High wind or rain
    # genuinely changes pass/RB efficiency, so we dampen OUR OWN projection
    # (never the DK price) for players in flagged games, recompute value, and
    # tag the player. Rides the slate cache so it can lag a fresh forecast by
    # one cache TTL; the banner in the UI reads the same flags. Domed stadiums
    # and calm games are untouched. This is our labeled projection layer, not a
    # betting line.
    weather_flags: dict[str, dict] = {}
    if sport == "nfl" and players:
        _gflags: dict[str, dict] = {}
        try:
            import data
            for gid, home in home_of_game.items():
                coords = NFL_STADIUMS.get(str(home).upper())
                if not coords:
                    continue
                w = data.fetch_weather(*coords) or {}
                daily = w.get("daily") or {}
                times = daily.get("time") or []
                if not times:
                    continue
                idx = 3 if len(times) > 3 else len(times) - 1  # slate Sunday
                wind = float((daily.get("wind_speed_10m_max") or [0] * len(times))[idx] or 0)
                precip_mm = float((daily.get("precipitation_sum") or [0] * len(times))[idx] or 0)
                if wind < WIND_FLAG and precip_mm < PRECIP_FLAG:
                    continue
                bits = []
                if wind >= WIND_FLAG:
                    bits.append(f"Wind {wind:.0f} mph")
                if precip_mm >= PRECIP_FLAG:
                    bits.append(f"Rain {precip_mm / 25.4:.2f} in")
                _gflags[gid] = {
                    "home": str(home).upper(),
                    "wind": round(wind, 0),
                    "precip_mm": precip_mm,
                    "precip_in": round(precip_mm / 25.4, 2),
                    "warn": " + ".join(bits),
                }
        except Exception:
            _gflags = {}
        if _gflags:
            wf = {"QB": 1.0, "WR": 1.0, "TE": 1.0, "RB": 1.0}
            for f in _gflags.values():
                if f["wind"] >= WIND_FLAG:
                    for pos in ("QB", "WR", "TE"):
                        wf[pos] = min(wf[pos], 0.92)
                    wf["RB"] = min(wf["RB"], 0.96)
                if f["precip_mm"] >= PRECIP_FLAG:
                    for pos in ("QB", "WR", "TE"):
                        wf[pos] = min(wf[pos], 0.94)
                    wf["RB"] = min(wf["RB"], 0.97)
            for rec in players.values():
                f = _gflags.get(str(rec.get("game_id") or ""))
                if not f or rec["position"] in ("DST",):
                    continue
                factor = wf.get(rec["position"], 1.0)
                if factor >= 1.0:
                    continue
                rec["proj"] = round(rec["proj"] * factor, 3)
                rec["value_per_1k"] = round(rec["proj"] / rec["salary"] * 1000, 2)
                rec["weather_note"] = f["warn"]
            weather_flags = _gflags
            if form_note:
                form_note += " Weather hits dampen the projection (US units)."
            else:
                form_note = "Projections are dampened for flagged high-wind/rain games."

    exc_counts = Counter(s for _, s in excluded.values())
    if dropped_qbs:
        exc_counts["QB backup"] = len(dropped_qbs)
    flagged_q = sorted({p["name"] for p in players.values() if p.get("dk_status") == "Q"})

    result = {
        "draftgroup_id": dg, "slate_name": name, "start_time": start,
        "sport": sport, "label": cfg["label"],
        "players": list(players.values()),
        "games": sorted({str(p.get("game")) for p in players.values() if p.get("game")}),
        "home_of_game": dict(home_of_game),
        "weather_flags": weather_flags,
        "form_note": form_note,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "excluded": {
            "total": len(excluded) + len(dropped_qbs),
            "counts": dict(exc_counts),
            "names": sorted({n for n, _ in excluded.values()} | set(dropped_qbs.values())),
            "questionable": flagged_q,
            "qb_backups": sorted(dropped_qbs.values()),
        },
        "excluded_note": (
            f"Dropped {len(excluded) + len(dropped_qbs)} player entries before "
            + "building ("
            + ", ".join(f"{s} x{c}" for s, c in sorted(exc_counts.items()))
            + ")."
            if (excluded or dropped_qbs)
            else ""
        ),
    }
    _cache(cache_key, result)
    return result


def _norm_players(players: list[dict], sport: str = "nfl") -> list[dict]:
    cfg = roster(sport)
    eligible_slots: set[str] = set()
    for pool in cfg["pools"].values():
        eligible_slots |= set(pool)
    out = []
    for i, p in enumerate(players):
        raw = str(p.get("position", "")).upper()
        positions = {x.strip() for x in raw.replace("DEF", "DST").split("/") if x.strip()}
        rec = dict(p)
        rec["_i"] = i
        rec["_positions"] = positions
        rec["_slot"] = "DST" if "DST" in positions else raw
        rec["_proj"] = float(p.get("proj") or p.get("avg_points") or 0.0)
        rec["_salary"] = int(p.get("salary") or 0)
        rec["_valid"] = bool(positions & eligible_slots) and rec["_salary"] > 0
        out.append(rec)
    return out


MAX_LINEUPS = 10


def build_lineups(players: list[dict], sport: str = "nfl", count: int = 1,
                  rules: dict | None = None) -> dict:
    """
    Build up to MAX_LINEUPS distinct DraftKings classic lineups via ILP.

    One solver for every sport: dfs.ROSTERS[sport]['slots'] is the list of
    starting slots (e.g. NFL QB/RB/RB/WR/WR/WR/TE/FLEX/DST, NBA PG..UTIL,
    Golf 6x G) and 'pools' says which player positions may fill each slot.

    Each lineup is a two-stage MILP:
      1. Maximize projected points subject to the $50K salary cap, roster
         rules, and a pairwise overlap constraint vs. every lineup already
         built (share at most slots - diff players).
      2. Holding points at the maximum, maximize salary used (push toward cap).

    diff starts at ceil(slots/3) — NFL 3, NBA 3, Golf 2 — and relaxes down
    to 1 if the slate can't support it; diff=1 still guarantees the lineups
    are not identical. Returns {"lineups": [...], "requested", "built",
    "min_diff"} or {"error": ...}.
    """
    cfg = roster(sport)
    slots: list[str] = cfg["slots"]
    pools: dict[str, set] = cfg["pools"]
    label = cfg["label"]
    n_slots = len(slots)

    try:
        count = int(count or 1)
    except (TypeError, ValueError):
        count = 1
    count = max(1, min(count, MAX_LINEUPS))

    # Construction rules — defaults come from the roster config (which encodes
    # the Milly Maker winner study), caller may override per request.
    rule_cfg = dict(cfg.get("rules") or {})
    if isinstance(rules, dict):
        rule_cfg.update(rules)
    qb_stack = bool(rule_cfg.get("qb_stack", False))
    bring_back = bool(rule_cfg.get("bring_back", False))
    double_stack = bool(rule_cfg.get("double_stack", False))
    game_stack = bool(rule_cfg.get("game_stack", False))
    fade_chalk = bool(rule_cfg.get("fade_chalk", False))
    chalk_cap = max(0, int(rule_cfg.get("chalk_cap") or 0))
    style = str(rule_cfg.get("style") or "balanced").strip().lower()
    # A game stack is the concentrated version of the two stack halves: pick ONE
    # game and put QB + pass-catchers + bring-back in it. Implies both halves.
    if game_stack:
        qb_stack = True
        bring_back = True

    try:
        from pulp import (
            LpProblem, LpVariable, LpMaximize, LpBinary, LpStatusOptimal,
            PULP_CBC_CMD, value as lp_value,
        )
    except Exception:
        return {"error": "pulp (MILP solver) is not installed",
                "lineups": [], "requested": count, "built": 0}

    FADE_COUNT = 5           # most-chalky players faded by the fade toggle
    CHALK_THRESH = 20        # est_own % that counts as "chalky"
    pool = _norm_players(players, sport)
    eligible = [p for p in pool if p["_valid"]]
    if fade_chalk:
        elite = sorted(eligible, key=lambda p: float(p.get("est_own") or 0),
                       reverse=True)[:FADE_COUNT]
        fade_ids = {p["_i"] for p in elite}
        eligible = [p for p in eligible if p["_i"] not in fade_ids]
    if len(eligible) < n_slots:
        return {"error": f"Not enough valid {label} player data to build a lineup",
                "lineups": [], "requested": count, "built": 0}

    # Same slate + same count + same rules => same lineups. Solving 10 NFL
    # lineups takes tens of seconds, so cache the whole result for the TTL.
    digest = hashlib.blake2b(
        repr([(p.get("player_id"), p["_salary"], round(p["_proj"], 3))
              for p in eligible]).encode(), digest_size=12).hexdigest()
    flags = (f"{int(qb_stack)}{int(bring_back)}{int(double_stack)}"
             f"{int(game_stack)}{int(fade_chalk)}{chalk_cap}{style}")
    cache_key = f"lineups2_{sport}_{count}_{flags}_{digest}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    # Assignment variables y[player_index, slot_index] — built once and shared
    # by every lineup's problems (each solve overwrites the values we read back).
    y: dict = {}
    y_by_player: dict[int, list] = {p["_i"]: [] for p in eligible}
    for p in eligible:
        for si, s in enumerate(slots):
            if p["_positions"] & pools[s]:
                v = LpVariable(f"y_{si}_{p['_i']}", cat=LpBinary)
                y[(p["_i"], si)] = v
                y_by_player[p["_i"]].append((si, v))

    for si in range(n_slots):
        if not any((p["_i"], si) in y for p in eligible):
            return {"error": f"No eligible player for roster slot {slots[si]}",
                    "lineups": [], "requested": count, "built": 0}

    def selected(p):
        return sum(v for (_, v) in y_by_player[p["_i"]])

    target_diff = max(2, -(-n_slots // 3))  # ceil(slots / 3)

    # --- Construction rules (Milly Maker winner study + user toggles) ----------
    # QB stack:  if a team's QB is selected, so is >=1 WR/TE from that team.
    #             (76/89 Sunday winners, 85.4%; RBs don't count as the stack.)
    # Bring-back: if a team's QB is selected, so is >=1 opposing offensive
    #             player in his game (38/89, 42.7%) — optional in the research.
    # Double-stack: >=2 WR/TE with the QB (37.1% of winners).
    # Game-stack:   concentrate both halves into ONE game (>=4 offensive
    #               players via the QB's game, incl. the bring-back side).
    # Chalk:        fade the top-FADE_COUNT most-estimated-own players, or cap
    #               the number of >=20% chalky players (both default off).
    # Stars:        at least 2 players priced at/above the "star" threshold.
    # All are linear: exactly one QB is selected, so for each team T,
    #   sum(pass catchers on T) >= sum(QBs on T)  binds only when T's QB is in.
    qb_by_team: dict[str, list] = {}
    pc_by_team: dict[str, list] = {}
    off_by_team: dict[str, list] = {}
    opp_of: dict[str, str] = {}
    qb_by_game: dict[str, list] = {}
    off_by_game: dict[str, list] = {}
    for p in eligible:
        t = str(p.get("team") or "").strip().upper()
        if not t:
            continue
        pos = p["_positions"]
        if "QB" in pos:
            qb_by_team.setdefault(t, []).append(p)
        if pos & {"WR", "TE"}:
            pc_by_team.setdefault(t, []).append(p)
        if pos - {"DST"}:  # any offensive position (research excludes DST)
            off_by_team.setdefault(t, []).append(p)
        g = str(p.get("game_id") or "").strip()
        if g:
            if "QB" in pos:
                qb_by_game.setdefault(g, []).append(p)
            if pos - {"DST"}:
                off_by_game.setdefault(g, []).append(p)
        o = str(p.get("opponent") or "").strip().upper()
        if o and t not in opp_of:
            opp_of[t] = o

    stack_teams = [t for t in qb_by_team if pc_by_team.get(t)]
    double_teams = [t for t in qb_by_team if len(pc_by_team.get(t, [])) >= 2]
    back_pairs = [(t, opp_of[t]) for t in qb_by_team
                  if opp_of.get(t) and off_by_team.get(opp_of[t])]
    game_stackable = [g for g in qb_by_game if len(off_by_game.get(g, [])) >= 4]
    chalked = [p for p in eligible if float(p.get("est_own") or 0) >= CHALK_THRESH]
    star_thresh = max(7000, int(SALARY_CAP / n_slots * 1.4))
    stars = [p for p in eligible if p["_salary"] >= star_thresh]
    # Only a rule we can actually express on this pool counts as "applied".
    rules_applied = {
        "qb_stack": bool(qb_stack and stack_teams),
        "bring_back": bool(bring_back and back_pairs),
        "double_stack": bool(double_stack and double_teams),
        "game_stack": bool(game_stack and game_stackable),
        "fade_chalk": bool(fade_chalk),
        "chalk_cap": bool(chalk_cap and chalked),
        "stars": bool(style == "stars" and len(stars) >= 2),
    }

    def add_rule_constraints(prob):
        if rules_applied["qb_stack"]:
            for t in stack_teams:
                prob += (sum(selected(p) for p in pc_by_team[t])
                         >= sum(selected(p) for p in qb_by_team[t]))
        if rules_applied["double_stack"]:
            for t in double_teams:
                prob += (sum(selected(p) for p in pc_by_team[t])
                         >= 2 * sum(selected(p) for p in qb_by_team[t]))
        if rules_applied["game_stack"]:
            for g in game_stackable:
                prob += (sum(selected(p) for p in off_by_game[g])
                         >= 4 * sum(selected(p) for p in qb_by_game[g]))
        if rules_applied["bring_back"]:
            for t, o in back_pairs:
                prob += (sum(selected(p) for p in off_by_team[o])
                         >= sum(selected(p) for p in qb_by_team[t]))
        if rules_applied["chalk_cap"]:
            prob += sum(selected(p) for p in chalked) <= chalk_cap
        if rules_applied["stars"]:
            prob += sum(selected(p) for p in stars) >= 2

    def lineup_diag(rows):
        """Research diagnostics for one built lineup (reporting, not forcing)."""
        qb = next((r for r in rows if r["position"] == "QB"), None)
        qteam = str(qb.get("team") or "") if qb else ""
        qopp = opp_of.get(qteam, "") if qteam else ""
        q_game = str(qb.get("game_id") or "") if qb else ""
        qpcs = [r for r in rows if r["position"] in ("WR", "TE") and r.get("team") == qteam]
        stacked = bool(qb) and bool(qpcs)
        double = bool(qb) and len(qpcs) >= 2
        bring = bool(qb) and qopp and any(
            r["position"] != "DST" and r.get("team") == qopp for r in rows)
        gamed = bool(qb) and bool(q_game) and sum(
            1 for r in rows if r["position"] != "DST" and r.get("game_id") == q_game) >= 4
        six_x = sum(1 for r in rows
                    if r["position"] != "DST" and r["value_per_1k"] >= 6)
        owns = [r.get("est_own") or 0 for r in rows if r["position"] != "DST"]
        return {
            "stacked": bool(stacked),
            "double": bool(double),
            "bring_back": bool(bring),
            "game_stacked": bool(gamed),
            "qb_team": qteam,
            "qb_opponent": qopp,
            "six_x": six_x,
            "max_own": round(max(owns), 1) if owns else 0.0,
            "chalky": sum(1 for r in rows if (r.get("est_own") or 0) >= CHALK_THRESH),
            "est_pool_pts": round(sum(r["avg_points"] for r in rows), 1),
        }

    def add_roster_constraints(prob, pts_bound=None):
        for si in range(n_slots):
            prob += sum(y[(p["_i"], si)] for p in eligible if (p["_i"], si) in y) == 1
        for p in eligible:
            prob += sum(v for (_, v) in y_by_player[p["_i"]]) <= 1
        prob += sum(p["_salary"] * selected(p) for p in eligible) <= SALARY_CAP
        if pts_bound is not None:
            prob += sum(p["_proj"] * selected(p) for p in eligible) >= pts_bound
        add_rule_constraints(prob)

    def add_overlap_constraints(prob, prev_ids, diff,
                                banned_ids=None, banned_games=None):
        for ids in prev_ids:
            if not ids:
                continue
            prob += (sum(selected(p) for p in eligible if p["_i"] in ids)
                     <= n_slots - diff)
        for i_ in banned_ids or []:
            prob += selected(eligible[i_]) == 0
        for g in banned_games or []:
            prob += sum(selected(p) for p in eligible
                        if str(p.get("game_id") or "") == g) == 0

    def extract_rows():
        """Read the current variable assignment back into slot-ordered rows."""
        chosen: dict[int, dict] = {}
        for p in eligible:
            for si, v in y_by_player[p["_i"]]:
                if (v.value() or 0) > 0.5:
                    chosen[si] = p
                    break
        if len(chosen) != n_slots:
            return None
        rows = []
        for si in range(n_slots):
            o = chosen[si]
            rows.append({
                "player_id": o.get("player_id"), "name": o.get("name"),
                "position": o.get("position"), "team": o.get("team"),
                "salary": o["_salary"], "avg_points": round(o["_proj"], 1),
                "fppg": round(float(o.get("avg_points") or o["_proj"]), 1),
                "recent_avg": round(float(o.get("recent_avg") or 0.0), 1),
                "est_own": float(o.get("est_own") or 0.0),
                "game": o.get("game") or "", "game_id": o.get("game_id") or "",
                "news": o.get("news") or "",
                "value_per_1k": round(o["_proj"] / o["_salary"] * 1000, 1) if o["_salary"] else 0.0,
                "dk_status": o.get("dk_status") or "",
                "weather_note": o.get("weather_note") or "",
                "slot": slots[si],
            })
        return rows

    def solve_stages(prev_ids, diff, banned_ids=None, banned_games=None):
        """Two-stage solve for one lineup; None when no feasible solution."""
        # gapRel=0.01: CBC proves optimality inside 1% (measured ~3x faster
        # than an exact proof on the NFL pool, same optimal lineup).
        def solve(prob):
            return prob.solve(PULP_CBC_CMD(msg=False, timeLimit=15, gapRel=0.01))

        # Stage 1: maximize points under roster + overlap constraints.
        prob1 = LpProblem("dfs_points", LpMaximize)
        add_roster_constraints(prob1)
        add_overlap_constraints(prob1, prev_ids, diff, banned_ids, banned_games)
        prob1 += sum(p["_proj"] * selected(p) for p in eligible)
        st1 = solve(prob1)
        if st1 != LpStatusOptimal and lp_value(prob1.objective) is None:
            return None
        max_pts = round(float(lp_value(prob1.objective)), 3)
        rows1, salary1 = extract_rows(), None
        if rows1 is None:
            return None
        salary1 = sum(r["salary"] for r in rows1)

        # Stage 2: keep the point level, maximize salary used (near-cap).
        prob2 = LpProblem("dfs_capfill", LpMaximize)
        add_roster_constraints(prob2, pts_bound=max_pts - 0.001)
        add_overlap_constraints(prob2, prev_ids, diff, banned_ids, banned_games)
        prob2 += sum(p["_salary"] * selected(p) for p in eligible)
        st2 = solve(prob2)
        if st2 != LpStatusOptimal and lp_value(prob2.objective) is None:
            rows, salary_used = rows1, salary1
        else:
            rows = extract_rows() or rows1
            salary_used = int(round(float(lp_value(prob2.objective))))
        return rows, salary_used, max_pts

    lineups: list[dict] = []
    prev_ids: list[set[int]] = []
    id_sets: list[set[int]] = []
    min_diff = 0
    error = ""
    relax_note = ""

    # Portfolio hygiene across the lineup set: keep a single player out of too
    # many lineups and a single game out of too many lineups, so one bust can't
    # sink the whole entry. Only meaningful for multi-lineup builds.
    exposure_on = count >= 4
    player_cap = max(2, -(-count // 2))      # 10 lineups -> <=5 appearances
    game_cap = max(3, -(-count * 3 // 5))    # 10 lineups -> <=6 wears per game

    def build_all(with_rules=True, with_exposure=True):
        nonlocal lineups, prev_ids, id_sets, min_diff, error
        lineups, prev_ids, id_sets, min_diff, error = [], [], [], 0, ""
        player_uses = Counter()
        game_uses = Counter()
        if not with_rules:
            for k in rules_applied:
                rules_applied[k] = False
        for idx in range(count):
            banned_ids = []
            banned_games = []
            if with_exposure:
                banned_ids = [p["_i"] for p in eligible
                              if player_uses[p["_i"]] >= player_cap]
                banned_games = [g for g, n in game_uses.items() if n >= game_cap]
            solved = None
            used_diff = 0
            for diff in range(target_diff, 0, -1):
                solved = solve_stages(prev_ids, diff, banned_ids, banned_games)
                if solved:
                    used_diff = diff
                    break
            if not solved:
                if not lineups:
                    return False
                error = (f"Built {len(lineups)} distinct lineups "
                         f"(slate can't support {count} under the overlap rule)")
                break
            rows, salary_used, max_pts = solved
            ids = {p["_i"] for p in eligible
                   for (_si, v) in y_by_player[p["_i"]] if (v.value() or 0) > 0.5}
            for i_ in ids:
                player_uses[i_] += 1
            used_games = {str(r.get("game_id") or "") for r in rows if r.get("game_id")}
            for g in used_games:
                game_uses[g] += 1
            lineups.append({
                "lineup": rows,
                "salary_used": salary_used,
                "cap": SALARY_CAP,
                "salary_remaining": SALARY_CAP - salary_used,
                "projected": round(sum(r["avg_points"] for r in rows), 1),
                "count": len(rows),
                "optimal_points": max_pts,
                "sport": sport,
                "label": label,
                "summary": cfg["summary"],
                "idx": idx + 1,
                "shared_with_first": (len(ids & id_sets[0]) if id_sets else len(ids)),
                "diag": lineup_diag(rows),
            })
            prev_ids.append(ids)
            id_sets.append(ids)
            min_diff = used_diff if not min_diff else min(min_diff, used_diff)
        return True

    if not build_all(True, exposure_on):
        # A rule or exposure cap can make this pool infeasible. Never show a
        # blank page for a preference: relax in layers, retry, and say so.
        dropped = []
        if any(rules_applied.values()):
            dropped.append("construction rules")
            build_all(False, exposure_on)
            relax_note = ("Active construction rules were dropped for this slate — "
                          "no feasible lineup satisfied them.")
        if not lineups and exposure_on:
            dropped.append("exposure caps")
            build_all(False, False)
            if relax_note:
                relax_note += " Exposure caps were also relaxed."
            else:
                relax_note = "Exposure caps were relaxed to fit this slate."

    if not lineups:
        return {"error": error or "No feasible lineup could be solved",
                "lineups": [], "requested": count, "built": 0,
                "rules_applied": rules_applied}

    result = {
        "lineups": lineups,
        "requested": count,
        "built": len(lineups),
        "min_diff": min_diff,
        "rules_applied": rules_applied,
        "relaxed": relax_note,
        "error": error,
    }
    if lineups:
        _cache(cache_key, result)
    return result


def build_lineup(players: list[dict], sport: str = "nfl") -> dict:
    """Build a single lineup (back-compat wrapper around build_lineups)."""
    res = build_lineups(players, sport, 1)
    if res.get("error"):
        return {"error": res["error"], "lineup": [], "count": 0}
    if not res.get("lineups"):
        return {"error": "No feasible lineup could be solved", "lineup": [], "count": 0}
    return res["lineups"][0]


MAX_Q_SWAPS = 3


def suggest_q_swaps(lineups: list[dict], players: list[dict],
                    sport: str = "nfl", rules: dict | None = None) -> dict:
    """For every questionable (Q) player sitting in a built lineup, suggest the
    best affordable replacements that fit the same slot with the rest of the
    lineup pinned and the active construction rules still satisfied.

    Returns {lineup_idx: {player_name: [candidate, ...]}}, each candidate list
    sorted by projected points (top MAX_Q_SWAPS).
    """
    cfg = roster(sport)
    norm = _norm_players(players, sport)
    by_id = {p["player_id"]: p for p in norm}
    out: dict[int, dict[str, list[dict]]] = {}

    def ok(pinned, cand):
        if not rules:
            return True
        qb = None
        for r in list(pinned) + [cand]:
            if r["position"] == "QB":
                qb = r
                break
        if qb is None:
            return True
        allr = list(pinned) + [cand]
        if rules.get("qb_stack") and qb["team"] not in {
                r["team"] for r in allr if r["position"] in ("WR", "TE")}:
            return False
        if ((rules.get("double_stack") or rules.get("game_stack"))
                and sum(1 for r in allr if r["position"] in ("WR", "TE")
                        and r["team"] == qb["team"]) < 2):
            return False
        if (rules.get("bring_back") or rules.get("game_stack")):
            opp = qb.get("opponent") or ""
            if opp and opp not in {r["team"] for r in allr if r["position"] != "DST"}:
                return False
        return True

    for lu in lineups:
        rows = lu.get("lineup") or []
        others = [o for o in (by_id.get(r["player_id"]) for r in rows) if o is not None]
        for rw in rows:
            if rw.get("dk_status") != "Q":
                continue
            qrec = by_id.get(rw["player_id"])
            if qrec is None:
                continue
            slot = rw.get("slot") or rw.get("position") or "FLEX"
            elig = set(cfg["pools"].get(slot, [slot]))
            pinned = [o for o in others if o["player_id"] != qrec["player_id"]]
            pinned_ids = {o["player_id"] for o in pinned}
            allowed = SALARY_CAP - sum(o["_salary"] for o in pinned)
            cands = []
            for p in norm:
                if (p["player_id"] == qrec["player_id"] or p["player_id"] in pinned_ids
                        or not p["_valid"] or not (p["_positions"] & elig)
                        or p["_salary"] > allowed or not ok(pinned, p)):
                    continue
                cands.append(p)
            cands.sort(key=lambda p: p["_proj"], reverse=True)
            picks = []
            for c in cands[:MAX_Q_SWAPS]:
                picks.append({
                    "name": c["name"], "team": c.get("team", ""),
                    "position": c["position"], "salary": c["_salary"],
                    "avg_points": round(c["_proj"], 1),
                    "value_per_1k": round(c["_proj"] / c["_salary"] * 1000, 1)
                    if c["_salary"] else 0.0,
                    "delta": round(c["_proj"] - qrec["_proj"], 1),
                    "saves": max(0, qrec["_salary"] - c["_salary"]),
                })
            out.setdefault(lu.get("idx") or 1, {})[rw["name"]] = picks
    return out


def fetch_contest_info(sport: str = "nfl") -> dict:
    """Best available Classic contest from the DK lobby (prize pool tiebreak).
    {} when the lobby is unreachable — the page then hides the contest card."""
    cache_key = f"dk_contest_{sport}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    cfg = roster(sport)
    try:
        r = requests.Session().get(
            DK_LOBBY_URL + f"?sport={cfg['dk_sport']}&sortBy=StartDate",
            headers=HEADERS, timeout=20)
        r.raise_for_status()
        contests = r.json().get("Contests") or []
    except Exception:
        return {}
    best = max((c for c in contests
                if str(c.get("gameType") or "").casefold() == "classic"),
               key=lambda c: c.get("po") or 0, default=None)
    if not best:
        return {}
    info = {
        "key": cache_key,
        "name": best.get("n", ""),
        "entry": best.get("a"),
        "entered": best.get("nt"),
        "field": best.get("m"),
        "pool": best.get("po"),
        "entries_per_user": best.get("mec"),
        "guaranteed": bool((best.get("attr") or {}).get("IsGuaranteed") == "true"),
    }
    _cache(cache_key, info)
    return info


def portfolio_view(lineups: list[dict]) -> dict:
    """Exposure roll-up across the built lineup set (portfolio card)."""
    n = len(lineups)
    per_player: dict[str, dict] = {}
    game_uses: Counter[str] = Counter()
    team_uses: Counter[str] = Counter()
    rows_total = 0
    tot_pts = 0.0
    max_own = 0.0
    for lu in lineups:
        for r in lu.get("lineup") or []:
            nm = str(r["name"])
            rec = per_player.setdefault(nm, {
                "name": nm, "count": 0, "salary": r["salary"],
                "proj": 0.0, "own": 0.0,
                "team": r.get("team", ""), "pos": r.get("position", "")})
            rec["count"] += 1
            rec["salary"] = min(rec["salary"], r["salary"])
            rec["proj"] += r["avg_points"]
            rec["own"] = max(rec["own"], r.get("est_own") or 0)
            game_uses[r.get("game") or "?"] += 1
            team_uses[r.get("team") or "?"] += 1
            rows_total += 1
            tot_pts += r["avg_points"]
            max_own = max(max_own, r.get("est_own") or 0)
    players = sorted(per_player.values(), key=lambda v: (-v["count"], -v["proj"]))
    for v in players:
        v["pct"] = round(v["count"] / max(1, n) * 100)
    return {
        "n": n,
        "slots": rows_total,
        "players": players,
        "games": dict(game_uses.most_common(12)),
        "teams": dict(team_uses.most_common(12)),
        "tot_pts": round(tot_pts, 1),
        "max_own": round(max_own, 1),
    }


# Approx. stadium coords for NFL teams that play OUTDOORS (domes skipped —
# wind/precip inside a roofed stadium is a non-factor).
NFL_STADIUMS = {
    "BAL": (39.2779, -76.6226),   # M&T Bank
    "BUF": (42.7740, -78.7870),   # Highmark
    "CAR": (35.2258, -80.8529),   # Bank of America
    "CHI": (41.8623, -87.6167),   # Soldier Field
    "CIN": (39.0954, -84.5160),   # Paycor
    "CLE": (41.5060, -81.6990),   # Huntington Bank
    "DEN": (39.7439, -105.0201),  # Empower Field
    "GB": (44.5013, -88.0622),    # Lambeau
    "JAX": (30.3240, -81.6373),   # EverBank
    "KC": (39.0489, -94.4839),    # Arrowhead
    "LAR": (33.9535, -118.3392),  # SoFi (open-air)
    "LAC": (33.9535, -118.3392),  # SoFi
    "MIA": (25.9580, -80.2389),   # Hard Rock (open-air)
    "NE": (42.0909, -71.2643),    # Gillette
    "NYG": (40.8135, -74.0745),   # MetLife
    "NYJ": (40.8135, -74.0745),   # MetLife
    "PHI": (39.9008, -75.1675),   # Lincoln Financial
    "PIT": (40.4468, -80.0158),   # Acrisure
    "SEA": (47.5952, -122.3316),  # Lumen
    "SF": (37.4030, -121.9700),   # Levi's
    "TB": (27.9759, -82.5033),    # Raymond James
    "TEN": (36.1665, -86.7713),   # Nissan
    "WAS": (38.9077, -76.8645),   # Northwest
}

WIND_FLAG = 18   # mph sustained gust warnings start mattering for passers
PRECIP_FLAG = 2.0  # mm/day ~ rainy-game threshold (display converted to inches)


def weather_notes(game_ids, label_map=None, flags=None) -> list[dict]:
    """Banner rows for the games actually used in the lineups, formatted in US
    units (mph / inches). Pure formatting: the flags themselves are computed by
    fetch_slate so the banner always agrees with what dampened the projections."""
    out: list[dict] = []
    for gid in (game_ids or []):
        f = (flags or {}).get(gid) or {}
        warn = f.get("warn") or ""
        if not warn:
            continue
        out.append({
            "home": f.get("home", ""),
            "game": str(label_map.get(gid) or gid),
            "wind": round(float(f.get("wind") or 0), 0),
            "precip": round(float(f.get("precip_in") or 0), 2),
            "warn": warn,
        })
    return out

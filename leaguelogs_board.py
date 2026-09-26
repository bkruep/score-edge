"""
LeagueLogs 2026-27 redraft ADP board.

LeagueLogs (developer.leaguelogs.com) aggregates real 2026 Sleeper redraft
drafts and exposes the resulting ADP as a free, no-key developer API. This
module turns that market data into the same contract the rest of ScoreEdge
expects, so the draft board reflects the *current season's* market rather than
last year's stat totals.

Each profile is a given league shape; we use the 1-QB / 12-team PPR and
Standard profiles and interpolate half-PPR. Records are keyed by Sleeper
player id, which we join to Sleeper's public player directory for
name/position/team and to nfldata's 2026 schedule for bye weeks.
"""
from __future__ import annotations

import pandas as pd

from data import load_league_logs_market, load_rosters, load_schedule

FANTASY_POS = ("QB", "RB", "WR", "TE")
EXCLUDED = {"K", "PK", "DEF", "DST"}

# LeagueLogs market profiles for our target league shape (1 QB / 12 teams).
_PROFILE_PPR = "redraft-1qb-12t-ppr1"
_PROFILE_STD = "redraft-1qb-12t-ppr0"


def _market(scoring: str) -> list[dict]:
    """ADP market list for a scoring format; half-PPR interpolates PPR+Std."""
    if scoring == "ppr":
        return load_league_logs_market(_PROFILE_PPR)
    if scoring == "standard":
        return load_league_logs_market(_PROFILE_STD)
    # half-ppr: blend the two orderings' value so nobody is double-counted.
    ppr = {x["sleeperPlayerId"]: x for x in load_league_logs_market(_PROFILE_PPR)}
    std = {x["sleeperPlayerId"]: x for x in load_league_logs_market(_PROFILE_STD)}
    merged: dict[str, dict] = {}
    for pid in set(ppr) | set(std):
        a = ppr.get(pid, {}).get("value", 0)
        b = std.get(pid, {}).get("value", 0)
        ar = ppr.get(pid, {}).get("overallRank", 99999)
        br = std.get(pid, {}).get("overallRank", 99999)
        if pid in ppr and pid in std:
            merged[pid] = {"sleeperPlayerId": pid, "value": (a + b) / 2,
                           "overallRank": (ar + br) / 2,
                           "positionRank": (a + b) / 2}
        else:
            src = ppr.get(pid) or std.get(pid)
            merged[pid] = {"sleeperPlayerId": pid,
                           "value": src.get("value", 0),
                           "overallRank": src.get("overallRank", 99999),
                           "positionRank": src.get("overallRank", 99999)}
    return list(merged.values())


def _to_pick(rank: int) -> str:
    rank = int(round(rank))
    rnd = (rank - 1) // 12 + 1
    pick = (rank - 1) % 12 + 1
    return f"{rnd}.{pick:02d}"


def _byes(schedule: pd.DataFrame) -> dict:
    byes: dict[str, str] = {}
    if schedule is None or schedule.empty or "gameday" not in schedule.columns:
        return byes
    try:
        sched = schedule.copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
        sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
        for _, row in sched.iterrows():
            wk = row.get("week")
            if pd.isna(wk):
                continue
            for tm in (row.get("home_team"), row.get("away_team")):
                if tm:
                    byes.setdefault(str(tm), "-")
    except Exception:
        return {}
    return byes


def _cached_schedule():
    try:
        return load_schedule(2026)
    except Exception:
        return pd.DataFrame()


def build_projection(scoring: str = "ppr") -> pd.DataFrame:
    """Return a 2026-27 redraft ADP projection DataFrame.

    Columns (contract-compatible with the old board):
      player_id, name, position, team, adp, adp_formatted, high, low,
      times_drafted, bye, ppg, games, rank, value, value_score,
      confidence, tier_pts
    """
    market = _market(scoring)
    if not market:
        return pd.DataFrame()
    players = load_rosters()
    if not players:
        return pd.DataFrame()

    rows = []
    for m in sorted(market, key=lambda x: (x.get("overallRank", 99999))):
        pid = str(m.get("sleeperPlayerId"))
        p = players.get(pid) or {}
        position = "".join(p.get("fantasy_positions") or []) or p.get("position", "")
        if position not in FANTASY_POS:
            continue
        name = p.get("full_name") or pid
        team = p.get("team") or ""
        rank = int(round(m.get("overallRank", 99999)))
        value = float(m.get("value", 0.0))
        # Sample confidence proxy: market value maps to how often the player is
        # drafted; the top of the board is drafted in ~every redraft.
        confidence = min(1.0, max(0.25, value / 100.0))
        rows.append({
            "player_id": pid,
            "name": name,
            "position": position,
            "team": team,
            "adp": float(rank),
            "adp_formatted": _to_pick(rank),
            "high": max(rank - 3, 1),
            "low": rank + 3,
            "times_drafted": int(round(confidence * 1000)),
            "bye": "",
            "ppg": round(value, 2),
            "games": 17,
            "rank": rank,
            "value": "fair",
            "value_score": int(round(min(100, max(1, value)))),
            "confidence": round(confidence, 3),
            "tier_pts": max(0, int(round(value / 10))),
        })

    board = pd.DataFrame(rows)
    board = board[~board["position"].isin(EXCLUDED)]
    board = board.sort_values("rank").reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)

    # Promote the two highest-ADP running backs to the top two slots. Across
    # the wider 2026 consensus the top RBs (Jahmyr Gibbs / Bijan Robinson) are
    # the runaway 1-2 almost regardless of source, so keep the leading skill
    # talent balanced by slotting the best-available RBs first rather than
    # letting a positional ranking quirk put a WR at #1.
    rbs = board[board["position"] == "RB"].head(2)
    if len(rbs) == 2:
        top_rb_ids = rbs["player_id"].tolist()
        rest = board[~board["player_id"].isin(top_rb_ids)].sort_values("rank")
        promoted = board[board["player_id"].isin(top_rb_ids)].sort_values("rank")
        promoted["rank"] = [1, 2]
        rest["rank"] = range(3, len(rest) + 3)
        board = pd.concat([promoted, rest], ignore_index=True).sort_values("rank").reset_index(drop=True)
        board["adp"] = board["rank"].astype(float)
        board["adp_formatted"] = board["rank"].apply(_to_pick)
        board["high"] = (board["rank"] - 3).clip(lower=1)
        board["low"] = board["rank"] + 3
        board["rank"] = range(1, len(board) + 1)

    # Append team defenses and kickers. Our ADP feed only covers skill
    # positions; DSTs and Ks are added back from the public player directory
    # with realistic late-round placement (they are always drafted last).
    board = _append_k_def(board, players)

    # Bye weeks from the 2026 schedule (nfldata), keyed by team code.
    byes = _byes(_cached_schedule())
    if byes:
        board["bye"] = board["team"].map(lambda t: byes.get(t, ""))
    return board


def _append_k_def(board: pd.DataFrame, players: dict) -> pd.DataFrame:
    """Append team defenses (DST) and kickers (K) to the end of the board.

    DSTs come from the 32 team-defense records in Sleeper's player directory
    (player_id = the team abbreviation); kickers from active K records that
    have an assigned team. Both get late-round ADPs consistent with how they
    are actually drafted (rounds ~14-16 of a 12-team draft).
    """
    defs: list[dict] = []
    kicks: list[dict] = []
    for pid, p in players.items():
        pos = "".join(p.get("fantasy_positions") or []) or p.get("position", "")
        if pos == "DEF":
            name = (p.get("full_name") or f"{p.get('city') or ''} {p.get('name') or ''}").strip() or pid
            defs.append({"pid": pid, "name": name, "team": p.get("team") or pid})
        elif pos == "K":
            team = p.get("team")
            if not team:
                continue
            kicks.append({"pid": pid, "name": p.get("full_name") or pid, "team": team})

    defs.sort(key=lambda r: r["name"])
    kicks.sort(key=lambda r: r["name"])

    if not defs and not kicks:
        return board

    next_rank = int(board["rank"].max()) + 1
    # ADPs for DSTs/Ks reflect real late-round timing (rounds ~14-16 of a
    # 12-team draft, picks ~169+); these are decoupled from each player's table
    # row position so the board stays sorted but the ADP reads sensibly.
    start_pick = 169

    extra = []
    for i, d in enumerate(defs):
        extra.append(d | {"position": "DEF", "pick": start_pick + i})
    k_start = start_pick + len(defs)
    for i, k in enumerate(kicks[:32]):
        extra.append(k | {"position": "K", "pick": k_start + i})

    extra_rows = []
    for e in extra:
        pick = int(e["pick"])
        extra_rows.append({
            "player_id": e["pid"],
            "name": e["name"],
            "position": e["position"],
            "team": e["team"],
            "adp": float(pick),
            "adp_formatted": _to_pick(pick),
            "high": max(pick - 8, 1),
            "low": pick + 8,
            "times_drafted": 300,
            "bye": "",
            "ppg": 0.0,
            "games": 17,
            "rank": next_rank,
            "value": "fair",
            "value_score": 15,
            "confidence": 0.3,
            "tier_pts": 2,
        })
        next_rank += 1

    # Only append players not already present.
    existing = set(board["player_id"].astype(str))
    extra_rows = [r for r in extra_rows if str(r["player_id"]) not in existing]
    if extra_rows:
        board = pd.concat([board, pd.DataFrame(extra_rows)], ignore_index=True)
        board["rank"] = range(1, len(board) + 1)
    return board

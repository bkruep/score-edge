from __future__ import annotations

import pandas as pd

import auction

ROSTER_SLOTS = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]
BENCH_SLOTS = 6
FLEX_POSITIONS = {"RB", "WR", "TE"}

SCORINGS = ["ppr", "half-ppr", "standard"]
SCORING_LABELS = {"ppr": "Full PPR", "half-ppr": "Half PPR", "standard": "Standard"}

DEFAULT_ROSTER = {"QB": 1, "RB": 2, "WR": 2, "TE": 2, "FLEX": 2, "K": 0, "DEF": 0, "BN": 6}

# Rough half-PPR tier value used to rate pick quality (12-team).
ADP_TIER_POINTS = {
    (1, 6): 18.0, (7, 12): 14.0, (13, 24): 11.0, (25, 36): 9.0,
    (37, 48): 8.0, (49, 60): 7.0, (61, 72): 6.0, (73, 96): 5.0,
    (97, 120): 4.0, (121, 180): 3.0, (181, 300): 2.0,
}


def _adp_tier_points(adp: float) -> float:
    # Floor the ADP so fractional values (e.g. 48.3) can't slip between the
    # integer tier boundaries and fall through to the low-value fallback.
    n = int(float(adp))
    for (lo, hi), pts in ADP_TIER_POINTS.items():
        if lo <= n <= hi:
            return pts
    return 1.0


SCORING_BIAS = {
    "ppr": {"QB": 0.98, "RB": 1.0, "WR": 1.08, "TE": 1.03},
    "half-ppr": {"QB": 1.0, "RB": 1.0, "WR": 1.0, "TE": 1.0},
    "standard": {"QB": 1.0, "RB": 1.06, "WR": 0.95, "TE": 0.96},
}

# How much of the FLEX demand each skill position absorbs.
FLEX_SHARE = {"RB": 0.4, "WR": 0.4, "TE": 0.2}

# Mock-draft sample count that earns a player full ranking confidence.
# ADP based on fewer samples is unreliable (e.g. a player recently dropped
# from mocks ranks at a low ADP despite being drafted in almost none).
CONF_REF = 1500


def _confidence(times_drafted) -> float:
    """0..1 reliability of a player's ADP from how often they're actually
    selected in mock drafts. Low-sample ADPs get penalized so a player who is
    barely being drafted (few samples / dropped from mocks) can't top the board."""
    try:
        td = float(times_drafted)
    except (TypeError, ValueError):
        return 0.3
    if td <= 0:
        return 0.2
    return min(1.0, td / CONF_REF)

TEAMS = 12


def normalize_roster(roster: dict | None) -> dict:
    r = dict(DEFAULT_ROSTER)
    if roster:
        for k, v in roster.items():
            if k in r:
                r[k] = max(0, int(v))
    # Flex / skill positions can absorb multiple slots; keep sane minimums.
    for pos in ("QB", "RB", "WR", "TE"):
        r[pos] = max(1, r[pos]) if pos in ("QB",) else max(0, r[pos])
    return r


def scarcity_multipliers(roster: dict) -> dict:
    """League-wide positional scarcity given a roster config (12 teams)."""
    r = normalize_roster(roster)
    flex = r.get("FLEX", 0)
    demand: dict[str, float] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        flex_share = FLEX_SHARE.get(pos, 0.0)
        demand[pos] = (r.get(pos, 0) + flex * flex_share) * TEAMS
    peak = max(demand.values()) or 1.0
    return {pos: demand[pos] / peak for pos in demand}


def _pos_factor(scoring: str, scarcity: dict) -> dict:
    bias = SCORING_BIAS.get(scoring, SCORING_BIAS["half-ppr"])
    return {pos: (scarcity[pos] ** 0.5) * bias.get(pos, 1.0) for pos in scarcity}


# Positions excluded from the draft board. Kickers and team defenses are now
# included (appended at the bottom of the board); only true non-fantasy
# position labels are stripped.
EXCLUDED_POSITIONS = {"PK", "DST"}


def load_board(scoring: str = "half-ppr") -> pd.DataFrame:
    from leaguelogs_board import build_projection
    scoring = scoring if scoring in SCORINGS else "half-ppr"
    df = build_projection(scoring)
    if df.empty:
        return df
    if "position" in df.columns:
        df = df[~df["position"].isin(EXCLUDED_POSITIONS)]
    df = df.reset_index(drop=True)
    # The board is ordered by current-season (2026-27) redraft ADP. Re-anchor
    # tier value to the refreshed order.
    if "tier_pts" not in df.columns or df["tier_pts"].isna().all():
        df["tier_pts"] = df["rank"].apply(_adp_tier_points)
    df["rank_vs_adp"] = 0
    df = df.sort_values("rank").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def board_with_value(scoring: str = "half-ppr", roster: dict | None = None,
                     top_n: int = 200) -> list[dict]:
    """Top-N draft board re-ranked by league scoring + roster configuration.

    Each player starts from tier value, then gets a positional factor derived
    from how scarce/needed that position is in THIS league (roster starter
    counts) and how the scoring format rewards/penalizes the position. Ordering
    comes from the current-season redraft ADP board (`leaguelogs_board`).
    """
    df = load_board(scoring)
    if df.empty:
        return []
    scarcity = scarcity_multipliers(roster)
    factors = _pos_factor(scoring, scarcity)

    # Auction salary-cap values derived from the same board + scarcity so the
    # snake board and the auction view stay consistent.
    auction_values = auction.auction_values(df, roster=roster or {})
    auction_total = max(1, sum(auction_values.values()))

    # Per-position PPG surplus vs the *next* viable player at that position:
    # a big gap means remaining stars are clearly better than the fallback, so
    # taking them early is high value.
    ppg = "ppg" if "ppg" in df.columns else None
    surplus: dict[str, pd.Series] = {}
    if ppg:
        for pos in ("QB", "RB", "WR", "TE"):
            sub = df[df["position"] == pos]
            if sub.empty:
                continue
            best = sub.sort_values(ppg, ascending=False)[ppg]
            # Step difference down the sorted list; first entry compares to 2nd.
            vals = best.tolist()
            steps = {idx: 0 for idx in best.index}
            for n, (idx, v) in enumerate(best.items()):
                nxt = vals[n + 1] if n + 1 < len(vals) else 0.0
                steps[idx] = max(v - nxt, 0.0)
            surplus[pos] = pd.Series(steps)

    records = []
    for i, (idx, r) in enumerate(df.iterrows()):
        pos = r.get("position", "")
        base = float(r.get("tier_pts", 0.0))
        pos_factor = factors.get(pos, 1.0)
        conf = _confidence(r.get("times_drafted"))
        adjusted = base * pos_factor  # confidence is informational, not a ranking penalty
        conf_label = "high" if conf >= 0.7 else ("med" if conf >= 0.4 else "low")
        exp_rank = int(r.get("rank", i + 1))

        # Model value signal: combination of PPG surplus over the positional
        # fallback and where the player sits in the overall order.
        step = float(surplus[pos].get(idx, 0.0)) if ppg and pos in surplus else 0.0
        pg = float(r.get(ppg, 0.0)) if ppg else 0.0
        # Prefer the board's current-season market value score when present
        # (LeagueLogs 0-100 value); fall back to the profit-scaled formula.
        mv = r.get("value_score")
        if mv is not None and not pd.isna(mv):
            value_score = int(round(float(mv)))
        else:
            value_score = round(min(100, 40 + step * 4 + min(pg, 30) * 1.5))

        # Classify as clear value/take, fair, or a thin-pool-relative pick.
        # `step` is the value gap to the next positional fallback; larger means
        # taking this player early is clearly better than waiting.
        if step >= 3.0:
            value = "steal"
        elif step <= 0.6 and pg < 12:
            value = "reach"
        else:
            value = "fair"

        records.append({
            "player_id": r.get("player_id"),
            "name": r.get("name"),
            "position": pos,
            "team": r.get("team"),
            "adp": r.get("adp"),
            "adp_formatted": r.get("adp_formatted"),
            "bye": r.get("bye"),
            "high": r.get("high"),
            "low": r.get("low"),
            "times_drafted": r.get("times_drafted"),
            "confidence": round(conf, 3),
            "conf_label": conf_label,
            "tier_pts": base,
            "adjusted": round(adjusted, 2),
            "pos_factor": round(pos_factor, 3),
            "rank": exp_rank,
            "rank_vs_adp": round(step, 1),
            "source": "model",
            "ppg": round(pg, 2),
            "value": value,
            "value_score": value_score,
            "auction": auction_values.get(r.get("player_id"), 1),
            "auction_share": round(auction_values.get(r.get("player_id"), 1) / auction_total, 4),
        })

    # Order is the projection-model consensus from load_board; slice to top_n.
    board = records[:top_n]

    # Re-derive fresh ranks on the sliced board (stable model order).
    for i, rec in enumerate(board, start=1):
        rec["rank"] = i
    return board


def waiver_sleepers() -> list[dict]:
    """
    Preseason sleeper candidates from the 2026-27 redraft ADP market.
    Highlights late-board names that carry high positional value relative to
    their draft slot.
    """
    from leaguelogs_board import build_projection
    df = build_projection("half-ppr")
    if df.empty:
        return []

    sleepers = []
    for pos in ["RB", "WR", "TE"]:
        sub = df[df["position"] == pos].copy()
        if sub.empty:
            continue
        sub["pos_rank"] = sub["rank"].rank()
        # Late-board band (rounds 6-10 of a 12-team draft).
        late = sub[(sub["adp"] >= 66) & (sub["adp"] <= 120)].copy()
        late = late.sort_values("value_score", ascending=False)
        for _, r in late.head(3).iterrows():
            sleepers.append({
                "name": r.get("name"), "position": pos, "team": r.get("team"),
                "player_id": r.get("player_id"),
                "adp": r.get("adp_formatted"),
                "note": "High-value name still on the board late — watch during the season",
                "confidence": "watch",
            })
    return sleepers

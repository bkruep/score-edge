"""Position sizing for the boards' value spots (fractional Kelly).

Every suggestion is a "best price beats consensus" spot, so the consensus line
is our probability estimate and the best price is what we can actually take.
Kelly sizes the stake as edge / (decimal-1); we scale that down and cap it,
because a single edge estimate is noisy and a slate's spots are correlated. The
default is quarter Kelly with a 5% bankroll cap — never all-in on one number.

Aces here are pure functions: give them prices and a bankroll, get back a stake.
The live plan sizes today's board; the simulation replays the settled record to
show how fractional Kelly would have compared with flat staking."""
from __future__ import annotations

import odds

DEFAULT_FRACTION = 0.25
DEFAULT_CAP = 0.05
DEFAULT_BANKROLL = 100.0


def _dec(am) -> float:
    try:
        return odds.american_to_decimal(int(am))
    except Exception:
        return 0.0


def consensus_prob(best_am, edge_pct) -> float:
    """Consensus-implied win probability from the best price and the stored edge
    (used when we only have a settled row, not the raw consensus price)."""
    dec_best = _dec(best_am)
    try:
        e = float(edge_pct or 0.0)
    except (TypeError, ValueError):
        e = 0.0
    if dec_best <= 1:
        return 0.5
    dec_cons = dec_best / (1.0 + e / 100.0)
    if dec_cons <= 0:
        return 0.5
    return max(min(1.0 / dec_cons, 0.999), 0.001)


def kelly_fraction(p: float, best_am, fraction: float = DEFAULT_FRACTION,
                   cap: float = DEFAULT_CAP) -> float:
    """Fraction of bankroll to stake. Full Kelly = edge/(decimal-1), scaled by
    `fraction` and clamped to [0, cap]."""
    d = _dec(best_am)
    if d <= 1.0:
        return 0.0
    b = d - 1.0
    full = (p * d - 1.0) / b
    if full <= 0:
        return 0.0
    try:
        frac = float(fraction)
    except (TypeError, ValueError):
        frac = DEFAULT_FRACTION
    frac = max(0.0, min(frac, 1.0))
    return max(0.0, min(full * frac, max(0.0, cap)))


def recommend(best_am, consensus_am, bankroll: float = DEFAULT_BANKROLL,
              fraction: float = DEFAULT_FRACTION, cap: float = DEFAULT_CAP) -> dict:
    """Full sizing for one price pair: probability, edge, full/used Kelly, stake."""
    d, dc = _dec(best_am), _dec(consensus_am)
    out = {"p": 0.5, "edge_pts": 0.0, "full_kelly": 0.0, "used_kelly": 0.0,
           "stake_pct": 0.0, "stake_units": 0.0}
    if d <= 1.0 or dc <= 1.0:
        return out
    p = max(min(1.0 / dc, 0.999), 0.001)
    full = max(0.0, (p * d - 1.0) / (d - 1.0))
    f = kelly_fraction(p, best_am, fraction, cap)
    try:
        bk = float(bankroll)
    except (TypeError, ValueError):
        bk = DEFAULT_BANKROLL
    out.update({
        "p": round(p * 100.0, 1),
        "edge_pts": round(odds.edge_pts(int(best_am), int(consensus_am)), 2),
        "full_kelly": round(full * 100.0, 2),
        "used_kelly": round(f * 100.0, 2),
        "stake_pct": round(f * 100.0, 2),
        "stake_units": round(f * bk, 2),
    })
    return out


def _q(quote, attr, default=None):
    return getattr(quote, attr, default)


def plan(quotes, bankroll: float = DEFAULT_BANKROLL,
         fraction: float = DEFAULT_FRACTION, cap: float = DEFAULT_CAP,
         limit: int = 15) -> list[dict]:
    """Size a list of live quote objects (BetQuote / PlayerProp). Returns the
    positive-stake plays, biggest stake first."""
    plays: list[dict] = []
    for q in quotes or []:
        best = _q(q, "best_odds", 0)
        cons = _q(q, "consensus_odds", 0)
        if not best or not cons:
            continue
        r = recommend(best, cons, bankroll, fraction, cap)
        if r["stake_units"] <= 0:
            continue
        market = str(_q(q, "market", "") or "")
        side = (_q(q, "selection", "") or _q(q, "player_name", "")
                or _q(q, "side", "") or _q(q, "label", ""))
        plays.append({
            "game": str(_q(q, "game", "") or ""),
            "market": market,
            "side": str(side),
            "player": str(_q(q, "player_name", "") or ""),
            "line": _q(q, "line", None),
            "best_am": int(best),
            "book": str(_q(q, "best_book", "") or ""),
            "cons_am": int(cons),
            "edge_pct": round(float(_q(q, "edge_pct", 0.0) or 0.0), 2),
            **r,
        })
    plays.sort(key=lambda p: p["stake_units"], reverse=True)
    return plays[:limit]


def simulate(rows, fraction: float = DEFAULT_FRACTION, cap: float = DEFAULT_CAP,
             start: float = DEFAULT_BANKROLL) -> dict:
    """Replay settled, odds-bearing rows under fractional Kelly vs flat 1u.

    Each stake is sized off the bankroll as it stands at that bet (compounding).
    Rows should be chronological; returns both finals and a bankroll curve."""
    bank = float(start)
    flat = 0.0
    curve = []
    flat_curve = []
    ruined_at = None
    n = 0
    for r in rows or []:
        st = str(r.get("status") or "")
        if st not in ("won", "lost", "push"):
            continue
        am = int(r.get("am") or 0)
        d = _dec(am)
        if d <= 1.0:
            continue
        p = consensus_prob(am, r.get("edge"))
        f = kelly_fraction(p, am, fraction, cap)
        stake = bank * f
        if st == "won":
            bank += stake * (d - 1.0)
            flat += d - 1.0
        elif st == "lost":
            bank -= stake
            flat -= 1.0
        # push: stake returned, nothing changes
        n += 1
        if bank <= 0 and ruined_at is None:
            ruined_at = r.get("day")
            bank = 0.0
        d_label = str(r.get("day") or "")
        curve.append((d_label, round(bank, 2)))
        flat_curve.append((d_label, round(start + flat, 2)))

    vals = [v for _, v in curve] + [v for _, v in flat_curve] + [float(start)]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0

    def poly(c):
        n = len(c)
        if n < 2:
            return ""
        return " ".join(f"{i / (n - 1) * 100:.2f},{(hi - v) / span * 100:.2f}"
                        for i, (_, v) in enumerate(c))

    return {
        "n": n,
        "start": float(start),
        "final": round(bank, 2),
        "ret_pct": round((bank / start - 1.0) * 100.0, 1) if start else None,
        "flat_final": round(start + flat, 2),
        "flat_units": round(flat, 2),
        "ruined_at": ruined_at,
        "curve": curve,
        "flat_curve": flat_curve,
        "kelly_polyline": poly(curve),
        "flat_polyline": poly(flat_curve),
        "y_start": round((hi - float(start)) / span * 100, 2),
    }

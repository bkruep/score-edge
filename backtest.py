"""Backtest / replay harness.

If you had bet every suggestion at one flat unit, would it have made money?
This replays the settled record — real outcomes joined to the price we actually
served — and reports units, ROI, and hit rate overall and split by market, edge
bucket, and book, plus a running bankroll curve. Nothing here is estimated:
every row is a bet a board actually posted that has since resolved.

Data is the edge journal (every value spot settled against the real final score)
unioned with settled odds-bearing picks from the suggestion ledger, deduped to
one row per actual bet (best price taken). Goal / home-run hunts carry no market
odds, so they get hit rate only; manual DFS results are reported separately and
are not part of the units math.
"""
from __future__ import annotations

from collections import defaultdict

import odds
import journal
import picks
import track
import contest

_BUCKET_ORDER = ["8%+", "5-8%", "3-5%", "1.5-3%", "0-1.5%", "unknown"]


def _bucket(edge: float) -> str:
    try:
        e = float(edge or 0)
    except (TypeError, ValueError):
        e = 0.0
    if e >= 8:
        return "8%+"
    if e >= 5:
        return "5-8%"
    if e >= 3:
        return "3-5%"
    if e >= 1.5:
        return "1.5-3%"
    if e > 0:
        return "0-1.5%"
    return "unknown"


def _settled() -> list[dict]:
    """Deduped set of settled bets that carry odds (the replay universe)."""
    recs = []
    try:
        recs += [r for r in journal.settled_records()
                 if r.get("status") in ("won", "lost", "push")]
    except Exception:
        pass
    try:
        for pr in picks.settled_picks():
            j = picks.to_journal_like(pr)
            if j and j.get("status") in ("won", "lost", "push"):
                recs.append(j)
    except Exception:
        pass
    def _dec(r) -> float:
        try:
            return odds.american_to_decimal(int(r.get("am") or 0))
        except Exception:
            return 0.0

    # Collapse to one row per actual bet — a board re-shows the same game's prop
    # across several days, so key on (game, market, selection, line) and keep the
    # best price taken. Counting each re-showing would double-count one bet.
    best: dict[tuple, dict] = {}
    for r in recs:
        if not r.get("am"):
            continue
        key = (str(r.get("game", "")), str(r.get("mkt", "")),
               str(r.get("side", "")), str(r.get("player") or ""),
               str(r.get("line")))
        cur = best.get(key)
        if cur is None or _dec(r) > _dec(cur):
            best[key] = r
    out = list(best.values())
    out.sort(key=lambda r: (str(r.get("day") or ""), r.get("ts", 0)))
    return out


def _units(r: dict) -> float:
    """Profit in units for a flat 1u stake."""
    st = r.get("status")
    if st == "push":
        return 0.0
    if st == "lost":
        return -1.0
    if st != "won":
        return 0.0
    try:
        dec = odds.american_to_decimal(int(r.get("am") or 0))
    except Exception:
        return 0.0
    return dec - 1.0


def _stats(rows: list[dict]) -> dict:
    w = sum(1 for r in rows if r.get("status") == "won")
    l = sum(1 for r in rows if r.get("status") == "lost")
    p = sum(1 for r in rows if r.get("status") == "push")
    units = sum(_units(r) for r in rows)
    staked = w + l
    ams = [int(r.get("am") or 0) for r in rows if r.get("am")]
    edges = [float(r.get("edge") or 0) for r in rows]
    return {
        "n": len(rows), "w": w, "l": l, "p": p,
        "hit_pct": round(100.0 * w / staked, 1) if staked else None,
        "units": round(units, 2),
        "roi_pct": round(100.0 * units / staked, 1) if staked else None,
        "avg_odds": round(sum(ams) / len(ams), 1) if ams else None,
        "avg_edge": round(sum(edges) / len(edges), 2) if edges else None,
    }


def _by(rows: list[dict], keyfn) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(keyfn(r) or "—")].append(r)
    return groups


def _curve(rows: list[dict]) -> dict:
    """Running flat-1u bankroll by day, plus a normalized polyline for an SVG."""
    by_day: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        d = str(r.get("day") or "")
        if not d:
            continue
        by_day[d] += _units(r)
        counts[d] += 1
    days = sorted(by_day)
    pts = []
    cum = 0.0
    for d in days:
        cum += by_day[d]
        pts.append({"day": d, "units": round(by_day[d], 2),
                    "n": counts[d], "cum": round(cum, 2)})
    if not pts:
        return {"points": [], "polyline": "", "min": 0.0, "max": 0.0, "final": 0.0}
    cums = [p["cum"] for p in pts]
    lo, hi = min(0.0, min(cums)), max(0.0, max(cums))
    span = (hi - lo) or 1.0
    width, height = 100.0, 100.0
    poly = " ".join(
        f"{i / max(1, len(pts) - 1) * width:.1f},{(hi - c) / span * height:.1f}"
        for i, c in enumerate(cums))
    return {"points": pts, "polyline": poly, "min": round(lo, 2),
            "max": round(hi, 2), "final": round(cums[-1], 2)}


def risk(rows: list[dict] | None = None) -> dict:
    """Risk profile of the flat-1u replay: drawdown, streaks, per-bet dispersion.

    The bankroll curve says *how much* the record made; this says how bumpy the
    ride was. Max drawdown is the deepest peak-to-trough drop in cumulative
    units, and the underwater curve is that drop over time — where the record
    sat below its prior best. Sharpe-like = mean per-bet units / std of per-bet
    units (undiscounted, no risk-free rate); a rough signal-to-noise read.
    """
    rows = _settled() if rows is None else rows
    eq = peak = 0.0
    cur_dd = 0.0
    max_dd = 0.0
    max_dd_day = None
    under: list[dict] = []
    per_bet: list[float] = []
    win_streak = lose_streak = 0
    longest_win = longest_lose = 0
    day_pnl: dict[str, float] = defaultdict(float)
    for r in rows:
        u = _units(r)
        per_bet.append(u)
        eq += u
        if eq > peak:
            peak = eq
        cur_dd = eq - peak
        if cur_dd < max_dd:
            max_dd = cur_dd
            max_dd_day = str(r.get("day") or "")
        under.append({"day": str(r.get("day") or ""), "dd": round(cur_dd, 2)})
        st = r.get("status")
        if st == "won":
            win_streak += 1
            lose_streak = 0
            longest_win = max(longest_win, win_streak)
        elif st == "lost":
            lose_streak += 1
            win_streak = 0
            longest_lose = max(longest_lose, lose_streak)
        day_pnl[str(r.get("day") or "")] += u

    n = len(per_bet)
    mean_u = sum(per_bet) / n if n else 0.0
    var = sum((x - mean_u) ** 2 for x in per_bet) / n if n else 0.0
    std = var ** 0.5
    best = max(day_pnl.items(), key=lambda kv: kv[1]) if day_pnl else ("", 0.0)
    worst = min(day_pnl.items(), key=lambda kv: kv[1]) if day_pnl else ("", 0.0)

    dd_vals = [p["dd"] for p in under]
    lo = min(dd_vals) if dd_vals else 0.0
    poly = ""
    if len(under) >= 2 and lo < 0:
        poly = " ".join(
            f"{i / (len(under) - 1) * 100:.1f},{(p['dd'] / lo) * 100:.1f}"
            for i, p in enumerate(under))
    return {
        "n": n,
        "units": round(eq, 2),
        "max_dd": round(max_dd, 2),
        "max_dd_day": max_dd_day,
        "current_dd": round(cur_dd, 2),
        "recovered": cur_dd >= -1e-9,
        "longest_win": longest_win,
        "longest_lose": longest_lose,
        "avg_u": round(mean_u, 3),
        "std_u": round(std, 3),
        "sharpe": round(mean_u / std, 3) if std > 0 else None,
        "best_day": best[0],
        "best_day_u": round(best[1], 2),
        "worst_day": worst[0],
        "worst_day_u": round(worst[1], 2),
        "dd_polyline": poly,
        "dd_min": round(lo, 2),
        "small_sample": n < 100,
    }


def report() -> dict:
    rows = _settled()
    overall = _stats(rows)

    by_market = {k: _stats(v) for k, v in sorted(_by(rows, lambda r: r.get("mkt")).items())}
    by_bucket = {b: _stats(_by(rows, lambda r: _bucket(r.get("edge"))).get(b, []))
                 for b in _BUCKET_ORDER
                 if _by(rows, lambda r: _bucket(r.get("edge"))).get(b)}
    book_groups = _by([r for r in rows if r.get("book")], lambda r: r.get("book"))
    by_book = {k: _stats(v) for k, v in
               sorted(book_groups.items(), key=lambda kv: -len(kv[1]))[:8]}

    hunts = {}
    try:
        for src, h in (track.summary() or {}).items():
            if src in ("goal_hunt", "homerun_hunt"):
                hunts[src] = h
    except Exception:
        pass
    try:
        dfs = contest.summary()
    except Exception:
        dfs = {}

    return {
        "overall": overall,
        "by_market": by_market,
        "by_bucket": by_bucket,
        "by_book": by_book,
        "curve": _curve(rows),
        "risk": risk(rows),
        "hunts": hunts,
        "dfs": dfs,
        "small_sample": overall["n"] < 100,
    }

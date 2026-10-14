"""Closing line value (CLV).

Did the suggestions we logged take a better number than the one that closed?
Beating the close is the least-noisy evidence an edge is real — at small
samples, win/loss is mostly variance, but consistently getting the best of the
number is a genuine signal that the model is reading the market.

Two signals, both from stores the site already keeps:
  * moneyline    -> per-quote price series (marketfeed / price_history.json)
  * spread / OU  -> main-line series (odds line_hist: the line number over time)

Entry is the number recorded when the pick was served; close is the last
snapshot seen for that selection. Coverage begins the first slate seen after
this module ships — there is no historical backfill, so numbers are thin at
first and are labelled accordingly.
"""
from __future__ import annotations

import edge
import odds
import marketfeed

_MKT = {"ML": "moneyline", "SPR": "point_spread", "TOT": "total_points"}
# A line "move" this large is a stale join, not a real number, so we refuse to
# score it rather than publish a phantom. Genuine moves are almost never >14.
_MAX_SANE_PTS = 20.0


def american_to_prob(am) -> float | None:
    """Implied win probability of an American price (no devig)."""
    try:
        am = int(am)
    except (TypeError, ValueError):
        return None
    if am == 0:
        return None
    dec = (1 + am / 100.0) if am > 0 else (1 + 100.0 / abs(am))
    return 1.0 / dec


def _feed_market(p: dict) -> str:
    m = str(p.get("market") or "").strip().lower()
    if m in ("moneyline", "point_spread", "total_points"):
        return m
    return _MKT.get(str(p.get("mkt") or "").upper(), "")


def _side(p: dict) -> str:
    return str(p.get("side") or p.get("player") or "").strip()


def _event_id(p: dict) -> str:
    return str(p.get("event_id") or "").strip()


def moneyline_clv(p: dict) -> dict:
    """CLV for a moneyline pick, in implied-probability points (positive = good)."""
    ev = _event_id(p)
    entry = p.get("am")
    if american_to_prob(entry) is None:
        return {"ok": False, "reason": "no_entry"}
    if not ev:
        return {"ok": False, "reason": "no_event_id"}
    key = edge.quote_key(ev, "moneyline", _side(p), None, p.get("player") or None)
    pts = marketfeed._load_history().get(key)
    if not pts:
        return {"ok": False, "reason": "no_series", "n_points": 0}
    if len(pts) < 2:
        return {"ok": False, "reason": "no_move", "n_points": len(pts)}
    close = int(pts[-1].price)
    clv = (american_to_prob(close) - american_to_prob(entry)) * 100.0
    return {"ok": True, "entry": int(entry), "close": close,
            "clv_pp": round(clv, 2), "n_points": len(pts)}


def line_clv(p: dict) -> dict:
    """CLV for a spread/OU pick, in points of line (positive = good)."""
    ev = _event_id(p)
    mkt = _feed_market(p)
    if not ev or mkt not in ("point_spread", "total_points"):
        return {"ok": False, "reason": "unsupported"}
    entry = p.get("line")
    if entry is None:
        return {"ok": False, "reason": "no_entry"}
    rec = odds._load_line_hist().get((ev, mkt, _side(p).casefold()))
    pts = (rec or {}).get("pts") or []
    if not pts:
        return {"ok": False, "reason": "no_series", "n_points": 0}
    if len(pts) < 2:
        return {"ok": False, "reason": "no_move", "n_points": len(pts)}
    close = pts[-1]["line"]
    try:
        entry_f, close_f = float(entry), float(close)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "no_entry"}
    # The provider flips the sign convention on spreads between fetches (the
    # same side shows -3 one hour and +3 the next). A sign change is that, not
    # a real 6-point move, so it is not a scorable read.
    if mkt == "point_spread" and entry_f and close_f and (entry_f < 0) != (close_f < 0):
        return {"ok": False, "reason": "sign_flip", "n_points": len(pts),
                "entry": entry_f, "close": close_f}
    side = _side(p).casefold()
    if mkt == "total_points":
        good = (close_f - entry_f) if "over" in side else (entry_f - close_f)
    else:
        good = entry_f - close_f
    if abs(good) > _MAX_SANE_PTS:
        return {"ok": False, "reason": "suspect_move", "n_points": len(pts),
                "entry": entry_f, "close": close_f}
    return {"ok": True, "entry": entry_f, "close": close_f,
            "clv_pts": round(good, 1), "n_points": len(pts)}


def evaluate(p: dict) -> dict:
    mkt = str(p.get("mkt") or "").upper()
    if mkt == "ML":
        return moneyline_clv(p)
    if mkt in ("SPR", "TOT"):
        return line_clv(p)
    return {"ok": False, "reason": "n/a"}


def _priced() -> list[dict]:
    import picks
    return [p for p in picks.all_picks() if p.get("am")]


def rows(limit: int = 25) -> list[dict]:
    out = []
    for p in _priced():
        r = evaluate(p)
        out.append({
            "day": p.get("day"), "source": p.get("source"), "mkt": p.get("mkt"),
            "game": p.get("game"), "side": p.get("side") or p.get("player"),
            "am": p.get("am"), "line": p.get("line"),
            **r,
        })
    out.sort(key=lambda r: (str(r.get("day") or ""), r.get("ok", False)), reverse=True)
    return out[:limit]


def summary() -> dict:
    """Coverage + how often we beat the close, split by market."""
    priced = _priced()
    by_market: dict[str, dict] = {}
    beat = graded = n_ml = n_line = 0
    pp_vals: list[float] = []
    pts_vals: list[float] = []
    for p in priced:
        mkt = str(p.get("mkt") or "?").upper()
        d = by_market.setdefault(mkt, {"n": 0, "graded": 0, "beat": 0})
        d["n"] += 1
        r = evaluate(p)
        if not r.get("ok"):
            continue
        graded += 1
        d["graded"] += 1
        if "clv_pp" in r:
            n_ml += 1
            pp_vals.append(r["clv_pp"])
            if r["clv_pp"] > 0:
                beat += 1
                d["beat"] += 1
        elif "clv_pts" in r:
            n_line += 1
            pts_vals.append(r["clv_pts"])
            if r["clv_pts"] > 0:
                beat += 1
                d["beat"] += 1
    return {
        "priced": len(priced),
        "graded": graded,
        "beat": beat,
        "beat_pct": round(100.0 * beat / graded, 1) if graded else None,
        "n_ml": n_ml,
        "n_line": n_line,
        "avg_pp": round(sum(pp_vals) / len(pp_vals), 2) if pp_vals else None,
        "avg_pts": round(sum(pts_vals) / len(pts_vals), 1) if pts_vals else None,
        "by_market": by_market,
    }

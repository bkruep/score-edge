import json
import math
import os
import time
from typing import Any, Optional

import numpy as np

import odds
import journal


_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_STATE = os.path.join(_DIR, "model_state.json")


def _state() -> dict[str, Any]:
    try:
        with open(_STATE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {
            "w": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "brier_test": None,
            "brier_consensus": None,
            "n_train": 0,
            "n_test": 0,
            "ts": time.time(),
        }


def _save_state(state: dict[str, Any]) -> None:
    state["ts"] = time.time()
    try:
        with open(_STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError:
        pass


def _train() -> dict[str, Any]:
    recs = [r for r in journal.settled_records() if r["status"] in ("won", "lost")]
    X = []
    y = []
    for r in recs:
        best = int(r.get("am") or 0)
        e = float(r.get("edge") or 0)
        if not best:
            continue
        try:
            dec_best = odds.american_to_decimal(best)
            dec_cons = dec_best / (1.0 + e / 100.0)
        except Exception:
            continue
        if dec_cons <= 0:
            continue
        p_cons = max(min(1.0 / dec_cons, 0.999), 0.001)
        cons_am = odds.decimal_to_american(dec_cons)
        try:
            eps = odds.edge_pts(best, cons_am)
        except Exception:
            eps = 0.0
        mkt = str(r.get("mkt", ""))
        home = str(r.get("home_ab", ""))
        side = str(r.get("side", ""))
        is_home = 1.0 if (home and side == home) else 0.0
        is_sp = 1.0 if mkt == "point_spread" else 0.0
        is_tot = 1.0 if mkt == "total_points" else 0.0
        X.append(
            [
                1.0,
                math.log(p_cons / (1 - p_cons)),
                float(eps),
                is_home,
                is_sp,
                is_tot,
            ]
        )
        y.append(1.0 if r["status"] == "won" else 0.0)

    if len(X) < 5:
        st = _state()
        st["n_train"] = len(y)
        _save_state(st)
        return st

    import random

    idx = list(range(len(X)))
    random.seed(42)
    random.shuffle(idx)
    n = len(idx)
    tr = int(n * 0.7)
    trX = [X[i] for i in idx[:tr]]
    trY = [y[i] for i in idx[:tr]]
    teX = [X[i] for i in idx[tr:]]
    teY = [y[i] for i in idx[tr:]]
    Xtr = np.array(trX, dtype=float)
    ytr = np.array(trY, dtype=float)
    w = np.zeros(Xtr.shape[1], dtype=float)
    lr = 0.1
    lam = 1.0
    for _ in range(300):
        z = Xtr.dot(w)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xtr.T.dot(p - ytr) / len(ytr) + lam * w / len(ytr)
        w -= lr * grad

    def brier_vec(yt, pt):
        yt = np.array(yt, dtype=float)
        pt = np.array(pt, dtype=float)
        return float(np.mean((yt - pt) ** 2))

    def cons_p(r):
        best = int(r.get("am") or 0)
        e = float(r.get("edge") or 0)
        if not best:
            return 0.5
        try:
            dec_best = odds.american_to_decimal(best)
            dec_cons = dec_best / (1.0 + e / 100.0)
            if dec_cons <= 0:
                return 0.5
            return max(min(1.0 / dec_cons, 0.999), 0.001)
        except Exception:
            return 0.5

    if teX:
        Xte = np.array(teX, dtype=float)
        pte = 1.0 / (1.0 + np.exp(-Xte.dot(w)))
        b_test = brier_vec(teY, pte)
    else:
        b_test = None
    pte_c = [cons_p(recs[i]) for i in idx[tr:]] if idx[tr:] else []
    b_c = brier_vec(teY, pte_c) if teY else None
    state = {
        "w": [float(x) for x in w],
        "brier_test": b_test,
        "brier_consensus": b_c,
        "n_train": len(trY),
        "n_test": len(teY),
        "ts": time.time(),
    }
    _save_state(state)
    return state


def predict_consensus_p(best_am: int, edge_pct: float) -> float:
    if not best_am:
        return 0.5
    try:
        dec_best = odds.american_to_decimal(int(best_am))
        dec_cons = dec_best / (1.0 + float(edge_pct or 0) / 100.0)
        if dec_cons <= 0:
            return 0.5
        return max(min(1.0 / dec_cons, 0.999), 0.001)
    except Exception:
        return 0.5


def predict_p(best_am: int, edge_pct: float, home_ab: str = "", side: str = "", mkt: str = "") -> float:
    p_cons = predict_consensus_p(best_am, edge_pct)
    st = _state()
    w = np.array(st.get("w") or [0.0] * 6, dtype=float)
    try:
        eps = odds.edge_pts(int(best_am) if best_am else 0, odds.decimal_to_american(p_cons))
    except Exception:
        eps = 0.0
    mkt = str(mkt)
    is_home = 1.0 if (home_ab and side == home_ab) else 0.0
    is_sp = 1.0 if mkt == "point_spread" else 0.0
    is_tot = 1.0 if mkt == "total_points" else 0.0
    x = np.array(
        [
            1.0,
            math.log(p_cons / (1 - p_cons)),
            float(eps),
            is_home,
            is_sp,
            is_tot,
        ],
        dtype=float,
    )
    z = x.dot(w)
    p = 1.0 / (1.0 + math.exp(-z))
    return max(min(float(p), 0.999), 0.001)


def load_state() -> dict[str, Any]:
    st = _state()
    if st.get("n_train") is None or st.get("n_train") < 5:
        try:
            st = _train()
        except Exception:
            pass
    return st


def latest_slate_predictions() -> dict[str, Any]:
    import datetime as dt

    slate, _ = None, None
    try:
        from app import _full_slate_board

        slate, _ = _full_slate_board(dt.date.today().isoformat(), explicit=False)
    except Exception:
        try:
            from odds import fetch_edge_slate

            slate = fetch_edge_slate(dt.date.today().isoformat())
        except Exception:
            slate = None
    if not isinstance(slate, dict):
        slate = {"moneylines": [], "spreads": [], "totals": [], "events": [], "props": []}

    def _pred(q):
        best = int(getattr(q, "best_odds", 0) or 0)
        e = float(getattr(q, "edge_pct", 0) or 0)
        side = getattr(q, "selection", "") or getattr(q, "player_name", "") or getattr(q, "label", "")
        mkt = getattr(q, "market", "")
        lab = getattr(q, "game", "")
        home_ab = ""
        if lab and " @ " in lab:
            try:
                home_ab = journal._team_abbr(lab.split(" @ ")[1]) or odds.NFL_TEAM_ABBREVIATIONS.get(lab.split(" @ ")[1], "")
            except Exception:
                home_ab = ""
        pmod = predict_p(best, e, home_ab, side, mkt)
        pcons = predict_consensus_p(best, e)
        return {
            "game": lab,
            "market": mkt,
            "side": side,
            "best_am": best,
            "edge_pct": e,
            "p_consensus": round(pcons * 100.0, 1),
            "p_model": round(pmod * 100.0, 1),
            "model_over": round((pmod - pcons) * 100.0, 1),
        }

    preds = []
    for mk in ("moneylines", "spreads", "totals"):
        for q in slate.get(mk) or []:
            try:
                preds.append(_pred(q))
            except Exception:
                continue
    preds.sort(key=lambda x: abs(x.get("model_over", 0)), reverse=True)
    return {
        "slate_day": slate.get("fetched_at") or "",
        "preds": preds[:60],
    }


if __name__ == "__main__":
    print(json.dumps(_train()))

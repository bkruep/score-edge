"""Golf Top-20 best bets — DataGolf finish-position market support.

The tool ranks golfers by model-vs-market value on the top-20 market of the
current (or selected) tournament:
  * model P(top 20) comes from DataGolf's pre-tournament model,
  * the market fair is Shin de-vigged consensus across the sportsbooks'
    own top-20 prices (betting-tools/outrights, market=top_20),
  * edge = model minus market fair, in percentage points.

Honest labels: the model number is an estimate, the market fair is de-vigged
book prices — neither is a guarantee. Nothing here fabricates a book price we
don't have; without a DataGolf key the page shows how to connect one.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

import edge
import toolbox

_BASE = "https://feeds.datagolf.com"
_SETTINGS = "_toolbox_settings.json"
_CACHE = "_golf_top20.json"
_CACHE_TTL = 600.0  # DataGolf refreshes in batch; 10 minutes is plenty

TOURS = (("pga", "PGA Tour"), ("eur", "DP World Tour"),
         ("chall", "Korn Ferry / Challenge"))

# Value thresholds in percentage points of model-minus-market.
_VALUE_PT = 2.0


def get_key() -> str:
    """DataGolf API key: env var wins, then the locally saved setting."""
    return (os.environ.get("DATAGOLF_KEY") or
            str(toolbox._load(_SETTINGS, {}).get("datagolf_key") or "")).strip()


def set_key(key: str) -> str:
    s = toolbox._load(_SETTINGS, {})
    s["datagolf_key"] = (key or "").strip()
    toolbox._save(_SETTINGS, s)
    return get_key()


def _norm_prob(v: Any) -> float | None:
    """Model probability as a 0..1 fraction (DataGolf may send 0.13 or 13)."""
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    if p > 1.0:
        p /= 100.0
    return p if 0.0 < p < 1.0 else None


def _american(v: Any) -> int | None:
    try:
        a = int(v)
    except (TypeError, ValueError):
        return None
    return a if a else None


_MODEL_KEYS = ("model", "prediction", "pred", "proj", "model_prob",
               "model_prediction", "dg_model", "baseline", "prob")
_ODDS_KEYS = ("odds", "book_odds", "books", "sportsbook_odds")
_NAME_KEYS = ("player_name", "player", "name", "golfer")


def _pick(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _book_odds(v: Any) -> dict[str, int]:
    """Book-name -> american odds from whatever the payload put there."""
    out: dict[str, int] = {}
    if isinstance(v, dict):
        for book, price in v.items():
            a = _american(price if not isinstance(price, dict) else _pick(price, ("odds", "american", "price")))
            if a:
                out[str(book)] = a
    return out


def _iter_records(payload: Any) -> list[dict]:
    """Pull player records out of the (documented but shape-soft) payload.

    Handles a top-level list of rows, a {player: {...}} map, and a wrapped
    {"data": [...] or {...}} container — in that order of likelihood.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for wrap in ("data", "results", "players"):
            inner = payload.get(wrap)
            if isinstance(inner, (list, dict)) and inner is not payload:
                recs = _iter_records(inner)
                if recs:
                    return recs
        # A direct {player_name: {…}} map: promote keys into name fields.
        recs: list[dict] = []
        for k, v in payload.items():
            if isinstance(v, dict):
                r = dict(v)
                if not any(kk in r for kk in _NAME_KEYS):
                    r["player_name"] = k
                recs.append(r)
        if recs:
            return recs
    return []


def _find_str(payload: Any, keys: tuple[str, ...], depth: int = 0) -> str:
    if depth > 3:
        return ""
    if isinstance(payload, dict):
        for k in keys:
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in payload.values():
            if isinstance(v, (dict, list)):
                found = _find_str(v, keys, depth + 1)
                if found:
                    return found
    elif isinstance(payload, list):
        for v in payload[:50]:
            found = _find_str(v, keys, depth + 1)
            if found:
                return found
    return ""


def _parse(payload: Any, tour: str) -> dict:
    records = _iter_records(payload)
    rows: list[dict] = []
    books_seen: set[str] = set()
    for rec in records:
        player = str(_pick(rec, _NAME_KEYS) or "").strip()
        if not player:
            continue
        model = _norm_prob(_pick(rec, _MODEL_KEYS))
        odds_raw = _pick(rec, _ODDS_KEYS)
        if odds_raw is None and "odds" not in rec:
            # Some shapes nest the book map one level down.
            for v in rec.values():
                if isinstance(v, dict) and _book_odds(v):
                    odds_raw = v
                    break
        books = _book_odds(odds_raw)
        if not books and _american(_pick(rec, ("best_odds", "odds_american"))):
            best = _american(_pick(rec, ("best_odds", "odds_american")))
            books = {"best": best}
        if not books or model is None:
            continue
        prices = list(books.values())
        try:
            fair = edge.shin_devig(prices)[0] if len(prices) >= 2 else edge.implied_probability(prices[0])
        except Exception:
            fair = None
        if not fair or not (0.0 < fair < 1.0):
            continue
        best_am = max(prices, key=lambda p: -edge.implied_probability(p))
        best_book = max(books, key=lambda b: -edge.implied_probability(books[b]))
        edge_pts = (model - fair) * 100.0
        books_seen.update(books)
        rows.append({
            "player": player,
            "model_pct": round(model * 100.0, 1),
            "fair_pct": round(fair * 100.0, 1),
            "best_am": int(best_am),
            "best_book": best_book,
            "n_books": len(books),
            "edge_pts": round(edge_pts, 1),
            "fair_dec": round(1.0 / fair, 2),
            "verdict": ("VALUE" if edge_pts >= _VALUE_PT else
                        "AVOID" if edge_pts <= -_VALUE_PT else "PASS"),
        })
    rows.sort(key=lambda r: r["edge_pts"], reverse=True)
    event = _find_str(payload, ("event_name", "tournament_name", "tournament", "event"))
    return {
        "ok": bool(rows),
        "error": "" if rows else "no_rows",
        "tour": tour,
        "event": event,
        "rows": rows,
        "field": len(rows),
        "books": sorted(books_seen),
        "value_n": sum(1 for r in rows if r["verdict"] == "VALUE"),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "parse_keys": (list(payload.keys())[:8] if isinstance(payload, dict) else
                       ([list(payload[0].keys())[:8] if payload and isinstance(payload[0], dict) else []]
                        if isinstance(payload, list) else [])),
    }


def fetch_top20(tour: str = "pga", event: str = "", force: int = 0) -> dict:
    """Model-vs-market Top-20 board for one tournament (10-minute cache)."""
    tour = tour if tour in {t for t, _ in TOURS} else "pga"
    if not get_key():
        return {"ok": False, "error": "no_key", "rows": [], "field": 0,
                "books": [], "value_n": 0, "tour": tour, "event": event,
                "fetched_at": "", "parse_keys": []}

    cached = toolbox._load(_CACHE, {})
    if (not force and cached.get("tour") == tour and cached.get("event") == event
            and time.time() - float(cached.get("ts") or 0) < _CACHE_TTL
            and cached.get("result")):
        return cached["result"]

    params: dict[str, Any] = {
        "tour": tour, "market": "top_20", "odds_format": "american",
        "file_format": "json", "key": get_key(),
    }
    if event:
        params["event"] = event
    try:
        r = requests.get(f"{_BASE}/betting-tools/outrights", params=params, timeout=30)
        payload = r.json() if "json" in r.headers.get("content-type", "") else None
        if payload is None:
            payload = {"error": f"non-json response (http {r.status_code})"}
    except Exception as exc:
        return {"ok": False, "error": f"request failed: {exc}", "rows": [],
                "field": 0, "books": [], "value_n": 0, "tour": tour,
                "event": event, "fetched_at": "", "parse_keys": []}

    if isinstance(payload, dict) and payload.get("error"):
        return {"ok": False, "error": str(payload.get("error"))[:200], "rows": [],
                "field": 0, "books": [], "value_n": 0, "tour": tour,
                "event": event, "fetched_at": "", "parse_keys": []}

    result = _parse(payload, tour)
    if event and not result.get("event"):
        result["event"] = event
    if result["ok"]:
        toolbox._save(_CACHE, {"ts": time.time(), "tour": tour,
                               "event": event, "result": result})
    return result

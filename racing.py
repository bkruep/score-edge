"""Horse Racing Value Finder — FormFav race form, model and prices.

FormFav (api.formfav.com) supplies international thoroughbred / harness /
greyhound racing:
  * GET /v1/form/meetings?date=&race_code=      -> every meeting + full card (free)
  * GET /v1/form?date=&track=&race=             -> race form + runners (free)
  * GET /v1/predictions?date=&track=&race=      -> model win/place probs (Pro)
  * GET /v1/prices?date=&track=&race=           -> live bookmaker win/place (Premium)

The value board ranks runners by model win probability vs the Shin de-vigged
consensus of the bookmakers' own win prices, in percentage points. Two honest
caveats:
  * Coverage is international (AUS/NZ/HK/GB etc.) — FormFav does not cover US
    tracks.
  * The edge only appears on a Pro (predictions) + Premium (prices) key. On the
    free tier the board degrades to a form reader with no edge column printed.

Nothing here fabricates a price we don't have: without a key the page shows how
to connect one, and without a paid tier it labels the missing side plainly.
"""
from __future__ import annotations

import os
import time
from datetime import date as _date, timedelta
from typing import Any

import requests

import edge
import toolbox

_BASE = "https://api.formfav.com/v1"
_SETTINGS = "_toolbox_settings.json"
_CACHE = "_racing_board.json"
_CACHE_TTL = 900.0

RACE_CODES = (("gallops", "Thoroughbred"), ("harness", "Harness"),
              ("greyhounds", "Greyhounds"))

_VALUE_PT = 2.0  # model minus market-fair points to flag VALUE


def _today() -> str:
    return _date.today().isoformat()


def get_key() -> str:
    return (os.environ.get("FORMFAV_KEY") or
            str(toolbox._load(_SETTINGS, {}).get("formfav_key") or "")).strip()


def set_key(key: str) -> str:
    s = toolbox._load(_SETTINGS, {})
    s["formfav_key"] = (key or "").strip()
    toolbox._save(_SETTINGS, s)
    return get_key()


def _get(path: str, params: dict) -> tuple[int, Any]:
    if not get_key():
        return 0, "no_key"
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, timeout=25,
                         headers={"X-API-Key": get_key(),
                                  "User-Agent": "ScoreEdge/1.0"})
    except Exception as exc:
        return 0, f"request failed: {exc}"
    if r.status_code != 200:
        return r.status_code, (r.text or "")[:160]
    try:
        return 200, r.json()
    except Exception:
        return r.status_code, "non-json response"


# ------------------------------------------------------------------ parsing

_NAME_KEYS = ("name", "runnerName", "horse", "horseName", "runner", "selection")
_NUM_KEYS = ("number", "no", "runnerNumber", "tab", "saddle", "barrier")
_JOCKEY_KEYS = ("jockey", "jockeyName", "rider", "jockey_full")
_TRAINER_KEYS = ("trainer", "trainerName")


def _pick(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _num(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _runners(payload: Any) -> list[dict]:
    """Runners out of a {runners: [...]} / {data:{runners:[...]}} payload."""
    if isinstance(payload, dict):
        for key in ("runners", "data", "race", "results"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                got = _runners(inner)
                if got:
                    return got
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                if any(k in inner[0] for k in _NAME_KEYS):
                    return inner
        # maybe the dict maps runner id -> runner
        recs = [v for v in payload.values() if isinstance(v, dict) and _pick(v, _NAME_KEYS)]
        if recs:
            return recs
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _name_of(r: dict) -> str:
    return str(_pick(r, _NAME_KEYS) or "").strip()


def _dec(v: Any) -> float | None:
    try:
        d = float(v)
    except (TypeError, ValueError):
        return None
    return d if d > 1.0 else None


def _runner_prices(r: dict) -> dict[str, float]:
    """book -> best decimal win price for one runner record."""
    out: dict[str, float] = {}
    raw = r.get("prices") if isinstance(r.get("prices"), dict) else None
    if raw is None and isinstance(r.get("odds"), dict):
        raw = r["odds"]
    if isinstance(raw, dict):
        for book, val in raw.items():
            if isinstance(val, dict):
                d = _dec(val.get("win") or val.get("decimal") or val.get("price"))
            else:
                d = _dec(val)
            if d:
                out[str(book)] = d
    return out


def _join_key(r: dict) -> tuple:
    n = _num(_pick(r, _NUM_KEYS))
    if n is not None:
        return ("n", n)
    return ("s", _name_of(r).lower())


# ------------------------------------------------------------------ endpoints

def fetch_meetings(date_str: str, race_code: str, timezone: str = "") -> dict:
    params = {"date": date_str, "race_code": race_code}
    if timezone:
        params["timezone"] = timezone
    status, data = _get("form/meetings", params)
    if status != 200:
        return {"ok": False, "status": status, "error": data, "meetings": []}
    meetings = []
    for m in (data.get("meetings") or []):
        races = []
        for rc in (m.get("races") or []):
            races.append({
                "number": _num(_pick(rc, ("raceNumber", "number", "race", "no"))),
                "name": _pick(rc, ("name", "raceName", "title")) or "",
                "startTime": _pick(rc, ("startTime", "time", "start")) or "",
                "distance": _pick(rc, ("distance", "distanceM", "distanceMetres")) or "",
            })
        meetings.append({
            "slug": m.get("slug") or m.get("track") or "",
            "name": m.get("name") or m.get("trackName") or m.get("track") or m.get("slug") or "Meeting",
            "country": (m.get("country") or m.get("countryCode") or "").upper(),
            "raceType": m.get("raceType") or race_code,
            "timezone": m.get("timezone") or timezone,
            "races": races,
        })
    return {"ok": True, "status": 200, "error": "", "meetings": meetings}


def fetch_race(date_str: str, track: str, race: int) -> dict:
    status, data = _get("form", {"date": date_str, "track": track, "race": race})
    return {"status": status, "data": data if status == 200 else None,
            "error": "" if status == 200 else data}


def fetch_predictions(date_str: str, track: str, race: int) -> dict:
    status, data = _get("predictions", {"date": date_str, "track": track, "race": race})
    probs: dict[tuple, float] = {}
    if status == 200 and isinstance(data, dict):
        for p in (data.get("predictions") or []):
            k = _join_key(p)
            wp = p.get("winProb", p.get("win_prob", p.get("probability")))
            try:
                wp = float(wp)
            except (TypeError, ValueError):
                wp = None
            if wp is not None:
                probs[k] = wp if wp <= 1.0 else wp / 100.0
    return {"status": status, "probs": probs, "error": "" if status == 200 else data}


def fetch_prices(date_str: str, track: str, race: int) -> dict:
    status, data = _get("prices", {"date": date_str, "track": track, "race": race})
    books: dict[tuple, dict[str, float]] = {}
    if status == 200 and isinstance(data, dict):
        for r in _runners(data):
            pr = _runner_prices(r)
            if pr:
                books[_join_key(r)] = pr
    return {"status": status, "books": books, "error": "" if status == 200 else data}


def _devig_best(best_decs: list[float]) -> list[float] | None:
    if len(best_decs) < 2:
        return None
    americans = [edge.to_american(d) for d in best_decs]
    try:
        return edge.shin_devig(americans)
    except Exception:
        return None


def build_board(date_str: str = "", race_code: str = "gallops", track: str = "",
                timezone: str = "", max_races: int = 8, force: int = 0) -> dict:
    date_str = date_str or _today()
    race_code = race_code if race_code in {c for c, _ in RACE_CODES} else "gallops"
    key = get_key()
    base = {"date": date_str, "race_code": race_code, "key_set": bool(key),
            "active_track": track, "meetings": [], "races": [], "tier": "free",
            "error": "", "ok": False,
            "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    if not key:
        return {**base, "error": "no_key"}

    ck = f"{date_str}|{race_code}|{track}|{timezone}"
    if not force:
        cached = toolbox._load(_CACHE, {})
        if (cached.get("key") == ck and cached.get("result")
                and time.time() - float(cached.get("ts") or 0) < _CACHE_TTL):
            return cached["result"]

    mt = fetch_meetings(date_str, race_code, timezone)
    if not mt["ok"]:
        msg = mt["error"] if isinstance(mt["error"], str) else "meetings request failed"
        return {**base, "error": f"meetings: {msg}", "status": mt["status"]}
    meetings = mt["meetings"]
    base["meetings"] = [{"slug": m["slug"], "name": m["name"], "country": m["country"],
                         "race_n": len(m["races"])} for m in meetings]
    if not meetings:
        return {**base, "error": "no_meetings"}

    chosen = None
    if track:
        chosen = next((m for m in meetings if m["slug"] == track), None)
    chosen = chosen or meetings[0]
    base["active_track"] = chosen["slug"]
    base["active_meeting"] = {"slug": chosen["slug"], "name": chosen["name"],
                              "country": chosen["country"]}

    model_tier = None  # "ok" | "denied"
    price_tier = None
    for rc in chosen["races"][:max_races]:
        num = rc["number"]
        if num is None:
            continue
        form = fetch_race(date_str, chosen["slug"], num)
        preds = fetch_predictions(date_str, chosen["slug"], num)
        prices = fetch_prices(date_str, chosen["slug"], num)
        if preds["status"] == 403:
            model_tier = "denied"
        elif preds["status"] == 200:
            model_tier = "ok"
        if prices["status"] == 403:
            price_tier = "denied"
        elif prices["status"] == 200:
            price_tier = "ok"

        runners = _runners((form.get("data") or {}) if form["data"] else {})
        rows: list[dict] = []
        best_decs: list[float] = []
        for r in runners:
            k = _join_key(r)
            pr = prices["books"].get(k) or _runner_prices(r)
            best_book = max(pr, key=lambda b: pr[b]) if pr else ""
            best_dec = pr[best_book] if pr else None
            rows.append({
                "number": _num(_pick(r, _NUM_KEYS)),
                "name": _name_of(r) or "—",
                "jockey": str(_pick(r, _JOCKEY_KEYS) or "").strip(),
                "trainer": str(_pick(r, _TRAINER_KEYS) or "").strip(),
                "form": str(_pick(r, ("form", "recentForm", "lastStarts")) or "").strip()[:18],
                "model": (preds["probs"].get(k) or 0.0) * 100.0 if preds["probs"] else None,
                "best_dec": best_dec,
                "best_book": best_book,
                "fair": None,
                "edge_pts": None,
                "verdict": "—",
            })
            if best_dec:
                best_decs.append(best_dec)
        fairs = _devig_best(best_decs) if best_decs else None
        if fairs is not None:
            # match de-vig list back to rows in the same order best_decs was built
            di = 0
            for row in rows:
                if row["best_dec"]:
                    row["fair"] = fairs[di] * 100.0
                    di += 1
        for row in rows:
            if row["model"] is not None and row["fair"] is not None:
                e = row["model"] - row["fair"]
                row["edge_pts"] = round(e, 1)
                row["verdict"] = ("VALUE" if e >= _VALUE_PT else
                                  "AVOID" if e <= -_VALUE_PT else "PASS")
        rows.sort(key=lambda x: (x["edge_pts"] if x["edge_pts"] is not None else -999,
                                 x["model"] if x["model"] is not None else -999), reverse=True)
        base["races"].append({
            "number": num, "name": rc["name"], "startTime": rc["startTime"],
            "distance": rc["distance"], "runners": rows,
            "model_ok": bool(preds["probs"]), "market_ok": bool(prices["books"]),
            "form_error": form.get("error") or "",
        })

    if model_tier == "denied" or price_tier == "denied":
        base["tier"] = "free"
    elif model_tier == "ok" and price_tier == "ok":
        base["tier"] = "pro+premium"
    elif model_tier == "ok":
        base["tier"] = "pro"
    base["model_available"] = model_tier == "ok"
    base["market_available"] = price_tier == "ok"
    base["ok"] = bool(base["races"])
    if not base["ok"]:
        base["error"] = "no_races"
    toolbox._save(_CACHE, {"ts": time.time(), "key": ck, "result": base})
    return base

"""Calibration ledger: an append-only prediction log for the statistical
(non-market) boards, plus best-effort outcome resolution against the leagues'
public game feeds, plus honest calibration math.

Nothing here invents numbers: predictions are logged exactly as served, results
are joined from real final games, and the /calibration page reports raw
collected/resolved/pending splits with Brier / log-loss / per-bucket hit rates.
"""
from __future__ import annotations

import os
import time
import json
import math
import threading
import datetime as _dt
from collections import defaultdict
from typing import Any

_DIR = os.environ.get("SCOREEDGE_TRACK_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_PRED = os.path.join(_DIR, "pred_log.jsonl")
_RES = os.path.join(_DIR, "pred_results.jsonl")
_TS = os.path.join(_DIR, "resolve_ts.json")
_lock = threading.Lock()


def _read(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    out.append(json.loads(ln))
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    return out


def _write(path: str, rows: list[dict]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    except OSError:
        pass


def _sig(source: str, day: str, item: str) -> str:
    return f"{source}|{day}|{item}"


def log(source: str, day: str, item: str, prob: float,
        meta: dict | None = None) -> bool:
    """Append one prediction line (deduped by source|day|item)."""
    with _lock:
        rows = _read(_PRED)
        key = _sig(source, day, item)
        if any(_sig(r.get("source", ""), r.get("day", ""), r.get("item", "")) == key
               for r in rows):
            return False
        rows.append({
            "source": source, "day": day, "item": item,
            "prob": round(float(prob or 0), 3),
            "meta": meta or {}, "ts": time.time(),
        })
        rows.sort(key=lambda r: r.get("day", ""))
        _write(_PRED, rows)
        return True


def mark(source: str, day: str, item: str, status: str,
         detail: str = "") -> None:
    with _lock:
        rows = _read(_RES)
        key = _sig(source, day, item)
        rows = [r for r in rows if _sig(r.get("source", ""), r.get("day", ""),
                                        r.get("item", "")) != key]
        rows.append({"source": source, "day": day, "item": item,
                     "status": status, "detail": detail, "ts": time.time()})
        _write(_RES, rows)


def joined(source: str | None = None) -> list[dict]:
    res = _read(_RES)
    rmap = {_sig(r.get("source", ""), r.get("day", ""), r.get("item", "")): r
            for r in res}
    out: list[dict] = []
    for p in _read(_PRED):
        if source and p.get("source") != source:
            continue
        r = rmap.get(_sig(p.get("source", ""), p.get("day", ""), p.get("item", "")))
        out.append({**p, "status": (r or {}).get("status", "pending"),
                    "detail": (r or {}).get("detail", "")})
    return out


def _brier_logloss(rows: list[dict]) -> tuple[float | None, float | None]:
    bin_rows = [r for r in rows if r["status"] in ("won", "lost")]
    if not bin_rows:
        return None, None
    p = [min(max(float(r.get("prob") or 0) / 100.0, 0.001), 0.999) for r in bin_rows]
    y = [1 if r["status"] == "won" else 0 for r in bin_rows]
    n = len(p)
    brier = sum((y[i] - p[i]) ** 2 for i in range(n)) / n
    ll = sum(-(y[i] * math.log(p[i]) + (1 - y[i]) * math.log(1 - p[i]))
             for i in range(n)) / n
    return round(brier, 4), round(ll, 4)


def summary(source: str | None = None) -> dict[str, dict]:
    j = joined(source)
    per: dict[str, dict] = defaultdict(lambda: {"logged": 0, "w": 0, "l": 0,
                                                 "push": 0, "pending": 0,
                                                 "days": set()})
    for r in j:
        s = per[r["source"]]
        s["logged"] += 1
        s["days"].add(r["day"])
        st = r["status"]
        if st == "won":
            s["w"] += 1
        elif st == "lost":
            s["l"] += 1
        elif st == "push":
            s["push"] += 1
        else:
            s["pending"] += 1
    out: dict[str, dict] = {}
    for k, v in per.items():
        res = v["w"] + v["l"] + v["push"]
        brier, ll = _brier_logloss([r for r in j if r["source"] == k])
        out[k] = {
            "logged": v["logged"], "won": v["w"], "lost": v["l"],
            "push": v["push"], "pending": v["pending"], "resolved": res,
            "days": sorted(v["days"]),
            "hit_pct": round(v["w"] / res * 100, 1) if res else None,
            "brier": brier, "logloss": ll,
        }
    return out


_BUCKETS = [("0-10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30),
            ("30-40", 30, 40), ("40-50", 40, 50), ("50+", 50, 101)]


def buckets(source: str | None = None) -> list[dict]:
    j = [r for r in joined(source) if r["status"] in ("won", "lost")]
    agg = defaultdict(lambda: {"n": 0, "w": 0, "ps": 0.0})
    for r in j:
        p = float(r.get("prob") or 0)
        for label, lo, hi in _BUCKETS:
            if lo <= p < hi:
                b = agg[label]
                b["n"] += 1
                b["w"] += 1 if r["status"] == "won" else 0
                b["ps"] += p
                break
    out = []
    for label, lo, hi in _BUCKETS:
        b = agg[label]
        if not b["n"]:
            continue
        out.append({
            "label": label, "n": b["n"],
            "pred": round(b["ps"] / b["n"], 1),
            "actual": round(b["w"] / b["n"] * 100, 1),
            "d": round(b["w"] / b["n"] * 100 - b["ps"] / b["n"], 1),
        })
    return out


def ingest(board: dict, source: str = "goal_hunt") -> int:
    """Log every prediction on a fresh board, then (for past days) try to
    resolve any still-pending rows from the first log batch cheaply."""
    day = board.get("date") or ""
    n = 0
    for r in (board.get("rows") or []):
        player = r.get("player") or r.get("p") or ""
        prob = r.get("prob")
        if not player or prob is None:
            continue
        if log(source, day, player, float(prob), {
                "team": r.get("team", ""), "pos": r.get("pos", ""),
                "game": r.get("game", "")}):
            n += 1
    sync(source, day)
    return n


def _attempted(source: str, day: str) -> bool:
    try:
        with open(_TS, "r", encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        m = {}
    return m.get(_sig(source, day, ""), 0) > time.time() - 3600


def _mark_attempt(source: str, day: str) -> None:
    try:
        with open(_TS, "r", encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        m = {}
    m[_sig(source, day, "")] = time.time()
    with open(_TS, "w", encoding="utf-8") as fh:
        json.dump(m, fh)


def _last_word(name: str) -> str:
    return (name or "").strip().lower().rsplit(None, 1)[-1] if (name or "").strip() else ""


def resolve_nhl(day: str) -> int:
    """Mark logged goal-hunt rows for a past day won/lost from NHL gamecenter."""
    import hockey
    try:
        sched = hockey.fetch_schedule(day)
    except Exception:
        return 0
    res = resolved_map("goal_hunt")
    pending = [r for r in joined("goal_hunt")
               if r["day"] == day and r["status"] == "pending"]
    if not pending or not sched:
        return 0
    scorer: dict[Any, dict[str, set[str]]] = {}
    games_for_abbr: dict[str, Any] = {}
    ready: set[Any] = set()
    for g in sched:
        if str(g.get("state") or "").upper() not in ("FINAL", "OFF", "FINAL SO", "FINAL/OT"):
            continue
        gid = g.get("id")
        try:
            landing = hockey._get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/landing")
        except Exception:
            continue
        scoring = (landing.get("summary") or {}).get("scoring")
        if scoring is None:
            continue
        teams: dict[str, set[str]] = {}
        for item in scoring:
            for gol in item.get("goals") or []:
                tab = ((gol.get("teamAbbrev") or {}).get("default") or "").lower()
                nm = (gol.get("name") or {}).get("default") or ""
                teams.setdefault(tab, set()).add(_last_word(nm))
        scorer[gid] = teams
        ready.add(gid)
        games_for_abbr[(g.get("away") or "").lower()] = gid
        games_for_abbr[(g.get("home") or "").lower()] = gid
    if not ready:
        return 0
    n = 0
    for r in pending:
        tab = (r.get("meta") or {}).get("team", "").lower()
        gid = games_for_abbr.get(tab)
        if gid is None or gid not in ready:
            continue
        names = (scorer.get(gid) or {}).get(tab) or set()
        hit = _last_word(r["item"]) in names
        mark("goal_hunt", day, r["item"], "won" if hit else "lost", "nhl_goal")
        n += 1
    return n


def resolve_mlb(day: str) -> int:
    """Mark logged home-run-hunt rows for a past day won/lost from MLB boxscores."""
    import baseball
    try:
        sched = baseball.fetch_schedule(day)
    except Exception:
        return 0
    pending = [r for r in joined("homerun_hunt")
               if r["day"] == day and r["status"] == "pending"]
    if not pending or not sched:
        return 0
    games_for_abbr: dict[str, dict[str, Any]] = {}
    for g in sched:
        games_for_abbr[(g.get("away") or "").lower()] = g
        games_for_abbr[(g.get("home") or "").lower()] = g
    hit_names: dict[str, set[str]] = {}
    ready: set[str] = set()
    for g in sched:
        if not str(g.get("status") or "").startswith("Final"):
            continue
        pk = g.get("gamePk")
        if not pk:
            continue
        try:
            box = baseball._get(f"game/{pk}/boxscore", {})
        except Exception:
            continue
        for ab, abbr_key in (("away", g.get("away")), ("home", g.get("home"))):
            side = (box.get("teams") or {}).get(ab) or {}
            players = side.get("players")
            if players is None:
                continue
            abbr = (abbr_key or "").lower()
            hit_names.setdefault(abbr, set())
            ready.add(abbr)
            for pid, node in players.items():
                st = (node.get("stats") or {}).get("batting") or {}
                if int(st.get("homeRuns") or 0) > 0:
                    nm = (node.get("person") or {}).get("fullName") or ""
                    if nm:
                        hit_names[abbr].add(_last_word(nm))
    if not ready:
        return 0
    n = 0
    for r in pending:
        tab = (r.get("meta") or {}).get("team", "").lower()
        if tab not in ready:
            continue
        names = hit_names.get(tab) or set()
        hit = _last_word(r["item"]) in names
        mark("homerun_hunt", day, r["item"], "won" if hit else "lost", "mlb_hr")
        n += 1
    return n


def sync(source: str, day: str) -> int:
    if not day or day >= _dt.date.today().isoformat():
        return 0
    if _attempted(source, day):
        return 0
    _mark_attempt(source, day)
    try:
        if source == "goal_hunt":
            return resolve_nhl(day)
        if source == "homerun_hunt":
            return resolve_mlb(day)
    except Exception:
        pass
    return 0


def resolve(source: str, day: str) -> int:
    if source == "goal_hunt":
        return resolve_nhl(day)
    if source == "homerun_hunt":
        return resolve_mlb(day)
    return 0


def resolved_map(source: str | None = None) -> dict[str, dict]:
    return {_sig(r.get("source", ""), r.get("day", ""), r.get("item", "")): r
            for r in _read(_RES) if not source or r.get("source") == source}
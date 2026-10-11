"""Suggested picks ledger: log recommendations (DFS lineups, best bets,
goal/homerun hunt, etc.) with provenance, then settle to won/lost/push and
feed settled examples into the learning model."""
from __future__ import annotations

import json
import os
import threading
import time
import datetime as _dt
from typing import Any

import journal
import track

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_PICKS = os.path.join(_DIR, "picks.jsonl")
_lock = threading.Lock()
_seen: set[str] = set()


def _today_iso() -> str:
    return _dt.date.today().isoformat()

def _sig(p: dict) -> str:
    day = str(p.get("day") or "")
    source = str(p.get("source") or "")
    mkt = str(p.get("mkt") or "")
    game = str(p.get("game") or "")
    player = str(p.get("player") or "")
    side = str(p.get("side") or "")
    line = p.get("line")
    return f"{day}|{source}|{mkt}|{game}|{player}|{side}|{line}"


def record_pick(pick: dict) -> bool:
    with _lock:
        rows = _read()
        p = dict(pick)
        p.setdefault('ts', time.time())
        p.setdefault('day', _today_iso())
        p.setdefault('year', str(p['day'])[:4])
        p.setdefault('status', 'pending')
        key = _sig(p)
        if key in _seen: return False
        for r in rows:
            if _sig(r) == key: return False
        _seen.add(key)
        rows.append(p)
        _write(rows)
        return True



def _read() -> list[dict]:
    out: list[dict] = []
    try:
        with open(_PICKS, "r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    out.append(json.loads(ln))
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    for r in out:
        k = _sig(r)
        if k not in _seen:
            _seen.add(k)
    return out


def _write(rows: list[dict]) -> None:
    rows.sort(key=lambda x: x.get("ts", 0))
    try:
        with open(_PICKS, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    except OSError:
        pass


def settled_picks(season: int | None = None) -> list[dict]:
    rows = _read()
    out: list[dict] = []
    for r in rows:
        st = r.get("status", "pending")
        if st not in ("won", "lost", "push"):
            try:
                sres = settle_pick(r)
                st = sres.get("status", st)
                r["status"] = st
            except Exception:
                pass
        if st in ("won", "lost", "push"):
            out.append(dict(r))
    return out


def settle_pick(r: dict) -> dict:
    mkt = str(r.get("mkt") or "").upper()
    if mkt in ("ML", "SPR", "TOT"):
        jres = journal.settle_entry({
            "market": mkt,
            "game": r.get("game", ""),
            "side": r.get("side", ""),
            "line": r.get("line"),
            "player": r.get("player", ""),
        })
        st = jres.get("status", "pending")
        return {"status": st, **jres}
    player = str(r.get("player") or r.get("item") or r.get("side") or "")
    source = str(r.get("source") or "").lower()
    day = str(r.get("day") or r.get("gameday") or "")
    if source in ("goal_hunt", "homerun_hunt") and player and day:
        try:
            resmap = track.resolved_map(source)
            key = f"{source}|{day}|{player}"
            if key in resmap:
                st = resmap[key].get("status", "pending")
                if st in ("won", "lost"):
                    return {"status": st}
            if day < _today_iso():
                track.sync(source, day)
                resmap = track.resolved_map(source)
                if key in resmap:
                    st = resmap[key].get("status", "pending")
                    if st in ("won", "lost"):
                        return {"status": st}
        except Exception:
            pass
    return {"status": str(r.get("status", "pending"))}


def sync_resolve_all() -> int:
    n = 0
    rows = _read()
    changed = False
    for r in rows:
        if str(r.get("status", "pending")) in ("pending", "undecided"):
            try:
                s = settle_pick(r).get("status", r.get("status"))
                if s != r.get("status"):
                    r["status"] = s
                    changed = True
                    if s in ("won", "lost", "push"):
                        n += 1
            except Exception:
                continue
    if changed:
        _write(rows)
    return n



def to_journal_like(r: dict) -> dict | None:
    st = str(r.get("status", "pending"))
    if st not in ("won", "lost"):
        return None
    am = int(r.get("am") or r.get("odds") or 0)
    edge = float(r.get("edge") or 0.0)
    if not am:
        return None
    game = str(r.get("game") or "")
    mkt = str(r.get("mkt") or "")
    side = str(r.get("side") or r.get("player") or r.get("item") or "")
    line = r.get("line")
    away_ab = journal._team_abbr(game.split(" @ ")[0]) if " @ " in game else None
    home_ab = journal._team_abbr(game.split(" @ ")[1]) if " @ " in game else None
    return {
        "day": r.get("day", _today_iso()),
        "year": str(r.get("day", _today_iso()))[:4],
        "game": game,
        "away_ab": away_ab or "",
        "home_ab": home_ab or "",
        "mkt": mkt,
        "side": side,
        "line": line,
        "am": am,
        "edge": round(edge, 2),
        "status": st,
        "source": r.get("source", ""),
    }

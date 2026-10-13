"""Manual DFS contest results.

DFS lineups carry no odds, so this is the hand-entered label source that lets the
site (over time) learn whether the lineups it suggests actually cash. The user
pastes the winning lineup and its total points — the cash line — and optionally
the score the site's suggested lineup put up.

Nothing here is inferred: names and scores are exactly what was submitted, and
"overlap" is a plain set comparison of last names between the submitted winning
lineup and the lineup the site logged for that slate.
"""
from __future__ import annotations

import json
import os
import threading
import time
import datetime as _dt

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_FILE = os.path.join(_DIR, "dfs_results.jsonl")
_lock = threading.Lock()

_POS = {
    "qb", "rb", "wr", "te", "k", "dst", "def", "flex", "pq", "sf",
    "pg", "sg", "sf", "pf", "c", "g", "f", "util", "sp", "rp", "p",
    "if", "of", "dh", "1b", "2b", "3b", "ss", "g1", "g2", "g3",
}


def _read() -> list[dict]:
    out: list[dict] = []
    try:
        with open(_FILE, "r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    out.append(json.loads(ln))
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    return out


def _write(rows: list[dict]) -> None:
    rows.sort(key=lambda r: (str(r.get("day", "")), r.get("ts", 0)))
    try:
        with open(_FILE, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    except OSError:
        pass


def _clean(name: str) -> str:
    s = " ".join((name or "").replace(".", " ").split())
    toks = s.split()
    if toks and toks[0].lower() in _POS:
        toks = toks[1:]
    return " ".join(toks).strip()


def _last(name: str) -> str:
    parts = [p for p in _clean(name).lower().split()
             if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return parts[-1] if parts else _clean(name).lower()


def parse_lineup(text: str) -> list[str]:
    """Split a pasted lineup into clean player names (newline, comma or ';')."""
    out: list[str] = []
    for row in (text or "").replace(";", "\n").split("\n"):
        for piece in row.split(","):
            c = _clean(piece)
            if c:
                out.append(c)
    return out


def _our_lineup(day: str, sport: str) -> list[str]:
    try:
        import picks
        names = []
        for r in picks._read():
            if str(r.get("day")) != str(day):
                continue
            if str(r.get("mkt")) != "dfs":
                continue
            src = str(r.get("source", ""))
            if sport and not src.startswith(f"dfs_lineup:{sport}"):
                continue
            nm = str(r.get("player") or "")
            if nm:
                names.append(nm)
        return names
    except Exception:
        return []


def record_result(day: str, sport: str, winning_text: str,
                  winning_total=None, our_total=None) -> dict:
    day = str(day or _dt.date.today().isoformat())
    sport = str(sport or "")
    winning = parse_lineup(winning_text)
    our = _our_lineup(day, sport)

    wset = {_last(w) for w in winning}
    oset = {_last(o) for o in our}
    overlap = sorted(wset & oset)

    status = "pending"
    try:
        wt = float(winning_total)
        if our_total not in (None, ""):
            status = "won" if float(our_total) >= wt else "lost"
    except (TypeError, ValueError):
        status = "pending"

    rec = {
        "ts": time.time(),
        "day": day,
        "sport": sport,
        "winning": winning,
        "winning_total": winning_total if winning_total not in (None, "") else None,
        "our": our,
        "our_total": our_total if our_total not in (None, "") else None,
        "overlap": overlap,
        "overlap_n": len(overlap),
        "winning_n": len(winning),
        "status": status,
    }
    with _lock:
        rows = [r for r in _read()
                if not (str(r.get("day")) == day and str(r.get("sport")) == sport)]
        rows.append(rec)
        _write(rows)
    return rec


def results() -> list[dict]:
    rows = _read()
    rows.sort(key=lambda r: (str(r.get("day", "")), r.get("ts", 0)), reverse=True)
    return rows


def summary() -> dict:
    rows = _read()
    decided = [r for r in rows if r.get("status") in ("won", "lost")]
    w = sum(1 for r in decided if r["status"] == "won")
    return {
        "slates": len(rows),
        "decided": len(decided),
        "wins": w,
        "losses": len(decided) - w,
        "hit_pct": round(w / len(decided) * 100, 1) if decided else None,
        "avg_overlap": round(sum(r.get("overlap_n", 0) for r in rows) / len(rows), 1) if rows else None,
    }

"""Edge journal: a persistent, self-settling record of every value spot the
boards have shown.

Each fresh slate appends one line per quote (deduped by signature). Once the
game finishes, ``settled_records`` joins the recorded line against the real
final score from the schedule and marks it won/lost/push, so the "PROOF THE
EDGE WORKS" band and the tracked-slip ledger are computed from real outcomes —
never estimated.
"""
from __future__ import annotations

import os
import time
import json
import threading
import datetime as _dt
from typing import Any

import odds

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_EDGE_FILE = os.path.join(_DIR, "nfl_edge_journal.jsonl")
_lock = threading.Lock()
_seen_sigs: set[str] = set()


def _team_abbr(team: str) -> str | None:
    return odds.NFL_TEAM_ABBREVIATIONS.get((team or "").strip())


def _side_parts(rec: dict) -> tuple[str | None, str | None]:
    """Canonical abbr of the side team (ML/SPR) or 'Over'/'Under' (totals)."""
    mkt = rec.get("mkt")
    side = rec.get("side", "")
    if mkt in ("point_spread", "moneyline"):
        ab = _team_abbr(side)
        return ab, None
    if mkt == "total_points":
        return None, side.strip().casefold()
    return None, None


def _today_iso() -> str:
    return _dt.date.today().isoformat()


def _load_schedule(season: int):
    import data
    return data.load_schedule(season)


def _game_result(rec: dict, season: int) -> dict | None:
    """Settle one journal record against the real final score.

    Returns {status: won|lost|push}, None when the game hasn't been scored.
    """
    side_ab, over_side = _side_parts(rec)
    if side_ab is None and over_side is None:
        return None
    day = str(rec.get("day") or "")
    if not day:
        return None
    y, m, d = day[:4], day[5:7], day[8:10]
    try:
        season = int(y) if y else season
        day_of = _dt.date(int(y), int(m), int(d))
    except (TypeError, ValueError):
        return None
    sched = _load_schedule(season)
    if sched is None or sched.empty:
        return None

    home_ab = str(rec.get("home_ab") or "").strip()
    away_ab = str(rec.get("away_ab") or "").strip()
    if not home_ab or not away_ab:
        return None

    rows = sched[(sched["home_team"] == home_ab) & (sched["away_team"] == away_ab)]
    if rows.empty:
        return None
    try:
        gd = _dt.date.fromisoformat(str(rows.iloc[0]["gameday"]))
    except (TypeError, ValueError):
        gd = day_of
    if abs((gd - day_of).days) > 2:
        return None
    try:
        hs = int(rows.iloc[0]["home_score"])
        as_ = int(rows.iloc[0]["away_score"])
    except (TypeError, ValueError):
        return None
    if hs <= 0 and as_ <= 0:
        # blanks/not played guard (scores are ints once final)
        if hs == 0 and as_ == 0 and str(rows.iloc[0].get("home_score")) == "0":
            return None
        if hs == 0 and as_ == 0:
            return None

    if over_side is not None:  # totals
        line = rec.get("line")
        if line is None:
            return None
        total = hs + as_
        if total > line:
            status = "won" if over_side == "over" else "lost"
        elif total < line:
            status = "lost" if over_side == "over" else "won"
        else:
            status = "push"
        return {"status": status, "total": total, "line": line}

    margin = (hs - as_) if side_ab == home_ab else (as_ - hs)
    if rec.get("mkt") == "moneyline":
        status = "won" if margin > 0 else ("lost" if margin < 0 else "push")
        return {"status": status, "margin": margin}
    # spread
    line = rec.get("line")
    if line is None:
        return None
    if margin > line:
        status = "won"
    elif margin < line:
        status = "lost"
    else:
        status = "push"
    return {"status": status, "margin": margin, "line": line}


def record_slate(slate: dict, day: str | None = None) -> int:
    """Append compact quote records for a slate (deduped). Returns new count."""
    if not isinstance(slate, dict):
        return 0
    day = day or _today_iso()
    lines: list[dict] = []
    now = time.time()
    for key in ("moneylines", "spreads", "totals", "props"):
        for q in slate.get(key) or []:
            e = getattr(q, "edge_pct", 0) or 0
            if e < 0.5:
                continue
            line = getattr(q, "line", None)
            if key == "props" and line is None:
                continue
            label = getattr(q, "game", "") or ""
            away_ab = home_ab = None
            if label and " @ " in label:
                away_ab = _team_abbr(label.split(" @ ")[0])
                home_ab = _team_abbr(label.split(" @ ")[1])
            rec = {
                "ts": now,
                "day": day,
                "year": day[:4],
                "game": label,
                "away_ab": away_ab or "",
                "home_ab": home_ab or "",
                "mkt": getattr(q, "market", "") or key,
                "side": (getattr(q, "selection", "") or
                         getattr(q, "player_name", "") or getattr(q, "label", "") or ""),
                "line": line,
                "am": getattr(q, "best_odds", 0),
                "book": getattr(q, "best_book", "") or "",
                "edge": round(e, 2),
                "pop": getattr(q, "is_value", False),
            }
            sig = (f'{rec["day"]}|{label}|{rec["mkt"]}|{rec["side"]}|'
                   f'{rec["line"]}|{rec["am"]}|{e:.2f}')
            if sig in _seen_sigs:
                continue
            _seen_sigs.add(sig)
            lines.append(rec)
    if not lines:
        return 0
    with _lock:
        with open(_EDGE_FILE, "a", encoding="utf-8") as fh:
            for rec in lines:
                fh.write(json.dumps(rec) + "\n")
    return len(lines)


def load_journal() -> list[dict]:
    out: list[dict] = []
    try:
        with open(_EDGE_FILE, "r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    out.append(json.loads(ln))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    out.sort(key=lambda r: r.get("ts", 0))
    for r in out:
        sig = (f'{r["day"]}|{r["game"]}|{r["mkt"]}|{r["side"]}|'
               f'{r["line"]}|{r["am"]}|{r["edge"]}')
        _seen_sigs.add(sig)
    return out


def prime_from_cache() -> int:
    """Backfill the journal from every slate stored in the disk cache so the
    proof band has real (usually already-settled) boards to show immediately."""
    disk = odds._load_disk_cache()
    n = 0
    for key in sorted(disk.keys()):
        if not (key.startswith("slate_") or key.startswith("bets_slate_")):
            continue
        entry = disk[key]
        val = entry[1] if isinstance(entry, tuple) else entry
        day = key.split("_", 1)[1] if "_" in key else None
        if not day or "-" not in day:
            continue
        if not isinstance(val, dict):
            continue
        try:
            n += record_slate(val, day)
        except Exception:
            continue
    return n


def settled_records(season: int | None = None) -> list[dict]:
    """Journal records joined to their real results (won/lost/push/pending)."""
    recs = load_journal()
    out = []
    for r in recs:
        res = _game_result(r, season or int(r.get("year") or 0) or _dt.date.today().year)
        out.append({**r, "status": (res or {}).get("status", "pending")})
    return out


def proof(season: int | None = None) -> dict:
    """The PROOF band payload: logged vs settled, record, flat-1u ROI, buckets."""
    from collections import defaultdict
    recs = settled_records(season)
    logged = len(recs)
    settled = [r for r in recs if r["status"] in ("won", "lost", "push")]
    w = sum(1 for r in settled if r["status"] == "won")
    l = sum(1 for r in settled if r["status"] == "lost")
    p = sum(1 for r in settled if r["status"] == "push")
    spend = sum(1 for r in settled)
    profit = 0.0
    for r in settled:
        if r["status"] != "won":
            continue
        try:
            dec = odds.american_to_decimal(int(r.get("am") or 0))
        except Exception:
            dec = 1.0
        profit += dec - 1.0
    roi_pct = round(profit / spend * 100.0, 1) if spend else 0.0

    buckets = defaultdict(lambda: {"n": 0, "w": 0, "l": 0})
    for r in recs:
        e = r.get("edge", 0.0) or 0.0
        b = ("8%+" if e >= 8 else "5-8%" if e >= 5 else "3-5%" if e >= 3 else "1.5-3%" if e >= 1.5 else "0-1.5%")
        buckets[b]["n"] += 1
        if r["status"] == "won":
            buckets[b]["w"] += 1
        elif r["status"] == "lost":
            buckets[b]["l"] += 1
    order = ["8%+", "5-8%", "3-5%", "1.5-3%", "0-1.5%"]
    by_bucket = []
    for b in order:
        d = buckets[b]
        by_bucket.append({
            "label": b, "n": d["n"], "w": d["w"], "l": d["l"],
            "hit": round(d["w"] / d["n"] * 100) if d["n"] else 0,
        })
    all_edges = [r.get("edge", 0.0) or 0.0 for r in recs]
    today = _today_iso()
    today_recs = [r for r in recs if r.get("day") == today]
    return {
        "logged": logged,
        "settled": len(settled),
        "wins": w, "losses": l, "pushes": p,
        "roi_pct": roi_pct,
        "hit_pct": round(w / (w + l) * 100, 1) if (w + l) else 0.0,
        "avg_edge": round(sum(all_edges) / len(all_edges), 2) if all_edges else 0.0,
        "top_edge": round(max(all_edges), 1) if all_edges else 0.0,
        "by_bucket": by_bucket,
        "today_n": len(today_recs),
        "pending": logged - len(settled),
    }


def settle_entry(entry: dict) -> dict:
    """Resolve one tracked-slip entry (POSTed from the client ledger) to its
    real result. Main markets settle from the schedule; props return 'undecided'
    (weekly player-stat detail isn't part of the aggregated stats feed)."""
    market = str(entry.get("market") or "").upper()
    game = str(entry.get("game") or "")
    side = str(entry.get("side") or "")
    line = entry.get("line")
    player = str(entry.get("player") or "")
    if market in ("ML", "SPR", "TOT"):
        mkt = {"ML": "moneyline", "SPR": "point_spread", "TOT": "total_points"}[market]
        if mkt == "total_points":
            side_full = "Over" if str(side).casefold().startswith("over") else "Under"
        else:
            side_full = side
        rec = {
            "day": _day_from_game(game),
            "year": _day_from_game(game)[:4],
            "game": game,
            "away_ab": _ab(0, game), "home_ab": _ab(1, game),
            "mkt": mkt, "side": side_full, "line": line,
        }
        res = _game_result(rec, int(rec["year"]) or _dt.date.today().year)
        return {"market": market, "game": game, "side": side, "line": line,
                "status": (res or {}).get("status", "pending")}
    return {"market": market, "game": game, "side": side, "line": line,
            "status": "undecided", "note": "prop results settle as weekly stats post"}


def _ab(idx: int, game: str) -> str:
    parts = (game or "").split(" @ ")
    return _team_abbr(parts[idx]) or "" if len(parts) == 2 else ""


def _day_from_game(game: str) -> str:
    for r in reversed(load_journal()):
        if r.get("game") == game and r.get("day"):
            return r["day"]
    return _today_iso()
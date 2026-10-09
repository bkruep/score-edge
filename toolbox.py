"""ScoreEdge toolbox: persistence plus the compute for the secondary tools.

Currently the parlay builder (fair-odds compounding of de-vigged legs). The
heavier tools (golf, and the new best-bets / hunt / racing boards) live in
their own modules; this file holds the shared helpers they use.

Design rules shared with the rest of the app:
  * Everything is labeled as estimate / heuristic when it is; the only things
    called "proof" settle against real final scores (journal).
  * No feeds are fabricated: all inputs come from the odds/slate/DFS modules.
  * Nothing here ever fakes book data we don't have.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any
from collections import defaultdict

import edge
import odds

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_DIR, exist_ok=True)
_lock = threading.Lock()

# Kicker prop markets: the parlay builder is for skill-position scoring props.
# Kicking points/FGs are low-signal and the user asked them out.
_KICKER_RE = re.compile(r"field_goal|kicking|extra_point", re.IGNORECASE)

# Novelty "longest" markets (longest reception/rush/pass): lottery props the
# user does not want cluttering the one-bet-per-player board.
_LONGEST_RE = re.compile(r"longest", re.IGNORECASE)

# Display names for the game-line markets on the parlay board.
_GAME_MARKET_LABELS = {
    "moneyline": "Moneyline",
    "point_spread": "Spread",
    "total_points": "Total",
}


def prop_side(p: Any) -> str:
    """The Over/Under (or team side) of a prop quote.

    NFL props are PlayerProp objects with `.side`; registry sports return
    BetQuote objects where the side lives in `.selection`.
    """
    return str(getattr(p, "side", "") or getattr(p, "selection", "") or "").strip()


def _load(name: str, default: Any) -> Any:
    path = os.path.join(_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return default


def _save(name: str, data: Any) -> None:
    path = os.path.join(_DIR, name)
    with _lock:
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=0)
            os.replace(tmp, path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 1) PARLAY FAIR-ODDS — de-vig each leg, compound the fair prices, and show the
# book's takeout. The board shows ONE best bet per player (the highest-edge
# market for that player) so a single name never repeats across receptions /
# yards / targets, capped per game. Sports with no prop markets (CFB/CBB) fall
# back to game-line legs (moneyline / spread / total).
# ---------------------------------------------------------------------------

def fair_prob(quote: Any) -> float:
    """De-vigged win probability for a quote-like object.

    Uses every book price when we have the book-by-book map (Shin removes the
    favourite-longshot bias); otherwise falls back to best-vs-consensus.
    """
    books = getattr(quote, "book_odds", None) or {}
    prices = [int(p) for p in books.values() if p]
    if len(prices) >= 2:
        try:
            fair = edge.shin_devig(prices)[0]
            if fair and 0.0 < fair < 1.0:
                return fair
        except Exception:
            pass
        prices = [int(quote.best_odds), int(getattr(quote, "consensus_odds", 0) or 0)]
        prices = [p for p in prices if p]
    if len(prices) >= 2:
        fair = edge.proportional_devig(prices)[0]
        return fair if 0.0 < fair < 1.0 else 0.0
    try:
        return edge.implied_probability(int(quote.best_odds))
    except (TypeError, ValueError):
        return 0.0


def _leg(kind: str, game: str, name: str, market: str, side: str,
         line: Any, am: int, fp: float, edge_pct: float, book: str) -> dict:
    dec = odds.american_to_decimal(am)
    return {
        "kind": kind,
        "game": game,
        "player": name,
        "market": market,
        "label": " ".join(str(x) for x in (name, side, line) if x not in (None, "")).strip(),
        "side": side,
        "line": line,
        "am": am,
        "dec": round(dec, 2),
        "fair_pct": round(fp * 100.0, 1),
        "fair_dec": round(1.0 / fp, 2) if fp and fp < 1.0 else None,
        "edge": round(edge_pct, 1),
        "book": book,
    }


def _prop_leg(p: Any, g: str) -> dict | None:
    market = str(getattr(p, "market", "") or "").strip()
    if _KICKER_RE.search(market) or _LONGEST_RE.search(market):
        return None
    side = prop_side(p)
    am = int(getattr(p, "best_odds", 0) or 0)
    if not side or not am:
        return None
    fp = fair_prob(p)
    if not (0.0 < fp < 1.0):
        return None
    return _leg(
        "prop", g, str(getattr(p, "player_name", "") or "").strip(), market,
        side, getattr(p, "line", None), am, fp,
        float(getattr(p, "edge_pct", 0) or 0),
        str(getattr(p, "best_book", "") or "").strip(),
    )


def _game_leg(q: Any) -> dict | None:
    market = str(getattr(q, "market", "") or "").strip()
    sel = str(getattr(q, "selection", "") or "").strip()
    am = int(getattr(q, "best_odds", 0) or 0)
    if not sel or not am:
        return None
    fp = fair_prob(q)
    if not (0.0 < fp < 1.0):
        return None
    return _leg(
        "game", str(getattr(q, "game", "") or "").strip(), sel,
        _GAME_MARKET_LABELS.get(market, market), sel.rstrip("0123456789.+- "),
        getattr(q, "line", None), am, fp,
        float(getattr(q, "edge_pct", 0) or 0),
        str(getattr(q, "best_book", "") or "").strip(),
    )


def parlay_legs(slate: dict, game: str = "", per_game: int = 10) -> list[dict]:
    """Curated parlay legs for a game (or all games).

    Prop sports: one best (highest-edge) bet per player, capped at `per_game`
    players per game. Sports with no prop markets fall back to game-line legs.
    """
    props = slate.get("props") or []
    if props:
        by_game: dict[str, list] = defaultdict(list)
        for p in props:
            g = str(getattr(p, "game", "") or "").strip()
            if g:
                by_game[g].append(p)
        targets = [game] if game else sorted(by_game, key=str.casefold)
        out: list[dict] = []
        for g in targets:
            best: dict[str, dict] = {}
            for p in by_game.get(g) or []:
                leg = _prop_leg(p, g)
                if not leg:
                    continue
                key = (leg["player"] or leg["market"]).casefold()
                cur = best.get(key)
                if cur is None or leg["edge"] > cur["edge"]:
                    best[key] = leg
            ranked = sorted(best.values(), key=lambda r: r["edge"], reverse=True)[:per_game]
            out.extend(ranked)
        out.sort(key=lambda r: (r["game"].casefold(), -r["edge"]))
        return out

    # No prop markets (CFB/CBB): build legs from the game markets instead.
    quotes = (slate.get("moneylines") or []) + (slate.get("spreads") or []) + (slate.get("totals") or [])
    by_game_q: dict[str, list] = defaultdict(list)
    for q in quotes:
        g = str(getattr(q, "game", "") or "").strip()
        if g:
            by_game_q[g].append(q)
    targets = [game] if game else sorted(by_game_q, key=str.casefold)
    out = []
    for g in targets:
        legs = [lg for lg in (_game_leg(q) for q in by_game_q.get(g) or []) if lg]
        legs.sort(key=lambda r: r["edge"], reverse=True)
        out.extend(legs[:per_game])
    out.sort(key=lambda r: (r["game"].casefold(), -r["edge"]))
    return out

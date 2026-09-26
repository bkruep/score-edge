from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd

import data
import odds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# How much the live season's own scoring rate is shrunk toward the prior
# season's rate while the sample is small (games). ~8 games = 50/50 blend.
PRIOR_STRENGTH = 8.0

# Weight each position's scoring toward the deep ball for the DK "Longest TD"
# bonus. WRs (and explosive RBs) carry the long TDs; TEs are red-zone/
# short-yardage; QBs rare carriers.
_DEEP_BIAS = {"WR": 1.0, "RB": 0.95, "TE": 0.35, "QB": 0.2}

_ANYTIME_TD_MARKETS = {"player_touchdowns", "player_anytime_touchdown"}
_LONGEST_MARKETS = {"player_longest_rush", "player_longest_reception"}

_STATS_CACHE: dict[int, pd.DataFrame] = {}
_ROSTERS: dict | None = None


def current_season() -> int:
    """NFL season label by calendar (season N starts in September of year N)."""
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1


def _stats_for(season: int) -> pd.DataFrame:
    """Player season stats for one season — on-disk cache first (instant, immune
    to nfldata.org's flaky TLS), live fetch last. Empty frame when the season has
    no results posted yet (e.g. the current season before Week 1 finishes)."""
    if season in _STATS_CACHE:
        return _STATS_CACHE[season]
    df = pd.DataFrame()
    try:
        with open(os.path.join(SCRIPT_DIR, ".data_cache", f"nfl_stats_{season}.json"),
                  "r", encoding="utf-8") as fh:
            disk = data._materialize(json.load(fh))
        if disk is not None and len(disk):
            df = disk
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        try:
            live = data.load_player_stats(season)
            if len(live):
                df = live
        except Exception:
            df = pd.DataFrame()
    _STATS_CACHE[season] = df
    return df


def _stats_any() -> pd.DataFrame:
    """Most recent season with data (current first, then prior) — kept for
    backward compatibility with callers that just want 'the stats'."""
    season = current_season()
    for s in (season, season - 1, season - 2):
        df = _stats_for(s)
        if not df.empty:
            return df
    return pd.DataFrame()


def _rosters_any() -> dict:
    """Player identity map (full name, position, current team) — Sleeper."""
    global _ROSTERS
    if _ROSTERS:
        return _ROSTERS
    try:
        with open(os.path.join(SCRIPT_DIR, ".data_cache", "sleeper_rosters.json"),
                  "r", encoding="utf-8") as fh:
            _ROSTERS = json.load(fh)
        if _ROSTERS:
            return _ROSTERS
    except Exception:
        pass
    try:
        r = data.load_rosters()
        if r:
            _ROSTERS = r
            return r
    except Exception:
        return {}


def _pos_td_priors(stats: pd.DataFrame) -> dict[str, float]:
    """Mean scoring-TD rate (rush + rec + ST) per position from the reference
    season — the self-taught baseline the picks regress toward."""
    if stats.empty:
        return {}
    td_cols = ["rushing_tds", "receiving_tds", "special_teams_tds"]
    have = [c for c in td_cols if c in stats.columns]
    if not have or "games" not in stats.columns or "position" not in stats.columns:
        return {}
    work = stats.copy()
    work["score_tds"] = work[have].fillna(0).sum(axis=1)
    work["td_rate"] = work["score_tds"] / work["games"].clip(lower=1)
    priors: dict[str, float] = {}
    for pos in SKILL_POSITIONS:
        sub = work[work["position"] == pos]
        if len(sub):
            priors[pos] = float(sub["td_rate"].mean())
    return priors


def _stats_map(stats: pd.DataFrame) -> dict[str, dict]:
    """name(casefold) -> player season aggregates for TD scoring."""
    if stats.empty:
        return {}
    td_cols = [c for c in ("rushing_tds", "receiving_tds", "special_teams_tds")
               if c in stats.columns]
    out: dict[str, dict] = {}
    for _, row in stats.iterrows():
        name = str(row.get("player_display_name") or "").strip()
        if not name:
            continue
        games = float(row.get("games") or 0)
        tds = sum(float(row.get(c) or 0) for c in td_cols)
        out[name.casefold()] = {
            "td_rate": tds / max(games, 1.0),
            "games": int(games),
            "tds": int(tds),
        }
    return out


def _market_signal(props: list) -> dict[str, dict]:
    """Per-player market evidence from the merged props:
    anytime-TD quotes (when books post them) and the Longest Rush/Reception
    over-side implied probabilities (deep-ball ability)."""
    signal: dict[str, dict] = {}
    for p in props:
        name = str(getattr(p, "player_name", "") or "").strip()
        market = str(getattr(p, "market", "") or "").casefold()
        if not name or not market:
            continue
        key = name.casefold()
        entry = signal.setdefault(key, {"anytime": 0.0, "longest": 0.0, "longest_mkt": ""})
        try:
            implied = 1.0 / odds.american_to_decimal(int(getattr(p, "best_odds", 0)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if market in _ANYTIME_TD_MARKETS:
            entry["anytime"] = max(entry["anytime"], implied)
        elif market in _LONGEST_MARKETS:
            if implied > entry["longest"]:
                entry["longest"] = implied
                entry["longest_mkt"] = "Longest Rush" if market == "player_longest_rush" else "Longest Reception"
    return signal


def _display_team(team_abbr: str) -> str:
    try:
        from odds import NFL_TEAM_ABBREVIATIONS
        rev = {v: k for k, v in NFL_TEAM_ABBREVIATIONS.items()}
        return rev.get(team_abbr, team_abbr)
    except Exception:
        return team_abbr


def _canon_game_label(away: str, home: str) -> str:
    """Normalize a provider game label to the full-name 'Away @ Home' format the
    game chips use. Handles abbreviated sides ('CLE @ TB') and mixed input."""
    rev = {v: k for k, v in odds.NFL_TEAM_ABBREVIATIONS.items()}
    out = []
    for side in (away, home):
        side = side.strip()
        if not side:
            return ""
        if side in rev or side in odds.NFL_TEAM_ABBREVIATIONS:
            out.append(side)
        else:
            resolved = rev.get(side) or odds.NFL_TEAM_ABBREVIATIONS.get(side)
            out.append(resolved if resolved else side)
    if out[0].casefold() == out[1].casefold():
        return ""
    return f"{out[0]} @ {out[1]}"


def build(slate: dict) -> tuple[list[dict], list[dict], list[dict], str, str]:
    """DraftKings weekly TD-promo picks for this week's games.

    Returns (anytime-TD picks, longest-TD picks, value picks, warning,
    season label). Scoring rates come from the CURRENT season, shrunk toward
    the prior season while the in-season sample is small, so the picks keep
    learning automatically as 2026 results post. Falls back to the prior
    season alone when the current season has no results yet. Value picks
    weight P(TD) against how contested the payout is (elite/book-favorite
    names split the pot).
    """
    season = current_season()
    cur_stats = _stats_for(season)
    prev_stats = _stats_for(season - 1)
    stats = cur_stats if not cur_stats.empty else prev_stats
    rosters = _rosters_any()
    warn_parts = []
    if stats.empty:
        warn_parts.append("no player stats")
    if not rosters:
        warn_parts.append("no rosters")
    if not cur_stats.empty:
        season_label = f"{season} in-season + {season - 1} baseline"
    else:
        season_label = f"{season - 1} baseline · no {season} results posted yet"

    # Scope to teams on this week's board (events + prop game labels) and map
    # each team to its canonical game label so the chip filter works in-page.
    teams: set[str] = set()
    game_by_abbr: dict[str, str] = {}
    for ev in slate.get("events") or []:
        home = str(getattr(ev, "home_team", "") or "").strip()
        away = str(getattr(ev, "away_team", "") or "").strip()
        label = f"{away} @ {home}" if home and away else ""
        for field, abbr in ((home, odds.NFL_TEAM_ABBREVIATIONS.get(home)),
                            (away, odds.NFL_TEAM_ABBREVIATIONS.get(away))):
            if field and abbr:
                teams.add(abbr)
                if label and abbr not in game_by_abbr:
                    game_by_abbr[abbr] = label
    for p in slate.get("props") or []:
        game = str(getattr(p, "game", "") or "").strip()
        if " @" in game:
            away, home = (x.strip() for x in game.split(" @ ", 1))
            norm = _canon_game_label(away, home)
            if not norm:
                continue
            for x in (away, home):
                abbr = odds.NFL_TEAM_ABBREVIATIONS.get(x)
                if not abbr:
                    abbr = {v: k for k, v in odds.NFL_TEAM_ABBREVIATIONS.items()}.get(x)
                if abbr:
                    teams.add(abbr)
                    game_by_abbr.setdefault(abbr, norm)
    if not teams:
        warn_parts.append("no scoped games")
        return [], [], [], " · ".join(warn_parts) or "no data"

    priors = _pos_td_priors(stats)
    stats_map = _stats_map(stats)
    signal = _market_signal(list(slate.get("props") or []))

    anytime: list[dict] = []
    longest: list[dict] = []
    for info in rosters.values():
        if not isinstance(info, dict):
            continue
        pos = ""
        fp = info.get("fantasy_positions") or []
        if fp and isinstance(fp, list):
            pos = str(fp[0] or "").strip()
        if not pos:
            pos = str(info.get("position") or "").strip()
        team = str(info.get("team") or "").strip()
        if pos not in SKILL_POSITIONS or team not in teams:
            continue
        name = str(info.get("full_name") or "").strip()
        if not name:
            continue
        key = name.casefold()
        srow = stats_map.get(key)
        prior = priors.get(pos, 0.2)
        if srow:
            w = min(1.0, srow["games"] / 12.0)
            p_td = w * srow["td_rate"] + (1.0 - w) * prior
            note = f"{srow['tds']} TD / {srow['games']} gms"
        else:
            p_td = prior
            note = "rookie — position avg"
        p_td = min(p_td * 0.97, 0.85)

        sig = signal.get(key, {})
        anytime_booked = bool(sig.get("anytime"))
        deep = _DEEP_BIAS.get(pos, 0.5)
        longest_mkt = ""
        if sig.get("longest"):
            deep = min(deep + 0.75 * sig["longest"], 1.6)
            longest_mkt = sig.get("longest_mkt", "")
        longest_score = p_td * deep

        base = {
            "name": name,
            "pos": pos,
            "team": team,
            "team_name": _display_team(team),
            "p_pct": round(p_td * 100.0, 1),
            "note": note,
            "game": game_by_abbr.get(team, ""),
        }
        anytime.append(dict(base, booked=anytime_booked))
        longest.append(dict(
            base,
            deep=round(deep, 2),
            score=round(longest_score, 3),
            booked=bool(longest_mkt),
            longest_mkt=longest_mkt,
        ))

    anytime.sort(key=lambda r: r["p_pct"], reverse=True)
    longest.sort(key=lambda r: r["score"], reverse=True)

    # Tag every row with a global rank (position within the panel across ALL
    # games) and a per-game rank (1..10 within its own game) so the board can
    # show EXACTLY ten rows in every filter state: global top-10 for "All",
    # one game's top-10 for a single chip, and the ten best scores across the
    # picked games for multi-select (see tdKeys() in bets.html).
    for _i, _r in enumerate(anytime):
        _r["global_rank"] = _i + 1
        _r["game_rank"] = None
        _r["score_map"] = _r["p_pct"]
    for _i, _r in enumerate(longest):
        _r["global_rank"] = _i + 1
        _r["game_rank"] = None
        _r["score_map"] = _r["score"]
    _per_game = {}
    for _r in anytime:
        _per_game.setdefault(_r["game"], []).append(_r)
    for _g, _rows in _per_game.items():
        _rows.sort(key=lambda r: r["p_pct"], reverse=True)
        for _i, _r in enumerate(_rows[:10], 1):
            _r["game_rank"] = _i

    # Value picks: DK splits the bonus among everyone who picked the winner, so
    # an elite book-favorite shares the pot with every sharp+square. Weight P(TD)
    # by an (honest, proxied) contestedness multiplier — DK does not expose pick
    # counts, so contest is inferred from book spotlight + elite status.
    pool_size = max(len(anytime), 1)
    value: list[dict] = []
    for i, r in enumerate(anytime):
        sig = signal.get(r["name"].casefold(), {})
        if sig.get("anytime"):
            contest = 2.2
            angle = "book favorite — pot splits"
        elif sig.get("longest"):
            contest = 1.6
            angle = "booked deep-ball — contested"
        elif i / pool_size < 0.2:
            contest = 1.5
            angle = "elite scorer — crowd pick"
        else:
            contest = 1.0
            angle = "under the radar — fuller share"
        value.append(dict(
            r,
            value_pct=round(r["p_pct"] / contest, 1),
            angle=angle,
        ))
    value.sort(key=lambda v: v["value_pct"], reverse=True)

    if not anytime and not longest:
        warn_parts.append("no skill players found")

    # Keep top-10 PER GAME (same contract as the Player Props / value board:
    # group server-side, cap 10 per game, flatten). The game chips then reveal
    # a full 10-name board; the "All" chip shows every game's 10 side by side.
    def _per_game_top(rows: list[dict], n: int = 10) -> list[dict]:
        by_game: dict[str, list[dict]] = {}
        for r in rows:
            by_game.setdefault(r["game"], []).append(r)
        out: list[dict] = []
        for g in sorted(by_game):
            out.extend(by_game[g][:n])
        return out

    # Keep top-10 PER GAME flattened for the game chips, with two tags on every
    # row so the board can reveal EXACTLY ten names in every filter state:
    #   scope="all"  -> the global top-10, shown when no game chip is active
    #   scope="game" -> that game's top-10, shown only when its chip is active
    #   grk         -> 1..10 rank within the game (used for multi-game caps)
    def _tagged(rows: list[dict], metric: str) -> list[dict]:
        global10 = [dict(r, scope="all", grk=None) for r in rows[:10]]
        by_game: dict[str, list[dict]] = {}
        for r in rows:
            by_game.setdefault(r.get("game") or "", []).append(r)
        game_rows: list[dict] = []
        for _g in sorted(by_game):
            for _i, _r in enumerate(by_game[_g][:10], 1):
                _r2 = dict(_r, scope="game", grk=_i)
                _r2.pop("scope_rank", None)
                game_rows.append(_r2)
        return global10 + game_rows

    return (
        _tagged(anytime, "p_pct"),
        _tagged(longest, "score"),
        _tagged(value, "value_pct"),
        " · ".join(warn_parts) or "",
    )
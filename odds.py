from __future__ import annotations

import os
import re
import time
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

# Make .env authoritative before reading API keys. The app is sometimes
# launched from a shell that already exports ODDS_API_KEY="" (or a stale
# value); without an explicit override, python-dotenv will not replace it and
# SharpAPI rejects every call with "missing_api_key". Load the file next to
# this module with override=True so this repo's .env always wins.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                 override=True)
except Exception:
    pass

API_KEY = os.getenv("ODDS_API_KEY", "").strip()

SHARPAPI_BASE_URL = "https://api.sharpapi.io/api/v1"
SHARPAPI_EVENTS_URL = f"{SHARPAPI_BASE_URL}/events"
SHARPAPI_BEST_ODDS_URL = f"{SHARPAPI_BASE_URL}/odds/best"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
}

NFL_GAME_MARKETS = ["moneyline", "point_spread", "total_points"]

NFL_PROP_MARKETS = ["player_touchdowns", "player_rushing_yards", "player_rushing_attempts",
                    "player_pass_yds", "player_pass_tds", "player_pass_attempts",
                    "player_pass_completions", "player_pass_interceptions",
                    "player_receiving_yards", "player_receptions", "player_receiving_tds"]

MARKET_LABELS = {
    "moneyline": "Moneyline",
    "point_spread": "Spread",
    "total_points": "Total",
}

PROP_LABELS = {
    "player_touchdowns": "Touchdowns",
    "player_anytime_touchdown": "Anytime TD",
    "player_rushing_yards": "Rushing Yards",
    "player_rushing_attempts": "Rushing Attempts",
    "player_pass_yds": "Passing Yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_attempts": "Pass Attempts",
    "player_pass_completions": "Pass Completions",
    "player_pass_interceptions": "Interceptions",
    "player_receiving_yards": "Receiving Yards",
    "player_receptions": "Receptions",
    "player_receiving_tds": "Receiving TDs",
}

NFL_TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 900
MAX_LINE_HISTORY = 24  # rolling per-market sample cap for the line-move charts

import pickle
import threading

_DISK_CACHE_FILE = os.path.join(os.path.dirname(__file__), "odds_cache.pkl")
_DISK_KEYS = ("slate_", "bets_active_", "props_", "line_", "bets_slate_", "sport_slate_", "league_odds_")

GAME_MARKETS = ["moneyline", "point_spread", "total_points"]
COLLEGE_GAME_MARKETS = GAME_MARKETS

# Display labels for the canonical market grouping, overrideable per league so
# MLB shows "Run Line"/"Total Runs" while the generic default stays Spread/Total.
DEFAULT_MARKET_NAMES = {"moneyline": "Moneyline", "point_spread": "Spread", "total_points": "Total"}

# Prop market display labels (typography-safe, used across the prop board).
PROP_LABELS = {
    "player_home_runs": "Home Runs",
    "player_hits": "Hits",
    "player_total_bases": "Total Bases",
    "player_rbis": "RBIs",
    "player_hits_+_runs_+_rbis": "H+R+RBIs",
    "player_runs": "Runs",
    "player_singles": "Singles",
    "player_doubles": "Doubles",
    "player_triples": "Triples",
    "player_strikeouts": "Strikeouts",
    "player_stolen_bases": "Stolen Bases",
    "player_walks": "Walks",
    "player_fantasy_score": "Fantasy Score",
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_made_threes": "Threes",
    "player_blocks": "Blocks",
    "player_points_+_rebounds": "Points + Rebounds",
    "player_points_+_assists": "Points + Assists",
    "player_rebounds_+_assists": "Rebounds + Assists",
    "player_points_+_rebounds_+_assists": "Pts + Reb + Ast",
    "player_double_double": "Double Double",
    "player_triple_double": "Triple Double",
    "player_goals": "Goals",
    "player_shots": "Shots",
    "player_shots_on_goal": "Shots on Goal",
    "player_plus_minus": "Plus/Minus",
    "player_power_play_points": "Power-Play Points",
    "player_blocked_shots": "Blocked Shots",
    "player_minutes_played": "Minutes",
    "player_faceoffs_won": "Faceoffs Won",
    "player_saves": "Saves",
    "player_passing_yards": "Passing Yards",
    "player_rushing_yards": "Rushing Yards",
    "player_receptions": "Receptions",
    "player_rushing_attempts": "Rush Attempts",
    "player_passing_attempts": "Pass Attempts",
    "player_longest_reception": "Longest Reception",
    "player_longest_rush": "Longest Rush",
}

# League registry: every board route is a row here.
#   markets   the provider's exact market types for that league (defaults to
#             GAME_MARKETS). For MLB that is run_line + total_runs; NHL is
#             puck_line + total_goals; NBA already uses point_spread/total_points.
#   aliases   normalize sport-specific market types onto the canonical grouping
#             the board tables understand (moneyline / point_spread / total_points).
#   collapse_bets  True = college best-bets boards: one row per game, best
#             value side ranked by edge. Absent = per-team rows like the NFL board.
SPORTS = {
    "cfb": {"sport": "football", "league": "ncaaf", "name": "College Football", "icon": "🏈",
            "college": True, "scoped": True, "collapse_bets": True},
    "cbb": {"sport": "basketball", "league": "ncaab", "name": "College Basketball", "icon": "🏀",
            "college": True, "collapse_bets": True},
    "mlb": {"sport": "baseball", "league": "mlb", "name": "Major League Baseball", "icon": "⚾",
            "markets": ["moneyline", "run_line", "total_runs"],
            "aliases": {"run_line": "point_spread", "total_runs": "total_points"},
            "market_names": {"moneyline": "Moneyline", "point_spread": "Run Line", "total_points": "Total Runs"},
            "prop_markets": ["player_home_runs", "player_hits", "player_total_bases", "player_rbis",
                             "player_runs", "player_singles", "player_doubles", "player_strikeouts",
                             "player_walks", "player_stolen_bases"]},
    "nba": {"sport": "basketball", "league": "nba", "name": "NBA", "icon": "🏀",
            "markets": ["moneyline", "point_spread", "total_points"],
            "market_names": {"moneyline": "Moneyline", "point_spread": "Spread", "total_points": "Total"},
            "prop_markets": ["player_points", "player_rebounds", "player_assists",
                             "player_made_threes", "player_points_+_rebounds",
                             "player_points_+_assists", "player_rebounds_+_assists",
                             "player_points_+_rebounds_+_assists", "player_fantasy_score"]},
    "nhl": {"sport": "hockey", "league": "nhl", "name": "NHL", "icon": "🏒",
            "markets": ["moneyline", "puck_line", "total_goals"],
            "aliases": {"puck_line": "point_spread", "total_goals": "total_points"},
            "market_names": {"moneyline": "Moneyline", "point_spread": "Puck Line", "total_points": "Total Goals"},
            "prop_markets": ["player_goals", "player_assists", "player_points",
                             "player_shots_on_goal", "player_shots", "player_plus_minus"]},
    # Soccer moneyline carries a third "draw" outcome. The board's two-sided
    # arbitrage and complement math assume home/away, so MLS quotes get the
    # draw collapsed into a single best-price side per game the same way
    # college boards collapse their boards (collapse_bets).
    "mls": {"sport": "soccer", "league": "mls", "name": "MLS", "icon": "⚽",
            "markets": ["moneyline"],
            "market_names": {"moneyline": "Moneyline"},
            "collapse_bets": True,
            "prop_markets": ["player_shots_on_target", "player_shots", "player_goals",
                             "player_assists", "player_cards"]},
}
COLLEGE_SPORTS = {t: SPORTS[t] for t in ("cfb", "cbb")}

# "Suggestible" band for value spots. Ratio edge (%) explodes on longshots: a
# +8000/-5620 mismatch reads as "+41% edge" while buying only ~0.5 points of
# real win probability. Restrict suggestions to lines priced like a real play
# (-500..+300) and rank them by true edge in points of implied probability.
VALUE_ODDS_MIN = -500
VALUE_ODDS_MAX = 300
VALUE_MIN_PTS = 1.0

# ---- CFB board scoping: allow only Power-4 + Group-of-5 games -------------
# SharpAPI's CFB feed carries team names but no conference metadata, and it
# posts every FBS + FCS game (64+ matchups on a normal Saturday). To keep the
# board on real, bettable games only, map every allowed FBS school to its
# conference grouping (ACC / B1G / B12 / SEC for the Power-4, "G5" for AAC,
# C-USA, MAC, Mountain West, Sun Belt + the 2026 Pac-12 raid teams). Any team
# absent from this map (FCS schools, independents like Notre Dame) makes the
# whole game out of scope. Keys are normalized (lowercase, punctuation ->
# spaces; "Texas A&M" a&m -> "a m").
_FB_P4 = {"ACC", "B1G", "B12", "SEC"}

# Group of 5, tagged as G5. MW+2026-Pac-12 flux teams are kept under G5 so the
# board doesn't drop Boise/CSU/Fresno/SDSU/Utah State over the realignment.
_FB_MAP: dict[str, str] = {
    # ACC
    "boston college": "ACC", "california": "ACC", "cal": "ACC", "clemson": "ACC",
    "duke": "ACC", "florida state": "ACC", "georgia tech": "ACC", "louisville": "ACC",
    "miami florida": "ACC", "miami": "ACC", "north carolina": "ACC", "nc state": "ACC",
    "north carolina state": "ACC", "pittsburgh": "ACC", "pitt": "ACC", "smu": "ACC",
    "stanford": "ACC", "syracuse": "ACC", "virginia": "ACC", "virginia tech": "ACC",
    "wake forest": "ACC",
    # Big Ten
    "illinois": "B1G", "indiana": "B1G", "iowa": "B1G", "maryland": "B1G",
    "michigan": "B1G", "michigan state": "B1G", "minnesota": "B1G", "nebraska": "B1G",
    "northwestern": "B1G", "ohio state": "B1G", "oregon": "B1G", "penn state": "B1G",
    "purdue": "B1G", "rutgers": "B1G", "ucla": "B1G", "usc": "B1G",
    "southern california": "B1G", "washington": "B1G", "wisconsin": "B1G",
    # Big 12
    "arizona": "B12", "arizona state": "B12", "baylor": "B12", "byu": "B12",
    "cincinnati": "B12", "colorado": "B12", "houston": "B12", "iowa state": "B12",
    "kansas": "B12", "kansas state": "B12", "oklahoma state": "B12", "tcu": "B12",
    "texas tech": "B12", "ucf": "B12", "central florida": "B12", "utah": "B12",
    "west virginia": "B12",
    # SEC
    "alabama": "SEC", "arkansas": "SEC", "auburn": "SEC", "florida": "SEC",
    "georgia": "SEC", "kentucky": "SEC", "lsu": "SEC", "mississippi": "SEC",
    "ole miss": "SEC", "mississippi state": "SEC", "missouri": "SEC", "oklahoma": "SEC",
    "south carolina": "SEC", "tennessee": "SEC", "texas a m": "SEC", "texas": "SEC",
    "texas a&m": "SEC", "vanderbilt": "SEC",
    # Group of 5 --- AAC
    "army": "G5", "navy": "G5", "charlotte": "G5", "east carolina": "G5",
    "florida atlantic": "G5", "fau": "G5", "memphis": "G5", "north texas": "G5",
    "rice": "G5", "south florida": "G5", "usf": "G5", "temple": "G5", "tulane": "G5",
    "tulsa": "G5", "uab": "G5", "alabama birmingham": "G5", "utsa": "G5",
    # Group of 5 --- C-USA
    "delaware": "G5", "florida international": "G5", "fiu": "G5",
    "jacksonville state": "G5", "kennesaw state": "G5", "liberty": "G5",
    "louisiana tech": "G5", "middle tennessee": "G5", "mtsu": "G5",
    "sam houston": "G5", "sam houston state": "G5", "western kentucky": "G5",
    "wku": "G5",
    # Group of 5 --- MAC
    "akron": "G5", "ball state": "G5", "bowling green": "G5", "buffalo": "G5",
    "central michigan": "G5", "eastern michigan": "G5", "kent state": "G5",
    "massachusetts": "G5", "umass": "G5", "miami ohio": "G5", "northern illinois": "G5",
    "ohio": "G5", "toledo": "G5", "western michigan": "G5",
    # Group of 5 --- Mountain West / 2026 Pac-12 raid (kept as G5)
    "air force": "G5", "boise state": "G5", "colorado state": "G5", "fresno state": "G5",
    "hawaii": "G5", "nevada": "G5", "new mexico state": "G5", "new mexico": "G5",
    "san diego state": "G5", "san jose state": "G5", "unlv": "G5", "utah state": "G5",
    "wyoming": "G5",
    # Group of 5 --- Sun Belt
    "appalachian state": "G5", "app state": "G5", "arkansas state": "G5",
    "coastal carolina": "G5", "georgia southern": "G5", "georgia state": "G5",
    "james madison": "G5", "jmu": "G5", "louisiana": "G5", "marshall": "G5",
    "old dominion": "G5", "odu": "G5", "south alabama": "G5", "southern miss": "G5",
    "southern mississippi": "G5", "texas state": "G5", "troy": "G5",
    "ul monroe": "G5", "louisiana monroe": "G5", "ulm": "G5",
    # Abbr/nickname aliases the feeds post alongside full names ("KANSAS ST",
    # "DEL", "ALA", "(11) Texas Tech"). Longest-prefix match resolves them to
    # the same school as the full name.
    "kansas st": "kansas state", "mississippi st": "mississippi state",
    "florida st": "florida state", "michigan st": "michigan state",
    "ohio st": "ohio state", "oklahoma st": "oklahoma state",
    "texas st": "texas state", "nc st": "north carolina state",
    "sam houston st": "sam houston state", "akr": "akron", "del": "delaware",
    "ala": "alabama", "ill": "illinois", "ark": "arkansas",
    "bama": "alabama", "miami fl": "miami florida",
    "san diego st": "san diego state", "san jose st": "san jose state",
    "fresno st": "fresno state", "colorado st": "colorado state",
    "boise st": "boise state", "utah st": "utah state",
    "washington st": "washington state", "kent st": "kent state",
    "ball st": "ball state", "app st": "appalachian state",
}

# FCS (or low-division) schools that share a Power-4/G5 prefix and would
# otherwise false-match ("Tennessee State" -> "Tennessee", "North Carolina
# Central" -> "North Carolina", "Missouri State" -> "Missouri", ...). Blocking
# them first keeps every member of this list out of the board.
_FB_BLOCK = (
    "tennessee state", "texas southern", "texas a m commerce", "texas a&m commerce",
    "alabama a m", "alabama a&m", "alabama state", "indiana state", "florida a m",
    "florida a&m", "florida gulf coast", "jackson state", "south carolina state",
    "north carolina central", "north carolina a t", "north carolina a&t",
    "houston christian", "houston baptist", "missouri state", "delaware state",
)

_FB_SORTED = sorted(_FB_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)

# Display names for aliases that don't title-case cleanly ("texas a m",
# "ucf", "app state", ...). Default fallback is key.title().
_FB_DISPLAY = {
    "texas a m": "Texas A&M", "texas a&m": "Texas A&M",
    "california": "Cal", "cal": "Cal",
    "miami florida": "Miami (FL)", "miami": "Miami (FL)",
    "alabama birmingham": "UAB", "uab": "UAB",
    "central florida": "UCF", "ucf": "UCF",
    "south florida": "USF", "usf": "USF",
    "florida atlantic": "FAU", "fau": "FAU",
    "florida international": "FIU", "fiu": "FIU",
    "middle tennessee": "MTSU", "mtsu": "MTSU",
    "western kentucky": "WKU", "wku": "WKU",
    "louisiana monroe": "ULM", "ul monroe": "ULM", "ulm": "ULM",
    "north carolina state": "NC State", "nc state": "NC State", "nc st": "NC State",
    "appalachian state": "App State", "app state": "App State",
    "app st": "App State",
    "southern miss": "Southern Miss", "southern mississippi": "Southern Miss",
    "pittsburgh": "Pitt", "pitt": "Pitt",
    "mississippi": "Ole Miss", "ole miss": "Ole Miss",
    "sam houston": "Sam Houston", "sam houston state": "Sam Houston",
    "sam houston st": "Sam Houston",
    "southern california": "USC", "usc": "USC",
    "james madison": "JMU", "jmu": "JMU",
    "unlv": "UNLV", "utsa": "UTSA", "utep": "UTEP", "lsu": "LSU",
    "byu": "BYU", "tcu": "TCU", "ucla": "UCLA", "akron": "Akron",
    "san diego st": "San Diego State", "san jose st": "San Jose State",
    "fresno st": "Fresno State", "colorado st": "Colorado State",
    "boise st": "Boise State", "utah st": "Utah State",
    "washington st": "Washington State", "kent st": "Kent State",
    "ball st": "Ball State",
}


def _norm_team(name_raw: str) -> str:
    """Identity key for a team label: accents folded, punctuation dropped,
    ranking prefix removed ("(11) Texas Tech" -> "texas tech").

    Apostrophes and periods must vanish without leaving a space, or "St. John's"
    and "St. Louis" normalize to "st john s" / "st louis" and stop matching
    their alias tables. Accents are folded to ASCII rather than dropped, so
    "Montréal" stays "montreal" instead of fragmenting into "montr al".
    """
    text = unicodedata.normalize("NFKD", (name_raw or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"['`.&]", "", text)          # join: st johns, st louis
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not text:
        return ""
    toks = text.split()
    while toks and toks[0].isdigit():
        toks.pop(0)
    return " ".join(toks)


def _canonical_college_team(selection: str) -> str:
    """Map any SharpAPI college label (full name, mascot variant, \"st\"/abbr,
    ranked prefix) to one canonical school name so per-team rows dedupe."""
    norm = _norm_team(selection)
    if not norm:
        return ""
    for b in _FB_BLOCK:
        if norm.startswith(b):
            return selection.strip() or norm
    for alias, _ in _FB_SORTED:
        if norm.startswith(alias):
            return _FB_DISPLAY.get(alias, alias.title())
    return selection.strip() or norm


# --- Pro team nickname maps ------------------------------------------------
# Canonical display name -> set of provider aliases (full names, official
# nicknames, city codes, common variants). Matching is longest-prefix against
# the normalized label so longer aliases ("los angeles dodgers") win over
# shorter ones ("dodgers"). A league without a map falls back to _pro_team.
_MLB_TEAM_ALIASES = {
    "New York Yankees": {"new york yankees", "ny yankees", "yankees", "nyy"},
    "Boston Red Sox": {"boston red sox", "red sox", "bos"},
    "Baltimore Orioles": {"baltimore orioles", "orioles", "bal"},
    "Tampa Bay Rays": {"tampa bay rays", "tampa rays", "tb rays", "rays", "tb"},
    "Toronto Blue Jays": {"toronto blue jays", "blue jays", "jays", "tor"},
    "Chicago White Sox": {"chicago white sox", "white sox", "cws"},
    "Cleveland Guardians": {"cleveland guardians", "guardians", "cle"},
    "Detroit Tigers": {"detroit tigers", "tigers", "det"},
    "Kansas City Royals": {"kansas city royals", "kc royals", "royals", "kc"},
    "Minnesota Twins": {"minnesota twins", "twins", "min"},
    "Houston Astros": {"houston astros", "astros", "hou"},
    "Los Angeles Angels": {"los angeles angels", "la angels", "angels", "laa"},
    "Oakland Athletics": {"oakland athletics", "athletics", "as", "oak"},
    "Seattle Mariners": {"seattle mariners", "mariners", "sea"},
    "Texas Rangers": {"texas rangers", "texas", "rangers", "tex"},
    "Atlanta Braves": {"atlanta braves", "atl braves", "atla braves", "braves", "atl"},
    "Miami Marlins": {"miami marlins", "marlins", "mia"},
    "New York Mets": {"new york mets", "ny mets", "mets", "nym"},
    "Philadelphia Phillies": {"philadelphia phillies", "phillies", "philly", "phi"},
    "Washington Nationals": {"washington nationals", "nationals", "nats", "was"},
    "Chicago Cubs": {"chicago cubs", "cubs", "chc"},
    "Cincinnati Reds": {"cincinnati reds", "reds", "cin"},
    "Milwaukee Brewers": {"milwaukee brewers", "mil brewers", "brewers", "mil", "mke"},
    "Pittsburgh Pirates": {"pittsburgh pirates", "pirates", "pit"},
    "St. Louis Cardinals": {"st louis cardinals", "cardinals", "stl"},
    "Arizona Diamondbacks": {"arizona diamondbacks", "diamondbacks", "d backs", "dbacks", "snakes", "ari"},
    "Colorado Rockies": {"colorado rockies", "rockies", "col"},
    "Los Angeles Dodgers": {"los angeles dodgers", "la dodgers", "dodgers", "lad"},
    "San Diego Padres": {"san diego padres", "sd padres", "padres", "sd"},
    "San Francisco Giants": {"san francisco giants", "sf giants", "giants", "sf"},
}
_MLB_SORTED = sorted(
    ((alias, display) for display, aliases in _MLB_TEAM_ALIASES.items() for alias in aliases),
    key=lambda pair: len(pair[0]), reverse=True)


def _canonical_pro_mlb(selection: str) -> str:
    """Map any SharpAPI MLB label ("LA Dodgers", "ATL Braves", "Brewers",
    full name) onto the canonical team name so per-team rows dedupe."""
    norm = _norm_team(selection)
    if not norm:
        return ""
    for alias, display in _MLB_SORTED:
        if norm.startswith(alias):
            return display
    return selection.strip() or norm


def _pro_team(selection: str) -> str:
    """Conservative identity key kept as the fallback pro canonicalizer for
    leagues that have no team map yet (returns the label unchanged after
    whitespace collapse so no rows are thrown together accidentally)."""
    return " ".join((selection or "").split())


def _canonical_pro_mls(selection: str) -> str:
    """Map any SharpAPI MLS label onto one canonical club.

    Soccer clubs break a naive "last word" identity badly, because both halves
    of the name are load-bearing: the feed posts "Inter Miami" and "Inter Miami
    CF" (same club), "Sporting KC" and "Sporting Kansas City" (same club), and
    "Columbus" for "Columbus Crew". Without this map a single club splits into
    two outcomes, a three-way market reads as four, and the devigged overround
    balloons to 14%.
    """
    norm = _norm_team(selection)
    if not norm:
        return ""
    for alias, display in _MLS_SORTED:
        if norm == alias or norm.startswith(alias + " "):
            return display
    return selection.strip() or norm


# Club name -> every label the feed is known to use for it. Club suffixes
# ("FC", "CF", "SC") are dropped during normalization, so "Inter Miami CF" and
# "Inter Miami" collide on purpose.
_MLS_TEAM_ALIASES = {
    "Atlanta United": {"atlanta united", "atl utd", "atlanta utd"},
    "Austin FC": {"austin fc", "austin"},
    "CF Montreal": {"cf montreal", "montreal", "cf montreal"},
    "Charlotte FC": {"charlotte fc", "charlotte"},
    "Chicago Fire": {"chicago fire", "chi fire"},
    "Colorado Rapids": {"colorado rapids", "col rapids", "colorado"},
    "Columbus Crew": {"columbus crew", "columbus", "clb"},
    "D.C. United": {"dc united", "d c united", "dc utd", "dc"},
    "FC Cincinnati": {"fc cincinnati", "cincinnati", "cin"},
    "FC Dallas": {"fc dallas", "dallas"},
    "Houston Dynamo": {"houston dynamo", "hou dynamo", "houston"},
    "Inter Miami": {"inter miami", "mia", "int miami", "miami fc", "miami"},
    "Kansas City Current": {"kansas city current", "kc current"},
    "LA Galaxy": {"la galaxy", "lag"},
    "Los Angeles FC": {"los angeles fc", "lafc"},
    "Minnesota United": {"minnesota united", "mn united", "min united"},
    "Nashville SC": {"nashville sc", "nashville"},
    "New England Revolution": {"new england revolution", "ne revolution", "new england"},
    "New York Red Bulls": {"new york red bulls", "ny red bulls", "ny rb"},
    "New York City FC": {"new york city fc", "nycfc", "ny city fc"},
    "Orlando City": {"orlando city", "orlando"},
    "Philadelphia Union": {"philadelphia union", "philadelphia", "phi union"},
    "Portland Timbers": {"portland timbers", "por timbers", "portland"},
    "Real Salt Lake": {"real salt lake", "rs l", "salt lake"},
    "San Diego FC": {"san diego fc", "sd fc"},
    "San Jose Earthquakes": {"san jose earthquakes", "sj earthquakes", "quakes"},
    "Seattle Sounders": {"seattle sounders", "sea sounders", "sounders"},
    "Sporting Kansas City": {"sporting kansas city", "sporting kc", "skc", "kansas city"},
    "St. Louis City": {"st louis city", "stl city"},
    "Toronto FC": {"toronto fc", "toronto", "tfc"},
    "Vancouver Whitecaps": {"vancouver whitecaps", "van whitecaps", "whitecaps"},
}
_MLS_SORTED = sorted(
    ((alias, display) for display, aliases in _MLS_TEAM_ALIASES.items() for alias in aliases),
    key=lambda pair: len(pair[0]), reverse=True)


_PRO_CANON = {"mlb": _canonical_pro_mlb, "mls": _canonical_pro_mls}


def _canonical_game(label_raw: str) -> str:
    """One canonical \"Away @ Home\" label per event. SharpAPI varies the feed
    per market (\"Texas Longhorns @ Tennessee Volunteers\" vs \"Texas @
    Tennessee\"), which makes the same game look like two rows; both teams'
    quotes are resolved to the same canonical school names so a game renders
    once across every market."""
    parts = [p for p in (label_raw or "").split("@")]
    if len(parts) == 2:
        away = _canonical_college_team(parts[0])
        home = _canonical_college_team(parts[1])
        if away and home:
            return f"{away} @ {home}"
    return label_raw.strip() if label_raw else ""


def _team_grouping(name_raw: str) -> str | None:
    """Conference grouping for one team string, or None if out of scope."""
    norm = _norm_team(name_raw)
    if not norm:
        return None
    for b in _FB_BLOCK:
        if norm.startswith(b):
            return None
    for alias, grouping in _FB_SORTED:
        if norm.startswith(alias):
            return grouping
    return None


def _game_grouping(label_raw: str) -> str | None:
    """Grouping label for a full \"Away @ Home\" game, or None if out of scope
    (either side is an FCS school or an independent)."""
    parts = [p.strip() for p in (label_raw or "").split("@")]
    if len(parts) != 2:
        return None
    g1 = _team_grouping(parts[0])
    g2 = _team_grouping(parts[1])
    if g1 is None or g2 is None:
        return None
    for g in (g1, g2):
        if g in _FB_P4:
            return g
    return "G5"

def edge_pts(best_american: int, consensus_american: int) -> float:
    """True betting value: points of implied probability the best price buys
    over consensus. Longshot-invariant (a +8000/+5620 gap is ~0.5 pts, not
    41%), so it ranks favorites and underdogs on the same scale."""
    if not best_american or not consensus_american:
        return 0.0
    p_best = 1.0 / american_to_decimal(best_american)
    p_cons = 1.0 / american_to_decimal(consensus_american)
    return (p_cons - p_best) * 100.0
_disk_lock = threading.Lock()


def _load_disk_cache() -> dict:
    try:
        with open(_DISK_CACHE_FILE, "rb") as fh:
            raw = pickle.load(fh)
        if isinstance(raw, dict):
            return raw
    except (OSError, pickle.PickleError, EOFError):
        pass
    return {}


def _save_disk_cache(snapshot: dict) -> None:
    try:
        with open(_DISK_CACHE_FILE, "wb") as fh:
            pickle.dump(snapshot, fh)
    except (OSError, pickle.PickleError):
        pass


def _cached(key: str):
    if key in _CACHE:
        ts, val = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return val
    if any(key.startswith(p) for p in _DISK_KEYS):
        with _disk_lock:
            disk = _load_disk_cache()
        entry = disk.get(key)
        if entry is not None:
            ts, val = entry
            if time.time() - ts < CACHE_TTL:
                _CACHE[key] = entry
                return val
    return None


def _cache_set(key: str, val) -> None:
    now = time.time()
    _CACHE[key] = (now, val)
    if any(key.startswith(p) for p in _DISK_KEYS):
        with _disk_lock:
            disk = _load_disk_cache()
            disk[key] = (now, val)
            _save_disk_cache(disk)


class OddsError(RuntimeError):
    """Raised when SharpAPI odds cannot be fetched."""


@dataclass(frozen=True, slots=True)
class NflEvent:
    event_id: str
    away_team: str
    home_team: str
    start_time: str

    def label(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


@dataclass(frozen=True, slots=True)
class BetQuote:
    event_id: str
    game: str
    selection: str
    market: str
    best_odds: int
    consensus_odds: int
    best_book: str
    edge_pct: float
    line: float | None = None
    player_name: str | None = None
    book_odds: dict[str, int] | None = None
    book_edges: dict[str, float] | None = None
    sport: str = ""
    league: str = ""
    # The feed's own "this is the line the market is on" flag. Alt lines are
    # posted with both teams carrying the same sign and no main-line flag, so
    # without this the dedupe can pick a nonsense pair whose two sides are both
    # heavy favourites and derive a 36% overround from it.
    is_main_line: bool = False

    def book_ladder(self, top: int = 4) -> list[dict]:
        """Top books by price for this quote: [{book, am, edge}] best first."""
        if not self.book_odds:
            return []
        items = sorted(self.book_odds.items(), key=lambda kv: (kv[1], ), reverse=True)
        out = []
        for book, am in items[:top]:
            e = (self.book_edges or {}).get(book)
            out.append({"book": book, "am": am, "edge": round(e, 1) if e is not None else None})
        return out

    @property
    def is_value(self) -> bool:
        """Best price meaningfully beats consensus (real value spot)."""
        return self.edge_pct >= 1.5

    def edge_pts(self) -> float:
        """True betting value in points of implied probability (longshot-invariant)."""
        return edge_pts(self.best_odds, self.consensus_odds)

    @property
    def is_playable(self) -> bool:
        """Sane suggestion: priced in the real-play band with genuine points edge."""
        if not self.best_odds:
            return False
        if not (VALUE_ODDS_MIN <= self.best_odds <= VALUE_ODDS_MAX):
            return False
        return self.edge_pts() >= VALUE_MIN_PTS

    def slip_payload(self) -> dict:
        """Compact payload the tracked-slip drawer locks: m=market tag,
        g=game, s=selection, l=line, a=american price, b=book, e=edge %,
        p=player (props only)."""
        m = {"moneyline": "ML", "point_spread": "SPR", "total_points": "O/U"}\
            .get(self.market, "PROP" if self.player_name else self.market)
        return {
            "m": m, "g": self.game, "s": self.selection,
            "l": self.line, "a": self.best_odds, "b": self.best_book,
            "e": round(self.edge_pct, 2) if self.edge_pct is not None else None,
            "p": self.player_name or "",
            "sport": self.sport, "league": self.league,
        }


@dataclass(frozen=True, slots=True)
class PlayerProp:
    event_id: str
    game: str
    player_name: str
    market: str
    label: str
    stat_category: str
    side: str
    line: float | None
    best_odds: int
    consensus_odds: int
    best_book: str
    edge_pct: float
    book_odds: dict[str, int] | None = None

    @property
    def is_value(self) -> bool:
        return self.edge_pct >= 1.5

    def book_ladder(self, top: int = 4) -> list[dict]:
        """Top books by price for this prop: [{book, am, edge}] best first."""
        if not self.book_odds:
            return []
        items = sorted(self.book_odds.items(), key=lambda kv: (kv[1], ), reverse=True)
        out = []
        for book, am in items[:top]:
            out.append({"book": book, "am": am, "edge": None})
        return out

    def slip_payload(self) -> dict:
        """Compact tracked-slip payload (m=PROP). See BetQuote.slip_payload."""
        return {
            "m": "PROP", "g": self.game, "s": self.player_name or self.side,
            "l": self.line, "a": self.best_odds, "b": self.best_book,
            "e": round(self.edge_pct, 2) if self.edge_pct is not None else None,
            "p": self.player_name or "",
        }


def american_to_decimal(american: int) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / float(-american)


def decimal_to_american(decimal: float) -> int:
    if decimal <= 1.0:
        return 500
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def odds_html(american: int) -> str:
    if american > 0:
        return f"+{american}"
    return str(american)


def implied_prob(american: int) -> float:
    """Vig-free implied probability of an american price, as a percentage."""
    if not american:
        return 0.0
    return 100.0 / american_to_decimal(american)


def matches_team(selection: str, team_name: str) -> bool:
    if not selection or not team_name:
        return False
    selection = selection.strip()
    if selection.casefold() == team_name.casefold():
        return True
    abbreviation = NFL_TEAM_ABBREVIATIONS.get(team_name)
    if not abbreviation:
        return False
    tokens = [t.upper() for t in selection.split()]
    return any(t == abbreviation or abbreviation.startswith(t) for t in tokens)


def _canonical_team(selection: str) -> str | None:
    """Map a provider team label (abbr+mascot, full name) to the canonical full
    name. Returns None when the label can't be resolved to a real NFL team."""
    sel = (selection or "").strip()
    if not sel:
        return None
    lower = sel.casefold()
    if lower in NFL_TEAM_ABBREVIATIONS:
        return sel
    tokens = {t.upper() for t in sel.split() if t}
    for team, abbr in NFL_TEAM_ABBREVIATIONS.items():
        if abbr in tokens:
            return team
        mascot = team.split()[-1].upper()
        if mascot in tokens and mascot not in _MASCOT_AMBIGUOUS:
            return team
        team_tokens = {t.upper() for t in team.split()}
        if tokens == team_tokens:
            return team
    return None


_MASCOT_AMBIGUOUS = {"NEW", "LOS", "LAS"}


def _request_json(session: requests.Session, url: str, params: dict[str, Any], timeout: int) -> Any:
    headers = {**HEADERS, "X-API-Key": API_KEY}
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, timeout=timeout, headers=headers, params=params)
        except requests.RequestException as error:
            raise OddsError(f"Could not reach SharpAPI: {error}") from error

        if response.status_code == 429:
            if attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise OddsError("SharpAPI rate limit hit. Try again shortly.")
        if response.status_code == 401:
            raise OddsError("SharpAPI rejected the API key. Check ODDS_API_KEY.")
        if response.status_code == 403:
            raise OddsError("SharpAPI tier restriction — upgrade to unlock this data.")
        if response.status_code != 200:
            raise OddsError(f"SharpAPI returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError as error:
            raise OddsError("SharpAPI response was not valid JSON.") from error


def _parse_start_time(raw: Any) -> str:
    try:
        return (
            datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .strftime("%b %d, %I:%M %p ET")
        )
    except ValueError:
        return str(raw)


def fetch_nfl_events(date: str) -> list[NflEvent]:
    cache_key = f"events_{date}"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    payload = _request_json(session, SHARPAPI_EVENTS_URL, {
        "sport": "football", "league": "nfl", "date": date[:10], "limit": 200,
    }, timeout=20)

    events: list[NflEvent] = []
    seen_matchups: set[tuple[str, str]] = set()
    for entry in payload.get("data") or []:
        if entry.get("is_live"):
            continue
        event_id = str(entry.get("id", "")).strip()
        away = str(entry.get("away_team", "")).strip()
        home = str(entry.get("home_team", "")).strip()
        # SharpAPI's feed mixes real games with pseudo-events (alt spreads,
        # "Team goes to @ ...", "Over X @ Under Y", winning-margin markets).
        # Keep only events where both sides are real NFL teams.
        if not event_id or away not in NFL_TEAM_ABBREVIATIONS or home not in NFL_TEAM_ABBREVIATIONS:
            continue
        matchup = (home, away)
        if matchup in seen_matchups:
            continue        # dedupe mirrored/duplicate entries for the same game
        seen_matchups.add(matchup)
        events.append(NflEvent(event_id, away, home, _parse_start_time(entry.get("start_time", ""))))

    _cache_set(cache_key, events)
    return events


def _parse_best_odds_rows(payload: Any, labels: dict[str, str]) -> list[BetQuote]:
    quotes: list[BetQuote] = []
    for entry in payload.get("data") or []:
        best = entry.get("best_odds") or {}
        consensus = entry.get("consensus_odds") or {}
        if not best or not consensus:
            continue
        try:
            best_am = int(best.get("american"))
            cons_am = int(consensus.get("american"))
        except (TypeError, ValueError):
            continue
        if not best_am or not cons_am:
            continue

        market = str(entry.get("market_type", "")).strip().casefold()
        line = None
        try:
            if entry.get("line") is not None:
                line = float(entry.get("line"))
        except (TypeError, ValueError):
            line = None

        book_odds: dict[str, int] | None = None
        book_edges: dict[str, float] | None = None
        all_books = entry.get("all_books") or []
        if all_books:
            book_odds, book_edges = {}, {}
            for b in all_books:
                bname = str(b.get("book") or b.get("sportsbook") or "").strip()
                bo = b.get("odds") or {}
                try:
                    bam = int(bo.get("american"))
                except (TypeError, ValueError):
                    continue
                if not bname or not bam:
                    continue
                book_odds[bname] = bam
                try:
                    if b.get("edge") is not None:
                        book_edges[bname] = float(b.get("edge"))
                except (TypeError, ValueError):
                    pass
            if not book_odds:
                book_odds = book_edges = None

        quotes.append(BetQuote(
            event_id=str(entry.get("event_id", "")).strip(),
            game=labels.get(str(entry.get("event_id", "")).strip(),
                            str(entry.get("event_name", "")).strip() or ""),
            selection=str(entry.get("selection", "")).strip(),
            market=market,
            best_odds=best_am,
            consensus_odds=cons_am,
            best_book=str(entry.get("best_book", "")).strip(),
            edge_pct=(american_to_decimal(best_am) / american_to_decimal(cons_am) - 1.0) * 100.0,
            line=line,
            player_name=str(entry.get("player_name") or "").strip() or None,
            book_odds=book_odds,
            book_edges=book_edges,
            is_main_line=bool(entry.get("is_main_line")),
        ))
    return quotes


def fetch_best_odds(event_ids: list[str], markets: str, labels: dict[str, str],
                    max_rows: int = 500, request_delay: float = 0.2) -> list[BetQuote]:
    if not event_ids:
        return []
    cache_key = f"odds_{','.join(event_ids[:5])}_{markets}"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    quotes: list[BetQuote] = []
    offset = 0
    while True:
        payload = _request_json(session, SHARPAPI_BEST_ODDS_URL, {
            "sport": "football", "league": "nfl",
            "event_id": ",".join(event_ids), "market": markets,
            "limit": 200, "offset": offset,
        }, timeout=20)
        page = _parse_best_odds_rows(payload, labels)
        quotes.extend(page)
        pagination = payload.get("pagination") or {}
        next_offset = pagination.get("next_offset")
        if not pagination.get("has_more") or len(quotes) >= max_rows or not next_offset:
            break
        offset = int(next_offset)
        if request_delay > 0:
            time.sleep(request_delay)

    quotes = quotes[:max_rows]
    _cache_set(cache_key, quotes)
    return quotes


def _dedupe_side_quotes(quotes: list[BetQuote], *, side_by_team: bool,
                        canonicalizer=None) -> list[BetQuote]:
    """Collapse per-team/per-side duplicates into one best row per game.

    SharpAPI posts one row per team (or per over/under side) using both the full
    team name and an abbreviated alias, and occasionally posts moved lines too.
    For moneyline/spreads (side_by_team=True) keep one row per canonical team;
    for totals keep one row per side (Over/Under). Only the main line for the
    game is kept, preferring the line closest to consensus over a stale moved one.

    canonicalizer maps a provider team label to its canonical name; defaults to
    the NFL canonicalizer for football week slates (college boards pass their
    own school canonicalizer).
    """
    from collections import defaultdict
    from dataclasses import replace
    if canonicalizer is None:
        canonicalizer = _canonical_team

    # (event, canonical side) -> list of quotes; the main line for the game is the
    # one with the smallest |edge| (closest to consensus market).
    by_side: dict[tuple[str, str], list[BetQuote]] = defaultdict(list)
    for q in quotes:
        team = canonicalizer(q.selection) if side_by_team else (q.selection or "").strip().casefold()
        if side_by_team and not team:
            continue
        by_side[(q.event_id, team)].append(q)

    result: list[BetQuote] = []
    # The main line is a property of the market, not of each side: home and away
    # must be quoted on the same line or the pair is incoherent. Resolve it once
    # per event (the flagged line with the most sides behind it), then hold every
    # side of that event to it.
    event_line: dict[str, float] = {}
    flagged_sides: dict[tuple[str, float], set[str]] = {}
    for q in quotes:
        if not getattr(q, "is_main_line", False) or q.line is None:
            continue
        key = (q.event_id, round(float(q.line), 2))
        flagged_sides.setdefault(key, set()).add(
            (canonicalizer(q.selection) if side_by_team else (q.selection or "").casefold()))
    for (evt, line), sides in flagged_sides.items():
        if evt not in event_line or len(sides) > len(flagged_sides.get(
                (evt, event_line[evt]), set())):
            event_line[evt] = line

    for (evt, side), group in by_side.items():
        want = event_line.get(evt)
        on_line = [x for x in group
                   if want is not None and x.line is not None
                   and round(float(x.line), 2) == want]
        if on_line:
            pool = on_line
        else:
            # No flagged row for this side: fall back to the flagged main line's
            # mirror for spreads, else closest-to-consensus on any line.
            pool = [x for x in group if x.is_main_line] or group
        pool.sort(key=lambda x: (abs(x.edge_pct), x.best_odds), reverse=False)
        chosen = pool[0]  # main line
        # Surface the canonical team name instead of the provider alias.
        canonical = canonicalizer(side) if side_by_team else side
        if side_by_team and canonical and canonical.casefold() != chosen.selection.casefold():
            # dataclasses.replace (not a positional rebuild) so sport/league and
            # any future fields survive the rename.
            chosen = replace(chosen, selection=canonical)
        result.append(chosen)
    return result


def _last_good_slate(cache_key: str) -> dict | None:
    """Best cached slate for a date regardless of age.

    Used as a fallback when SharpAPI is rate-limited or returns no main market
    lines for a day that previously carried them. Events can exist while the
    best-odds fetch is throttled (silently returning no quotes); preferring the
    last known lines keeps the board populated instead of blanking that day.
    """
    with _disk_lock:
        disk = _load_disk_cache()
    entry = disk.get(cache_key)
    if entry is None:
        return None
    _, val = entry
    if not isinstance(val, dict):
        return None
    if val.get("moneylines") or val.get("spreads") or val.get("totals"):
        return val
    return None


def fetch_edge_slate(date: str) -> dict:
    """Fetch best moneyline/spread/total odds for a date across books."""
    cache_key = f"slate_{date}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        events = fetch_nfl_events(date)
        labels = {e.event_id: e.label() for e in events}
        event_ids = [e.event_id for e in events]
        quotes = fetch_best_odds(event_ids, ",".join(NFL_GAME_MARKETS), labels) if event_ids else []

        props = fetch_player_props(labels)
    except OddsError:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            _cache_set(cache_key, stale)
            return stale
        raise

    moneylines = _dedupe_side_quotes([q for q in quotes if q.market == "moneyline"], side_by_team=True)
    spreads = _dedupe_side_quotes([q for q in quotes if q.market == "point_spread"], side_by_team=True)
    totals = _dedupe_side_quotes([q for q in quotes if q.market == "total_points"], side_by_team=False)

    # Totals: show only the best side per game (the stronger edge), never both
    # Over and Under — betting both sides of one line is a coin flip, not a bet.
    totals_by_game: dict[str, BetQuote] = {}
    for q in totals:
        cur = totals_by_game.get(q.event_id)
        if cur is None or q.edge_pct > cur.edge_pct:
            totals_by_game[q.event_id] = q
    totals = list(totals_by_game.values())

    value_spots = build_value_spots(moneylines, spreads, totals)

    slate = {
        "events": events,
        "moneylines": moneylines,
        "spreads": spreads,
        "totals": totals,
        "props": props,
        "value_spots": value_spots,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    # Only cache slates that actually carried lines. Events can exist even if
    # the best-odds fetch was rate-limited (silently returning no quotes), and
    # caching that would poison the hot path for the TTL window.
    if moneylines or spreads or totals:
        _cache_set(cache_key, slate)
    else:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            return stale
    # Seed baselines from the same deduped main-line rows the pages show —
    # raw quotes include alt lines and mirrored perspectives, which seeded
    # inconsistent references and made normal lines look like steam moves.
    record_line_baselines(moneylines + spreads + totals)
    record_line_history(moneylines + spreads + totals)
    log_slate_snapshot(slate, "fetch")
    try:
        import journal
        journal.record_slate(slate, date)
    except Exception:
        pass
    return slate


def build_value_spots(moneylines: list[BetQuote], spreads: list[BetQuote],
                      totals: list[BetQuote]) -> list[BetQuote]:
    """One value row per market per game.

    Never show both sides of the same bet (e.g. Over AND Under on one total) —
    that's a coin flip, not a value signal. For totals keep only the stronger
    side; for moneylines and spreads each side is its own fair price, but only
    if it independently beats consensus.
    """
    def _best_side(quotes: list[BetQuote]) -> list[BetQuote]:
        by_key: dict[tuple, BetQuote] = {}
        for q in quotes:
            key = (q.event_id, q.market, q.selection.casefold())
            cur = by_key.get(key)
            if cur is None or abs(q.edge_pct) > abs(cur.edge_pct):
                by_key[key] = q
        return list(by_key.values())

    total_side_keys: dict[tuple, BetQuote] = {}
    for q in totals:
        line_key = (q.event_id, q.line)
        cur = total_side_keys.get(line_key)
        if not q.is_value:
            continue
        if cur is None or q.edge_pct > cur.edge_pct:
            total_side_keys[line_key] = q

    game_quotes = _best_side(moneylines) + _best_side(spreads) + list(total_side_keys.values())
    return sorted(
        [q for q in game_quotes if q.is_value],
        key=lambda q: q.edge_pct, reverse=True,
    )


def _fetch_league_best_odds(sport: str, league: str,
                            markets: str = ",".join(COLLEGE_GAME_MARKETS),
                            max_rows: int = 900) -> list[BetQuote]:
    """Page the full best-odds board for a league (no event filter).

    The NFL path pre-filters by event ids from the events feed; college leagues
    have no canonical team map to validate against, so page every row SharpAPI
    returns for the league's main markets and keep the ones that carry lines.
    Game labels come straight from each row's event_name.
    """
    cache_key = f"league_odds_{league}_{markets}"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    quotes: list[BetQuote] = []
    offset = 0
    while True:
        payload = _request_json(session, SHARPAPI_BEST_ODDS_URL, {
            "sport": sport, "league": league, "market": markets,
            "limit": 200, "offset": offset,
        }, timeout=25)
        page = _parse_best_odds_rows(payload, {})
        quotes.extend(page)
        pagination = payload.get("pagination") or {}
        next_offset = pagination.get("next_offset")
        if not pagination.get("has_more") or len(quotes) >= max_rows or not next_offset:
            break
        offset = int(next_offset)
        time.sleep(0.15)

    quotes = quotes[:max_rows]
    _cache_set(cache_key, quotes)
    return quotes


def fetch_sport_slate(tag: str, include_all: bool = False) -> dict:
    """Best moneyline/spread/total odds for any registry league (no props yet).

    Same board shape as fetch_edge_slate (moneylines/spreads/totals/value_spots)
    so the bets-board markup and slip drawer work unchanged. Totals keep only the
    stronger side per game. Sport-specific market types (run_line, puck_line,
    total_goals, total_runs) are aliased onto moneyline/point_spread/total_points
    and every quote is stamped with its sport/league for the cross-sport board.

    CFB scopes to Power-4 + Group-of-5 games unless include_all=True; other
    leagues stay unscoped.
    """
    cfg = SPORTS.get(tag)
    if cfg is None:
        raise OddsError(f"Unknown sport tag: {tag}")

    cache_key = f"sport_slate_{tag}_all" if include_all else f"sport_slate_{tag}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        quotes = _fetch_league_best_odds(
            cfg["sport"], cfg["league"],
            markets=",".join(cfg.get("markets") or GAME_MARKETS))
    except OddsError:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            _cache_set(cache_key, stale)
            return stale
        raise

    # Normalize provider-specific market types onto the canonical grouping and
    # stamp sport/league on every quote (survives dataclasses.replace; powers
    # the later cross-sport daily best-bets board).
    from dataclasses import replace
    aliases = cfg.get("aliases") or {}
    quotes = [
        replace(q, market=aliases.get(q.market, q.market),
                sport=cfg["sport"], league=cfg["league"])
        for q in quotes
    ]

    # Unify game labels so the same matchup shows identically across every
    # market/row (providers vary the team naming per market).
    if cfg.get("college"):
        game_labels: dict[str, str] = {}
        for q in quotes:
            if q.event_id and q.event_id not in game_labels:
                game_labels[q.event_id] = _canonical_game(q.game)
        canoned: list[BetQuote] = []
        for q in quotes:
            canon_label = game_labels.get(q.event_id)
            canoned.append(replace(q, game=canon_label) if canon_label else q)
        quotes = canoned

    # Scope the board: only keep games where BOTH teams are in an allowed
    # conference. cfb uses the P4+G5 map; other leagues carry no conference map.
    conferences: dict[str, str] = {}
    if cfg.get("scoped") and not include_all:
        for q in quotes:
            if q.game and q.game not in conferences:
                conferences[q.game] = _game_grouping(q.game)
        quotes = [q for q in quotes if conferences.get(q.game)]

    # Per-team rows collapse across the provider's label variants (full name,
    # mascot, "st"/abbr, ranked prefix) so each school is one row per game.
    # Unmapped schools fall back to their raw label, so nothing disappears.
    board_canon = _canonical_college_team if cfg.get("college") else (_PRO_CANON.get(tag) or _pro_team)
    moneylines = _dedupe_side_quotes([q for q in quotes if q.market == "moneyline"], side_by_team=True, canonicalizer=board_canon)
    spreads = _dedupe_side_quotes([q for q in quotes if q.market == "point_spread"], side_by_team=True, canonicalizer=board_canon)
    totals = _dedupe_side_quotes([q for q in quotes if q.market == "total_points"], side_by_team=False)

    totals_by_game: dict[str, BetQuote] = {}
    for q in totals:
        cur = totals_by_game.get(q.event_id)
        if cur is None or q.edge_pts() > cur.edge_pts():
            totals_by_game[q.event_id] = q
    totals = list(totals_by_game.values())

    # Value spots on true edge (points of implied probability), restricted to
    # lines in the real-play band (-500..+300) so the board never "suggests"
    # fantasy longshots like a +8000 moneyline underdog.
    playables = [q for q in (moneylines + spreads) if q.is_playable]
    playables += [q for q in totals if q.is_playable]
    playables.sort(key=lambda q: q.edge_pts(), reverse=True)
    value_spots = playables

    # Snapshot every side *before* any per-game collapsing below. The board UI
    # wants one best row per game, but the analytics engine needs complete
    # markets — all three soccer outcomes, both sides of a spread — because a
    # lone side has nothing to devig against.
    all_sides = list(moneylines) + list(spreads) + list(totals)

    if cfg.get("collapse_bets"):
        # Moneyline best bets: one row per game — the single best-value side
        # only (real play), ranked by edge. No per-team double rows.
        best_ml_by_game: dict[str, BetQuote] = {}
        for q in moneylines:
            if not q.is_playable:
                continue
            cur = best_ml_by_game.get(q.game)
            if cur is None or q.edge_pts() > cur.edge_pts():
                best_ml_by_game[q.game] = q
        moneylines = sorted(best_ml_by_game.values(), key=lambda q: q.edge_pts(), reverse=True)

        # Spread best bets: one row per game — the single best-value side
        # (real play), ranked by edge. No per-team double rows.
        best_sp_by_game: dict[str, BetQuote] = {}
        for q in spreads:
            if not q.is_playable:
                continue
            cur = best_sp_by_game.get(q.game)
            if cur is None or q.edge_pts() > cur.edge_pts():
                best_sp_by_game[q.game] = q
        spreads = sorted(best_sp_by_game.values(), key=lambda q: q.edge_pts(), reverse=True)

        # Value Plays: one row per game (best edge play only) so a matchup
        # never repeats for ML + Spread + Total.
        best_play_by_game: dict[str, BetQuote] = {}
        for q in value_spots:
            cur = best_play_by_game.get(q.game)
            if cur is None or q.edge_pts() > cur.edge_pts():
                best_play_by_game[q.game] = q
        value_spots = sorted(best_play_by_game.values(), key=lambda q: q.edge_pts(), reverse=True)[:12]

    game_rows: dict[str, str] = {}
    for q in moneylines + spreads + totals:
        game_rows[q.event_id] = q.game
    events = [{"event_id": eid, "label": label}
              for eid, label in sorted(game_rows.items(), key=lambda kv: (kv[1] or "").casefold())]

    slate = {
        "events": events,
        "moneylines": moneylines,
        "spreads": spreads,
        "totals": totals,
        "all_sides": all_sides,
        "props": [],
        "value_spots": value_spots,
        "conferences": conferences,
        "scoped": bool(tag == "cfb" and not include_all),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    # Only cache slates that carried lines; a pre-season league legitimately
    # returns zero rows until its season starts, and caching that would blank
    # the board for the whole TTL window.
    if moneylines or spreads or totals:
        _cache_set(cache_key, slate)
    else:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            return stale
    return slate


def _fetch_league_props_quotes(sport: str, league: str, markets: str,
                               max_rows: int = 1200, request_delay: float = 0.25) -> list[BetQuote]:
    """Aggregate per-book player props into best-price BetQuotes.

    /odds/best strips player identity for some sports (baseball), so this pages
    the raw /odds endpoint (which carries player_name/stat_category) filtered to
    main lines, then groups by (event, player, market, side, line): best price =
    max decimal, consensus = mean decimal, edge = best vs consensus.
    """
    session = requests.Session()
    raw_rows: list[dict] = []
    offset = 0
    while True:
        payload = _request_json(session, f"{SHARPAPI_BASE_URL}/odds", {
            "sport": sport, "league": league, "market": markets,
            "is_main_line": "true", "limit": 200, "offset": offset,
        }, timeout=25)
        rows = payload.get("data") or []
        raw_rows.extend(rows)
        pagination = payload.get("pagination") or {}
        next_offset = pagination.get("next_offset")
        if not pagination.get("has_more") or len(raw_rows) >= max_rows or not next_offset:
            break
        offset = int(next_offset)
        time.sleep(request_delay)

    groups: dict[tuple, dict] = {}
    for row in raw_rows[:max_rows]:
        player = str(row.get("player_name") or "").strip()
        market = str(row.get("market_type") or "").strip().casefold()
        if not player or not market:
            continue
        try:
            american = int(row.get("odds_american"))
        except (TypeError, ValueError):
            continue
        if not american:
            continue
        decimal = american_to_decimal(american)
        line = None
        try:
            if row.get("line") is not None and row.get("line") != "":
                line = float(row.get("line"))
        except (TypeError, ValueError):
            line = None
        if line is None:
            continue
        event_id = str(row.get("event_id") or "").strip()
        side = str(row.get("selection_type") or row.get("selection") or "").strip()
        key = (event_id, player.casefold(), market, str(side).casefold(), float(line))
        g = groups.setdefault(key, {
            "event_id": event_id,
            "game": str(row.get("event_name") or "").strip(),
            "player": player, "market": market, "side": side, "line": line,
            "best": 0.0, "best_am": 0, "best_book": "", "book_odds": {}, "decimals": [],
        })
        if decimal > g["best"]:
            g["best"] = decimal
            g["best_am"] = american
            g["best_book"] = str(row.get("sportsbook") or "").strip()
        book_name = str(row.get("sportsbook") or "").strip()
        if book_name:
            g["book_odds"][book_name] = american
        if not g["game"]:
            g["game"] = str(row.get("event_name") or "").strip()
        g["decimals"].append(decimal)

    quotes: list[BetQuote] = []
    for g in groups.values():
        if not g["decimals"]:
            continue
        consensus = sum(g["decimals"]) / len(g["decimals"])
        edge = (g["best"] / consensus - 1.0) * 100.0 if consensus else 0.0
        cons_am = decimal_to_american(consensus) if consensus else g["best_am"]
        quotes.append(BetQuote(
            event_id=g["event_id"], game=g["game"], selection=g["side"] or "Over",
            market=g["market"], best_odds=g["best_am"], consensus_odds=cons_am,
            best_book=g["best_book"], edge_pct=round(edge, 2), line=g["line"],
            player_name=g["player"], book_odds=(g["book_odds"] or None),
            book_edges=None, sport=sport, league=league))
    return quotes


def _sport_game_labels(tag: str) -> dict[str, str]:
    """event_id -> matchup label for a registry league, from the cached slate.

    The raw player-prop feed skips event names for some sports (baseball), so
    props fall back to the main-market slate's canonical labels. Only consults
    the in-memory/disk slate cache — never triggers a fresh fetch.
    """
    slate = _cached(f"sport_slate_{tag}") or _cached(f"sport_slate_{tag}_all")
    if not slate:
        return {}
    return {e.get("event_id"): e.get("label") for e in (slate.get("events") or [])
            if e.get("event_id") and e.get("label")}


def fetch_sport_props(tag: str) -> list:
    """Player-prop best odds for a registry league (raw + edge, no model layer).

    Consumes the league's configured prop market types (e.g. MLB home runs,
    hits, total bases, RBIs) and stamps sport/league on every quote; rows are
    grouped per (event, player, market, side, line) with the best price vs
    consensus edge. Returns [] for leagues with no prop_markets configured.
    """
    cfg = SPORTS.get(tag)
    if cfg is None:
        raise OddsError(f"Unknown sport tag: {tag}")
    prop_markets = cfg.get("prop_markets") or []
    if not prop_markets:
        return []

    cache_key = f"sport_props_{tag}"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    try:
        props = _fetch_league_props_quotes(
            cfg["sport"], cfg["league"], ",".join(prop_markets))
    except OddsError:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            _cache_set(cache_key, stale)
            return list(stale)
        raise

    # The raw prop feed can omit event names (baseball). Backfill each quote's
    # game from the slate's canonical matchup labels so rows stay aligned and
    # the prop is findable by game.
    game_labels = _sport_game_labels(tag)
    if game_labels:
        from dataclasses import replace
        props = [
            replace(q, game=game_labels.get(q.event_id, q.game)) if not getattr(q, "game", "") else q
            for q in props
        ]

    props.sort(key=lambda q: (q.market, (q.player_name or "").casefold(), q.line or 0))
    if props:
        _cache_set(cache_key, props)
    else:
        stale = _last_good_slate(cache_key)
        if stale is not None:
            _cache_set(cache_key, stale)
            return list(stale)
    return props


def _load_line_hist() -> dict:
    """Main-line movement series, read from disk directly (ignores the TTL —
    history should survive idle gaps, unlike the ephemeral line snapshot)."""
    with _disk_lock:
        disk = _load_disk_cache()
    entry = disk.get("line_hist")
    if entry is None:
        return {}
    _, val = entry
    return val if isinstance(val, dict) else {}


def _line_steam_key(q: BetQuote) -> tuple:
    """Stable identity for one main-line series: (event, market, side).

    Deliberately excludes the price and the line itself — those are exactly what
    the series tracks. Side is casefolded so a label casing change between
    fetches cannot fork one series into two.
    """
    return (q.event_id, q.market, (q.selection or "").strip().casefold())


def record_line_history(quotes: list[BetQuote]) -> None:
    """Append a snapshot of each main spread/total line to a rolling series.

    SharpAPI serves a snapshot, not an opening-line endpoint, so the closest
    honest "line move" chart samples the main line on every fresh fetch and
    diffs against the first-seen value (the baseline). Unchanged samples are
    skipped to avoid bloat and the series is capped per market; the disk cache
    persists it across restarts so overnight/weekly moves are still charted.
    """
    hist = _load_line_hist()
    changed = False
    for q in quotes:
        if q.line is None or q.market not in ("point_spread", "total_points"):
            continue
        key = _line_steam_key(q)
        rec = hist.get(key)
        if rec is None:
            hist[key] = {
                "game": q.game or "",
                "market": q.market,
                "side": (q.selection or ""),
                "pts": [{"t": time.time(), "line": q.line}],
            }
            changed = True
            continue
        if rec["pts"][-1]["line"] == q.line:
            continue
        rec["pts"].append({"t": time.time(), "line": q.line})
        rec["pts"] = rec["pts"][-MAX_LINE_HISTORY:]
        hist[key] = rec
        changed = True
    if changed:
        _cache_set("line_hist", hist)


def line_series() -> list[dict]:
    """Main-line movement series for the steam charts, movers first."""
    hist = _load_line_hist()
    out = []
    for key, rec in hist.items():
        pts = rec.get("pts") or []
        if not pts:
            continue
        out.append({
            "key": key,
            "game": rec.get("game", ""),
            "market": rec.get("market", ""),
            "side": rec.get("side", ""),
            "base": pts[0]["line"],
            "pts": [{"t": p["t"], "line": p["line"]} for p in pts],
        })
    out.sort(
        key=lambda r: (len(r["pts"]) > 1, abs(r["pts"][-1]["line"] - r["base"])),
        reverse=True,
    )
    return out


def record_line_baselines(quotes: list[BetQuote]) -> None:
    """Persist the first-seen main line per (event, market, side) as a baseline.

    SharpAPI only serves a current snapshot — there is no opening line — so the
    first fetch seeds the reference and later fetches reveal movement against it.
    Baselines survive restarts via the disk cache, so a line that moved overnight
    (or week-to-week on a Thursday/Sunday slate) is still detected.
    """
    snap = _cached("line_snap") or {}
    changed = False
    for q in quotes:
        if q.line is None or q.market not in ("point_spread", "total_points"):
            continue
        k = _line_steam_key(q)
        if k not in snap:
            # Store the magnitude: the provider flips the sign convention on
            # spreads between fetches (the same side shows -3.5 one hour and
            # +3.5 the next), which produced phantom moves of 7 points.
            snap[k] = {"line": abs(q.line), "ts": time.time()}
            changed = True
    if changed:
        _cache_set("line_snap", snap)


def line_moves(quotes: list[BetQuote], min_delta: float = 1.5) -> dict[tuple, dict]:
    """Movement vs each line's first-seen baseline (steam-move signal).

    Compares line *magnitudes* — a sign flip with the same magnitude is a
    provider perspective change, not a real move; a magnitude change of
    >= min_delta is genuine movement on either side of zero.
    """
    snap = _cached("line_snap") or {}
    out: dict[tuple, dict] = {}
    for q in quotes:
        if q.line is None or q.market not in ("point_spread", "total_points"):
            continue
        k = _line_steam_key(q)
        base = snap.get(k)
        if not base:
            continue
        try:
            base_line = abs(float(base["line"]))
        except (TypeError, ValueError):
            continue
        cur_line = abs(q.line)
        delta = abs(cur_line - base_line)
        if delta >= min_delta:
            out[k] = {"from": round(base_line, 1), "to": round(cur_line, 1),
                      "delta": round(delta, 1)}
    return out


def steam(q, moves: dict[tuple, dict]):
    """Steam-move info for a quote against the moves map, or None."""
    m = moves.get(_line_steam_key(q))
    if not m:
        return None
    return {"from": m["from"], "to": m["to"], "delta": m["delta"],
            "up": m["to"] > m["from"]}


# ---------------------------------------------------------------------------
# Snapshot history — a compact KPI time series so the dashboard can draw real
# trend sparklines (edge counts, avg edge, board size) as snapshots accrue.
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
os.makedirs(_SNAPSHOT_DIR, exist_ok=True)
_SNAPSHOT_FILE = os.path.join(_SNAPSHOT_DIR, "nfl_odds_history.jsonl")
_snapshot_lock = threading.Lock()
_last_snapshot = {"ts": 0.0, "sig": ""}


def _quote_list(slate: dict) -> list:
    out: list = []
    for key in ("moneylines", "spreads", "totals", "props"):
        out.extend(slate.get(key) or [])
    return out


def log_slate_snapshot(slate: dict, source: str = "fetch") -> None:
    """Append one compact record per observed slate so trends build over time.

    Cheap (single json line) and idempotent — identical observations within 60s
    are skipped, and any I/O failure is swallowed so logging never breaks the
    page that triggered it.
    """
    global _last_snapshot
    if not isinstance(slate, dict):
        return
    try:
        quotes = _quote_list(slate)
        value = [q for q in quotes if (getattr(q, "edge_pct", 0) or 0) >= 1.5]
        top = 0.0
        if value:
            top = round(max(q.edge_pct for q in value), 1)
        rec = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).strftime("%H:%M"),
            "games": len(slate.get("events") or []),
            "pieces": len(quotes),
            "value": len(value),
            "avg_edge": round(sum(q.edge_pct for q in value) / len(value), 1) if value else 0.0,
            "top_edge": top,
            "source": source,
        }
        sig = f"{rec['games']}:{rec['value']}:{rec['top_edge']}:{rec['avg_edge']}"
        now = rec["ts"]
        if sig == _last_snapshot["sig"] and now - _last_snapshot["ts"] < 60:
            return
        _last_snapshot = {"ts": now, "sig": sig}
        with _snapshot_lock:
            with open(_SNAPSHOT_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def snapshot_history(max_age_s: float = 36 * 3600) -> list[dict]:
    """Read back snapshot records (oldest→newest) within max_age_s seconds."""
    try:
        out: list[dict] = []
        with open(_SNAPSHOT_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if time.time() - rec.get("ts", 0) <= max_age_s:
                        out.append(rec)
                except (ValueError, TypeError):
                    continue
        out.sort(key=lambda r: r.get("ts", 0))
        return out
    except Exception:
        return []


_TEAM_SLUGS: dict[str, str] = {}


def _team_from_slug(slug: str) -> str | None:
    """Map a lowercase event-id token (e.g. '49ers', 'rams') to the canonical
    full team name."""
    if not _TEAM_SLUGS:
        for name in NFL_TEAM_ABBREVIATIONS:
            mascot = name.split()[-1].casefold().replace(" ", "")
            _TEAM_SLUGS[mascot] = name
            _TEAM_SLUGS[name.casefold().replace(" ", "")] = name
    key = slug.casefold().replace(" ", "")
    return _TEAM_SLUGS.get(key)


def _game_label_from_event_id(event_id: str) -> str:
    """Resolve a game label from a SharpAPI event id ('nfl_49ers_rams_...').

    The props feed carries events across the week (TNF/SNF), so the slate's label
    map misses them. Parse the two team slug tokens out of the event id instead.
    """
    if not event_id:
        return ""
    parts = event_id.split("_")
    if len(parts) < 4 or not parts[0] == "nfl":
        return ""
    away = _team_from_slug(parts[1])
    home = _team_from_slug(parts[2])
    if away and home:
        return f"{away} @ {home}"
    return ""


def _human_prop_label(stat_category: str, market: str) -> str:
    if market == "player_touchdowns":
        return "Anytime Touchdown"
    if market in PROP_LABELS:
        return PROP_LABELS[market]
    if stat_category:
        return stat_category.replace("_", " ").title()
    return market.replace("player_", "").replace("_", " ").title()


def _group_props(raw_rows: list[dict], labels: dict[str, str]) -> list[PlayerProp]:
    """Aggregate raw per-book prop odds into best-price rows with edge, keyed by
    (player, market_type, selection_type, line) per SharpAPI canonical spec."""
    groups: dict[tuple, dict] = {}
    for row in raw_rows:
        player = str(row.get("player_name") or "").strip()
        market = str(row.get("market_type") or "").strip().casefold()
        if not player or not market:
            continue
        try:
            american = int(row.get("odds_american"))
        except (TypeError, ValueError):
            continue
        if not american:
            continue
        decimal = american_to_decimal(american)
        line = None
        try:
            if row.get("line") is not None and row.get("line") != "":
                line = float(row.get("line"))
        except (TypeError, ValueError):
            line = None
        if line is None:
            # Skip degenerate bins (yes/no combo rows, "2+ rec" longshots) that
            # produce meaningless edges — the canonical prop has a threshold line.
            continue
        side = str(row.get("selection_type") or row.get("selection") or "").strip()
        key = (player.casefold(), market, str(side).casefold(), line)
        group = groups.setdefault(key, {
            "event_id": str(row.get("event_id") or "").strip(),
            "game": "",
            "player": player,
            "market": market,
            "label": _human_prop_label(str(row.get("stat_category") or ""), market),
            "stat_category": str(row.get("stat_category") or "").strip(),
            "side": side,
            "line": line,
            "best_decimal": 0.0,
            "best_american": 0,
            "best_book": "",
            "decimals": [],
            "books": [],
            "book_odds": {},
            "count": 0,
        })
        decimal_sum = group["best_decimal"]
        if decimal > group["best_decimal"]:
            group["best_decimal"] = decimal
            group["best_american"] = american
            group["best_book"] = str(row.get("sportsbook") or "").strip()
        book_name = str(row.get("sportsbook") or "").strip()
        if book_name:
            group["book_odds"][book_name] = american
        if not group["game"]:
            group["game"] = (
                labels.get(group["event_id"])
                or _game_label_from_event_id(group["event_id"])
                or str(row.get("event_name") or "").strip()
            )
        group["decimals"].append(decimal)
        if row.get("sportsbook"):
            group["books"].append(str(row.get("sportsbook")))
        group["count"] += 1

    props: list[PlayerProp] = []
    for g in groups.values():
        if not g["decimals"]:
            continue
        consensus = sum(g["decimals"]) / len(g["decimals"]) if g["decimals"] else 0.0
        edge = (g["best_decimal"] / consensus - 1.0) * 100.0 if consensus else 0.0
        cons_am = decimal_to_american(consensus) if consensus else g["best_american"]
        props.append(PlayerProp(
            event_id=g["event_id"], game=g["game"], player_name=g["player"],
            market=g["market"], label=g["label"], stat_category=g["stat_category"],
            side=g["side"], line=g["line"],
            best_odds=g["best_american"], consensus_odds=cons_am,
            best_book=g["best_book"], edge_pct=round(edge, 2),
            book_odds=(g["book_odds"] or None),
        ))

    # Collapse over/under duplicates: keep a single row per (player, market,
    # line), preferring the higher-edge side so the board isn't full of mirrors.
    by_line: dict[tuple, PlayerProp] = {}
    for p in props:
        key = (p.player_name.casefold(), p.market.casefold(),
               None if p.line is None else round(float(p.line), 4))
        cur = by_line.get(key)
        if cur is None or p.edge_pct > cur.edge_pct:
            by_line[key] = p
    props = sorted(by_line.values(), key=lambda p: p.edge_pct, reverse=True)
    return props


def fetch_player_props(labels: dict[str, str], max_rows: int = 800,
                       request_delay: float = 0.25) -> list[PlayerProp]:
    """Fetch NFL player props across books and aggregate best-price + edge.

    Uses the raw /odds endpoint (which carries player identity) filtered to the
    `props` alias. Props are sparse until books release the full weekly slate
    (typically Tue-Wed before games); returns [] when none are posted yet."""
    cache_key = "props_all"
    cached = _cached(cache_key)
    if cached is not None:
        return list(cached)

    session = requests.Session()
    raw_rows: list[dict] = []
    offset = 0
    page_error = False
    while True:
        try:
            payload = _request_json(session, f"{SHARPAPI_BASE_URL}/odds", {
                "sport": "football", "league": "nfl", "market": "props",
                "is_main_line": "true", "limit": 200, "offset": offset,
            }, timeout=25)
        except OddsError:
            # Don't cache a partial board: surface no-props so the next request
            # retries the full pagination instead of freezing a truncated set.
            if raw_rows:
                page_error = True
            break
        raw_rows.extend(payload.get("data") or [])
        pagination = payload.get("pagination") or {}
        next_offset = pagination.get("next_offset")
        if not pagination.get("has_more") or len(raw_rows) >= max_rows or not next_offset:
            break
        offset = int(next_offset)
        if request_delay > 0:
            time.sleep(request_delay)

    props = _group_props(raw_rows[:max_rows], labels)
    if raw_rows and not page_error:
        _cache_set(cache_key, props)
    return props

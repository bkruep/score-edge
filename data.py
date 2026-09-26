from __future__ import annotations
import io, os, time, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Shared cache (in-memory + on-disk so restarts/expiry don't re-download)
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 1800  # 30 min in memory

_DISK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
_DISK_TTL = 86400 * 7  # 7 days on disk
os.makedirs(_DISK_DIR, exist_ok=True)


def _disk_path(key: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in key)
    return os.path.join(_DISK_DIR, f"{safe}.json")


def _get_disk(key: str):
    path = _disk_path(key)
    try:
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > _DISK_TTL:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _put_disk(key: str, val):
    try:
        with open(_disk_path(key), "w", encoding="utf-8") as f:
            json.dump(_serializable(val), f)
    except Exception:
        pass


def _serializable(val):
    if isinstance(val, pd.DataFrame):
        return {"__df__": True, "records": val.to_dict("records"), "columns": list(val.columns)}
    if isinstance(val, (list, tuple)):
        return [_serializable(v) for v in val]
    if isinstance(val, dict):
        return {k: _serializable(v) for k, v in val.items()}
    try:
        json.dumps(val)
        return val
    except (TypeError, ValueError):
        return str(val)


def _materialize(val):
    if isinstance(val, dict) and val.get("__df__"):
        try:
            return pd.DataFrame(val["records"], columns=val.get("columns"))
        except Exception:
            pass
    return val


def _get_cached(key: str):
    if key in _CACHE:
        ts, val = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return val
    # Fall back to disk (load once per process to avoid repeated disk reads)
    if key not in _CACHE:
        disk = _get_disk(key)
        if disk is not None:
            val = _materialize(disk)
            _CACHE[key] = (time.time(), val)
            return val
    return None


def _set_cached(key: str, val):
    _CACHE[key] = (time.time(), val)
    _put_disk(key, val)

# ---------------------------------------------------------------------------
# Odds API (shared across sports)
# ---------------------------------------------------------------------------

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE = "https://api.the-odds-api.com/v4"

def fetch_odds(sport: str = "americanfootball_nfl", market: str = "h2h") -> list[dict]:
    if not ODDS_API_KEY:
        return []
    key = f"odds_{sport}_{market}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        url = f"{ODDS_BASE}/sports/{sport}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": market, "oddsFormat": "american"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        _set_cached(key, data)
        return data
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Weather (shared across sports)
# ---------------------------------------------------------------------------

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_weather(lat: float, lon: float) -> dict:
    key = f"weather_{lat}_{lon}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "America/New_York", "forecast_days": 7
        }
        r = requests.get(WEATHER_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        _set_cached(key, data)
        return data
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# NFL — nfldata.org (free, no key) + Sleeper (free, no key)
# ---------------------------------------------------------------------------

NFLDATA_BASE = "https://api.nfldata.org/v1"
SLEEPER_BASE = "https://api.sleeper.app/v1"

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF", "DST"}


def _get_retry(url: str, params: dict, attempts: int = 3, timeout: int = 12) -> requests.Response | None:
    """GET with retry+backoff. nfldata.org is flaky (drops TLS connections and
    rate-limits), so retry transient failures before giving up. Returns None if
    every attempt fails, else the last non-exception response."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = r
        except Exception as e:  # TLS EOF, timeouts, connection resets
            last = e
        if i < attempts - 1:
            time.sleep(0.4 * (i + 1))  # small backoff
    return last


def _fetch_stats_page(season: int, offset: int, limit: int) -> list[dict]:
    url = f"{NFLDATA_BASE}/stats/season"
    params = {"season": season, "limit": limit, "offset": offset}
    r = _get_retry(url, params)
    if r is None or not isinstance(r, requests.Response) or r.status_code != 200:
        return []
    return r.json().get("data", [])


def load_player_stats(season: int = 2025) -> pd.DataFrame:
    key = f"nfl_stats_{season}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        # NOTE: nfldata.org drops TLS connections under concurrent requests
        # (SSL UNEXPECTED_EOF_WHILE_READING), so pages must be fetched serially.
        # The disk cache below makes repeat loads near-instant regardless.
        limit = 500
        all_rows: list[dict] = []
        offset = 0
        while True:
            page = _fetch_stats_page(season, offset, limit)
            all_rows.extend(page)
            if len(page) < limit:
                break
            offset += limit
            if offset > limit * 24:  # safety cap (~12k rows)
                break
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        _set_cached(key, df)
        return df
    except Exception:
        return pd.DataFrame()


def load_schedule(season: int = 2025) -> pd.DataFrame:
    key = f"nfl_schedule_{season}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        def fetch_week(week: int) -> list[dict]:
            url = f"{NFLDATA_BASE}/games"
            params = {"season": season, "week": week}
            r = _get_retry(url, params)
            if r is not None and isinstance(r, requests.Response) and r.status_code == 200:
                return r.json().get("data", [])
            return []

        # NOTE: fetched serially — nfldata.org drops concurrent TLS connections
        # (SSL UNEXPECTED_EOF_WHILE_READING). Disk cache keeps repeat loads fast.
        all_games = []
        for week in range(1, 19):
            all_games.extend(fetch_week(week))
        if not all_games:
            return pd.DataFrame()
        df = pd.DataFrame(all_games)
        _set_cached(key, df)
        return df
    except Exception:
        return pd.DataFrame()

def load_rosters() -> dict:
    key = "sleeper_rosters"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        url = f"{SLEEPER_BASE}/players/nfl"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        players = r.json()
        _set_cached(key, players)
        return players
    except Exception:
        return {}

LEAGUELOGS_BASE = "https://developer.leaguelogs.com/v1"


def load_league_logs_market(profile: str = "redraft-1qb-12t-ppr1") -> list[dict]:
    """Real 2026-27 Sleeper redraft ADP via the LeagueLogs developer API.

    Returns an ordered list of player records keyed by `sleeperPlayerId` with
    `overallRank`, `positionRank`, and a market `value` (100 = consensus top).
    Covers QB/RB/WR/TE only. This is a free no-key API; attribution to LeagueLogs
    is expected and provided via the returned meta/by the UI footer.
    """
    key = f"leaguelogs_{profile}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        url = f"{LEAGUELOGS_BASE}/market/{profile}"
        r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json().get("data", [])
        _set_cached(key, data)
        return data
    except Exception:
        return []


FFC_BASE = "https://fantasyfootballcalculator.com/api/v1"

def load_adp(scoring: str = "ppr", year: int | None = None) -> pd.DataFrame:
    if year is None:
        from datetime import datetime as _dt
        now = _dt.now()
        year = now.year if now.month >= 8 else now.year - 1
    key = f"ffc_adp_{scoring}_{year}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        url = f"{FFC_BASE}/adp/{scoring}"
        params = {"teams": 12, "position": "all", "year": year}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        players = data.get("players", [])
        if not players:
            return pd.DataFrame()
        df = pd.DataFrame(players)
        _set_cached(key, df)
        return df
    except Exception:
        return pd.DataFrame()

NFL_STADIUMS = {
    "ARI": {"name": "State Farm Stadium", "lat": 33.5276, "lon": -112.2626, "roof": "dome"},
    "ATL": {"name": "Mercedes-Benz Stadium", "lat": 33.7554, "lon": -84.401, "roof": "dome"},
    "BAL": {"name": "M&T Bank Stadium", "lat": 39.278, "lon": -76.6227, "roof": "open"},
    "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.787, "roof": "open"},
    "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8528, "roof": "open"},
    "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167, "roof": "open"},
    "CIN": {"name": "Paycor Stadium", "lat": 39.0955, "lon": -84.5161, "roof": "open"},
    "CLE": {"name": "Cleveland Browns Stadium", "lat": 41.5061, "lon": -81.6995, "roof": "open"},
    "DAL": {"name": "AT&T Stadium", "lat": 32.7473, "lon": -97.0945, "roof": "dome"},
    "DEN": {"name": "Empower Field", "lat": 39.7439, "lon": -105.02, "roof": "open"},
    "DET": {"name": "Ford Field", "lat": 42.34, "lon": -83.0456, "roof": "dome"},
    "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622, "roof": "open"},
    "HOU": {"name": "NRG Stadium", "lat": 29.6847, "lon": -95.4107, "roof": "dome"},
    "IND": {"name": "Lucas Oil Stadium", "lat": 39.7601, "lon": -86.1639, "roof": "dome"},
    "JAX": {"name": "EverBank Stadium", "lat": 30.3239, "lon": -81.6373, "roof": "open"},
    "KC": {"name": "Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839, "roof": "open"},
    "LA": {"name": "SoFi Stadium", "lat": 33.9534, "lon": -118.339, "roof": "dome"},
    "LAC": {"name": "SoFi Stadium", "lat": 33.9534, "lon": -118.339, "roof": "dome"},
    "LV": {"name": "Allegiant Stadium", "lat": 36.0908, "lon": -115.183, "roof": "dome"},
    "MIA": {"name": "Hard Rock Stadium", "lat": 25.958, "lon": -80.2389, "roof": "open"},
    "MIN": {"name": "U.S. Bank Stadium", "lat": 44.9736, "lon": -93.2575, "roof": "dome"},
    "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643, "roof": "open"},
    "NO": {"name": "Caesars Superdome", "lat": 29.9511, "lon": -90.0812, "roof": "dome"},
    "NYG": {"name": "MetLife Stadium", "lat": 40.8128, "lon": -74.0742, "roof": "open"},
    "NYJ": {"name": "MetLife Stadium", "lat": 40.8128, "lon": -74.0742, "roof": "open"},
    "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675, "roof": "open"},
    "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158, "roof": "open"},
    "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316, "roof": "open"},
    "SF": {"name": "Levi's Stadium", "lat": 37.4033, "lon": -121.9694, "roof": "open"},
    "TB": {"name": "Raymond James Stadium", "lat": 27.9759, "lon": -82.5033, "roof": "open"},
    "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713, "roof": "open"},
    "WAS": {"name": "Northwest Stadium", "lat": 38.9076, "lon": -76.8645, "roof": "open"},
}

# ---------------------------------------------------------------------------
# NBA — planned (nba_api / basketball-reference)
# ---------------------------------------------------------------------------

# def load_nba_player_stats(season: str = "2025-26") -> pd.DataFrame: ...
# def load_nba_schedule(season: str = "2025-26") -> pd.DataFrame: ...
# def load_nba_teams() -> pd.DataFrame: ...
# NBA_ARENAS = { ... }

# ---------------------------------------------------------------------------
# MLB — planned (pybaseball / statsapi)
# ---------------------------------------------------------------------------

# def load_mlb_player_stats(season: int = 2026) -> pd.DataFrame: ...
# def load_mlb_schedule(season: int = 2026) -> pd.DataFrame: ...
# def load_mlb_teams() -> pd.DataFrame: ...
# MLB_STADIUMS = { ... }

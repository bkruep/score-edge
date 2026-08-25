from __future__ import annotations
import io, os, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 1800

def _get_cached(key: str):
    if key in _CACHE:
        ts, val = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return val
    return None

def _set_cached(key: str, val):
    _CACHE[key] = (time.time(), val)

def _download_csv(url: str, key: str) -> pd.DataFrame:
    cached = _get_cached(key)
    if cached is not None:
        return cached
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        _set_cached(key, df)
        return df
    except Exception:
        return pd.DataFrame()

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
# NFL — nflverse (CC0) + The Odds API
# ---------------------------------------------------------------------------

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

def load_player_stats(season: int = 2024) -> pd.DataFrame:
    key = f"nfl_player_stats_{season}"
    url = f"{NFLVERSE_BASE}/player_stats/player_stats_{season}.csv"
    df = _download_csv(url, key)
    if not df.empty:
        return df
    for fallback in range(season - 1, 2019, -1):
        key2 = f"nfl_player_stats_{fallback}"
        url2 = f"{NFLVERSE_BASE}/player_stats/player_stats_{fallback}.csv"
        df2 = _download_csv(url2, key2)
        if not df2.empty:
            return df2
    return pd.DataFrame()

def load_schedule(season: int = 2024) -> pd.DataFrame:
    key = "nfl_schedule_all"
    url = f"{NFLVERSE_BASE}/schedules/games.csv"
    df = _download_csv(url, key)
    if not df.empty and "season" in df.columns:
        filtered = df[df["season"] == season]
        if not filtered.empty:
            return filtered
        latest = int(df["season"].max())
        return df[df["season"] == latest]
    return df

def load_teams() -> pd.DataFrame:
    key = "nfl_teams"
    url = f"{NFLVERSE_BASE}/teams/teams_colors_logos.csv"
    return _download_csv(url, key)

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

# def load_nba_player_stats(season: str = "2024-25") -> pd.DataFrame: ...
# def load_nba_schedule(season: str = "2024-25") -> pd.DataFrame: ...
# def load_nba_teams() -> pd.DataFrame: ...
# NBA_ARENAS = { ... }

# ---------------------------------------------------------------------------
# MLB — planned (pybaseball / statsapi)
# ---------------------------------------------------------------------------

# def load_mlb_player_stats(season: int = 2025) -> pd.DataFrame: ...
# def load_mlb_schedule(season: int = 2025) -> pd.DataFrame: ...
# def load_mlb_teams() -> pd.DataFrame: ...
# MLB_STADIUMS = { ... }

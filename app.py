from __future__ import annotations
import hashlib, os
from datetime import datetime
from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd

import data
import analytics

app = FastAPI(title="ScoreEdge")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

CACHE_BUST = hashlib.md5(str(datetime.now().hour).encode()).hexdigest()[:8]
LATEST_SEASON = 2024

app.state.cache_bust = CACHE_BUST
app.state.current_year = LATEST_SEASON

@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if not path.startswith("/static") and request.headers.get("accept", "") == "*/*":
        pass
    if path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif not path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

_data_loaded = False

def ensure_data():
    global _data_loaded
    if _data_loaded:
        return
    import logging, time
    log = logging.getLogger(__name__)
    t0 = time.time()
    try:
        data.load_player_stats(LATEST_SEASON)
        data.load_schedule(LATEST_SEASON)
        _data_loaded = True
        log.warning(f"ScoreEdge data loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        log.error(f"ScoreEdge data load FAILED: {e}")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    ensure_data()
    schedule = data.load_schedule(LATEST_SEASON)
    players = data.load_player_stats(LATEST_SEASON)
    stats = data.load_player_stats(LATEST_SEASON)
    odds = data.fetch_odds()

    upcoming = []
    if not schedule.empty:
        cols = schedule.columns.tolist()
        date_col = next((c for c in cols if c.lower() in ("date", "gameday", "game_date")), None)
        if date_col:
            schedule[date_col] = pd.to_datetime(schedule[date_col], errors="coerce")
            future = schedule[schedule[date_col] >= pd.Timestamp.now()].head(10)
            for _, row in future.iterrows():
                game = {"week": row.get("week", ""), "date": str(row.get(date_col, ""))[:10]}
                for col in ["home_team", "away_team", "home_score", "away_score"]:
                    if col in row.index:
                        game[col] = row[col]
                upcoming.append(game)

    top_passers, top_rushers, top_receivers = [], [], []
    if not stats.empty:
        for col in ["passing_yards", "rushing_yards", "receiving_yards"]:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
        for col in ["passing_tds", "rushing_tds", "receiving_tds", "receptions"]:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
        if "player_name" in stats.columns:
            if "passing_yards" in stats.columns:
                _team_col = "recent_team" if "recent_team" in stats.columns else "team"
                top_passers = stats.nlargest(5, "passing_yards")[["player_name", _team_col, "passing_yards", "passing_tds"]].rename(columns={_team_col: "team"}).to_dict("records")
            if "rushing_yards" in stats.columns:
                _team_col = "recent_team" if "recent_team" in stats.columns else "team"
                top_rushers = stats.nlargest(5, "rushing_yards")[["player_name", _team_col, "rushing_yards", "rushing_tds"]].rename(columns={_team_col: "team"}).to_dict("records")
            if "receiving_yards" in stats.columns:
                _team_col = "recent_team" if "recent_team" in stats.columns else "team"
                top_receivers = stats.nlargest(5, "receiving_yards")[["player_name", _team_col, "receiving_yards", "receiving_tds"]].rename(columns={_team_col: "team"}).to_dict("records")

    return templates.TemplateResponse(request, "home.html", {
        "request": request, "active_page": "home",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
        "upcoming": upcoming, "odds": odds[:10],
        "top_passers": top_passers, "top_rushers": top_rushers, "top_receivers": top_receivers,
        "player_count": len(players) if not players.empty else 0,
    })

@app.get("/players", response_class=HTMLResponse)
def players_page(request: Request, q: str = "", pos: str = "", sort: str = "fantasy_pts", season: int = 0):
    ensure_data()
    season = season or LATEST_SEASON
    stats = data.load_player_stats(season)

    if stats.empty:
        return templates.TemplateResponse(request, "players.html", {
            "request": request, "active_page": "players",
            "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
            "players": [], "positions": [], "q": q, "pos": pos, "sort": sort, "season": season,
        })

    for col in ["passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds",
                 "receiving_tds", "receptions", "interceptions", "carries", "targets"]:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)

    stats["fantasy_pts_ppr"] = analytics.compute_fantasy_points(stats, "ppr")

    team_map = {}
    if "player_name" in stats.columns:
        if "recent_team" in stats.columns:
            team_map = stats.groupby("player_id")["recent_team"].first().to_dict()
        grouped = stats.groupby(["player_id", "player_name"], as_index=False).agg({
            col: "sum" for col in stats.select_dtypes(include="number").columns
        })
        if team_map:
            grouped["recent_team"] = grouped["player_id"].map(team_map)
    else:
        grouped = stats

    if q:
        grouped = grouped[grouped["player_name"].str.contains(q, case=False, na=False)]
    if pos and pos != "ALL":
        if "position" in grouped.columns:
            grouped = grouped[grouped["position"] == pos]

    positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
    sort_col = sort if sort in grouped.columns else "fantasy_pts_ppr"
    grouped = grouped.sort_values(sort_col, ascending=False).head(100)

    players = grouped.to_dict("records")

    return templates.TemplateResponse(request, "players.html", {
        "request": request, "active_page": "players",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
        "players": players, "positions": positions, "q": q, "pos": pos, "sort": sort, "season": season,
    })

@app.get("/players/{player_id}", response_class=HTMLResponse)
def player_profile(request: Request, player_id: str):
    ensure_data()
    stats = data.load_player_stats(LATEST_SEASON)
    player = {}
    seasons = []
    if not stats.empty and "player_id" in stats.columns:
        p = stats[stats["player_id"] == player_id]
        if not p.empty:
            for col in ["passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds",
                         "receiving_tds", "receptions", "interceptions"]:
                if col in p.columns:
                    p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
            numeric_cols = p.select_dtypes(include="number").columns.tolist()
            season_group = p.groupby("season")[numeric_cols].sum().reset_index()
            player = p.iloc[0].to_dict()
            for col in numeric_cols:
                player[col] = season_group[col].sum()
            if "recent_team" in player and "team" not in player:
                player["team"] = player["recent_team"]
            seasons = season_group.to_dict("records")

    return templates.TemplateResponse(request, "player.html", {
        "request": request, "active_page": "players",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
        "player": player, "seasons": seasons, "player_id": player_id,
    })

@app.get("/odds", response_class=HTMLResponse)
def odds_page(request: Request):
    ensure_data()
    h2h = data.fetch_odds("americanfootball_nfl", "h2h")
    spreads = data.fetch_odds("americanfootball_nfl", "spreads")
    totals = data.fetch_odds("americanfootball_nfl", "totals")
    return templates.TemplateResponse(request, "odds.html", {
        "request": request, "active_page": "odds",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
        "h2h": h2h, "spreads": spreads, "totals": totals,
        "api_key_set": bool(data.ODDS_API_KEY),
    })

@app.get("/rankings", response_class=HTMLResponse)
def rankings_page(request: Request, scoring: str = "ppr", position: str = "ALL"):
    ensure_data()
    stats = data.load_player_stats(LATEST_SEASON)
    if stats.empty:
        return templates.TemplateResponse(request, "rankings.html", {
            "request": request, "active_page": "rankings",
            "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
            "rankings": [], "scoring": scoring, "position": position,
        })
    for col in ["passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds",
                 "receiving_tds", "receptions", "interceptions"]:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
    stats["fantasy_pts"] = analytics.compute_fantasy_points(stats, scoring)
    team_map = {}
    if "player_name" in stats.columns:
        if "recent_team" in stats.columns:
            team_map = stats.groupby("player_id")["recent_team"].first().to_dict()
        grouped = stats.groupby(["player_id", "player_name"], as_index=False).agg({
            col: "sum" for col in stats.select_dtypes(include="number").columns
        })
        if team_map:
            grouped["recent_team"] = grouped["player_id"].map(team_map)
    else:
        grouped = stats
    if position and position != "ALL" and "position" in grouped.columns:
        grouped = grouped[grouped["position"] == position]
    grouped = grouped.sort_values("fantasy_pts", ascending=False).head(200).reset_index(drop=True)
    grouped["rank"] = range(1, len(grouped) + 1)
    positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
    return templates.TemplateResponse(request, "rankings.html", {
        "request": request, "active_page": "rankings",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
        "rankings": grouped.to_dict("records"), "scoring": scoring, "position": position, "positions": positions,
    })

@app.get("/weather", response_class=HTMLResponse)
def weather_page(request: Request):
    ensure_data()
    games = data.load_schedule(LATEST_SEASON)
    weather_data = []
    if not games.empty:
        date_col = next((c for c in games.columns if c.lower() in ("date", "gameday", "game_date")), None)
        if date_col:
            games[date_col] = pd.to_datetime(games[date_col], errors="coerce")
            today = pd.Timestamp.now().normalize()
            upcoming = games[(games[date_col] >= today) & (games[date_col] <= today + pd.Timedelta(days=7))]
            for _, row in upcoming.head(10).iterrows():
                home = str(row.get("home_team", ""))
                stadium = data.NFL_STADIUMS.get(home, {})
                if stadium and stadium.get("roof") == "open":
                    w = data.fetch_weather(stadium["lat"], stadium["lon"])
                    daily = w.get("daily", {})
                    weather_data.append({
                        "home": home, "away": row.get("away_team", ""),
                        "stadium": stadium.get("name", ""),
                        "temp_max": daily.get("temperature_2m_max", [None])[0],
                        "temp_min": daily.get("temperature_2m_min", [None])[0],
                        "precip": daily.get("precipitation_sum", [None])[0],
                        "wind": daily.get("wind_speed_10m_max", [None])[0],
                    })
                else:
                    weather_data.append({
                        "home": home, "away": row.get("away_team", ""),
                        "stadium": stadium.get("name", "Indoor"),
                        "temp_max": None, "temp_min": None, "precip": None, "wind": None,
                    })

    return templates.TemplateResponse(request, "weather.html", {
        "request": request, "active_page": "weather",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
        "weather_data": weather_data,
    })

@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request):
    return templates.TemplateResponse(request, "trades.html", {
        "request": request, "active_page": "trades",
        "cache_bust": CACHE_BUST, "current_year": LATEST_SEASON,
    })

@app.get("/api/player-search", response_class=HTMLResponse)
def player_search_api(request: Request, q: str = ""):
    ensure_data()
    if len(q) < 2:
        return HTMLResponse("")
    stats = data.load_player_stats(LATEST_SEASON)
    if stats.empty or "player_name" not in stats.columns:
        return HTMLResponse("")
    matches = stats[stats["player_name"].str.contains(q, case=False, na=False)].head(10)
    if matches.empty:
        return HTMLResponse('<div class="p-4 text-sm text-slate-500">No players found</div>')
    rows = []
    for _, r in matches.iterrows():
        name = r.get("player_name", "")
        pid = r.get("player_id", "")
        pos = r.get("position", "")
        team = r.get("recent_team", r.get("team", ""))
        rows.append(f'<a href="/players/{pid}" class="flex items-center gap-3 px-4 py-2.5 hover:bg-blue-500/10 transition-colors border-b border-white/5 last:border-0"><div class="flex-1"><div class="text-sm font-medium text-white">{name}</div><div class="text-xs text-slate-500">{team} · {pos}</div></div><svg class="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a>')
    return HTMLResponse(content=''.join(rows))

@app.get("/robots.txt")
def robots_txt():
    return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /static/\n", media_type="text/plain")

@app.get("/sitemap.xml")
def sitemap_xml():
    base = "https://scoreedge.onrender.com"
    pages = ["", "/players", "/odds", "/rankings", "/weather", "/trades"]
    urls = "\n".join(f'  <url><loc>{base}{p}</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>' for p in pages)
    xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}\n</urlset>'
    return PlainTextResponse(xml, media_type="application/xml")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(request, "404.html", {
        "request": request, "active_page": "404"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8512)

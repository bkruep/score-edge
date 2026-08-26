from __future__ import annotations
import hashlib, os
from datetime import datetime
from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd

import data
import analytics

app = FastAPI(title="ScoreEdge")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

CACHE_BUST = hashlib.md5(str(datetime.now().hour).encode()).hexdigest()[:8]
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

app.state.cache_bust = CACHE_BUST

@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif not path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

_data_loaded = False
_latest_season = None

def get_latest_season():
    global _latest_season
    if _latest_season is not None:
        return _latest_season
    import requests as _req
    for yr in range(datetime.now().year, 2019, -1):
        url = f"{data.NFLVERSE_BASE}/player_stats/player_stats_{yr}.csv"
        try:
            r = _req.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                _latest_season = yr
                return yr
        except Exception:
            continue
    _latest_season = 2024
    return 2024

def ensure_data():
    global _data_loaded
    if _data_loaded:
        return
    import logging, time
    log = logging.getLogger(__name__)
    t0 = time.time()
    season = get_latest_season()
    try:
        data.load_player_stats(season)
        data.load_schedule(season)
        _data_loaded = True
        log.warning(f"ScoreEdge data loaded in {time.time()-t0:.1f}s (season {season})")
    except Exception as e:
        log.error(f"ScoreEdge data load FAILED: {e}")

def _filter_fantasy(df: pd.DataFrame) -> pd.DataFrame:
    if "position" in df.columns:
        return df[df["position"].isin(FANTASY_POSITIONS)]
    return df

def _get_season():
    return get_latest_season()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    ensure_data()
    season = _get_season()
    schedule = data.load_schedule(season)
    stats = _filter_fantasy(data.load_player_stats(season))
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
        for col in ["passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds", "receiving_tds", "receptions"]:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
        if "player_name" in stats.columns:
            _team = "recent_team" if "recent_team" in stats.columns else "team"
            if "passing_yards" in stats.columns:
                top_passers = stats.nlargest(5, "passing_yards")[["player_name", _team, "passing_yards", "passing_tds"]].rename(columns={_team: "team"}).to_dict("records")
            if "rushing_yards" in stats.columns:
                top_rushers = stats.nlargest(5, "rushing_yards")[["player_name", _team, "rushing_yards", "rushing_tds"]].rename(columns={_team: "team"}).to_dict("records")
            if "receiving_yards" in stats.columns:
                top_receivers = stats.nlargest(5, "receiving_yards")[["player_name", _team, "receiving_yards", "receiving_tds"]].rename(columns={_team: "team"}).to_dict("records")

    return templates.TemplateResponse(request, "home.html", {
        "request": request, "active_page": "home",
        "cache_bust": CACHE_BUST, "current_year": season,
        "upcoming": upcoming, "odds": odds[:10],
        "top_passers": top_passers, "top_rushers": top_rushers, "top_receivers": top_receivers,
        "player_count": len(stats) if not stats.empty else 0,
    })

@app.get("/players", response_class=HTMLResponse)
def players_page(request: Request, q: str = "", pos: str = "", sort: str = "fantasy_pts", season: int = 0):
    ensure_data()
    season = season or _get_season()
    stats = _filter_fantasy(data.load_player_stats(season))

    if stats.empty:
        return templates.TemplateResponse(request, "players.html", {
            "request": request, "active_page": "players",
            "cache_bust": CACHE_BUST, "current_year": season,
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
        "cache_bust": CACHE_BUST, "current_year": season,
        "players": players, "positions": positions, "q": q, "pos": pos, "sort": sort, "season": season,
    })

@app.get("/players/{player_id}", response_class=HTMLResponse)
def player_profile(request: Request, player_id: str):
    ensure_data()
    season = _get_season()
    stats = data.load_player_stats(season)
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
        "cache_bust": CACHE_BUST, "current_year": season,
        "player": player, "seasons": seasons, "player_id": player_id,
    })

@app.get("/odds", response_class=HTMLResponse)
def odds_page(request: Request):
    ensure_data()
    season = _get_season()
    h2h = data.fetch_odds("americanfootball_nfl", "h2h")
    spreads = data.fetch_odds("americanfootball_nfl", "spreads")
    totals = data.fetch_odds("americanfootball_nfl", "totals")

    weather_data = []
    schedule = data.load_schedule(season)
    if not schedule.empty:
        cols = schedule.columns.tolist()
        date_col = next((c for c in cols if c.lower() in ("date", "gameday", "game_date")), None)
        if date_col:
            schedule[date_col] = pd.to_datetime(schedule[date_col], errors="coerce")
            today = pd.Timestamp.now().normalize()
            upcoming = schedule[(schedule[date_col] >= today) & (schedule[date_col] <= today + pd.Timedelta(days=7))]
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

    return templates.TemplateResponse(request, "odds.html", {
        "request": request, "active_page": "odds",
        "cache_bust": CACHE_BUST, "current_year": season,
        "h2h": h2h, "spreads": spreads, "totals": totals,
        "weather_data": weather_data,
        "api_key_set": bool(data.ODDS_API_KEY),
    })

@app.get("/rankings", response_class=HTMLResponse)
def rankings_page(request: Request, scoring: str = "ppr", position: str = "ALL"):
    ensure_data()
    season = _get_season()
    stats = _filter_fantasy(data.load_player_stats(season))
    if stats.empty:
        return templates.TemplateResponse(request, "rankings.html", {
            "request": request, "active_page": "rankings",
            "cache_bust": CACHE_BUST, "current_year": season,
            "rankings": [], "scoring": scoring, "position": position, "positions": [],
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
        "cache_bust": CACHE_BUST, "current_year": season,
        "rankings": grouped.to_dict("records"), "scoring": scoring, "position": position, "positions": positions,
    })

@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request):
    ensure_data()
    season = _get_season()
    return templates.TemplateResponse(request, "trades.html", {
        "request": request, "active_page": "trades",
        "cache_bust": CACHE_BUST, "current_year": season,
    })

@app.get("/api/player-search", response_class=HTMLResponse)
def player_search_api(request: Request, q: str = ""):
    ensure_data()
    if len(q) < 2:
        return HTMLResponse("")
    stats = _filter_fantasy(data.load_player_stats(_get_season()))
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

@app.get("/api/trade-compare", response_class=HTMLResponse)
def trade_compare_api(request: Request, a: str = "", b: str = ""):
    ensure_data()
    if not a or not b:
        return HTMLResponse("")
    stats = _filter_fantasy(data.load_player_stats(_get_season()))
    if stats.empty or "player_id" not in stats.columns:
        return HTMLResponse("")
    for col in ["passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds",
                 "receiving_tds", "receptions", "interceptions", "carries", "targets"]:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
    stats["fantasy_pts"] = analytics.compute_fantasy_points(stats, "ppr")

    pa = stats[stats["player_id"] == a]
    pb = stats[stats["player_id"] == b]
    if pa.empty or pb.empty:
        return HTMLResponse('<div class="p-4 text-sm text-slate-500">Player not found</div>')

    def _summarize(p):
        num_cols = p.select_dtypes(include="number").columns
        s = p[num_cols].sum()
        name = p.iloc[0].get("player_name", "")
        pos = p.iloc[0].get("position", "")
        team = p.iloc[0].get("recent_team", p.iloc[0].get("team", ""))
        return {"name": name, "pos": pos, "team": team, "stats": s.to_dict()}

    sa = _summarize(pa)
    sb = _summarize(pb)

    a_pts = sa["stats"].get("fantasy_pts", 0)
    b_pts = sb["stats"].get("fantasy_pts", 0)

    html = f'''<div class="p-5">
<div class="grid grid-cols-2 gap-4 mb-4">
  <div class="text-center">
    <div class="text-lg font-bold text-white">{sa['name']}</div>
    <div class="text-xs text-slate-400">{sa['team']} · {sa['pos']}</div>
    <div class="text-2xl font-display font-bold text-brand-light mt-2">{a_pts:.1f}</div>
    <div class="text-xs text-slate-500">Fantasy Pts (PPR)</div>
  </div>
  <div class="text-center">
    <div class="text-lg font-bold text-white">{sb['name']}</div>
    <div class="text-xs text-slate-400">{sb['team']} · {sb['pos']}</div>
    <div class="text-2xl font-display font-bold text-accent mt-2">{b_pts:.1f}</div>
    <div class="text-xs text-slate-500">Fantasy Pts (PPR)</div>
  </div>
</div>
<table class="w-full text-sm">
<thead><tr class="border-b border-white/5"><th class="px-3 py-2 text-left text-xs text-slate-400">Stat</th><th class="px-3 py-2 text-right text-xs text-slate-400">{sa['name']}</th><th class="px-3 py-2 text-right text-xs text-slate-400">{sb['name']}</th></tr></thead>
<tbody class="divide-y divide-white/5">'''

    stat_labels = [
        ("passing_yards", "Pass Yards"), ("passing_tds", "Pass TD"), ("interceptions", "INT"),
        ("rushing_yards", "Rush Yards"), ("rushing_tds", "Rush TD"),
        ("receiving_yards", "Rec Yards"), ("receiving_tds", "Rec TD"), ("receptions", "Receptions"),
        ("fantasy_pts", "Fantasy Pts"),
    ]
    for key, label in stat_labels:
        va = sa["stats"].get(key, 0)
        vb = sb["stats"].get(key, 0)
        winner_a = "text-brand-light" if va > vb else ""
        winner_b = "text-accent" if vb > va else ""
        html += f'<tr><td class="px-3 py-2 text-slate-400">{label}</td><td class="px-3 py-2 text-right font-mono {winner_a}">{va:.0f}</td><td class="px-3 py-2 text-right font-mono {winner_b}">{vb:.0f}</td></tr>'

    adv = "Side A" if a_pts > b_pts else "Side B" if b_pts > a_pts else "Even"
    diff = abs(a_pts - b_pts)
    html += f'''</tbody></table>
<div class="mt-4 text-center text-sm text-slate-400">Advantage: <span class="text-white font-medium">{adv}</span> (+{diff:.1f} pts)</div>
</div>'''
    return HTMLResponse(content=html)

@app.get("/robots.txt")
def robots_txt():
    return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /static/\n", media_type="text/plain")

@app.get("/sitemap.xml")
def sitemap_xml():
    base = "https://scoreedge.onrender.com"
    pages = ["", "/players", "/odds", "/rankings"]
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

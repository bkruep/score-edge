from __future__ import annotations
import hashlib, os
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd

import data

app = FastAPI(title="ScoreEdge")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

CACHE_BUST = hashlib.md5(str(datetime.now().hour).encode()).hexdigest()[:8]
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF", "DST"}

SPORTS = {
    "nfl": {"name": "NFL", "href": "/", "icon": "🏈"},
    "nba": {"name": "NBA", "href": "/nba", "icon": "🏀"},
    "cfb": {"name": "CFB", "href": "/cfb", "icon": "🏈"},
    "cbb": {"name": "CBB", "href": "/cbb", "icon": "🏀"},
    "mlb": {"name": "MLB", "href": "/mlb", "icon": "⚾"},
    "nhl": {"name": "NHL", "href": "/nhl", "icon": "🏒"},
    "mls": {"name": "MLS", "href": "/mls", "icon": "⚽"},
}

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

def _get_season() -> int:
    return 2025

def _season_display(season: int) -> str:
    return f"{season + 1}"

def _is_nfl_regular_season() -> bool:
    now = datetime.now()
    month = now.month
    day = now.day
    if month in (9, 10, 11, 12):
        return True
    if month == 1:
        return True
    if month == 2 and day <= 14:
        return True
    return False

def ensure_data():
    global _data_loaded
    if _data_loaded:
        return
    import logging, time
    log = logging.getLogger(__name__)
    t0 = time.time()
    season = _get_season()
    try:
        data.load_player_stats(season)
        data.load_schedule(season)
        data.load_adp("ppr")
        data.load_adp("half-ppr")
        data.load_adp("standard")
        _data_loaded = True
        log.warning(f"ScoreEdge data loaded in {time.time()-t0:.1f}s (season {season})")
    except Exception as e:
        log.error(f"ScoreEdge data load FAILED: {e}")

def _filter_fantasy(df: pd.DataFrame) -> pd.DataFrame:
    if "position" in df.columns:
        return df[df["position"].isin(FANTASY_POSITIONS)]
    return df

def _normalize_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renames = {
        "player_display_name": "player_name",
        "passing_interceptions": "interceptions",
        "fantasy_points_ppr": "pts_ppr",
        "fantasy_points": "pts_std",
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    num_cols = ["passing_yards", "rushing_yards", "receiving_yards",
                "passing_tds", "rushing_tds", "receiving_tds", "receptions",
                "interceptions", "carries", "targets",
                "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
                "rushing_fumbles_lost", "receiving_fumbles_lost",
                "special_teams_tds", "games", "pass_att", "pass_cmp",
                "pts_ppr", "pts_std",
                "pos_rank_ppr", "pos_rank_half_ppr", "pos_rank_std"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "player_name" not in df.columns:
        if "player_id" in df.columns:
            df["player_name"] = df["player_id"]
        else:
            df["player_name"] = "Unknown"

    if "pts_half_ppr" not in df.columns:
        if "pts_ppr" in df.columns and "pts_std" in df.columns:
            df["pts_half_ppr"] = ((df["pts_ppr"] + df["pts_std"]) / 2).round(2)
        elif "pts_ppr" in df.columns:
            df["pts_half_ppr"] = df["pts_ppr"]
        else:
            df["pts_half_ppr"] = 0

    return df

def _ctx(request: Request, sport: str = "nfl", **extra):
    season = _get_season()
    base = {"request": request, "active_sport": sport, "cache_bust": CACHE_BUST, "current_year": _season_display(season)}
    base.update(extra)
    return base

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    ensure_data()
    season = _get_season()
    schedule = data.load_schedule(season)
    stats = _filter_fantasy(_normalize_stats(data.load_player_stats(season)))
    odds = data.fetch_odds()
    adp = data.load_adp("ppr")

    upcoming = []
    if not schedule.empty and "gameday" in schedule.columns:
        schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
        future = schedule[schedule["gameday"] >= pd.Timestamp.now()].head(10)
        for _, row in future.iterrows():
            game = {"week": row.get("week", ""), "date": str(row.get("gameday", ""))[:10]}
            for col in ["home_team", "away_team", "home_score", "away_score"]:
                if col in row.index:
                    game[col] = row[col]
            upcoming.append(game)

    draft_board = []
    if not adp.empty:
        cols = ["name", "position", "team", "adp", "adp_formatted", "bye"]
        available = [c for c in cols if c in adp.columns]
        draft_board = adp[available].head(50).to_dict("records")

    return templates.TemplateResponse(request, "home.html", {
        **_ctx(request, "nfl", active_page="home"),
        "upcoming": upcoming, "odds": odds[:10],
        "draft_board": draft_board,
        "player_count": len(stats) if not stats.empty else 0,
        "draft_count": len(adp) if not adp.empty else 0,
    })

@app.get("/players", response_class=HTMLResponse)
def players_page(request: Request, q: str = "", pos: str = "", sort: str = "fantasy_pts"):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()
    adp = data.load_adp("ppr")

    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(season)))
        if stats.empty:
            return templates.TemplateResponse(request, "players.html", {
                **_ctx(request, "nfl", active_page="players"),
                "players": [], "positions": [], "q": q, "pos": pos, "sort": sort, "in_season": True,
            })
        sort_map = {"fantasy_pts": "pts_ppr", "passing_yards": "passing_yards",
                    "rushing_yards": "rushing_yards", "receiving_yards": "receiving_yards"}
        sort_col = sort_map.get(sort, "pts_ppr")
        if q:
            stats = stats[stats["player_name"].str.contains(q, case=False, na=False)]
        if pos and pos != "ALL" and "position" in stats.columns:
            stats = stats[stats["position"] == pos]
        positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
        stats = stats.sort_values(sort_col, ascending=False).head(200)
        return templates.TemplateResponse(request, "players.html", {
            **_ctx(request, "nfl", active_page="players"),
            "players": stats.to_dict("records"), "positions": positions, "q": q, "pos": pos, "sort": sort, "in_season": True,
        })
    else:
        df = adp.copy() if not adp.empty else pd.DataFrame()
        if q and not df.empty and "name" in df.columns:
            df = df[df["name"].str.contains(q, case=False, na=False)]
        if pos and pos != "ALL" and not df.empty and "position" in df.columns:
            df = df[df["position"] == pos]
        positions = sorted(df["position"].dropna().unique().tolist()) if not df.empty and "position" in df.columns else []
        df = df.sort_values("adp").head(200).reset_index(drop=True) if not df.empty else df
        return templates.TemplateResponse(request, "players.html", {
            **_ctx(request, "nfl", active_page="players"),
            "players": df.to_dict("records") if not df.empty else [], "positions": positions,
            "q": q, "pos": pos, "sort": "adp", "in_season": False,
        })

@app.get("/players/{player_id}", response_class=HTMLResponse)
def player_profile(request: Request, player_id: str):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()

    if in_season:
        stats = _normalize_stats(data.load_player_stats(season))
        player = {}
        if not stats.empty and "player_id" in stats.columns:
            p = stats[stats["player_id"] == player_id]
            if not p.empty:
                player = p.iloc[0].to_dict()
        return templates.TemplateResponse(request, "player.html", {
            **_ctx(request, "nfl", active_page="players"),
            "player": player, "player_id": player_id, "in_season": True,
        })
    else:
        adp = data.load_adp("ppr")
        player = {}
        if not adp.empty:
            p = adp[adp["player_id"].astype(str) == str(player_id)]
            if not p.empty:
                player = p.iloc[0].to_dict()
        return templates.TemplateResponse(request, "player.html", {
            **_ctx(request, "nfl", active_page="players"),
            "player": player, "player_id": player_id, "in_season": False,
        })

@app.get("/bets", response_class=HTMLResponse)
def bets_page(request: Request):
    ensure_data()
    season = _get_season()
    h2h = data.fetch_odds("americanfootball_nfl", "h2h")
    spreads = data.fetch_odds("americanfootball_nfl", "spreads")
    totals = data.fetch_odds("americanfootball_nfl", "totals")

    weather_data = []
    schedule = data.load_schedule(season)
    if not schedule.empty and "gameday" in schedule.columns:
        schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        upcoming = schedule[(schedule["gameday"] >= today) & (schedule["gameday"] <= today + pd.Timedelta(days=7))]
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

    return templates.TemplateResponse(request, "bets.html", {
        **_ctx(request, "nfl", active_page="bets"),
        "h2h": h2h, "spreads": spreads, "totals": totals,
        "weather_data": weather_data,
        "api_key_set": bool(data.ODDS_API_KEY),
    })

@app.get("/rankings", response_class=HTMLResponse)
def rankings_page(request: Request, scoring: str = "ppr", position: str = "ALL"):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()

    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(season)))
        if stats.empty:
            return templates.TemplateResponse(request, "rankings.html", {
                **_ctx(request, "nfl", active_page="rankings"),
                "rankings": [], "scoring": scoring, "position": position, "positions": [], "source": "stats",
            })
        pts_col = {"ppr": "pts_ppr", "standard": "pts_std", "half_ppr": "pts_half_ppr"}.get(scoring, "pts_ppr")
        stats["fantasy_pts"] = stats[pts_col]
        if position and position != "ALL" and "position" in stats.columns:
            stats = stats[stats["position"] == position]
        stats = stats.sort_values("fantasy_pts", ascending=False).head(200).reset_index(drop=True)
        stats["rank"] = range(1, len(stats) + 1)
        positions = sorted(stats["position"].dropna().unique().tolist()) if "position" in stats.columns else []
        return templates.TemplateResponse(request, "rankings.html", {
            **_ctx(request, "nfl", active_page="rankings"),
            "rankings": stats.to_dict("records"), "scoring": scoring, "position": position,
            "positions": positions, "source": "stats",
        })
    else:
        adp = data.load_adp(scoring)
        if not adp.empty:
            df = adp.copy()
            if position and position != "ALL" and "position" in df.columns:
                df = df[df["position"] == position]
            df = df.sort_values("adp").head(200).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
            positions = sorted(adp["position"].dropna().unique().tolist()) if "position" in adp.columns else []
            return templates.TemplateResponse(request, "rankings.html", {
                **_ctx(request, "nfl", active_page="rankings"),
                "rankings": df.to_dict("records"), "scoring": scoring, "position": position,
                "positions": positions, "source": "adp",
            })
        return templates.TemplateResponse(request, "rankings.html", {
            **_ctx(request, "nfl", active_page="rankings"),
            "rankings": [], "scoring": scoring, "position": position, "positions": [], "source": "adp",
        })

@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request):
    ensure_data()
    season = _get_season()
    in_season = _is_nfl_regular_season()
    return templates.TemplateResponse(request, "trades.html", {
        **_ctx(request, "nfl", active_page="trades"),
        "in_season": in_season,
    })

@app.get("/api/player-search", response_class=HTMLResponse)
def player_search_api(request: Request, q: str = ""):
    ensure_data()
    if len(q) < 2:
        return HTMLResponse("")
    in_season = _is_nfl_regular_season()
    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
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
    else:
        adp = data.load_adp("ppr")
        if adp.empty or "name" not in adp.columns:
            return HTMLResponse("")
        matches = adp[adp["name"].str.contains(q, case=False, na=False)].head(10)
        if matches.empty:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">No players found</div>')
        rows = []
        for _, r in matches.iterrows():
            name = r.get("name", "")
            pid = r.get("player_id", "")
            pos = r.get("position", "")
            team = r.get("team", "")
            adp_val = r.get("adp_formatted", "")
            rows.append(f'<a href="/players/{pid}" class="flex items-center gap-3 px-4 py-2.5 hover:bg-blue-500/10 transition-colors border-b border-white/5 last:border-0"><div class="flex-1"><div class="text-sm font-medium text-white">{name}</div><div class="text-xs text-slate-500">{team} · {pos} · ADP {adp_val}</div></div><svg class="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a>')
        return HTMLResponse(content=''.join(rows))

@app.get("/api/trade-compare", response_class=HTMLResponse)
def trade_compare_api(request: Request, a: str = "", b: str = ""):
    ensure_data()
    if not a or not b:
        return HTMLResponse("")
    in_season = _is_nfl_regular_season()

    if in_season:
        stats = _filter_fantasy(_normalize_stats(data.load_player_stats(_get_season())))
        if stats.empty or "player_id" not in stats.columns:
            return HTMLResponse("")
        pa = stats[stats["player_id"] == a]
        pb = stats[stats["player_id"] == b]
        if pa.empty or pb.empty:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">Player not found</div>')
        def _summarize(p):
            row = p.iloc[0]
            return {"name": row.get("player_name", ""), "pos": row.get("position", ""),
                    "team": row.get("recent_team", ""), "pts": float(row.get("pts_ppr", 0)), "row": row.to_dict()}
        sa, sb = _summarize(pa), _summarize(pb)
        html = _trade_html(sa, sb, stat_mode=True)
        return HTMLResponse(content=html)
    else:
        adp = data.load_adp("ppr")
        if adp.empty:
            return HTMLResponse("")
        pa = adp[adp["player_id"].astype(str) == str(a)]
        pb = adp[adp["player_id"].astype(str) == str(b)]
        if pa.empty or pb.empty:
            return HTMLResponse('<div class="p-4 text-sm text-slate-500">Player not found</div>')
        def _summarize_adp(p):
            row = p.iloc[0]
            adp_val = float(row.get("adp", 999))
            return {"name": row.get("name", ""), "pos": row.get("position", ""),
                    "team": row.get("team", ""), "adp": adp_val, "adp_formatted": row.get("adp_formatted", ""),
                    "bye": row.get("bye", ""), "row": row.to_dict()}
        sa, sb = _summarize_adp(pa), _summarize_adp(pb)
        html = _trade_html(sa, sb, stat_mode=False)
        return HTMLResponse(content=html)

def _trade_html(a: dict, b: dict, stat_mode: bool) -> str:
    if stat_mode:
        html = f'''<div class="glass-card rounded-xl p-5">
<div class="grid grid-cols-2 gap-4 mb-4">
  <div class="text-center">
    <div class="text-lg font-bold text-white">{a['name']}</div>
    <div class="text-xs text-slate-400">{a['team']} · {a['pos']}</div>
    <div class="text-2xl font-display font-bold text-brand-light mt-2">{a['pts']:.1f}</div>
    <div class="text-xs text-slate-500">Fantasy Pts (PPR)</div>
  </div>
  <div class="text-center">
    <div class="text-lg font-bold text-white">{b['name']}</div>
    <div class="text-xs text-slate-400">{b['team']} · {b['pos']}</div>
    <div class="text-2xl font-display font-bold text-accent mt-2">{b['pts']:.1f}</div>
    <div class="text-xs text-slate-500">Fantasy Pts (PPR)</div>
  </div>
</div>
<table class="w-full text-sm">
<thead><tr class="border-b border-white/5"><th class="px-3 py-2 text-left text-xs text-slate-400">Stat</th><th class="px-3 py-2 text-right text-xs text-slate-400">{a['name']}</th><th class="px-3 py-2 text-right text-xs text-slate-400">{b['name']}</th></tr></thead>
<tbody class="divide-y divide-white/5">'''
        stat_labels = [
            ("passing_yards", "Pass Yards"), ("passing_tds", "Pass TD"), ("interceptions", "INT"),
            ("rushing_yards", "Rush Yards"), ("rushing_tds", "Rush TD"),
            ("receiving_yards", "Rec Yards"), ("receiving_tds", "Rec TD"), ("receptions", "Receptions"),
            ("pts_ppr", "Fantasy Pts"),
        ]
        for key, label in stat_labels:
            va = float(a["row"].get(key, 0))
            vb = float(b["row"].get(key, 0))
            wa = "text-brand-light" if va > vb else ""
            wb = "text-accent" if vb > va else ""
            html += f'<tr><td class="px-3 py-2 text-slate-400">{label}</td><td class="px-3 py-2 text-right font-mono {wa}">{va:.0f}</td><td class="px-3 py-2 text-right font-mono {wb}">{vb:.0f}</td></tr>'
        adv = "Side A" if a["pts"] > b["pts"] else "Side B" if b["pts"] > a["pts"] else "Even"
        diff = abs(a["pts"] - b["pts"])
        html += f'''</tbody></table>
<div class="mt-4 text-center text-sm text-slate-400">Advantage: <span class="text-white font-medium">{adv}</span> (+{diff:.1f} pts)</div>
</div>'''
    else:
        a_rank = int(a["adp"])
        b_rank = int(b["adp"])
        a_val = max(0, 300 - a_rank)
        b_val = max(0, 300 - b_rank)
        adv = "Side A" if a_rank < b_rank else "Side B" if b_rank < a_rank else "Even"
        diff = abs(a_rank - b_rank)
        html = f'''<div class="glass-card rounded-xl p-5">
<div class="grid grid-cols-2 gap-4 mb-4">
  <div class="text-center">
    <div class="text-lg font-bold text-white">{a['name']}</div>
    <div class="text-xs text-slate-400">{a['team']} · {a['pos']}</div>
    <div class="text-2xl font-display font-bold text-brand-light mt-2">#{a['adp_formatted']}</div>
    <div class="text-xs text-slate-500">Avg Draft Position</div>
  </div>
  <div class="text-center">
    <div class="text-lg font-bold text-white">{b['name']}</div>
    <div class="text-xs text-slate-400">{b['team']} · {b['pos']}</div>
    <div class="text-2xl font-display font-bold text-accent mt-2">#{b['adp_formatted']}</div>
    <div class="text-xs text-slate-500">Avg Draft Position</div>
  </div>
</div>
<table class="w-full text-sm">
<thead><tr class="border-b border-white/5"><th class="px-3 py-2 text-left text-xs text-slate-400">Metric</th><th class="px-3 py-2 text-right text-xs text-slate-400">{a['name']}</th><th class="px-3 py-2 text-right text-xs text-slate-400">{b['name']}</th></tr></thead>
<tbody class="divide-y divide-white/5">
<tr><td class="px-3 py-2 text-slate-400">ADP (12-team)</td><td class="px-3 py-2 text-right font-mono {"text-brand-light" if a_rank < b_rank else ""}">{a['adp_formatted']}</td><td class="px-3 py-2 text-right font-mono {"text-accent" if b_rank < a_rank else ""}">{b['adp_formatted']}</td></tr>
<tr><td class="px-3 py-2 text-slate-400">Position</td><td class="px-3 py-2 text-right font-mono text-white">{a['pos']}</td><td class="px-3 py-2 text-right font-mono text-white">{b['pos']}</td></tr>
<tr><td class="px-3 py-2 text-slate-400">Team</td><td class="px-3 py-2 text-right font-mono text-white">{a['team']}</td><td class="px-3 py-2 text-right font-mono text-white">{b['team']}</td></tr>
<tr><td class="px-3 py-2 text-slate-400">Bye Week</td><td class="px-3 py-2 text-right font-mono text-white">{a.get('bye', '—')}</td><td class="px-3 py-2 text-right font-mono text-white">{b.get('bye', '—')}</td></tr>
<tr><td class="px-3 py-2 text-slate-400">Draft Value Score</td><td class="px-3 py-2 text-right font-mono {"text-brand-light" if a_val > b_val else ""}">{a_val}</td><td class="px-3 py-2 text-right font-mono {"text-accent" if b_val > a_val else ""}">{b_val}</td></tr>
</tbody></table>
<div class="mt-4 text-center text-sm text-slate-400">Advantage: <span class="text-white font-medium">{adv}</span> (picked {diff} spots earlier)</div>
</div>'''
    return html

@app.get("/nba", response_class=HTMLResponse)
def nba_page(request: Request):
    return templates.TemplateResponse(request, "sport_coming_soon.html", {
        **_ctx(request, "nba"), "sport_name": "NBA", "sport_icon": "🏀",
    })

@app.get("/cfb", response_class=HTMLResponse)
def cfb_page(request: Request):
    return templates.TemplateResponse(request, "sport_coming_soon.html", {
        **_ctx(request, "cfb"), "sport_name": "College Football", "sport_icon": "🏈",
    })

@app.get("/cbb", response_class=HTMLResponse)
def cbb_page(request: Request):
    return templates.TemplateResponse(request, "sport_coming_soon.html", {
        **_ctx(request, "cbb"), "sport_name": "College Basketball", "sport_icon": "🏀",
    })

@app.get("/mlb", response_class=HTMLResponse)
def mlb_page(request: Request):
    return templates.TemplateResponse(request, "sport_coming_soon.html", {
        **_ctx(request, "mlb"), "sport_name": "MLB", "sport_icon": "⚾",
    })

@app.get("/nhl", response_class=HTMLResponse)
def nhl_page(request: Request):
    return templates.TemplateResponse(request, "sport_coming_soon.html", {
        **_ctx(request, "nhl"), "sport_name": "NHL", "sport_icon": "🏒",
    })

@app.get("/mls", response_class=HTMLResponse)
def mls_page(request: Request):
    return templates.TemplateResponse(request, "sport_coming_soon.html", {
        **_ctx(request, "mls"), "sport_name": "MLS", "sport_icon": "⚽",
    })

@app.get("/odds", response_class=HTMLResponse)
def odds_redirect(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/bets", status_code=301)

@app.get("/robots.txt")
def robots_txt():
    return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /static/\n", media_type="text/plain")

@app.get("/sitemap.xml")
def sitemap_xml():
    base = "https://scoreedge.onrender.com"
    pages = ["", "/players", "/bets", "/rankings", "/nba", "/cfb", "/cbb", "/mlb", "/nhl", "/mls"]
    urls = "\n".join(f'  <url><loc>{base}{p}</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>' for p in pages)
    xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}\n</urlset>'
    return PlainTextResponse(xml, media_type="application/xml")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(request, "404.html", {
        **_ctx(request, "nfl"), "active_page": "404"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8512)

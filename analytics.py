from __future__ import annotations
import pandas as pd
import numpy as np

POSITION_MAP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "K": "K", "DEF": "DEF", "DST": "DEF",
}

def pos_group(pos: str) -> str:
    return POSITION_MAP.get(str(pos).upper(), "FLEX")

FANTASY_PTS_PPR = {
    "passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 0.1, "rushing_tds": 6,
    "receiving_yards": 0.1, "receiving_tds": 6, "receptions": 1,
    "rushing_fumbles_lost": -2, "receiving_fumbles_lost": -2, "sack_fumbles_lost": -2,
    "passing_2pt_conversions": 2, "rushing_2pt_conversions": 2, "receiving_2pt_conversions": 2,
    "special_teams_tds": 6,
}

FANTASY_PTS_STD = {k: v for k, v in FANTASY_PTS_PPR.items()}
FANTASY_PTS_STD["receptions"] = 0

FANTASY_PTS_HALF = {k: v for k, v in FANTASY_PTS_PPR.items()}
FANTASY_PTS_HALF["receptions"] = 0.5

def compute_fantasy_points(df: pd.DataFrame, scoring: str = "ppr") -> pd.Series:
    pts_map = {"ppr": FANTASY_PTS_PPR, "standard": FANTASY_PTS_STD, "half_ppr": FANTASY_PTS_HALF}[scoring]
    total = pd.Series(0.0, index=df.index)
    for col, mult in pts_map.items():
        if col in df.columns:
            total += pd.to_numeric(df[col], errors="coerce").fillna(0) * mult
    return total.round(2)

def compute_per_game(df: pd.DataFrame, stat_cols: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    if stat_cols is None:
        stat_cols = [c for c in ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                                  "receiving_yards", "receiving_tds", "receptions", "interceptions",
                                  "sacks", "fantasy_points"] if c in df.columns]
    games = result["games"].replace(0, 1) if "games" in result.columns else 1
    for col in stat_cols:
        if col in result.columns:
            result[f"{col}_pg"] = (pd.to_numeric(result[col], errors="coerce").fillna(0) / games).round(2)
    return result

def compute_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "passing_yards" in result.columns and "attempts" in result.columns:
        att = pd.to_numeric(result["attempts"], errors="coerce").replace(0, np.nan)
        result["yards_per_attempt"] = (pd.to_numeric(result["passing_yards"], errors="coerce") / att).round(2)
    if "rushing_yards" in result.columns and "carries" in result.columns:
        car = pd.to_numeric(result["carries"], errors="coerce").replace(0, np.nan)
        result["yards_per_carry"] = (pd.to_numeric(result["rushing_yards"], errors="coerce") / car).round(2)
    if "receiving_yards" in result.columns and "receptions" in result.columns:
        rec = pd.to_numeric(result["receptions"], errors="coerce").replace(0, np.nan)
        result["yards_per_reception"] = (pd.to_numeric(result["receiving_yards"], errors="coerce") / rec).round(2)
    return result

def compute_rankings(df: pd.DataFrame, scoring: str = "ppr") -> pd.DataFrame:
    result = df.copy()
    result["fantasy_pts"] = compute_fantasy_points(result, scoring)
    result = result.sort_values("fantasy_pts", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result

def get_player_stats_summary(stats: pd.DataFrame, player_id: str) -> dict:
    if stats.empty:
        return {}
    p = stats[stats["player_id"] == player_id] if "player_id" in stats.columns else pd.DataFrame()
    if p.empty:
        return {}
    numeric_cols = p.select_dtypes(include=[np.number]).columns
    return {
        "games_played": len(p),
        "seasons": int(p["season"].nunique()) if "season" in p.columns else 1,
        "totals": p[numeric_cols].sum().to_dict(),
        "averages": p[numeric_cols].mean().to_dict(),
    }

def compute_trade_value(player_a_stats: dict, player_b_stats: dict) -> dict:
    a_pts = player_a_stats.get("averages", {}).get("fantasy_pts", 0)
    b_pts = player_b_stats.get("averages", {}).get("fantasy_pts", 0)
    diff = a_pts - b_pts
    advantage = "Side A" if diff > 0 else "Side B" if diff < 0 else "Even"
    return {
        "player_a_avg_pts": round(a_pts, 2),
        "player_b_avg_pts": round(b_pts, 2),
        "difference": round(abs(diff), 2),
        "advantage": advantage,
    }

def compute_similarity(df: pd.DataFrame, target_id: str, feat_cols: list[str] | None = None, top_n: int = 5) -> list[dict]:
    if df.empty or "player_id" not in df.columns:
        return []
    if feat_cols is None:
        feat_cols = [c for c in ["fantasy_pts", "passing_yards", "rushing_yards", "receiving_yards",
                                  "passing_tds", "rushing_tds", "receiving_tds"] if c in df.columns]
    if not feat_cols:
        return []
    work = df[feat_cols].copy().fillna(0)
    if target_id not in df["player_id"].values:
        return []
    target_idx = df[df["player_id"] == target_id].index[0]
    target_vec = work.loc[target_idx].values.astype(float)
    norms = work.values.astype(float)
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return []
    sims = []
    for i, row in enumerate(norms):
        if i == target_idx:
            continue
        norm = np.linalg.norm(row)
        if norm == 0:
            continue
        sim = float(np.dot(target_vec, row) / (target_norm * norm))
        sims.append({"player_id": df.iloc[i]["player_id"],
                      "player_name": df.iloc[i].get("player_name", ""),
                      "similarity": round(sim, 4)})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return sims[:top_n]

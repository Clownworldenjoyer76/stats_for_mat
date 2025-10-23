#!/usr/bin/env python3
"""
Player Projections Builder

Reads:
  data/raw/stats/passing.csv
  data/raw/stats/rushing.csv
  data/processed/games.csv
  data/team_stadiums.csv
  data/processed/nfl_unified_with_metrics.csv

Writes:
  data/processed/player_projections.csv
"""

from pathlib import Path
import pandas as pd

# ---------- Paths ----------
P_PASS   = Path("data/raw/stats/passing.csv")
P_RUSH   = Path("data/raw/stats/rushing.csv")
P_GAMES  = Path("data/processed/games.csv")
P_TEAMS  = Path("data/team_stadiums.csv")
P_MET    = Path("data/processed/nfl_unified_with_metrics.csv")
P_OUT    = Path("data/processed/player_projections.csv")

# ---------- Import weather_bundle from your team script (with a safe fallback) ----------
def _fallback_weather_bundle(temp_f, wind_mph, precip_in, dome):
    if str(dome).strip().lower() == "yes":
        return dict(pass_mult=1.0, rush_mult=1.0, to_delta=0.0,
                    rz_mult=1.0, drives_mult=1.0, fg_pct_delta=0.0, net_punt_delta=0.0)

    def _flt(x, default):
        try:
            return float(x)
        except Exception:
            return default

    t = _flt(temp_f, 70.0)
    w = _flt(wind_mph, 0.0)
    r = _flt(precip_in, 0.0)

    if r > 0 or w >= 20:
        sev = "heavy"
    elif w >= 10:
        sev = "moderate"
    elif w >= 5:
        sev = "light"
    elif t < 35:
        sev = "cold"
    else:
        sev = "none"

    table = {
        "none":     (1.00, 1.00, 0.00, 1.00, 1.00,  0.00,  0.0),
        "light":    (0.98, 1.00, 0.00, 1.00, 1.00, -0.01, -0.1),
        "moderate": (0.95, 1.03, 0.10, 0.98, 0.99, -0.02, -0.3),
        "heavy":    (0.92, 1.06, 0.30, 0.95, 0.97, -0.05, -0.8),
        "cold":     (0.97, 1.02, 0.10, 0.98, 0.99, -0.02, -0.2),
    }
    pm, rm, to, rz, dr, fg, np = table[sev]
    return dict(pass_mult=pm, rush_mult=rm, to_delta=to, rz_mult=rz,
                drives_mult=dr, fg_pct_delta=fg, net_punt_delta=np)

try:
    from src.predict_outcomes import weather_bundle  # type: ignore
except Exception:
    weather_bundle = _fallback_weather_bundle

# ---------- Helpers ----------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def _normalize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, variants in mapping.items():
        for v in variants:
            if v.lower() in lower_map:
                rename[lower_map[v.lower()]] = canon
                break
    return df.rename(columns=rename)

def _ensure_columns(df: pd.DataFrame, required: list, src_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{src_name} missing required columns: {missing}")

# ---------- Load ----------
pass_df = _read_csv(P_PASS)
rush_df = _read_csv(P_RUSH)
games   = _read_csv(P_GAMES)
teams   = _read_csv(P_TEAMS)
metrics = _read_csv(P_MET)

# ---------- Normalize column names to canonical ----------
# Your passing.csv uses PYDS (yards) and PTD (pass TDs) — mapped below.
pass_df = _normalize_columns(pass_df, {
    "PLAYER":   ["PLAYER","Player","player","Name"],
    "TEAM":     ["TEAM","Team","team"],
    "PASS_YDS": ["PYDS","PASS_YDS","PassYds","Pass_Yds","PASSING_YDS","YDS"],
    "PASS_TD":  ["PTD","PASS_TD","PassTD","TD","PASSING_TD"],  # <-- includes PTD
    "INT":      ["INT","Interceptions","PASS_INT","PICKS"],
})

# Your rushing.csv uses RYDS (yards) and RTD (rush TDs) — mapped below.
rush_df = _normalize_columns(rush_df, {
    "PLAYER":   ["PLAYER","Player","player","Name"],
    "TEAM":     ["TEAM","Team","team"],
    "RUSH_YDS": ["RYDS","RUSH_YDS","RushYds","RushingYds","RUSHING_YDS","YDS"],
    "RUSH_TD":  ["RTD","RUSH_TD","RushTD","TD","RUSHING_TD"],  # <-- includes RTD
    "CARRIES":  ["RATT","CARRIES","Att","ATT","Attempts"],      # your file has RATT
})

metrics = _normalize_columns(metrics, {
    "TEAM":            ["TEAM","Team","team"],
    "defense_PYDS/G":  ["defense_PYDS/G","DEF_PYDS_G","defense_PYDS_g"],
    "defense_RYDS/G":  ["defense_RYDS/G","DEF_RYDS_G","defense_RYDS_g"],
})

# ---------- Validate minimal requirements ----------
_ensure_columns(pass_df, ["PLAYER","TEAM","PASS_YDS","PASS_TD","INT"], "passing.csv")
_ensure_columns(rush_df, ["PLAYER","TEAM","RUSH_YDS","RUSH_TD","CARRIES"], "rushing.csv")
_ensure_columns(metrics, ["TEAM","defense_PYDS/G","defense_RYDS/G"], "nfl_unified_with_metrics.csv")

# ---------- Per-player averages ----------
pass_avg = pass_df.groupby(["PLAYER","TEAM"], as_index=False)[["PASS_YDS","PASS_TD","INT"]].mean()
rush_avg = rush_df.groupby(["PLAYER","TEAM"], as_index=False)[["RUSH_YDS","RUSH_TD","CARRIES"]].mean()

# ---------- Build weekly matchup map (team -> opponent + weather from home site) ----------
home_side = games[["home_team","away_team","temp_f","wind_mph","precip_in"]].copy()
home_side["team"] = home_side["home_team"]
home_side["opponent"] = home_side["away_team"]
home_side["is_home"] = 1

away_side = games[["home_team","away_team","temp_f","wind_mph","precip_in"]].copy()
away_side["team"] = away_side["away_team"]
away_side["opponent"] = away_side["home_team"]
away_side["is_home"] = 0

matchups = pd.concat([home_side, away_side], ignore_index=True)[["team","opponent","is_home","temp_f","wind_mph","precip_in"]]

# Map home stadium's dome flag to both teams in that game
teams_dome = teams[["TEAM","dome"]].copy()
home_dome = games[["home_team"]].merge(teams_dome, left_on="home_team", right_on="TEAM", how="left").rename(columns={"dome":"home_dome"})
home_dome_map = pd.concat([
    games[["home_team","away_team"]].assign(team=games["home_team"]).merge(home_dome[["home_team","home_dome"]], on="home_team", how="left"),
    games[["home_team","away_team"]].assign(team=games["away_team"]).merge(home_dome[["home_team","home_dome"]], on="home_team", how="left"),
], ignore_index=True)[["team","home_dome"]].drop_duplicates()
matchups = matchups.merge(home_dome_map, on="team", how="left").rename(columns={"home_dome":"dome"})

# ---------- Merge opponent defensive context ----------
metrics_ctx = metrics[["TEAM","defense_PYDS/G","defense_RYDS/G"]].rename(columns={"TEAM":"opponent"})
matchups = matchups.merge(metrics_ctx, on="opponent", how="left")

# ---------- League averages for scaling ----------
league_avg_pass_def = metrics["defense_PYDS/G"].mean()
league_avg_rush_def = metrics["defense_RYDS/G"].mean()

def def_adj_pass(opp_def):
    if pd.isna(opp_def) or pd.isna(league_avg_pass_def) or league_avg_pass_def == 0:
        return 1.0
    return 1 - ((opp_def - league_avg_pass_def) / league_avg_pass_def) * 0.6

def def_adj_rush(opp_def):
    if pd.isna(opp_def) or pd.isna(league_avg_rush_def) or league_avg_rush_def == 0:
        return 1.0
    return 1 - ((opp_def - league_avg_rush_def) / league_avg_rush_def) * 0.5

# ---------- Build unified player base and attach matchup row ----------
players = pd.merge(pass_avg, rush_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(matchups, left_on="TEAM", right_on="team", how="left")
players = players[players["opponent"].notna()].copy()

# ---------- Apply multipliers per player ----------
rows = []
for _, r in players.iterrows():
    player = r["PLAYER"]
    team   = r["TEAM"]
    opp    = r["opponent"]
    is_home = int(r.get("is_home", 0) or 0)
    home_mult = 1.02 if is_home == 1 else 0.98

    wb = weather_bundle(r.get("temp_f"), r.get("wind_mph"), r.get("precip_in"), r.get("dome"))

    pass_def_mult = def_adj_pass(r.get("defense_PYDS/G"))
    rush_def_mult = def_adj_rush(r.get("defense_RYDS/G"))

    pass_yds = float(r.get("PASS_YDS") or 0.0)
    pass_td  = float(r.get("PASS_TD")  or 0.0)
    picks    = float(r.get("INT")      or 0.0)

    rush_yds = float(r.get("RUSH_YDS") or 0.0)
    rush_td  = float(r.get("RUSH_TD")  or 0.0)

    proj_pass_yds = pass_yds * wb["pass_mult"] * home_mult * pass_def_mult
    proj_pass_td  = pass_td  * wb["pass_mult"] * home_mult * pass_def_mult
    proj_int      = picks    * (1.00 + max(0.0, wb["to_delta"]))

    proj_rush_yds = rush_yds * wb["rush_mult"] * home_mult * rush_def_mult
    proj_rush_td  = rush_td  * wb["rush_mult"] * home_mult * rush_def_mult

    rows.append({
        "player": player,
        "team": team,
        "opponent": opp,
        "is_home": is_home,
        "temp_f": r.get("temp_f"),
        "wind_mph": r.get("wind_mph"),
        "dome": r.get("dome"),
        "proj_pass_yds": round(proj_pass_yds, 1),
        "proj_pass_td": round(proj_pass_td, 2),
        "proj_int": round(proj_int, 2),
        "proj_rush_yds": round(proj_rush_yds, 1),
        "proj_rush_td": round(proj_rush_td, 2),
    })

out_df = pd.DataFrame(rows)
P_OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(P_OUT, index=False)
print(f"Wrote {P_OUT} with {len(out_df)} player rows.")

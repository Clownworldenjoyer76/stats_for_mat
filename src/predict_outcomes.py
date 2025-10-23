#!/usr/bin/env python3
"""
Predict NFL game outcomes using Projection_System formulas and your processed data.

Inputs
------
data/team_stadiums.csv
  Columns: TEAM, STADIUM, LATITUDE, LONGITUDE, dome

data/processed/games.csv
  Columns: Day, Date, Time, away_team, home_team, Stadium, temp_f, wind_mph, wind_dir,
           precip_in, chance_of_rain, Odds, favorite, total, game_id

data/processed/nfl_unified_with_metrics.csv
  Team-level metrics used for offense/defense, turnovers, special teams.

Output
------
data/processed/projections.csv
  Columns:
    game_id, away_team, home_team,
    pred_away_pts, pred_home_pts, point_diff, win_prob_home,
    pass_yards_away, pass_yards_home, rush_yards_away, rush_yards_home

Implements (aligned with Projection_System.pdf):
- Weather Adjustment (ignored for dome='yes' on home team).
- Home Field Advantage baseline (+2.0).
- Turnover differential adjustment (Poisson-inspired).
- Special teams EPA-style tweak (FG% and net punt diff).
- Market Blend: 0.6 * Model + 0.4 * Vegas implied totals.
"""
import math
from pathlib import Path
import pandas as pd

# ---------- Paths ----------
P_GAMES   = Path("data/processed/games.csv")
P_TEAMS   = Path("data/team_stadiums.csv")
P_METRICS = Path("data/processed/nfl_unified_with_metrics.csv")
P_OUT     = Path("data/processed/projections.csv")

# ---------- Load ----------
games   = pd.read_csv(P_GAMES)
stadia  = pd.read_csv(P_TEAMS)
metrics = pd.read_csv(P_METRICS)

# Index helpers
M = metrics.set_index("TEAM")
S = stadia.set_index("TEAM")

# Safe getters
def g(df, team, col, default=0.0):
    try:
        v = df.at[team, col]
        return float(v) if pd.notna(v) else default
    except Exception:
        return default

# Vegas implied points from total & spread
def vegas_implied(total, spread):
    """
    Standard split:
      home = total/2 - spread/2
      away = total - home
    'spread' should be the home-line (negative means home favored by |spread|).
    """
    if pd.isna(total) or pd.isna(spread):
        return None, None
    try:
        t = float(total)
        s = float(spread)
    except Exception:
        return None, None
    home = t/2 - s/2
    away = t - home
    return away, home

# Build a home spread where negative = home favored by |spread|
def compute_home_spread(row):
    odds = row.get("Odds")
    fav  = row.get("favorite")
    if pd.isna(odds) or pd.isna(fav):
        return None
    try:
        sp = float(str(odds).replace("+","").strip())
    except Exception:
        return None
    return -abs(sp) if fav == row.get("home_team") else +abs(sp)

# Weather passing multiplier (Projection_System weather rules)
def weather_pass_multiplier(temp_f, wind_mph, precip_in, dome_flag: str):
    # Dome? No weather adjustment
    if isinstance(dome_flag, str) and dome_flag.strip().lower() == "yes":
        return 1.0

    mult = 1.0
    # Temp ref 70°F, coefficient 0.004 per degree (PassYds_Adjust in Projection_System)  [oai_citation:0‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    try:
        if pd.notna(temp_f):
            mult *= (1 - 0.004 * (70 - float(temp_f)))
    except Exception:
        pass
    # Wind tiers (10–20: -1.5%, 20+: -3%)  [oai_citation:1‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    try:
        w = float(wind_mph) if pd.notna(wind_mph) else 0.0
        if 10 <= w < 20:
            mult *= (1 - 0.015)
        elif w >= 20:
            mult *= (1 - 0.03)
    except Exception:
        pass
    # Precip: −6% on passing yardage  [oai_citation:2‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    try:
        r = float(precip_in) if pd.notna(precip_in) else 0.0
        if r > 0:
            mult *= (1 - 0.06)
    except Exception:
        pass
    # Avoid extreme boosts/drops
    return max(0.75, mult)

# Logistic from point diff → home win probability (calibrated margin scale)
def win_prob_from_diff(point_diff):
    return 1/(1+math.exp(-point_diff/6.5))

# Model points from offense/defense yardage (yards-to-points ~ 15 yds/pt)
YDS_PER_POINT = 15.0

def model_points(off_team, def_team, weather_mult):
    # Off passing vs opponent pass defense (Projection_System “Behind-the-Scenes Logic”)  [oai_citation:3‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    off_pass = g(M, off_team, "passing_PYDS/G", 0.0)
    def_pass = g(M, def_team, "defense_PYDS/G", 0.0)
    pass_yards = (off_pass + def_pass) / 2.0
    pass_yards *= weather_mult

    # Off rushing vs opponent rush defense (same logic)
    off_rush = g(M, off_team, "rushing_RYDS/G", 0.0)
    def_rush = g(M, def_team, "defense_RYDS/G", 0.0)
    rush_yards = (off_rush + def_rush) / 2.0

    # Base points from yards (Yards→Points translation concept)  [oai_citation:4‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    base_pts = (pass_yards + rush_yards) / YDS_PER_POINT

    # Turnover differential (Poisson-inspired expected loss/gain)  [oai_citation:5‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    team_give = g(M, off_team, "turnovers_GA", 0.0) / max(1.0, g(M, off_team, "turnovers_GP", 1.0))
    opp_take  = (g(M, def_team, "turnovers_DEF_INT", 0.0) + g(M, def_team, "turnovers_FUMR", 0.0)) / \
                max(1.0, g(M, def_team, "turnovers_GP", 1.0))
    lambda_to = (team_give + opp_take)/2.0
    to_adj_pts = 3.2 * (lambda_to - 1.4) * 0.6  # small tilt

    # Special teams adjustment (FG% + Net punt diff)  [oai_citation:6‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    fg_off = g(M, off_team, "kicking_FG_PCT", 0.0)
    fg_def = g(M, def_team, "kicking_FG_PCT", 0.0)
    net_punt_off = g(M, off_team, "kickoffs_punts_NET_AVG", 0.0)
    net_punt_def = g(M, def_team, "kickoffs_punts_NET_AVG", 0.0)
    st_adj = 0.02 * (fg_off - fg_def) + 0.04 * (net_punt_off - net_punt_def)

    return base_pts + to_adj_pts + st_adj, pass_yards, rush_yards

# ---------- Compute per game ----------
rows = []
for _, grow in games.iterrows():
    home = grow["home_team"]
    away = grow["away_team"]

    # Dome check based on home team (your rule)
    dome_flag = S.at[home, "dome"] if home in S.index and "dome" in S.columns else "no"

    # Weather mult from games.csv (ignored if dome='yes')
    w_mult = weather_pass_multiplier(grow.get("temp_f"), grow.get("wind_mph"),
                                     grow.get("precip_in"), dome_flag)

    # Model points (each side vs opponent defense)
    away_model_pts, away_pass_yds, away_rush_yds = model_points(away, home, w_mult)
    home_model_pts, home_pass_yds, home_rush_yds = model_points(home, away, w_mult)

    # Home Field Advantage baseline (+2.0)  [oai_citation:7‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    home_model_pts += 2.0

    # Vegas implied split
    home_spread = compute_home_spread(grow)
    v_away, v_home = vegas_implied(grow.get("total"), home_spread)

    # Market blend (0.6 model + 0.4 vegas). Fallback to model-only if vegas missing.  [oai_citation:8‡Projection_System.pdf](sediment://file_000000006ab861f7866beb9956e6c9f5)
    if v_away is not None and v_home is not None:
        away_pts = 0.6*away_model_pts + 0.4*v_away
        home_pts = 0.6*home_model_pts + 0.4*v_home
    else:
        away_pts, home_pts = away_model_pts, home_model_pts

    # Clip to plausible ranges
    away_pts = max(6.0, float(away_pts))
    home_pts = max(6.0, float(home_pts))

    diff = home_pts - away_pts
    winp_home = 1/(1+math.exp(-diff/6.5))

    rows.append({
        "game_id": grow["game_id"],
        "away_team": away,
        "home_team": home,
        "pred_away_pts": round(away_pts, 1),
        "pred_home_pts": round(home_pts, 1),
        "point_diff": round(diff, 1),
        "win_prob_home": round(winp_home, 3),
        "pass_yards_away": int(round(away_pass_yds)),
        "pass_yards_home": int(round(home_pass_yds)),
        "rush_yards_away": int(round(away_rush_yds)),
        "rush_yards_home": int(round(home_rush_yds)),
    })

out_df = pd.DataFrame(rows)
P_OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(P_OUT, index=False)
print(f"Wrote {P_OUT} with {len(out_df)} games.")

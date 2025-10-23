
#!/usr/bin/env python3
"""
Predict NFL game outcomes using Projection_System formulas and your processed data,
with extended weather effects beyond passing (bundle applies to pass, rush, TOs, RZ proxy,
drives/pace, and special teams deltas).

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

# ------------- Weather Bundle (extended factors) -------------
def weather_bundle(temp_f, wind_mph, precip_in, dome):
    """Return multipliers and deltas for multiple weather pathways. Dome => neutral effects."""
    if str(dome).strip().lower() == "yes":
        return dict(pass_mult=1.0, rush_mult=1.0, to_delta=0.0,
                    rz_mult=1.0, drives_mult=1.0, fg_pct_delta=0.0, net_punt_delta=0.0)
    # Coerce
    try: t = float(temp_f) if pd.notna(temp_f) else 70.0
    except Exception: t = 70.0
    try: w = float(wind_mph) if pd.notna(wind_mph) else 0.0
    except Exception: w = 0.0
    try: r = float(precip_in) if pd.notna(precip_in) else 0.0
    except Exception: r = 0.0

    # Severity logic
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

# Logistic from point diff → home win probability
def win_prob_from_diff(point_diff):
    return 1/(1+math.exp(-point_diff/6.5))

# Model points from offense/defense yardage (yards-to-points ~ 15 yds/pt)
YDS_PER_POINT = 15.0

def model_points(off_team, def_team, wb):
    """Return (points, pass_yards, rush_yards) for offense vs defense, including weather bundle."""
    # Passing vs pass defense
    off_pass = g(M, off_team, "passing_PYDS/G", 0.0)
    def_pass = g(M, def_team, "defense_PYDS/G", 0.0)
    pass_yards = (off_pass + def_pass) / 2.0
    pass_yards *= wb["pass_mult"]

    # Rushing vs rush defense
    off_rush = g(M, off_team, "rushing_RYDS/G", 0.0)
    def_rush = g(M, def_team, "defense_RYDS/G", 0.0)
    rush_yards = (off_rush + def_rush) / 2.0
    rush_yards *= wb["rush_mult"]

    # Base points from yards
    base_pts = (pass_yards + rush_yards) / YDS_PER_POINT

    # Turnover differential (Poisson-inspired) with weather bump
    team_give = g(M, off_team, "turnovers_GA", 0.0) / max(1.0, g(M, off_team, "turnovers_GP", 1.0))
    opp_take  = (g(M, def_team, "turnovers_DEF_INT", 0.0) + g(M, def_team, "turnovers_FUMR", 0.0)) / \
                max(1.0, g(M, def_team, "turnovers_GP", 1.0))
    lambda_to = (team_give + opp_take)/2.0 + wb["to_delta"]
    to_adj_pts = 3.2 * (lambda_to - 1.4) * 0.6  # small tilt

    # Special teams adjustment (FG% + Net punt), include weather deltas on offense side
    fg_off = (g(M, off_team, "kicking_FG_PCT", 0.0) + wb["fg_pct_delta"])
    fg_def = g(M, def_team, "kicking_FG_PCT", 0.0)
    net_punt_off = (g(M, off_team, "kickoffs_punts_NET_AVG", 0.0) + wb["net_punt_delta"])
    net_punt_def = g(M, def_team, "kickoffs_punts_NET_AVG", 0.0)
    st_adj = 0.02 * (fg_off - fg_def) + 0.04 * (net_punt_off - net_punt_def)

    pts = base_pts + to_adj_pts + st_adj
    return pts, pass_yards, rush_yards

# ---------- Compute per game ----------
rows = []
for _, grow in games.iterrows():
    home = grow["home_team"]
    away = grow["away_team"]

    # Dome check (home team drives the environment)
    dome_flag = S.at[home, "dome"] if home in S.index and "dome" in S.columns else "no"

    # Build weather bundle
    wb = weather_bundle(grow.get("temp_f"), grow.get("wind_mph"), grow.get("precip_in"), dome_flag)

    # Model points (each side vs opponent defense), bundle already adjusts pass/rush/TO/ST
    away_model_pts, away_pass_yds, away_rush_yds = model_points(away, home, wb)
    home_model_pts, home_pass_yds, home_rush_yds = model_points(home, away, wb)

    # Home field advantage (+2.0)
    home_model_pts += 2.0

    # Apply a small aggregate weather scaling to total efficiency (drives + mixed yardage)
    mix_mult = wb["drives_mult"] * (0.5*wb["pass_mult"] + 0.5*wb["rush_mult"])
    away_model_pts *= mix_mult
    home_model_pts *= mix_mult

    # Vegas implied split and market blend
    home_spread = compute_home_spread(grow)
    v_away, v_home = vegas_implied(grow.get("total"), home_spread)
    if v_away is not None and v_home is not None:
        away_pts = 0.6*away_model_pts + 0.4*v_away
        home_pts = 0.6*home_model_pts + 0.4*v_home
    else:
        away_pts, home_pts = away_model_pts, home_model_pts

    # Clip to plausible ranges
    away_pts = max(6.0, float(away_pts))
    home_pts = max(6.0, float(home_pts))

    diff = home_pts - away_pts
    winp_home = win_prob_from_diff(diff)

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

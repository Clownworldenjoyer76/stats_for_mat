#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# Paths
PROCESSED_DIR = "data/processed/stats"
OUT_PATH = "data/processed/player_projections.csv"

# Expected input files
FILES = {
    "passing": "passing_per_game.csv",
    "rushing": "rushing_per_game.csv",
    "receiving": "receiving_per_game.csv",
    "returning": "returning_per_game.csv",
    "kicking": "kicking_per_game.csv",
    "kickoffs_punts": "kickoffs_punts_per_game.csv",
    "defensive": "defensive_per_game.csv",
    "scoring": "scoring_per_game.csv"
}

# Adjustment factors (can later be customized)
HOME_ADV = 1.05
AWAY_PENALTY = 0.95
BAD_WEATHER_PENALTY = 0.9

def load_csv(name):
    path = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    required = {"PLAYER", "TEAM", "GP"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df

def apply_adjustments(df, is_home=1, weather_good=True):
    """Apply simple multipliers for home/away and weather."""
    if is_home:
        df = df * HOME_ADV
    else:
        df = df * AWAY_PENALTY
    if not weather_good:
        df = df * BAD_WEATHER_PENALTY
    return df

def main():
    # Load all per-game stats
    dfs = {k: load_csv(v) for k, v in FILES.items()}

    # Start base from scoring for TD probability
    base = dfs["scoring"][["PLAYER", "TEAM", "TD"]].copy()
    base["TD_PG"] = dfs["scoring"]["TD"] / dfs["scoring"]["GP"]
    base["expected_anytime_td"] = base["TD_PG"].clip(lower=0)
    base["anytime_td_prob"] = 1 - np.exp(-base["expected_anytime_td"])

    # Merge in other stats
    def merge_stats(src, cols):
        for c in cols:
            if c in src.columns:
                base[c] = src.set_index("PLAYER")[c]
    
    merge_stats(dfs["passing"], ["PYDS", "PTD", "INT"])
    merge_stats(dfs["rushing"], ["RYDS", "RTD"])
    merge_stats(dfs["receiving"], ["RECYDS", "RECTD", "TGT", "REC", "YAC"])
    merge_stats(dfs["returning"], ["K_RET_YDS", "K_RET_TD", "P_RET_YDS", "P_RET_TD"])
    merge_stats(dfs["kicking"], ["FGM", "FGA", "XPM", "XPA", "FG_PCT"])
    merge_stats(dfs["defensive"], ["TCKL", "SCK", "DEF_INT"])
    merge_stats(dfs["kickoffs_punts"], ["TB %", "NET AVG", "P-AVG", "K-AVG"])

    # Rename to projection-style column names
    base = base.rename(columns={
        "PYDS": "proj_pass_yds",
        "PTD": "proj_pass_td",
        "INT": "proj_int",
        "RYDS": "proj_rush_yds",
        "RTD": "proj_rush_td",
        "RECYDS": "proj_rec_yards",
        "RECTD": "proj_rec_td",
        "TGT": "proj_targets",
        "REC": "proj_rec",
        "YAC": "proj_yac",
        "K_RET_YDS": "proj_kr_yds",
        "K_RET_TD": "proj_kr_td",
        "P_RET_YDS": "proj_pr_yds",
        "P_RET_TD": "proj_pr_td",
        "FGM": "proj_fgm",
        "FGA": "proj_fga",
        "XPM": "proj_xpm",
        "XPA": "proj_xpa",
        "FG_PCT": "proj_fg_pct",
        "TCKL": "proj_tackles",
        "SCK": "proj_sacks",
        "DEF_INT": "proj_def_int",
        "TB %": "proj_tb_pct",
        "NET AVG": "proj_net_avg",
        "P-AVG": "proj_punt_avg",
        "K-AVG": "proj_kick_avg"
    })

    # Add contextual fields
    base["opponent"] = ""
    base["is_home"] = 1
    base["temp_f"] = 72
    base["wind_mph"] = 5
    base["dome"] = "no"

    # Apply adjustments (demonstration only)
    adj_cols = [c for c in base.columns if c.startswith("proj_") and base[c].dtype != "object"]
    base[adj_cols] = apply_adjustments(base[adj_cols], is_home=1, weather_good=True)

    # Order columns
    ordered_cols = [
        "PLAYER","TEAM","opponent","is_home","temp_f","wind_mph","dome",
        "proj_pass_yds","proj_pass_td","proj_int","proj_rush_yds","proj_rush_td",
        "proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac",
        "proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td",
        "proj_fgm","proj_fga","proj_xpm","proj_xpa","proj_fg_pct",
        "proj_tackles","proj_sacks","proj_def_int",
        "proj_tb_pct","proj_net_avg","proj_punt_avg","proj_kick_avg",
        "expected_anytime_td","anytime_td_prob"
    ]
    for c in ordered_cols:
        if c not in base.columns:
            base[c] = np.nan
    base = base[ordered_cols].sort_values(["TEAM","PLAYER"], na_position="last")

    # Output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    base.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(base):,} rows.")

if __name__ == "__main__":
    main()

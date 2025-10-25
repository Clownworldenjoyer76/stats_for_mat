# src/player_projections.py
#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd

PROC_DIR = "data/processed/stats"
OUT_PATH = "data/processed/player_projections.csv"

# Optional inputs (if missing, adjustments default to neutral = no change)
GAMES_PATH = "data/raw/games.csv"  # columns: player,team,opponent,is_home,temp_f,wind_mph,dome
DEF_FCTRS = "data/raw/team_defense_factors.csv"  # columns: team,pass_def,rush_def,rec_def,kick_def (1.00 = league avg)

# -------------------------
# Helpers (minimal, robust)
# -------------------------
def load_csv_maybe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def nz(x):
    """numeric with NaN -> 0"""
    return pd.to_numeric(x, errors="coerce").fillna(0.0)

def ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# -------------------------
# Load per-game baselines
# -------------------------
receiving = load_csv_maybe(os.path.join(PROC_DIR, "receiving_per_game.csv"))
rushing   = load_csv_maybe(os.path.join(PROC_DIR, "rushing_per_game.csv"))
passing   = load_csv_maybe(os.path.join(PROC_DIR, "passing_per_game.csv"))
returning = load_csv_maybe(os.path.join(PROC_DIR, "returning_per_game.csv"))
kicking   = load_csv_maybe(os.path.join(PROC_DIR, "kicking_per_game.csv"))
defense   = load_csv_maybe(os.path.join(PROC_DIR, "defensive_per_game.csv"))
kickpunts = load_csv_maybe(os.path.join(PROC_DIR, "kickoffs_punts_per_game.csv"))
scoring   = load_csv_maybe(os.path.join(PROC_DIR, "scoring_per_game.csv"))

# Normalize key identifiers
for df in [receiving, rushing, passing, returning, kicking, defense, kickpunts, scoring]:
    if not df.empty:
        if "PLAYER" in df.columns: df.rename(columns={"PLAYER":"player"}, inplace=True)
        if "Team"   in df.columns: df.rename(columns={"Team":"TEAM"}, inplace=True)
        if "TEAM"   in df.columns: df.rename(columns={"TEAM":"team"}, inplace=True)
        for c in df.columns:
            if c not in ("player","team"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------------
# Assemble base projection
# -------------------------
# Start with union of all players we can see
players = pd.Series(dtype=str)
for df in [receiving, rushing, passing, returning, kicking, defense, scoring]:
    if not df.empty and "player" in df.columns:
        players = pd.concat([players, df["player"]], ignore_index=True)
players = players.dropna().drop_duplicates()

base = pd.DataFrame({"player": players})

# Attach team (prefer receiving -> passing -> rushing -> returning -> kicking -> defense -> scoring)
def attach_team_pref(target, sources):
    target = target.copy()
    target["team"] = np.nan
    for src in sources:
        if src.empty or "player" not in src.columns or "team" not in src.columns:
            continue
        target = target.merge(src[["player","team"]], on="player", how="left", suffixes=("","_src"))
        target["team"] = target["team"].fillna(target["team_src"])
        target.drop(columns=[c for c in target.columns if c.endswith("_src")], inplace=True)
    return target

base = attach_team_pref(base, [receiving, passing, rushing, returning, kicking, defense, scoring])
base["team"] = base["team"].fillna("")

# Merge in per-game stats to standardized projection fields
# Passing
if not passing.empty:
    p = passing.rename(columns={
        "PYDS":"proj_pass_yds",
        "PTD":"proj_pass_td",
        "INT":"proj_int"
    })
    base = base.merge(p[["player","proj_pass_yds","proj_pass_td","proj_int"]], on="player", how="left")

# Rushing
if not rushing.empty:
    ru = rushing.rename(columns={
        "RYDS":"proj_rush_yds",
        "RTD":"proj_rush_td"
    })
    base = base.merge(ru[["player","proj_rush_yds","proj_rush_td"]], on="player", how="left")

# Receiving
if not receiving.empty:
    rc = receiving.rename(columns={
        "RECYDS":"proj_rec_yards",
        "RECTD":"proj_rec_td",
        "TGT":"proj_targets",
        "REC":"proj_rec",
        "YAC":"proj_yac"
    })
    use_cols = [c for c in ["player","proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac"] if c in rc.columns]
    base = base.merge(rc[use_cols], on="player", how="left")

# Returns
if not returning.empty:
    ret = returning.rename(columns={
        "K_RET_YDS":"proj_kr_yds",
        "K_RET_TD":"proj_kr_td",
        "P_RET_YDS":"proj_pr_yds",
        "P_RET_TD":"proj_pr_td"
    })
    use_cols = [c for c in ["player","proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td"] if c in ret.columns]
    base = base.merge(ret[use_cols], on="player", how="left")

# Kicking totals (leave as totals, not per-game probabilities)
if not kicking.empty:
    kk = kicking.rename(columns={
        "FGM":"proj_fgm",
        "FGA":"proj_fga",
        "XPM":"proj_xpm",
        "XPA":"proj_xpa",
        "FG_PCT":"proj_fg_pct"
    })
    use_cols = [c for c in ["player","proj_fgm","proj_fga","proj_xpm","proj_xpa","proj_fg_pct"] if c in kk.columns]
    base = base.merge(kk[use_cols], on="player", how="left")

# Defensive counting (light, optional)
if not defense.empty:
    df_ = defense.rename(columns={
        "SCK":"proj_sacks",
        "DEF_INT":"proj_def_int",
        "TCKL":"proj_tackles"
    })
    use_cols = [c for c in ["player","proj_tackles","proj_sacks","proj_def_int"] if c in df_.columns]
    base = base.merge(df_[use_cols], on="player", how="left")

# Kick/punt specialist rates (optional)
if not kickpunts.empty:
    # Try flexible column names
    kp = kickpunts.copy()
    rename_map = {}
    for src, dst in [
        ("TB %","proj_tb_pct"),
        ("TB%","proj_tb_pct"),
        ("TB_PCT","proj_tb_pct"),
        ("NET AVG","proj_net_avg"),
        ("NET_AVG","proj_net_avg"),
        ("P-AVG","proj_punt_avg"),
        ("P_AVG","proj_punt_avg"),
        ("K-AVG","proj_kick_avg"),
        ("K_AVG","proj_kick_avg"),
    ]:
        if src in kp.columns: rename_map[src] = dst
    kp.rename(columns=rename_map, inplace=True)
    keep = [c for c in ["player","proj_tb_pct","proj_net_avg","proj_punt_avg","proj_kick_avg"] if c in kp.columns]
    if keep:
        base = base.merge(kp[keep], on="player", how="left")

# Anytime TD (λ per game from scoring -> probability)
if not scoring.empty and {"player","TD","GP"}.issubset(scoring.columns):
    s = scoring.copy()
    s["td_pg"] = s["TD"] / s["GP"].replace(0, np.nan)
    s["td_pg"] = s["td_pg"].fillna(0.0)
    s["expected_anytime_td"] = s["td_pg"].clip(lower=0)
    s["anytime_td_prob"] = 1.0 - np.exp(-s["expected_anytime_td"])
    base = base.merge(s[["player","expected_anytime_td","anytime_td_prob"]], on="player", how="left")
else:
    base["expected_anytime_td"] = np.nan
    base["anytime_td_prob"] = np.nan

# -------------------------
# Match context (weather / H/A / opponent)
# -------------------------
games = load_csv_maybe(GAMES_PATH)
if not games.empty:
    # Normalize expected headers
    colmap = {
        "PLAYER":"player","Team":"team","TEAM":"team","OPPONENT":"opponent",
        "IS_HOME":"is_home","TEMP_F":"temp_f","WIND_MPH":"wind_mph","DOME":"dome"
    }
    gm = games.rename(columns={c: colmap.get(c, c) for c in games.columns})
    for col in ["is_home","temp_f","wind_mph"]:
        if col in gm.columns:
            gm[col] = pd.to_numeric(gm[col], errors="coerce")
    if "dome" in gm.columns:
        gm["dome"] = gm["dome"].astype(str).str.strip().str.lower()
    # Prefer player-level merge; if player missing, allow team-level fill
    base = base.merge(gm[["player","opponent","is_home","temp_f","wind_mph","dome"]],
                      on="player", how="left")
else:
    # If no games file, create neutral defaults
    base["opponent"] = ""
    base["is_home"] = 0
    base["temp_f"] = 60.0
    base["wind_mph"] = 5.0
    base["dome"] = "no"

# -------------------------
# Opponent strength factors
# -------------------------
def_factors = load_csv_maybe(DEF_FCTRS)
if not def_factors.empty:
    dfc = def_factors.copy()
    # standardize
    if "team" not in dfc.columns:
        # try TEAM
        if "TEAM" in dfc.columns:
            dfc.rename(columns={"TEAM":"team"}, inplace=True)
    for col in ["pass_def","rush_def","rec_def","kick_def"]:
        if col in dfc.columns:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce").fillna(1.0)
        else:
            dfc[col] = 1.0
    # Join opponent factors
    base = base.merge(dfc.rename(columns={"team":"opponent"})[
        ["opponent","pass_def","rush_def","rec_def","kick_def"]
    ], on="opponent", how="left")
else:
    base["pass_def"] = 1.0
    base["rush_def"] = 1.0
    base["rec_def"]  = 1.0
    base["kick_def"] = 1.0

# -------------------------
# Adjustments
# -------------------------
# Weather:
# - If dome == "yes": neutral (1.00)
# - Pass/kick slight penalties with cold (below 50F) and wind (>10 mph)
# - Rush slight boost in bad weather
def weather_factors(row):
    if str(row.get("dome","")).lower() == "yes":
        return 1.0, 1.0, 1.0  # pass, rush, kick
    temp = float(row.get("temp_f", 60.0) or 60.0)
    wind = float(row.get("wind_mph", 5.0) or 5.0)
    # Pass & Kick penalties
    cold_pen = max(0.0, (50.0 - temp) * 0.005)  # 0.5% per °F below 50
    wind_pen = max(0.0, (wind - 10.0) * 0.01)   # 1% per mph above 10
    pass_f = max(0.80, 1.0 - (cold_pen + wind_pen))
    kick_f = max(0.75, 1.0 - (cold_pen*1.2 + wind_pen*1.2))
    # Rush slight boost in bad weather (up to +8%)
    rush_boost = min(0.08, cold_pen*0.5 + wind_pen*0.5)
    rush_f = 1.0 + rush_boost
    return pass_f, rush_f, kick_f

# Home/Away:
# - Home small bump (+5%), Away slight dip (-2%)
def ha_factor(is_home):
    try:
        return 1.05 if int(is_home) == 1 else 0.98
    except Exception:
        return 1.00

# Apply factors
def apply_adj(series, factor):
    if series is None:
        return None
    return series * factor

# Initialize numeric columns we’ll adjust
for col in [
    "proj_pass_yds","proj_pass_td","proj_int",
    "proj_rush_yds","proj_rush_td",
    "proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac",
    "proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td",
    "proj_fgm","proj_fga","proj_xpm","proj_xpa"
]:
    if col not in base.columns:
        base[col] = np.nan

# Row-wise adjustments
adj_cols = [
    ("pass", ["proj_pass_yds","proj_pass_td","proj_int"]),
    ("rush", ["proj_rush_yds","proj_rush_td"]),
    ("rec" , ["proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac"]),
    ("kick", ["proj_fgm","proj_fga","proj_xpm","proj_xpa"]),
]

def row_adjust(r):
    p_f, r_f, k_f = weather_factors(r)
    ha_f = ha_factor(r.get("is_home", 0))
    # Opponent factors
    pf = float(r.get("pass_def", 1.0))
    rf = float(r.get("rush_def", 1.0))
    cf = float(r.get("rec_def", 1.0))
    kf = float(r.get("kick_def", 1.0))

    # Compose factors
    pass_mult = p_f * ha_f * (1.0 / pf)   # tougher defense (>1) reduces output
    rush_mult = r_f * ha_f * (1.0 / rf)
    rec_mult  = p_f * ha_f * (1.0 / cf)   # receiving tied closer to passing/weather
    kick_mult = k_f * ha_f * (1.0 / kf)

    out = {}
    for c in adj_cols:
        kind, cols = c
        mult = {"pass":pass_mult,"rush":rush_mult,"rec":rec_mult,"kick":kick_mult}[kind]
        for col in cols:
            if col in r and pd.notna(r[col]):
                out[col] = r[col] * mult
    # Returns: mildly impacted like passing/kicking (use pass_mult for yards, kick_mult for attempts/points)
    for col in ["proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td"]:
        if pd.notna(r.get(col, np.nan)):
            out[col] = r[col] * pass_mult
    return pd.Series(out)

base = pd.concat([base, base.apply(row_adjust, axis=1)], axis=1)

# -------------------------
# Final ordering + write
# -------------------------
cols_order = [
    "player","team","opponent","is_home","temp_f","wind_mph","dome",
    "proj_pass_yds","proj_pass_td","proj_int","proj_rush_yds","proj_rush_td",
    "proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac",
    "proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td",
    "proj_fgm","proj_fga","proj_xpm","proj_xpa","proj_fg_pct",
    "proj_tackles","proj_sacks","proj_def_int",
    "proj_tb_pct","proj_net_avg","proj_punt_avg","proj_kick_avg",
    "expected_anytime_td","anytime_td_prob"
]
base = ensure_cols(base, cols_order)
base = base[cols_order].sort_values(["team","player"], na_position="last")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
base.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH} with {len(base):,} rows.")

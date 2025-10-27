#!/usr/bin/env python3
# src/player_projections.py
import os
import math
import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed/stats"
OUT_PATH = "data/processed/player_projections.csv"

# ---------- helpers ----------

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase headers and normalize symbols/spaces so merges are robust."""
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace("%", "PCT", regex=False)
          .str.replace("-", "_", regex=False)
          .str.replace("/", "_", regex=False)
          .str.replace(" ", "_", regex=False)
          .str.upper()
    )
    # standardize key cols if they exist in any casing
    ren = {}
    for want, alts in {
        "PLAYER": ["PLAYER", "Player", "player"],
        "TEAM":   ["TEAM", "Team", "team"],
    }.items():
        for a in alts:
            if a in df.columns and want not in df.columns:
                ren[a] = want
    if ren:
        df = df.rename(columns=ren)
    return df

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return norm_cols(df)

def get(df: pd.DataFrame, col: str):
    """Return a Series if column exists, else zeros of proper length."""
    if df.empty:
        return pd.Series(dtype=float)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series([0]*len(df), index=df.index, dtype=float)

def left_merge(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Merge selected columns from b (if they exist)."""
    if b.empty:
        for c in cols:
            if c not in a.columns:
                a[c] = np.nan
        return a
    keep = [c for c in ["PLAYER"] + cols if c in b.columns]
    b2 = b[keep].copy()
    return a.merge(b2, on="PLAYER", how="left")

# simple adjustment hooks (kept neutral unless fields are provided upstream)
def home_away_factor(is_home: pd.Series) -> pd.Series:
    # 1.03 home, 0.97 away if field exists; else 1.0
    if is_home is None:
        return 1.0
    return np.where(pd.to_numeric(is_home, errors="coerce").fillna(0).astype(int) == 1, 1.03, 0.97)

def weather_factor(temp_f: pd.Series, wind_mph: pd.Series, dome: pd.Series) -> pd.Series:
    if temp_f is None or wind_mph is None or dome is None:
        return 1.0
    dome_mask = (dome.astype(str).str.lower() == "yes") | (dome.astype(str).str.lower() == "y")
    wind = pd.to_numeric(wind_mph, errors="coerce").fillna(0)
    temp = pd.to_numeric(temp_f, errors="coerce").fillna(70)
    # wind penalty above 15mph; cold penalty below 25F
    w_mult = np.where(wind > 15, 0.97, 1.0)
    t_mult = np.where(temp < 25, 0.98, 1.0)
    mult = w_mult * t_mult
    mult = np.where(dome_mask, 1.0, mult)
    return mult

def opp_factor(series: pd.Series) -> pd.Series:
    # placeholder neutral (1.0) unless a numeric difficulty column is present
    if series is None:
        return 1.0
    v = pd.to_numeric(series, errors="coerce")
    if v.isna().all():
        return 1.0
    # assume 1.0 is neutral; >1 tough; <1 easy
    return v.fillna(1.0)

# ---------- load per-game inputs (already averaged) ----------
rec   = load_csv(os.path.join(PROCESSED_DIR, "receiving_per_game.csv"))
ret   = load_csv(os.path.join(PROCESSED_DIR, "returning_per_game.csv"))
passg = load_csv(os.path.join(PROCESSED_DIR, "passing_per_game.csv"))
rush  = load_csv(os.path.join(PROCESSED_DIR, "rushing_per_game.csv"))
kick  = load_csv(os.path.join(PROCESSED_DIR, "kicking_per_game.csv"))
kp    = load_csv(os.path.join(PROCESSED_DIR, "kickoffs_punts_per_game.csv"))
defn  = load_csv(os.path.join(PROCESSED_DIR, "defensive_per_game.csv"))
score = load_csv(os.path.join(PROCESSED_DIR, "scoring_per_game.csv"))

# players universe (union)
players = pd.Series(dtype=str)
for df in [rec, ret, passg, rush, kick, kp, defn, score]:
    if not df.empty and "PLAYER" in df.columns:
        players = pd.concat([players, df["PLAYER"]], ignore_index=True)
players = players.dropna().drop_duplicates()

base = pd.DataFrame({"PLAYER": players})
# attach TEAM from first source that has it
for df in [rec, ret, passg, rush, kick, defn, score, kp]:
    if not df.empty and "TEAM" in df.columns:
        base = base.merge(df[["PLAYER","TEAM"]], on="PLAYER", how="left")
        if "TEAM_x" in base.columns:
            base["TEAM"] = base["TEAM_x"].fillna(base["TEAM_y"])
            base = base.drop(columns=["TEAM_x","TEAM_y"])
        if "TEAM" in base.columns and base["TEAM"].notna().any():
            break
base["TEAM"] = base.get("TEAM", pd.Series([""]*len(base)))

# placeholders (neutral) — you can populate these upstream if you have data
for c, val in [("OPPONENT",""), ("IS_HOME",0), ("TEMP_F",0.0), ("WIND_MPH",0.0), ("DOME","no")]:
    base[c] = val

# ---------- merge per-game baselines ----------
# Receiving -> projections
if not rec.empty:
    rec_pg = rec.rename(columns={
        "RECYDS":"PROJ_REC_YARDS",
        "RECTD":"PROJ_REC_TD",
        "TGT":"PROJ_TARGETS",
        "REC":"PROJ_REC",
        "YAC":"PROJ_YAC",
    })
    base = left_merge(base, rec_pg, ["PROJ_REC_YARDS","PROJ_REC_TD","PROJ_TARGETS","PROJ_REC","PROJ_YAC"])
else:
    for c in ["PROJ_REC_YARDS","PROJ_REC_TD","PROJ_TARGETS","PROJ_REC","PROJ_YAC"]:
        base[c] = np.nan

# Returning
if not ret.empty:
    ret_pg = ret.rename(columns={
        "K_RET_YDS":"PROJ_KR_YDS",
        "K_RET_TD":"PROJ_KR_TD",
        "P_RET_YDS":"PROJ_PR_YDS",
        "P_RET_TD":"PROJ_PR_TD",
    })
    base = left_merge(base, ret_pg, ["PROJ_KR_YDS","PROJ_KR_TD","PROJ_PR_YDS","PROJ_PR_TD"])
else:
    for c in ["PROJ_KR_YDS","PROJ_KR_TD","PROJ_PR_YDS","PROJ_PR_TD"]:
        base[c] = np.nan

# Passing
if not passg.empty:
    p_pg = passg.rename(columns={
        "PYDS":"PROJ_PASS_YDS",
        "PTD":"PROJ_PASS_TD",
        "INT":"PROJ_INT",
    })
    base = left_merge(base, p_pg, ["PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_INT"])
else:
    for c in ["PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_INT"]:
        base[c] = np.nan

# Rushing
if not rush.empty:
    r_pg = rush.rename(columns={
        "RYDS":"PROJ_RUSH_YDS",
        "RTD":"PROJ_RUSH_TD",
    })
    base = left_merge(base, r_pg, ["PROJ_RUSH_YDS","PROJ_RUSH_TD"])
else:
    for c in ["PROJ_RUSH_YDS","PROJ_RUSH_TD"]:
        base[c] = np.nan

# Kicking
if not kick.empty:
    k_pg = kick.rename(columns={
        "FGM":"PROJ_FGM",
        "FGA":"PROJ_FGA",
        "XPM":"PROJ_XPM",
        "XPA":"PROJ_XPA",
        "FG_PCT":"PROJ_FG_PCT",
    })
    base = left_merge(base, k_pg, ["PROJ_FGM","PROJ_FGA","PROJ_XPM","PROJ_XPA","PROJ_FG_PCT"])
else:
    for c in ["PROJ_FGM","PROJ_FGA","PROJ_XPM","PROJ_XPA","PROJ_FG_PCT"]:
        base[c] = np.nan

# Defensive
if not defn.empty:
    d_pg = defn.rename(columns={
        "TCKL":"PROJ_TACKLES",
        "SCK":"PROJ_SACKS",
        "DEF_INT":"PROJ_DEF_INT",
    })
    base = left_merge(base, d_pg, ["PROJ_TACKLES","PROJ_SACKS","PROJ_DEF_INT"])
else:
    for c in ["PROJ_TACKLES","PROJ_SACKS","PROJ_DEF_INT"]:
        base[c] = np.nan

# Kickoffs/Punts (header-insensitive)
if not kp.empty:
    # after normalization we expect TB_PCT, NET_AVG, P_AVG, K_AVG
    rename_map = {}
    for src, dst in {
        "TB_PCT":"PROJ_TB_PCT",
        "NET_AVG":"PROJ_NET_AVG",
        "P_AVG":"PROJ_PUNT_AVG",
        "K_AVG":"PROJ_KICK_AVG",
    }.items():
        if src in kp.columns:
            rename_map[src] = dst
    kp2 = kp.rename(columns=rename_map)
    base = left_merge(base, kp2, list(rename_map.values()))
else:
    for c in ["PROJ_TB_PCT","PROJ_NET_AVG","PROJ_PUNT_AVG","PROJ_KICK_AVG"]:
        base[c] = np.nan

# Scoring → anytime TD λ (per-game TD is already averaged)
if not score.empty and "TD" in score.columns:
    td = score[["PLAYER","TD"]].rename(columns={"TD":"TD_PG"})
    base = base.merge(td, on="PLAYER", how="left")
else:
    base["TD_PG"] = 0.0

# ---------- apply neutral adjustments (no GP re-scaling) ----------
# If upstream supplies IS_HOME/TEMP_F/WIND_MPH/DOME/OPP_* columns later,
# these hooks will scale accordingly; otherwise factors == 1.0.
ha = home_away_factor(base.get("IS_HOME"))
wx = weather_factor(base.get("TEMP_F"), base.get("WIND_MPH"), base.get("DOME"))

adj_cols = [
    "PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_INT","PROJ_RUSH_YDS","PROJ_RUSH_TD",
    "PROJ_REC_YARDS","PROJ_REC_TD","PROJ_TARGETS","PROJ_REC","PROJ_YAC",
    "PROJ_KR_YDS","PROJ_KR_TD","PROJ_PR_YDS","PROJ_PR_TD",
    "PROJ_FGM","PROJ_FGA","PROJ_XPM","PROJ_XPA","PROJ_FG_PCT",
    "PROJ_TACKLES","PROJ_SACKS","PROJ_DEF_INT",
    "PROJ_TB_PCT","PROJ_NET_AVG","PROJ_PUNT_AVG","PROJ_KICK_AVG",
]
for c in adj_cols:
    if c in base.columns:
        base[c] = pd.to_numeric(base[c], errors="coerce")
        base[c] = base[c] * ha * wx

# Anytime TD probability (Poisson, λ = TD per game)
base["TD_PG"] = pd.to_numeric(base["TD_PG"], errors="coerce").fillna(0)
base["EXPECTED_ANYTIME_TD"] = base["TD_PG"].clip(lower=0)
base["ANYTIME_TD_PROB"] = 1 - np.exp(-base["EXPECTED_ANYTIME_TD"])

# final ordering
cols_order = [
    "PLAYER","TEAM","OPPONENT","IS_HOME","TEMP_F","WIND_MPH","DOME",
    "PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_INT","PROJ_RUSH_YDS","PROJ_RUSH_TD",
    "PROJ_REC_YARDS","PROJ_REC_TD","PROJ_TARGETS","PROJ_REC","PROJ_YAC",
    "PROJ_KR_YDS","PROJ_KR_TD","PROJ_PR_YDS","PROJ_PR_TD",
    "PROJ_FGM","PROJ_FGA","PROJ_XPM","PROJ_XPA","PROJ_FG_PCT",
    "PROJ_TACKLES","PROJ_SACKS","PROJ_DEF_INT",
    "PROJ_TB_PCT","PROJ_NET_AVG","PROJ_PUNT_AVG","PROJ_KICK_AVG",
    "EXPECTED_ANYTIME_TD","ANYTIME_TD_PROB",
]
for c in cols_order:
    if c not in base.columns:
        base[c] = np.nan

base = base[cols_order].sort_values(["TEAM","PLAYER"], na_position="last")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
base.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH} with {len(base):,} rows.")

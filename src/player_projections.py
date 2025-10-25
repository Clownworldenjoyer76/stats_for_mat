#!/usr/bin/env python3
# src/player_projections.py
import os
import math
import numpy as np
import pandas as pd

PG_DIR   = "data/processed/stats"
OUT_PATH = "data/processed/player_projections.csv"

# Optional game context:
# columns: player,team,opponent,is_home,temp_f,wind_mph,dome
CONTEXT_PATH = "data/raw/context/player_game_context.csv"

# Optional opponent mods:
# columns: team,pass_def_mod,rush_def_mod,rec_mod,kick_mod,return_mod,defense_mod
OPP_MODS_PATH = "data/processed/opponent_mods.csv"


# ----------------------------- helpers ---------------------------------------
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def col(df, *cands):
    """Return first existing column name (case-insensitive)."""
    if df.empty:
        return None
    cols_map = {c.lower(): c for c in df.columns}
    for c in cands:
        lc = c.lower()
        if lc in cols_map:
            return cols_map[lc]
    return None

def get_series(df, cname, default=0.0):
    """Safe numeric series (fillna->0)."""
    if df is None or df.empty or cname is None or cname not in df.columns:
        return pd.Series(default, index=pd.RangeIndex(0))
    s = pd.to_numeric(df[cname], errors="coerce").fillna(0.0)
    return s

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def simple_home_mod(is_home):
    # modest +5% at home, -5% away
    return np.where(is_home.astype(int) == 1, 1.05, 0.95)

def simple_weather_mod(temp_f, wind_mph, dome):
    # Dome nullifies weather effects
    dome_mask = (str(dome).lower() == "yes") if isinstance(dome, str) else False
    if isinstance(dome, pd.Series):
        dome_mask = dome.astype(str).str.lower().eq("yes")

    # Start at 1.0, then apply small penalties/bonuses
    # Passing/Receiving more sensitive to wind and cold
    temp_f = pd.to_numeric(temp_f, errors="coerce").fillna(60.0)
    wind_mph = pd.to_numeric(wind_mph, errors="coerce").fillna(0.0)

    pass_rec = np.ones_like(temp_f, dtype=float)
    rush     = np.ones_like(temp_f, dtype=float)
    kick     = np.ones_like(temp_f, dtype=float)
    ret      = np.ones_like(temp_f, dtype=float)

    # Wind effects
    windy = wind_mph > 15
    pass_rec = np.where(windy, pass_rec * 0.90, pass_rec)
    kick     = np.where(windy, kick * 0.92, kick)

    # Cold effects
    cold = temp_f < 40
    pass_rec = np.where(cold, pass_rec * 0.95, pass_rec)
    kick     = np.where(cold, kick * 0.95, kick)
    rush     = np.where(cold | windy, rush * 1.03, rush)  # small run lean

    # Dome overrides (reset to 1.0)
    if isinstance(dome_mask, (pd.Series, np.ndarray)):
        pass_rec = np.where(dome_mask, 1.0, pass_rec)
        rush     = np.where(dome_mask, 1.0, rush)
        kick     = np.where(dome_mask, 1.0, kick)
        ret      = np.where(dome_mask, 1.0, ret)
    elif dome_mask:  # scalar true
        pass_rec = rush = kick = ret = np.ones_like(pass_rec, dtype=float)

    return pass_rec, rush, kick, ret

def opponent_mods(opponent_series, mods_df, mod_col, default=1.0):
    if mods_df.empty or mod_col not in mods_df.columns:
        return np.full(len(opponent_series), default, dtype=float)
    # Normalize opponent key
    key = "team"
    df = mods_df.copy()
    df[key] = df[key].astype(str).str.strip()
    opp = opponent_series.astype(str).str.strip()
    m = opp.map(df.set_index(key)[mod_col]).astype(float)
    return m.fillna(default).to_numpy()


# ------------------------ load per-game baselines -----------------------------
passing   = load_csv(os.path.join(PG_DIR, "passing_per_game.csv"))
rushing   = load_csv(os.path.join(PG_DIR, "rushing_per_game.csv"))
receiving = load_csv(os.path.join(PG_DIR, "receiving_per_game.csv"))
returning = load_csv(os.path.join(PG_DIR, "returning_per_game.csv"))
kicking   = load_csv(os.path.join(PG_DIR, "kicking_per_game.csv"))
defense   = load_csv(os.path.join(PG_DIR, "defensive_per_game.csv"))
kickpunts = load_csv(os.path.join(PG_DIR, "kickoffs_punts_per_game.csv"))
scoring   = load_csv(os.path.join(PG_DIR, "scoring_per_game.csv"))

# Player + team spine (outer union)
parts = []
for df in (passing, rushing, receiving, returning, kicking, defense, kickpunts, scoring):
    if not df.empty and col(df, "PLAYER"):
        sub = df[[col(df, "PLAYER")]].copy()
        sub.columns = ["player"]
        parts.append(sub)
players = pd.concat(parts, ignore_index=True).drop_duplicates()

base = pd.DataFrame({"player": players["player"]})
# Attach a team from the first source that has it
for src in (receiving, rushing, passing, returning, kicking, defense):
    if not src.empty and col(src, "PLAYER") and col(src, "TEAM"):
        base = base.merge(
            src[[col(src, "PLAYER"), col(src, "TEAM")]].rename(
                columns={col(src, "PLAYER"): "player", col(src, "TEAM"): "team"}
            ),
            on="player", how="left"
        )
        if base["team"].notna().any():
            base["team"] = base["team"].ffill().bfill()
base["team"] = base["team"].fillna("")

# ------------------------ bring in context (optional) -------------------------
ctx = load_csv(CONTEXT_PATH)
if not ctx.empty and col(ctx, "player"):
    ctx = ctx.rename(columns={col(ctx, "player"): "player"})
    # Keep expected fields if present
    for want, cands in {
        "team":      ("team",),
        "opponent":  ("opponent",),
        "is_home":   ("is_home",),
        "temp_f":    ("temp_f",),
        "wind_mph":  ("wind_mph",),
        "dome":      ("dome",),
    }.items():
        c = col(ctx, *cands)
        if c and c != want:
            ctx = ctx.rename(columns={c: want})

    base = base.merge(ctx[["player","team","opponent","is_home","temp_f","wind_mph","dome"]]
                      .drop_duplicates("player"),
                      on="player", how="left")
else:
    # Defaults if no context
    base["opponent"] = ""
    base["is_home"]  = 0
    base["temp_f"]   = 60.0
    base["wind_mph"] = 0.0
    base["dome"]     = "no"

# ------------------------ optional opponent modifiers -------------------------
opp_mods = load_csv(OPP_MODS_PATH)

# ------------------------ merge per-family baselines --------------------------
# Passing
if not passing.empty and col(passing, "PLAYER"):
    df = passing.rename(columns={col(passing, "PLAYER"): "player"})
    base = base.merge(
        df[["player",
            col(df, "PYDS"), col(df, "PTD"), col(df, "INT")]].rename(
            columns={
                col(df, "PYDS"): "pass_yards_pg",
                col(df, "PTD"):  "pass_td_pg",
                col(df, "INT"):  "pass_int_pg",
            }
        ),
        on="player", how="left"
    )

# Rushing
if not rushing.empty and col(rushing, "PLAYER"):
    df = rushing.rename(columns={col(rushing, "PLAYER"): "player"})
    base = base.merge(
        df[["player",
            col(df, "RYDS"), col(df, "RTD")]].rename(
            columns={
                col(df, "RYDS"): "rush_yards_pg",
                col(df, "RTD"):  "rush_td_pg",
            }
        ),
        on="player", how="left"
    )

# Receiving
if not receiving.empty and col(receiving, "PLAYER"):
    df = receiving.rename(columns={col(receiving, "PLAYER"): "player"})
    base = base.merge(
        df[["player",
            col(df, "RECYDS"), col(df, "RECTD"),
            col(df, "TGT"), col(df, "REC"), col(df, "YAC")]].rename(
            columns={
                col(df, "RECYDS"): "rec_yards_pg",
                col(df, "RECTD"):  "rec_td_pg",
                col(df, "TGT"):    "targets_pg",
                col(df, "REC"):    "receptions_pg",
                col(df, "YAC"):    "yac_pg",
            }
        ),
        on="player", how="left"
    )

# Returning
if not returning.empty and col(returning, "PLAYER"):
    df = returning.rename(columns={col(returning, "PLAYER"): "player"})
    base = base.merge(
        df[["player",
            col(df, "K_RET_YDS"), col(df, "K_RET_TD"),
            col(df, "P_RET_YDS"), col(df, "P_RET_TD")]].rename(
            columns={
                col(df, "K_RET_YDS"): "kr_yards_pg",
                col(df, "K_RET_TD"):  "kr_td_pg",
                col(df, "P_RET_YDS"): "pr_yards_pg",
                col(df, "P_RET_TD"):  "pr_td_pg",
            }
        ),
        on="player", how="left"
    )

# Kicking
if not kicking.empty and col(kicking, "PLAYER"):
    df = kicking.rename(columns={col(kicking, "PLAYER"): "player"})
    base = base.merge(
        df[["player",
            col(df, "FGM"), col(df, "FGA"), col(df, "FG_PCT"),
            col(df, "XPM"), col(df, "XPA")]].rename(
            columns={
                col(df, "FGM"):    "fgm_pg",
                col(df, "FGA"):    "fga_pg",
                col(df, "FG_PCT"): "fg_pct_pg",
                col(df, "XPM"):    "xpm_pg",
                col(df, "XPA"):    "xpa_pg",
            }
        ),
        on="player", how="left"
    )

# Defense (IDP)
if not defense.empty and col(defense, "PLAYER"):
    df = defense.rename(columns={col(defense, "PLAYER"): "player"})
    base = base.merge(
        df[["player",
            col(df, "TCKL"), col(df, "SCK"), col(df, "DEF_INT")]].rename(
            columns={
                col(df, "TCKL"):    "tackles_pg",
                col(df, "SCK"):     "sacks_pg",
                col(df, "DEF_INT"): "def_int_pg",
            }
        ),
        on="player", how="left"
    )

# Kickoffs/Punts
if not kickpunts.empty and col(kickpunts, "Player"):
    df = kickpunts.rename(columns={col(kickpunts, "Player"): "player"})
    # Percent/averages are already rates; just carry over
    # Guard for variants like "TB %", "NET AVG", "P-AVG", "K-AVG"
    tb_pct  = col(df, "TB %", "TB_%", "TB_PCT")
    net_avg = col(df, "NET AVG", "NET_AVG")
    p_avg   = col(df, "P-AVG", "P_AVG", "P-AVG.")
    k_avg   = col(df, "K-AVG", "K_AVG", "K-AVG.")
    keep = ["player"]
    if tb_pct:  keep.append(tb_pct)
    if net_avg: keep.append(net_avg)
    if p_avg:   keep.append(p_avg)
    if k_avg:   keep.append(k_avg)
    df_keep = df[keep].copy()
    ren = {}
    if tb_pct:  ren[tb_pct]  = "tb_pct_pg"
    if net_avg: ren[net_avg] = "net_avg_pg"
    if p_avg:   ren[p_avg]   = "punt_avg_pg"
    if k_avg:   ren[k_avg]   = "kick_avg_pg"
    base = base.merge(df_keep.rename(columns=ren), on="player", how="left")

# Scoring (λ for anytime TD)
if not scoring.empty and col(scoring, "PLAYER"):
    df = scoring.rename(columns={col(scoring, "PLAYER"): "player"})
    if col(df, "TD"):
        base = base.merge(df[["player", col(df, "TD")]].rename(columns={col(df, "TD"): "td_pg"}),
                          on="player", how="left")
    else:
        base["td_pg"] = 0.0
else:
    base["td_pg"] = 0.0

# ------------------------ adjustments (per-game ONLY) -------------------------
# Factors
home_factor = simple_home_mod(base["is_home"].fillna(0))
pr_mod, ru_mod, ki_mod, re_mod = simple_weather_mod(
    base["temp_f"].fillna(60.0),
    base["wind_mph"].fillna(0.0),
    base["dome"].fillna("no"),
)

# Opponent mods (optional; default 1.0 if file missing)
opp_pass = opponent_mods(base["opponent"].fillna(""), opp_mods, "pass_def_mod", default=1.0)
opp_rush = opponent_mods(base["opponent"].fillna(""), opp_mods, "rush_def_mod", default=1.0)
opp_rec  = opponent_mods(base["opponent"].fillna(""), opp_mods, "rec_mod",       default=1.0)
opp_kick = opponent_mods(base["opponent"].fillna(""), opp_mods, "kick_mod",      default=1.0)
opp_ret  = opponent_mods(base["opponent"].fillna(""), opp_mods, "return_mod",    default=1.0)
opp_idp  = opponent_mods(base["opponent"].fillna(""), opp_mods, "defense_mod",   default=1.0)

# Build projections = per-game × home × weather × opponent
def proj(x, *mods):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    f = np.ones(len(base), dtype=float)
    for m in mods:
        f = f * m
    return x * f

# Passing
base["proj_pass_yds"] = proj(base.get("pass_yards_pg", 0.0), home_factor, pr_mod, opp_pass)
base["proj_pass_td"]  = proj(base.get("pass_td_pg", 0.0),    home_factor, pr_mod, opp_pass)
base["proj_int"]      = proj(base.get("pass_int_pg", 0.0),   1.0)  # keep ints neutral

# Rushing
base["proj_rush_yds"] = proj(base.get("rush_yards_pg", 0.0), home_factor, ru_mod, opp_rush)
base["proj_rush_td"]  = proj(base.get("rush_td_pg", 0.0),    home_factor, ru_mod, opp_rush)

# Receiving
base["proj_rec_yards"] = proj(base.get("rec_yards_pg", 0.0), home_factor, pr_mod, opp_rec)
base["proj_rec_td"]    = proj(base.get("rec_td_pg", 0.0),    home_factor, pr_mod, opp_rec)
base["proj_targets"]   = proj(base.get("targets_pg", 0.0),   home_factor, pr_mod, opp_rec)
base["proj_rec"]       = proj(base.get("receptions_pg", 0.0),home_factor, pr_mod, opp_rec)
base["proj_yac"]       = proj(base.get("yac_pg", 0.0),       home_factor, pr_mod, opp_rec)

# Returning
base["proj_kr_yds"] = proj(base.get("kr_yards_pg", 0.0), re_mod, opp_ret)
base["proj_kr_td"]  = proj(base.get("kr_td_pg", 0.0),    re_mod, opp_ret)
base["proj_pr_yds"] = proj(base.get("pr_yards_pg", 0.0), re_mod, opp_ret)
base["proj_pr_td"]  = proj(base.get("pr_td_pg", 0.0),    re_mod, opp_ret)

# Kicking
base["proj_fgm"]    = proj(base.get("fgm_pg", 0.0), ki_mod, opp_kick)
base["proj_fga"]    = proj(base.get("fga_pg", 0.0), ki_mod, opp_kick)
base["proj_xpm"]    = proj(base.get("xpm_pg", 0.0), ki_mod, opp_kick)
base["proj_xpa"]    = proj(base.get("xpa_pg", 0.0), ki_mod, opp_kick)
# FG% is already a rate; carry through (cap 0..100)
base["proj_fg_pct"] = pd.to_numeric(base.get("fg_pct_pg", 0.0), errors="coerce").clip(lower=0, upper=100).fillna(0.0)

# IDP
base["proj_tackles"] = proj(base.get("tackles_pg", 0.0), opp_idp)
base["proj_sacks"]   = proj(base.get("sacks_pg", 0.0),   opp_idp)
base["proj_def_int"] = proj(base.get("def_int_pg", 0.0), opp_idp)

# Kickoffs/Punts (already rates/averages)
base["proj_tb_pct"]   = pd.to_numeric(base.get("tb_pct_pg", 0.0),  errors="coerce").fillna(0.0)
base["proj_net_avg"]  = pd.to_numeric(base.get("net_avg_pg", 0.0), errors="coerce").fillna(0.0)
base["proj_punt_avg"] = pd.to_numeric(base.get("punt_avg_pg", 0.0),errors="coerce").fillna(0.0)
base["proj_kick_avg"] = pd.to_numeric(base.get("kick_avg_pg", 0.0),errors="coerce").fillna(0.0)

# Anytime TD probability (Poisson with λ = total TD per game)
base["td_pg"] = pd.to_numeric(base.get("td_pg", 0.0), errors="coerce").fillna(0.0)
base["expected_anytime_td"] = base["td_pg"].clip(lower=0)
base["anytime_td_prob"] = 1 - np.exp(-base["expected_anytime_td"])

# Output schema & order
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

# Write
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
base.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH} with {len(base):,} rows.")

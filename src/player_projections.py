# src/player_projections.py
import os
import math
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/stats"
OUT_PATH = "data/processed/player_projections.csv"

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct")
    )
    return df

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def pick_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_val(df: pd.DataFrame, player: str, col_candidates) -> float | str:
    """
    Return the value for 'player' from any of the provided candidate columns.
    If the column or row doesn't exist, return np.nan (keeps pipeline alive).
    """
    if df is None or df.empty or "player" not in df.columns:
        return np.nan
    col = pick_col(df, col_candidates)
    if col is None:
        return np.nan
    rows = df.loc[df["player"] == player, col]
    if rows.empty:
        return np.nan
    return pd.to_numeric(rows.iloc[0], errors="coerce")

def load_csv_maybe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = clean_columns(df)
        return df
    return pd.DataFrame()

# --- Load core input files ----------------------------------------------------
receiving = load_csv_maybe(os.path.join(RAW_DIR, "receiving.csv"))
returning = load_csv_maybe(os.path.join(RAW_DIR, "returning.csv"))
scoring   = load_csv_maybe(os.path.join(RAW_DIR, "scoring.csv"))

# Optional (may not exist in repo). If missing, we will default to NaNs.
kickpunts = load_csv_maybe(os.path.join(RAW_DIR, "kickpunts.csv"))

# --- Normalize types ----------------------------------------------------------
for df in [receiving, returning, scoring, kickpunts]:
    if df.empty:
        continue
    # standard numeric coercions
    for c in df.columns:
        if c not in ("player", "team"):
            df[c] = to_num(df[c])

# --- Per-game computations ----------------------------------------------------
# Receiving
rec_cols_present = not receiving.empty and {"player","team","gp"}.issubset(receiving.columns)
if rec_cols_present:
    r = receiving.copy()
    r["gp"] = r["gp"].replace(0, np.nan)
    rec_pg = pd.DataFrame({
        "player": r["player"],
        "team": r.get("team", pd.Series([""]*len(r))),
        "proj_rec_yards": r.get("recyds", 0) / r["gp"],
        "proj_rec_td":    r.get("rectd", 0) / r["gp"],
        "proj_targets":   r.get("tgt", 0)    / r["gp"],
        "proj_rec":       r.get("rec", 0)    / r["gp"],
        "proj_yac":       r.get("yac", 0)    / r["gp"],
    })
else:
    rec_pg = pd.DataFrame(columns=["player","team","proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac"])

# Returning
ret_cols_present = not returning.empty and {"player","team","gp"}.issubset(returning.columns)
if ret_cols_present:
    t = returning.copy()
    t["gp"] = t["gp"].replace(0, np.nan)
    ret_pg = pd.DataFrame({
        "player": t["player"],
        "team": t.get("team", pd.Series([""]*len(t))),
        "proj_kr_yds":  t.get("k_ret_yds", 0) / t["gp"],
        "proj_kr_td":   t.get("k_ret_td", 0)  / t["gp"],
        "proj_pr_yds":  t.get("p_ret_yds", 0) / t["gp"],
        "proj_pr_td":   t.get("p_ret_td", 0)  / t["gp"],
    })
else:
    ret_pg = pd.DataFrame(columns=["player","team","proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td"])

# Scoring (for anytime TD λ)
score_cols_present = not scoring.empty and {"player","team","gp","td"}.issubset(scoring.columns)
if score_cols_present:
    s = scoring.copy()
    s["gp"] = s["gp"].replace(0, np.nan)
    # total TD rate per game (safer than summing partial columns that may be NaN/blank)
    s["td_pg"] = s["td"] / s["gp"]
    td_pg = s[["player","team","td_pg"]].copy()
else:
    td_pg = pd.DataFrame(columns=["player","team","td_pg"])

# --- Assemble player list (outer union of sources) ---------------------------
players = pd.Series(dtype=str)
for df in [rec_pg, ret_pg, td_pg]:
    if not df.empty and "player" in df.columns:
        players = pd.concat([players, df["player"]], ignore_index=True)

players = players.dropna().drop_duplicates()

base = pd.DataFrame({
    "player": players,
})

# Merge in team (prefer receiving -> returning -> scoring)
def attach_team(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return target
    return target.merge(source[["player","team"]], on="player", how="left", suffixes=("",""))

base = attach_team(base, rec_pg)
base["team"] = base["team"].fillna("")
base = attach_team(base.rename(columns={"team":"team_left"}), ret_pg).rename(columns={"team":"team_right"})
base["team"] = base["team_left"].where(base["team_left"].ne("") & base["team_left"].notna(),
                                       base["team_right"])
base = base.drop(columns=["team_left","team_right"])
base = attach_team(base.rename(columns={"team":"team_left"}), td_pg).rename(columns={"team":"team_right"})
base["team"] = base["team_left"].where(base["team_left"].ne("") & base["team_left"].notna(),
                                       base["team_right"])
base = base.drop(columns=["team_left","team_right"])

# --- Projected weather/opponent placeholders (kept for schema continuity) ----
base["opponent"] = ""
base["is_home"]  = 0
base["temp_f"]   = 0.0
base["wind_mph"] = 0.0
base["dome"]     = "no"

# --- Bring in per-game metrics ------------------------------------------------
base = base.merge(rec_pg.drop(columns=["team"]), on="player", how="left")
base = base.merge(ret_pg.drop(columns=["team"]), on="player", how="left")
base = base.merge(td_pg.drop(columns=["team"]), on="player", how="left")

# --- Kick/Punt metrics (robust, non-crashing) --------------------------------
# Accept multiple header variants if the optional file is present.
if not kickpunts.empty:
    # Already cleaned to lower/underscored names in load step
    pass

# Compute per-player special-teams projections safely
special_cols = {
    "proj_tb_pct":  ["tb_pct", "touchback_pct", "tbpercent", "tb_rate"],
    "proj_net_avg": ["net_avg", "netavg", "net_average", "net"],
    "proj_punt_avg":["punt_avg", "puntaverage", "p_avg", "avg"],
    "proj_kick_avg":["kick_avg", "kickaverage", "k_avg"],
}

for out_col, candidates in special_cols.items():
    base[out_col] = base["player"].apply(lambda p: safe_val(kickpunts, p, candidates))

# --- Passing placeholders (kept for schema continuity) -----------------------
for c in ["proj_pass_yds","proj_pass_td","proj_int","proj_rush_yds","proj_rush_td"]:
    if c not in base.columns:
        base[c] = np.nan

# --- Kicking placeholders (schema continuity) --------------------------------
for c in ["proj_fgm","proj_fga","proj_xpm","proj_xpa","proj_fg_pct",
          "proj_tackles","proj_sacks","proj_def_int"]:
    if c not in base.columns:
        base[c] = np.nan

# --- Compute anytime TD probability ------------------------------------------
# λ = total TDs per game (from scoring). P(≥1 TD) = 1 - exp(-λ).
base["td_pg"] = base["td_pg"].fillna(0)
base["expected_anytime_td"] = base["td_pg"].clip(lower=0)
base["anytime_td_prob"] = 1 - np.exp(-base["expected_anytime_td"])

# --- Column order to match prior outputs -------------------------------------
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

# Ensure all expected columns exist
for c in cols_order:
    if c not in base.columns:
        base[c] = np.nan

base = base[cols_order].sort_values(["team","player"], na_position="last")

# --- Write -------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
base.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH} with {len(base):,} rows.")

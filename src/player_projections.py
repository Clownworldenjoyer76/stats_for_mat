#!/usr/bin/env python3
# src/player_projections.py
import os
import numpy as np
import pandas as pd

STATS_DIR = "data/processed/stats"
OUT_PATH  = "data/processed/player_projections.csv"

# ---------- helpers ----------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower, strip, underscore, and normalize percent signs."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct")
    )
    return df

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return clean_columns(pd.read_csv(path))

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# ---------- load per-game inputs (from processed) ----------
receiving = load_csv(os.path.join(STATS_DIR, "receiving_per_game.csv"))
returning = load_csv(os.path.join(STATS_DIR, "returning_per_game.csv"))
scoring   = load_csv(os.path.join(STATS_DIR, "scoring_per_game.csv"))
kickpunts = load_csv(os.path.join(STATS_DIR, "kickoffs_punts_per_game.csv"))

# numeric coercion where appropriate
for df in (receiving, returning, scoring, kickpunts):
    if df.empty:
        continue
    for c in df.columns:
        if c not in ("player", "team"):
            df[c] = to_num(df[c])

# ---------- build base player list ----------
players = pd.Series(dtype=str)
for df in (receiving, returning, scoring):
    if not df.empty and "player" in df.columns:
        players = pd.concat([players, df["player"]], ignore_index=True)

players = players.dropna().drop_duplicates()

base = pd.DataFrame({"player": players})

# attach team (prefer receiving -> returning -> scoring)
def attach_team(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    if source.empty or "player" not in source.columns or "team" not in source.columns:
        return target
    return target.merge(source[["player", "team"]], on="player", how="left")

base = attach_team(base, receiving)
base.rename(columns={"team": "team_r"}, inplace=True)
base = attach_team(base, returning)
base.rename(columns={"team": "team_ret"}, inplace=True)
base = attach_team(base, scoring)
base["team"] = (
    base["team_r"].fillna("")
    .where(base["team_r"].fillna("") != "", base["team_ret"])
    .where(lambda s: s.fillna("") != "", base.get("team", pd.Series([""] * len(base))))
)
base.drop(columns=[c for c in ["team_r", "team_ret"] if c in base.columns], inplace=True)
base["team"] = base["team"].fillna("")

# placeholders for game context (can be filled upstream later)
for c, v in [("opponent",""), ("is_home",0), ("temp_f",0.0), ("wind_mph",0.0), ("dome","no")]:
    base[c] = v

# ---------- merge skill projections ----------
# receiving
if not receiving.empty:
    keep = [c for c in ["player","recyds","rectd","tgt","rec","yac"] if c in receiving.columns]
    r = receiving[keep].rename(columns={
        "recyds":"proj_rec_yards",
        "rectd":"proj_rec_td",
        "tgt":"proj_targets",
        "rec":"proj_rec",
        "yac":"proj_yac",
    })
    base = base.merge(r, on="player", how="left")

# returning
if not returning.empty:
    keep = [c for c in ["player","k_ret_yds","k_ret_td","p_ret_yds","p_ret_td"] if c in returning.columns]
    t = returning[keep].rename(columns={
        "k_ret_yds":"proj_kr_yds",
        "k_ret_td":"proj_kr_td",
        "p_ret_yds":"proj_pr_yds",
        "p_ret_td":"proj_pr_td",
    })
    base = base.merge(t, on="player", how="left")

# scoring (for anytime TD rate)
if not scoring.empty and {"player","td","gp"}.issubset(scoring.columns):
    s = scoring[["player","td","gp"]].copy()
    s["gp"] = s["gp"].replace(0, np.nan)
    s["td_pg"] = s["td"] / s["gp"]
    base = base.merge(s[["player","td_pg"]], on="player", how="left")
else:
    base["td_pg"] = np.nan

# ---------- kick/punt metrics (robust to headers) ----------
if not kickpunts.empty:
    kp = kickpunts.copy()
    # ensure 'player' exists; if not, try to recover or skip
    if "player" not in kp.columns:
        # try common variants before giving up
        for alt in ["name", "kicker", "punter", "returner", "Player"]:
            if alt.lower() in kp.columns:
                kp.rename(columns={alt.lower(): "player"}, inplace=True)
                break
    if "player" in kp.columns:
        # normalize possible column variants -> standard names
        rename_map = {}
        variants = {
            "proj_tb_pct":  ["tb_pct", "tb_pct.", "tb_pct_%", "tb_pct", "tb_pct_", "tb_pct__",
                             "tb_pctg", "tb_pctpercentage", "tb_pctpercent", "tb_pct_rate", "tb_pct_rate_%", "tb_pctrate",
                             "tb_pct_", "tb_pctg_", "tb_pct_percent", "tb_pct_percent_",
                             "tb_pctage", "tb_pct_opp", "tb_pct_opponent", "tb_pct_allowed", "tb_pct_allowed_",
                             "tb_pct_allowed_percent", "tb_pct_allowed_rate", "tb_pct_allowed_%", "tb_pct_allowed_percent_",
                             "tb_pct_allowed_rate_", "tb_pct_allowed_%_"],
            "proj_net_avg": ["net_avg", "netavg", "net_average", "net"],
            "proj_punt_avg":["p_avg","p-avg","p_avg.","p-avg.","punt_avg","pavg","pavg."],
            "proj_kick_avg":["k_avg","k-avg","k_avg.","k-avg.","kick_avg","kavg","kavg."],
        }
        # build map for any matching columns
        for std, alts in variants.items():
            for a in alts:
                if a in kp.columns:
                    rename_map[a] = std
        kp.rename(columns=rename_map, inplace=True)

        keep = [c for c in ["player","proj_tb_pct","proj_net_avg","proj_punt_avg","proj_kick_avg"] if c in kp.columns]
        if keep:
            base = base.merge(kp[keep], on="player", how="left")
# if we couldn't merge, create empty columns to preserve schema
for c in ["proj_tb_pct","proj_net_avg","proj_punt_avg","proj_kick_avg"]:
    if c not in base.columns:
        base[c] = np.nan

# ---------- placeholders for other schema fields ----------
for c in ["proj_pass_yds","proj_pass_td","proj_int","proj_rush_yds","proj_rush_td",
          "proj_fgm","proj_fga","proj_xpm","proj_xpa","proj_fg_pct",
          "proj_tackles","proj_sacks","proj_def_int"]:
    if c not in base.columns:
        base[c] = np.nan

# ---------- anytime TD probability ----------
base["td_pg"] = base["td_pg"].fillna(0)
base["expected_anytime_td"] = base["td_pg"].clip(lower=0)
base["anytime_td_prob"] = 1 - np.exp(-base["expected_anytime_td"])

# ---------- column order ----------
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
for c in cols_order:
    if c not in base.columns:
        base[c] = np.nan

base = base[cols_order].sort_values(["team","player"], na_position="last")

# ---------- write ----------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
base.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH} with {len(base):,} rows.")

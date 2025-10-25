#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
from pathlib import Path

# ---------- File locations ----------
DATA_DIR = Path("data")
RAW_STATS = DATA_DIR / "raw" / "stats"
PROCESSED_DIR = DATA_DIR / "processed"

TEAM_STADIUMS_CSV = DATA_DIR / "team_stadiums.csv"
GAMES_CSV = PROCESSED_DIR / "games.csv"
NFL_TEAM_METRICS = PROCESSED_DIR / "nfl_unified_with_metrics.csv"  # used for opponent context

# Player stat files provided
PASSING_CSV = RAW_STATS / "passing.csv"
RUSHING_CSV = RAW_STATS / "rushing.csv"
RECEIVING_CSV = RAW_STATS / "receiving.csv"
RETURNING_CSV = RAW_STATS / "returning.csv"
DEFENSIVE_CSV = RAW_STATS / "defensive.csv"
KICKING_CSV = RAW_STATS / "kicking.csv"
KICKOFFS_PUNTS_CSV = RAW_STATS / "kickoffs_punts.csv"
SCORING_CSV = RAW_STATS / "scoring.csv"

OUT_CSV = PROCESSED_DIR / "player_projections.csv"

# ---------- Helpers ----------

def safe_div(a, b):
    try:
        if b and float(b) != 0:
            return float(a) / float(b)
        return 0.0
    except Exception:
        return 0.0

def get_col(df, row, col, default=0.0):
    try:
        v = row.get(col, default)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default

def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower-cased, stripped column names (but keep original df intact for file integrity)."""
    x = df.copy()
    x.columns = [c.strip() for c in x.columns]
    return x

def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        df = lower_cols(df)
        return df
    return pd.DataFrame()

# Very light opponent adjustment scaffolding (kept simple and robust to columns present)
def opponent_pass_factor(op_team_row: pd.Series) -> float:
    # Use defense_PYDS/G if available (higher -> tougher on pass yards allowed? Inverse effect)
    val = 0.0
    if isinstance(op_team_row, pd.Series) and "defense_PYDS/G" in op_team_row.index:
        try:
            val = float(op_team_row["defense_PYDS/G"])
        except Exception:
            val = 0.0
    # Normalize around a notional league mean ~220
    if val <= 0:
        return 1.0
    return max(0.85, min(1.15, 220.0 / val))

def opponent_rush_factor(op_team_row: pd.Series) -> float:
    # Use defense_RYDS/G if available
    val = 0.0
    if isinstance(op_team_row, pd.Series) and "defense_RYDS/G" in op_team_row.index:
        try:
            val = float(op_team_row["defense_RYDS/G"])
        except Exception:
            val = 0.0
    # Normalize around a notional league mean ~112
    if val <= 0:
        return 1.0
    return max(0.85, min(1.15, 112.0 / val))

def env_pass_factor(dome: bool, temp_f: float, wind_mph: float, chance_of_rain: float) -> float:
    if dome:
        return 1.0
    f = 1.0
    # wind penalties
    if wind_mph >= 20: f *= 0.90
    elif wind_mph >= 15: f *= 0.94
    elif wind_mph >= 10: f *= 0.97
    # cold penalty
    if temp_f <= 32: f *= 0.94
    elif temp_f <= 40: f *= 0.97
    # rain chance penalty
    if chance_of_rain >= 70: f *= 0.94
    elif chance_of_rain >= 40: f *= 0.97
    return f

def env_rush_factor(dome: bool, temp_f: float, wind_mph: float, chance_of_rain: float) -> float:
    if dome:
        return 1.0
    f = 1.0
    # bad weather nudges run usage a bit
    bump = 0.0
    if wind_mph >= 15: bump += 0.03
    if temp_f <= 40:   bump += 0.03
    if chance_of_rain >= 50: bump += 0.02
    return 1.0 + min(0.08, bump)

def poisson_anytime_prob(lam: float) -> float:
    lam = max(0.0, float(lam))
    try:
        return 1.0 - math.exp(-lam)
    except OverflowError:
        return 1.0

# ---------- Load inputs ----------
passing = load_csv_or_empty(PASSING_CSV)
rushing = load_csv_or_empty(RUSHING_CSV)
receiving = load_csv_or_empty(RECEIVING_CSV)
returning = load_csv_or_empty(RETURNING_CSV)
defensive = load_csv_or_empty(DEFENSIVE_CSV)
kicking = load_csv_or_empty(KICKING_CSV)
kickpunts = load_csv_or_empty(KICKOFFS_PUNTS_CSV)
scoring = load_csv_or_empty(SCORING_CSV)

games = load_csv_or_empty(GAMES_CSV)
team_stadiums = load_csv_or_empty(TEAM_STADIUMS_CSV)
team_metrics = load_csv_or_empty(NFL_TEAM_METRICS)

# Normalize TEAM lookups
def norm_team_col(df: pd.DataFrame) -> pd.DataFrame:
    if "TEAM" in df.columns:
        df["TEAM"] = df["TEAM"].astype(str)
    if "team" in df.columns:
        df["team"] = df["team"].astype(str)
    return df

for d in (passing, rushing, receiving, returning, defensive, kicking, kickpunts, scoring, team_metrics, team_stadiums, games):
    norm_team_col(d)

# Build schedule map from games.csv
# Each row is one game; both teams share same environment (home stadium)
schedule = {}
if not games.empty:
    # ensure required columns exist with defaults if missing
    for col in ["home_team", "away_team", "temp_f", "wind_mph", "chance_of_rain", "Stadium"]:
        if col not in games.columns:
            games[col] = None
    # get dome from team_stadiums by STADIUM or TEAM
    domes_by_team = {}
    if not team_stadiums.empty:
        # dome is "yes"/"no"
        tmp = team_stadiums[["TEAM", "dome"]].copy()
        tmp["TEAM"] = tmp["TEAM"].astype(str)
        tmp["dome"] = tmp["dome"].astype(str).str.strip().str.lower()
        domes_by_team = dict(zip(tmp["TEAM"], tmp["dome"]))

    for _, g in games.iterrows():
        home = str(g.get("home_team", "") or "")
        away = str(g.get("away_team", "") or "")
        temp_f = float(g.get("temp_f", 0.0) or 0.0)
        wind_mph = float(g.get("wind_mph", 0.0) or 0.0)
        cor = float(g.get("chance_of_rain", 0.0) or 0.0)

        # Dome rule: decide from home team dome status
        dome_val = False
        if home in domes_by_team:
            dome_val = (domes_by_team[home] == "yes")

        # map
        if home:
            schedule[home] = {"opponent": away, "is_home": 1, "temp_f": temp_f, "wind_mph": wind_mph, "dome": dome_val, "chance_of_rain": cor}
        if away:
            schedule[away] = {"opponent": home, "is_home": 0, "temp_f": temp_f, "wind_mph": wind_mph, "dome": dome_val, "chance_of_rain": cor}

# Opponent metrics by team
team_metrics_map = {}
if not team_metrics.empty and "TEAM" in team_metrics.columns:
    for _, r in team_metrics.iterrows():
        team_metrics_map[str(r["TEAM"])] = r

# League means for shrinkage (non-pass TD rates)
def league_means_for_nonpass():
    # compute per-game means for the categories we actually use
    rush_td_pg = 0.0
    rec_td_pg = 0.0
    kret_td_pg = 0.0
    pret_td_pg = 0.0
    def_td_pg = 0.0

    n_rush = len(rushing) if not rushing.empty else 0
    if n_rush:
        rush_td_pg = (rushing["RTD"] / rushing["GP"]).replace([pd.NA, pd.NaT], 0).fillna(0).astype(float).clip(lower=0).mean()

    n_recv = len(receiving) if not receiving.empty else 0
    if n_recv:
        rec_td_pg = (receiving["RECTD"] / receiving["GP"]).replace([pd.NA, pd.NaT], 0).fillna(0).astype(float).clip(lower=0).mean()

    n_ret = len(returning) if not returning.empty else 0
    if n_ret:
        kret_td_pg = (returning["K_RET_TD"] / returning["GP"]).replace([pd.NA, pd.NaT], 0).fillna(0).astype(float).clip(lower=0).mean()
        pret_td_pg = (returning["P_RET_TD"] / returning["GP"]).replace([pd.NA, pd.NaT], 0).fillna(0).astype(float).clip(lower=0).mean()

    # Defensive TDs: use INTTD + TD if present; otherwise 0
    n_def = len(defensive) if not defensive.empty else 0
    if n_def:
        def_td_pg = ((defensive.get("INTTD", 0).fillna(0) + defensive.get("TD", 0).fillna(0)) / defensive["GP"]).astype(float).clip(lower=0).mean()

    league_nonpass_td_pg = rush_td_pg + rec_td_pg + kret_td_pg + pret_td_pg + def_td_pg
    return max(0.0, float(league_nonpass_td_pg))

LEAGUE_NONPASS_TD_PG = league_means_for_nonpass()

# Union of players across all files
all_players = set()
for df in (passing, rushing, receiving, returning, defensive, kicking, kickpunts, scoring):
    if not df.empty and "PLAYER" in df.columns:
        all_players.update(df["PLAYER"].astype(str).tolist())

def row_by_player(df: pd.DataFrame, player: str):
    if df.empty or "PLAYER" not in df.columns:
        return None
    rows = df[df["PLAYER"] == player]
    if rows.empty:
        return None
    return rows.iloc[0]

def team_for_row(row: pd.Series) -> str:
    if row is None:
        return ""
    # TEAM or Team columns (present in files you provided)
    for c in ("TEAM", "Team", "team"):
        if c in row.index:
            return str(row[c])
    return ""

def per_game_vals(row: pd.Series, fields: dict) -> dict:
    """
    fields: { "output_name": "CSV_COLUMN_NAME" }
    Returns per-game values: value_of_column / GP
    """
    out = {}
    if row is None:
        for k in fields.keys():
            out[k] = 0.0
        out["GP"] = 0.0
        return out
    gp = float(get_col(row, "GP", 0.0))
    out["GP"] = gp
    for out_name, col in fields.items():
        v = float(get_col(row, col, 0.0))
        out[out_name] = safe_div(v, gp) if gp > 0 else 0.0
    return out

# ---------- Build projections ----------
records = []

for player in sorted(all_players):
    # pull each row if present
    prow = row_by_player(passing, player)
    rrow = row_by_player(rushing, player)
    recrow = row_by_player(receiving, player)
    retrow = row_by_player(returning, player)
    drow = row_by_player(defensive, player)
    krow = row_by_player(kicking, player)
    kprow = row_by_player(kickpunts, player)
    scrow = row_by_player(scoring, player)

    # choose team: prefer the first row that has TEAM
    team = ""
    for rr in (prow, rrow, recrow, retrow, drow, krow, kprow, scrow):
        t = team_for_row(rr)
        if t:
            team = t
            break

    # schedule / opponent & env
    game_info = schedule.get(team, {})
    opponent = str(game_info.get("opponent", ""))
    is_home = int(game_info.get("is_home", 0))
    temp_f = float(game_info.get("temp_f", 0.0))
    wind_mph = float(game_info.get("wind_mph", 0.0))
    dome = bool(game_info.get("dome", False))
    chance_of_rain = float(game_info.get("chance_of_rain", 0.0))

    # opponent metrics row
    op_row = team_metrics_map.get(opponent, pd.Series(dtype=float))

    # --- Passing per game ---
    pass_pg = {"py_pg": 0.0, "ptd_pg": 0.0, "pint_pg": 0.0, "GP": 0.0}
    if prow is not None:
        pass_pg = per_game_vals(
            prow,
            {
                "py_pg": "PYDS",
                "ptd_pg": "PTD",
                "pint_pg": "INT",
            },
        )
        # For yards we prefer the per-game column if present (PYDS/G)
        if "PYDS/G" in prow.index:
            try:
                pass_pg["py_pg"] = float(prow["PYDS/G"])
            except Exception:
                pass_pg["py_pg"] = pass_pg["py_pg"]

    # --- Rushing per game ---
    rush_pg = {"ry_pg": 0.0, "rtd_pg": 0.0, "GP": 0.0}
    if rrow is not None:
        rush_pg = per_game_vals(
            rrow,
            {
                "ry_pg": "RYDS",
                "rtd_pg": "RTD",
            },
        )
        if "RYDS/G" in rrow.index:
            try:
                rush_pg["ry_pg"] = float(rrow["RYDS/G"])
            except Exception:
                rush_pg["ry_pg"] = rush_pg["ry_pg"]

    # --- Receiving per game ---
    recv_pg = {"recy_pg": 0.0, "rectd_pg": 0.0, "tgt_pg": 0.0, "rec_pg": 0.0, "yac_pg": 0.0, "GP": 0.0}
    if recrow is not None:
        recv_pg = per_game_vals(
            recrow,
            {
                "recy_pg": "RECYDS",
                "rectd_pg": "RECTD",
                "tgt_pg": "TGT",
                "rec_pg": "REC",
                "yac_pg": "YAC",
            },
        )
        if "YDS/G" in recrow.index:
            try:
                recv_pg["recy_pg"] = float(recrow["YDS/G"])
            except Exception:
                recv_pg["recy_pg"] = recv_pg["recy_pg"]

    # --- Returning per game ---
    ret_pg = {"kr_yds_pg": 0.0, "kret_td_pg": 0.0, "pr_yds_pg": 0.0, "pret_td_pg": 0.0, "GP": 0.0}
    if retrow is not None:
        ret_pg = per_game_vals(
            retrow,
            {
                "kr_yds_pg": "K_RET_YDS",
                "kret_td_pg": "K_RET_TD",
                "pr_yds_pg": "P_RET_YDS",
                "pret_td_pg": "P_RET_TD",
            },
        )

    # --- Defensive per game ---
    def_pg = {"tackles_pg": 0.0, "sacks_pg": 0.0, "def_int_pg": 0.0, "def_td_pg": 0.0, "GP": 0.0}
    if drow is not None:
        def_pg = per_game_vals(
            drow,
            {
                "tackles_pg": "TCKL",   # using total tackles column from your file
                "sacks_pg": "SCK",
                "def_int_pg": "DEF_INT",
                # TD columns: INTTD and TD (total) may exist
            },
        )
        gp_d = float(get_col(drow, "GP", 0.0))
        inttd = float(get_col(drow, "INTTD", 0.0))
        any_td = float(get_col(drow, "TD", 0.0))
        def_td = 0.0
        if gp_d > 0:
            # Use provided INTTD and TD if present; TD may already include INTTD but we will sum conservatively
            def_td = safe_div(inttd, gp_d) + safe_div(any_td, gp_d)
        def_pg["def_td_pg"] = def_td

    # --- Kicking per game ---
    kick_pg = {"fgm_pg": 0.0, "fga_pg": 0.0, "xpm_pg": 0.0, "xpa_pg": 0.0, "fg_pct": None}
    if krow is not None:
        gp_k = float(get_col(krow, "GP", 0.0))
        fgm = float(get_col(krow, "FGM", 0.0))
        fga = float(get_col(krow, "FGA", 0.0))
        xpm = float(get_col(krow, "XPM", 0.0))
        xpa = float(get_col(krow, "XPA", 0.0))
        fg_pct = get_col(krow, "FG_PCT", None)
        kick_pg["fgm_pg"] = safe_div(fgm, gp_k)
        kick_pg["fga_pg"] = safe_div(fga, gp_k)
        kick_pg["xpm_pg"] = safe_div(xpm, gp_k)
        kick_pg["xpa_pg"] = safe_div(xpa, gp_k)
        try:
            kick_pg["fg_pct"] = float(fg_pct) if fg_pct is not None and not pd.isna(fg_pct) else None
        except Exception:
            kick_pg["fg_pct"] = None

    # --- Kickoffs/Punts (use as-is, not per-game for the %/avg columns) ---
    tb_pct = None
    net_avg = None
    p_avg = None
    k_avg = None
    if kprow is not None:
        # Column names exactly as provided (with spaces and hyphens)
        if "TB %" in kprow.index:
            tb_pct = kprow["TB %"]
        if "NET AVG" in kprow.index:
            net_avg = kprow["NET AVG"]
        if "P-AVG" in kprow.index:
            p_avg = kprow["P-AVG"]
        if "K-AVG" in kprow.index:
            k_avg = kprow["K-AVG"]

    # ---------- Apply opponent + environment multipliers ----------
    # Passing env/opp
    pf = env_pass_factor(dome, temp_f, wind_mph, chance_of_rain) * opponent_pass_factor(op_row)
    # Rushing env/opp
    rf = env_rush_factor(dome, temp_f, wind_mph, chance_of_rain) * opponent_rush_factor(op_row)

    proj_pass_yds = pass_pg["py_pg"] * pf
    proj_pass_td  = pass_pg["ptd_pg"] * pf
    proj_int      = pass_pg["pint_pg"]  # leave INT unadjusted by env (optional tweak)

    proj_rush_yds = rush_pg["ry_pg"] * rf
    proj_rush_td  = rush_pg["rtd_pg"] * rf

    # Receiving yards/TDs can be slightly affected by pass environment as well
    proj_rec_yards = recv_pg["recy_pg"] * pf
    proj_rec_td    = recv_pg["rectd_pg"] * pf
    proj_targets   = recv_pg["tgt_pg"]
    proj_rec       = recv_pg["rec_pg"]
    proj_yac       = recv_pg["yac_pg"]

    # Returns
    proj_kr_yds = ret_pg["kr_yds_pg"]
    proj_kr_td  = ret_pg["kret_td_pg"]
    proj_pr_yds = ret_pg["pr_yds_pg"]
    proj_pr_td  = ret_pg["pret_td_pg"]

    # Defensive
    proj_tackles = def_pg["tackles_pg"]
    proj_sacks   = def_pg["sacks_pg"]
    proj_def_int = def_pg["def_int_pg"]

    # Kicking (already per game)
    proj_fgm = kick_pg["fgm_pg"]
    proj_fga = kick_pg["fga_pg"]
    proj_xpm = kick_pg["xpm_pg"]
    proj_xpa = kick_pg["xpa_pg"]
    proj_fg_pct = kick_pg["fg_pct"]

    # Kickoffs/Punts point-in-time marks
    proj_tb_pct = tb_pct
    proj_net_avg = net_avg
    proj_punt_avg = p_avg
    proj_kick_avg = k_avg

    # ---------- Anytime TD model ----------
    # Baseline non-pass TD per-game from components the player actually has
    baseline_components = []
    if rrow is not None:
        baseline_components.append(proj_rush_td)
    if recrow is not None:
        baseline_components.append(proj_rec_td)
    if retrow is not None:
        baseline_components.append(proj_kr_td + proj_pr_td)
    if drow is not None:
        baseline_components.append(def_pg["def_td_pg"])

    baseline_raw = sum([c for c in baseline_components if pd.notna(c)])
    # Shrink toward league mean ONLY if baseline_raw > 0 (do NOT lift true zeros)
    gp_for_shrink = 4.0  # small sample prior
    if baseline_raw > 0:
        baseline_shrunk = ((gp_for_shrink * baseline_raw) + (4.0 * LEAGUE_NONPASS_TD_PG)) / (gp_for_shrink + 4.0)
    else:
        baseline_shrunk = 0.0

    # Very light home edge (skill players) – only if not a kicker/defender only
    is_skillish = (rrow is not None) or (recrow is not None) or (retrow is not None)
    home_mult = 1.05 if (is_skillish and is_home == 1) else 1.0

    # Final lambda for anytime TD (non-passing only)
    expected_anytime_td = max(0.0, baseline_shrunk) * home_mult
    anytime_td_prob = poisson_anytime_prob(expected_anytime_td)

    # ---------- Assemble row ----------
    rec = {
        "player": player,
        "team": team,
        "opponent": opponent,
        "is_home": is_home,
        "temp_f": round(temp_f, 1),
        "wind_mph": round(wind_mph, 1),
        "dome": "yes" if dome else "no",

        "proj_pass_yds": round(proj_pass_yds, 1) if proj_pass_yds else "",
        "proj_pass_td": round(proj_pass_td, 3) if proj_pass_td else "",
        "proj_int": round(proj_int, 3) if proj_int else "",

        "proj_rush_yds": round(proj_rush_yds, 1) if proj_rush_yds else "",
        "proj_rush_td": round(proj_rush_td, 3) if proj_rush_td else "",

        "proj_rec_yards": round(proj_rec_yards, 1) if proj_rec_yards else "",
        "proj_rec_td": round(proj_rec_td, 3) if proj_rec_td else "",
        "proj_targets": round(proj_targets, 1) if proj_targets else "",
        "proj_rec": round(proj_rec, 1) if proj_rec else "",
        "proj_yac": round(proj_yac, 1) if proj_yac else "",

        "proj_kr_yds": round(proj_kr_yds, 1) if proj_kr_yds else "",
        "proj_kr_td": round(proj_kr_td, 3) if proj_kr_td else "",
        "proj_pr_yds": round(proj_pr_yds, 1) if proj_pr_yds else "",
        "proj_pr_td": round(proj_pr_td, 3) if proj_pr_td else "",

        "proj_fgm": round(proj_fgm, 2) if proj_fgm else "",
        "proj_fga": round(proj_fga, 2) if proj_fga else "",
        "proj_xpm": round(proj_xpm, 2) if proj_xpm else "",
        "proj_xpa": round(proj_xpa, 2) if proj_xpa else "",
        "proj_fg_pct": round(proj_fg_pct, 1) if (proj_fg_pct is not None and proj_fg_pct != "") else "",

        "proj_tackles": round(proj_tackles, 2) if proj_tackles else "",
        "proj_sacks": round(proj_sacks, 2) if proj_sacks else "",
        "proj_def_int": round(proj_def_int, 2) if proj_def_int else "",

        "proj_tb_pct": proj_tb_pct if (proj_tb_pct is not None and proj_tb_pct != "") else "",
        "proj_net_avg": proj_net_avg if (proj_net_avg is not None and proj_net_avg != "") else "",
        "proj_punt_avg": proj_punt_avg if (proj_punt_avg is not None and proj_punt_avg != "") else "",
        "proj_kick_avg": proj_kick_avg if (proj_kick_avg is not None and proj_kick_avg != "") else "",

        "expected_anytime_td": round(expected_anytime_td, 4),
        "anytime_td_prob": round(anytime_td_prob, 4),
    }

    records.append(rec)

# ---------- Output ----------
os.makedirs(PROCESSED_DIR, exist_ok=True)
cols = [
    "player","team","opponent","is_home","temp_f","wind_mph","dome",
    "proj_pass_yds","proj_pass_td","proj_int",
    "proj_rush_yds","proj_rush_td",
    "proj_rec_yards","proj_rec_td","proj_targets","proj_rec","proj_yac",
    "proj_kr_yds","proj_kr_td","proj_pr_yds","proj_pr_td",
    "proj_fgm","proj_fga","proj_xpm","proj_xpa","proj_fg_pct",
    "proj_tackles","proj_sacks","proj_def_int",
    "proj_tb_pct","proj_net_avg","proj_punt_avg","proj_kick_avg",
    "expected_anytime_td","anytime_td_prob",
]

pd.DataFrame.from_records(records, columns=cols).to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}")

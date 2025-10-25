#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
from pathlib import Path

# ---------- Paths ----------
DATA_DIR = Path("data")
RAW_STATS = DATA_DIR / "raw" / "stats"
PROCESSED_DIR = DATA_DIR / "processed"

TEAM_STADIUMS_CSV = DATA_DIR / "team_stadiums.csv"
GAMES_CSV = PROCESSED_DIR / "games.csv"
NFL_TEAM_METRICS = PROCESSED_DIR / "nfl_unified_with_metrics.csv"

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
        b = float(b)
        if b != 0:
            return float(a) / b
        return 0.0
    except Exception:
        return 0.0

def val(row, col, default=0.0):
    if row is None:
        return default
    try:
        v = row.get(col, default)
    except Exception:
        v = default
    if pd.isna(v):
        return default
    return v

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def team_from_row(r: pd.Series) -> str:
    if r is None: return ""
    for c in ("TEAM", "Team", "team"):
        if c in r.index:
            v = r[c]
            return "" if pd.isna(v) else str(v)
    return ""

def poisson_anytime_prob(lam: float) -> float:
    lam = max(0.0, float(lam))
    try:
        return 1.0 - math.exp(-lam)
    except OverflowError:
        return 1.0

# Opponent/context factors (kept conservative)
def opponent_pass_factor(team_row: pd.Series) -> float:
    if isinstance(team_row, pd.Series) and "defense_PYDS/G" in team_row.index:
        try:
            ypg = float(team_row["defense_PYDS/G"])
            if ypg > 0:
                return max(0.85, min(1.15, 220.0 / ypg))
        except Exception:
            pass
    return 1.0

def opponent_rush_factor(team_row: pd.Series) -> float:
    if isinstance(team_row, pd.Series) and "defense_RYDS/G" in team_row.index:
        try:
            ypg = float(team_row["defense_RYDS/G"])
            if ypg > 0:
                return max(0.85, min(1.15, 112.0 / ypg))
        except Exception:
            pass
    return 1.0

def env_pass_factor(dome: bool, temp_f: float, wind_mph: float, cor: float) -> float:
    if dome: return 1.0
    f = 1.0
    # wind
    if wind_mph >= 20: f *= 0.90
    elif wind_mph >= 15: f *= 0.94
    elif wind_mph >= 10: f *= 0.97
    # cold
    if temp_f <= 32: f *= 0.94
    elif temp_f <= 40: f *= 0.97
    # precip
    if cor >= 70: f *= 0.94
    elif cor >= 40: f *= 0.97
    return f

def env_rush_factor(dome: bool, temp_f: float, wind_mph: float, cor: float) -> float:
    if dome: return 1.0
    bump = 0.0
    if wind_mph >= 15: bump += 0.03
    if temp_f <= 40:   bump += 0.03
    if cor >= 50:      bump += 0.02
    return 1.0 + min(0.08, bump)

# ---------- Load data ----------
passing   = load_csv(PASSING_CSV)
rushing   = load_csv(RUSHING_CSV)
receiving = load_csv(RECEIVING_CSV)
returning = load_csv(RETURNING_CSV)
defensive = load_csv(DEFENSIVE_CSV)
kicking   = load_csv(KICKING_CSV)
kickpunts = load_csv(KICKOFFS_PUNTS_CSV)
scoring   = load_csv(SCORING_CSV)

games         = load_csv(GAMES_CSV)
team_stadiums = load_csv(TEAM_STADIUMS_CSV)
team_metrics  = load_csv(NFL_TEAM_METRICS)

# ---------- Build schedule & dome map ----------
domes_by_team = {}
if not team_stadiums.empty and "TEAM" in team_stadiums.columns and "dome" in team_stadiums.columns:
    t = team_stadiums[["TEAM", "dome"]].copy()
    t["TEAM"] = t["TEAM"].astype(str)
    t["dome"] = t["dome"].astype(str).str.strip().str.lower()
    domes_by_team = dict(zip(t["TEAM"], t["dome"]))

schedule = {}
if not games.empty:
    need_cols = ["home_team", "away_team", "temp_f", "wind_mph", "chance_of_rain"]
    for c in need_cols:
        if c not in games.columns:
            games[c] = 0.0 if c != "home_team" and c != "away_team" else ""
    for _, g in games.iterrows():
        home = str(g.get("home_team", "") or "")
        away = str(g.get("away_team", "") or "")
        temp_f = float(g.get("temp_f", 0.0) or 0.0)
        wind_mph = float(g.get("wind_mph", 0.0) or 0.0)
        cor = float(g.get("chance_of_rain", 0.0) or 0.0)
        dome = domes_by_team.get(home, "no") == "yes"
        if home:
            schedule[home] = {"opponent": away, "is_home": 1, "temp_f": temp_f, "wind_mph": wind_mph, "dome": dome, "cor": cor}
        if away:
            schedule[away] = {"opponent": home, "is_home": 0, "temp_f": temp_f, "wind_mph": wind_mph, "dome": dome, "cor": cor}

# opponent metrics map
team_metrics_map = {}
if not team_metrics.empty and "TEAM" in team_metrics.columns:
    for _, r in team_metrics.iterrows():
        team_metrics_map[str(r["TEAM"])] = r

# ---------- League mean for shrinkage (non-passing) ----------
def league_nonpass_td_pg():
    vals = []

    if not rushing.empty and "RTD" in rushing.columns and "GP" in rushing.columns:
        vals.append((rushing["RTD"] / rushing["GP"]).fillna(0).clip(lower=0))

    if not receiving.empty and "RECTD" in receiving.columns and "GP" in receiving.columns:
        vals.append((receiving["RECTD"] / receiving["GP"]).fillna(0).clip(lower=0))

    if not returning.empty and "K_RET_TD" in returning.columns and "GP" in returning.columns:
        vals.append((returning["K_RET_TD"] / returning["GP"]).fillna(0).clip(lower=0))
    if not returning.empty and "P_RET_TD" in returning.columns and "GP" in returning.columns:
        vals.append((returning["P_RET_TD"] / returning["GP"]).fillna(0).clip(lower=0))

    if not defensive.empty and "GP" in defensive.columns:
        # best-effort defensive TD rate: INTTD + TD if present
        d_inttd = defensive.get("INTTD", pd.Series([0]*len(defensive))).fillna(0)
        d_td    = defensive.get("TD", pd.Series([0]*len(defensive))).fillna(0)
        vals.append((d_inttd + d_td) / defensive["GP"])

    if not vals:
        return 0.0
    combo = pd.concat(vals, axis=0)
    return float(combo.mean())

LEAGUE_NONPASS_TD_PG = league_nonpass_td_pg()

# ---------- Per-player lookups ----------
def row_for(df: pd.DataFrame, player: str):
    if df.empty or "PLAYER" not in df.columns: return None
    m = df[df["PLAYER"] == player]
    return None if m.empty else m.iloc[0]

def per_game_from(row: pd.Series, num_col: str) -> float:
    if row is None: return 0.0
    gp = float(val(row, "GP", 0.0))
    n  = float(val(row, num_col, 0.0))
    return safe_div(n, gp) if gp > 0 else 0.0

# fallback from scoring.csv for TD components
def scoring_pg(player: str, col: str) -> float:
    r = row_for(scoring, player)
    if r is None or "GP" not in r.index or col not in r.index: return 0.0
    return per_game_from(r, col)

# ---------- Build player universe ----------
all_players = set()
for df in (passing, rushing, receiving, returning, defensive, kicking, kickpunts, scoring):
    if not df.empty and "PLAYER" in df.columns:
        all_players.update(df["PLAYER"].astype(str))

# ---------- Projections ----------
records = []

for player in sorted(all_players):
    # rows
    prow  = row_for(passing,   player)
    rrow  = row_for(rushing,   player)
    recrow= row_for(receiving, player)
    retrow= row_for(returning, player)
    drow  = row_for(defensive, player)
    krow  = row_for(kicking,   player)
    kprow = row_for(kickpunts, player)

    # team
    team = ""
    for r in (prow, rrow, recrow, retrow, drow, krow, kprow):
        t = team_from_row(r)
        if t:
            team = t
            break

    # schedule/env
    gi = schedule.get(team, {})
    opponent = gi.get("opponent", "")
    is_home  = int(gi.get("is_home", 0))
    temp_f   = float(gi.get("temp_f", 0.0))
    wind_mph = float(gi.get("wind_mph", 0.0))
    cor      = float(gi.get("cor", 0.0))
    dome     = bool(gi.get("dome", False))

    # opponent context row
    op_row = team_metrics_map.get(opponent, pd.Series(dtype=float))

    # per-game core stats
    # Passing
    py_pg  = per_game_from(prow, "PYDS")
    ptd_pg = per_game_from(prow, "PTD")
    pint_pg= per_game_from(prow, "INT")
    # prefer explicit per-game if present
    if prow is not None and "PYDS/G" in prow.index:
        try: py_pg = float(prow["PYDS/G"])
        except Exception: pass

    # Rushing
    ry_pg  = per_game_from(rrow, "RYDS")
    rtd_pg = per_game_from(rrow, "RTD")
    if rrow is not None and "RYDS/G" in rrow.index:
        try: ry_pg = float(rrow["RYDS/G"])
        except Exception: pass

    # Receiving
    recy_pg   = per_game_from(recrow, "RECYDS")
    rectd_pg  = per_game_from(recrow, "RECTD")
    tgt_pg    = per_game_from(recrow, "TGT")
    rec_pg    = per_game_from(recrow, "REC")
    yac_pg    = per_game_from(recrow, "YAC")
    if recrow is not None and "YDS/G" in recrow.index:
        try: recy_pg = float(recrow["YDS/G"])
        except Exception: pass

    # If we still have 0 receiving TDs per game but scoring has REC_TD, use that as fallback
    if rectd_pg == 0.0:
        rectd_pg = scoring_pg(player, "REC_TD")

    # If rush TDs per game 0 but scoring has RUSH_TD, use it
    if rtd_pg == 0.0:
        rtd_pg = scoring_pg(player, "RUSH_TD")

    # Returns
    kr_yds_pg  = per_game_from(retrow, "K_RET_YDS")
    kret_td_pg = per_game_from(retrow, "K_RET_TD")
    pr_yds_pg  = per_game_from(retrow, "P_RET_YDS")
    pret_td_pg = per_game_from(retrow, "P_RET_TD")

    # Defense
    tackles_pg = per_game_from(drow, "TCKL")
    sacks_pg   = per_game_from(drow, "SCK")
    def_int_pg = per_game_from(drow, "DEF_INT")
    # defensive TD best-effort: INTTD + TD columns if present
    def_td_pg = 0.0
    if drow is not None and "GP" in drow.index:
        gp_d = float(val(drow, "GP", 0.0))
        if gp_d > 0:
            def_td_pg = safe_div(float(val(drow, "INTTD", 0.0)), gp_d) + safe_div(float(val(drow, "TD", 0.0)), gp_d)

    # Kicking (per-game)
    fgm_pg = fga_pg = xpm_pg = xpa_pg = 0.0
    fg_pct = None
    if krow is not None:
        gp_k = float(val(krow, "GP", 0.0))
        fgm_pg = safe_div(float(val(krow, "FGM", 0.0)), gp_k)
        fga_pg = safe_div(float(val(krow, "FGA", 0.0)), gp_k)
        xpm_pg = safe_div(float(val(krow, "XPM", 0.0)), gp_k)
        xpa_pg = safe_div(float(val(krow, "XPA", 0.0)), gp_k)
        fp = val(krow, "FG_PCT", None)
        try:
            fg_pct = float(fp) if fp is not None and not pd.isna(fp) else None
        except Exception:
            fg_pct = None

    # Kickoffs/Punts (single-game descriptors)
    tb_pct = kickpunts.loc[kickpunts["PLAYER"] == player, "TB %"].iloc[0] if (not kickpunts.empty and "PLAYER" in kickpunts.columns and "TB %" in kickpunts.columns and not kickpunts[kickpunts["PLAYER"] == player].empty) else ""
    net_avg = kickpunts.loc[kickpunts["PLAYER"] == player, "NET AVG"].iloc[0] if (not kickpunts.empty and "NET AVG" in kickpunts.columns and not kickpunts[kickpunts["PLAYER"] == player].empty) else ""
    p_avg = kickpunts.loc[kickpunts["PLAYER"] == player, "P-AVG"].iloc[0] if (not kickpunts.empty and "P-AVG" in kickpunts.columns and not kickpunts[kickpunts["PLAYER"] == player].empty) else ""
    k_avg = kickpunts.loc[kickpunts["PLAYER"] == player, "K-AVG"].iloc[0] if (not kickpunts.empty and "K-AVG" in kickpunts.columns and not kickpunts[kickpunts["PLAYER"] == player].empty) else ""

    # ---------- Environment & opponent adjustments ----------
    pf = env_pass_factor(dome, temp_f, wind_mph, cor) * opponent_pass_factor(op_row)
    rf = env_rush_factor(dome, temp_f, wind_mph, cor) * opponent_rush_factor(op_row)

    proj_pass_yds = py_pg * pf
    proj_pass_td  = ptd_pg * pf
    proj_int      = pint_pg  # can later blend env if desired

    proj_rush_yds = ry_pg * rf
    proj_rush_td  = rtd_pg * rf

    proj_rec_yards = recy_pg * pf
    proj_rec_td    = rectd_pg * pf
    proj_targets   = tgt_pg
    proj_rec       = rec_pg
    proj_yac       = yac_pg

    proj_kr_yds = kr_yds_pg
    proj_kr_td  = kret_td_pg
    proj_pr_yds = pr_yds_pg
    proj_pr_td  = pret_td_pg

    # ---------- Anytime TD λ (non-passing only) ----------
    baseline_raw = 0.0
    baseline_raw += max(0.0, proj_rush_td)
    baseline_raw += max(0.0, proj_rec_td)
    baseline_raw += max(0.0, proj_kr_td + proj_pr_td)
    baseline_raw += max(0.0, def_td_pg)

    # DO NOT lift true zeros
    if baseline_raw > 0:
        prior_weight = 4.0
        baseline_shrunk = ((prior_weight * baseline_raw) + (4.0 * LEAGUE_NONPASS_TD_PG)) / (prior_weight + 4.0)
    else:
        baseline_shrunk = 0.0

    # mild home boost for skill roles
    is_skill = (rrow is not None) or (recrow is not None) or (retrow is not None)
    home_mult = 1.05 if (is_skill and is_home == 1) else 1.0

    expected_anytime_td = max(0.0, baseline_shrunk) * home_mult
    anytime_td_prob = poisson_anytime_prob(expected_anytime_td)

    # ---------- Build record (always numeric; no empty strings) ----------
    def r1(x):  return round(float(x), 1)
    def r2(x):  return round(float(x), 2)
    def r3(x):  return round(float(x), 3)
    def rN(x):  return float(x)

    rec_out = {
        "player": player,
        "team": team,
        "opponent": opponent,
        "is_home": int(is_home),
        "temp_f": r1(temp_f),
        "wind_mph": r1(wind_mph),
        "dome": "yes" if dome else "no",

        "proj_pass_yds": r1(proj_pass_yds),
        "proj_pass_td": r3(proj_pass_td),
        "proj_int": r3(proj_int),

        "proj_rush_yds": r1(proj_rush_yds),
        "proj_rush_td": r3(proj_rush_td),

        "proj_rec_yards": r1(proj_rec_yards),
        "proj_rec_td": r3(proj_rec_td),
        "proj_targets": r1(proj_targets),
        "proj_rec": r1(proj_rec),
        "proj_yac": r1(proj_yac),

        "proj_kr_yds": r1(proj_kr_yds),
        "proj_kr_td": r3(proj_kr_td),
        "proj_pr_yds": r1(proj_pr_yds),
        "proj_pr_td": r3(proj_pr_td),

        "proj_fgm": r2(fgm_pg),
        "proj_fga": r2(fga_pg),
        "proj_xpm": r2(xpm_pg),
        "proj_xpa": r2(xpa_pg),
        "proj_fg_pct": (round(fg_pct, 1) if fg_pct is not None else 0.0),

        "proj_tackles": r2(tackles_pg),
        "proj_sacks": r2(sacks_pg),
        "proj_def_int": r2(def_int_pg),

        "proj_tb_pct": rN(tb_pct) if tb_pct != "" else 0.0,
        "proj_net_avg": rN(net_avg) if net_avg != "" else 0.0,
        "proj_punt_avg": rN(p_avg) if p_avg != "" else 0.0,
        "proj_kick_avg": rN(k_avg) if k_avg != "" else 0.0,

        "expected_anytime_td": round(expected_anytime_td, 4),
        "anytime_td_prob": round(anytime_td_prob, 4),
    }

    records.append(rec_out)

# ---------- Save ----------
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

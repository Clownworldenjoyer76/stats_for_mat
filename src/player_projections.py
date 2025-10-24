#!/usr/bin/env python3
"""
Player Projections (per-game) + Anytime TD Scorer

Inputs (exact columns used):
  data/raw/stats/passing.csv    -> PLAYER, TEAM, GP, PYDS, PTD, INT
  data/raw/stats/rushing.csv    -> PLAYER, TEAM, GP, RATT, RYDS, RTD
  data/raw/stats/receiving.csv  -> PLAYER, TEAM, GP, TGT, REC, RECYDS, RECTD, RECAVG, YAC
  data/raw/stats/returning.csv  -> PLAYER, TEAM, GP, K_RET, K_RET_YDS, K_RET_AVG, K_RET_TD, P_RET, P_RET_YDS, P_RET_AVG, P_RET_TD
  data/raw/stats/kicking.csv    -> PLAYER, TEAM, GP, FGM, FGA, FG_PCT, XPM, XPA
  data/raw/stats/defensive.csv  -> PLAYER, TEAM, GP, TCKL, SCK, DEF_INT
  data/raw/stats/kickoffs_punts.csv
                                 -> Player, Team, GP, KO, YDS, K-AVG, TB, TB %, OSKA, OSK, AVG,
                                    PUNTS, P-YDS, P-AVG, P-LNG, IN20, IN20 %, P-TB, P-TB %, BLK, NET AVG
  data/raw/stats/scoring.csv    -> PLAYER, TEAM, GP, TD, PASS_TD, RUSH_TD, REC_TD, K_RET_TD, P_RET_TD, INT_TD

Context:
  data/processed/games.csv
  data/team_stadiums.csv
  data/processed/nfl_unified_with_metrics.csv   (defense_PYDS/G, defense_RYDS/G)

Output:
  data/processed/player_projections.csv
"""

from pathlib import Path
import math
import pandas as pd

# ---------- Paths ----------
BASE_RAW = Path("data/raw/stats")
P_PASS   = BASE_RAW / "passing.csv"
P_RUSH   = BASE_RAW / "rushing.csv"
P_RECV   = BASE_RAW / "receiving.csv"
P_RET    = BASE_RAW / "returning.csv"
P_KICK   = BASE_RAW / "kicking.csv"
P_DEF    = BASE_RAW / "defensive.csv"
P_PUNT   = BASE_RAW / "kickoffs_punts.csv"
P_SCORE  = BASE_RAW / "scoring.csv"

P_GAMES  = Path("data/processed/games.csv")
P_TEAMS  = Path("data/team_stadiums.csv")
P_MET    = Path("data/processed/nfl_unified_with_metrics.csv")
P_OUT    = Path("data/processed/player_projections.csv")

# ---------- weather_bundle (import with safe fallback) ----------
def _fallback_weather_bundle(temp_f, wind_mph, precip_in, dome):
    if str(dome).strip().lower() == "yes":
        return dict(pass_mult=1.0, rush_mult=1.0, to_delta=0.0,
                    rz_mult=1.0, drives_mult=1.0, fg_pct_delta=0.0, net_punt_delta=0.0)
    def _f(x, d):
        try: return float(x)
        except Exception: return d
    t = _f(temp_f, 70.0); w = _f(wind_mph, 0.0); r = _f(precip_in, 0.0)
    if r > 0 or w >= 20: sev = "heavy"
    elif w >= 10:        sev = "moderate"
    elif w >= 5:         sev = "light"
    elif t < 35:         sev = "cold"
    else:                sev = "none"
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

try:
    from src.predict_outcomes import weather_bundle  # type: ignore
except Exception:
    weather_bundle = _fallback_weather_bundle

# ---------- Helpers ----------
def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None

def _norm(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, variants in mapping.items():
        for v in variants:
            if v.lower() in lower_map:
                rename[lower_map[v.lower()]] = canon
                break
    return df.rename(columns=rename)

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def per_game(df: pd.DataFrame, totals: list[str], gp_col: str = "GP") -> None:
    if df is None or gp_col not in df.columns:
        return
    df[gp_col] = pd.to_numeric(df[gp_col], errors="coerce").fillna(0)
    gp = df[gp_col].replace(0, pd.NA)
    for col in totals:
        if col in df.columns:
            df[f"{col}_pg"] = pd.to_numeric(df[col], errors="coerce") / gp

def avg_by_player(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["PLAYER","TEAM"] + cols)
    return df.groupby(["PLAYER","TEAM"], as_index=False)[cols].mean()

def home_mult(is_home: int) -> float:
    return 1.02 if int(is_home or 0) == 1 else 0.98

def _nz(x: object) -> float:
    """Return numeric value or 0.0 (never NaN/None)."""
    try:
        if pd.isna(x):
            return 0.0
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return 0.0

# ---------- Load context ----------
games   = pd.read_csv(P_GAMES)
teams   = pd.read_csv(P_TEAMS)
metrics = pd.read_csv(P_MET)

# Build matchups: team -> opponent (+ weather from home stadium) + dome flag
home_side = games[["home_team","away_team","temp_f","wind_mph","precip_in"]].copy()
home_side["team"] = home_side["home_team"]; home_side["opponent"] = home_side["away_team"]; home_side["is_home"] = 1
away_side = games[["home_team","away_team","temp_f","wind_mph","precip_in"]].copy()
away_side["team"] = away_side["away_team"]; away_side["opponent"] = away_side["home_team"]; away_side["is_home"] = 0
matchups = pd.concat([home_side, away_side], ignore_index=True)[
    ["team","opponent","is_home","temp_f","wind_mph","precip_in"]
]
teams_dome = teams[["TEAM","dome"]].copy()
home_dome = games[["home_team"]].merge(teams_dome, left_on="home_team", right_on="TEAM", how="left").rename(columns={"dome":"home_dome"})
home_dome_map = pd.concat([
    games[["home_team","away_team"]].assign(team=games["home_team"]).merge(home_dome[["home_team","home_dome"]], on="home_team", how="left"),
    games[["home_team","away_team"]].assign(team=games["away_team"]).merge(home_dome[["home_team","home_dome"]], on="home_team", how="left"),
], ignore_index=True)[["team","home_dome"]].drop_duplicates().rename(columns={"home_dome":"dome"})
matchups = matchups.merge(home_dome_map, on="team", how="left")

# Opponent defensive context
metrics = _norm(metrics, {
    "TEAM": ["TEAM","Team","team"],
    "defense_PYDS/G": ["defense_PYDS/G","DEF_PYDS_G","defense_PYDS_g"],
    "defense_RYDS/G": ["defense_RYDS/G","DEF_RYDS_G","defense_RYDS_g"],
})
coerce_numeric(metrics, ["defense_PYDS/G","defense_RYDS/G"])
metrics_ctx = metrics[["TEAM","defense_PYDS/G","defense_RYDS/G"]].rename(columns={"TEAM":"opponent"})
matchups = matchups.merge(metrics_ctx, on="opponent", how="left")

league_avg_pass_def = metrics["defense_PYDS/G"].mean()
league_avg_rush_def = metrics["defense_RYDS/G"].mean()

def def_adj_pass(opp_def):
    if pd.isna(opp_def) or pd.isna(league_avg_pass_def) or league_avg_pass_def == 0:
        return 1.0
    return 1 - ((float(opp_def) - league_avg_pass_def) / league_avg_pass_def) * 0.6

def def_adj_rush(opp_def):
    if pd.isna(opp_def) or pd.isna(league_avg_rush_def) or league_avg_rush_def == 0:
        return 1.0
    return 1 - ((float(opp_def) - league_avg_rush_def) / league_avg_rush_def) * 0.5

# ---------- Load player files, coerce numeric, create per-game ----------
# Passing
pass_df = _read_csv(P_PASS)
if pass_df is not None:
    pass_df = _norm(pass_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    coerce_numeric(pass_df, ["GP","PYDS","PTD","INT"])
    per_game(pass_df, ["PYDS","PTD","INT"], gp_col="GP")
    pass_avg = avg_by_player(pass_df, ["PYDS_pg","PTD_pg","INT_pg"])
else:
    pass_avg = pd.DataFrame(columns=["PLAYER","TEAM","PYDS_pg","PTD_pg","INT_pg"])

# Rushing
rush_df = _read_csv(P_RUSH)
if rush_df is not None:
    rush_df = _norm(rush_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    coerce_numeric(rush_df, ["GP","RATT","RYDS","RTD"])
    per_game(rush_df, ["RATT","RYDS","RTD"], gp_col="GP")
    rush_avg = avg_by_player(rush_df, ["RATT_pg","RYDS_pg","RTD_pg"])
else:
    rush_avg = pd.DataFrame(columns=["PLAYER","TEAM","RATT_pg","RYDS_pg","RTD_pg"])

# Receiving
recv_df = _read_csv(P_RECV)
if recv_df is not None:
    recv_df = _norm(recv_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    coerce_numeric(recv_df, ["GP","TGT","REC","RECYDS","RECTD","RECAVG","YAC"])
    per_game(recv_df, ["TGT","REC","RECYDS","RECTD","YAC"], gp_col="GP")
    recv_avg = avg_by_player(recv_df, ["TGT_pg","REC_pg","RECYDS_pg","RECTD_pg","RECAVG","YAC_pg"])
else:
    recv_avg = pd.DataFrame(columns=["PLAYER","TEAM","TGT_pg","REC_pg","RECYDS_pg","RECTD_pg","RECAVG","YAC_pg"])

# Returning
ret_df = _read_csv(P_RET)
if ret_df is not None:
    ret_df = _norm(ret_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    coerce_numeric(ret_df, ["GP","K_RET","K_RET_YDS","K_RET_AVG","K_RET_TD","P_RET","P_RET_YDS","P_RET_AVG","P_RET_TD"])
    per_game(ret_df, ["K_RET","K_RET_YDS","K_RET_TD","P_RET","P_RET_YDS","P_RET_TD"], gp_col="GP")
    ret_avg = avg_by_player(ret_df, ["K_RET_pg","K_RET_YDS_pg","K_RET_AVG","K_RET_TD_pg",
                                     "P_RET_pg","P_RET_YDS_pg","P_RET_AVG","P_RET_TD_pg"])
else:
    ret_avg = pd.DataFrame(columns=["PLAYER","TEAM","K_RET_pg","K_RET_YDS_pg","K_RET_AVG","K_RET_TD_pg",
                                    "P_RET_pg","P_RET_YDS_pg","P_RET_AVG","P_RET_TD_pg"])

# Kicking
kick_df = _read_csv(P_KICK)
if kick_df is not None:
    kick_df = _norm(kick_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    coerce_numeric(kick_df, ["GP","FGM","FGA","FG_PCT","XPM","XPA"])
    per_game(kick_df, ["FGM","FGA","XPM","XPA"], gp_col="GP")
    kick_avg = avg_by_player(kick_df, ["FGM_pg","FGA_pg","FG_PCT","XPM_pg","XPA_pg"])
else:
    kick_avg = pd.DataFrame(columns=["PLAYER","TEAM","FGM_pg","FGA_pg","FG_PCT","XPM_pg","XPA_pg"])

# Defensive
def_df = _read_csv(P_DEF)
if def_df is not None:
    def_df = _norm(def_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    coerce_numeric(def_df, ["GP","TCKL","SCK","DEF_INT"])
    per_game(def_df, ["TCKL","SCK","DEF_INT"], gp_col="GP")
    def_avg = avg_by_player(def_df, ["TCKL_pg","SCK_pg","DEF_INT_pg"])
else:
    def_avg = pd.DataFrame(columns=["PLAYER","TEAM","TCKL_pg","SCK_pg","DEF_INT_pg"])

# Kickoffs/Punts
punt_df = _read_csv(P_PUNT)
if punt_df is not None:
    punt_df = _norm(punt_df, {
        "PLAYER":["Player","PLAYER","Name"], "TEAM":["Team","TEAM"],
        "KO":["KO"], "K_YDS":["YDS"], "K_AVG":["K-AVG"], "TB":["TB"], "TB_pct":["TB %"],
        "OSKA":["OSKA"], "OSK":["OSK"], "OSK_AVG":["AVG"],
        "PUNTS":["PUNTS"], "P_YDS":["P-YDS"], "P_AVG":["P-AVG"], "P_LNG":["P-LNG"],
        "IN20":["IN20"], "IN20_pct":["IN20 %"], "P_TB":["P-TB"], "P_TB_pct":["P-TB %"],
        "BLK":["BLK"], "NET_AVG":["NET AVG"], "GP":["GP"]
    })
    coerce_numeric(punt_df, ["GP","KO","K_YDS","K_AVG","TB","TB_pct","OSKA","OSK","OSK_AVG",
                             "PUNTS","P_YDS","P_AVG","P_LNG","IN20","IN20_pct","P_TB","P_TB_pct","BLK","NET_AVG"])
    per_game(punt_df, ["KO","K_YDS","PUNTS","P_YDS"], gp_col="GP")
    punt_avg = avg_by_player(punt_df, ["KO_pg","K_YDS_pg","K_AVG","TB","TB_pct","PUNTS_pg","P_YDS_pg","P_AVG","NET_AVG","IN20","IN20_pct"])
else:
    punt_avg = pd.DataFrame(columns=["PLAYER","TEAM","KO_pg","K_YDS_pg","K_AVG","TB","TB_pct","PUNTS_pg","P_YDS_pg","P_AVG","NET_AVG","IN20","IN20_pct"])

# Scoring (baseline non-passing TD rate) + carry GP for shrinkage + league baseline
score_df = _read_csv(P_SCORE)
if score_df is not None:
    score_df = _norm(score_df, {"PLAYER":["PLAYER","Player","Name"], "TEAM":["TEAM","Team"]})
    for c in ["GP","TD","PASS_TD","RUSH_TD","REC_TD","K_RET_TD","P_RET_TD","INT_TD"]:
        if c in score_df.columns:
            score_df[c] = pd.to_numeric(score_df[c], errors="coerce").fillna(0)
    score_df["GP_safe"] = score_df["GP"].replace({0: 1})
    nonpass_td = score_df[["RUSH_TD","REC_TD","K_RET_TD","P_RET_TD","INT_TD"]].sum(axis=1)
    score_df["baseline_nonpass_td_pg"] = nonpass_td / score_df["GP_safe"]
    # league average non-pass TD per game across players with GP>0
    league_td_pg = (nonpass_td[score_df["GP"] > 0] / score_df.loc[score_df["GP"] > 0, "GP"]).mean()
    if pd.isna(league_td_pg) or league_td_pg <= 0:
        league_td_pg = 0.25  # fallback
    score_avg = score_df[["PLAYER","TEAM","baseline_nonpass_td_pg","GP"]].copy()
else:
    league_td_pg = 0.25
    score_avg = pd.DataFrame(columns=["PLAYER","TEAM","baseline_nonpass_td_pg","GP"])

# ---------- Build player universe ----------
players = pass_avg.merge(rush_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(recv_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(ret_avg,  on=["PLAYER","TEAM"], how="outer")
players = players.merge(kick_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(def_avg,  on=["PLAYER","TEAM"], how="outer")
players = players.merge(punt_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(score_avg, on=["PLAYER","TEAM"], how="left")

# Attach matchup row (team/opponent/weather/dome/home)
players = players.merge(matchups, left_on="TEAM", right_on="team", how="left")
players = players[players["opponent"].notna()].copy()

league_avg_pass_def = metrics["defense_PYDS/G"].mean()
league_avg_rush_def = metrics["defense_RYDS/G"].mean()

# ---------- Projections (per-game) ----------
rows = []
for _, r in players.iterrows():
    player = r["PLAYER"]; team = r["TEAM"]; opp = r["opponent"]
    is_home = int(r.get("is_home", 0) or 0)
    hmult = home_mult(is_home)
    wb = weather_bundle(r.get("temp_f"), r.get("wind_mph"), r.get("precip_in"), r.get("dome"))

    p_mult = def_adj_pass(r.get("defense_PYDS/G"))
    r_mult = def_adj_rush(r.get("defense_RYDS/G"))

    out = {
        "player": player, "team": team, "opponent": opp, "is_home": is_home,
        "temp_f": r.get("temp_f"), "wind_mph": r.get("wind_mph"), "dome": r.get("dome"),
        "proj_pass_yds": None, "proj_pass_td": None, "proj_int": None,
        "proj_rush_yds": None, "proj_rush_td": None,
        "proj_rec_yards": None, "proj_rec_td": None, "proj_targets": None, "proj_rec": None, "proj_yac": None,
        "proj_kr_yds": None, "proj_kr_td": None, "proj_pr_yds": None, "proj_pr_td": None,
        "proj_fgm": None, "proj_fga": None, "proj_xpm": None, "proj_xpa": None, "proj_fg_pct": None,
        "proj_tackles": None, "proj_sacks": None, "proj_def_int": None,
        "proj_tb_pct": None, "proj_net_avg": None, "proj_punt_avg": None, "proj_kick_avg": None,
        "expected_anytime_td": None, "anytime_td_prob": None,
    }

    # ----- Passing (per-game) -----
    if pd.notna(r.get("PYDS_pg")):
        pass_yds_pg = _nz(r.get("PYDS_pg")); ptd_pg = _nz(r.get("PTD_pg")); int_pg = _nz(r.get("INT_pg"))
        out["proj_pass_yds"] = round(pass_yds_pg * wb["pass_mult"] * hmult * p_mult, 1)
        out["proj_pass_td"]  = round(ptd_pg      * wb["pass_mult"] * hmult * p_mult, 3)
        out["proj_int"]      = round(int_pg      * (1.00 + max(0.0, wb["to_delta"])), 3)

    # ----- Rushing (per-game) -----
    if pd.notna(r.get("RYDS_pg")):
        rush_yds_pg = _nz(r.get("RYDS_pg")); rtd_pg = _nz(r.get("RTD_pg"))
        out["proj_rush_yds"] = round(rush_yds_pg * wb["rush_mult"] * hmult * r_mult, 1)
        out["proj_rush_td"]  = round(rtd_pg      * wb["rush_mult"] * hmult * r_mult, 3)

    # ----- Receiving (per-game) -----
    if pd.notna(r.get("RECYDS_pg")):
        recyds_pg = _nz(r.get("RECYDS_pg")); rectd_pg = _nz(r.get("RECTD_pg"))
        rec_pg = _nz(r.get("REC_pg")); tgt_pg = _nz(r.get("TGT_pg")); yac_pg = _nz(r.get("YAC_pg"))
        adj = wb["pass_mult"] * hmult * p_mult
        out["proj_rec_yards"] = round(recyds_pg * adj, 1)
        out["proj_rec_td"]    = round(rectd_pg  * adj, 3)
        out["proj_targets"]   = round(tgt_pg * (0.97 + 0.03*wb["drives_mult"]), 1)
        out["proj_rec"]       = round(rec_pg * (0.97 + 0.03*wb["drives_mult"]), 1)
        out["proj_yac"]       = round(yac_pg * (0.98 if wb["pass_mult"] < 1.0 else 1.0), 1)

    # ----- Returns (per-game) -----
    if pd.notna(r.get("K_RET_pg")) or pd.notna(r.get("P_RET_pg")):
        kr_pg = _nz(r.get("K_RET_pg")); kr_avg = _nz(r.get("K_RET_AVG"))
        pr_pg = _nz(r.get("P_RET_pg")); pr_avg = _nz(r.get("P_RET_AVG"))
        pace = wb["drives_mult"]
        wind_eff = 0.98 if wb["fg_pct_delta"] < 0 else 1.0
        out["proj_kr_yds"] = round(kr_pg * kr_avg * pace * wind_eff, 1) if kr_pg > 0 else None
        out["proj_kr_td"]  = _nz(r.get("K_RET_TD_pg"))
        out["proj_pr_yds"] = round(pr_pg * pr_avg * pace * wind_eff, 1) if pr_pg > 0 else None
        out["proj_pr_td"]  = _nz(r.get("P_RET_TD_pg"))

    # ----- Kickers (per-game) -----
    if pd.notna(r.get("FGA_pg")):
        fga_pg = _nz(r.get("FGA_pg")); fgm_pg = _nz(r.get("FGM_pg"))
        xpa_pg = _nz(r.get("XPA_pg")); xpm_pg = _nz(r.get("XPM_pg"))
        fg_pct_val = _nz(r.get("FG_PCT"))
        fg_pct = fg_pct_val if fg_pct_val > 0 else (_nz(fgm_pg) / fga_pg * 100 if fga_pg > 0 else 0.0)
        adj_fg_pct = max(0.0, min(100.0, fg_pct + (wb["fg_pct_delta"] * 100)))
        out["proj_fga"] = round(fga_pg * (0.97 + 0.03*wb["drives_mult"]), 2)
        out["proj_fgm"] = round(out["proj_fga"] * (adj_fg_pct / 100.0), 2)
        out["proj_xpa"] = round(xpa_pg * (0.98 + 0.02*wb["drives_mult"]), 2)
        out["proj_xpm"] = round(min(out["proj_xpa"], xpm_pg * (1 + wb["fg_pct_delta"])), 2)
        out["proj_fg_pct"] = round(adj_fg_pct, 1)

    # ----- Defensive (per-game) -----
    if pd.notna(r.get("TCKL_pg")):
        tckl_pg = _nz(r.get("TCKL_pg")); sck_pg = _nz(r.get("SCK_pg")); dint_pg = _nz(r.get("DEF_INT_pg"))
        pace = wb["drives_mult"]
        out["proj_tackles"] = round(tckl_pg * (0.98 + 0.02*pace), 2)
        out["proj_sacks"]   = round(sck_pg   * (1.00 + 0.05*(1 - wb["pass_mult"])), 3)
        out["proj_def_int"] = round(dint_pg  * (1.00 + max(0.0, wb["to_delta"])), 3)

    # ----- Kickoffs / Punting (per-game where relevant) -----
    if pd.notna(r.get("NET_AVG")) or pd.notna(r.get("K_AVG")) or pd.notna(r.get("P_AVG")):
        if pd.notna(r.get("NET_AVG")):
            out["proj_net_avg"] = round(_nz(r.get("NET_AVG")) + wb["net_punt_delta"], 1)
        if pd.notna(r.get("P_AVG")):
            out["proj_punt_avg"] = round(_nz(r.get("P_AVG")) + wb["net_punt_delta"], 1)
        if pd.notna(r.get("K_AVG")):
            out["proj_kick_avg"] = round(_nz(r.get("K_AVG")) * (1.0 + (wb["fg_pct_delta"] * 0.5)), 1)
        if pd.notna(r.get("TB_pct")):
            out["proj_tb_pct"] = round(_nz(r.get("TB_pct")), 
                                           # ----- Anytime TD λ (per-game; shrunk + reweighted blend) -----
    # Determine if we have a real baseline from scoring.csv
    gp_for_shrink = _nz(r.get("GP"))
    baseline_raw = _nz(r.get("baseline_nonpass_td_pg"))

    have_baseline = (gp_for_shrink > 0) and (baseline_raw > 0 or baseline_raw == 0) and (not pd.isna(r.get("baseline_nonpass_td_pg")))

    # League avg non-pass TDs per game (computed earlier), use only to shrink real baselines
    league_nonpass_td_pg = _nz(league_td_pg) if _nz(league_td_pg) > 0 else 0.25

    if have_baseline:
        # Shrink real baseline toward league average with k=4 prior games
        baseline_shrunk = ((gp_for_shrink * baseline_raw) + (4.0 * league_nonpass_td_pg)) / (gp_for_shrink + 4.0)
    else:
        # No scoring row for this player → don't inject league average; use 0
        baseline_shrunk = 0.0

    # Per-component projected TDs
    comp_rush = _nz(out.get("proj_rush_td"))
    comp_rec  = _nz(out.get("proj_rec_td"))
    comp_kr   = _nz(out.get("proj_kr_td"))
    comp_pr   = _nz(out.get("proj_pr_td"))
    comp_def  = 0.15 * _nz(out.get("proj_def_int"))
    comp_sum = comp_rush + comp_rec + comp_kr + comp_pr + comp_def

    # Drive tempo modifier
    pace_mult = 0.97 + 0.03 * _nz(wb.get("drives_mult"))

    # Weighted blend: 35% baseline, 65% components, then clamp λ to [0, 2]
    lam_raw = (0.35 * baseline_shrunk * pace_mult) + (0.65 * comp_sum)

    # If there's neither a baseline nor any component signal, force λ=0
    if baseline_shrunk == 0.0 and comp_sum == 0.0:
        lam = 0.0
    else:
        lam = min(max(lam_raw, 0.0), 2.0)

    out["expected_anytime_td"] = round(lam, 4)
    out["anytime_td_prob"] = round(1 - math.exp(-lam), 4) if lam > 0 else 0.0

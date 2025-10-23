#!/usr/bin/env python3
"""
Unified Player Projections + Anytime TD Scorer

Reads (only the columns explicitly listed):
  data/raw/stats/passing.csv    -> PLAYER, TEAM, PYDS, PTD, INT
  data/raw/stats/rushing.csv    -> PLAYER, TEAM, RATT, RYDS, RTD
  data/raw/stats/receiving.csv  -> PLAYER, TEAM, TGT, REC, RECYDS, RECTD, RECAVG, YAC
  data/raw/stats/returning.csv  -> PLAYER, TEAM, K_RET, K_RET_YDS, K_RET_AVG, K_RET_TD, P_RET, P_RET_YDS, P_RET_AVG, P_RET_TD
  data/raw/stats/kicking.csv    -> PLAYER, TEAM, FGM, FGA, FG_PCT, XPM, XPA
  data/raw/stats/defensive.csv  -> PLAYER, TEAM, TCKL, SCK, DEF_INT
  data/raw/stats/scoring.csv    -> PLAYER, TEAM, GP, TD, PASS_TD, RUSH_TD, REC_TD, K_RET_TD, P_RET_TD, INT_TD

Context:
  data/processed/games.csv, data/team_stadiums.csv,
  data/processed/nfl_unified_with_metrics.csv (defense_PYDS/G, defense_RYDS/G)

Outputs:
  data/processed/player_projections.csv with:
    - Weather/dome-aware, defense-adjusted projections across roles
    - Anytime TD expected value and probability:
        expected_anytime_td, anytime_td_prob
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
P_SCORE  = BASE_RAW / "scoring.csv"

P_GAMES  = Path("data/processed/games.csv")
P_TEAMS  = Path("data/team_stadiums.csv")
P_MET    = Path("data/processed/nfl_unified_with_metrics.csv")
P_OUT    = Path("data/processed/player_projections.csv")

# ---------- Import weather_bundle (with fallback) ----------
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

# ---------- Utilities ----------
def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None

def _norm(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename found variants to canonical names; ignore others."""
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, variants in mapping.items():
        for v in variants:
            if v.lower() in lower_map:
                rename[lower_map[v.lower()]] = canon
                break
    return df.rename(columns=rename)

def _avg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None: 
        return pd.DataFrame(columns=["PLAYER","TEAM"] + cols)
    return df.groupby(["PLAYER","TEAM"], as_index=False)[cols].mean()

def home_mult(is_home: int) -> float:
    return 1.02 if int(is_home or 0) == 1 else 0.98

# ---------- Load context ----------
games   = pd.read_csv(P_GAMES)
teams   = pd.read_csv(P_TEAMS)
metrics = pd.read_csv(P_MET)

# Matchups: team -> opponent + weather (from HOME stadium) + dome flag
home_side = games[["home_team","away_team","temp_f","wind_mph","precip_in"]].copy()
home_side["team"] = home_side["home_team"]; home_side["opponent"] = home_side["away_team"]; home_side["is_home"] = 1
away_side = games[["home_team","away_team","temp_f","wind_mph","precip_in"]].copy()
away_side["team"] = away_side["away_team"]; away_side["opponent"] = away_side["home_team"]; away_side["is_home"] = 0
matchups = pd.concat([home_side, away_side], ignore_index=True)[
    ["team","opponent","is_home","temp_f","wind_mph","precip_in"]
]

# Dome flag comes from the HOME team's stadium and applies to both teams in the game
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
metrics_ctx = metrics[["TEAM","defense_PYDS/G","defense_RYDS/G"]].rename(columns={"TEAM":"opponent"})
matchups = matchups.merge(metrics_ctx, on="opponent", how="left")

league_avg_pass_def = metrics["defense_PYDS/G"].mean()
league_avg_rush_def = metrics["defense_RYDS/G"].mean()

def def_adj_pass(opp_def):
    if pd.isna(opp_def) or pd.isna(league_avg_pass_def) or league_avg_pass_def == 0: 
        return 1.0
    return 1 - ((opp_def - league_avg_pass_def) / league_avg_pass_def) * 0.6

def def_adj_rush(opp_def):
    if pd.isna(opp_def) or pd.isna(league_avg_rush_def) or league_avg_rush_def == 0: 
        return 1.0
    return 1 - ((opp_def - league_avg_rush_def) / league_avg_rush_def) * 0.5

# ---------- Load & normalize each category ----------
pass_df = _read_csv(P_PASS)
if pass_df is not None:
    pass_df = _norm(pass_df, {
        "PLAYER": ["PLAYER","Player","Name"],
        "TEAM":   ["TEAM","Team"],
        "PYDS":   ["PYDS"],   # from file
        "PTD":    ["PTD"],    # from file
        "INT":    ["INT"],    # from file
    })

rush_df = _read_csv(P_RUSH)
if rush_df is not None:
    rush_df = _norm(rush_df, {
        "PLAYER": ["PLAYER","Player","Name"],
        "TEAM":   ["TEAM","Team"],
        "RATT":   ["RATT"],   # from file
        "RYDS":   ["RYDS"],   # from file
        "RTD":    ["RTD"],    # from file
    })

recv_df = _read_csv(P_RECV)
if recv_df is not None:
    recv_df = _norm(recv_df, {
        "PLAYER":  ["PLAYER","Player","Name"],
        "TEAM":    ["TEAM","Team"],
        "TGT":     ["TGT"],
        "REC":     ["REC"],
        "RECYDS":  ["RECYDS"],
        "RECTD":   ["RECTD"],
        "RECAVG":  ["RECAVG"],
        "YAC":     ["YAC"],
    })

ret_df = _read_csv(P_RET)
if ret_df is not None:
    ret_df = _norm(ret_df, {
        "PLAYER":      ["PLAYER","Player","Name"],
        "TEAM":        ["TEAM","Team"],
        "K_RET":       ["K_RET"],
        "K_RET_YDS":   ["K_RET_YDS"],
        "K_RET_AVG":   ["K_RET_AVG"],
        "K_RET_TD":    ["K_RET_TD"],
        "P_RET":       ["P_RET"],
        "P_RET_YDS":   ["P_RET_YDS"],
        "P_RET_AVG":   ["P_RET_AVG"],
        "P_RET_TD":    ["P_RET_TD"],
    })

kick_df = _read_csv(P_KICK)
if kick_df is not None:
    kick_df = _norm(kick_df, {
        "PLAYER": ["PLAYER","Player","Name"],
        "TEAM":   ["TEAM","Team"],
        "FGM":    ["FGM"],
        "FGA":    ["FGA"],
        "FG_PCT": ["FG_PCT"],
        "XPM":    ["XPM"],
        "XPA":    ["XPA"],
    })

def_df = _read_csv(P_DEF)
if def_df is not None:
    def_df = _norm(def_df, {
        "PLAYER":  ["PLAYER","Player","Name"],
        "TEAM":    ["TEAM","Team"],
        "TCKL":    ["TCKL"],
        "SCK":     ["SCK"],
        "DEF_INT": ["DEF_INT"],
    })

score_df = _read_csv(P_SCORE)
if score_df is not None:
    score_df = _norm(score_df, {
        "PLAYER":   ["PLAYER","Player","Name"],
        "TEAM":     ["TEAM","Team"],
        "GP":       ["GP"],
        "TD":       ["TD"],
        "PASS_TD":  ["PASS_TD"],
        "RUSH_TD":  ["RUSH_TD"],
        "REC_TD":   ["REC_TD"],
        "K_RET_TD": ["K_RET_TD"],
        "P_RET_TD": ["P_RET_TD"],
        "INT_TD":   ["INT_TD"],
    })
    # per-game baseline for non-passing TDs
    for c in ["RUSH_TD","REC_TD","K_RET_TD","P_RET_TD","INT_TD"]:
        if c not in score_df.columns: score_df[c] = 0.0
    score_df["GP_safe"] = score_df["GP"].replace({0: 1})
    score_df["baseline_nonpass_td_pg"] = (
        score_df["RUSH_TD"].fillna(0)
      + score_df["REC_TD"].fillna(0)
      + score_df["K_RET_TD"].fillna(0)
      + score_df["P_RET_TD"].fillna(0)
      + score_df["INT_TD"].fillna(0)
    ) / score_df["GP_safe"]
    score_avg = score_df[["PLAYER","TEAM","baseline_nonpass_td_pg"]].copy()
else:
    score_avg = pd.DataFrame(columns=["PLAYER","TEAM","baseline_nonpass_td_pg"])

# ---------- Per-player averages by category ----------
pass_avg = _avg(pass_df, ["PYDS","PTD","INT"])
rush_avg = _avg(rush_df, ["RATT","RYDS","RTD"])
recv_avg = _avg(recv_df, ["TGT","REC","RECYDS","RECTD","RECAVG","YAC"])
ret_avg  = _avg(ret_df,  ["K_RET","K_RET_YDS","K_RET_AVG","K_RET_TD","P_RET","P_RET_YDS","P_RET_AVG","P_RET_TD"])
kick_avg = _avg(kick_df, ["FGM","FGA","FG_PCT","XPM","XPA"])
def_avg  = _avg(def_df,  ["TCKL","SCK","DEF_INT"])

# ---------- Build player universe ----------
players = pass_avg.merge(rush_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(recv_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(ret_avg,  on=["PLAYER","TEAM"], how="outer")
players = players.merge(kick_avg, on=["PLAYER","TEAM"], how="outer")
players = players.merge(def_avg,  on=["PLAYER","TEAM"], how="outer")
players = players.merge(score_avg, on=["PLAYER","TEAM"], how="left")

# Attach matchup row
players = players.merge(matchups, left_on="TEAM", right_on="team", how="left")
players = players[players["opponent"].notna()].copy()

# ---------- Project per player ----------
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
        "expected_anytime_td": None, "anytime_td_prob": None,
    }

    # ----- Passing -----
    if pd.notna(r.get("PYDS")):
        pass_yds = float(r["PYDS"]); ptd = float(r.get("PTD") or 0.0); intr = float(r.get("INT") or 0.0)
        out["proj_pass_yds"] = round(pass_yds * wb["pass_mult"] * hmult * p_mult, 1)
        out["proj_pass_td"]  = round(ptd      * wb["pass_mult"] * hmult * p_mult, 2)
        out["proj_int"]      = round(intr     * (1.00 + max(0.0, wb["to_delta"])), 2)

    # ----- Rushing -----
    if pd.notna(r.get("RYDS")):
        r_yds = float(r["RYDS"]); rtd = float(r.get("RTD") or 0.0)
        out["proj_rush_yds"] = round(r_yds * wb["rush_mult"] * hmult * r_mult, 1)
        out["proj_rush_td"]  = round(rtd   * wb["rush_mult"] * hmult * r_mult, 2)

    # ----- Receiving -----
    if pd.notna(r.get("RECYDS")):
        recyds = float(r["RECYDS"]); rectd = float(r.get("RECTD") or 0.0)
        rec = float(r.get("REC") or 0.0); tgt = float(r.get("TGT") or 0.0); yac = float(r.get("YAC") or 0.0)
        adj = wb["pass_mult"] * hmult * p_mult
        out["proj_rec_yards"] = round(recyds * adj, 1)
        out["proj_rec_td"]    = round(rectd  * adj, 2)
        out["proj_targets"]   = round(tgt * (0.97 + 0.03*wb["drives_mult"]), 1)
        out["proj_rec"]       = round(rec * (0.97 + 0.03*wb["drives_mult"]), 1)
        out["proj_yac"]       = round(yac * (0.98 if wb["pass_mult"] < 1.0 else 1.0), 1)

    # ----- Returns (KR/PR) -----
    # Yards per return modestly impacted by wind; TD rates are rare -> keep from history
    if (pd.notna(r.get("K_RET")) and pd.notna(r.get("K_RET_AVG"))) or (pd.notna(r.get("P_RET")) and pd.notna(r.get("P_RET_AVG"))):
        kr = float(r.get("K_RET") or 0.0); kr_avg = float(r.get("K_RET_AVG") or 0.0)
        pr = float(r.get("P_RET") or 0.0); pr_avg = float(r.get("P_RET_AVG") or 0.0)
        pace = wb["drives_mult"]
        wind_eff = 0.98 if wb["fg_pct_delta"] < 0 else 1.0
        out["proj_kr_yds"] = round(kr * kr_avg * pace * wind_eff, 1) if kr > 0 else None
        out["proj_kr_td"]  = float(r.get("K_RET_TD") or 0.0)
        out["proj_pr_yds"] = round(pr * pr_avg * pace * wind_eff, 1) if pr > 0 else None
        out["proj_pr_td"]  = float(r.get("P_RET_TD") or 0.0)

    # ----- Kickers -----
    if pd.notna(r.get("FGA")):
        fga = float(r["FGA"]); fgm = float(r.get("FGM") or 0.0)
        xpa = float(r.get("XPA") or 0.0); xpm = float(r.get("XPM") or 0.0)
        fg_pct = float(r.get("FG_PCT") or (fgm / fga * 100 if fga > 0 else 0.0))
        adj_fg_pct = max(0.0, min(100.0, fg_pct + (wb["fg_pct_delta"] * 100)))
        out["proj_fga"] = round(fga * (0.97 + 0.03*wb["drives_mult"]), 1)
        out["proj_fgm"] = round(out["proj_fga"] * (adj_fg_pct / 100.0), 1)
        out["proj_xpa"] = round(xpa * (0.98 + 0.02*wb["drives_mult"]), 1)
        out["proj_xpm"] = round(min(out["proj_xpa"], xpm * (1 + wb["fg_pct_delta"])), 1)
        out["proj_fg_pct"] = round(adj_fg_pct, 1)

    # ----- Defensive -----
    if pd.notna(r.get("TCKL")):
        tckl = float(r["TCKL"]); sck = float(r.get("SCK") or 0.0); dint = float(r.get("DEF_INT") or 0.0)
        pace = wb["drives_mult"]
        out["proj_tackles"] = round(tckl * (0.98 + 0.02*pace), 1)
        out["proj_sacks"]   = round(sck   * (1.00 + 0.05*(1 - wb["pass_mult"])), 2)
        out["proj_def_int"] = round(dint  * (1.00 + max(0.0, wb["to_delta"])), 2)

    # ----- Anytime TD (non-passing) -----
    base_nonpass_pg = float(r.get("baseline_nonpass_td_pg") or 0.0)
    # Component-based expected TD from projections
    comp_rush = float(out["proj_rush_td"] or 0.0)
    comp_rec  = float(out["proj_rec_td"]  or 0.0)
    comp_kr   = float(out["proj_kr_td"]   or 0.0)
    comp_pr   = float(out["proj_pr_td"]   or 0.0)
    # Defensive INT TD proxy: small fraction of projected INTs
    comp_def  = 0.15 * float(out["proj_def_int"] or 0.0)

    comp_sum = comp_rush + comp_rec + comp_kr + comp_pr + comp_def
    # Blend baseline (pace-adjusted) and components
    pace_mult = (0.97 + 0.03 * wb["drives_mult"])
    lam = max(0.0, 0.5 * (base_nonpass_pg * pace_mult) + 0.5 * comp_sum)

    out["expected_anytime_td"] = round(lam, 3)
    out["anytime_td_prob"] = round(1 - math.exp(-lam), 3) if lam > 0 else 0.0

    rows.append(out)

out_df = pd.DataFrame(rows)
P_OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(P_OUT, index=False)
print(f"Wrote {P_OUT} with {len(out_df)} player rows.")

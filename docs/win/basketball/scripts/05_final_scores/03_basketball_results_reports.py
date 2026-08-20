#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/03_basketball_results_reports.py
#
# Reads work_{league}.csv from script 02 and produces summary/detail CSVs.
# All "by X" files share a canonical schema (see SCHEMA below). All sides files
# follow Option A naming: "_home_away_summary" for ML/spread, "_over_under_summary"
# for total. Crosses are emitted as a single long-format file per market.
#
# Inputs:
#   docs/win/basketball/05_final_scores/work_nba.csv
#   docs/win/basketball/05_final_scores/work_ncaam.csv
#   docs/win/basketball/05_final_scores/work_wnba.csv
#
# Outputs (per league: nba, ncaam, wnba):
#   docs/win/basketball/05_final_scores/{league}_summary_overall.csv
#   docs/win/basketball/05_final_scores/{league}_summary_grand_total.csv
#   docs/win/basketball/05_final_scores/reports/{league}/{moneyline,spread,total,overview}/*.csv
#
# Log:
#   docs/win/basketball/errors/05_final_scores/03_basketball_results_reports.txt

from datetime import datetime, UTC
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

BASE       = Path("docs/win/basketball/05_final_scores")
REPORT_DIR = BASE / "reports"
ERROR_DIR  = Path("docs/win/basketball/errors/05_final_scores")
LOG_FILE   = ERROR_DIR / "03_basketball_results_reports.txt"

# Where work files live
WORK_FILES = {lg: BASE / f"work_{lg}.csv" for lg in LEAGUES}

ERROR_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOGGING
# =========================

RUN_STARTED = datetime.now(UTC)
WARNING_COUNT = 0
ERROR_COUNT = 0
INPUT_FILE_COUNT = 0
INPUT_ROW_COUNT = 0
OUTPUT_FILE_COUNT = 0
OUTPUT_ROW_COUNT = 0

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== 03_basketball_results_reports ===\n")
    f.write(f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def log(level: str, message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level} | {message}\n")


def warn(message: str) -> None:
    global WARNING_COUNT
    WARNING_COUNT += 1
    log("WARNING", message)


def error(message: str) -> None:
    global ERROR_COUNT
    ERROR_COUNT += 1
    log("ERROR", message)


def log_input(path: Path, rows: int, exists: bool = True) -> None:
    global INPUT_FILE_COUNT, INPUT_ROW_COUNT
    INPUT_FILE_COUNT += 1
    INPUT_ROW_COUNT += rows
    log("INFO", f"INPUT | file={path} | exists={int(exists)} | rows={rows}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    global OUTPUT_FILE_COUNT, OUTPUT_ROW_COUNT
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    rows = len(df)
    OUTPUT_FILE_COUNT += 1
    OUTPUT_ROW_COUNT += rows
    log("INFO", f"OUTPUT | file={path} | rows={rows}")


def finish(status: str) -> None:
    ended = datetime.now(UTC)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"INPUT_SUMMARY | files={INPUT_FILE_COUNT} | rows={INPUT_ROW_COUNT}\n")
        f.write(f"OUTPUT_SUMMARY | files={OUTPUT_FILE_COUNT} | rows={OUTPUT_ROW_COUNT}\n")
        f.write(f"WARNING_COUNT: {WARNING_COUNT}\n")
        f.write(f"ERROR_COUNT: {ERROR_COUNT}\n")
        f.write(f"END_TIMESTAMP_UTC: {ended.isoformat()}\n")
        f.write(f"STATUS: {status}\n")


# =========================
# CANONICAL SCHEMA
# =========================

CANON_COLS_NO_SIDE = [
    "league", "market_type", "bucket_dimension", "bucket",
    "bets", "wins", "losses", "pushes", "total",
    "win_pct",
    "units_flat", "roi_flat",
    "units_kelly", "roi_kelly",
    "avg_ev", "avg_edge_vs_market_pp", "avg_kelly_pct",
    "avg_model_prob", "avg_odds_american",
]

CANON_COLS_WITH_SIDE = [
    "league", "market_type", "side_group", "bucket_dimension", "bucket",
    "bets", "wins", "losses", "pushes", "total",
    "win_pct",
    "units_flat", "roi_flat",
    "units_kelly", "roi_kelly",
    "avg_ev", "avg_edge_vs_market_pp", "avg_kelly_pct",
    "avg_model_prob", "avg_odds_american",
]


# =========================
# HELPERS
# =========================

def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def aggregate_block(df: pd.DataFrame, league: str, market_type: str | None,
                    bucket_dimension: str, bucket_col: str,
                    side_group_col: str | None = None) -> pd.DataFrame:
    """Build one canonical aggregation DataFrame."""
    if df.empty:
        cols = CANON_COLS_WITH_SIDE if side_group_col else CANON_COLS_NO_SIDE
        return pd.DataFrame(columns=cols)

    work = df.copy()

    for c in ("profit_unit", "profit_kelly", "bet_stake_pct",
              "bet_ev", "bet_edge_vs_market", "bet_kelly",
              "bet_model_prob", "bet_odds_american"):
        if c in work.columns:
            work[c] = to_num(work[c])

    res = work["bet_result"].astype(str).str.strip().str.lower() if "bet_result" in work.columns else pd.Series([""] * len(work))
    work["_is_win"]  = (res == "win").astype(int)
    work["_is_loss"] = (res == "loss").astype(int)
    work["_is_push"] = (res == "push").astype(int)

    group_cols = [bucket_col]
    if side_group_col:
        group_cols = [side_group_col] + group_cols

    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        wins   = int(sub["_is_win"].sum())
        losses = int(sub["_is_loss"].sum())
        pushes = int(sub["_is_push"].sum())
        bets   = wins + losses + pushes
        total  = bets

        units_flat  = float(sub["profit_unit"].sum(skipna=True))   if "profit_unit"  in sub.columns else 0.0
        units_kelly = float(sub["profit_kelly"].sum(skipna=True))  if "profit_kelly" in sub.columns else 0.0
        stake_total = float(sub["bet_stake_pct"].sum(skipna=True)) if "bet_stake_pct" in sub.columns else 0.0

        roi_flat  = (units_flat  / bets) if bets > 0 else np.nan
        roi_kelly = (units_kelly / stake_total) if stake_total > 0 else np.nan
        win_pct = (wins / (wins + losses)) if (wins + losses) > 0 else np.nan

        avg_ev      = float(sub["bet_ev"].mean(skipna=True))             if "bet_ev"             in sub.columns else np.nan
        avg_edgepp  = float(sub["bet_edge_vs_market"].mean(skipna=True)) if "bet_edge_vs_market" in sub.columns else np.nan
        avg_kpct    = float(sub["bet_kelly"].mean(skipna=True))          if "bet_kelly"          in sub.columns else np.nan
        avg_mp      = float(sub["bet_model_prob"].mean(skipna=True))     if "bet_model_prob"     in sub.columns else np.nan
        avg_odds    = float(sub["bet_odds_american"].mean(skipna=True))  if "bet_odds_american"  in sub.columns else np.nan

        if market_type is not None:
            mt = market_type
        else:
            mt_vals = sub["market_type"].astype(str).str.lower().unique() if "market_type" in sub.columns else []
            mt = mt_vals[0] if len(mt_vals) == 1 else "mixed"

        row = {
            "league":           league,
            "market_type":      mt,
            "bucket_dimension": bucket_dimension,
            "bucket":           keys[-1] if len(keys) == 1 else keys[1],
            "bets":             bets,
            "wins":             wins,
            "losses":           losses,
            "pushes":           pushes,
            "total":            total,
            "win_pct":          round(win_pct, 4) if not pd.isna(win_pct) else np.nan,
            "units_flat":       round(units_flat, 4),
            "roi_flat":         round(roi_flat, 4) if not pd.isna(roi_flat) else np.nan,
            "units_kelly":      round(units_kelly, 6),
            "roi_kelly":        round(roi_kelly, 4) if not pd.isna(roi_kelly) else np.nan,
            "avg_ev":           round(avg_ev, 4) if not pd.isna(avg_ev) else np.nan,
            "avg_edge_vs_market_pp": round(avg_edgepp, 4) if not pd.isna(avg_edgepp) else np.nan,
            "avg_kelly_pct":    round(avg_kpct, 4) if not pd.isna(avg_kpct) else np.nan,
            "avg_model_prob":   round(avg_mp, 4) if not pd.isna(avg_mp) else np.nan,
            "avg_odds_american":round(avg_odds, 1) if not pd.isna(avg_odds) else np.nan,
        }
        if side_group_col:
            row["side_group"] = keys[0]

        rows.append(row)

    cols = CANON_COLS_WITH_SIDE if side_group_col else CANON_COLS_NO_SIDE
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out[cols].sort_values(by=[c for c in cols if c in ("side_group", "bucket")]).reset_index(drop=True)
    return out


# =========================
# PER-MARKET REPORTS
# =========================

COMMON_BUCKETS = [
    ("ev",             "ev_bucket"),
    ("kelly",          "kelly_bucket"),
    ("odds",           "odds_bucket"),
    ("win_prob",       "model_prob_bucket"),
    ("edge_vs_market", "edge_vs_market_bucket"),
    ("dow",            "dow_bucket"),
    ("month",          "month_bucket"),
]


def side_suffix(market_type: str) -> str:
    return "over_under" if market_type == "total" else "home_away"


def write_market_reports(work_df: pd.DataFrame, league: str, market_type: str,
                         out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = work_df[work_df["market_type"].astype(str).str.lower() == market_type].copy()
    if sub.empty:
        log("INFO", f"[{league} / {market_type}] no rows; skipping per-market reports")
        return

    side_col = "side_group" if "side_group" in sub.columns else None
    sfx = side_suffix(market_type)

    for label, bucket_col in COMMON_BUCKETS:
        if bucket_col not in sub.columns:
            warn(f"[{league} / {market_type}] missing bucket column {bucket_col}; skipping {label}")
            continue

        agg = aggregate_block(sub, league=league, market_type=market_type,
                              bucket_dimension=label, bucket_col=bucket_col)
        write_csv(agg, out_dir / f"{league}_{market_type}_by_{label}.csv")

        if side_col:
            agg_s = aggregate_block(sub, league=league, market_type=market_type,
                                    bucket_dimension=label, bucket_col=bucket_col,
                                    side_group_col=side_col)
            write_csv(agg_s, out_dir / f"{league}_{market_type}_by_{label}_{sfx}_summary.csv")

    if market_type in ("spread", "total") and side_col:
        agg = aggregate_block(sub, league=league, market_type=market_type,
                              bucket_dimension="side", bucket_col=side_col)
        write_csv(agg, out_dir / f"{league}_{market_type}_by_side.csv")

        agg_s = agg.copy()
        if not agg_s.empty:
            agg_s.insert(2, "side_group", agg_s["bucket"])
            agg_s = agg_s[CANON_COLS_WITH_SIDE]
        else:
            agg_s = pd.DataFrame(columns=CANON_COLS_WITH_SIDE)
        write_csv(agg_s, out_dir / f"{league}_{market_type}_by_side_{sfx}_summary.csv")

    if market_type == "total" and "total_bucket" in sub.columns:
        agg = aggregate_block(sub, league=league, market_type=market_type,
                              bucket_dimension="total_range", bucket_col="total_bucket")
        write_csv(agg, out_dir / f"{league}_{market_type}_by_total_range.csv")
        if side_col:
            agg_s = aggregate_block(sub, league=league, market_type=market_type,
                                    bucket_dimension="total_range", bucket_col="total_bucket",
                                    side_group_col=side_col)
            write_csv(agg_s, out_dir / f"{league}_{market_type}_by_total_range_{sfx}_summary.csv")


# =========================
# CROSSES
# =========================

CROSS_DIMS = [
    ("ev",             "ev_bucket"),
    ("kelly",          "kelly_bucket"),
    ("odds",           "odds_bucket"),
    ("win_prob",       "model_prob_bucket"),
    ("edge_vs_market", "edge_vs_market_bucket"),
    ("dow",            "dow_bucket"),
    ("month",          "month_bucket"),
    ("side",           "side_group"),
]

CROSS_COLS = [
    "league", "market_type",
    "dimension_1", "bucket_1",
    "dimension_2", "bucket_2",
    "bets", "wins", "losses", "pushes", "total",
    "win_pct",
    "units_flat", "roi_flat",
    "units_kelly", "roi_kelly",
    "avg_ev", "avg_edge_vs_market_pp", "avg_kelly_pct",
    "avg_model_prob", "avg_odds_american",
]


def aggregate_cross(df: pd.DataFrame, league: str, market_type: str,
                    dim1_label: str, dim1_col: str,
                    dim2_label: str, dim2_col: str) -> pd.DataFrame:
    if df.empty or dim1_col not in df.columns or dim2_col not in df.columns:
        return pd.DataFrame(columns=CROSS_COLS)

    work = df.copy()
    for c in ("profit_unit", "profit_kelly", "bet_stake_pct",
              "bet_ev", "bet_edge_vs_market", "bet_kelly",
              "bet_model_prob", "bet_odds_american"):
        if c in work.columns:
            work[c] = to_num(work[c])

    res = work["bet_result"].astype(str).str.strip().str.lower() if "bet_result" in work.columns else pd.Series([""] * len(work))
    work["_is_win"]  = (res == "win").astype(int)
    work["_is_loss"] = (res == "loss").astype(int)
    work["_is_push"] = (res == "push").astype(int)

    rows = []
    for (b1, b2), sub in work.groupby([dim1_col, dim2_col], dropna=False, observed=True):
        wins   = int(sub["_is_win"].sum())
        losses = int(sub["_is_loss"].sum())
        pushes = int(sub["_is_push"].sum())
        bets   = wins + losses + pushes

        units_flat  = float(sub["profit_unit"].sum(skipna=True))   if "profit_unit"  in sub.columns else 0.0
        units_kelly = float(sub["profit_kelly"].sum(skipna=True))  if "profit_kelly" in sub.columns else 0.0
        stake_total = float(sub["bet_stake_pct"].sum(skipna=True)) if "bet_stake_pct" in sub.columns else 0.0

        roi_flat  = (units_flat  / bets) if bets > 0 else np.nan
        roi_kelly = (units_kelly / stake_total) if stake_total > 0 else np.nan
        win_pct   = (wins / (wins + losses)) if (wins + losses) > 0 else np.nan

        rows.append({
            "league":           league,
            "market_type":      market_type,
            "dimension_1":      dim1_label,
            "bucket_1":         b1,
            "dimension_2":      dim2_label,
            "bucket_2":         b2,
            "bets":             bets,
            "wins":             wins,
            "losses":           losses,
            "pushes":           pushes,
            "total":            bets,
            "win_pct":          round(win_pct, 4) if not pd.isna(win_pct) else np.nan,
            "units_flat":       round(units_flat, 4),
            "roi_flat":         round(roi_flat, 4) if not pd.isna(roi_flat) else np.nan,
            "units_kelly":      round(units_kelly, 6),
            "roi_kelly":        round(roi_kelly, 4) if not pd.isna(roi_kelly) else np.nan,
            "avg_ev":           round(float(sub["bet_ev"].mean(skipna=True)), 4) if "bet_ev" in sub.columns and not sub["bet_ev"].dropna().empty else np.nan,
            "avg_edge_vs_market_pp": round(float(sub["bet_edge_vs_market"].mean(skipna=True)), 4) if "bet_edge_vs_market" in sub.columns and not sub["bet_edge_vs_market"].dropna().empty else np.nan,
            "avg_kelly_pct":    round(float(sub["bet_kelly"].mean(skipna=True)), 4) if "bet_kelly" in sub.columns and not sub["bet_kelly"].dropna().empty else np.nan,
            "avg_model_prob":   round(float(sub["bet_model_prob"].mean(skipna=True)), 4) if "bet_model_prob" in sub.columns and not sub["bet_model_prob"].dropna().empty else np.nan,
            "avg_odds_american":round(float(sub["bet_odds_american"].mean(skipna=True)), 1) if "bet_odds_american" in sub.columns and not sub["bet_odds_american"].dropna().empty else np.nan,
        })

    return pd.DataFrame(rows, columns=CROSS_COLS)


def write_market_crosses(work_df: pd.DataFrame, league: str, market_type: str,
                         out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = work_df[work_df["market_type"].astype(str).str.lower() == market_type].copy()
    if sub.empty:
        return

    pieces = []
    n = len(CROSS_DIMS)
    for i in range(n):
        for j in range(i + 1, n):
            d1_label, d1_col = CROSS_DIMS[i]
            d2_label, d2_col = CROSS_DIMS[j]
            if d1_col not in sub.columns or d2_col not in sub.columns:
                continue
            pieces.append(aggregate_cross(sub, league, market_type,
                                          d1_label, d1_col, d2_label, d2_col))

    if pieces:
        all_crosses = pd.concat(pieces, ignore_index=True)
        write_csv(all_crosses, out_dir / f"{league}_{market_type}_crosses.csv")


# =========================
# OVERVIEW
# =========================

def write_overview(work_df: pd.DataFrame, league: str, overview_dir: Path) -> None:
    overview_dir.mkdir(parents=True, exist_ok=True)

    if work_df.empty:
        log("INFO", f"[{league}] no rows; skipping overview")
        return

    by_mkt = []
    for mt, sub in work_df.groupby("market_type", dropna=False, observed=True):
        agg = aggregate_block(sub, league=league, market_type=str(mt).lower(),
                              bucket_dimension="market_type", bucket_col="market_type")
        by_mkt.append(agg)
    if by_mkt:
        out = pd.concat(by_mkt, ignore_index=True)
        write_csv(out, overview_dir / f"{league}_summary_by_market.csv")

    if "side_group" in work_df.columns:
        agg = aggregate_block(work_df, league=league, market_type=None,
                              bucket_dimension="side_group", bucket_col="side_group")
        write_csv(agg, overview_dir / f"{league}_summary_by_side_group.csv")

    if "game_date" in work_df.columns:
        agg = aggregate_block(work_df, league=league, market_type=None,
                              bucket_dimension="game_date", bucket_col="game_date")
        write_csv(agg, overview_dir / f"{league}_summary_by_date.csv")

    log_cols = [
        "game_date", "league", "market_type", "side_group",
        "home_team", "away_team",
        "bet_side", "bet_line", "bet_odds_american",
        "bet_ev", "bet_edge_vs_market", "bet_kelly", "bet_model_prob",
        "bet_stake_pct",
        "ev_bucket", "edge_vs_market_bucket", "kelly_bucket", "odds_bucket",
        "model_prob_bucket", "spread_bucket", "total_bucket",
        "dow_bucket", "month_bucket",
        "bet_result", "profit_unit", "profit_kelly",
    ]
    existing = [c for c in log_cols if c in work_df.columns]
    write_csv(work_df[existing], overview_dir / f"{league}_bet_log.csv")

    overall = build_summary_overall(work_df, league)
    write_csv(overall, overview_dir / f"{league}_summary_overall.csv")


# =========================
# TOP-LEVEL SUMMARIES
# =========================

def build_summary_overall(work_df: pd.DataFrame, league: str) -> pd.DataFrame:
    rows = []
    if work_df.empty:
        return pd.DataFrame(columns=["league","market_type","Win","Loss","Push","Total","Win_Pct"])

    for mt in ["moneyline", "spread", "total"]:
        sub = work_df[work_df["market_type"].astype(str).str.lower() == mt]
        res = sub["bet_result"].astype(str).str.strip().str.lower() if "bet_result" in sub.columns else pd.Series([""] * len(sub))
        wins   = int((res == "win").sum())
        losses = int((res == "loss").sum())
        pushes = int((res == "push").sum())
        total  = wins + losses + pushes
        wp     = round(wins / (wins + losses), 4) if (wins + losses) > 0 else np.nan
        rows.append({
            "league":      league.upper(),
            "market_type": mt,
            "Win":         wins,
            "Loss":        losses,
            "Push":        pushes,
            "Total":       total,
            "Win_Pct":     wp,
        })
    return pd.DataFrame(rows)


def build_summary_grand_total(work_df: pd.DataFrame, league: str) -> pd.DataFrame:
    if work_df.empty:
        return pd.DataFrame([{
            "league": league.upper(),
            "bets": 0, "wins": 0, "losses": 0, "pushes": 0, "total": 0,
            "win_pct": np.nan,
            "units_flat": 0.0, "roi_flat": np.nan,
            "units_kelly": 0.0, "roi_kelly": np.nan,
            "avg_ev": np.nan, "avg_edge_vs_market_pp": np.nan,
            "avg_kelly_pct": np.nan, "avg_model_prob": np.nan,
            "avg_odds_american": np.nan,
        }])

    res = work_df["bet_result"].astype(str).str.strip().str.lower()
    wins   = int((res == "win").sum())
    losses = int((res == "loss").sum())
    pushes = int((res == "push").sum())
    bets   = wins + losses + pushes

    units_flat  = float(to_num(work_df.get("profit_unit",  pd.Series(dtype=float))).sum(skipna=True))
    units_kelly = float(to_num(work_df.get("profit_kelly", pd.Series(dtype=float))).sum(skipna=True))
    stake_total = float(to_num(work_df.get("bet_stake_pct", pd.Series(dtype=float))).sum(skipna=True))

    roi_flat  = (units_flat / bets) if bets > 0 else np.nan
    roi_kelly = (units_kelly / stake_total) if stake_total > 0 else np.nan
    win_pct   = (wins / (wins + losses)) if (wins + losses) > 0 else np.nan

    return pd.DataFrame([{
        "league": league.upper(),
        "bets": bets, "wins": wins, "losses": losses, "pushes": pushes, "total": bets,
        "win_pct": round(win_pct, 4) if not pd.isna(win_pct) else np.nan,
        "units_flat":  round(units_flat, 4),
        "roi_flat":    round(roi_flat, 4) if not pd.isna(roi_flat) else np.nan,
        "units_kelly": round(units_kelly, 6),
        "roi_kelly":   round(roi_kelly, 4) if not pd.isna(roi_kelly) else np.nan,
        "avg_ev":      round(float(to_num(work_df.get("bet_ev", pd.Series(dtype=float))).mean(skipna=True)), 4) if "bet_ev" in work_df.columns else np.nan,
        "avg_edge_vs_market_pp": round(float(to_num(work_df.get("bet_edge_vs_market", pd.Series(dtype=float))).mean(skipna=True)), 4) if "bet_edge_vs_market" in work_df.columns else np.nan,
        "avg_kelly_pct":    round(float(to_num(work_df.get("bet_kelly", pd.Series(dtype=float))).mean(skipna=True)), 4) if "bet_kelly" in work_df.columns else np.nan,
        "avg_model_prob":   round(float(to_num(work_df.get("bet_model_prob", pd.Series(dtype=float))).mean(skipna=True)), 4) if "bet_model_prob" in work_df.columns else np.nan,
        "avg_odds_american":round(float(to_num(work_df.get("bet_odds_american", pd.Series(dtype=float))).mean(skipna=True)), 1) if "bet_odds_american" in work_df.columns else np.nan,
    }])


# =========================
# RUN
# =========================

def run_one(league: str) -> None:
    work_path = WORK_FILES[league]
    if not work_path.exists():
        log_input(work_path, 0, exists=False)
        warn(f"[{league}] missing work file: {work_path}")
        return

    work = pd.read_csv(work_path)
    log_input(work_path, len(work), exists=True)
    if work.empty:
        warn(f"[{league}] empty work file; skipping")
        return

    if "market_type" in work.columns:
        work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    if "side_group" in work.columns:
        work["side_group"] = work["side_group"].astype(str).str.strip().str.upper()

    write_csv(build_summary_overall(work, league), BASE / f"{league}_summary_overall.csv")
    write_csv(build_summary_grand_total(work, league), BASE / f"{league}_summary_grand_total.csv")

    for mt in ["moneyline", "spread", "total"]:
        out_dir = REPORT_DIR / league / mt
        write_market_reports(work, league, mt, out_dir)
        write_market_crosses(work, league, mt, out_dir)

    overview_dir = REPORT_DIR / league / "overview"
    write_overview(work, league, overview_dir)

    log("INFO", f"[{league}] reports written under {REPORT_DIR / league}")


def run() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for league in LEAGUES:
        run_one(league)
    log("INFO", "Basketball reports complete.")


def main() -> None:
    status = "FAILED"
    try:
        run()
        status = "SUCCESS"
    except Exception as exc:
        error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        trace = traceback.format_exc()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(trace)
            if not trace.endswith("\n"):
                f.write("\n")
        raise
    finally:
        finish(status)


if __name__ == "__main__":
    main()

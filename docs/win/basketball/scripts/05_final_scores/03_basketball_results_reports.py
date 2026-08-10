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

from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

BASE       = Path("docs/win/basketball/05_final_scores")
REPORT_DIR = BASE / "reports"

# Where work files live
WORK_FILES = {lg: BASE / f"work_{lg}.csv" for lg in LEAGUES}


# =========================
# CANONICAL SCHEMA
# =========================
#
# Every "by X" file uses these columns. Side-aware files insert side_group
# right after market_type. The 'bucket_dimension' column lets multiple files
# be unioned cleanly later.

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
    """
    Build one canonical aggregation DataFrame.

    df: already filtered to the appropriate market(s).
    market_type: literal value to populate the column. If None, derive from df.
    bucket_dimension: literal label (e.g. "ev", "kelly", "model_prob", "dow").
    bucket_col: column in df that holds the bucket label.
    side_group_col: if provided, also group by this column and include it in output.
    """
    if df.empty:
        cols = CANON_COLS_WITH_SIDE if side_group_col else CANON_COLS_NO_SIDE
        return pd.DataFrame(columns=cols)

    work = df.copy()

    # Numeric coercion (defensive — these may already be numeric)
    for c in ("profit_unit", "profit_kelly", "bet_stake_pct",
              "bet_ev", "bet_edge_vs_market", "bet_kelly",
              "bet_model_prob", "bet_odds_american"):
        if c in work.columns:
            work[c] = to_num(work[c])

    # Result flags
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

        # Resolve market_type for this row
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

# (bucket_dimension_label, bucket_col_in_work) tuples used for per-market buckets.
# Note: 'win_prob' is sourced from model_prob_bucket (your stated mapping).
COMMON_BUCKETS = [
    ("ev",             "ev_bucket"),
    ("kelly",          "kelly_bucket"),
    ("odds",           "odds_bucket"),
    ("win_prob",       "model_prob_bucket"),
    ("edge_vs_market", "edge_vs_market_bucket"),
    ("dow",            "dow_bucket"),
    ("month",          "month_bucket"),
]

# Side-naming convention (Option A)
def side_suffix(market_type: str) -> str:
    return "over_under" if market_type == "total" else "home_away"


def write_market_reports(work_df: pd.DataFrame, league: str, market_type: str,
                         out_dir: Path) -> None:
    """Per-market: write all _by_X (no side) and _by_X_{home_away|over_under}_summary files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = work_df[work_df["market_type"].astype(str).str.lower() == market_type].copy()
    if sub.empty:
        print(f"  [{league} / {market_type}] no rows; skipping per-market reports")
        return

    side_col = "side_group" if "side_group" in sub.columns else None
    sfx = side_suffix(market_type)

    for label, bucket_col in COMMON_BUCKETS:
        if bucket_col not in sub.columns:
            continue

        # No-side aggregate
        agg = aggregate_block(sub, league=league, market_type=market_type,
                              bucket_dimension=label, bucket_col=bucket_col)
        agg.to_csv(out_dir / f"{league}_{market_type}_by_{label}.csv", index=False)

        # Side-aware aggregate
        if side_col:
            agg_s = aggregate_block(sub, league=league, market_type=market_type,
                                    bucket_dimension=label, bucket_col=bucket_col,
                                    side_group_col=side_col)
            agg_s.to_csv(out_dir / f"{league}_{market_type}_by_{label}_{sfx}_summary.csv", index=False)

    # Spread/total: by_side files (the side_group itself IS the bucket)
    if market_type in ("spread", "total") and side_col:
        # _by_side: just bucket = side_group, no side column repeated
        agg = aggregate_block(sub, league=league, market_type=market_type,
                              bucket_dimension="side", bucket_col=side_col)
        agg.to_csv(out_dir / f"{league}_{market_type}_by_side.csv", index=False)
        # _by_side_{sfx}_summary: identical content but matches the naming pattern;
        # included so the family is consistent. side_group column = bucket value.
        agg_s = agg.copy()
        if not agg_s.empty:
            agg_s.insert(2, "side_group", agg_s["bucket"])
            agg_s = agg_s[CANON_COLS_WITH_SIDE]
        else:
            agg_s = pd.DataFrame(columns=CANON_COLS_WITH_SIDE)
        agg_s.to_csv(out_dir / f"{league}_{market_type}_by_side_{sfx}_summary.csv", index=False)

    # Total only: by_total_range (uses total_bucket which is the book total binning)
    if market_type == "total" and "total_bucket" in sub.columns:
        agg = aggregate_block(sub, league=league, market_type=market_type,
                              bucket_dimension="total_range", bucket_col="total_bucket")
        agg.to_csv(out_dir / f"{league}_{market_type}_by_total_range.csv", index=False)
        if side_col:
            agg_s = aggregate_block(sub, league=league, market_type=market_type,
                                    bucket_dimension="total_range", bucket_col="total_bucket",
                                    side_group_col=side_col)
            agg_s.to_csv(out_dir / f"{league}_{market_type}_by_total_range_{sfx}_summary.csv", index=False)


# =========================
# CROSSES (single long-format file per market)
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

    out = pd.DataFrame(rows, columns=CROSS_COLS)
    return out


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
        all_crosses.to_csv(out_dir / f"{league}_{market_type}_crosses.csv", index=False)


# =========================
# OVERVIEW
# =========================

def write_overview(work_df: pd.DataFrame, league: str, overview_dir: Path) -> None:
    overview_dir.mkdir(parents=True, exist_ok=True)

    if work_df.empty:
        print(f"  [{league}] no rows; skipping overview")
        return

    # by market
    by_mkt = []
    for mt, sub in work_df.groupby("market_type", dropna=False, observed=True):
        agg = aggregate_block(sub, league=league, market_type=str(mt).lower(),
                              bucket_dimension="market_type", bucket_col="market_type")
        by_mkt.append(agg)
    if by_mkt:
        out = pd.concat(by_mkt, ignore_index=True)
        out.to_csv(overview_dir / f"{league}_summary_by_market.csv", index=False)

    # by side_group
    if "side_group" in work_df.columns:
        agg = aggregate_block(work_df, league=league, market_type=None,
                              bucket_dimension="side_group", bucket_col="side_group")
        agg.to_csv(overview_dir / f"{league}_summary_by_side_group.csv", index=False)

    # by date
    if "game_date" in work_df.columns:
        agg = aggregate_block(work_df, league=league, market_type=None,
                              bucket_dimension="game_date", bucket_col="game_date")
        agg.to_csv(overview_dir / f"{league}_summary_by_date.csv", index=False)

    # bet log (per-bet listing using the new column names)
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
    work_df[existing].to_csv(overview_dir / f"{league}_bet_log.csv", index=False)

    # local copy of summary_overall (also written at top level)
    overall = build_summary_overall(work_df, league)
    overall.to_csv(overview_dir / f"{league}_summary_overall.csv", index=False)


# =========================
# TOP-LEVEL SUMMARIES
# =========================

def build_summary_overall(work_df: pd.DataFrame, league: str) -> pd.DataFrame:
    """One row per market_type (per your schema:
       league, market_type, Win, Loss, Push, Total, Win_Pct)."""
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
    """Single-row roll-up across all markets."""
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
        print(f"[{league}] missing work file: {work_path}")
        return

    work = pd.read_csv(work_path)
    if work.empty:
        print(f"[{league}] empty work file; skipping")
        return

    # Normalize
    if "market_type" in work.columns:
        work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    if "side_group" in work.columns:
        work["side_group"] = work["side_group"].astype(str).str.strip().str.upper()

    # Top-level per-league summaries
    build_summary_overall(work,    league).to_csv(BASE / f"{league}_summary_overall.csv",     index=False)
    build_summary_grand_total(work,league).to_csv(BASE / f"{league}_summary_grand_total.csv", index=False)

    # Per-market reports
    for mt in ["moneyline", "spread", "total"]:
        out_dir = REPORT_DIR / league / mt
        write_market_reports(work, league, mt, out_dir)
        write_market_crosses(work, league, mt, out_dir)

    # Overview
    overview_dir = REPORT_DIR / league / "overview"
    write_overview(work, league, overview_dir)

    print(f"[{league}] reports written under {REPORT_DIR / league}")


def run():
    BASE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for league in LEAGUES:
        run_one(league)
    print("Basketball reports complete.")


if __name__ == "__main__":
    run()

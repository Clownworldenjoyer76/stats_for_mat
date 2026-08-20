#!/usr/bin/env python3
# docs/win/basketball/scripts/03_edges/compute_ev_kelly.py
#
# Reads merguiced files (per market, per league) and adds EV, edge-vs-market,
# and Kelly columns using the model/market probabilities ALREADY computed
# upstream in build_juice_files.py.
#
# Inputs:
#   docs/win/basketball/01_merge/01_merguiced/{league}/{market}/*.csv
#       where league in {nba, ncaam, wnba}
#       and market in {moneyline, spread, total}
#
# Outputs:
#   docs/win/basketball/03_edges/ev_kelly/{league}/{market}/*.csv
#
# Key design points:
# - Spread cover probability is NOT recomputed. It comes from the
#   home_spread_model_prob / away_spread_model_prob columns in the merguiced
#   file, which were computed in build_juice_files.py with the correct
#   sign convention. Recomputing here historically reintroduced a sign bug.
# - Total over/under probability is NOT round-tripped through fair_over.
#   It comes from over_model_prob / under_model_prob directly.
# - EV at offered price = (model_prob * book_decimal) - 1
# - Edge vs devigged market = model_prob - market_prob
# - Kelly fraction = (b*p - q)/b where b=d-1, q=1-p, clipped to >= 0
# - All three leagues (NBA, NCAAM, WNBA) supported uniformly.

import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

INPUT_DIR  = Path("docs/win/basketball/01_merge/01_merguiced")
OUTPUT_DIR = Path("docs/win/basketball/03_edges/ev_kelly")
ERROR_DIR  = Path("docs/win/basketball/errors/03_edges")
LOG_FILE   = ERROR_DIR / "compute_ev_kelly.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 70,
        f"SUMMARY  {_now()}",
        "=" * 70,
        f"  files_processed   : {summary['files_processed']}",
        f"  rows_processed    : {summary['rows_processed']}",
        f"  moneyline_files   : {summary['moneyline_files']}",
        f"  spread_files      : {summary['spread_files']}",
        f"  total_files       : {summary['total_files']}",
        f"  skipped           : {summary['skipped']}",
        f"  schema_errors     : {summary['schema_errors']}",
        f"  errors            : {summary['errors']}",
        "",
        f"  {'file':<48} {'league':>6} {'market':<10} {'rows':>6} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<48} {pf['league']:>6} {pf['market']:<10} "
            f"{pf['rows']:>6} {pf['status']:>10}"
        )
    status = "SUCCESS" if (summary["errors"] == 0 and summary["schema_errors"] == 0) \
        else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 70]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_ev(model_prob, book_decimal):
    """EV per $1 staked at the book's offered decimal odds."""
    return (model_prob * book_decimal) - 1


def compute_kelly(model_prob, book_decimal):
    """
    Kelly fraction for a binary bet: f = (b*p - q) / b
    where b = decimal - 1, p = model_prob, q = 1 - p.
    Negative results (negative-EV bets) are returned as NaN so downstream
    code can cleanly distinguish "no bet" from "bet 0".
    """
    b = book_decimal - 1
    q = 1 - model_prob
    k = ((b * model_prob) - q) / b
    return k.clip(lower=0).fillna(0)

def atomic_write_csv(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# =========================
# MARKET PROCESSORS
# =========================

def process_moneyline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required columns from merguiced ML file:
      home_model_prob, away_model_prob          (model)
      home_market_prob, away_market_prob        (devigged market)
      home_dk_moneyline_decimal, away_dk_moneyline_decimal  (book)
    """
    df = to_numeric(df, [
        "home_model_prob", "away_model_prob",
        "home_market_prob", "away_market_prob",
        "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    ])
    validate_columns(df, [
        "home_model_prob", "away_model_prob",
        "home_market_prob", "away_market_prob",
        "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    ])

    # EV at offered price
    df["home_ml_ev"] = compute_ev(df["home_model_prob"], df["home_dk_moneyline_decimal"])
    df["away_ml_ev"] = compute_ev(df["away_model_prob"], df["away_dk_moneyline_decimal"])

    # Edge vs devigged market
    df["home_ml_edge_vs_market"] = df["home_model_prob"] - df["home_market_prob"]
    df["away_ml_edge_vs_market"] = df["away_model_prob"] - df["away_market_prob"]

    # Kelly fraction (NaN if negative EV)
    df["home_ml_kelly"] = compute_kelly(df["home_model_prob"], df["home_dk_moneyline_decimal"])
    df["away_ml_kelly"] = compute_kelly(df["away_model_prob"], df["away_dk_moneyline_decimal"])

    # Convenience: percent versions
    df["home_ml_ev_pct"]              = df["home_ml_ev"] * 100
    df["away_ml_ev_pct"]              = df["away_ml_ev"] * 100
    df["home_ml_edge_vs_market_pct"]  = df["home_ml_edge_vs_market"] * 100
    df["away_ml_edge_vs_market_pct"]  = df["away_ml_edge_vs_market"] * 100

    return df


def process_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required columns from merguiced spread file:
      home_spread_model_prob, away_spread_model_prob       (model — sign fix applied upstream)
      home_spread_market_prob, away_spread_market_prob     (devigged market)
      home_dk_spread_decimal, away_dk_spread_decimal       (book)
    """
    df = to_numeric(df, [
        "home_spread_model_prob", "away_spread_model_prob",
        "home_spread_market_prob", "away_spread_market_prob",
        "home_dk_spread_decimal", "away_dk_spread_decimal",
    ])
    validate_columns(df, [
        "home_spread_model_prob", "away_spread_model_prob",
        "home_spread_market_prob", "away_spread_market_prob",
        "home_dk_spread_decimal", "away_dk_spread_decimal",
    ])

    df["home_spread_ev"] = compute_ev(df["home_spread_model_prob"], df["home_dk_spread_decimal"])
    df["away_spread_ev"] = compute_ev(df["away_spread_model_prob"], df["away_dk_spread_decimal"])

    df["home_spread_edge_vs_market"] = df["home_spread_model_prob"] - df["home_spread_market_prob"]
    df["away_spread_edge_vs_market"] = df["away_spread_model_prob"] - df["away_spread_market_prob"]

    df["home_spread_kelly"] = compute_kelly(df["home_spread_model_prob"], df["home_dk_spread_decimal"])
    df["away_spread_kelly"] = compute_kelly(df["away_spread_model_prob"], df["away_dk_spread_decimal"])

    df["home_spread_ev_pct"]              = df["home_spread_ev"] * 100
    df["away_spread_ev_pct"]              = df["away_spread_ev"] * 100
    df["home_spread_edge_vs_market_pct"]  = df["home_spread_edge_vs_market"] * 100
    df["away_spread_edge_vs_market_pct"]  = df["away_spread_edge_vs_market"] * 100

    return df


def process_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required columns from merguiced total file:
      over_model_prob, under_model_prob              (model — Poisson removed upstream)
      over_market_prob, under_market_prob            (devigged market)
      dk_total_over_decimal, dk_total_under_decimal  (book)
    """
    df = to_numeric(df, [
        "over_model_prob", "under_model_prob",
        "over_market_prob", "under_market_prob",
        "dk_total_over_decimal", "dk_total_under_decimal",
    ])
    validate_columns(df, [
        "over_model_prob", "under_model_prob",
        "over_market_prob", "under_market_prob",
        "dk_total_over_decimal", "dk_total_under_decimal",
    ])

    df["over_ev"]  = compute_ev(df["over_model_prob"],  df["dk_total_over_decimal"])
    df["under_ev"] = compute_ev(df["under_model_prob"], df["dk_total_under_decimal"])

    df["over_edge_vs_market"]  = df["over_model_prob"]  - df["over_market_prob"]
    df["under_edge_vs_market"] = df["under_model_prob"] - df["under_market_prob"]

    df["over_kelly"]  = compute_kelly(df["over_model_prob"],  df["dk_total_over_decimal"])
    df["under_kelly"] = compute_kelly(df["under_model_prob"], df["dk_total_under_decimal"])

    df["over_ev_pct"]               = df["over_ev"]  * 100
    df["under_ev_pct"]              = df["under_ev"] * 100
    df["over_edge_vs_market_pct"]   = df["over_edge_vs_market"]  * 100
    df["under_edge_vs_market_pct"]  = df["under_edge_vs_market"] * 100

    return df


MARKET_PROCESSORS = {
    "moneyline": process_moneyline,
    "spread":    process_spread,
    "total":     process_total,
}


# =========================
# MAIN LOOP
# =========================

def process_one(path: Path, league: str, market: str,
                summary: dict, per_file: list) -> None:
    pf = {"name": path.name, "league": league.upper(), "market": market,
          "rows": 0, "status": "ok"}

    try:
        df = pd.read_csv(path)
        if df.empty:
            _log(f"{path.name} empty — skipping")
            pf["status"] = "empty"
            summary["skipped"] += 1
            per_file.append(pf)
            return

        pf["rows"] = len(df)
        summary["rows_processed"] += len(df)

        df = MARKET_PROCESSORS[market](df)

        out_dir = OUTPUT_DIR / league / market
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name
        atomic_write_csv(df, out_path)

        summary["files_processed"]      += 1
        summary[f"{market}_files"]      += 1
        _log(f"WROTE: {out_path} ({len(df)} rows)")

    except ValueError as e:
        _log(f"{path.name} schema error: {e}", "ERROR")
        pf["status"] = "schema_error"
        summary["schema_errors"] += 1
    except Exception as e:
        _log(f"{path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
        pf["status"] = "error"
        summary["errors"] += 1

    per_file.append(pf)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== compute_ev_kelly RUN {_now()} ===\n")

    summary = {
        "files_processed": 0, "rows_processed": 0,
        "moneyline_files": 0, "spread_files": 0, "total_files": 0,
        "skipped": 0, "schema_errors": 0, "errors": 0,
    }
    per_file = []

    # Wipe old outputs
    for league in LEAGUES:
        for market in MARKETS:
            folder = OUTPUT_DIR / league / market
            folder.mkdir(parents=True, exist_ok=True)
            for stale in folder.glob("*.csv"):
                stale.unlink(missing_ok=True)
    _log("Wiped old outputs.")

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    try:
        for league in LEAGUES:
            for market in MARKETS:
                folder = INPUT_DIR / league / market
                if not folder.exists():
                    _log(f"INPUT FOLDER MISSING: {folder}", "WARN")
                    continue

                files = sorted(folder.glob("*.csv"))
                if not files:
                    _log(f"NO FILES: league={league} market={market}", "WARN")
                    continue

                for f in files:
                    process_one(f, league, market, summary, per_file)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)
    print("compute_ev_kelly complete.")


if __name__ == "__main__":
    main()

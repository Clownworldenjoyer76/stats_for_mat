#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/build_juice_files.py

import csv
import traceback
import sys
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import norm
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

LEAGUE_SETTINGS = {
    "NBA": {
        "ML_EDGE":     0.020,
        "TOTAL_EDGE":  0.030,
        "SPREAD_EDGE": 0.050,
        "TOTAL_STD":   19.1345,
        "SPREAD_STD":  14.1837,
    },
    "NCAAM": {
        "ML_EDGE":     0.015,
        "TOTAL_EDGE":  0.03,
        "SPREAD_EDGE": 0.100,
        "TOTAL_STD":   17.7495,
        "SPREAD_STD":  11.2375,
    },
    "WNBA": {
        "ML_EDGE":     0.005,
        "TOTAL_EDGE":  0.030,
        "SPREAD_EDGE": 0.050,
        "TOTAL_STD":   16.6675,
        "SPREAD_STD":  13.0424,
    },
}

LEAGUES = ["nba", "ncaam", "wnba"]

# ============================================================
# PATHS
# ============================================================

INPUT_DIR  = Path("docs/win/basketball/01_merge")
OUTPUT_DIR = Path("docs/win/basketball/01_merge/01_merguiced")
ERROR_DIR  = Path("docs/win/basketball/errors/01_merge")
LOG_FILE   = ERROR_DIR / "build_juice_files.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== build_juice_files RUN {datetime.now(timezone.utc).isoformat()} ===\n\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def audit(stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            f.write(f"  MSG: {msg}\n")
        if df is not None:
            f.write(f"  ROWS: {len(df)}\n")
        f.write("-" * 40 + "\n")


# ============================================================
# HELPERS
# ============================================================

def american_to_decimal(odds):
    if pd.isna(odds) or str(odds).strip() == "":
        return ""
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return ""
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def to_american(dec):
    if dec == "" or pd.isna(dec) or float(dec) <= 1:
        return ""
    dec = float(dec)
    if dec >= 2:
        return f"+{int((dec - 1) * 100)}"
    return f"-{int(100 / (dec - 1))}"


def clamp_probability(p):
    # Loosened from [0.05, 0.95] to [0.01, 0.99]. Trust the model further out
    # on heavy favorites/extreme totals where the predictions are calibrated.
    return min(max(p, 0.01), 0.99)


def safe_implied_prob(decimal_value):
    """Convert a decimal odds value to implied probability. Returns '' if invalid."""
    if decimal_value == "" or pd.isna(decimal_value):
        return ""
    try:
        d = float(decimal_value)
    except (ValueError, TypeError):
        return ""
    if d <= 0:
        return ""
    return 1.0 / d


def devig_pair(p_a, p_b):
    """
    Normalize a pair of implied probabilities so they sum to 1.
    Removes the bookmaker's overround (vig).
    Returns ('', '') if either input is missing/invalid.
    """
    if p_a == "" or p_b == "" or pd.isna(p_a) or pd.isna(p_b):
        return "", ""
    try:
        a = float(p_a)
        b = float(p_b)
    except (ValueError, TypeError):
        return "", ""
    s = a + b
    if s <= 0:
        return "", ""
    return a / s, b / s


def wipe_outputs():
    for league in LEAGUES:
        for subdir in ["moneyline", "spread", "total"]:
            folder = OUTPUT_DIR / league / subdir
            folder.mkdir(parents=True, exist_ok=True)
            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)
    log("Wiped all output folders.")


# ============================================================
# PROCESS MONEYLINE
# ============================================================

def process_moneyline(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    ML_EDGE = settings["ML_EDGE"]

    ml_df = df.copy()

    # Decimal odds (recompute from American to be safe)
    ml_df["away_decimal"] = ml_df["away_dk_moneyline_american"].apply(american_to_decimal)
    ml_df["home_decimal"] = ml_df["home_dk_moneyline_american"].apply(american_to_decimal)

    # Raw implied probabilities from book decimals (not yet devigged)
    ml_df["away_implied_prob"] = ml_df["away_decimal"].apply(safe_implied_prob)
    ml_df["home_implied_prob"] = ml_df["home_decimal"].apply(safe_implied_prob)

    # Devigged market probabilities — what the book "really" thinks after removing the hold
    market_pairs = ml_df.apply(
        lambda r: devig_pair(r["away_implied_prob"], r["home_implied_prob"]),
        axis=1,
    )
    ml_df["away_market_prob"] = market_pairs.apply(lambda t: t[0])
    ml_df["home_market_prob"] = market_pairs.apply(lambda t: t[1])

    # Model probabilities pass through directly from the predictions stage
    ml_df["home_model_prob"] = pd.to_numeric(ml_df["home_prob"], errors="coerce")
    ml_df["away_model_prob"] = pd.to_numeric(ml_df["away_prob"], errors="coerce")

    # Fair decimal odds = 1 / model_prob
    ml_df["away_fair"] = ml_df["away_model_prob"].apply(lambda x: 1 / float(x) if pd.notna(x) and float(x) > 0 else "")
    ml_df["home_fair"] = ml_df["home_model_prob"].apply(lambda x: 1 / float(x) if pd.notna(x) and float(x) > 0 else "")

    ml_df["away_acceptable_decimal_moneyline"]  = ml_df["away_fair"].apply(lambda x: float(x) * (1 + ML_EDGE) if x != "" else "")
    ml_df["home_acceptable_decimal_moneyline"]  = ml_df["home_fair"].apply(lambda x: float(x) * (1 + ML_EDGE) if x != "" else "")
    ml_df["away_acceptable_american_moneyline"] = ml_df["away_acceptable_decimal_moneyline"].apply(to_american)
    ml_df["home_acceptable_american_moneyline"] = ml_df["home_acceptable_decimal_moneyline"].apply(to_american)

    out_path = OUTPUT_DIR / league / "moneyline" / f"{date}_{league_upper}_moneyline.csv"
    ml_df.to_csv(out_path, index=False)
    return out_path, len(ml_df)


# ============================================================
# PROCESS TOTALS
# ============================================================

def process_totals(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    TOTAL_EDGE = settings["TOTAL_EDGE"]
    TOTAL_STD  = settings["TOTAL_STD"]

    total_df = df.copy()

    over_model_prob  = []
    under_model_prob = []
    fair_over        = []
    fair_under       = []
    acc_over         = []
    acc_under        = []

    for _, row in total_df.iterrows():
        try:
            T    = float(row["total"])
            mean = float(row["total_projected_points"])
        except (ValueError, TypeError):
            over_model_prob.append("")
            under_model_prob.append("")
            fair_over.append("")
            fair_under.append("")
            acc_over.append("")
            acc_under.append("")
            continue

        # Normal distribution for all leagues. NCAAM previously used Poisson,
        # which forces variance = mean. With NCAAM totals around 140 the
        # implied std was ~12, far tighter than reality (observed total MAE
        # was ~14 implying std ~18+). The Poisson assumption produced
        # artificially confident probabilities, hit the clamp, and inflated
        # apparent edges. Normal with the league's TOTAL_STD reflects
        # observed variance.
        z       = (T - mean) / TOTAL_STD
        p_under = norm.cdf(z)

        p_under = clamp_probability(p_under)
        p_over  = 1 - p_under

        over_model_prob.append(p_over)
        under_model_prob.append(p_under)

        fair_over_dec  = 1 / p_over
        fair_under_dec = 1 / p_under
        fair_over.append(fair_over_dec)
        fair_under.append(fair_under_dec)
        acc_over.append(fair_over_dec  * (1 + TOTAL_EDGE))
        acc_under.append(fair_under_dec * (1 + TOTAL_EDGE))

    total_df["over_model_prob"]  = over_model_prob
    total_df["under_model_prob"] = under_model_prob
    total_df["fair_over"]        = fair_over
    total_df["fair_under"]       = fair_under
    total_df["acceptable_over"]  = acc_over
    total_df["acceptable_under"] = acc_under

    # Devigged market probabilities from the over/under decimal odds
    total_df["over_implied_prob"]  = total_df["dk_total_over_decimal"].apply(safe_implied_prob)
    total_df["under_implied_prob"] = total_df["dk_total_under_decimal"].apply(safe_implied_prob)

    market_pairs = total_df.apply(
        lambda r: devig_pair(r["over_implied_prob"], r["under_implied_prob"]),
        axis=1,
    )
    total_df["over_market_prob"]  = market_pairs.apply(lambda t: t[0])
    total_df["under_market_prob"] = market_pairs.apply(lambda t: t[1])

    out_path = OUTPUT_DIR / league / "total" / f"{date}_{league_upper}_total.csv"
    total_df.to_csv(out_path, index=False)
    return out_path, len(total_df)


# ============================================================
# PROCESS SPREAD
# ============================================================

def process_spread(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    SPREAD_EDGE = settings["SPREAD_EDGE"]
    SPREAD_STD  = settings["SPREAD_STD"]

    spread_df = df.copy()

    home_model_prob = []
    away_model_prob = []
    fair_home       = []
    fair_away       = []
    acc_home        = []
    acc_away        = []

    for _, row in spread_df.iterrows():
        try:
            mean_margin = float(row["home_projected_points"]) - float(row["away_projected_points"])
            home_line   = float(row["home_spread"])
        except (ValueError, TypeError):
            home_model_prob.append("")
            away_model_prob.append("")
            fair_home.append("")
            fair_away.append("")
            acc_home.append("")
            acc_away.append("")
            continue

        # Spread cover convention:
        # If home_spread = -3.5 (home favored by 3.5), home covers iff
        # home_margin > 3.5, i.e., home_margin > -home_line.
        # We compute P(margin > -home_line) where margin ~ N(mean_margin, SPREAD_STD).
        cover_threshold = -home_line
        p_home = 1 - norm.cdf(cover_threshold, loc=mean_margin, scale=SPREAD_STD)
        p_home = clamp_probability(p_home)
        p_away = 1 - p_home

        home_model_prob.append(p_home)
        away_model_prob.append(p_away)

        fair_home_dec = 1 / p_home
        fair_away_dec = 1 / p_away
        fair_home.append(fair_home_dec)
        fair_away.append(fair_away_dec)
        acc_home.append(fair_home_dec * (1 + SPREAD_EDGE))
        acc_away.append(fair_away_dec * (1 + SPREAD_EDGE))

    spread_df["home_spread_model_prob"]          = home_model_prob
    spread_df["away_spread_model_prob"]          = away_model_prob
    spread_df["fair_home_spread_decimal"]        = fair_home
    spread_df["fair_away_spread_decimal"]        = fair_away
    spread_df["home_acceptable_spread_decimal"]  = acc_home
    spread_df["away_acceptable_spread_decimal"]  = acc_away
    spread_df["home_acceptable_spread_american"] = spread_df["home_acceptable_spread_decimal"].apply(to_american)
    spread_df["away_acceptable_spread_american"] = spread_df["away_acceptable_spread_decimal"].apply(to_american)

    # Devigged market probabilities from the spread decimal odds
    spread_df["home_spread_implied_prob"] = spread_df["home_dk_spread_decimal"].apply(safe_implied_prob)
    spread_df["away_spread_implied_prob"] = spread_df["away_dk_spread_decimal"].apply(safe_implied_prob)

    market_pairs = spread_df.apply(
        lambda r: devig_pair(r["home_spread_implied_prob"], r["away_spread_implied_prob"]),
        axis=1,
    )
    spread_df["home_spread_market_prob"] = market_pairs.apply(lambda t: t[0])
    spread_df["away_spread_market_prob"] = market_pairs.apply(lambda t: t[1])

    out_path = OUTPUT_DIR / league / "spread" / f"{date}_{league_upper}_spread.csv"
    spread_df.to_csv(out_path, index=False)
    return out_path, len(spread_df)


# ============================================================
# MAIN
# ============================================================

def main():
    files_written = []
    files_skipped = 0

    try:
        wipe_outputs()

        for league in LEAGUES:
            league_upper = league.upper()
            settings     = LEAGUE_SETTINGS[league_upper]

            for market_type in ["moneyline", "spread", "total"]:
                input_folder = INPUT_DIR / league / market_type
                if not input_folder.exists():
                    log(f"INPUT FOLDER NOT FOUND: {input_folder}")
                    continue

                input_files = sorted(input_folder.glob(f"*_{league_upper}_{market_type}.csv"))

                if not input_files:
                    log(f"NO INPUT FILES: {input_folder}")
                    continue

                for file_path in input_files:
                    try:
                        df = pd.read_csv(file_path)

                        if df.empty:
                            log(f"EMPTY: {file_path.name} — skipping")
                            files_skipped += 1
                            continue

                        # Parse date from filename: {date}_{LEAGUE}_{type}.csv
                        date = file_path.stem.replace(f"_{league_upper}_{market_type}", "")

                        if market_type == "moneyline":
                            out_path, count = process_moneyline(df, date, league_upper, settings, league)
                        elif market_type == "total":
                            out_path, count = process_totals(df, date, league_upper, settings, league)
                        elif market_type == "spread":
                            out_path, count = process_spread(df, date, league_upper, settings, league)

                        files_written.append((str(out_path), count))
                        log(f"WROTE {out_path.name} ({count} rows)")
                        audit(market_type.upper(), "SUCCESS", file_path.name, df)

                    except Exception as e:
                        log(f"ERROR processing {file_path.name}: {e}\n{traceback.format_exc()}")
                        files_skipped += 1

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")
        log(f"Files skipped: {files_skipped}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")
        print("build_juice_files complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

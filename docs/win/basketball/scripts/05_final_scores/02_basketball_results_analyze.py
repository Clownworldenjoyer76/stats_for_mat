#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/02_basketball_results_analyze.py
#
# Reads graded bet files from script 01 and adds derived columns + bucket
# labels for downstream reporting. Uses the bet_* columns already on the row
# (bet_ev, bet_edge_vs_market, bet_kelly, bet_model_prob, bet_odds_american,
# bet_line, bet_stake_pct) — does NOT look up home/away columns.
#
# Inputs:
#   docs/win/basketball/05_final_scores/results/nba/graded/NBA_final.csv
#   docs/win/basketball/05_final_scores/results/ncaam/graded/NCAAM_final.csv
#   docs/win/basketball/05_final_scores/results/wnba/graded/WNBA_final.csv
#
# Outputs:
#   docs/win/basketball/05_final_scores/work_nba.csv
#   docs/win/basketball/05_final_scores/work_ncaam.csv
#   docs/win/basketball/05_final_scores/work_wnba.csv
#
# Buckets added per row:
#   ev_bucket               (signed; pulls from bet_ev)
#   edge_vs_market_bucket   (signed; pulls from bet_edge_vs_market in pp)
#   kelly_bucket            (pulls from bet_kelly)
#   model_prob_bucket       (pulls from bet_model_prob)
#   odds_bucket             (pulls from bet_odds_american; meaningful for ML)
#   spread_bucket           (pulls from bet_line; meaningful for spread)
#   total_bucket            (pulls from bet_line; meaningful for total)
#   dow_bucket              (Mon..Sun from game_date)
#   month_bucket            (Jan..Dec from game_date)
#
# profit_unit and profit_kelly carry through unchanged from script 01.

from datetime import datetime
from pathlib import Path

import pandas as pd

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

BASE       = Path("docs/win/basketball")
INPUT_DIR  = BASE / "05_final_scores/results"
OUTPUT_DIR = BASE / "05_final_scores"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Wipe basketball intermediate outputs before regenerating
for league in LEAGUES:
    (OUTPUT_DIR / f"work_{league}.csv").unlink(missing_ok=True)


# =========================
# HELPERS
# =========================

def to_float(value):
    try:
        if value is None or pd.isna(value):
            return pd.NA
        return float(value)
    except Exception:
        return pd.NA


def build_side_group(row):
    market_type = str(row.get("market_type", "")).strip().lower()
    bet_side    = str(row.get("bet_side", "")).strip().lower()

    if market_type in {"moneyline", "spread"}:
        if bet_side == "home":
            return "HOME"
        if bet_side == "away":
            return "AWAY"

    if market_type == "total":
        if bet_side == "over":
            return "OVER"
        if bet_side == "under":
            return "UNDER"

    return ""


# ---- bucket functions ----

def ev_bucket(value):
    """Signed buckets on bet_ev (decimal). Negative bucket exists because
    ev can technically be <= 0 if a row leaked through filters."""
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < 0:
        return "<0"
    if v < 0.01:
        return "0.00_to_0.0099"
    if v < 0.02:
        return "0.01_to_0.0199"
    if v < 0.03:
        return "0.02_to_0.0299"
    if v < 0.04:
        return "0.03_to_0.0399"
    if v < 0.05:
        return "0.04_to_0.0499"
    if v < 0.075:
        return "0.05_to_0.0749"
    if v < 0.10:
        return "0.075_to_0.0999"
    return "0.10_plus"


def edge_vs_market_bucket(value):
    """bet_edge_vs_market is in percentage points (model_prob - market_prob) * 100.
    Signed; negative means model thinks the bet is worse than the market does."""
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < -10:
        return "below_neg10pp"
    if v < -5:
        return "neg10_to_neg5pp"
    if v < -2:
        return "neg5_to_neg2pp"
    if v < 0:
        return "neg2_to_0pp"
    if v < 1:
        return "0_to_1pp"
    if v < 2:
        return "1_to_2pp"
    if v < 3:
        return "2_to_3pp"
    if v < 5:
        return "3_to_5pp"
    if v < 7:
        return "5_to_7pp"
    if v < 10:
        return "7_to_10pp"
    return "10pp_plus"


def kelly_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < 0:
        return "<0"
    if v < 0.025:
        return "0.0_to_2.5pct"
    if v < 0.05:
        return "2.5_to_5pct"
    if v < 0.10:
        return "5_to_10pct"
    if v < 0.20:
        return "10_to_20pct"
    return "20pct_plus"


def model_prob_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < 0.20:
        return "below_0.20"
    if v < 0.30:
        return "0.20_to_0.30"
    if v < 0.40:
        return "0.30_to_0.40"
    if v < 0.50:
        return "0.40_to_0.50"
    if v < 0.60:
        return "0.50_to_0.60"
    if v < 0.70:
        return "0.60_to_0.70"
    if v < 0.80:
        return "0.70_to_0.80"
    if v < 0.90:
        return "0.80_to_0.90"
    return "0.90_plus"


def odds_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v <= -200:
        return "minus_200_or_lower"
    if v <= -150:
        return "minus_199_to_minus_150"
    if v <= -125:
        return "minus_149_to_minus_125"
    if v <= -110:
        return "minus_124_to_minus_110"
    if v <= -101:
        return "minus_109_to_minus_101"
    if v <= 100:
        return "minus_100_to_plus_100"
    if v <= 125:
        return "plus_101_to_plus_125"
    if v <= 150:
        return "plus_126_to_plus_150"
    if v <= 200:
        return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def spread_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    abs_v = abs(float(v))
    if abs_v < 1:    return "0.0_to_0.9"
    if abs_v < 2:    return "1.0_to_1.9"
    if abs_v < 3:    return "2.0_to_2.9"
    if abs_v < 4:    return "3.0_to_3.9"
    if abs_v < 5:    return "4.0_to_4.9"
    if abs_v < 6:    return "5.0_to_5.9"
    if abs_v < 7:    return "6.0_to_6.9"
    if abs_v < 8:    return "7.0_to_7.9"
    if abs_v < 9:    return "8.0_to_8.9"
    if abs_v < 10:   return "9.0_to_9.9"
    if abs_v < 12:   return "10.0_to_11.9"
    if abs_v < 15:   return "12.0_to_14.9"
    return "15.0_plus"


def total_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    start = int(float(v) // 5) * 5
    end = start + 4.9
    return f"{start}_to_{end:.1f}"


# ---- date buckets ----

DOW_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def parse_date(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    # game_date in your pipeline is YYYY_MM_DD; tolerate a couple of common variants
    for fmt in ("%Y_%m_%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def dow_bucket(value):
    dt = parse_date(value)
    if dt is None:
        return "UNBUCKETED"
    return DOW_NAMES[dt.weekday()]


def month_bucket(value):
    dt = parse_date(value)
    if dt is None:
        return "UNBUCKETED"
    return MONTH_NAMES[dt.month - 1]


# =========================
# PREPARE
# =========================

def prepare(df: pd.DataFrame, league_label: str) -> pd.DataFrame:
    work = df.copy()

    # Normalize string columns
    if "market_type" in work.columns:
        work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    if "bet_side" in work.columns:
        work["bet_side"] = work["bet_side"].astype(str).str.strip().str.lower()
    if "bet_result" in work.columns:
        work["bet_result"] = work["bet_result"].astype(str).str.strip().str.title()

    # Tag with league
    work["market"] = league_label

    # Side grouping (HOME/AWAY/OVER/UNDER)
    work["side_group"] = work.apply(build_side_group, axis=1)

    # Source columns (use bet_* directly — already on the row from stage 04)
    if "bet_ev" not in work.columns:
        work["bet_ev"] = pd.NA
    if "bet_edge_vs_market" not in work.columns:
        work["bet_edge_vs_market"] = pd.NA
    if "bet_kelly" not in work.columns:
        work["bet_kelly"] = pd.NA
    if "bet_model_prob" not in work.columns:
        work["bet_model_prob"] = pd.NA
    if "bet_odds_american" not in work.columns:
        work["bet_odds_american"] = pd.NA
    if "bet_line" not in work.columns:
        work["bet_line"] = pd.NA
    if "bet_stake_pct" not in work.columns:
        work["bet_stake_pct"] = pd.NA

    # profit_unit and profit_kelly should already be on the row from script 01;
    # if missing, leave them missing rather than try to recompute here.
    if "profit_unit" not in work.columns:
        work["profit_unit"] = pd.NA
    if "profit_kelly" not in work.columns:
        work["profit_kelly"] = pd.NA

    # Bucket columns
    work["ev_bucket"]              = work["bet_ev"].apply(ev_bucket)
    work["edge_vs_market_bucket"]  = work["bet_edge_vs_market"].apply(edge_vs_market_bucket)
    work["kelly_bucket"]           = work["bet_kelly"].apply(kelly_bucket)
    work["model_prob_bucket"]      = work["bet_model_prob"].apply(model_prob_bucket)

    # odds_bucket only applies cleanly to moneyline rows — but compute for all
    # so the column exists; rows where it doesn't apply will be UNBUCKETED.
    work["odds_bucket"] = work["bet_odds_american"].apply(odds_bucket)

    # spread_bucket only meaningful for spread rows; total_bucket only for total rows.
    # Compute selectively so unrelated markets get UNBUCKETED rather than nonsense.
    def spread_for_row(row):
        if str(row.get("market_type", "")).lower() == "spread":
            return spread_bucket(row.get("bet_line"))
        return "UNBUCKETED"

    def total_for_row(row):
        if str(row.get("market_type", "")).lower() == "total":
            return total_bucket(row.get("bet_line"))
        return "UNBUCKETED"

    work["spread_bucket"] = work.apply(spread_for_row, axis=1)
    work["total_bucket"]  = work.apply(total_for_row,  axis=1)

    # Date buckets from game_date
    if "game_date" in work.columns:
        work["dow_bucket"]   = work["game_date"].apply(dow_bucket)
        work["month_bucket"] = work["game_date"].apply(month_bucket)
    else:
        work["dow_bucket"]   = "UNBUCKETED"
        work["month_bucket"] = "UNBUCKETED"

    return work


# =========================
# RUN
# =========================

def run_one(league: str) -> None:
    upper = league.upper()
    in_path  = INPUT_DIR / league / "graded" / f"{upper}_final.csv"
    out_path = OUTPUT_DIR / f"work_{league}.csv"

    if not in_path.exists():
        print(f"[{upper}] input missing: {in_path} — skipping")
        return

    df = pd.read_csv(in_path)
    if df.empty:
        print(f"[{upper}] input is empty: {in_path} — writing empty work file")
        df.to_csv(out_path, index=False)
        return

    work = prepare(df, upper)
    work.to_csv(out_path, index=False)
    print(f"[{upper}] wrote {len(work)} rows -> {out_path}")


def run():
    for league in LEAGUES:
        run_one(league)
    print("Analyze complete.")


if __name__ == "__main__":
    run()

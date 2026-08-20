#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/02_mlb_results_analyze_AM.py

import pandas as pd
from pathlib import Path

MLB_INPUT  = Path("docs/win/baseball/mlb/05_final_scores/morning/results/graded/MLB_final.csv")
OUTPUT_DIR = Path("docs/win/baseball/mlb/05_final_scores/morning/intermediate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


###############################################################
######################## HELPERS ##############################
###############################################################

def to_float(value):
    try:
        return float(value)
    except Exception:
        return pd.NA


###############################################################
######################## FIELD BUILDERS #######################
###############################################################

def build_side_group(row):
    market_type = str(row.get("market_type", "")).strip().lower()
    bet_side    = str(row.get("bet_side",    "")).strip().lower()

    if market_type in {"moneyline", "run_line"}:
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


def build_odds_value(row):
    return to_float(row.get("dk_odds_american"))


def build_moneyline_odds_value(row):
    if str(row.get("market_type", "")).strip().lower() != "moneyline":
        return pd.NA
    return to_float(row.get("dk_odds_american"))


def build_run_line_value(row):
    side_group = str(row.get("side_group", "")).strip().upper()

    if side_group == "HOME":
        return to_float(row.get("home_run_line"))
    if side_group == "AWAY":
        return to_float(row.get("away_run_line"))

    return pd.NA


def build_total_value(row):
    return to_float(row.get("total"))


def build_day_night(row):
    raw = str(row.get("game_time", "")).strip()

    if not raw:
        return ""

    try:
        for fmt in ("%I:%M %p", "%H:%M", "%I:%M%p", "%H:%M:%S"):
            try:
                from datetime import datetime
                t = datetime.strptime(raw, fmt)
                return "Day" if t.hour < 17 else "Night"
            except ValueError:
                continue
    except Exception:
        pass

    return ""


###############################################################
######################## BUCKET FUNCTIONS #####################
###############################################################

def ev_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"
    if value < 0:
        return "<0"
    if value < 0.01:
        return "0.00_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.04:
        return "0.03_to_0.0399"
    if value < 0.05:
        return "0.04_to_0.0499"
    if value < 0.075:
        return "0.05_to_0.0749"
    if value < 0.10:
        return "0.075_to_0.0999"

    return "0.10_plus"


def odds_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"
    if value <= -200:
        return "minus_200_or_lower"
    if value <= -150:
        return "minus_199_to_minus_150"
    if value <= -125:
        return "minus_149_to_minus_125"
    if value <= -110:
        return "minus_124_to_minus_110"
    if value <= -101:
        return "minus_109_to_minus_101"
    if value <= 100:
        return "minus_100_to_plus_100"
    if value <= 125:
        return "plus_101_to_plus_125"
    if value <= 150:
        return "plus_126_to_plus_150"
    if value <= 200:
        return "plus_151_to_plus_200"

    return "plus_201_or_higher"


def run_line_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"

    abs_val = abs(float(value))

    if abs_val < 1:  return "0.0_to_0.9"
    if abs_val < 2:  return "1.0_to_1.9"
    if abs_val < 3:  return "2.0_to_2.9"
    if abs_val < 4:  return "3.0_to_3.9"
    if abs_val < 5:  return "4.0_to_4.9"
    if abs_val < 6:  return "5.0_to_5.9"
    if abs_val < 7:  return "6.0_to_6.9"
    if abs_val < 8:  return "7.0_to_7.9"
    if abs_val < 9:  return "8.0_to_8.9"
    if abs_val < 10: return "9.0_to_9.9"
    if abs_val < 12: return "10.0_to_11.9"
    if abs_val < 15: return "12.0_to_14.9"

    return "15.0_plus"


def total_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"

    start = int(float(value) // 5) * 5
    return f"{start}_to_{start + 4.9:.1f}"


def kelly_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"
    if value <= 0:
        return "zero_or_below"
    if value < 0.01:
        return "0.001_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.05:
        return "0.03_to_0.0499"
    if value < 0.10:
        return "0.05_to_0.0999"
    if value < 0.15:
        return "0.10_to_0.1499"
    if value < 0.20:
        return "0.15_to_0.1999"

    return "0.20_plus"


def model_prob_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"

    pct = float(value) * 100 if float(value) <= 1.0 else float(value)

    if pct < 50: return "<50"
    if pct < 55: return "50_to_54.9"
    if pct < 60: return "55_to_59.9"
    if pct < 65: return "60_to_64.9"
    if pct < 70: return "65_to_69.9"
    if pct < 75: return "70_to_74.9"
    if pct < 80: return "75_to_79.9"

    return "80_plus"


def total_range_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"

    import math
    floor = math.floor(float(value) * 2) / 2
    hi = floor + 0.5

    return f"{floor:.1f}_to_{hi:.1f}"


def run_line_side_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"
    if value > 0:
        return "+1.5"
    if value < 0:
        return "-1.5"

    return "0"


###############################################################
######################## PREPARE ##############################
###############################################################

def prepare(df):
    work = df.copy()

    work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    work["bet_side"]    = work["bet_side"].astype(str).str.strip().str.lower()
    work["bet_result"]  = work["bet_result"].astype(str).str.strip().str.title()

    work["side_group"]           = work.apply(build_side_group, axis=1)
    work["ev_value"]             = work["ev"].apply(to_float)
    work["odds_value"]           = work.apply(build_odds_value, axis=1)
    work["moneyline_odds_value"] = work.apply(build_moneyline_odds_value, axis=1)
    work["run_line_value"]       = work.apply(build_run_line_value, axis=1)
    work["total_value"]          = work.apply(build_total_value, axis=1)
    work["kelly_value"]          = work["kelly"].apply(to_float)
    work["model_prob_value"]     = work["model_prob"].apply(to_float)
    work["day_night"]            = work.apply(build_day_night, axis=1)

    work["low_confidence"] = (
        work["low_confidence"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: "True" if x in {"true", "1", "yes"} else ("False" if x in {"false", "0", "no"} else ""))
    )

    work["ev_bucket"]            = work["ev_value"].apply(ev_bucket)
    work["odds_bucket"]          = work["odds_value"].apply(odds_bucket)
    work["run_line_bucket"]      = work["run_line_value"].apply(run_line_bucket)
    work["total_bucket"]         = work["total_value"].apply(total_bucket)
    work["kelly_bucket"]         = work["kelly_value"].apply(kelly_bucket)
    work["model_prob_bucket"]    = work["model_prob_value"].apply(model_prob_bucket)
    work["win_prob_bucket"]      = work["model_prob_value"].apply(model_prob_bucket)
    work["total_range_bucket"]   = work["total_value"].apply(total_range_bucket)
    work["run_line_side"]        = work["run_line_value"].apply(run_line_side_bucket)

    return work


###############################################################
######################## MAIN #################################
###############################################################

def run():
    if not MLB_INPUT.exists():
        print(f"ERROR: input file not found: {MLB_INPUT}")
        return

    mlb = pd.read_csv(MLB_INPUT, dtype=str)
    mlb = prepare(mlb)

    out = OUTPUT_DIR / "work_mlb.csv"
    mlb.to_csv(out, index=False)

    print(f"MLB morning analyze complete. Rows={len(mlb)} | Out={out}")


if __name__ == "__main__":
    run()

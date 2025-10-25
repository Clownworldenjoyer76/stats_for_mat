#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
from pandas.api.types import is_numeric_dtype

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns never divided (identifiers/meta)
ID_COLS = {"PLAYER", "TEAM", "OPPONENT", "POS", "POSITION", "IS_HOME", "HOME", "AWAY",
           "DOME", "STADIUM", "WEEK", "SEASON", "YEAR", "DATE"}

# Columns considered already per-game / averages / percentages (skip dividing)
# Keep this SIMPLE on purpose.
SKIP_REGEXES = [
    r"/G$",          # per-game columns like YDS/G, TD/G
    r"AVG$",         # averages like RECAVG
    r"AVERAGE$",     # explicit AVERAGE
    r"PCT$",         # percent columns like FG_PCT, XP_PCT, CMP_PCT
    r"%$",           # columns ending with %
    r"RATE$",        # passer rate, etc.
]

def is_skip_col(col: str) -> bool:
    c = col.strip()
    if c.upper() in ID_COLS or c.upper() == "GP":
        return True
    for pat in SKIP_REGEXES:
        if re.search(pat, c, flags=re.IGNORECASE):
            return True
    return False

def out_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def process_one(csv_path: str):
    df = pd.read_csv(csv_path)
    if "GP" not in df.columns:
        raise ValueError(f"{csv_path}: missing GP column")

    # For each numeric column that is NOT a skip column, divide by GP (row-wise).
    for col in df.columns:
        if is_skip_col(col):
            continue
        if not is_numeric_dtype(df[col]):
            # leave non-numeric columns as-is
            continue
        # divide totals by GP; if GP == 0, leave the original value
        df[col] = df[col].div(df["GP"]).where(df["GP"] != 0, df[col])

    out_fp = out_name(csv_path)
    df.to_csv(out_fp, index=False)
    print(f"Wrote {out_fp}")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")
    for fp in files:
        process_one(fp)

if __name__ == "__main__":
    main()

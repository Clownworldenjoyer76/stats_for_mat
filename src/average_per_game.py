#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed/stats"   # <-- corrected as requested
os.makedirs(OUT_DIR, exist_ok=True)

# Columns to never divide by GP (identifiers / meta)
ID_COLS = {
    "PLAYER","TEAM","OPPONENT","POS","POSITION","IS_HOME","HOME","AWAY",
    "DOME","STADIUM","WEEK","SEASON","YEAR","DATE","TEMP_F","WIND_MPH"
}

# Columns already per-game/rates/averages/percentages (skip dividing)
SKIP_DIVIDE_PATTERNS = [
    r"/G$",                 # ends with per-game already
    r"AVG$",                # averages (RECAVG, RAVG, etc.)
    r"AVERAGE$",            # any explicit AVERAGE
    r"PCT$",                # FG_PCT, XP_PCT, CMP_PCT, etc.
    r"%$",                  # columns literally ending with %
    r"RATE$",               # passer rating etc.
    r"QB\s*RAT(ING)?$",     # QB rating variants
    r"Y/R$|Y/T$|Y/A$|Y/ATT$",  # yards per X
    r"ATT/G$|YDS/G$|TD/G$", # common per-game composites
]

def col_matches(name: str, patterns) -> bool:
    for pat in patterns:
        if re.search(pat, name, flags=re.IGNORECASE):
            return True
    return False

def is_skip_divide_col(name: str) -> bool:
    if name.upper() in ID_COLS:
        return True
    return col_matches(name, SKIP_DIVIDE_PATTERNS)

def out_name_from(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def process_file(fp: str):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]

    if "GP" not in df.columns:
        raise ValueError(f"{fp}: Missing required 'GP' column.")

    # Divide totals by GP for numeric columns that are NOT already per-game/averages/percentages
    for col in df.columns:
        if col == "GP" or col.upper() in ID_COLS:
            continue
        if is_skip_divide_col(col):
            continue
        # Only operate on columns that are already numeric dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] / df["GP"]

    df.to_csv(out_name_from(fp), index=False)
    print(f"✓ Wrote {out_name_from(fp)}")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")
    for fp in files:
        process_file(fp)

if __name__ == "__main__":
    main()

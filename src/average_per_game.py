#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns that should NEVER be divided by GP (identifiers / meta)
ID_COLS = {
    "PLAYER","TEAM","OPPONENT","POS","POSITION","IS_HOME","HOME","AWAY",
    "DOME","STADIUM","WEEK","SEASON","YEAR","DATE","TEMP_F","WIND_MPH"
}

# Regex rules: columns to SKIP dividing by GP (already rates/averages/percentages)
SKIP_DIVIDE_PATTERNS = [
    r"/G$",                 # ends with per-game already
    r"AVG$",                # averages (RECAVG, RAVG, PAVG, etc.)
    r"AVERAGE$",            # any explicit AVERAGE
    r"PCT$",                # FG_PCT, XP_PCT, CMP_PCT, etc.
    r"%$",                  # columns literally ending with %
    r"RATE$",               # passer rating etc.
    r"QB\s*RAT(ING)?$",     # QB rating variants
    r"Y/R$|Y/T$|Y/A$|Y/ATT$",  # yards per X
    r"ATT/G$|YDS/G$|TD/G$", # common per-game composites
]

# Columns that are clearly PERCENT/PROPORTION and need 0-1 → 0-100 scaling
PERCENT_HINT_PATTERNS = [
    r"%$", r"PCT$", r"TB_%$", r"IN20_%$", r"P-TB_%$", r"TB_PCT$", r"XP_PCT$", r"FG_PCT$"
]

# Strings that should be treated as zeros when cleaning numerics
ZERO_LIKE = {"", " ", "-", "—", "NA", "N/A", "na", "n/a"}

# Helper: is this column name matching any of the given regex patterns?
def col_matches(name: str, patterns) -> bool:
    for pat in patterns:
        if re.search(pat, name, flags=re.IGNORECASE):
            return True
    return False

# Helper: coerce a series to numeric safely, mapping zero-like to 0
def to_numeric_clean(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":  # already numeric-ish
        return pd.to_numeric(s, errors="coerce").fillna(0)
    # map zero-like strings to '0'
    s2 = s.astype(str).str.strip().replace({z: "0" for z in ZERO_LIKE})
    return pd.to_numeric(s2, errors="coerce").fillna(0)

# Decide if a column is "percentage-like"
def is_percent_col(name: str) -> bool:
    return col_matches(name, PERCENT_HINT_PATTERNS)

# Decide if a column is already a rate/avg (skip dividing by GP)
def is_skip_divide_col(name: str) -> bool:
    if name.upper() in ID_COLS:
        return True
    return col_matches(name, SKIP_DIVIDE_PATTERNS)

# Build output filename with _per_game
def out_name_from(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def process_file(fp: str):
    df = pd.read_csv(fp)

    # Normalize column names: strip spaces
    df.columns = [c.strip() for c in df.columns]

    # Ensure GP column exists and is numeric
    if "GP" not in df.columns:
        raise ValueError(f"{fp}: Missing required 'GP' column.")
    df["GP"] = to_numeric_clean(df["GP"]).replace(0, pd.NA)

    # Clean all numeric-like columns first
    for col in df.columns:
        if col.upper() in ID_COLS:
            continue
        # Keep strings for clearly categorical columns only (PLAYER/TEAM handled above)
        df[col] = to_numeric_clean(df[col])

    # Divide by GP for appropriate columns
    for col in df.columns:
        if col in ("GP",) or col.upper() in ID_COLS:
            continue
        if is_skip_divide_col(col):
            # Do not divide per-game / average / percent columns again
            continue
        # divide totals by GP (row-wise), preserving 0 if GP is NA/0
        df[col] = (df[col] / df["GP"]).fillna(0)

    # Percent scaling fix: if values look like proportions (<=1), scale to percent
    for col in df.columns:
        if is_percent_col(col):
            ser = df[col]
            # If many values are in [0,1], interpret as proportions → scale by 100
            # Use a heuristic: at least half of non-zero values <= 1
            nonzero = ser[ser > 0]
            if len(nonzero) > 0 and (nonzero.le(1).sum() / len(nonzero)) >= 0.5:
                df[col] = ser * 100

    # Replace any remaining NaNs with 0 for numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    # Save
    out_fp = out_name_from(fp)
    df.to_csv(out_fp, index=False)
    print(f"✓ Wrote {out_fp}")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")
    for fp in files:
        process_file(fp)

if __name__ == "__main__":
    main()

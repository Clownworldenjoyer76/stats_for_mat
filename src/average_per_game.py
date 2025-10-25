#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed/stats"
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
    """Check if column name matches any given regex pattern."""
    for pat in patterns:
        if re.search(pat, name, flags=re.IGNORECASE):
            return True
    return False

def is_skip_divide_col(name: str) -> bool:
    """Return True if column should NOT be divided by GP."""
    if name.upper() in ID_COLS:
        return True
    return col_matches(name, SKIP_DIVIDE_PATTERNS)

def out_name_from(path: str) -> str:
    """Generate output file path with _per_game suffix."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def process_file(fp: str):
    """Process a single stat file into per-game averages."""
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
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] / df["GP"]

    # --- Specific fix for scoring_per_game.csv ---
    if os.path.basename(fp).lower().startswith("scoring"):
        # Fill empty or whitespace-only cells with 0
        df = df.replace(r"^\s*$", 0, regex=True)
        # Coerce any non-numeric cells in numeric columns to 0
        for col in df.columns:
            if col.upper() not in ID_COLS and col != "GP":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Round numeric values to 2 decimals for readability
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)

    # Save output
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

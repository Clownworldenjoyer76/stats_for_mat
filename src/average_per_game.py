#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import numpy as np

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns that should never be divided
ID_COLS = {
    "PLAYER","TEAM","OPPONENT","POS","POSITION","IS_HOME","HOME","AWAY",
    "DOME","STADIUM","WEEK","SEASON","YEAR","DATE","TEMP_F","WIND_MPH"
}

# Heuristics to detect columns that are already per-game or averages
AVG_LIKE_PAT = re.compile(
    r"(?:/G|_PER_GAME|AVG$|AVERAGE$|RATE$|RATING$|Y/ATT$|Y/A$|Y/R$|Y/T$|"
    r"ATT/G$|YDS/G$|TD/G$|NET_AVG$|PUNT_AVG$|KICK_AVG$|RECAVG$|RAVG$|PAVG$)",
    re.IGNORECASE
)

# Percent-like columns (leave as-is; do NOT divide)
PCT_LIKE_PAT = re.compile(r"(?:%$|_PCT$|PCT$)", re.IGNORECASE)

# Treat these strings as zeros when converting to numeric
ZERO_LIKE = {"", " ", "-", "—", "NA", "N/A", "na", "n/a"}

def is_avg_like(col: str) -> bool:
    return bool(AVG_LIKE_PAT.search(col))

def is_pct_like(col: str) -> bool:
    return bool(PCT_LIKE_PAT.search(col))

def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Coerce to numeric; map zero-like strings to 0 first."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0)
    s = series.astype(str).str.strip()
    s = s.replace({z: "0" for z in ZERO_LIKE})
    return pd.to_numeric(s, errors="coerce").fillna(0)

def out_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def process_csv(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "GP" not in df.columns:
        raise ValueError(f"{path}: Missing required 'GP' column.")

    # Clean GP and guard against division by 0
    df["GP"] = to_numeric_clean(df["GP"]).astype(float)
    safe_gp = df["GP"].replace(0, np.nan)

    # Work column-by-column
    out = df.copy()
    for col in df.columns:
        col_up = col.upper()
        if col_up in ID_COLS or col == "GP":
            continue

        # Try to make it numeric; if it isn't numeric after cleaning, leave as-is
        cleaned = to_numeric_clean(df[col])
        if not pd.api.types.is_numeric_dtype(cleaned):
            out[col] = df[col]
            continue

        # Skip dividing for averages/rates and percentages
        if is_avg_like(col) or is_pct_like(col):
            out[col] = cleaned.fillna(0)
            continue

        # Divide totals by GP (per-game); rows with GP==0 become 0
        out[col] = (cleaned / safe_gp).fillna(0)

    # Ensure numeric NaNs are 0
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)

    out_path = out_name(path)
    out.to_csv(out_path, index=False)
    print(f"✓ Wrote {out_path}")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")
    for fp in files:
        process_csv(fp)

if __name__ == "__main__":
    main()

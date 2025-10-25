#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed/stats"   # keep outputs here as requested
os.makedirs(OUT_DIR, exist_ok=True)

# Columns to never divide by GP (identifiers / meta)
ID_COLS = {
    "PLAYER","TEAM","OPPONENT","POS","POSITION","IS_HOME","HOME","AWAY",
    "DOME","STADIUM","WEEK","SEASON","YEAR","DATE","TEMP_F","WIND_MPH"
}

# Columns to ALWAYS skip dividing (explicit requests)
SKIP_ALWAYS = {
    "GS",           # binary starter flag (don’t average)
    "KICK_PTS",     # scoring totals (don’t per-game)
    "TWO_PT",       # scoring totals (don’t per-game)
}

# Already per-game / rates / averages / percentages (skip dividing)
SKIP_DIVIDE_PATTERNS = [
    r"/G$",                     # ends with per-game already
    r"AVG$",                    # averages (RECAVG, RAVG, PAVG, etc.)
    r"AVERAGE$",                # any explicit AVERAGE
    r"PCT$",                    # FG_PCT, XP_PCT, CMP_PCT, etc.
    r"%$",                      # columns literally ending with %
    r"RATE$",                   # passer rating etc.
    r"QB\s*RAT(ING)?$",         # QB rating variants
    r"Y/R$|Y/T$|Y/A$|Y/ATT$",   # yards per X
    r"ATT/G$|YDS/G$|TD/G$",     # common per-game composites
    r"LNG$",                    # any "long" metric (e.g., LNG, FG_LNG)
]

def col_matches(name: str, patterns) -> bool:
    for pat in patterns:
        if re.search(pat, name, flags=re.IGNORECASE):
            return True
    return False

def is_skip_divide_col(name: str) -> bool:
    u = name.upper()
    if u in ID_COLS or u in SKIP_ALWAYS:
        return True
    return col_matches(u, SKIP_DIVIDE_PATTERNS)

def out_name_from(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def postprocess_by_file(stem: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply file-specific fixes you listed.
    - defensive_per_game.csv: round to 2 dp.
    - kicking_per_game.csv: skip 'LNG' already handled by pattern; just round.
    - kickoffs_punts_per_game.csv: ensure rate/avg columns skipped (pattern handles) and
      standardize '–' placeholders to 0 in numeric columns; round.
    - passing_per_game.csv: skip per-game/rate cols & GS (handled); round.
    - receiving_per_game.csv: skip RECAVG, YDS/G (handled); leave GS; round.
    - returning_per_game.csv: don't divide *_AVG (handled); fill blanks with 0; round.
    - rushing_per_game.csv: skip ATT/G, RAVG, RYDS/G; don't divide GS (handled); round.
    - scoring_per_game.csv: don't divide PTS_G (handled by /G pattern); fill blanks with 0;
      skip KICK_PTS & TWO_PT (handled); round.
    """
    lower = stem.lower()

    # Standardize placeholders and blanks for specific files
    if "kickoffs_punts" in lower:
        # Replace dash-like placeholders with 0 in numeric columns only
        dash_like = {"-", "–", "—"}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Already numeric; nothing to do
                continue
            # For object columns, map dash-like to 0 then coerce if they are numeric-like
            s = df[col].astype(str).str.strip()
            if s.isin(dash_like | {""}).any():
                tmp = s.replace({d: "0" for d in dash_like})
                # Try to coerce to numeric; if it works for many values, keep it numeric
                coerced = pd.to_numeric(tmp, errors="coerce")
                # If coercion produced numbers for at least half of non-empty cells, adopt it; else keep as is
                if coerced.notna().sum() >= (tmp.ne("").sum() / 2 if tmp.ne("").sum() else 0):
                    df[col] = coerced.fillna(0)

    if "returning" in lower or "scoring" in lower:
        # Fill NaNs (from blanks) with 0 for numeric columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)

    # Round numeric columns to 2 decimals for readability (requested across files)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)

    return df

def process_file(fp: str):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]

    if "GP" not in df.columns:
        raise ValueError(f"{fp}: Missing required 'GP' column.")

    # Divide totals by GP for numeric columns that are NOT already per-game/averages/percentages
    for col in df.columns:
        if col == "GP" or col.upper() in ID_COLS or col.upper() in SKIP_ALWAYS:
            continue
        if is_skip_divide_col(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] / df["GP"]

    # File-specific postprocessing
    stem = os.path.splitext(os.path.basename(fp))[0]
    df = postprocess_by_file(stem, df)

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

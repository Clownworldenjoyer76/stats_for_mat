#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import subprocess

RAW_DIR = "data/raw/stats"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns that should never be averaged
ID_COLS = {"PLAYER", "TEAM", "OPPONENT", "POS", "POSITION", "IS_HOME", "DOME", "STADIUM", "WEEK", "SEASON", "YEAR"}

# Patterns for columns that are already per-game or percentage-based
SKIP_PATTERNS = [
    r"/G$", r"AVG$", r"AVERAGE$", r"PCT$", r"%$", r"RATE$", r"RATING$", 
    r"Y/R$", r"Y/T$", r"Y/A$", r"Y/ATT$", r"ATT/G$"
]

def is_skip_col(col: str) -> bool:
    if col.upper() in ID_COLS:
        return True
    for pat in SKIP_PATTERNS:
        if re.search(pat, col, flags=re.IGNORECASE):
            return True
    return False

def out_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return os.path.join(OUT_DIR, f"{stem}_per_game.csv")

def process_file(fp: str):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]

    if "GP" not in df.columns:
        raise ValueError(f"{fp} missing GP column")

    df["GP"] = pd.to_numeric(df["GP"], errors="coerce").replace(0, pd.NA)

    for col in df.columns:
        if col == "GP" or is_skip_col(col):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = (df[col] / df["GP"]).fillna(0)

    out_fp = out_name(fp)
    df.to_csv(out_fp, index=False)
    print(f"✓ Wrote {out_fp}")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {RAW_DIR}")
    for fp in files:
        process_file(fp)

    # Git add + commit + push
    try:
        subprocess.run(["git", "add", "data/processed"], check=True)
        subprocess.run(["git", "commit", "-m", "Update per-game processed stats"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✓ Changes committed and pushed to repo.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Git operation failed: {e}")

if __name__ == "__main__":
    main()

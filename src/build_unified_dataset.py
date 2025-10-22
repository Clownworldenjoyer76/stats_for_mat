#!/usr/bin/env python3
"""
Build a unified team-level dataset from data/raw/team_stats ONLY.
- No week/season detection (files are already up to date).
- Ignores data/raw/stats (player stats).
- Merges on detected team column; prefixes columns by source file stem.
- Fills numeric NaNs with 0 to avoid blanks.

Output: data/processed/nfl_unified_with_metrics.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[1]
TEAM_RAW = ROOT / "data" / "raw" / "team_stats"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "nfl_unified_with_metrics.csv"

TEAM_COL_PAT = re.compile(r"\bteam\b", re.IGNORECASE)

def detect_team_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if TEAM_COL_PAT.search(str(c)):
            return c
    return None

def numericize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "team":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def load_and_prepare(file_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(file_path)
    team_col = detect_team_col(df)
    if team_col is None:
        print(f"[SKIP] {file_path.name} (no 'team' column detected)")
        return None

    df = df.rename(columns={team_col: "team"})
    # Keep one row per team; if multiple, take the first (files are season-to-date)
    df = df.dropna(subset=["team"]).copy()
    df["team"] = df["team"].astype(str).str.strip()
    # If duplicates exist, keep the first per team
    df = df.drop_duplicates(subset=["team"], keep="first").reset_index(drop=True)

    # Prefix non-team columns by file stem
    prefix = file_path.stem.lower()
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "team"}
    df = df.rename(columns=rename_map)

    # Coerce numerics (after rename)
    df = numericize(df)
    return df

def main():
    csvs = sorted(TEAM_RAW.glob("*.csv"))
    if not csvs:
        print(f"[INFO] No CSVs found in {TEAM_RAW}")
        OUT_PATH.write_text("")  # produce an empty file to signal run
        return

    merged = None
    for fp in csvs:
        try:
            part = load_and_prepare(fp)
            if part is None or part.empty:
                continue
            if merged is None:
                merged = part
            else:
                merged = merged.merge(part, on="team", how="outer")
            print(f"[OK] merged {fp.name}")
        except Exception as e:
            print(f"[WARN] {fp.name}: {e}")

    if merged is None or merged.empty:
        print("[INFO] Nothing to write; no valid team tables.")
        OUT_PATH.write_text("")
        return

    # Fill numeric NaNs with 0; leave text columns (just 'team') as-is
    num_cols = [c for c in merged.columns if c != "team"]
    merged[num_cols] = merged[num_cols].fillna(0)

    merged.to_csv(OUT_PATH, index=False)
    print(f"[DONE] {len(merged)} teams → {OUT_PATH}")

if __name__ == "__main__":
    main()

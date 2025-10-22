#!/usr/bin/env python3
# Builds a unified team-level CSV from data/raw/team_stats ONLY.
# - Fixes specific team-name variants (incl. "Pittsb")
# - Merges on canonical team names
# - Fills numeric NaNs with 0
# - Drops rows that are all-zero across numeric columns

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEAM_RAW = ROOT / "data" / "raw" / "team_stats"
OUT_PATH = ROOT / "data" / "processed" / "nfl_unified_with_metrics.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Explicit fixes/aliases (expand as needed)
TEAM_FIXES = {
    "Pittsb": "Pittsburgh Steelers",
    "LA Rams": "Los Angeles Rams",
    "Oakland": "Las Vegas Raiders",
    "SD Chargers": "Los Angeles Chargers",
    "San Diego Chargers": "Los Angeles Chargers",
    "NY Jets": "New York Jets",
    "Washington": "Washington Commanders",
    "Jax Jaguars": "Jacksonville Jaguars",
    "Tampa Bay Bucs": "Tampa Bay Buccaneers",
    "NE Patriots": "New England Patriots",
}

def clean_team_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    n = name.strip()
    return TEAM_FIXES.get(n, " ".join(w.capitalize() for w in n.split()))

def detect_team_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "team" in str(c).lower():
            return c
    return None

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "team":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def load_and_prepare(file_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(
        file_path,
        dtype=str,                # read everything as string first
        engine="python",
        on_bad_lines="skip"       # skip malformed lines that can truncate names
    )
    team_col = detect_team_col(df)
    if not team_col:
        print(f"[SKIP] {file_path.name}: no team column")
        return None

    df = df.rename(columns={team_col: "team"})
    df["team"] = df["team"].map(clean_team_name)

    # group by team to collapse any duplicates within this file; numeric → max
    num_cols = [c for c in df.columns if c != "team"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    grouped = df.groupby("team", dropna=False)[num_cols].max().reset_index()

    # prefix non-team columns with file stem
    prefix = file_path.stem.lower()
    grouped = grouped.rename(columns={c: f"{prefix}_{c}" for c in num_cols})

    return grouped

def drop_all_zero_rows(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c != "team"]
    if not num_cols:
        return df
    mask_all_zero = (df[num_cols].fillna(0) == 0).all(axis=1)
    return df.loc[~mask_all_zero].reset_index(drop=True)

def main():
    files = sorted(TEAM_RAW.glob("*.csv"))
    if not files:
        print(f"[INFO] No files in {TEAM_RAW}")
        OUT_PATH.write_text("")
        return

    merged = None
    for fp in files:
        part = load_and_prepare(fp)
        if part is None or part.empty:
            continue
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on="team", how="outer")
        print(f"[OK] merged {fp.name}")

    if merged is None or merged.empty:
        print("[INFO] Nothing to write.")
        OUT_PATH.write_text("")
        return

    # fill numeric NaNs with 0
    num_cols = [c for c in merged.columns if c != "team"]
    merged[num_cols] = merged[num_cols].fillna(0)

    # drop zero-only rows
    merged = drop_all_zero_rows(merged)

    # sort for stability
    merged = merged.sort_values("team").reset_index(drop=True)

    merged.to_csv(OUT_PATH, index=False)
    print(f"[DONE] {len(merged)} teams → {OUT_PATH}")

if __name__ == "__main__":
    main()

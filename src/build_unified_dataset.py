#!/usr/bin/env python3
# Build a unified team-level CSV from data/raw/team_stats ONLY.
# - Requires every source file to have a 'TEAM' column (exact match).
# - Converts all non-TEAM columns to numeric (no string accessor usage).
# - Prefixes columns by source filename (stem).
# - Uses INNER MERGE on TEAM to avoid NaNs/zero blocks when all files contain the same teams.
# - Writes data/processed/nfl_unified_with_metrics.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEAM_RAW = ROOT / "data" / "raw" / "team_stats"
OUT_PATH = ROOT / "data" / "processed" / "nfl_unified_with_metrics.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def read_team_table(path: Path) -> pd.DataFrame:
    """Read a single team_stats CSV, validate headers, coerce numerics, prefix columns."""
    df = pd.read_csv(path, dtype=object)  # read as objects; we will coerce numerics next

    # Validate TEAM header exists exactly
    if "TEAM" not in df.columns:
        raise ValueError(f"{path.name}: Missing required 'TEAM' column")

    # Standardize TEAM to string and strip
    df["TEAM"] = df["TEAM"].astype(str).str.strip()

    # Convert non-TEAM columns to numeric
    non_team_cols = [c for c in df.columns if c != "TEAM"]
    for c in non_team_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # One row per team (in case of accidental duplicates in a source file)
    df = df.groupby("TEAM", as_index=False)[non_team_cols].max()

    # Prefix stat columns by file stem
    prefix = path.stem.lower()
    df = df.rename(columns={c: f"{prefix}_{c}" for c in non_team_cols})

    return df

def main() -> None:
    files = sorted(TEAM_RAW.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found in {TEAM_RAW}")

    merged: pd.DataFrame | None = None
    for fp in files:
        tbl = read_team_table(fp)
        merged = tbl if merged is None else merged.merge(tbl, on="TEAM", how="inner")
        print(f"[OK] merged {fp.name}")

    # Final sort for stability
    merged = merged.sort_values("TEAM").reset_index(drop=True)

    # Write output
    merged.to_csv(OUT_PATH, index=False)
    print(f"[DONE] {len(merged)} teams → {OUT_PATH}")

if __name__ == "__main__":
    main()

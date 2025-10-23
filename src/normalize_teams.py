#!/usr/bin/env python3
"""
normalize_teams.py

Normalize abbreviated team names to canonical team names using data/team_aliases.csv.

- In data/raw/week_matchups_odds.csv: normalize TEAM_A and TEAM_B
- In all CSVs under data/raw/stats: normalize TEAM

Updates files in place and reports what changed.
"""

import argparse
import sys
import pathlib
import pandas as pd

def load_alias_map(aliases_path: pathlib.Path) -> dict:
    df = pd.read_csv(aliases_path, dtype=str).rename(columns=str.strip)
    required = {"ABBR", "TEAM"}
    if not required.issubset(df.columns):
        raise ValueError(f"{aliases_path} must contain columns: {sorted(required)}")
    # Build case-insensitive map from ABBR -> TEAM
    df["ABBR"] = df["ABBR"].astype(str).str.strip().str.upper()
    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    # Drop duplicates preferring the first occurrence
    df = df.drop_duplicates(subset=["ABBR"], keep="first")
    return dict(zip(df["ABBR"], df["TEAM"]))

def _normalize_series(s: pd.Series, abbr_map: dict, colname: str, file_path: pathlib.Path) -> pd.Series:
    original = s.astype(str)
    cleaned = original.str.strip().str.upper()

    # Map using ABBR; if value not found, leave as original (unmodified string)
    mapped = cleaned.map(abbr_map)
    # Keep original where no mapping found
    out = original.where(mapped.isna(), mapped)

    # Report unknowns (only once per file/column)
    unknown = sorted(set(cleaned[ mapped.isna() & cleaned.ne("") ].dropna().unique()))
    if unknown:
        print(f"[WARN] {file_path.name} :: {colname} :: {len(unknown)} unknown ABBR(s) not found in aliases: {unknown[:20]}{' ...' if len(unknown)>20 else ''}")
    return out

def normalize_week_matchups(week_file: pathlib.Path, abbr_map: dict) -> int:
    if not week_file.exists():
        print(f"[INFO] Skipping (not found): {week_file}")
        return 0
    df = pd.read_csv(week_file, dtype=str).rename(columns=str.strip)
    changed = 0
    for col in ("TEAM_A", "TEAM_B"):
        if col in df.columns:
            before = df[col].copy()
            df[col] = _normalize_series(df[col], abbr_map, col, week_file)
            changed += (before != df[col]).sum()
        else:
            print(f"[WARN] {week_file.name} missing column '{col}'")
    df.to_csv(week_file, index=False)
    print(f"[OK]   Updated {week_file} ({changed} cell(s) changed)")
    return changed

def normalize_stats_folder(stats_dir: pathlib.Path, abbr_map: dict) -> int:
    if not stats_dir.exists():
        print(f"[INFO] Skipping (not found): {stats_dir}")
        return 0
    total_changed = 0
    files = sorted(stats_dir.glob("*.csv"))
    if not files:
        print(f"[INFO] No CSV files in {stats_dir}")
        return 0
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str).rename(columns=str.strip)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue
        if "TEAM" not in df.columns:
            print(f"[INFO] Skipping {f.name} (no TEAM column)")
            continue
        before = df["TEAM"].copy()
        df["TEAM"] = _normalize_series(df["TEAM"], abbr_map, "TEAM", f)
        changed = (before != df["TEAM"]).sum()
        df.to_csv(f, index=False)
        print(f"[OK]   Updated {f} ({changed} cell(s) changed)")
        total_changed += changed
    return total_changed

def main():
    parser = argparse.ArgumentParser(description="Normalize team abbreviations to canonical names.")
    parser.add_argument("--aliases", default="data/team_aliases.csv", help="Path to alias CSV (columns: ABBR, TEAM)")
    parser.add_argument("--week-file", default="data/raw/week_matchups_odds.csv", help="Week matchups CSV to normalize (TEAM_A, TEAM_B)")
    parser.add_argument("--stats-dir", default="data/raw/stats", help="Directory of stats CSVs to normalize (TEAM column)")
    args = parser.parse_args()

    aliases_path = pathlib.Path(args.aliases)
    week_file = pathlib.Path(args.week_file)
    stats_dir = pathlib.Path(args.stats_dir)

    try:
        abbr_map = load_alias_map(aliases_path)
    except Exception as e:
        print(f"[ERROR] Failed to load aliases: {e}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(abbr_map)} team aliases from {aliases_path}")
    total = 0
    total += normalize_week_matchups(week_file, abbr_map)
    total += normalize_stats_folder(stats_dir, abbr_map)
    print(f"[DONE] Total cells updated: {total}")

if __name__ == "__main__":
    main()

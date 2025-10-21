#!/usr/bin/env python3
"""
Build a unified, season-to-date team-level dataset
from data/raw/stats and data/raw/team_stats.

Output: data/processed/nfl_unified_with_metrics.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def infer_numeric_cols(df):
    numeric_cols = []
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > len(s) * 0.5:
                numeric_cols.append(c)
        except Exception:
            pass
    return numeric_cols

def normalize_team_names(df):
    for c in df.columns:
        if re.search(r'team', c, re.I):
            df[c] = df[c].astype(str).str.strip()
    return df

def extract_team_col(df):
    for c in df.columns:
        if re.search(r'team', c, re.I):
            return c
    return None

def classify_file(filename, columns):
    name = filename.lower()
    joined = " ".join(columns).lower()
    if "pass" in name or "pass" in joined:
        return "passing"
    if "rush" in name or "rush" in joined:
        return "rushing"
    if "defense" in name or "def" in joined:
        return "defense"
    if "kick" in name or "punt" in joined:
        return "special"
    if "return" in name:
        return "returning"
    if "turnover" in name:
        return "turnovers"
    if "down" in name:
        return "downs"
    if "yard" in name:
        return "yardage"
    if "score" in name:
        return "scoring"
    return "misc"

def summarize_team(df, category):
    team_col = extract_team_col(df)
    if not team_col:
        return pd.DataFrame()

    df = normalize_team_names(df)
    df = df.rename(columns={team_col: "team"})
    numeric_cols = infer_numeric_cols(df)
    if not numeric_cols:
        return pd.DataFrame()

    agg = df.groupby("team")[numeric_cols].mean().reset_index()
    agg["category"] = category
    return agg

def build_unified():
    frames = []
    for folder in ["stats", "team_stats"]:
        base = RAW / folder
        for file in sorted(base.glob("*.csv")):
            try:
                df = pd.read_csv(file)
                cat = classify_file(file.stem, df.columns)
                summary = summarize_team(df, cat)
                if not summary.empty:
                    frames.append(summary)
                    print(f"[OK] {file.name} → {cat}")
                else:
                    print(f"[SKIP] {file.name} (no numeric data)")
            except Exception as e:
                print(f"[WARN] {file.name}: {e}")

    if not frames:
        print("No data to unify.")
        return

    combined = pd.concat(frames, ignore_index=True)
    wide = combined.pivot_table(index="team", columns="category", aggfunc="mean")
    wide.columns = [f"{c[1]}_{c[0]}" if c[1] else c[0] for c in wide.columns]
    wide = wide.reset_index()

    wide.to_csv(PROCESSED / "nfl_unified_with_metrics.csv", index=False)
    print(f"[DONE] Wrote {len(wide)} teams → {PROCESSED/'nfl_unified_with_metrics.csv'}")

if __name__ == "__main__":
    build_unified()

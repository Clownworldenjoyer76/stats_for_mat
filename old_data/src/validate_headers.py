#!/usr/bin/env python3
"""
Validate the headers of CSV files in data/raw/stats and data/raw/team_stats
against the known header summaries in config/.

Usage:
    python src/validate_headers.py

Exit codes:
    0 = all files valid
    1 = one or more files invalid
"""
from pathlib import Path
import json
import pandas as pd
import sys

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
RAW = REPO / "data" / "raw"

SUMMARY = {
    "stats": CONFIG / "stats_headers_summary.json",
    "team_stats": CONFIG / "team_stats_headers_summary.json",
}

def load_summary(path: Path) -> dict[str, list[str]]:
    with open(path, "r") as f:
        return json.load(f)

def read_headers(csv_path: Path) -> list[str]:
    try:
        df = pd.read_csv(csv_path, nrows=0)
        return list(df.columns)
    except Exception as e:
        return [f"Error reading file: {e}"]

def validate_folder(folder: str, summary: dict[str, list[str]]) -> dict[str, dict]:
    base = RAW / folder
    results = {}
    for csv_path in sorted(base.glob("*.csv")):
        name = csv_path.stem
        actual = read_headers(csv_path)
        expected = summary.get(name)
        if expected is None:
            results[name] = {"status": "missing_in_summary", "headers": actual}
            continue
        missing = [h for h in expected if h not in actual]
        extra = [h for h in actual if h not in expected]
        status = "ok" if not missing and not extra else "mismatch"
        results[name] = {"status": status, "missing": missing, "extra": extra}
    return results

def main() -> int:
    overall_ok = True
    for folder, summary_path in SUMMARY.items():
        if not summary_path.exists():
            print(f"[WARN] Missing summary: {summary_path}")
            continue
        summary = load_summary(summary_path)
        results = validate_folder(folder, summary)
        print(f"\n=== {folder.upper()} ===")
        for name, info in results.items():
            if info["status"] == "ok":
                print(f"[OK] {name}")
            else:
                overall_ok = False
                print(f"[{info['status'].upper()}] {name}")
                if "missing" in info and info["missing"]:
                    print(f"  Missing: {info['missing']}")
                if "extra" in info and info["extra"]:
                    print(f"  Extra: {info['extra']}")
    if not overall_ok:
        print("\n[FAIL] Header validation detected mismatches.")
        return 1
    print("\n[PASS] All headers validated successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

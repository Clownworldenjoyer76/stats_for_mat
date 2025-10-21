#!/usr/bin/env python3
"""
Convert the data-dump files in ./stats/ and ./team_stats/ into CSVs.

- Reads every file in:
    ./stats/
    ./team_stats/
- Attempts to parse as:
    1) JSON (array or object) → normalized table
    2) Delimited text (auto-detected delimiter via csv.Sniffer)
    3) Fallback: single 'raw' column with original lines

- Writes CSVs to:
    ./data/raw/stats/<original_name>.csv
    ./data/raw/team_stats/<original_name>.csv

This script is intentionally defensive and will not crash the workflow on a single
bad file; it logs the issue and moves on.
"""

import os
import io
import csv
import json
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRS = {
    "stats": REPO_ROOT / "stats",
    "team_stats": REPO_ROOT / "team_stats",
}
OUT_ROOT = REPO_ROOT / "data" / "raw"
OUT_DIRS = {
    "stats": OUT_ROOT / "stats",
    "team_stats": OUT_ROOT / "team_stats",
}


def ensure_dirs() -> None:
    for p in OUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    # Try utf-8, then latin-1 as fallback
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def try_parse_json(text: str) -> Optional[pd.DataFrame]:
    t = text.strip()
    if not t:
        return pd.DataFrame()
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None

    # If it's a list of dicts → DataFrame directly; otherwise normalize
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return pd.DataFrame(obj)
        # list of primitives: store as single column
        return pd.DataFrame({"value": obj})
    if isinstance(obj, dict):
        # flatten one level
        return pd.json_normalize(obj, sep=".")
    # Fallback
    return pd.DataFrame({"value": [obj]})


def sniff_delimiter(sample: str) -> Optional[str]:
    # Use csv.Sniffer; try common fallbacks if it fails
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|", " "])
        return dialect.delimiter
    except csv.Error:
        # Heuristic: pick the delimiter that appears in most lines
        candidates = [",", "\t", ";", "|"]
        counts = {d: sample.count(d) for d in candidates}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else None


def to_dataframe_from_delimited(text: str) -> pd.DataFrame:
    # Use the first 32k to sniff
    head = text[:32768]
    delim = sniff_delimiter(head)
    if delim is None:
        # Treat as lines → single column
        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
        return pd.DataFrame({"raw": lines})

    # Detect header
    sniffer = csv.Sniffer()
    has_header = False
    try:
        has_header = sniffer.has_header(head)
    except csv.Error:
        has_header = False

    # Read with pandas
    buf = io.StringIO(text)
    df = pd.read_csv(
        buf,
        sep=delim,
        engine="python",
        header=0 if has_header else None,
        dtype=str  # keep original forms; convert later downstream if needed
    )
    if not has_header:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    return df


def convert_folder(src_dir: Path, out_dir: Path) -> List[Path]:
    written: List[Path] = []
    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir():
            # Skip nested dirs to keep it simple/explicit
            continue
        try:
            text = read_text(entry)
            # Try JSON
            df = try_parse_json(text)
            if df is None:
                # Try delimited parsing
                df = to_dataframe_from_delimited(text)

            # Empty → write a minimal CSV so the workflow still commits a file
            if df is None or df.empty:
                df = pd.DataFrame({"raw": []})

            out_path = out_dir / (entry.name + ("" if entry.suffix.lower() == ".csv" else ".csv"))
            # Normalize line endings and index
            df.to_csv(out_path, index=False)
            print(f"[OK] {entry.relative_to(REPO_ROOT)} → {out_path.relative_to(REPO_ROOT)}")
            written.append(out_path)
        except Exception as e:
            print(f"[WARN] Failed to convert {entry.name}: {e}", file=sys.stderr)
            continue
    return written


def main() -> int:
    ensure_dirs()
    total_written = []

    for key, src in SRC_DIRS.items():
        out = OUT_DIRS[key]
        if not src.exists():
            print(f"[INFO] Source folder missing, skipping: {src.relative_to(REPO_ROOT)}")
            continue
        written = convert_folder(src, out)
        total_written.extend(written)

    if not total_written:
        print("[INFO] No files converted.")
    else:
        print(f"[DONE] Wrote {len(total_written)} CSV file(s) under {OUT_ROOT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

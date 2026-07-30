#!/usr/bin/env python3
"""
Convert all files in ./stats/ into CSVs and save them under ./data/raw/stats/

Features:
- Reads any file type (text, JSON, delimited, etc.) from ./stats/
- Automatically detects JSON arrays/objects or delimited text
- Converts all to clean CSV files (UTF-8, quoted, no index)
- Creates target directories if missing
"""

import os
import io
import csv
import json
from pathlib import Path
from typing import Optional
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "stats"
OUT_DIR = ROOT / "data" / "raw" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_text(path: Path) -> str:
    """Read a text file safely with UTF-8 fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def try_parse_json(text: str) -> Optional[pd.DataFrame]:
    """Try to parse JSON array or object into a DataFrame."""
    t = text.strip()
    if not t:
        return pd.DataFrame()
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return pd.DataFrame(obj)
        return pd.DataFrame({"value": obj})
    if isinstance(obj, dict):
        return pd.json_normalize(obj, sep=".")
    return pd.DataFrame({"value": [obj]})

def sniff_delimiter(sample: str) -> Optional[str]:
    """Detect delimiter in text content."""
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|", " "])
        return dialect.delimiter
    except csv.Error:
        # fallback: choose most common
        candidates = [",", "\t", ";", "|"]
        counts = {d: sample.count(d) for d in candidates}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else None

def to_dataframe_from_delimited(text: str) -> pd.DataFrame:
    """Convert delimited text to DataFrame."""
    head = text[:32768]
    delim = sniff_delimiter(head)
    if delim is None:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return pd.DataFrame({"raw": lines})

    sniffer = csv.Sniffer()
    try:
        has_header = sniffer.has_header(head)
    except csv.Error:
        has_header = False

    buf = io.StringIO(text)
    df = pd.read_csv(buf, sep=delim, engine="python",
                     header=0 if has_header else None, dtype=str)
    if not has_header:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    return df

# ---------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------
def convert_folder(src_dir: Path, out_dir: Path):
    """Convert every file in src_dir into CSV in out_dir."""
    written = []
    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir():
            continue
        try:
            text = read_text(entry)
            df = try_parse_json(text)
            if df is None:
                df = to_dataframe_from_delimited(text)
            if df is None or df.empty:
                df = pd.DataFrame({"raw": []})

            out_path = out_dir / (entry.stem + ".csv")
            df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', lineterminator="\n")
            print(f"[OK] {entry.name} → {out_path.relative_to(ROOT)}")
            written.append(out_path)
        except Exception as e:
            print(f"[WARN] Failed to convert {entry.name}: {e}")
    print(f"[DONE] {len(written)} CSVs written to {out_dir.relative_to(ROOT)}")

def main():
    convert_folder(SRC_DIR, OUT_DIR)

if __name__ == "__main__":
    main()

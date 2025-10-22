#!/usr/bin/env python3
"""
Convert the data-dump files in ./stats/ and ./team_stats/ into clean CSVs.

Fixes & features:
- Normalize Unicode (prevent truncations like "Pittsb")
- Auto-detect delimiter, robust quoting on output
- Canonicalize TEAM column name and apply alias map
- Clean numeric values safely (no fake zeros)
- Validate missing/duplicate team rows
"""

import os, io, csv, json, sys, re, unicodedata, yaml
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRS = {"stats": REPO_ROOT / "stats", "team_stats": REPO_ROOT / "team_stats"}
OUT_ROOT = REPO_ROOT / "data" / "raw"
OUT_DIRS = {"stats": OUT_ROOT / "stats", "team_stats": OUT_ROOT / "team_stats"}
ALIASES_PATH = REPO_ROOT / "config" / "team_aliases.yaml"

# ---------------------------------------------------------------------------
# Canonical team set (strict 32)
# ---------------------------------------------------------------------------
CANON_TEAMS = {
    "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
    "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos",
    "Detroit Lions","Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars",
    "Kansas City Chiefs","Las Vegas Raiders","Los Angeles Chargers","Los Angeles Rams","Miami Dolphins",
    "Minnesota Vikings","New England Patriots","New Orleans Saints","New York Giants","New York Jets",
    "Philadelphia Eagles","Pittsburgh Steelers","San Francisco 49ers","Seattle Seahawks",
    "Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    for p in OUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)
    return s.replace("\u2014","-").replace("\u2013","-").replace("\xa0"," ").strip()

def sniff_delimiter(sample: str) -> Optional[str]:
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[",","\t",";","|"," "])
        return dialect.delimiter
    except csv.Error:
        candidates = [",","\t",";","|"]
        counts = {d: sample.count(d) for d in candidates}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else None

def to_dataframe_from_delimited(text: str) -> pd.DataFrame:
    head = text[:32768]
    delim = sniff_delimiter(head)
    if delim is None:
        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
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

# ---------------------------------------------------------------------------
# Team name canonicalization
# ---------------------------------------------------------------------------
def _norm_key(s: str) -> str:
    s = (s or "").lower().replace("&","and").replace("."," ")
    return re.sub(r"\s+"," ",s.strip())

def load_aliases() -> dict:
    if ALIASES_PATH.exists():
        m = yaml.safe_load(ALIASES_PATH.read_text()) or {}
        return {_norm_key(k): v for k,v in m.items()}
    return {}

ALIASES = load_aliases()

CITY_FALLBACK = {
    "oakland":"Las Vegas Raiders","st louis":"Los Angeles Rams",
    "ny jets":"New York Jets","n y jets":"New York Jets",
    "ny giants":"New York Giants","n y giants":"New York Giants",
    "la rams":"Los Angeles Rams","sd chargers":"Los Angeles Chargers",
    "san diego chargers":"Los Angeles Chargers",
    "washington":"Washington Commanders","pittsb":"Pittsburgh Steelers"
}

def canon_team(v: str) -> str:
    base = _norm_key(v)
    if base in ALIASES:
        return ALIASES[base]
    if base in CITY_FALLBACK:
        return CITY_FALLBACK[base]
    # fallback to title-case
    guess = " ".join(w.capitalize() for w in base.split())
    return guess if guess in CANON_TEAMS else v.strip()

def coerce_team_col(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if any(t in str(c).lower() for t in ("team","club","franchise","name")):
            df = df.rename(columns={c:"TEAM"})
            break
    return df

# ---------------------------------------------------------------------------
# Numeric cleaning
# ---------------------------------------------------------------------------
def clean_num(s):
    if s is None: return s
    s = str(s).strip()
    if s in {"", "-", "—", "–", "N/A","n/a","NA"}:
        return np.nan
    if "/" in s:
        a,b = s.split("/",1)
        try:
            return float(a)/float(b) if float(b)!=0 else np.nan
        except:
            return np.nan
    s = s.replace(",","").replace("%","")
    try:
        return float(s)
    except:
        return np.nan

# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------
def convert_folder(src_dir: Path, out_dir: Path):
    written = []
    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir(): continue
        try:
            text = read_text(entry)
            df = None
            t = text.strip()
            if t.startswith("{") or t.startswith("["):
                try:
                    obj = json.loads(t)
                    if isinstance(obj, list):
                        df = pd.DataFrame(obj)
                    elif isinstance(obj, dict):
                        df = pd.json_normalize(obj, sep=".")
                except Exception:
                    pass
            if df is None:
                df = to_dataframe_from_delimited(text)

            df = df.applymap(normalize_text)
            df = coerce_team_col(df)
            if "TEAM" in df.columns:
                df["TEAM"] = df["TEAM"].map(canon_team)

            # Clean numerics
            for c in df.columns:
                if c != "TEAM":
                    df[c] = df[c].apply(clean_num)

            # Validation
            if "TEAM" in df.columns:
                missing = int(df["TEAM"].isna().sum())
                dups = int(df["TEAM"].duplicated().sum())
                if missing or dups:
                    print(f"[VALIDATION] {entry.name}: missing TEAM={missing}, duplicate TEAM={dups}")

            out_path = out_dir / (entry.name + ("" if entry.suffix.lower() == ".csv" else ".csv"))
            df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', lineterminator='\n')
            print(f"[OK] {entry.relative_to(REPO_ROOT)} → {out_path.relative_to(REPO_ROOT)}")
            written.append(out_path)
        except Exception as e:
            print(f"[WARN] Failed to convert {entry.name}: {e}", file=sys.stderr)
    return written

def main() -> int:
    ensure_dirs()
    total_written = []
    for key, src in SRC_DIRS.items():
        out = OUT_DIRS[key]
        if not src.exists():
            print(f"[INFO] Missing source: {src}")
            continue
        written = convert_folder(src, out)
        total_written.extend(written)
    print(f"[DONE] {len(total_written)} CSVs written to {OUT_ROOT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

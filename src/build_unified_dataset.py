#!/usr/bin/env python3
# Build a unified team-level CSV from data/raw/team_stats ONLY.
# - Robust team-column detection (Team/Club/Franchise/Name)
# - Canonical team-name normalization (with aliases incl. "Pittsb")
# - Cleans numeric strings (commas, %, dashes) before conversion
# - Safe merge on canonical team names
# - Fills numeric NaNs with 0, then drops rows that are all-zero

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEAM_RAW = ROOT / "data" / "raw" / "team_stats"
OUT_PATH = ROOT / "data" / "processed" / "nfl_unified_with_metrics.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- TEAM NAME NORMALIZATION ---

ALIASES = {
    # broken / truncated
    "pittsb": "Pittsburgh Steelers",
    # short/legacy variants
    "la rams": "Los Angeles Rams",
    "st louis rams": "Los Angeles Rams",
    "oakland": "Las Vegas Raiders",
    "oakland raiders": "Las Vegas Raiders",
    "sd chargers": "Los Angeles Chargers",
    "san diego chargers": "Los Angeles Chargers",
    "ny jets": "New York Jets",
    "n.y. jets": "New York Jets",
    "ny giants": "New York Giants",
    "n.y. giants": "New York Giants",
    "washington": "Washington Commanders",
    "jax jaguars": "Jacksonville Jaguars",
    "tampa bay bucs": "Tampa Bay Buccaneers",
    "ne patriots": "New England Patriots",
}

CITY_CANON = {
    "buffalo":"Buffalo Bills","miami":"Miami Dolphins","new england":"New England Patriots",
    "new york jets":"New York Jets","new york giants":"New York Giants","dallas":"Dallas Cowboys",
    "philadelphia":"Philadelphia Eagles","washington commanders":"Washington Commanders",
    "chicago":"Chicago Bears","detroit":"Detroit Lions","green bay":"Green Bay Packers",
    "minnesota":"Minnesota Vikings","atlanta":"Atlanta Falcons","carolina":"Carolina Panthers",
    "new orleans":"New Orleans Saints","tampa bay":"Tampa Bay Buccaneers","arizona":"Arizona Cardinals",
    "los angeles rams":"Los Angeles Rams","seattle":"Seattle Seahawks","san francisco":"San Francisco 49ers",
    "kansas city":"Kansas City Chiefs","las vegas":"Las Vegas Raiders","los angeles chargers":"Los Angeles Chargers",
    "denver":"Denver Broncos","baltimore":"Baltimore Ravens","cincinnati":"Cincinnati Bengals",
    "cleveland":"Cleveland Browns","pittsburgh":"Pittsburgh Steelers","houston":"Houston Texans",
    "indianapolis":"Indianapolis Colts","jacksonville":"Jacksonville Jaguars","tennessee":"Tennessee Titans",
    "minneapolis":"Minnesota Vikings","phoenix":"Arizona Cardinals","st louis":"Los Angeles Rams",  # safety
}

def _norm_str(s: str) -> str:
    s = ("" if not isinstance(s, str) else s).strip().lower()
    s = s.replace("&", "and").replace(".", "")
    s = " ".join(s.split())
    return s

def clean_team_name(name: str) -> str:
    base = _norm_str(name)
    if base in ALIASES:
        return ALIASES[base]
    if base in CITY_CANON:
        return CITY_CANON[base]
    # title-case fallback
    return " ".join(w.capitalize() for w in base.split())

# --- CSV HANDLING ---

def detect_team_col(df: pd.DataFrame) -> str | None:
    names = {c: str(c).strip().lower() for c in df.columns}
    for target in ("team", "club", "franchise", "name"):
        for c, lc in names.items():
            if target == lc or target in lc:
                return c
    return None

def clean_numeric_series(s: pd.Series) -> pd.Series:
    # Strip commas, percent signs, dashes, blanks → NaN
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "-": np.nan, "—": np.nan, "–": np.nan, "N/A": np.nan, "na": np.nan, "None": np.nan}, regex=True)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def load_and_prepare(file_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(
        file_path,
        dtype=str,                # read as strings then clean
        engine="python",
        on_bad_lines="skip"
    )
    team_col = detect_team_col(df)
    if not team_col:
        print(f"[SKIP] {file_path.name}: no team column")
        return None

    df = df.rename(columns={team_col: "team"})
    df["team"] = df["team"].map(clean_team_name)

    # Clean numeric cols
    num_cols = [c for c in df.columns if c != "team"]
    for c in num_cols:
        df[c] = clean_numeric_series(df[c])

    # Collapse duplicates in this file by canonical team (take max across rows)
    grouped = df.groupby("team", dropna=False)[num_cols].max().reset_index()

    # Prefix non-team columns by file stem
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
        merged = part if merged is None else merged.merge(part, on="team", how="outer")
        print(f"[OK] merged {fp.name}")

    if merged is None or merged.empty:
        print("[INFO] Nothing to write.")
        OUT_PATH.write_text("")
        return

    # Fill numeric NaNs with 0 and drop zero-only rows
    num_cols = [c for c in merged.columns if c != "team"]
    merged[num_cols] = merged[num_cols].fillna(0)
    merged = drop_all_zero_rows(merged)
    merged = merged.sort_values("team").reset_index(drop=True)

    merged.to_csv(OUT_PATH, index=False)
    print(f"[DONE] {len(merged)} teams → {OUT_PATH}")

if __name__ == "__main__":
    main()

# src/normalize_teams.py
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

YEAR = 2025  # <-- update if needed

ALIASES_PATH = Path("data/team_aliases.csv")

FILES_TO_NORMALIZE = [
    Path("data/raw/stats/defensive.csv"),
    Path("data/raw/stats/kicking.csv"),
    Path("data/raw/stats/kickoffs_punts.csv"),
    Path("data/raw/stats/passing.csv"),
    Path("data/raw/stats/receiving.csv"),
    Path("data/raw/stats/returning.csv"),
    Path("data/raw/stats/rushing.csv"),
    Path("data/raw/stats/scoring.csv"),
    Path("data/raw/week_matchups_odds.csv"),
]

def load_alias_map(aliases_csv: Path) -> dict:
    """
    Expects columns: ABBR, TEAM
    Returns dict mapping ABBR -> TEAM
    """
    df = pd.read_csv(aliases_csv, dtype=str).fillna("")
    if not {"ABBR", "TEAM"}.issubset(df.columns):
        raise ValueError("team_aliases.csv must contain columns: ABBR, TEAM")
    return dict(zip(df["ABBR"].str.strip(), df["TEAM"].str.strip()))

def normalize_series_with_aliases(series: pd.Series, alias_map: dict) -> pd.Series:
    """
    Replace values that match an ABBR key with the canonical TEAM value.
    If a value doesn't match any ABBR, it is left as-is.
    """
    return series.astype(str).apply(lambda v: alias_map.get(v.strip(), v.strip()))

def normalize_stats_file(path: Path, alias_map: dict) -> None:
    """
    For stats CSVs: normalize TEAM column via ABBR->TEAM.
    """
    df = pd.read_csv(path)
    if "TEAM" not in df.columns:
        # Silent skip if the file doesn't have TEAM column (keeps script simple)
        return
    df["TEAM"] = normalize_series_with_aliases(df["TEAM"], alias_map)
    df.to_csv(path, index=False)

def normalize_matchups(path: Path, alias_map: dict) -> None:
    """
    Normalize TEAM_A, TEAM_B via ABBR->TEAM and add weatherapi_datetime column.
    Expects columns: DATE (e.g., 'Oct 23'), TIME (e.g., '8:15PM')
    """
    df = pd.read_csv(path)

    # Normalize teams if present
    for col in ["TEAM_A", "TEAM_B"]:
        if col in df.columns:
            df[col] = normalize_series_with_aliases(df[col], alias_map)

    # Add WeatherAPI datetime column
    if "DATE" in df.columns and "TIME" in df.columns:
        def to_weatherapi_datetime(row):
            d = str(row["DATE"]).strip()
            t = str(row["TIME"]).strip()
            try:
                date_iso = datetime.strptime(f"{d} {YEAR}", "%b %d %Y").strftime("%Y-%m-%d")
                time_24 = datetime.strptime(t, "%I:%M%p").strftime("%H:%M")
                return f"{date_iso} {time_24}"
            except Exception:
                return ""  # leave blank if parsing fails

        df["weatherapi_datetime"] = df.apply(to_weatherapi_datetime, axis=1)

    df.to_csv(path, index=False)

def main():
    if not ALIASES_PATH.exists():
        print(f"Aliases file not found: {ALIASES_PATH}", file=sys.stderr)
        sys.exit(1)

    alias_map = load_alias_map(ALIASES_PATH)

    for f in FILES_TO_NORMALIZE:
        if not f.exists():
            # Skip missing files without failing the whole run
            continue
        if f.name == "week_matchups_odds.csv":
            normalize_matchups(f, alias_map)
        else:
            normalize_stats_file(f, alias_map)

    print("Done.")

if __name__ == "__main__":
    main()

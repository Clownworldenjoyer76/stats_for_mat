#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/normalize_mlb_results.py

from pathlib import Path
from datetime import datetime, UTC
import pandas as pd

FINAL_SCORES_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/final_scores")
GAMES_DIR = Path("docs/win/baseball/mlb/00_intake/games")
PATTERN = "*_final_scores_MLB.csv"

ERROR_DIR = Path("docs/win/baseball/mlb/05_final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "normalize_results_log.txt"
NO_MATCH_FILE = ERROR_DIR / "normalize_results_no_match.csv"


def reset_outputs():
    LOG_FILE.write_text("", encoding="utf-8")
    if NO_MATCH_FILE.exists():
        NO_MATCH_FILE.unlink()


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


def parse_time_to_minutes(time_str):
    """Parse a time string to total minutes since midnight. Returns None if unparseable."""
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None
    time_str = str(time_str).strip()
    for fmt in ("%I:%M %p", "%H:%M", "%I:%M%p", "%H:%M:%S"):
        try:
            t = datetime.strptime(time_str, fmt)
            return t.hour * 60 + t.minute
        except ValueError:
            continue
    return None


def load_games_file(game_date_str):
    """Load the daily games file for a given date (YYYY-MM-DD)."""
    date_formatted = game_date_str.replace("-", "_")
    games_file = GAMES_DIR / f"{date_formatted}_games.csv"
    if not games_file.exists():
        return None
    try:
        df = pd.read_csv(games_file, dtype=str)
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        return df
    except Exception as e:
        log(f"ERROR loading games file | {games_file} | {e}")
        return None


def normalize_date_key(val):
    """Normalize date to YYYY-MM-DD regardless of input format."""
    val = str(val).strip()
    return val.replace("_", "-")


def find_game_id(row, games_df):
    """
    Given a final score row and the games DataFrame for that date,
    return the matched game_id or None.
    """
    home = str(row["home_team"]).strip().lower()
    away = str(row["away_team"]).strip().lower()

    candidates = games_df[
        (games_df["home_team"].str.strip().str.lower() == home) &
        (games_df["away_team"].str.strip().str.lower() == away)
    ].copy()

    if candidates.empty:
        return None

    # Single game (non-doubleheader)
    if len(candidates) == 1:
        return candidates.iloc[0]["game_id"]

    # Doubleheader — all rows should have doubleheader == 'Y'
    # Match on closest game_time
    fs_time = parse_time_to_minutes(row.get("game_time"))

    if fs_time is None:
        # Can't resolve by time — return None and flag it
        return None

    best_id = None
    best_diff = float("inf")

    for _, cand_row in candidates.iterrows():
        cand_time = parse_time_to_minutes(cand_row.get("game_time"))
        if cand_time is None:
            continue
        diff = abs(fs_time - cand_time)
        if diff < best_diff:
            best_diff = diff
            best_id = cand_row["game_id"]

    return best_id


def normalize_file(file_path: Path):
    no_match_rows = []

    try:
        df = pd.read_csv(file_path, dtype=str)
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    except Exception as e:
        log(f"ERROR reading | {file_path} | {e}")
        return []

    if "game_id" not in df.columns:
        log(f"SKIP — no game_id column | {file_path}")
        return []

    df["game_id"] = df["game_id"].astype(object)
    rows_needing_id = df["game_id"].isna() | (df["game_id"].astype(str).str.strip() == "")

    if not rows_needing_id.any():
        log(f"NO ACTION — all game_ids present | {file_path}")
        return []

    injected = 0
    games_cache = {}

    for idx in df[rows_needing_id].index:
        row = df.loc[idx]
        raw_date = str(row.get("game_date", "")).strip()
        norm_date = normalize_date_key(raw_date)  # YYYY-MM-DD

        if norm_date not in games_cache:
            games_cache[norm_date] = load_games_file(norm_date)

        games_df = games_cache[norm_date]

        if games_df is None:
            no_match_rows.append({
                "file_name": file_path.name,
                "row_index": idx,
                "game_date": raw_date,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "game_time": row.get("game_time"),
                "reason": "games file not found",
            })
            continue

        game_id = find_game_id(row, games_df)

        if game_id is None:
            no_match_rows.append({
                "file_name": file_path.name,
                "row_index": idx,
                "game_date": raw_date,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "game_time": row.get("game_time"),
                "reason": "no matching game_id found",
            })
            continue

        df.at[idx, "game_id"] = game_id
        injected += 1

    df.to_csv(file_path, index=False)
    log(f"NORMALIZED | {file_path} | injected={injected} | unmatched={len(no_match_rows)}")

    return no_match_rows


def main():
    reset_outputs()
    all_no_match = []

    files = sorted(FINAL_SCORES_DIR.glob(PATTERN))
    if not files:
        log(f"NO FILES FOUND | {FINAL_SCORES_DIR}")
        print("No final score files found.")
        return

    for file_path in files:
        all_no_match.extend(normalize_file(file_path))

    if all_no_match:
        no_match_df = pd.DataFrame(all_no_match).drop_duplicates()
        no_match_df = no_match_df.sort_values(
            ["file_name", "game_date", "home_team"],
            kind="mergesort"
        )
        no_match_df.to_csv(NO_MATCH_FILE, index=False)
        log(f"NO MATCH CSV WRITTEN | {NO_MATCH_FILE} | rows={len(no_match_df)}")
    else:
        pd.DataFrame(columns=["file_name", "row_index", "game_date", "home_team", "away_team", "game_time", "reason"]).to_csv(NO_MATCH_FILE, index=False)
        log(f"NO MATCH CSV WRITTEN EMPTY | {NO_MATCH_FILE}")

    print("MLB final score normalization complete.")


if __name__ == "__main__":
    main()

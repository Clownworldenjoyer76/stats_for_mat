#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/merge_intake.py

import csv
import traceback
from pathlib import Path
from datetime import datetime

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

INTAKE_DIR      = Path("docs/win/basketball/00_intake")
PREDICTIONS_DIR = INTAKE_DIR / "predictions" / "predictions_cleaned"
SPORTSBOOK_DIR  = INTAKE_DIR / "sportsbook"  / "sportsbook_cleaned"
MERGE_DIR       = Path("docs/win/basketball/01_merge")
ERROR_DIR       = Path("docs/win/basketball/errors/01_merge")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "merge_intake.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== merge_intake RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# OUTPUT FIELDNAMES
# =========================

MONEYLINE_FIELDS = [
    "sport", "league", "game_id", "game_date", "game_time",
    "home_team", "away_team",
    "home_prob", "away_prob",
    "away_projected_points", "home_projected_points", "total_projected_points",
    "total",
    "home_dk_moneyline_american", "away_dk_moneyline_american",
    "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
]

SPREAD_FIELDS = [
    "sport", "league", "game_id", "game_date", "game_time",
    "home_team", "away_team",
    "home_prob", "away_prob",
    "away_projected_points", "home_projected_points", "total_projected_points",
    "total",
    "home_spread", "away_spread",
    "home_dk_spread_american", "away_dk_spread_american",
    "home_dk_spread_decimal", "away_dk_spread_decimal",
]

TOTAL_FIELDS = [
    "sport", "league", "game_id", "game_date", "game_time",
    "home_team", "away_team",
    "home_prob", "away_prob",
    "away_projected_points", "home_projected_points", "total_projected_points",
    "total",
    "dk_total_over_american", "dk_total_under_american",
    "dk_total_over_decimal", "dk_total_under_decimal",
]


# =========================
# HELPERS
# =========================

def load_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_key(r: dict) -> tuple:
    return (
        r.get("game_date", "").strip(),
        r.get("home_team", "").strip(),
        r.get("away_team", "").strip(),
    )


def wipe_outputs():
    for league in LEAGUES:
        for subdir in ["moneyline", "spread", "total"]:
            folder = MERGE_DIR / league / subdir
            folder.mkdir(parents=True, exist_ok=True)
            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)
    log("Wiped all output folders.")


def write_csv(path: Path, fieldnames: list, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# =========================
# MAIN
# =========================

def main():
    files_written  = 0
    total_merged   = 0
    total_missing  = 0
    slates_skipped = 0

    try:
        wipe_outputs()

        for league in LEAGUES:
            league_upper = league.upper()
            pred_dir  = PREDICTIONS_DIR / league
            book_dir  = SPORTSBOOK_DIR  / league

            if not pred_dir.exists():
                log(f"PREDICTIONS DIR NOT FOUND: {pred_dir}")
                continue

            pred_files = sorted(pred_dir.glob(f"*_{league_upper}_predictions.csv"))

            if not pred_files:
                log(f"NO PREDICTION FILES: {pred_dir}")
                continue

            for pred_file in pred_files:
                # Parse date from filename: {date}_{LEAGUE}_predictions.csv
                date = pred_file.stem.replace(f"_{league_upper}_predictions", "")
                book_file = book_dir / f"{date}_{league_upper}_odds.csv"

                if not book_file.exists():
                    log(f"NO SPORTSBOOK FILE: {book_file} — skipping")
                    slates_skipped += 1
                    continue

                pred_rows = load_rows(pred_file)
                book_rows = load_rows(book_file)

                if not pred_rows:
                    log(f"EMPTY PREDICTIONS: {pred_file} — skipping")
                    slates_skipped += 1
                    continue

                if not book_rows:
                    log(f"EMPTY SPORTSBOOK: {book_file} — skipping")
                    slates_skipped += 1
                    continue

                book_map = {build_key(r): r for r in book_rows}

                ml_rows     = []
                spread_rows = []
                total_rows  = []

                for p in pred_rows:
                    key = build_key(p)
                    b   = book_map.get(key)

                    if b is None:
                        log(f"MISSING MATCH | {league_upper} {date} | {p.get('home_team')} vs {p.get('away_team')}")
                        total_missing += 1
                        continue

                    base = {
                        "sport":                   p.get("sport", ""),
                        "league":                  p.get("league", ""),
                        "game_id":                 p.get("game_id", ""),
                        "game_date":               p.get("game_date", ""),
                        "game_time":               p.get("game_time", ""),
                        "home_team":               p.get("home_team", ""),
                        "away_team":               p.get("away_team", ""),
                        "home_prob":               p.get("home_prob", ""),
                        "away_prob":               p.get("away_prob", ""),
                        "away_projected_points":   p.get("away_projected_points", ""),
                        "home_projected_points":   p.get("home_projected_points", ""),
                        "total_projected_points":  p.get("total_projected_points", ""),
                        "total":                   b.get("total", ""),
                    }

                    ml_rows.append({
                        **base,
                        "home_dk_moneyline_american": b.get("home_dk_moneyline_american", ""),
                        "away_dk_moneyline_american": b.get("away_dk_moneyline_american", ""),
                        "home_dk_moneyline_decimal":  b.get("home_dk_moneyline_decimal", ""),
                        "away_dk_moneyline_decimal":  b.get("away_dk_moneyline_decimal", ""),
                    })

                    spread_rows.append({
                        **base,
                        "home_spread":             b.get("home_spread", ""),
                        "away_spread":             b.get("away_spread", ""),
                        "home_dk_spread_american": b.get("home_dk_spread_american", ""),
                        "away_dk_spread_american": b.get("away_dk_spread_american", ""),
                        "home_dk_spread_decimal":  b.get("home_dk_spread_decimal", ""),
                        "away_dk_spread_decimal":  b.get("away_dk_spread_decimal", ""),
                    })

                    total_rows.append({
                        **base,
                        "dk_total_over_american":  b.get("dk_total_over_american", ""),
                        "dk_total_under_american": b.get("dk_total_under_american", ""),
                        "dk_total_over_decimal":   b.get("dk_total_over_decimal", ""),
                        "dk_total_under_decimal":  b.get("dk_total_under_decimal", ""),
                    })

                if not ml_rows:
                    log(f"NO MERGED ROWS: {league_upper} {date} — skipping")
                    slates_skipped += 1
                    continue

                ml_path     = MERGE_DIR / league / "moneyline" / f"{date}_{league_upper}_moneyline.csv"
                spread_path = MERGE_DIR / league / "spread"    / f"{date}_{league_upper}_spread.csv"
                total_path  = MERGE_DIR / league / "total"     / f"{date}_{league_upper}_total.csv"

                write_csv(ml_path,     MONEYLINE_FIELDS, ml_rows)
                write_csv(spread_path, SPREAD_FIELDS,    spread_rows)
                write_csv(total_path,  TOTAL_FIELDS,     total_rows)

                total_merged += len(ml_rows)
                files_written += 3
                log(f"WROTE {ml_path.name} | {spread_path.name} | {total_path.name} ({len(ml_rows)} rows each)")

        log("--- SUMMARY ---")
        log(f"Files written: {files_written}")
        log(f"Total rows merged: {total_merged}")
        log(f"Total missing matches: {total_missing}")
        log(f"Slates skipped: {slates_skipped}")
        log("STATUS: SUCCESS")

        print("merge_intake complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()

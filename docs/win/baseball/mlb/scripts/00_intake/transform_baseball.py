#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/transform_baseball.py

import csv
import json
import traceback
from datetime import datetime
from pathlib import Path

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "transform_baseball.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== transform_baseball RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# -------------------------
# PATHS
# -------------------------

RAW_DIR = Path("docs/win/baseball/mlb/00_intake/drat_raw")
PRED_DIR = Path("docs/win/baseball/mlb/00_intake/predictions")

PRED_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# HELPERS
# -------------------------

def parse_datetime(dt_str):
    dt = datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M %p")
    return dt, dt.strftime("%Y_%m_%d"), dt.strftime("%I:%M %p")


def clean_team(team_str):
    return team_str.split("(")[0].strip()


def pct_to_decimal(p):
    return str(round(float(p.replace("%", "")) / 100, 3))


# -------------------------
# ROW DETECTION
# Cell count is the reliable differentiator:
#   11 cells = future game / prediction row
#    8 cells = completed game row ignored by intake
# -------------------------

def is_future_game(row):
    return len(row) == 11


def is_completed_game(row):
    return len(row) == 8


SUMMARY_ROW_PREFIXES = {"Sportsbooks", "DRatings"}


def is_summary_row(row):
    return row and str(row[0]).strip() in SUMMARY_ROW_PREFIXES


# -------------------------
# WRITE HELPERS
# -------------------------

def write_csv(path, header, rows, files_written, label):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    files_written.append((str(path), len(rows)))
    log(f"WROTE {label} -> {path} ({len(rows)} rows)")


# -------------------------
# PROCESS
# -------------------------

def process_file(file_path, files_written):
    log(f"Processing {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions_by_date = {}

    parse_errors = 0
    skipped_summary = 0
    completed_rows_ignored = 0
    unknown_rows = 0

    for row in data:
        if not row or len(row) < 2:
            continue

        if is_summary_row(row):
            skipped_summary += 1
            continue

        try:
            dt, game_date, game_time = parse_datetime(row[0])
        except Exception:
            parse_errors += 1
            continue

        teams = row[1].split("\n")
        if len(teams) < 2:
            continue

        away_team = clean_team(teams[0])
        home_team = clean_team(teams[1])

        if is_future_game(row):
            # Cell layout (11 cells):
            #   0: date/time
            #   1: teams (with records)
            #   2: pitchers (away\nhome)
            #   3: win probs (away%\nhome%)
            #   4: moneyline (away\nhome)
            #   5: run line (away\nhome)
            #   6: projected runs (away\nhome)
            #   7: total projected runs
            #   8: over/under lines
            #   9: bet value label
            #  10: empty
            try:
                pitchers = row[2].split("\n")
                away_pitcher = pitchers[0].strip()
                home_pitcher = pitchers[1].strip() if len(pitchers) > 1 else ""

                probs = row[3].split("\n")
                away_prob = pct_to_decimal(probs[0])
                home_prob = pct_to_decimal(probs[1]) if len(probs) > 1 else ""

                runs = row[6].split("\n")
                away_runs = runs[0].strip()
                home_runs = runs[1].strip() if len(runs) > 1 else ""

                total_runs = row[7]

                pred_row = [
                    "",
                    "baseball",
                    "mlb",
                    game_date,
                    game_time,
                    home_team,
                    away_team,
                    home_pitcher,
                    away_pitcher,
                    home_prob,
                    away_prob,
                    away_runs,
                    home_runs,
                    total_runs,
                ]

                predictions_by_date.setdefault(game_date, []).append(pred_row)

            except Exception:
                parse_errors += 1
                continue

        elif is_completed_game(row):
            # Step 6: intake does not generate post-game score files.
            # Completed rows are handled only by the post-game final-score workflow.
            completed_rows_ignored += 1
            continue

        else:
            unknown_rows += 1
            log(f"  SKIPPED unknown row ({len(row)} cells): {row[0]} | {row[1]}")

    prediction_header = [
        "game_id",
        "sport",
        "league",
        "game_date",
        "game_time",
        "home_team",
        "away_team",
        "home_pitcher",
        "away_pitcher",
        "home_prob",
        "away_prob",
        "away_projected_runs",
        "home_projected_runs",
        "total_projected_runs",
    ]

    for date, rows in predictions_by_date.items():
        out = PRED_DIR / f"{date}_MLB.csv"
        write_csv(out, prediction_header, rows, files_written, "predictions")

    log(
        f"  parse_errors={parse_errors}, "
        f"skipped_summary={skipped_summary}, "
        f"completed_rows_ignored={completed_rows_ignored}, "
        f"unknown_rows={unknown_rows}, "
        f"predictions_dates={len(predictions_by_date)}"
    )


# -------------------------
# ENTRY
# -------------------------

def main():
    files_written = []

    try:
        raw_files = sorted(RAW_DIR.glob("*_mlb_raw.json"))
        log(f"Raw files found: {len(raw_files)}")
        log("Step 6 mode: intake writes predictions only; final-score generation is post-game only")

        for file in raw_files:
            process_file(file, files_written)

        log("--- SUMMARY ---")
        log(f"Raw files processed: {len(raw_files)}")
        log(f"Files written: {len(files_written)}")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("Baseball transform complete.")


if __name__ == "__main__":
    main()

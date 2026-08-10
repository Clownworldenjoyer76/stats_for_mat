
#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_daily_games.py

import csv
import traceback
from pathlib import Path
from datetime import datetime

# =========================
# PATHS
# =========================

LEAGUES = {
    "nba": {
        "league_label": "NBA",
        "input_dir": Path("docs/win/basketball/00_intake/sportsbook/nba"),
        "output_dir": Path("docs/win/basketball/daily_games/nba"),
    },
    "ncaam": {
        "league_label": "NCAAM",
        "input_dir": Path("docs/win/basketball/00_intake/sportsbook/ncaam"),
        "output_dir": Path("docs/win/basketball/daily_games/ncaam"),
    },
    "wnba": {
        "league_label": "WNBA",
        "input_dir": Path("docs/win/basketball/00_intake/sportsbook/wnba"),
        "output_dir": Path("docs/win/basketball/daily_games/wnba"),
    },
}

for cfg in LEAGUES.values():
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_daily_games.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== basketball_daily_games RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# HELPERS
# =========================

def clean_value(val):
    if val is None:
        return ""
    return str(val).strip()


def build_row(row: dict) -> dict:
    return {
        "sport": clean_value(row.get("sport")),
        "league": clean_value(row.get("league")),
        "game_date": clean_value(row.get("game_date")),
        "game_time": clean_value(row.get("game_time")),
        "home_team": clean_value(row.get("home_team")),
        "away_team": clean_value(row.get("away_team")),
        "game_id": clean_value(row.get("game_id")),
    }


def row_key(row: dict):
    return (
        clean_value(row.get("league")).upper(),
        clean_value(row.get("game_date")),
        clean_value(row.get("home_team")).lower(),
        clean_value(row.get("away_team")).lower(),
        clean_value(row.get("game_id")),
    )


def sort_key(row: dict):
    return (
        clean_value(row.get("game_date")),
        clean_value(row.get("game_time")),
        clean_value(row.get("home_team")),
        clean_value(row.get("away_team")),
        clean_value(row.get("game_id")),
    )


# =========================
# MAIN
# =========================

def main():
    files_written = []
    total_input_files = 0
    total_rows_read = 0
    total_rows_written = 0
    total_duplicates_skipped = 0

    fieldnames = [
        "sport",
        "league",
        "game_date",
        "game_time",
        "home_team",
        "away_team",
        "game_id",
    ]

    try:
        for league_key, cfg in LEAGUES.items():
            league_label = cfg["league_label"]
            input_dir = cfg["input_dir"]
            output_dir = cfg["output_dir"]

            if not input_dir.exists():
                log(f"INPUT DIR NOT FOUND: {input_dir}")
                continue

            csv_files = sorted(input_dir.glob("*.csv"))
            total_input_files += len(csv_files)

            if not csv_files:
                log(f"NO INPUT FILES: {input_dir}")
                continue

            by_date = {}
            seen_by_date = {}

            for csv_path in csv_files:
                log(f"PROCESSING {csv_path}")

                try:
                    with open(csv_path, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)

                        for row in reader:
                            total_rows_read += 1

                            out_row = build_row(row)
                            game_date = out_row["game_date"]

                            if not game_date:
                                log(f"  SKIPPED ROW (missing game_date) in {csv_path.name}")
                                continue

                            if game_date not in by_date:
                                by_date[game_date] = []
                                seen_by_date[game_date] = set()

                            key = row_key(out_row)
                            if key in seen_by_date[game_date]:
                                total_duplicates_skipped += 1
                                continue

                            seen_by_date[game_date].add(key)
                            by_date[game_date].append(out_row)

                except Exception as e:
                    log(f"ERROR processing {csv_path}: {e}\n{traceback.format_exc()}")

            for game_date, rows in sorted(by_date.items()):
                rows = sorted(rows, key=sort_key)
                out_path = output_dir / f"{game_date}_{league_label}.csv"

                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                files_written.append((str(out_path), len(rows)))
                total_rows_written += len(rows)
                log(f"WROTE {out_path} ({len(rows)} rows)")

        log("--- SUMMARY ---")
        log(f"Input files found: {total_input_files}")
        log(f"Rows read: {total_rows_read}")
        log(f"Duplicate rows skipped: {total_duplicates_skipped}")
        log(f"Rows written: {total_rows_written}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")

        print("Basketball daily games complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()
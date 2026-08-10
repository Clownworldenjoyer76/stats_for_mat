#!/usr/bin/env python3
# docs/win/basketball/scripts/04_select/daily_slate.py
#
# Builds per-league daily_slate selected-picks CSVs from per-date daily_picks CSVs.
#
# Inputs:
#   docs/win/basketball/04_select/nba/daily_picks/{daily}_nba_selected.csv
#   docs/win/basketball/04_select/wnba/daily_picks/{daily}_wnba_selected.csv
#   docs/win/basketball/04_select/ncaam/daily_picks/{daily}_ncaam_selected.csv
#
# Outputs:
#   docs/win/basketball/04_select/daily_slate/nba_selected.csv
#   docs/win/basketball/04_select/daily_slate/wnba_selected.csv
#   docs/win/basketball/04_select/daily_slate/ncaam_selected.csv
#
# Log:
#   docs/win/basketball/errors/04_select/daily_slate.txt
#
# Behavior:
#   - Overwrites the log every run.
#   - Clears each existing daily_slate output before rebuilding it.
#   - Appends all rows from the matching league daily_picks files into that league's
#     daily_slate output.
#   - Always writes each output file, even if there are zero input rows.
#   - Writes output headers in the fixed order below.

import csv
import traceback
from datetime import datetime, UTC
from pathlib import Path


SELECT_DIR = Path("docs/win/basketball/04_select")
DAILY_SLATE_DIR = SELECT_DIR / "daily_slate"
ERROR_DIR = Path("docs/win/basketball/errors/04_select")
LOG_FILE = ERROR_DIR / "daily_slate.txt"

LEAGUES = ["nba", "wnba", "ncaam"]

HEADERS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "away_projected_points",
    "home_projected_points",
    "total_projected_points",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "away_decimal",
    "home_decimal",
    "away_implied_prob",
    "home_implied_prob",
    "away_market_prob",
    "home_market_prob",
    "home_model_prob",
    "away_model_prob",
    "away_fair",
    "home_fair",
    "away_acceptable_decimal_moneyline",
    "home_acceptable_decimal_moneyline",
    "away_acceptable_american_moneyline",
    "home_acceptable_american_moneyline",
    "home_ml_ev",
    "away_ml_ev",
    "home_ml_edge_vs_market",
    "away_ml_edge_vs_market",
    "home_ml_kelly",
    "away_ml_kelly",
    "home_ml_ev_pct",
    "away_ml_ev_pct",
    "home_ml_edge_vs_market_pct",
    "away_ml_edge_vs_market_pct",
    "bet_side",
    "bet_line",
    "bet_odds_american",
    "bet_ev",
    "bet_kelly",
    "bet_model_prob",
    "bet_edge_vs_market",
    "bet_stake_pct",
    "market_type",
    "league_lower",
    "home_spread",
    "away_spread",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "home_spread_model_prob",
    "away_spread_model_prob",
    "fair_home_spread_decimal",
    "fair_away_spread_decimal",
    "home_acceptable_spread_decimal",
    "away_acceptable_spread_decimal",
    "home_acceptable_spread_american",
    "away_acceptable_spread_american",
    "home_spread_implied_prob",
    "away_spread_implied_prob",
    "home_spread_market_prob",
    "away_spread_market_prob",
    "home_spread_ev",
    "away_spread_ev",
    "home_spread_edge_vs_market",
    "away_spread_edge_vs_market",
    "home_spread_kelly",
    "away_spread_kelly",
    "home_spread_ev_pct",
    "away_spread_ev_pct",
    "home_spread_edge_vs_market_pct",
    "away_spread_edge_vs_market_pct",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "over_model_prob",
    "under_model_prob",
    "fair_over",
    "fair_under",
    "acceptable_over",
    "acceptable_under",
    "over_implied_prob",
    "under_implied_prob",
    "over_market_prob",
    "under_market_prob",
    "over_ev",
    "under_ev",
    "over_edge_vs_market",
    "under_edge_vs_market",
    "over_kelly",
    "under_kelly",
    "over_ev_pct",
    "under_ev_pct",
    "over_edge_vs_market_pct",
    "under_edge_vs_market_pct",
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def log(message: str, level: str = "INFO") -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{now_iso()} | {level:<5} | {message.rstrip()}\n")


def init_dirs_and_log() -> None:
    DAILY_SLATE_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== basketball daily_slate RUN {now_iso()} ===\n")


def output_path_for_league(league: str) -> Path:
    return DAILY_SLATE_DIR / f"{league}_selected.csv"


def input_dir_for_league(league: str) -> Path:
    return SELECT_DIR / league / "daily_picks"


def input_files_for_league(league: str) -> list[Path]:
    input_dir = input_dir_for_league(league)
    if not input_dir.exists():
        log(f"INPUT DIR MISSING: {input_dir}", "WARN")
        return []

    pattern = f"*_{league}_selected.csv"
    return sorted(input_dir.glob(pattern))


def normalize_row(row: dict) -> dict:
    return {header: row.get(header, "") for header in HEADERS}


def read_rows_from_file(path: Path) -> tuple[list[dict], list[str], list[str]]:
    rows = []
    missing_headers = []
    extra_headers = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        actual_headers = reader.fieldnames or []
        missing_headers = [h for h in HEADERS if h not in actual_headers]
        extra_headers = [h for h in actual_headers if h not in HEADERS]

        for row in reader:
            rows.append(normalize_row(row))

    return rows, missing_headers, extra_headers


def clear_output_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        log(f"CLEARED EXISTING OUTPUT FILE: {path}")
    else:
        log(f"NO EXISTING OUTPUT FILE TO CLEAR: {path}")


def write_output_file(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"WROTE OUTPUT FILE: {path} ({len(rows)} rows)")


def build_league_daily_slate(league: str) -> dict:
    out_path = output_path_for_league(league)

    result = {
        "league": league,
        "input_files": 0,
        "input_rows": 0,
        "output_rows": 0,
        "missing_input_dirs": 0,
        "files_with_missing_headers": 0,
        "files_with_extra_headers": 0,
        "errors": 0,
        "status": "ok",
    }

    clear_output_file(out_path)

    input_dir = input_dir_for_league(league)
    if not input_dir.exists():
        result["missing_input_dirs"] += 1

    files = input_files_for_league(league)
    result["input_files"] = len(files)

    all_rows = []

    for path in files:
        try:
            rows, missing_headers, extra_headers = read_rows_from_file(path)

            if missing_headers:
                result["files_with_missing_headers"] += 1
                log(
                    f"{path} missing headers filled as blank: {', '.join(missing_headers)}",
                    "WARN",
                )

            if extra_headers:
                result["files_with_extra_headers"] += 1
                log(
                    f"{path} extra headers ignored: {', '.join(extra_headers)}",
                    "WARN",
                )

            all_rows.extend(rows)
            result["input_rows"] += len(rows)
            log(f"READ: {path} ({len(rows)} rows)")

        except Exception as e:
            result["errors"] += 1
            result["status"] = "error"
            log(f"FAILED READING {path}: {e}\n{traceback.format_exc()}", "ERROR")

    write_output_file(out_path, all_rows)

    result["output_rows"] = len(all_rows)
    return result


def write_summary(results: list[dict]) -> None:
    total_input_files = sum(r["input_files"] for r in results)
    total_input_rows = sum(r["input_rows"] for r in results)
    total_output_rows = sum(r["output_rows"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_missing_input_dirs = sum(r["missing_input_dirs"] for r in results)
    total_missing_header_files = sum(r["files_with_missing_headers"] for r in results)
    total_extra_header_files = sum(r["files_with_extra_headers"] for r in results)

    lines = [
        "",
        "=" * 70,
        f"SUMMARY {now_iso()}",
        "=" * 70,
        f"  leagues_processed          : {len(results)}",
        f"  input_files                : {total_input_files}",
        f"  input_rows                 : {total_input_rows}",
        f"  output_rows                : {total_output_rows}",
        f"  missing_input_dirs         : {total_missing_input_dirs}",
        f"  files_with_missing_headers : {total_missing_header_files}",
        f"  files_with_extra_headers   : {total_extra_header_files}",
        f"  errors                     : {total_errors}",
        "",
        f"  {'league':<8} {'input_files':>11} {'input_rows':>11} {'output_rows':>12} {'errors':>8} {'status':>10}",
    ]

    for r in results:
        lines.append(
            f"  {r['league']:<8} "
            f"{r['input_files']:>11} "
            f"{r['input_rows']:>11} "
            f"{r['output_rows']:>12} "
            f"{r['errors']:>8} "
            f"{r['status']:>10}"
        )

    status = "SUCCESS" if total_errors == 0 else "COMPLETED WITH ERRORS"
    lines.extend(["", f"STATUS: {status}", "=" * 70])

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    init_dirs_and_log()

    log(f"SELECT_DIR      : {SELECT_DIR}")
    log(f"DAILY_SLATE_DIR : {DAILY_SLATE_DIR}")
    log(f"LOG_FILE        : {LOG_FILE}")

    results = []

    for league in LEAGUES:
        log(f"--- BUILD LEAGUE DAILY SLATE: {league} ---")
        try:
            results.append(build_league_daily_slate(league))
        except Exception as e:
            log(f"FATAL LEAGUE ERROR league={league}: {e}\n{traceback.format_exc()}", "ERROR")
            results.append({
                "league": league,
                "input_files": 0,
                "input_rows": 0,
                "output_rows": 0,
                "missing_input_dirs": 0,
                "files_with_missing_headers": 0,
                "files_with_extra_headers": 0,
                "errors": 1,
                "status": "fatal_error",
            })

    write_summary(results)
    print("basketball daily_slate complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Failure-signaling launcher for basketball_game_id_core.py.

The core still scans and repairs full history. After a successful run this launcher
rewrites the mismatch section of the operational log so it reports only the current
New York game date, preventing resolved/historical mismatch noise from looking like
an active pipeline problem.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CORE_PATH = Path(__file__).with_name("basketball_game_id_core.py")
BASE = Path("docs/win/basketball")
NY = ZoneInfo("America/New_York")
LABELS = {"nba": "NBA", "ncaam": "NCAAM", "wnba": "WNBA"}


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_game_id_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def key(row: dict) -> tuple[str, str, str]:
    return (
        clean(row.get("game_date")),
        clean(row.get("home_team")).casefold(),
        clean(row.get("away_team")).casefold(),
    )


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def unique_map(rows: list[dict]) -> tuple[dict[tuple[str, str, str], dict], list[dict]]:
    unique: dict[tuple[str, str, str], dict] = {}
    duplicates: list[dict] = []
    for row in rows:
        k = key(row)
        if not all(k):
            continue
        if k in unique:
            duplicates.append(row)
        else:
            unique[k] = row
    return unique, duplicates


def current_mismatches() -> tuple[str, list[dict], list[dict], list[dict]]:
    game_date = datetime.now(NY).strftime("%Y_%m_%d")
    daily_missing: list[dict] = []
    prediction_missing: list[dict] = []
    duplicates: list[dict] = []

    for league, label in LABELS.items():
        daily_path = BASE / f"daily_games/{league}/{game_date}_{label}.csv"
        pred_path = BASE / f"00_intake/predictions/{league}/{game_date}_{label}_predictions.csv"
        daily_rows = read_rows(daily_path)
        pred_rows = read_rows(pred_path)
        daily_map, daily_dupes = unique_map(daily_rows)
        pred_map, pred_dupes = unique_map(pred_rows)

        for k in sorted(set(daily_map) - set(pred_map)):
            row = daily_map[k]
            daily_missing.append({"league": label, **row})
        for k in sorted(set(pred_map) - set(daily_map)):
            row = pred_map[k]
            prediction_missing.append({"league": label, **row, "source_file": str(pred_path)})
        for row in daily_dupes:
            duplicates.append({"league": label, "source": "daily_games", **row})
        for row in pred_dupes:
            duplicates.append({"league": label, "source": "predictions", **row})

    return game_date, daily_missing, prediction_missing, duplicates


def rewrite_operational_log(log_path: Path, original_text: str) -> None:
    game_date, daily_missing, prediction_missing, duplicates = current_mismatches()
    lines = original_text.splitlines()

    for idx, line in enumerate(lines):
        if "Sportsbook rows with no prediction match:" in line and " | " in line:
            prefix = line.split("Sportsbook rows with no prediction match:", 1)[0]
            lines[idx] = f"{prefix}Current-date daily games with no prediction match: {len(daily_missing)}"
        elif "Prediction rows with no sportsbook match:" in line and " | " in line:
            prefix = line.split("Prediction rows with no sportsbook match:", 1)[0]
            lines[idx] = f"{prefix}Current-date prediction rows with no daily-game match: {len(prediction_missing)}"
        elif "Duplicate key rows:" in line and " | " in line:
            prefix = line.split("Duplicate key rows:", 1)[0]
            lines[idx] = f"{prefix}Current-date duplicate key rows: {len(duplicates)}"

    marker_index = next((i for i, line in enumerate(lines) if "--- SMALL MISMATCH REPORT ---" in line), None)
    status_index = None
    if marker_index is not None:
        for i in range(len(lines) - 1, marker_index, -1):
            if "STATUS:" in lines[i]:
                status_index = i
                break

    if marker_index is None or status_index is None:
        raise RuntimeError("Unable to locate mismatch report/status markers in basketball_game_id log")

    prefix_lines = lines[:marker_index]
    status_line = lines[status_index]
    report = [
        f"{datetime.now().isoformat()} | --- CURRENT-DATE MISMATCH REPORT ---",
        f"{datetime.now().isoformat()} | Operational mismatch date: {game_date} (America/New_York)",
        f"{datetime.now().isoformat()} | Current-date daily games with no prediction match:",
    ]
    if daily_missing:
        for row in daily_missing:
            report.append(
                f"{datetime.now().isoformat()} |   {row['league']} | {clean(row.get('game_date'))} | "
                f"{clean(row.get('home_team'))} | {clean(row.get('away_team'))} | "
                f"game_id={clean(row.get('game_id'))}"
            )
    else:
        report.append(f"{datetime.now().isoformat()} |   none")

    report.append(f"{datetime.now().isoformat()} | Current-date prediction rows with no daily-game match:")
    if prediction_missing:
        for row in prediction_missing:
            report.append(
                f"{datetime.now().isoformat()} |   {row['league']} | {clean(row.get('game_date'))} | "
                f"{clean(row.get('home_team'))} | {clean(row.get('away_team'))} | "
                f"file={row['source_file']}"
            )
    else:
        report.append(f"{datetime.now().isoformat()} |   none")

    report.append(f"{datetime.now().isoformat()} | Current-date duplicate key rows:")
    if duplicates:
        for row in duplicates:
            report.append(
                f"{datetime.now().isoformat()} |   {row['league']} | {clean(row.get('game_date'))} | "
                f"{clean(row.get('home_team'))} | {clean(row.get('away_team'))} | source={row['source']}"
            )
    else:
        report.append(f"{datetime.now().isoformat()} |   none")

    log_path.write_text("\n".join(prefix_lines + report + [status_line]) + "\n", encoding="utf-8")


def main() -> None:
    core = load_core()
    core.main()
    text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    fatal_markers = ("ERROR loading", "ERROR updating", "FATAL ERROR", "STATUS: FAILED")
    if any(marker in text for marker in fatal_markers):
        sys.exit(1)
    rewrite_operational_log(core.LOG_FILE, text)


if __name__ == "__main__":
    main()

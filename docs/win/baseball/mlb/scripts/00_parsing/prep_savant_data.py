#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_parsing/prep_savant_data.py
#
# Reads raw Savant files from data/ subfolders, cleans them,
# and writes _clean versions to the same subfolders.
# Run once when Savant data is refreshed.
#
# NOTE:
# This script intentionally does nothing to:
# docs/win/baseball/mlb/data/park_factors

import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd


BASE_DIR = Path("docs/win/baseball/mlb/data")
BATTING_DIR = BASE_DIR / "batting"
PITCHING_DIR = BASE_DIR / "pitching"
FIELDING_DIR = BASE_DIR / "fielding"
BASERUNNING_DIR = BASE_DIR / "baserunning"

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_parsing")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "prep_savant_data.txt"


def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


PA_THRESHOLDS = {
    "2026": 25,
    "2025": 100,
    "2024": 150,
    "2023": 150,
    "2022": 150,
}

SAMPLE_FLAG_LOW = {
    "2026": 75,
    "2025": 200,
}

BATTING_PITCHING_DROP = [
    "year",
    "player_age",
    "single",
    "double",
    "triple",
    "hit",
    "ab",
    "swords",
    "attack_direction",
    "vertical_swing_path",
]

BATTING_PITCHING_RENAME = {
    "last_name, first_name": "player_name",
    "k_percent": "k_pct",
    "bb_percent": "bb_pct",
    "barrel_batted_rate": "barrel_pct",
    "hard_hit_percent": "hard_hit_pct",
    "whiff_percent": "whiff_pct",
    "exit_velocity_avg": "exit_velo",
    "sweet_spot_percent": "sweet_spot_pct",
    "avg_swing_speed": "swing_speed",
    "slg_percent": "slg_pct",
    "on_base_percent": "obp",
    "on_base_plus_slg": "ops",
}

FIELDING_DROP = [
    "outs_2",
    "outs_3",
    "outs_4",
    "outs_5",
    "outs_6",
    "outs_7",
    "outs_8",
    "outs_9",
]


def _year_key(filename: str) -> str:
    """Extract year key from filename for PA threshold lookup."""
    stem = Path(filename).stem
    for yr in ["2026", "2025", "2024", "2023", "2022"]:
        if yr in stem:
            return yr
    return None


def _check_duplicates(df: pd.DataFrame, id_col: str, filepath: str) -> int:
    dupes = df[id_col].duplicated().sum()
    if dupes > 0:
        _log(
            f"  {filepath} — {dupes} duplicate {id_col} rows found",
            "WARN",
        )
    return dupes


def clean_batting_pitching(filepath: Path, summary: dict) -> None:
    label = filepath.name
    _log(f"--- {label}")

    year_key = _year_key(label)
    if year_key is None:
        _log(
            "  Cannot determine year key from filename — skipping",
            "WARN",
        )
        summary["skipped"] += 1
        return

    pa_min = PA_THRESHOLDS[year_key]

    try:
        df = pd.read_csv(filepath, dtype={"player_id": str})
    except Exception as e:
        _log(f"  READ ERROR: {e}", "ERROR")
        summary["errors"] += 1
        return

    rows_raw = len(df)
    _log(f"  Rows raw: {rows_raw}")

    df["pa"] = pd.to_numeric(df["pa"], errors="coerce")

    df = df[df["pa"] >= pa_min].copy()
    rows_after_pa = len(df)
    rows_dropped_pa = rows_raw - rows_after_pa
    _log(
        f"  Rows after PA filter (>={pa_min}): "
        f"{rows_after_pa} (dropped {rows_dropped_pa})"
    )
    summary["rows_dropped_pa"] += rows_dropped_pa

    cols_to_drop = [
        c for c in BATTING_PITCHING_DROP
        if c in df.columns
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    _log(f"  Dropped columns: {cols_to_drop}")

    rename_map = {
        k: v
        for k, v in BATTING_PITCHING_RENAME.items()
        if k in df.columns
    }
    df.rename(columns=rename_map, inplace=True)
    _log(f"  Renamed columns: {list(rename_map.keys())}")

    if year_key in SAMPLE_FLAG_LOW:
        threshold = SAMPLE_FLAG_LOW[year_key]
        df["sample_flag"] = df["pa"].apply(
            lambda x: "low"
            if pd.notna(x) and x < threshold
            else "ok"
        )
        low_count = (df["sample_flag"] == "low").sum()
        _log(
            f"  sample_flag=low count: {low_count} "
            f"(threshold PA<{threshold})"
        )
        summary["sample_flag_low"] += low_count
    else:
        df["sample_flag"] = "ok"

    _check_duplicates(df, "player_id", label)

    out_path = filepath.parent / (filepath.stem + "_clean.csv")
    df.to_csv(out_path, index=False)
    _log(f"  WROTE: {out_path.name} ({len(df)} rows)")
    summary["files_written"] += 1
    summary["rows_written"] += len(df)


def clean_fielding(filepath: Path, summary: dict) -> None:
    label = filepath.name
    _log(f"--- {label}")

    try:
        df = pd.read_csv(filepath, dtype={"id": str})
    except Exception as e:
        _log(f"  READ ERROR: {e}", "ERROR")
        summary["errors"] += 1
        return

    rows_raw = len(df)
    _log(f"  Rows raw: {rows_raw}")

    if "id" not in df.columns:
        _log(
            "  MISSING join key column 'id' — skipping",
            "ERROR",
        )
        summary["errors"] += 1
        return

    _log(
        "  Join key 'id' present — verify these are MLBAM "
        "player IDs before joining"
    )

    cols_to_drop = [
        c for c in FIELDING_DROP
        if c in df.columns
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    _log(f"  Dropped columns: {cols_to_drop}")

    _check_duplicates(df, "id", label)

    out_path = filepath.parent / (filepath.stem + "_clean.csv")
    df.to_csv(out_path, index=False)
    _log(f"  WROTE: {out_path.name} ({len(df)} rows)")
    summary["files_written"] += 1
    summary["rows_written"] += len(df)


def clean_baserunning(filepath: Path, summary: dict) -> None:
    label = filepath.name
    _log(f"--- {label}")

    try:
        df = pd.read_csv(filepath, dtype={"player_id": str})
    except Exception as e:
        _log(f"  READ ERROR: {e}", "ERROR")
        summary["errors"] += 1
        return

    rows_raw = len(df)
    _log(f"  Rows raw: {rows_raw}")

    if "entity_name" in df.columns:
        df.rename(
            columns={"entity_name": "player_name"},
            inplace=True,
        )
        _log("  Renamed: entity_name → player_name")

    _check_duplicates(df, "player_id", label)

    out_path = filepath.parent / (filepath.stem + "_clean.csv")
    df.to_csv(out_path, index=False)
    _log(f"  WROTE: {out_path.name} ({len(df)} rows)")
    summary["files_written"] += 1
    summary["rows_written"] += len(df)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== prep_savant_data RUN {_now()} ===\n")

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "rows_dropped_pa": 0,
        "sample_flag_low": 0,
        "skipped": 0,
        "errors": 0,
    }

    _log("=== BATTING ===")
    batting_files = sorted(BATTING_DIR.glob("batting_*.csv"))
    batting_files = [
        f for f in batting_files
        if "_clean" not in f.stem
    ]
    _log(f"Files found: {len(batting_files)}")

    for fp in batting_files:
        try:
            clean_batting_pitching(fp, summary)
        except Exception as e:
            _log(
                f"UNHANDLED ERROR {fp.name}: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            summary["errors"] += 1

    _log("=== PITCHING ===")
    pitching_files = sorted(PITCHING_DIR.glob("pitching_*.csv"))
    pitching_files = [
        f for f in pitching_files
        if "_clean" not in f.stem
    ]
    _log(f"Files found: {len(pitching_files)}")

    for fp in pitching_files:
        try:
            clean_batting_pitching(fp, summary)
        except Exception as e:
            _log(
                f"UNHANDLED ERROR {fp.name}: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            summary["errors"] += 1

    _log("=== FIELDING ===")
    fielding_files = sorted(FIELDING_DIR.glob("fielding_*.csv"))
    fielding_files = [
        f for f in fielding_files
        if "_clean" not in f.stem
    ]
    _log(f"Files found: {len(fielding_files)}")

    for fp in fielding_files:
        try:
            clean_fielding(fp, summary)
        except Exception as e:
            _log(
                f"UNHANDLED ERROR {fp.name}: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            summary["errors"] += 1

    _log("=== BASERUNNING ===")
    baserunning_files = sorted(
        BASERUNNING_DIR.glob("baserunning_*.csv")
    )
    baserunning_files = [
        f for f in baserunning_files
        if "_clean" not in f.stem
    ]
    _log(f"Files found: {len(baserunning_files)}")

    for fp in baserunning_files:
        try:
            clean_baserunning(fp, summary)
        except Exception as e:
            _log(
                f"UNHANDLED ERROR {fp.name}: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            summary["errors"] += 1

    _log("=== PARK FACTORS ===")
    _log(
        "Skipped intentionally. This script does not read, write, "
        "clean, or modify "
        "docs/win/baseball/mlb/data/park_factors."
    )

    status = (
        "SUCCESS"
        if summary["errors"] == 0
        else "COMPLETED WITH ERRORS"
    )

    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_written   : {summary['files_written']}",
        f"  rows_written    : {summary['rows_written']}",
        f"  rows_dropped_pa : {summary['rows_dropped_pa']}",
        f"  sample_flag_low : {summary['sample_flag_low']}",
        f"  skipped         : {summary['skipped']}",
        f"  errors          : {summary['errors']}",
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"prep_savant_data complete. "
        f"{summary['files_written']} files written. "
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()

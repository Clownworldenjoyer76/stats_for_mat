#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/normalize_final_score_mlb_names.py
#
# Normalizes MLB final-score team names.
#
# Master mapping file:
#   docs/win/baseball/mlb/maps/team_map_mlb.csv
#
# Expected map columns:
#   league,team_id,alias,canonical_team
#
# Behavior:
#   - Rewrites home_team / away_team to canonical_team.
#   - Requires the new combined map structure.
#   - Validates alias uniqueness.
#   - Validates canonical_team -> team_id consistency.
#   - Writes unmapped teams to team_normalization_no_map.csv.
#   - Hard fails if map is missing/malformed/ambiguous or if unmapped teams remain.

from pathlib import Path
from datetime import datetime, UTC
import sys
import traceback
import pandas as pd

BASE = Path("docs/win/baseball/mlb/05_final_scores/results/final_scores")

INPUT_DIR = BASE
PATTERN = "*_final_scores_MLB.csv"

MAP_FILE = Path("docs/win/baseball/mlb/maps/team_map_mlb.csv")
MAP_FILTER_COL = "league"
MAP_FILTER_VAL = "mlb"

ERROR_DIR = Path("docs/win/baseball/mlb/05_final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "team_normalization_log.txt"
NO_MAP_FILE = ERROR_DIR / "team_normalization_no_map.csv"

REQUIRED_MAP_COLUMNS = {"league", "team_id", "alias", "canonical_team"}
TEAM_COLUMNS = ["away_team", "home_team"]


def reset_outputs():
    LOG_FILE.write_text("", encoding="utf-8")
    if NO_MAP_FILE.exists():
        NO_MAP_FILE.unlink()


def log(msg, level="INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {level:<5} | {msg}\n")


def norm_key(val):
    return str(val or "").strip().lower()


def clean(val):
    return str(val or "").strip()


def write_no_map(rows):
    if rows:
        no_map_df = pd.DataFrame(rows).drop_duplicates()
        no_map_df = no_map_df.sort_values(
            ["league", "file_name", "team_col", "unmapped_value"],
            kind="mergesort",
        )
    else:
        no_map_df = pd.DataFrame(
            columns=["league", "file_name", "team_col", "unmapped_value"]
        )

    no_map_df.to_csv(NO_MAP_FILE, index=False)
    log(f"NO MAP CSV WRITTEN | {NO_MAP_FILE} | rows={len(no_map_df)}")


def load_map(map_file: Path, filter_col: str, filter_val: str):
    if not map_file.exists():
        raise FileNotFoundError(f"Missing required map file: {map_file}")

    df = pd.read_csv(map_file, dtype=str).fillna("")

    missing = REQUIRED_MAP_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{map_file} missing required columns: {sorted(missing)}")

    if filter_col in df.columns:
        df = df[
            df[filter_col].astype(str).str.strip().str.lower()
            == str(filter_val).strip().lower()
        ].copy()

    if df.empty:
        raise ValueError(f"No rows found in {map_file} for {filter_col}={filter_val}")

    mapping = {}
    team_id_to_canonical = {}
    canonical_to_team_id = {}
    duplicate_identical_alias_rows = 0

    for idx, row in df.iterrows():
        csv_row = idx + 2

        league = norm_key(row.get("league"))
        team_id = clean(row.get("team_id"))
        alias_raw = clean(row.get("alias"))
        alias_key = norm_key(alias_raw)
        canonical = clean(row.get("canonical_team"))

        if not league or not team_id or not alias_raw or not canonical:
            raise ValueError(
                f"{map_file}:{csv_row} blank required value: "
                f"league={league!r} team_id={team_id!r} "
                f"alias={alias_raw!r} canonical_team={canonical!r}"
            )

        existing_canonical = mapping.get(alias_key)
        if existing_canonical and existing_canonical != canonical:
            raise ValueError(
                f"{map_file}:{csv_row} ambiguous alias mapping: "
                f"alias={alias_raw!r} existing={existing_canonical!r} new={canonical!r}"
            )

        if existing_canonical == canonical:
            duplicate_identical_alias_rows += 1

        mapping[alias_key] = canonical

        existing_for_id = team_id_to_canonical.get(team_id)
        if existing_for_id and existing_for_id != canonical:
            raise ValueError(
                f"{map_file}:{csv_row} team_id maps to multiple canonical teams: "
                f"team_id={team_id!r} existing={existing_for_id!r} new={canonical!r}"
            )

        existing_id_for_canonical = canonical_to_team_id.get(canonical)
        if existing_id_for_canonical and existing_id_for_canonical != team_id:
            raise ValueError(
                f"{map_file}:{csv_row} canonical_team maps to multiple team_ids: "
                f"canonical_team={canonical!r} existing={existing_id_for_canonical!r} new={team_id!r}"
            )

        team_id_to_canonical[team_id] = canonical
        canonical_to_team_id[canonical] = team_id

    if not mapping:
        raise ValueError(f"No valid mappings loaded from {map_file}")

    log(
        f"MAP LOADED | {filter_val} | "
        f"aliases={len(mapping)} "
        f"team_ids={len(team_id_to_canonical)} "
        f"canonical_teams={len(canonical_to_team_id)} "
        f"duplicate_identical_alias_rows={duplicate_identical_alias_rows}"
    )

    return mapping


def normalize_file(file_path: Path, mapping: dict):
    df = pd.read_csv(file_path, dtype=str).fillna("")

    missing = set(TEAM_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} missing required team columns: {sorted(missing)}")

    no_map_rows = []
    names_updated = 0

    for col in TEAM_COLUMNS:
        original_col = f"{col}_original"
        key_col = f"{col}_key"

        df[original_col] = df[col].astype(str).str.strip()
        df[key_col] = df[original_col].map(norm_key)

        mapped = df[key_col].map(mapping)
        missing_mask = mapped.isna() & df[original_col].astype(str).str.strip().ne("")

        if missing_mask.any():
            for team_val in sorted(df.loc[missing_mask, original_col].dropna().unique()):
                no_map_rows.append({
                    "league": "mlb",
                    "file_name": file_path.name,
                    "team_col": col,
                    "unmapped_value": team_val,
                })

        changed_mask = mapped.notna() & (df[col].astype(str).str.strip() != mapped)
        names_updated += int(changed_mask.sum())

        df[col] = mapped.fillna(df[original_col])

    df = df.drop(
        columns=[
            "away_team_original",
            "home_team_original",
            "away_team_key",
            "home_team_key",
        ],
        errors="ignore",
    )

    df.to_csv(file_path, index=False)

    log(
        f"NORMALIZED | {file_path} | rows={len(df)} "
        f"names_updated={names_updated} unmapped={len(no_map_rows)}"
    )

    return no_map_rows, names_updated


def main():
    reset_outputs()

    all_no_map = []
    files_processed = 0
    files_failed = 0
    total_rows_updated = 0

    try:
        mapping = load_map(MAP_FILE, MAP_FILTER_COL, MAP_FILTER_VAL)

        files = sorted(INPUT_DIR.glob(PATTERN))
        if not files:
            log(f"NO FILES | mlb | {INPUT_DIR}", "WARN")
        else:
            log(f"FILES FOUND | mlb | count={len(files)}")

        for file_path in files:
            try:
                files_processed += 1
                no_map_rows, names_updated = normalize_file(file_path, mapping)
                all_no_map.extend(no_map_rows)
                total_rows_updated += names_updated
            except Exception as e:
                files_failed += 1
                log(f"ERROR | {file_path} | {e}\n{traceback.format_exc()}", "ERROR")

        write_no_map(all_no_map)

        log("--- SUMMARY ---")
        log(f"Files processed: {files_processed}")
        log(f"Files failed: {files_failed}")
        log(f"Names normalized: {total_rows_updated}")
        log(f"Unmapped teams: {len(pd.DataFrame(all_no_map).drop_duplicates()) if all_no_map else 0}")

        if files_failed or all_no_map:
            log("STATUS: FAILED", "ERROR")
            return 1

        log("STATUS: SUCCESS")
        return 0

    except Exception as e:
        write_no_map(all_no_map)
        log(f"FATAL ERROR | {e}\n{traceback.format_exc()}", "ERROR")
        log("--- SUMMARY ---")
        log(f"Files processed: {files_processed}")
        log(f"Files failed: {files_failed}")
        log(f"Names normalized: {total_rows_updated}")
        log(f"Unmapped teams: {len(pd.DataFrame(all_no_map).drop_duplicates()) if all_no_map else 0}")
        log("STATUS: FAILED", "ERROR")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(
        "MLB final score team normalization complete. "
        f"Status: {'SUCCESS' if exit_code == 0 else 'FAILED'}"
    )
    sys.exit(exit_code)

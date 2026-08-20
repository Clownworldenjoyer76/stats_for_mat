#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/name_normalization.py
#
# Normalizes MLB team names in sportsbook and prediction CSVs.
#
# Master mapping file:
#   mappings/baseball/team_map_mlb.csv
#
# Expected map columns:
#   league,team_id,alias,canonical_team
#
# Behavior:
#   - Rewrites home_team / away_team to canonical_team.
#   - Requires team_id in the map.
#   - Writes unmapped teams to docs/win/baseball/mlb/mappings/no_map/no_map_mlb.csv.
#   - Hard fails if the map is missing, malformed, ambiguous, or if any teams are unmapped.

import csv
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

# =========================
# PATHS
# =========================

SPORTSBOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/baseball/mlb/00_intake/predictions")

MAP_FILE = Path("docs/win/baseball/mlb/maps/team_map_mlb.csv")

NO_MAP_DIR = Path("docs/win/baseball/mlb/maps/no_map")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_mlb.csv"

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "name_normalization.txt"

REQUIRED_MAP_COLUMNS = {"league", "team_id", "alias", "canonical_team"}
TEAM_COLUMNS = ["home_team", "away_team"]


# =========================
# LOGGING
# =========================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_log() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== name_normalization RUN {utc_now()} ===\n")


def log(msg: str, level: str = "INFO") -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{utc_now()} | {level:<5} | {msg}\n")


# =========================
# HELPERS
# =========================

def clean(value) -> str:
    return str(value or "").strip()


def key_text(value) -> str:
    return clean(value).lower()


def load_team_map() -> dict:
    """
    Returns:
        dict[(league, alias_lower)] = {
            "team_id": team_id,
            "canonical_team": canonical_team,
        }
    """
    if not MAP_FILE.exists():
        raise FileNotFoundError(f"Missing required map file: {MAP_FILE}")

    team_map = {}
    canonical_to_ids = {}
    duplicate_same_rows = 0

    with open(MAP_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = REQUIRED_MAP_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(
                f"{MAP_FILE} is missing required column(s): {sorted(missing_columns)}"
            )

        for line_num, row in enumerate(reader, start=2):
            league = key_text(row.get("league"))
            team_id = clean(row.get("team_id"))
            alias = clean(row.get("alias"))
            alias_key = key_text(alias)
            canonical = clean(row.get("canonical_team"))

            if not league or not team_id or not alias or not canonical:
                raise ValueError(
                    f"{MAP_FILE}:{line_num} has blank required value: "
                    f"league={league!r}, team_id={team_id!r}, "
                    f"alias={alias!r}, canonical_team={canonical!r}"
                )

            map_key = (league, alias_key)
            map_value = {
                "team_id": team_id,
                "canonical_team": canonical,
            }

            if map_key in team_map:
                existing = team_map[map_key]
                if existing != map_value:
                    raise ValueError(
                        f"{MAP_FILE}:{line_num} ambiguous alias mapping for "
                        f"league={league!r}, alias={alias!r}: "
                        f"existing={existing}, new={map_value}"
                    )
                duplicate_same_rows += 1
                continue

            team_map[map_key] = map_value
            canonical_to_ids.setdefault((league, canonical), set()).add(team_id)

    bad_canonicals = {
        key: ids
        for key, ids in canonical_to_ids.items()
        if len(ids) > 1
    }

    if bad_canonicals:
        details = "; ".join(
            [
                f"league={league} canonical_team={canonical} team_ids={sorted(ids)}"
                for (league, canonical), ids in sorted(bad_canonicals.items())
            ]
        )
        raise ValueError(f"Canonical team maps to multiple team_ids: {details}")

    if not team_map:
        raise ValueError(f"No valid rows loaded from {MAP_FILE}")

    log(
        f"Team map loaded: {len(team_map)} alias entries "
        f"from {MAP_FILE} | duplicate identical rows skipped: {duplicate_same_rows}"
    )

    return team_map


def collect_target_files() -> list:
    target_files = []

    for f in SPORTSBOOK_DIR.glob("*_MLB.csv"):
        target_files.append(f)

    for f in PREDICTIONS_DIR.glob("*_MLB.csv"):
        target_files.append(f)

    target_files = sorted(set(target_files))
    log(f"Files to process: {len(target_files)}")

    return target_files


def validate_target_columns(csv_file: Path, fieldnames: list) -> None:
    missing = [col for col in TEAM_COLUMNS if col not in (fieldnames or [])]
    if missing:
        raise ValueError(
            f"{csv_file} missing required team column(s): {missing}"
        )

    if "league" not in (fieldnames or []):
        raise ValueError(
            f"{csv_file} missing required column: league"
        )


def write_unmapped(unmapped: set) -> None:
    with open(NO_MAP_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["league", "team"])
        for league, team in sorted(unmapped):
            writer.writerow([league, team])


# =========================
# PROCESS FILES
# =========================

def process_file(csv_file: Path, team_map: dict, unmapped: set) -> dict:
    stats = {
        "rows_processed": 0,
        "names_updated": 0,
        "file_modified": False,
    }

    updated_rows = []

    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        validate_target_columns(csv_file, fieldnames)

        for row_num, row in enumerate(reader, start=2):
            stats["rows_processed"] += 1

            league_raw = clean(row.get("league"))
            league = key_text(league_raw)

            if not league:
                raise ValueError(f"{csv_file}:{row_num} blank league")

            for side in TEAM_COLUMNS:
                original_team = clean(row.get(side))

                if not original_team:
                    continue

                map_key = (league, key_text(original_team))

                if map_key not in team_map:
                    unmapped.add((league, original_team))
                    continue

                canonical = team_map[map_key]["canonical_team"]

                if row.get(side) != canonical:
                    log(
                        f"{csv_file} row={row_num} {side}: "
                        f"{original_team!r} -> {canonical!r}"
                    )
                    row[side] = canonical
                    stats["file_modified"] = True
                    stats["names_updated"] += 1

            updated_rows.append(row)

    if stats["file_modified"]:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        log(f"UPDATED: {csv_file}")

    return stats


# =========================
# MAIN
# =========================

def main() -> int:
    init_log()

    files_processed = 0
    rows_processed = 0
    rows_updated = 0
    errors = 0
    unmapped = set()

    try:
        team_map = load_team_map()
        target_files = collect_target_files()

        for csv_file in target_files:
            try:
                files_processed += 1
                stats = process_file(csv_file, team_map, unmapped)
                rows_processed += stats["rows_processed"]
                rows_updated += stats["names_updated"]

            except Exception as e:
                errors += 1
                log(
                    f"ERROR processing {csv_file}: {e}\n{traceback.format_exc()}",
                    "ERROR",
                )

        write_unmapped(unmapped)

        if unmapped:
            log(
                f"UNMAPPED teams found. See {NO_MAP_FILE}. "
                f"Count={len(unmapped)}",
                "ERROR",
            )
            for league, team in sorted(unmapped):
                log(f"UNMAPPED: league={league} team={team}", "ERROR")

        log("--- SUMMARY ---")
        log(f"Files processed: {files_processed}")
        log(f"Rows processed: {rows_processed}")
        log(f"Names normalized: {rows_updated}")
        log(f"Unmapped teams: {len(unmapped)}")
        log(f"Errors: {errors}")

        if errors or unmapped:
            log("STATUS: FAILED", "ERROR")
            return 1

        log("STATUS: SUCCESS")
        return 0

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}", "ERROR")
        log("--- SUMMARY ---")
        log(f"Files processed: {files_processed}")
        log(f"Rows processed: {rows_processed}")
        log(f"Names normalized: {rows_updated}")
        log(f"Unmapped teams: {len(unmapped)}")
        log(f"Errors: {errors + 1}")
        log("STATUS: FAILED", "ERROR")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(
        "MLB name normalization complete. "
        f"Status: {'SUCCESS' if exit_code == 0 else 'FAILED'}"
    )
    sys.exit(exit_code)

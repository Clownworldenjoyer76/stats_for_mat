#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_parsing/mlb_savant_park_data.py
#
# Reads the 12 MLB Savant park-factor CSV files, validates team_id and venue_id
# against the MLB map files, and writes matching *_clean.csv files.
#
# Input park-factor header:
# "Team","Venue","Year","Park Factor","wOBAcon","xwOBAcon","BACON","xBACON","HardHit","R","OBP","H","1B","2B","3B","HR","BB","SO","PA","venue_id","team_id"
#
# Output clean header:
# team_id,venue_id,bat_side,condition,Park Factor,wOBAcon,xwOBAcon,HardHit,R,HR

import csv
import sys
from pathlib import Path


TEAM_MAP_PATH = Path("docs/win/baseball/mlb/maps/mlb_team_ids.csv")
VENUE_MAP_PATH = Path("docs/win/baseball/mlb/maps/mlb_venue_ids.csv")
PARK_DIR = Path("docs/win/baseball/mlb/data/park_factors")

TEAM_MAP_REQUIRED_COLUMNS = [
    "team_id",
    "name",
    "team_name",
    "location_name",
    "franchise_name",
    "club_name",
    "short_name",
    "abbreviation",
    "team_code",
    "file_code",
    "first_year_of_play",
    "active",
    "all_star_status",
    "link",
    "league_id",
    "league_name",
    "league_link",
    "division_id",
    "division_name",
    "division_link",
    "sport_id",
    "sport_name",
    "sport_link",
    "venue_id",
    "venue_name",
    "venue_link",
    "spring_venue_id",
    "spring_venue_name",
    "spring_venue_link",
    "spring_league_id",
    "spring_league_name",
    "spring_league_link",
]

VENUE_MAP_REQUIRED_COLUMNS = [
    "venue_id",
    "venue_name",
    "active",
    "link",
    "season",
    "venue_code",
    "time_zone_id",
    "time_zone_offset",
    "time_zone_tz",
    "location_city",
    "location_state",
    "location_state_abbrev",
    "location_country",
    "location_postal_code",
    "location_address1",
    "location_address2",
    "location_default_coords",
    "latitude",
    "longitude",
    "phone_number",
    "field_capacity",
    "turf_type",
    "roof_type",
    "left_line",
    "left",
    "left_center",
    "center",
    "right_center",
    "right",
    "right_line",
    "retrosheet",
    "wind_out_direction",
]

RAW_REQUIRED_COLUMNS = [
    "Team",
    "Venue",
    "Year",
    "Park Factor",
    "wOBAcon",
    "xwOBAcon",
    "BACON",
    "xBACON",
    "HardHit",
    "R",
    "OBP",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "BB",
    "SO",
    "PA",
    "venue_id",
    "team_id",
]

CLEAN_COLUMNS = [
    "team_id",
    "venue_id",
    "bat_side",
    "condition",
    "Park Factor",
    "wOBAcon",
    "xwOBAcon",
    "HardHit",
    "R",
    "HR",
]

CLEAN_VALUE_COLUMNS = [
    "Park Factor",
    "wOBAcon",
    "xwOBAcon",
    "HardHit",
    "R",
    "HR",
]

PARK_FILES = [
    ("park_B_day.csv", "B", "day"),
    ("park_B_night.csv", "B", "night"),
    ("park_B_open_air.csv", "B", "open_air"),
    ("park_B_roof_closed.csv", "B", "roof_closed"),
    ("park_L_day.csv", "L", "day"),
    ("park_L_night.csv", "L", "night"),
    ("park_L_open_air.csv", "L", "open_air"),
    ("park_L_roof_closed.csv", "L", "roof_closed"),
    ("park_R_day.csv", "R", "day"),
    ("park_R_night.csv", "R", "night"),
    ("park_R_open_air.csv", "R", "open_air"),
    ("park_R_roof_closed.csv", "R", "roof_closed"),
]


def clean(value):
    if value is None:
        return ""
    return str(value).strip().strip("\ufeff")


def normalize_header(value):
    return clean(value)


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{path}: missing header row")

        reader.fieldnames = [normalize_header(col) for col in reader.fieldnames]

        rows = []
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                cleaned[normalize_header(key)] = clean(value)
            if any(cleaned.values()):
                rows.append(cleaned)

    return reader.fieldnames, rows


def require_file(path, errors):
    if not path.exists():
        errors.append(f"missing file: {path}")
        return False
    if not path.is_file():
        errors.append(f"not a file: {path}")
        return False
    return True


def require_columns(path, actual_columns, required_columns, errors):
    actual = set(actual_columns)
    missing = [col for col in required_columns if col not in actual]

    if missing:
        errors.append(f"{path}: missing required columns: {', '.join(missing)}")
        return False

    return True


def load_team_ids(errors):
    if not require_file(TEAM_MAP_PATH, errors):
        return set()

    try:
        columns, rows = read_csv_rows(TEAM_MAP_PATH)
    except Exception as exc:
        errors.append(f"{TEAM_MAP_PATH}: read error: {exc}")
        return set()

    if not require_columns(
        TEAM_MAP_PATH,
        columns,
        TEAM_MAP_REQUIRED_COLUMNS,
        errors,
    ):
        return set()

    team_ids = set()

    for line_number, row in enumerate(rows, start=2):
        team_id = clean(row.get("team_id"))
        if not team_id:
            errors.append(f"{TEAM_MAP_PATH}: row {line_number}: blank team_id")
            continue
        if not team_id.isdigit():
            errors.append(
                f"{TEAM_MAP_PATH}: row {line_number}: "
                f"non-integer team_id: {team_id}"
            )
            continue
        team_ids.add(team_id)

    return team_ids


def load_venue_ids(errors):
    if not require_file(VENUE_MAP_PATH, errors):
        return set()

    try:
        columns, rows = read_csv_rows(VENUE_MAP_PATH)
    except Exception as exc:
        errors.append(f"{VENUE_MAP_PATH}: read error: {exc}")
        return set()

    if not require_columns(
        VENUE_MAP_PATH,
        columns,
        VENUE_MAP_REQUIRED_COLUMNS,
        errors,
    ):
        return set()

    venue_ids = set()

    for line_number, row in enumerate(rows, start=2):
        venue_id = clean(row.get("venue_id"))
        if not venue_id:
            errors.append(f"{VENUE_MAP_PATH}: row {line_number}: blank venue_id")
            continue
        if not venue_id.isdigit():
            errors.append(
                f"{VENUE_MAP_PATH}: row {line_number}: "
                f"non-integer venue_id: {venue_id}"
            )
            continue
        venue_ids.add(venue_id)

    return venue_ids


def normalize_integer_string(value):
    value = clean(value).replace(",", "")
    if not value:
        return ""
    if not value.lstrip("-").isdigit():
        return None
    return str(int(value))


def build_clean_rows(path, bat_side, condition, team_ids, venue_ids, errors):
    try:
        columns, rows = read_csv_rows(path)
    except Exception as exc:
        errors.append(f"{path}: read error: {exc}")
        return []

    if not require_columns(path, columns, RAW_REQUIRED_COLUMNS, errors):
        return []

    clean_rows = []
    seen_team_venue = set()

    for line_number, row in enumerate(rows, start=2):
        team_id = normalize_integer_string(row.get("team_id"))
        venue_id = normalize_integer_string(row.get("venue_id"))

        if team_id is None:
            errors.append(
                f"{path}: row {line_number}: "
                f"invalid team_id: {row.get('team_id')}"
            )
            continue
        if venue_id is None:
            errors.append(
                f"{path}: row {line_number}: "
                f"invalid venue_id: {row.get('venue_id')}"
            )
            continue
        if team_id == "":
            errors.append(f"{path}: row {line_number}: blank team_id")
            continue
        if venue_id == "":
            errors.append(f"{path}: row {line_number}: blank venue_id")
            continue

        if team_id not in team_ids:
            errors.append(
                f"{path}: row {line_number}: team_id not found in "
                f"{TEAM_MAP_PATH}: {team_id}"
            )
            continue
        if venue_id not in venue_ids:
            errors.append(
                f"{path}: row {line_number}: venue_id not found in "
                f"{VENUE_MAP_PATH}: {venue_id}"
            )
            continue

        pair_key = (team_id, venue_id)
        if pair_key in seen_team_venue:
            errors.append(
                f"{path}: row {line_number}: "
                f"duplicate team_id/venue_id pair: {team_id}/{venue_id}"
            )
            continue
        seen_team_venue.add(pair_key)

        clean_row = {
            "team_id": team_id,
            "venue_id": venue_id,
            "bat_side": bat_side,
            "condition": condition,
        }

        for col in CLEAN_VALUE_COLUMNS:
            value = normalize_integer_string(row.get(col))
            if value is None:
                errors.append(
                    f"{path}: row {line_number}: "
                    f"invalid integer value in {col}: {row.get(col)}"
                )
                value = ""
            if value == "":
                errors.append(
                    f"{path}: row {line_number}: blank value in {col}"
                )
            clean_row[col] = value

        clean_rows.append(clean_row)

    return clean_rows


def write_clean_file(path, rows):
    out_path = path.with_name(path.stem + "_clean.csv")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CLEAN_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def main():
    errors = []
    outputs_to_write = []

    team_ids = load_team_ids(errors)
    venue_ids = load_venue_ids(errors)

    for filename, bat_side, condition in PARK_FILES:
        path = PARK_DIR / filename

        if not require_file(path, errors):
            continue

        clean_rows = build_clean_rows(
            path=path,
            bat_side=bat_side,
            condition=condition,
            team_ids=team_ids,
            venue_ids=venue_ids,
            errors=errors,
        )

        outputs_to_write.append((path, clean_rows))

    if errors:
        print(
            "FAILED: mlb_savant_park_data.py found errors.",
            file=sys.stderr,
        )
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    files_written = 0
    rows_written = 0

    for path, rows in outputs_to_write:
        out_path = write_clean_file(path, rows)
        files_written += 1
        rows_written += len(rows)
        print(f"WROTE: {out_path} ({len(rows)} rows)")

    print(
        f"OK: wrote {files_written} clean park-factor files "
        f"with {rows_written} total rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

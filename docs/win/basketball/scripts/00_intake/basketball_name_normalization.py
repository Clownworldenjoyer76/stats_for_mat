#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_name_normalization.py

import csv
import traceback
from pathlib import Path
from datetime import datetime, timezone

# =========================
# PATHS
# =========================

SPORTSBOOK_DIR  = Path("docs/win/basketball/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/basketball/00_intake/predictions")

MAP_FILES = {
    "NBA":   Path("docs/win/basketball/maps/team_map_nba.csv"),
    "NCAAM": Path("docs/win/basketball/maps/team_map_ncaam.csv"),
    "WNBA":  Path("docs/win/basketball/maps/team_map_wnba.csv"),
}

NO_MAP_DIR  = Path("docs/win/basketball/maps/no_map")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_basketball.csv"

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE     = ERROR_DIR / "name_normalization.txt"
SUMMARY_FILE = ERROR_DIR / "condensed_summary.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== basketball_name_normalization RUN {datetime.now(timezone.utc).isoformat()} ===\n")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")

# =========================
# LOAD TEAM MAPS
# =========================

# team_map[(league, alias_lower)] = canonical_team
team_map = {}

for league, map_file in MAP_FILES.items():
    if not map_file.exists():
        log(f"WARNING: {map_file} not found — skipping {league} mappings")
        continue
    with open(map_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias     = row.get("alias", "").strip().lower()
            canonical = row.get("canonical_team", "").strip()
            if alias and canonical:
                team_map[(league, alias)] = canonical

log(f"Team map loaded: {len(team_map)} entries")

# =========================
# TARGET FILES
# =========================

target_files = []

for subdir in ["nba", "ncaam", "wnba"]:
    for f in (SPORTSBOOK_DIR / subdir).glob("*.csv"):
        target_files.append(f)
    for f in (PREDICTIONS_DIR / subdir).glob("*.csv"):
        target_files.append(f)

log(f"Files to process: {len(target_files)}")

# =========================
# PROCESS FILES
# =========================

unmapped        = set()
files_processed = 0
rows_processed  = 0
rows_updated    = 0

try:
    for csv_file in sorted(target_files):
        try:
            files_processed += 1
            updated_rows = []
            modified     = False

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader     = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                for row in reader:
                    rows_processed += 1
                    league = row.get("league", "").strip().upper()

                    for side in ["home_team", "away_team"]:
                        team = row.get(side, "").strip()
                        if not team:
                            continue

                        key = (league, team.lower())

                        if key in team_map:
                            canonical = team_map[key]
                            if row.get(side) != canonical:
                                row[side] = canonical
                                modified  = True
                                rows_updated += 1
                        else:
                            unmapped.add((league, team))

                    updated_rows.append(row)

            if modified and fieldnames:
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(updated_rows)
                log(f"UPDATED: {csv_file}")
            else:
                log(f"NO CHANGES: {csv_file}")

        except Exception as e:
            log(f"ERROR processing {csv_file}: {e}\n{traceback.format_exc()}")

    # =========================
    # WRITE UNMAPPED
    # =========================

    with open(NO_MAP_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["league", "team"])
        for league, team in sorted(unmapped):
            writer.writerow([league, team])

    # =========================
    # WRITE CONDENSED SUMMARY
    # =========================

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== basketball_name_normalization SUMMARY {ts} ===\n")
        f.write(f"  Files processed:       {files_processed}\n")
        f.write(f"  Rows processed:        {rows_processed}\n")
        f.write(f"  Team name replacements:{rows_updated}\n")
        f.write(f"  Unmapped teams:        {len(unmapped)}\n")

    log("--- SUMMARY ---")
    log(f"Files processed: {files_processed}")
    log(f"Rows processed: {rows_processed}")
    log(f"Names normalized: {rows_updated}")
    log(f"Unmapped teams: {len(unmapped)}")
    log("STATUS: SUCCESS")

except Exception as e:
    log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
    log("STATUS: FAILED")
    raise

print("Basketball name normalization complete.")
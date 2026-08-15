#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_daily_games.py
"""Build canonical per-date game identity files from sportsbook history.

Identity is date + normalized home + normalized away. When multiple source IDs exist
for the same identity, a numeric ESPN event ID is preferred over a legacy/custom ID.
Conflicting numeric IDs for the same identity are fatal instead of being silently
carried forward as duplicate games.
"""
from __future__ import annotations

import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path

LEAGUES = {
    "nba": {"league_label": "NBA", "input_dir": Path("docs/win/basketball/00_intake/sportsbook/nba"), "output_dir": Path("docs/win/basketball/daily_games/nba")},
    "ncaam": {"league_label": "NCAAM", "input_dir": Path("docs/win/basketball/00_intake/sportsbook/ncaam"), "output_dir": Path("docs/win/basketball/daily_games/ncaam")},
    "wnba": {"league_label": "WNBA", "input_dir": Path("docs/win/basketball/00_intake/sportsbook/wnba"), "output_dir": Path("docs/win/basketball/daily_games/wnba")},
}
for cfg in LEAGUES.values():
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_daily_games.txt"
ALIAS_FILE = ERROR_DIR / "basketball_game_id_aliases.csv"

FIELDNAMES = ["sport", "league", "game_date", "game_time", "home_team", "away_team", "game_id"]
ALIAS_FIELDS = ["league", "game_date", "home_team", "away_team", "alias_game_id", "canonical_game_id", "source_file"]


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def build_row(row: dict) -> dict:
    return {k: clean(row.get(k)) for k in FIELDNAMES}


def identity_key(row: dict) -> tuple[str, str, str, str]:
    return (
        clean(row.get("league")).upper(),
        clean(row.get("game_date")),
        clean(row.get("home_team")).casefold(),
        clean(row.get("away_team")).casefold(),
    )


def id_rank(game_id: str) -> int:
    gid = clean(game_id)
    if not gid:
        return 0
    return 2 if gid.isdigit() else 1


def sort_key(row: dict):
    return (clean(row.get("game_date")), clean(row.get("game_time")), clean(row.get("home_team")), clean(row.get("away_team")), clean(row.get("game_id")))


def choose(existing: dict, candidate: dict, source_file: str, aliases: list[dict]) -> dict:
    old_id = clean(existing.get("game_id"))
    new_id = clean(candidate.get("game_id"))
    if old_id.isdigit() and new_id.isdigit() and old_id != new_id:
        raise ValueError(
            "Conflicting numeric game_ids for the same game identity: "
            f"{identity_key(existing)} -> {old_id} vs {new_id}"
        )

    if id_rank(new_id) > id_rank(old_id):
        canonical, alias = new_id, old_id
        chosen = candidate.copy()
        # Preserve a useful time if the preferred candidate is blank there.
        if not clean(chosen.get("game_time")):
            chosen["game_time"] = existing.get("game_time", "")
    else:
        canonical, alias = old_id, new_id
        chosen = existing
        if not clean(chosen.get("game_time")) and clean(candidate.get("game_time")):
            chosen = chosen.copy()
            chosen["game_time"] = candidate.get("game_time", "")

    if alias and alias != canonical:
        aliases.append({
            "league": clean(chosen.get("league")),
            "game_date": clean(chosen.get("game_date")),
            "home_team": clean(chosen.get("home_team")),
            "away_team": clean(chosen.get("away_team")),
            "alias_game_id": alias,
            "canonical_game_id": canonical,
            "source_file": source_file,
        })
    return chosen


def write_aliases(rows: list[dict]) -> None:
    # Always rewrite the diagnostic so stale aliases cannot persist.
    unique = {}
    for row in rows:
        key = tuple(row.get(c, "") for c in ALIAS_FIELDS[:-1])
        unique[key] = row
    with open(ALIAS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALIAS_FIELDS)
        writer.writeheader()
        writer.writerows(unique.values())


def main() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== basketball_daily_games RUN {datetime.now().isoformat()} ===\n")

    files_written = []
    aliases: list[dict] = []
    total_input_files = total_rows_read = total_identity_duplicates = total_rows_written = 0
    errors = 0

    for league_key, cfg in LEAGUES.items():
        league_label = cfg["league_label"]
        input_dir = cfg["input_dir"]
        output_dir = cfg["output_dir"]
        if not input_dir.exists():
            log(f"INPUT DIR NOT FOUND: {input_dir}")
            continue

        csv_files = sorted(input_dir.glob("*.csv"))
        total_input_files += len(csv_files)
        canonical: dict[tuple[str, str, str, str], dict] = {}

        for csv_path in csv_files:
            log(f"PROCESSING {csv_path}")
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for source in reader:
                        total_rows_read += 1
                        row = build_row(source)
                        if not row["game_date"] or not row["home_team"] or not row["away_team"]:
                            log(f"SKIPPED malformed identity row in {csv_path.name}")
                            continue
                        key = identity_key(row)
                        if key in canonical:
                            total_identity_duplicates += 1
                            canonical[key] = choose(canonical[key], row, str(csv_path), aliases)
                        else:
                            canonical[key] = row
            except Exception as exc:
                errors += 1
                log(f"ERROR processing {csv_path}: {exc}\n{traceback.format_exc()}")

        by_date: dict[str, list[dict]] = {}
        for row in canonical.values():
            by_date.setdefault(row["game_date"], []).append(row)
        for game_date, rows in sorted(by_date.items()):
            rows = sorted(rows, key=sort_key)
            out_path = output_dir / f"{game_date}_{league_label}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            files_written.append((str(out_path), len(rows)))
            total_rows_written += len(rows)
            log(f"WROTE {out_path} ({len(rows)} rows)")

    write_aliases(aliases)
    log("--- SUMMARY ---")
    log(f"Input files found: {total_input_files}")
    log(f"Rows read: {total_rows_read}")
    log(f"Duplicate identities resolved: {total_identity_duplicates}")
    log(f"ID aliases recorded: {len(aliases)}")
    log(f"Rows written: {total_rows_written}")
    log(f"Files written: {len(files_written)}")
    log(f"Errors: {errors}")
    log(f"STATUS: {'SUCCESS' if errors == 0 else 'FAILED'}")

    if errors:
        sys.exit(1)
    print("Basketball daily games complete.")


if __name__ == "__main__":
    main()

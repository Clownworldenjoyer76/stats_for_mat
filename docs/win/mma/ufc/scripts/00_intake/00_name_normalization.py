"""
00_name_normalization.py

Normalizes fighter names in odds and predictions intake CSV files
using the fighter name map at docs/win/mma/ufc/mappings/fighter_name_map.csv.

Reads:
    docs/win/mma/ufc/00_intake/sportsbook/*_ufc_odds.csv
    docs/win/mma/ufc/00_intake/predictions/*_ufc_predictions.csv
    docs/win/mma/ufc/manual_files/*_ufc.csv

Normalizes fighter_1 and fighter_2 in place.
Outputs unmapped names to docs/win/mma/ufc/mappings/no_map_fighter_name.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

NAME_MAP_PATH = Path("docs/win/mma/ufc/mappings/fighter_name_map.csv")
NO_MAP_PATH = Path("docs/win/mma/ufc/mappings/no_map_fighter_name.csv")

INTAKE_FILES = [
    (Path("docs/win/mma/ufc/00_intake/sportsbook"), "*_ufc_odds.csv"),
    (Path("docs/win/mma/ufc/00_intake/predictions"), "*_ufc_predictions.csv"),
    (Path("docs/win/mma/ufc/manual_files"), "*_ufc.csv"),
]


def load_name_map(path: Path) -> dict[str, str]:
    name_map: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = row["alias"].strip()
            canonical = row["canonical"].strip()
            if alias and canonical:
                name_map[alias] = canonical
    return name_map


def normalize_name(name: str, name_map: dict[str, str], lower_map: dict[str, str]) -> tuple[str, bool]:
    if name in name_map:
        return name_map[name], True
    if name.lower() in lower_map:
        return lower_map[name.lower()], True
    return name, False


def normalize_file(filepath: Path, name_map: dict[str, str], lower_map: dict[str, str]) -> list[str]:
    with filepath.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        print(f"  SKIP {filepath.name} — no rows")
        return []

    unmapped = []
    for row in rows:
        for col in ["fighter_1", "fighter_2"]:
            if col not in row:
                continue
            original = row[col].strip()
            normalized, found = normalize_name(original, name_map, lower_map)
            if not found:
                unmapped.append(original)
            row[col] = normalized

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  OK {filepath.name} — {len(rows)} rows normalized, {len(unmapped)} unmapped")
    return unmapped


def write_no_map(unmapped: list[str]) -> None:
    NO_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    unique = []
    for name in unmapped:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    with NO_MAP_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unmapped_name"])
        for name in sorted(unique):
            writer.writerow([name])

    print(f"\nUnmapped names written to {NO_MAP_PATH} ({len(unique)} unique)")


def main() -> int:
    if not NAME_MAP_PATH.exists():
        print(f"ERROR: Name map not found at {NAME_MAP_PATH}")
        return 1

    name_map = load_name_map(NAME_MAP_PATH)
    lower_map = {k.lower(): v for k, v in name_map.items()}
    print(f"Loaded {len(name_map)} name mappings\n")

    files = []
    for directory, pattern in INTAKE_FILES:
        files.extend(sorted(directory.glob(pattern)))

    if not files:
        print("No intake files found")
        return 1

    print(f"Found {len(files)} files to normalize")
    all_unmapped = []
    for filepath in files:
        unmapped = normalize_file(filepath, name_map, lower_map)
        all_unmapped.extend(unmapped)

    if all_unmapped:
        write_no_map(all_unmapped)
    else:
        print("\nAll names matched — no unmapped names.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

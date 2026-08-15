#!/usr/bin/env python3
"""Season-aware launcher for the ESPN basketball odds collector.

The implementation remains in basketball_odds_core.py. Normal production runs only
query leagues that are in season for the current New York date. Set
BASKETBALL_FORCE_ALL_LEAGUES=1 (or true/yes/on) to run every league explicitly.

After a successful collection, raw sportsbook files are normalized to one row per
(date, home team, away team) matchup. Numeric ESPN game IDs outrank legacy IDs;
legacy alias rows are merged into the canonical row and removed. Conflicting numeric
IDs for the same matchup are a hard failure.
"""
from __future__ import annotations

import csv
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
CORE_PATH = Path(__file__).with_name("basketball_odds_core.py")
BASE = Path("docs/win/basketball")
SPORTSBOOK_BASE = BASE / "00_intake/sportsbook"
ERROR_DIR = BASE / "errors/00_intake"
ALIAS_REPORT = ERROR_DIR / "basketball_sportsbook_alias_cleanup.csv"
LEAGUE_LABELS = {"nba": "NBA", "ncaam": "NCAAM", "wnba": "WNBA"}
ALIAS_FIELDS = [
    "league", "source_file", "game_date", "home_team", "away_team",
    "alias_game_id", "canonical_game_id",
]


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def in_season(league: str, now: datetime) -> bool:
    league = league.lower()
    if league in {"nba", "ncaam"}:
        # Mirrors calculate_rolling_bias.py season boundaries:
        # Sep 1 through Jul 1, with Jul 2-Aug 31 treated as offseason.
        return now.month >= 9 or now.month <= 6 or (now.month == 7 and now.day == 1)
    if league == "wnba":
        # Operational WNBA window; full/offseason runs remain available via override.
        return 5 <= now.month <= 10
    return True


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_odds_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load odds core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def matchup_key(row: dict) -> tuple[str, str, str]:
    return (
        clean(row.get("game_date")),
        clean(row.get("home_team")).casefold(),
        clean(row.get("away_team")).casefold(),
    )


def id_rank(game_id: str) -> int:
    game_id = clean(game_id)
    if game_id.isdigit():
        return 2
    if game_id:
        return 1
    return 0


def row_score(row: dict) -> tuple[int, str]:
    # Prefer the row carrying the most usable data, then the latest odds timestamp.
    populated = sum(1 for key, value in row.items() if key != "game_id" and clean(value))
    return populated, clean(row.get("odds_last_update"))


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def collapse_file(path: Path, league: str) -> tuple[int, list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or not fieldnames:
        return 0, []

    grouped: dict[tuple[str, str, str], list[tuple[int, dict]]] = {}
    passthrough: list[tuple[int, dict]] = []
    for index, row in enumerate(rows):
        key = matchup_key(row)
        if not all(key):
            passthrough.append((index, row))
            continue
        grouped.setdefault(key, []).append((index, row))

    output: list[tuple[int, dict]] = list(passthrough)
    aliases: list[dict] = []
    removed = 0

    for key, items in grouped.items():
        group_rows = [row for _, row in items]
        numeric_ids = sorted({clean(r.get("game_id")) for r in group_rows if clean(r.get("game_id")).isdigit()})
        if len(numeric_ids) > 1:
            raise RuntimeError(
                f"Conflicting numeric sportsbook game IDs for {league.upper()} "
                f"{key[0]} {group_rows[0].get('away_team')} at {group_rows[0].get('home_team')}: "
                + ", ".join(numeric_ids)
            )

        canonical_id = numeric_ids[0] if numeric_ids else max(
            (clean(r.get("game_id")) for r in group_rows),
            key=lambda gid: (id_rank(gid), gid),
            default="",
        )
        best_index, best = max(items, key=lambda item: (id_rank(clean(item[1].get("game_id"))), row_score(item[1])))
        merged = dict(best)
        merged["game_id"] = canonical_id

        # Fill only blanks; never replace a populated canonical value with alias data.
        for _, row in sorted(items, key=lambda item: row_score(item[1]), reverse=True):
            for field in fieldnames:
                if not clean(merged.get(field)) and clean(row.get(field)):
                    merged[field] = row[field]
        merged["game_id"] = canonical_id
        output.append((min(index for index, _ in items), merged))

        if len(items) > 1:
            removed += len(items) - 1
            for _, row in items:
                alias_id = clean(row.get("game_id"))
                if alias_id != canonical_id:
                    aliases.append({
                        "league": league,
                        "source_file": str(path),
                        "game_date": clean(row.get("game_date")),
                        "home_team": clean(row.get("home_team")),
                        "away_team": clean(row.get("away_team")),
                        "alias_game_id": alias_id,
                        "canonical_game_id": canonical_id,
                    })

    if removed:
        output.sort(key=lambda item: item[0])
        atomic_write_csv(path, fieldnames, [row for _, row in output])
    return removed, aliases


def cleanup_sportsbook_aliases(core) -> tuple[int, int]:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    alias_rows: list[dict] = []
    changed_files = 0
    removed_rows = 0

    for league in LEAGUE_LABELS:
        folder = SPORTSBOOK_BASE / league
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            removed, aliases = collapse_file(path, league)
            if removed:
                changed_files += 1
                removed_rows += removed
            alias_rows.extend(aliases)

    atomic_write_csv(ALIAS_REPORT, ALIAS_FIELDS, alias_rows)
    core.log(
        "SPORTSBOOK ALIAS CLEANUP | "
        f"changed_files={changed_files} | removed_alias_rows={removed_rows} | "
        f"aliases={len(alias_rows)} | report={ALIAS_REPORT}"
    )
    return changed_files, removed_rows


def main() -> None:
    core = load_core()
    now = datetime.now(NY)
    force_all = truthy(os.getenv("BASKETBALL_FORCE_ALL_LEAGUES"))

    original = dict(core.LEAGUES)
    if force_all:
        active = original
    else:
        active = {k: v for k, v in original.items() if in_season(k, now)}

    skipped = sorted(set(original) - set(active))
    core.LEAGUES = active
    core.log(
        "SEASON GATE | "
        f"date={now.strftime('%Y_%m_%d')} | active={','.join(active) or 'none'} | "
        f"skipped={','.join(skipped) or 'none'} | force_all={int(force_all)}"
    )

    if active:
        core.main()

        # The core historically reported STATUS: SUCCESS even when per-event HTTP/build
        # errors were counted. Convert a non-zero current-run error count into a failed
        # process so GitHub Actions cannot be falsely green.
        text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
        errors = 0
        for line in reversed(text.splitlines()):
            if " | Errors: " in line:
                try:
                    errors = int(line.rsplit("Errors:", 1)[1].strip())
                except ValueError:
                    errors = 1
                break
        if "STATUS: FAILED" in text or errors:
            raise SystemExit(1)
    else:
        core.log("SEASON GATE: no leagues in season; collector skipped")

    try:
        cleanup_sportsbook_aliases(core)
    except Exception as exc:
        core.log(f"SPORTSBOOK ALIAS CLEANUP FAILED: {exc}")
        core.log("STATUS: FAILED")
        raise SystemExit(1) from exc

    if not active:
        core.log("STATUS: SUCCESS (no leagues in season)")


if __name__ == "__main__":
    main()

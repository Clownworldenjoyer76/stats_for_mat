#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/build_games_list.py
#
# Runs after scrape_mlb_raw.py and odds_parse.py.
# Joins mlb_raw to sportsbook to produce an authoritative {date}_games.csv.
#
# Matching rules:
#   1. Same home/away teams.
#   2. Never reuse a sportsbook row / game_id.
#   3. If same matchup has exactly one raw row and exactly one sportsbook row:
#      match them even if MLB raw time is stale, and log the time difference.
#   4. If same matchup has multiple raw rows and the same number of sportsbook rows:
#      pair by chronological order. This handles normal doubleheaders.
#   5. Otherwise, match to the closest unused sportsbook time within threshold.

import csv
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

MLB_RAW_DIR = Path("docs/win/baseball/mlb/00_intake/mlb_raw")
BOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")
MAPS_DIR = Path("docs/win/baseball/mlb/maps")

OUT_DIR = Path("docs/win/baseball/mlb/00_intake/games")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "build_games_list.txt"

MAX_TIME_DIFF_MINUTES = 90

OUTPUT_HEADER = [
    "gamePk",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "venue_id",
    "doubleheader",
    "gameNumber",
    "home_pitcher_id",
    "away_pitcher_id",
    "day_night",
]


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def norm(s):
    """Lowercase, strip punctuation, collapse spaces for team name matching."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_csv(path: Path) -> list:
    if not path.exists():
        log(f"MISSING: {path}", "WARN")
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_team_map() -> dict:
    """Returns dict: norm(team_name) -> team_id."""
    rows = load_csv(MAPS_DIR / "mlb_team_ids.csv")
    m = {}

    for r in rows:
        tid = r.get("team_id", "").strip()

        if not tid:
            continue

        for col in [
            "name",
            "team_name",
            "short_name",
            "club_name",
            "franchise_name",
        ]:
            val = norm(r.get(col, ""))

            if val:
                m[val] = tid

    return m


def build_id_to_name_map(rows: list) -> dict:
    """Returns dict: team_id -> full name."""
    return {
        r.get("team_id", "").strip(): r.get("name", "").strip()
        for r in rows
    }


def parse_int(value, default=0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def utc_to_local_datetime(
    utc_str: str,
    tz_id: str = "America/New_York",
):
    """Convert MLB UTC ISO time string to local aware datetime."""
    try:
        dt = datetime.fromisoformat(str(utc_str).replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo(tz_id))
    except Exception:
        return None


def parse_book_datetime(
    date_str: str,
    time_str: str,
    tz_id: str = "America/New_York",
):
    """Parse sportsbook date + HH:MM:SS as local aware datetime."""
    try:
        date_clean = str(date_str).replace("_", "-").strip()
        time_clean = str(time_str).strip()
        dt = datetime.strptime(
            f"{date_clean} {time_clean}",
            "%Y-%m-%d %H:%M:%S",
        )
        return dt.replace(tzinfo=ZoneInfo(tz_id))
    except Exception:
        return None


def sort_dt_key(entry: dict):
    dt = entry.get("local_dt") or entry.get("book_dt")

    if dt is None:
        return (
            1,
            datetime.max.replace(
                tzinfo=ZoneInfo("America/New_York")
            ),
        )

    return 0, dt


def minutes_between(a, b):
    if a is None or b is None:
        return None

    return abs((a - b).total_seconds()) / 60.0


def make_output_row(raw_entry: dict, book_entry: dict) -> dict:
    r = raw_entry["row"]
    b = book_entry["row"]

    return {
        "gamePk": r.get("gamePk", ""),
        "game_id": b.get("game_id", ""),
        "game_date": r.get("game_date", ""),
        "game_time": b.get("game_time", ""),
        "home_team": b.get("home_team", ""),
        "away_team": b.get("away_team", ""),
        "home_team_id": r.get("home_team_id", ""),
        "away_team_id": r.get("away_team_id", ""),
        "venue_id": r.get("venue_id", ""),
        "doubleheader": r.get("doubleheader", "N"),
        "gameNumber": r.get("gameNumber", "1"),
        "home_pitcher_id": r.get("home_pitcher_id", ""),
        "away_pitcher_id": r.get("away_pitcher_id", ""),
        "day_night": r.get("day_night", ""),
    }


def write_games_file(
    out_path: Path,
    output_rows: list,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_HEADER,
        )
        writer.writeheader()
        writer.writerows(output_rows)


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(
    date_str: str,
    team_map: dict,
    id_to_name: dict,
    summary: dict,
) -> None:
    raw_path = MLB_RAW_DIR / f"{date_str}_mlb_raw.csv"
    book_path = BOOK_DIR / f"{date_str}_MLB.csv"

    raw_rows = load_csv(raw_path)
    book_rows = load_csv(book_path)

    if not raw_rows:
        log(
            f"{date_str} | no mlb_raw — skipping",
            "WARN",
        )
        summary["skipped"] += 1
        return

    if not book_rows:
        log(
            f"{date_str} | no sportsbook — skipping",
            "WARN",
        )
        summary["skipped"] += 1
        return

    raw_groups = {}
    raw_key_order = []

    for r in raw_rows:
        home_tid = r.get("home_team_id", "").strip()
        away_tid = r.get("away_team_id", "").strip()

        home_name = id_to_name.get(home_tid, "")
        away_name = id_to_name.get(away_tid, "")

        key = (
            norm(home_name),
            norm(away_name),
        )

        if not key[0] or not key[1]:
            log(
                f"{date_str} | raw gamePk={r.get('gamePk', '')} "
                f"missing team name "
                f"home_team_id={home_tid} "
                f"away_team_id={away_tid}",
                "WARN",
            )
            continue

        if key not in raw_groups:
            raw_groups[key] = []
            raw_key_order.append(key)

        raw_groups[key].append(
            {
                "row": r,
                "key": key,
                "home_name": home_name,
                "away_name": away_name,
                "local_dt": utc_to_local_datetime(
                    r.get("game_time", "")
                ),
                "game_number": parse_int(
                    r.get("gameNumber", "1"),
                    1,
                ),
            }
        )

    book_groups = {}

    for idx, b in enumerate(book_rows):
        key = (
            norm(b.get("home_team", "")),
            norm(b.get("away_team", "")),
        )

        if key not in book_groups:
            book_groups[key] = []

        book_groups[key].append(
            {
                "row": b,
                "key": key,
                "book_dt": parse_book_datetime(
                    date_str,
                    b.get("game_time", ""),
                ),
                "used": False,
                "index": idx,
            }
        )

    output_rows = []
    matched = 0
    unmatched = 0

    for key in raw_key_order:
        raws = raw_groups.get(key, [])
        books = book_groups.get(key, [])

        matchup_label = (
            f"{raws[0]['away_name']} @ "
            f"{raws[0]['home_name']}"
            if raws
            else str(key)
        )

        if not books:
            for raw_entry in raws:
                log(
                    f"{date_str} | "
                    f"UNMATCHED no sportsbook rows: "
                    f"{matchup_label} "
                    f"gamePk="
                    f"{raw_entry['row'].get('gamePk', '')}",
                    "WARN",
                )
                unmatched += 1

            continue

        unused_books = [
            b
            for b in books
            if not b["used"]
        ]

        if len(raws) == 1 and len(unused_books) == 1:
            raw_entry = raws[0]
            book_entry = unused_books[0]
            book_entry["used"] = True

            output_rows.append(
                make_output_row(
                    raw_entry,
                    book_entry,
                )
            )
            matched += 1

            diff = minutes_between(
                raw_entry.get("local_dt"),
                book_entry.get("book_dt"),
            )
            diff_text = (
                ""
                if diff is None
                else f" diff_minutes={round(diff, 1)}"
            )

            level = "INFO"
            label = "MATCHED one-to-one"

            if (
                diff is not None
                and diff > MAX_TIME_DIFF_MINUTES
            ):
                level = "WARN"
                label = (
                    "MATCHED one-to-one "
                    "with time mismatch"
                )

            log(
                f"{date_str} | {label}: "
                f"{matchup_label} "
                f"gamePk="
                f"{raw_entry['row'].get('gamePk', '')} "
                f"gameNumber="
                f"{raw_entry['row'].get('gameNumber', '')} "
                f"game_id="
                f"{book_entry['row'].get('game_id', '')}"
                f"{diff_text}",
                level,
            )

            continue

        if (
            len(raws) > 1
            and len(unused_books) > 1
            and len(raws) == len(unused_books)
        ):
            sorted_raws = sorted(
                raws,
                key=lambda x: (
                    sort_dt_key(x),
                    x["game_number"],
                    x["row"].get("gamePk", ""),
                ),
            )
            sorted_books = sorted(
                unused_books,
                key=lambda x: (
                    sort_dt_key(x),
                    x["index"],
                ),
            )

            log(
                f"{date_str} | "
                f"ORDER MATCH duplicate matchup: "
                f"{matchup_label} "
                f"raw_count={len(sorted_raws)} "
                f"sportsbook_count={len(sorted_books)}"
            )

            for raw_entry, book_entry in zip(
                sorted_raws,
                sorted_books,
            ):
                book_entry["used"] = True

                output_rows.append(
                    make_output_row(
                        raw_entry,
                        book_entry,
                    )
                )
                matched += 1

                diff = minutes_between(
                    raw_entry.get("local_dt"),
                    book_entry.get("book_dt"),
                )
                diff_text = (
                    ""
                    if diff is None
                    else f" diff_minutes={round(diff, 1)}"
                )

                log(
                    f"{date_str} | MATCHED order: "
                    f"{matchup_label} "
                    f"gamePk="
                    f"{raw_entry['row'].get('gamePk', '')} "
                    f"gameNumber="
                    f"{raw_entry['row'].get('gameNumber', '')} "
                    f"game_id="
                    f"{book_entry['row'].get('game_id', '')}"
                    f"{diff_text}"
                )

            continue

        sorted_raws = sorted(
            raws,
            key=lambda x: (
                sort_dt_key(x),
                x["game_number"],
                x["row"].get("gamePk", ""),
            ),
        )

        for raw_entry in sorted_raws:
            available_books = [
                b
                for b in books
                if not b["used"]
            ]

            if not available_books:
                log(
                    f"{date_str} | "
                    f"UNMATCHED no unused sportsbook row: "
                    f"{matchup_label} "
                    f"gamePk="
                    f"{raw_entry['row'].get('gamePk', '')}",
                    "WARN",
                )
                unmatched += 1
                continue

            scored = []

            for book_entry in available_books:
                diff = minutes_between(
                    raw_entry.get("local_dt"),
                    book_entry.get("book_dt"),
                )
                scored.append(
                    (
                        diff,
                        book_entry,
                    )
                )

            scored_valid = [
                x
                for x in scored
                if x[0] is not None
            ]

            selected = None
            selected_diff = None

            if scored_valid:
                selected_diff, selected = min(
                    scored_valid,
                    key=lambda x: x[0],
                )

                if (
                    selected_diff
                    > MAX_TIME_DIFF_MINUTES
                ):
                    selected = None

            if selected is None:
                diffs_text = ", ".join(
                    [
                        f"{x[1]['row'].get('game_time', '')}:"
                        f"{'NA' if x[0] is None else round(x[0], 1)}"
                        for x in scored
                    ]
                )

                log(
                    f"{date_str} | "
                    f"UNMATCHED time threshold: "
                    f"{matchup_label} "
                    f"gamePk="
                    f"{raw_entry['row'].get('gamePk', '')} "
                    f"candidate_diffs={diffs_text}",
                    "WARN",
                )
                unmatched += 1
                continue

            selected["used"] = True

            output_rows.append(
                make_output_row(
                    raw_entry,
                    selected,
                )
            )
            matched += 1

            diff_text = (
                ""
                if selected_diff is None
                else (
                    f" diff_minutes="
                    f"{round(selected_diff, 1)}"
                )
            )

            log(
                f"{date_str} | MATCHED closest: "
                f"{matchup_label} "
                f"gamePk="
                f"{raw_entry['row'].get('gamePk', '')} "
                f"gameNumber="
                f"{raw_entry['row'].get('gameNumber', '')} "
                f"game_id="
                f"{selected['row'].get('game_id', '')}"
                f"{diff_text}"
            )

    for key, books in book_groups.items():
        unused = [
            b
            for b in books
            if not b["used"]
        ]

        for book_entry in unused:
            b = book_entry["row"]

            log(
                f"{date_str} | UNUSED sportsbook row: "
                f"{b.get('away_team', '')} @ "
                f"{b.get('home_team', '')} "
                f"game_id={b.get('game_id', '')} "
                f"game_time={b.get('game_time', '')}",
                "WARN",
            )

    seen_game_ids = {}
    duplicate_output_game_ids = 0

    for row in output_rows:
        gid = row.get("game_id", "")

        if not gid:
            continue

        if gid in seen_game_ids:
            duplicate_output_game_ids += 1

            log(
                f"{date_str} | "
                f"DUPLICATE OUTPUT game_id={gid} "
                f"first_gamePk={seen_game_ids[gid]} "
                f"second_gamePk="
                f"{row.get('gamePk', '')}",
                "ERROR",
            )
        else:
            seen_game_ids[gid] = row.get(
                "gamePk",
                "",
            )

    if duplicate_output_game_ids:
        summary["errors"] += (
            duplicate_output_game_ids
        )

    log(
        f"{date_str} | "
        f"matched={matched} "
        f"unmatched={unmatched}"
    )

    summary["total_matched"] += matched
    summary["total_unmatched"] += unmatched

    if output_rows:
        output_rows = sorted(
            output_rows,
            key=lambda r: (
                r.get("game_date", ""),
                r.get("game_time", ""),
                r.get("home_team", ""),
                r.get("away_team", ""),
                parse_int(
                    r.get("gameNumber", "1"),
                    1,
                ),
            ),
        )

        out_path = OUT_DIR / f"{date_str}_games.csv"
        write_games_file(
            out_path,
            output_rows,
        )

        log(
            f"{date_str} | "
            f"WROTE: {out_path} "
            f"({len(output_rows)} games)"
        )
        summary["files_written"] += 1
    else:
        log(
            f"{date_str} | "
            f"no matched games — file not written",
            "WARN",
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"=== build_games_list RUN {_now()} ===\n"
        )

    summary = {
        "files_written": 0,
        "total_matched": 0,
        "total_unmatched": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        team_map = load_team_map()
        team_rows = load_csv(
            MAPS_DIR / "mlb_team_ids.csv"
        )
        id_to_name = build_id_to_name_map(
            team_rows
        )

        log(
            f"Team map loaded: "
            f"{len(team_map)} entries | "
            f"id_to_name: "
            f"{len(id_to_name)} entries"
        )

        raw_files = sorted(
            MLB_RAW_DIR.glob("*_mlb_raw.csv")
        )
        log(
            f"mlb_raw files found: "
            f"{len(raw_files)}"
        )

        for rf in raw_files:
            date_str = rf.stem.replace(
                "_mlb_raw",
                "",
            )

            try:
                process_date(
                    date_str,
                    team_map,
                    id_to_name,
                    summary,
                )
            except Exception as e:
                log(
                    f"{date_str} FAILED: "
                    f"{e}\n"
                    f"{traceback.format_exc()}",
                    "ERROR",
                )
                summary["errors"] += 1

    except Exception as e:
        log(
            f"FATAL: {e}\n"
            f"{traceback.format_exc()}",
            "ERROR",
        )
        summary["errors"] += 1

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
        (
            f"  files_written   : "
            f"{summary['files_written']}"
        ),
        (
            f"  total_matched   : "
            f"{summary['total_matched']}"
        ),
        (
            f"  total_unmatched : "
            f"{summary['total_unmatched']}"
        ),
        (
            f"  skipped         : "
            f"{summary['skipped']}"
        ),
        (
            f"  errors          : "
            f"{summary['errors']}"
        ),
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"build_games_list complete. "
        f"{summary['files_written']} "
        f"files written. "
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/game_id_pred.py
#
# Injects game_id into prediction files using:
#   docs/win/baseball/mlb/00_intake/games/{date}_games.csv
#
# Input:
#   docs/win/baseball/mlb/00_intake/predictions/{date}_MLB.csv
#   docs/win/baseball/mlb/00_intake/games/{date}_games.csv
#   docs/win/baseball/mlb/00_intake/sportsbook/{date}_MLB.csv
#
# Output:
#   docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/{date}_MLB.csv
#   docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/{date}_MLB.csv
#
# Rejections:
#   docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id/rejections/{date}_unmatched_predictions.csv
#
# Matching rules:
#   1. A prediction row must have a matching sportsbook row to be eligible for game_id assignment.
#   2. Prediction rows without a matching sportsbook row are non-fatal rejections.
#   3. Non-fatal rejected rows are written to rejection CSV and omitted from pred_with_game_id output.
#   4. Eligible prediction rows are matched to games rows by same home/away teams.
#   5. Never reuse a games row / game_id.
#   6. If same matchup has exactly one eligible prediction row and exactly one games row:
#      match them even if source times differ, and log the time difference.
#   7. If same matchup has multiple eligible prediction rows and the same number of games rows:
#      pair by chronological order. This handles doubleheaders.
#   8. Otherwise, match to the closest unused games row time within threshold.
#
# Step 2 hardening:
#   - Never write blank game_id to pred_with_game_id output.
#   - Non-fatal reject prediction rows when the matching sportsbook row is missing.
#   - Non-fatal reject prediction rows when the games file is missing/empty for current or future dates.
#   - Hard-fail if a past-date games file is missing/empty.
#   - Hard-fail if an eligible prediction row cannot be assigned a game_id for a past date with games data.
#   - Hard-fail if duplicate game_id appears in output.
#   - Hard-fail if matched eligible output row count differs from eligible prediction row count.
#   - Write rejected prediction rows to rejection CSV.
#   - Print rejected prediction rows to stdout so GitHub Actions logs show the reason.
#   - Print fatal exception/traceback directly to stdout so failed GitHub runs show the actual error.

import csv
import math
import os
import re
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path


PRED_DIR = Path("docs/win/baseball/mlb/00_intake/predictions")
GAMES_DIR = Path("docs/win/baseball/mlb/00_intake/games")
BOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")

OUT_DIR = Path("docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id")
REJECTION_DIR = OUT_DIR / "rejections"
ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REJECTION_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "game_id_pred.txt"

MAX_TIME_DIFF_MINUTES = 90

OUTPUT_HEADER = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
]

REJECTION_HEADER = OUTPUT_HEADER + [
    "reject_reason",
    "reject_action",
    "fatal",
    "sportsbook_present",
    "sportsbook_match_detail",
    "candidate_games",
    "source_csv_row",
]

REQUIRED_PRED_COLS = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
]

REQUIRED_GAMES_COLS = [
    "gamePk",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "gameNumber",
]

REQUIRED_BOOK_COLS = [
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]


# ─────────────────────────────────────────────
# LOGGING / FAILURE
# ─────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def print_block(title: str, lines: list[str]):
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)
    for line in lines:
        print(line)
    print("=" * 80)
    print("")


def fail(msg: str):
    log(f"FATAL VALIDATION ERROR: {msg}", "ERROR")
    print_block("FATAL VALIDATION ERROR IN game_id_pred.py", [msg])
    raise RuntimeError(msg)


# ─────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────

def duplicate_columns(header):
    seen = set()
    dupes = []

    for col in header:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)

    return dupes


def assert_no_duplicate_columns(header, label):
    dupes = duplicate_columns(header)

    if dupes:
        fail(f"{label} has duplicate columns: {dupes}")


def assert_required_columns(path: Path, header: list[str], required_cols: list[str], label: str):
    missing = [col for col in required_cols if col not in header]

    if missing:
        fail(f"{label} missing required columns in {path}: {missing}")


def load_csv(path: Path, required_cols=None, label=None, required_file=False) -> list[dict]:
    if not path.exists():
        msg = f"MISSING: {path}"

        if required_file:
            fail(msg)

        log(msg, "WARN")
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        assert_no_duplicate_columns(header, f"{label or path} input")

        if required_cols:
            assert_required_columns(path, header, required_cols, label or str(path))

        return list(reader)


def write_csv(path: Path, header: list[str], rows: list[dict]):
    assert_no_duplicate_columns(header, f"{path} output")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"WROTE: {path} | rows={len(rows)}")


def write_output_csv(date_str: str, header: list[str], rows: list[dict], summary: dict) -> Path:
    out_path = OUT_DIR / f"{date_str}_MLB.csv"
    write_csv(out_path, header, rows)
    summary["files_written"] += 1
    return out_path


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces for team name matching."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_int(value, default=0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def clean_date(value: str) -> str:
    return str(value or "").strip().replace("_", "-")


def parse_date_value(value: str):
    value = clean_date(value)

    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def current_cutoff_date():
    env_date = os.environ.get("DATE", "")
    parsed_env_date = parse_date_value(env_date)

    if parsed_env_date is not None:
        return parsed_env_date

    return datetime.now(timezone.utc).date()


def is_current_or_future_date(date_str: str) -> bool:
    parsed_date = parse_date_value(date_str)

    if parsed_date is None:
        fail(f"Could not parse date for current/future validation: {date_str}")

    return parsed_date >= current_cutoff_date()


def parse_prediction_datetime(date_str: str, time_str: str):
    date_clean = clean_date(date_str)
    time_clean = re.sub(r"\s+", " ", str(time_str or "").strip()).upper()

    if not date_clean or not time_clean:
        return None

    formats = [
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    value = f"{date_clean} {time_clean}"

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def parse_games_datetime(date_str: str, time_str: str):
    date_clean = clean_date(date_str)
    time_clean = str(time_str or "").strip()

    if not date_clean or not time_clean:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M:%S %p",
    ]

    value = f"{date_clean} {time_clean}"

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def parse_book_datetime(date_str: str, time_str: str):
    return parse_games_datetime(date_str, time_str)


def minutes_between(a, b):
    if a is None or b is None:
        return None

    return abs((a - b).total_seconds()) / 60.0


def dt_sort_value(dt):
    if dt is None:
        return math.inf

    return dt.timestamp()


def make_output_row(pred_entry: dict, game_id: str) -> dict:
    p = pred_entry["row"]

    return {
        "game_id": game_id,
        "sport": p.get("sport", ""),
        "league": p.get("league", ""),
        "game_date": p.get("game_date", ""),
        "game_time": p.get("game_time", ""),
        "home_team": p.get("home_team", ""),
        "away_team": p.get("away_team", ""),
        "home_pitcher": p.get("home_pitcher", ""),
        "away_pitcher": p.get("away_pitcher", ""),
        "home_prob": p.get("home_prob", ""),
        "away_prob": p.get("away_prob", ""),
        "away_projected_runs": p.get("away_projected_runs", ""),
        "home_projected_runs": p.get("home_projected_runs", ""),
        "total_projected_runs": p.get("total_projected_runs", ""),
    }


def make_rejection_row(
    pred_entry: dict,
    reason: str,
    candidate_games: str = "",
    fatal: bool = True,
    sportsbook_present: bool = False,
    sportsbook_match_detail: str = "",
) -> dict:
    row = make_output_row(pred_entry, "")
    row["reject_reason"] = reason
    row["reject_action"] = "hard_failure" if fatal else "omit_and_continue"
    row["fatal"] = "1" if fatal else "0"
    row["sportsbook_present"] = "1" if sportsbook_present else "0"
    row["sportsbook_match_detail"] = sportsbook_match_detail
    row["candidate_games"] = candidate_games
    row["source_csv_row"] = str(pred_entry.get("csv_row", ""))
    return row


def matchup_label(row: dict) -> str:
    return f"{row.get('away_team', '')} @ {row.get('home_team', '')}"


def describe_game_entry(game_entry: dict) -> str:
    g = game_entry["row"]
    diff_fields = [
        f"game_id={g.get('game_id', '')}",
        f"gamePk={g.get('gamePk', '')}",
        f"time={g.get('game_time', '')}",
        f"gameNumber={g.get('gameNumber', '')}",
        f"away={g.get('away_team', '')}",
        f"home={g.get('home_team', '')}",
    ]
    return "|".join(diff_fields)


def describe_book_entry(book_entry: dict) -> str:
    b = book_entry["row"]
    diff_fields = [
        f"game_id={b.get('game_id', '')}",
        f"time={b.get('game_time', '')}",
        f"away={b.get('away_team', '')}",
        f"home={b.get('home_team', '')}",
    ]
    return "|".join(diff_fields)


def describe_candidates(scored: list[tuple]) -> str:
    parts = []

    for diff, game_entry in scored:
        diff_text = "NA" if diff is None else str(round(diff, 1))
        parts.append(f"{describe_game_entry(game_entry)}|diff_minutes={diff_text}")

    return "; ".join(parts)


def describe_book_candidates(scored: list[tuple]) -> str:
    parts = []

    for diff, book_entry in scored:
        diff_text = "NA" if diff is None else str(round(diff, 1))
        parts.append(f"{describe_book_entry(book_entry)}|diff_minutes={diff_text}")

    return "; ".join(parts)


def print_rejection_rows(date_str: str, rejection_path: Path, rejection_rows: list[dict]) -> None:
    print("")
    print("=" * 80)
    print(f"REJECTED PREDICTION ROWS | date={date_str}")
    print(f"rejection_csv={rejection_path.as_posix()}")
    print(f"count={len(rejection_rows)}")
    print("=" * 80)

    for idx, row in enumerate(rejection_rows, start=1):
        print(f"REJECTED #{idx}")
        print(f"  reject_reason={row.get('reject_reason', '')}")
        print(f"  reject_action={row.get('reject_action', '')}")
        print(f"  fatal={row.get('fatal', '')}")
        print(f"  sportsbook_present={row.get('sportsbook_present', '')}")
        print(f"  sportsbook_match_detail={row.get('sportsbook_match_detail', '')}")
        print(f"  source_csv_row={row.get('source_csv_row', '')}")
        print(f"  game_date={row.get('game_date', '')}")
        print(f"  game_time={row.get('game_time', '')}")
        print(f"  away_team={row.get('away_team', '')}")
        print(f"  home_team={row.get('home_team', '')}")
        print(f"  away_pitcher={row.get('away_pitcher', '')}")
        print(f"  home_pitcher={row.get('home_pitcher', '')}")
        print(f"  away_prob={row.get('away_prob', '')}")
        print(f"  home_prob={row.get('home_prob', '')}")
        print(f"  candidate_games={row.get('candidate_games', '')}")
        print("-" * 80)

    print("=" * 80)
    print("END REJECTED PREDICTION ROWS")
    print("=" * 80)
    print("")


# ─────────────────────────────────────────────
# GROUP BUILDERS
# ─────────────────────────────────────────────

def build_prediction_groups(pred_rows: list[dict]) -> tuple[dict, list]:
    groups = {}
    key_order = []

    for idx, p in enumerate(pred_rows):
        key = (norm(p.get("home_team", "")), norm(p.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"prediction row has blank team at csv_row={idx + 2}: "
                f"away={p.get('away_team', '')} home={p.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []
            key_order.append(key)

        groups[key].append({
            "row": p,
            "key": key,
            "index": idx,
            "csv_row": idx + 2,
            "dt": parse_prediction_datetime(
                p.get("game_date", ""),
                p.get("game_time", ""),
            ),
        })

    return groups, key_order


def build_games_groups(games_rows: list[dict], date_str: str) -> dict:
    groups = {}
    seen_game_ids = {}
    seen_gamepks = {}

    for idx, g in enumerate(games_rows):
        csv_row = idx + 2

        game_id = (g.get("game_id") or "").strip()
        game_pk = (g.get("gamePk") or "").strip()

        if not game_id:
            fail(f"{date_str} | games row has blank game_id at csv_row={csv_row}: {g}")

        if not game_pk:
            fail(f"{date_str} | games row has blank gamePk at csv_row={csv_row}: {g}")

        if game_id in seen_game_ids:
            fail(
                f"{date_str} | duplicate games game_id={game_id} "
                f"first_csv_row={seen_game_ids[game_id]} second_csv_row={csv_row}"
            )

        if game_pk in seen_gamepks:
            fail(
                f"{date_str} | duplicate games gamePk={game_pk} "
                f"first_csv_row={seen_gamepks[game_pk]} second_csv_row={csv_row}"
            )

        seen_game_ids[game_id] = csv_row
        seen_gamepks[game_pk] = csv_row

        key = (norm(g.get("home_team", "")), norm(g.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"{date_str} | games row has blank team at csv_row={csv_row}: "
                f"away={g.get('away_team', '')} home={g.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []

        groups[key].append({
            "row": g,
            "key": key,
            "index": idx,
            "csv_row": csv_row,
            "dt": parse_games_datetime(
                g.get("game_date", ""),
                g.get("game_time", ""),
            ),
            "game_number": parse_int(g.get("gameNumber", "1"), 1),
            "used": False,
        })

    return groups


def build_book_groups(book_rows: list[dict], date_str: str) -> dict:
    groups = {}
    seen_book_game_ids = {}

    for idx, b in enumerate(book_rows):
        csv_row = idx + 2
        game_id = (b.get("game_id") or "").strip()

        if not game_id:
            fail(f"{date_str} | sportsbook row has blank game_id at csv_row={csv_row}: {b}")

        if game_id in seen_book_game_ids:
            fail(
                f"{date_str} | duplicate sportsbook game_id={game_id} "
                f"first_csv_row={seen_book_game_ids[game_id]} second_csv_row={csv_row}"
            )

        seen_book_game_ids[game_id] = csv_row

        key = (norm(b.get("home_team", "")), norm(b.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"{date_str} | sportsbook row has blank team at csv_row={csv_row}: "
                f"away={b.get('away_team', '')} home={b.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []

        groups[key].append({
            "row": b,
            "key": key,
            "index": idx,
            "csv_row": csv_row,
            "dt": parse_book_datetime(
                b.get("game_date", ""),
                b.get("game_time", ""),
            ),
            "used": False,
        })

    return groups


# ─────────────────────────────────────────────
# SPORTSBOOK PRESENCE
# ─────────────────────────────────────────────

def build_sportsbook_presence(date_str: str, pred_groups: dict, pred_key_order: list, book_groups: dict) -> dict:
    presence = {}

    for key in pred_key_order:
        preds = pred_groups.get(key, [])
        books = book_groups.get(key, [])

        if not preds:
            continue

        label = matchup_label(preds[0]["row"])

        for pred_entry in preds:
            presence[pred_entry["index"]] = {
                "present": False,
                "detail": "",
            }

        if not books:
            log(
                f"{date_str} | sportsbook missing for prediction matchup: "
                f"{label} pred_count={len(preds)}",
                "WARN",
            )
            continue

        unused_books = [b for b in books if not b["used"]]

        if len(preds) == 1 and len(unused_books) == 1:
            pred_entry = preds[0]
            book_entry = unused_books[0]
            book_entry["used"] = True

            presence[pred_entry["index"]] = {
                "present": True,
                "detail": describe_book_entry(book_entry),
            }

            diff = minutes_between(pred_entry.get("dt"), book_entry.get("dt"))
            diff_text = "" if diff is None else f" diff_minutes={round(diff, 1)}"

            log(
                f"{date_str} | SPORTSBOOK MATCH one-to-one: {label} "
                f"pred_time={pred_entry['row'].get('game_time', '')} "
                f"book_time={book_entry['row'].get('game_time', '')}"
                f"{diff_text}"
            )
            continue

        if len(preds) > 1 and len(unused_books) > 1 and len(preds) == len(unused_books):
            sorted_preds = sorted(
                preds,
                key=lambda x: (
                    dt_sort_value(x.get("dt")),
                    x["index"],
                ),
            )

            sorted_books = sorted(
                unused_books,
                key=lambda x: (
                    dt_sort_value(x.get("dt")),
                    x["index"],
                ),
            )

            log(
                f"{date_str} | SPORTSBOOK ORDER MATCH duplicate matchup: "
                f"{label} pred_count={len(sorted_preds)} book_count={len(sorted_books)}"
            )

            for pred_entry, book_entry in zip(sorted_preds, sorted_books):
                book_entry["used"] = True

                presence[pred_entry["index"]] = {
                    "present": True,
                    "detail": describe_book_entry(book_entry),
                }

                diff = minutes_between(pred_entry.get("dt"), book_entry.get("dt"))
                diff_text = "" if diff is None else f" diff_minutes={round(diff, 1)}"

                log(
                    f"{date_str} | SPORTSBOOK MATCH order: {label} "
                    f"pred_time={pred_entry['row'].get('game_time', '')} "
                    f"book_time={book_entry['row'].get('game_time', '')}"
                    f"{diff_text}"
                )

            continue

        sorted_preds = sorted(
            preds,
            key=lambda x: (
                dt_sort_value(x.get("dt")),
                x["index"],
            ),
        )

        for pred_entry in sorted_preds:
            available_books = [b for b in books if not b["used"]]

            if not available_books:
                log(
                    f"{date_str} | sportsbook no unused row for prediction: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "WARN",
                )
                continue

            scored = []

            for book_entry in available_books:
                diff = minutes_between(pred_entry.get("dt"), book_entry.get("dt"))
                scored.append((diff, book_entry))

            scored_valid = [x for x in scored if x[0] is not None]

            selected = None
            selected_diff = None

            if scored_valid:
                selected_diff, selected = min(scored_valid, key=lambda x: x[0])

                if selected_diff > MAX_TIME_DIFF_MINUTES:
                    selected = None

            if selected is None:
                candidates = describe_book_candidates(scored)
                log(
                    f"{date_str} | sportsbook missing within time threshold: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')} "
                    f"candidate_books={candidates}",
                    "WARN",
                )
                continue

            selected["used"] = True

            presence[pred_entry["index"]] = {
                "present": True,
                "detail": describe_book_entry(selected),
            }

            diff_text = "" if selected_diff is None else f" diff_minutes={round(selected_diff, 1)}"

            log(
                f"{date_str} | SPORTSBOOK MATCH closest: {label} "
                f"pred_time={pred_entry['row'].get('game_time', '')} "
                f"book_time={selected['row'].get('game_time', '')}"
                f"{diff_text}"
            )

    return presence


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_output_rows(date_str: str, eligible_count: int, output_rows: list[dict]):
    if len(output_rows) != eligible_count:
        fail(
            f"{date_str} | eligible output row count mismatch: "
            f"eligible_prediction_rows={eligible_count} output_rows={len(output_rows)}"
        )

    seen_game_ids = {}

    for idx, row in enumerate(output_rows):
        csv_row = idx + 2
        game_id = (row.get("game_id") or "").strip()

        if not game_id:
            fail(f"{date_str} | output row has blank game_id at csv_row={csv_row}: {row}")

        if game_id in seen_game_ids:
            fail(
                f"{date_str} | duplicate output game_id={game_id} "
                f"first_csv_row={seen_game_ids[game_id]} second_csv_row={csv_row}"
            )

        seen_game_ids[game_id] = csv_row


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(date_str: str, pred_path: Path, summary: dict) -> None:
    games_path = GAMES_DIR / f"{date_str}_games.csv"
    book_path = BOOK_DIR / f"{date_str}_MLB.csv"
    rejection_path = REJECTION_DIR / f"{date_str}_unmatched_predictions.csv"

    pred_rows = load_csv(pred_path, REQUIRED_PRED_COLS, "prediction input")
    book_rows = load_csv(book_path, REQUIRED_BOOK_COLS, "sportsbook input", required_file=False)

    if not pred_rows:
        log(f"{date_str} | no prediction rows — skipping", "WARN")
        summary["skipped"] += 1
        return

    pred_groups, pred_key_order = build_prediction_groups(pred_rows)
    book_groups = build_book_groups(book_rows, date_str) if book_rows else {}

    sportsbook_presence = build_sportsbook_presence(
        date_str=date_str,
        pred_groups=pred_groups,
        pred_key_order=pred_key_order,
        book_groups=book_groups,
    )

    rejection_rows = []
    nonfatal_rejection_count = 0

    for key in pred_key_order:
        preds = pred_groups.get(key, [])

        for pred_entry in preds:
            presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

            if presence["present"]:
                continue

            rejection_rows.append(make_rejection_row(
                pred_entry=pred_entry,
                reason="sportsbook_game_missing_for_prediction",
                candidate_games="",
                fatal=False,
                sportsbook_present=False,
                sportsbook_match_detail=presence.get("detail", ""),
            ))

            nonfatal_rejection_count += 1

            log(
                f"{date_str} | non-fatal rejected prediction because sportsbook row is missing: "
                f"csv_row={pred_entry['csv_row']} "
                f"{matchup_label(pred_entry['row'])} "
                f"pred_time={pred_entry['row'].get('game_time', '')}",
                "WARN",
            )

    eligible_pred_indexes = {
        pred_entry["index"]
        for preds in pred_groups.values()
        for pred_entry in preds
        if sportsbook_presence.get(pred_entry["index"], {"present": False})["present"]
    }

    eligible_count = len(eligible_pred_indexes)

    if eligible_count == 0:
        if rejection_rows:
            write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
            print_rejection_rows(date_str, rejection_path, rejection_rows)

        write_output_csv(date_str, OUTPUT_HEADER, [], summary)

        log(
            f"{date_str} | no sportsbook-eligible prediction rows. "
            f"input_predictions={len(pred_rows)} nonfatal_rejections={nonfatal_rejection_count} "
            f"output_rows=0"
        )

        summary["rejected"] += len(rejection_rows)
        summary["nonfatal_rejections"] += nonfatal_rejection_count
        return

    if not games_path.exists():
        if is_current_or_future_date(date_str):
            for idx in sorted(eligible_pred_indexes):
                pred_entry = {
                    "row": pred_rows[idx],
                    "index": idx,
                    "csv_row": idx + 2,
                }
                presence = sportsbook_presence.get(idx, {"present": False, "detail": ""})

                already_rejected = any(
                    rejection_row.get("source_csv_row", "") == str(idx + 2)
                    for rejection_row in rejection_rows
                )

                if already_rejected:
                    continue

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="games_file_missing_current_or_future",
                    candidate_games=str(games_path),
                    fatal=False,
                    sportsbook_present=presence.get("present", False),
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                nonfatal_rejection_count += 1

                log(
                    f"{date_str} | non-fatal rejected prediction because current/future games file is missing: "
                    f"csv_row={idx + 2} games_path={games_path}",
                    "WARN",
                )

            if rejection_rows:
                write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
                print_rejection_rows(date_str, rejection_path, rejection_rows)

            write_output_csv(date_str, OUTPUT_HEADER, [], summary)

            log(
                f"{date_str} | current/future games file missing. "
                f"games_path={games_path} input_predictions={len(pred_rows)} "
                f"sportsbook_rows={len(book_rows)} eligible_predictions={eligible_count} "
                f"output_rows=0 nonfatal_rejections={nonfatal_rejection_count}"
            )

            summary["rejected"] += len(rejection_rows)
            summary["nonfatal_rejections"] += nonfatal_rejection_count
            return

        fail(f"{date_str} | past-date games file missing: {games_path}")

    games_rows = load_csv(games_path, REQUIRED_GAMES_COLS, "games input", required_file=True)

    if not games_rows:
        if is_current_or_future_date(date_str):
            for idx in sorted(eligible_pred_indexes):
                pred_entry = {
                    "row": pred_rows[idx],
                    "index": idx,
                    "csv_row": idx + 2,
                }
                presence = sportsbook_presence.get(idx, {"present": False, "detail": ""})

                already_rejected = any(
                    rejection_row.get("source_csv_row", "") == str(idx + 2)
                    for rejection_row in rejection_rows
                )

                if already_rejected:
                    continue

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="games_file_empty_current_or_future",
                    candidate_games=str(games_path),
                    fatal=False,
                    sportsbook_present=presence.get("present", False),
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                nonfatal_rejection_count += 1

                log(
                    f"{date_str} | non-fatal rejected prediction because current/future games file is empty: "
                    f"csv_row={idx + 2} games_path={games_path}",
                    "WARN",
                )

            if rejection_rows:
                write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
                print_rejection_rows(date_str, rejection_path, rejection_rows)

            write_output_csv(date_str, OUTPUT_HEADER, [], summary)

            log(
                f"{date_str} | current/future games file empty. "
                f"games_path={games_path} input_predictions={len(pred_rows)} "
                f"sportsbook_rows={len(book_rows)} eligible_predictions={eligible_count} "
                f"output_rows=0 nonfatal_rejections={nonfatal_rejection_count}"
            )

            summary["rejected"] += len(rejection_rows)
            summary["nonfatal_rejections"] += nonfatal_rejection_count
            return

        fail(f"{date_str} | past-date games file has zero rows: {games_path}")

    games_groups = build_games_groups(games_rows, date_str)

    output_by_pred_index = {}
    fatal_rejection_count = 0
    matched = 0

    for key in pred_key_order:
        preds = [
            pred_entry
            for pred_entry in pred_groups.get(key, [])
            if pred_entry["index"] in eligible_pred_indexes
        ]

        games = games_groups.get(key, [])

        if not preds:
            continue

        label = matchup_label(preds[0]["row"])

        if not games:
            for pred_entry in preds:
                presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="no_games_row_for_same_home_away",
                    candidate_games="",
                    fatal=True,
                    sportsbook_present=presence["present"],
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                fatal_rejection_count += 1

                log(
                    f"{date_str} | fatal unmatched sportsbook-eligible prediction no games row: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "ERROR",
                )
            continue

        unused_games = [g for g in games if not g["used"]]

        if len(preds) == 1 and len(unused_games) == 1:
            pred_entry = preds[0]
            game_entry = unused_games[0]
            game_entry["used"] = True

            game_id = (game_entry["row"].get("game_id") or "").strip()

            if not game_id:
                presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="matched_games_row_has_blank_game_id",
                    candidate_games=describe_game_entry(game_entry),
                    fatal=True,
                    sportsbook_present=presence["present"],
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                fatal_rejection_count += 1

                log(
                    f"{date_str} | matched games row has blank game_id: "
                    f"{label} games_csv_row={game_entry['csv_row']}",
                    "ERROR",
                )
                continue

            output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
            matched += 1

            diff = minutes_between(pred_entry.get("dt"), game_entry.get("dt"))
            diff_text = "" if diff is None else f" diff_minutes={round(diff, 1)}"

            level = "INFO"
            label_text = "MATCHED one-to-one"

            if diff is not None and diff > MAX_TIME_DIFF_MINUTES:
                level = "WARN"
                label_text = "MATCHED one-to-one with time mismatch"

            log(
                f"{date_str} | {label_text}: {label} "
                f"pred_time={pred_entry['row'].get('game_time', '')} "
                f"games_time={game_entry['row'].get('game_time', '')} "
                f"game_id={game_id}"
                f"{diff_text}",
                level,
            )

            continue

        if len(preds) > 1 and len(unused_games) > 1 and len(preds) == len(unused_games):
            sorted_preds = sorted(
                preds,
                key=lambda x: (
                    dt_sort_value(x.get("dt")),
                    x["index"],
                ),
            )

            sorted_games = sorted(
                unused_games,
                key=lambda x: (
                    dt_sort_value(x.get("dt")),
                    x["game_number"],
                    x["index"],
                ),
            )

            log(
                f"{date_str} | ORDER MATCH duplicate matchup: "
                f"{label} eligible_pred_count={len(sorted_preds)} games_count={len(sorted_games)}"
            )

            for pred_entry, game_entry in zip(sorted_preds, sorted_games):
                game_entry["used"] = True

                game_id = (game_entry["row"].get("game_id") or "").strip()

                if not game_id:
                    presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

                    rejection_rows.append(make_rejection_row(
                        pred_entry=pred_entry,
                        reason="matched_games_row_has_blank_game_id",
                        candidate_games=describe_game_entry(game_entry),
                        fatal=True,
                        sportsbook_present=presence["present"],
                        sportsbook_match_detail=presence.get("detail", ""),
                    ))

                    fatal_rejection_count += 1

                    log(
                        f"{date_str} | matched games row has blank game_id: "
                        f"{label} games_csv_row={game_entry['csv_row']}",
                        "ERROR",
                    )
                    continue

                output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
                matched += 1

                diff = minutes_between(pred_entry.get("dt"), game_entry.get("dt"))
                diff_text = "" if diff is None else f" diff_minutes={round(diff, 1)}"

                level = "INFO"
                if diff is not None and diff > MAX_TIME_DIFF_MINUTES:
                    level = "WARN"

                log(
                    f"{date_str} | MATCHED order: {label} "
                    f"pred_time={pred_entry['row'].get('game_time', '')} "
                    f"games_time={game_entry['row'].get('game_time', '')} "
                    f"gameNumber={game_entry['row'].get('gameNumber', '')} "
                    f"game_id={game_id}"
                    f"{diff_text}",
                    level,
                )

            continue

        sorted_preds = sorted(
            preds,
            key=lambda x: (
                dt_sort_value(x.get("dt")),
                x["index"],
            ),
        )

        for pred_entry in sorted_preds:
            available_games = [g for g in games if not g["used"]]

            if not available_games:
                presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="no_unused_games_row_for_same_home_away",
                    candidate_games="",
                    fatal=True,
                    sportsbook_present=presence["present"],
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                fatal_rejection_count += 1

                log(
                    f"{date_str} | fatal unmatched sportsbook-eligible prediction no unused games row: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "ERROR",
                )
                continue

            scored = []

            for game_entry in available_games:
                diff = minutes_between(pred_entry.get("dt"), game_entry.get("dt"))
                scored.append((diff, game_entry))

            scored_valid = [x for x in scored if x[0] is not None]

            selected = None
            selected_diff = None

            if scored_valid:
                selected_diff, selected = min(scored_valid, key=lambda x: x[0])

                if selected_diff > MAX_TIME_DIFF_MINUTES:
                    selected = None

            if selected is None:
                candidates = describe_candidates(scored)
                presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="no_games_row_within_time_threshold",
                    candidate_games=candidates,
                    fatal=True,
                    sportsbook_present=presence["present"],
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                fatal_rejection_count += 1

                log(
                    f"{date_str} | fatal unmatched sportsbook-eligible prediction time threshold: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')} "
                    f"candidate_games={candidates}",
                    "ERROR",
                )
                continue

            selected["used"] = True

            game_id = (selected["row"].get("game_id") or "").strip()

            if not game_id:
                presence = sportsbook_presence.get(pred_entry["index"], {"present": False, "detail": ""})

                rejection_rows.append(make_rejection_row(
                    pred_entry=pred_entry,
                    reason="matched_games_row_has_blank_game_id",
                    candidate_games=describe_game_entry(selected),
                    fatal=True,
                    sportsbook_present=presence["present"],
                    sportsbook_match_detail=presence.get("detail", ""),
                ))

                fatal_rejection_count += 1

                log(
                    f"{date_str} | matched games row has blank game_id: "
                    f"{label} games_csv_row={selected['csv_row']}",
                    "ERROR",
                )
                continue

            output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
            matched += 1

            diff_text = "" if selected_diff is None else f" diff_minutes={round(selected_diff, 1)}"

            log(
                f"{date_str} | MATCHED closest: {label} "
                f"pred_time={pred_entry['row'].get('game_time', '')} "
                f"games_time={selected['row'].get('game_time', '')} "
                f"game_id={game_id}"
                f"{diff_text}"
            )

    for key, games in games_groups.items():
        unused = [g for g in games if not g["used"]]
        for game_entry in unused:
            g = game_entry["row"]
            log(
                f"{date_str} | UNUSED games row: "
                f"{g.get('away_team', '')} @ {g.get('home_team', '')} "
                f"game_id={g.get('game_id', '')} game_time={g.get('game_time', '')}",
                "WARN",
            )

    for idx in sorted(eligible_pred_indexes):
        if idx in output_by_pred_index:
            continue

        already_rejected = any(
            rejection_row.get("source_csv_row", "") == str(idx + 2)
            for rejection_row in rejection_rows
        )

        if already_rejected:
            continue

        pred_entry = {
            "row": pred_rows[idx],
            "index": idx,
            "csv_row": idx + 2,
        }
        presence = sportsbook_presence.get(idx, {"present": False, "detail": ""})

        rejection_rows.append(make_rejection_row(
            pred_entry=pred_entry,
            reason="eligible_prediction_row_not_processed",
            candidate_games="",
            fatal=True,
            sportsbook_present=presence["present"],
            sportsbook_match_detail=presence.get("detail", ""),
        ))

        fatal_rejection_count += 1

        log(
            f"{date_str} | eligible prediction row not processed: "
            f"csv_row={idx + 2} away={pred_rows[idx].get('away_team', '')} "
            f"home={pred_rows[idx].get('home_team', '')}",
            "ERROR",
        )

    if rejection_rows:
        write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
        print_rejection_rows(date_str, rejection_path, rejection_rows)

    if fatal_rejection_count:
        summary["rejected"] += len(rejection_rows)
        summary["fatal_rejections"] += fatal_rejection_count
        summary["nonfatal_rejections"] += nonfatal_rejection_count
        summary["errors"] += fatal_rejection_count

        fail(
            f"{date_str} | fatal sportsbook-eligible prediction rows could not be assigned game_id: "
            f"fatal_count={fatal_rejection_count} "
            f"nonfatal_count={nonfatal_rejection_count} "
            f"rejection_csv={rejection_path}"
        )

    output_rows = []

    for idx in range(len(pred_rows)):
        if idx in output_by_pred_index:
            output_rows.append(output_by_pred_index[idx])

    validate_output_rows(date_str, eligible_count, output_rows)

    written_path = write_output_csv(date_str, OUTPUT_HEADER, output_rows, summary)

    log(
        f"{date_str} | WROTE: {written_path} | rows={len(output_rows)} "
        f"matched={matched} "
        f"input_predictions={len(pred_rows)} "
        f"sportsbook_rows={len(book_rows)} "
        f"eligible_predictions={eligible_count} "
        f"nonfatal_rejections={nonfatal_rejection_count} "
        f"fatal_rejections=0"
    )

    summary["total_rows"] += len(output_rows)
    summary["matched"] += matched
    summary["rejected"] += len(rejection_rows)
    summary["nonfatal_rejections"] += nonfatal_rejection_count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== game_id_pred RUN {_now()} ===\n")

    summary = {
        "files_written": 0,
        "total_rows": 0,
        "matched": 0,
        "rejected": 0,
        "nonfatal_rejections": 0,
        "fatal_rejections": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        pred_files = sorted(PRED_DIR.glob("*_MLB.csv"))

        pred_files = [
            p for p in pred_files
            if "pred_with_game_id" not in str(p)
        ]

        log(f"prediction files found: {len(pred_files)}")

        for pred_path in pred_files:
            date_str = pred_path.stem.replace("_MLB", "")
            process_date(date_str, pred_path, summary)

    except Exception as e:
        tb = traceback.format_exc()
        log(f"FATAL: {e}\n{tb}", "ERROR")

        print("")
        print("=" * 80)
        print("FATAL ERROR IN game_id_pred.py")
        print("=" * 80)
        print(f"error={e}")
        print("")
        print(tb)
        print("=" * 80)
        print("")

        if summary["errors"] == 0:
            summary["errors"] += 1

    status = "SUCCESS" if summary["errors"] == 0 and summary["fatal_rejections"] == 0 else "FAILED"

    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_written        : {summary['files_written']}",
        f"  total_rows           : {summary['total_rows']}",
        f"  matched              : {summary['matched']}",
        f"  rejected             : {summary['rejected']}",
        f"  nonfatal_rejections  : {summary['nonfatal_rejections']}",
        f"  fatal_rejections     : {summary['fatal_rejections']}",
        f"  skipped              : {summary['skipped']}",
        f"  errors               : {summary['errors']}",
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"game_id_pred complete. "
        f"{summary['files_written']} files written. "
        f"matched={summary['matched']} "
        f"rejected={summary['rejected']} "
        f"nonfatal_rejections={summary['nonfatal_rejections']} "
        f"fatal_rejections={summary['fatal_rejections']} "
        f"errors={summary['errors']} "
        f"Status: {status}"
    )

    if summary["errors"] != 0 or summary["fatal_rejections"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

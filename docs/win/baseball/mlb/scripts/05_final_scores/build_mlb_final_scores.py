#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/build_mlb_final_scores.py

import csv
import json
import traceback
from datetime import datetime, UTC
from pathlib import Path

ERROR_DIR = Path("docs/win/baseball/mlb/errors/05_final_scores")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_mlb_final_scores.txt"

RAW_DIR = Path("docs/win/baseball/mlb/00_intake/drat_raw")
GAMES_DIR = Path("docs/win/baseball/mlb/00_intake/games")
PRED_DIR = Path("docs/win/baseball/mlb/00_intake/predictions/pred_with_game_id")
SPORTSBOOK_DIR = Path("docs/win/baseball/mlb/00_intake/sportsbook")
SELECT_DIR = Path("docs/win/baseball/mlb/04_select")
FINAL_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/final_scores")
AUDIT_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/audit")

FINAL_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_AUDIT_FILE = AUDIT_DIR / "final_score_status_audit.csv"
KEY_AUDIT_FILE = AUDIT_DIR / "final_score_key_audit.csv"

RUN_TS = datetime.now(UTC).isoformat()
DOUBLEHEADER_TIME_TOLERANCE_MINUTES = 90

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== build_mlb_final_scores RUN {RUN_TS} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {msg}\n")


def fail(msg: str) -> None:
    log(f"FATAL: {msg}")
    raise RuntimeError(msg)


def parse_datetime(dt_str):
    dt = datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M %p")
    return dt, dt.strftime("%Y_%m_%d"), dt.strftime("%I:%M %p")


def parse_time_minutes(value):
    value = str(value).strip()
    if not value:
        return None

    for fmt in ["%I:%M %p", "%H:%M:%S", "%H:%M"]:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.hour * 60 + parsed.minute
        except ValueError:
            continue

    return None


def clean_team(team_str):
    return str(team_str).split("(")[0].strip()


def safe_get(row, key, default=""):
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def normalize_status(raw_status):
    raw = str(raw_status or "").strip().lower()

    if raw in {"final", "game over", "completed", "complete"}:
        return "final"

    if raw in {"postponed", "ppd"}:
        return "postponed"

    if raw in {"canceled", "cancelled"}:
        return "canceled"

    if raw in {"suspended"}:
        return "suspended"

    if raw in {"delayed", "delay"}:
        return "delayed"

    if raw in {"in progress", "live", "active"}:
        return "in_progress"

    if raw in {"scheduled", "pre-game", "pregame", "preview"}:
        return "scheduled"

    return "unknown"


def infer_game_status(row):
    """
    DRatings raw rows currently appear list-based.

    Existing behavior treated len(row) == 8 as completed/final-score rows.
    If an explicit status field exists in a dict-shaped source later, preserve it.
    Otherwise completed-score rows are inferred as final, and non-completed rows
    are treated as unknown for audit purposes.
    """
    explicit_status_fields = [
        "game_status",
        "status",
        "abstractGameState",
        "detailedState",
        "codedGameState",
        "statusCode",
    ]

    if isinstance(row, dict):
        for field in explicit_status_fields:
            val = row.get(field)
            if val not in (None, ""):
                return normalize_status(val), str(val).strip(), field, True

    if isinstance(row, list) and len(row) == 8:
        return "final", "final", "row_len_8_completed_score", False

    return "unknown", "unknown", "not_available_in_current_raw_shape", False


def is_completed_game(row):
    status_norm, _raw_status, _status_source, _status_available = infer_game_status(row)
    return status_norm == "final" and isinstance(row, list) and len(row) == 8


def closest_time_match(candidates, target_game_time, value_field):
    if not candidates:
        return ""

    if len(candidates) == 1:
        return candidates[0].get(value_field, "")

    target_minutes = parse_time_minutes(target_game_time)
    if target_minutes is None:
        return ""

    best_candidate = None
    best_diff = None

    for candidate in candidates:
        candidate_minutes = parse_time_minutes(candidate.get("game_time", ""))
        if candidate_minutes is None:
            continue

        diff = abs(candidate_minutes - target_minutes)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_candidate = candidate

    if best_candidate is None or best_diff > DOUBLEHEADER_TIME_TOLERANCE_MINUTES:
        return ""

    return best_candidate.get(value_field, "")


def closest_time_record_match(candidates, target_game_time):
    if not candidates:
        return {}

    if len(candidates) == 1:
        return candidates[0]

    target_minutes = parse_time_minutes(target_game_time)
    if target_minutes is None:
        return {}

    best_candidate = None
    best_diff = None

    for candidate in candidates:
        candidate_minutes = parse_time_minutes(candidate.get("game_time", ""))
        if candidate_minutes is None:
            continue

        diff = abs(candidate_minutes - target_minutes)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_candidate = candidate

    if best_candidate is None or best_diff > DOUBLEHEADER_TIME_TOLERANCE_MINUTES:
        return {}

    return best_candidate


def closest_time_book_match(candidates, target_game_time):
    return closest_time_record_match(candidates, target_game_time)


def assert_selected_files_exist():
    select_files = sorted(SELECT_DIR.glob("*_MLB.csv"))
    if not select_files:
        fail(
            f"No selected-bet files found in {SELECT_DIR}. "
            "Final-score generation is post-selection only. Run baseball_select_bets.py first."
        )

    log(f"Selected-bet files found before final-score build: {len(select_files)}")
    return {p.stem.replace("_MLB", "") for p in select_files}


def load_games_lookup(date):
    path = GAMES_DIR / f"{date}_games.csv"
    lookup = {}

    if not path.exists():
        log(f"GAMES FILE MISSING FOR FINAL-SCORE GAME_ID/GAMEPK LOOKUP: {path}")
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for r in reader:
            key = (
                r.get("home_team", "").strip(),
                r.get("away_team", "").strip(),
            )

            lookup.setdefault(key, []).append({
                "game_id": r.get("game_id", ""),
                "gamePk": r.get("gamePk", ""),
                "gameNumber": r.get("gameNumber", ""),
                "game_time": r.get("game_time", ""),
                "home_team": r.get("home_team", ""),
                "away_team": r.get("away_team", ""),
            })

    return lookup


def load_predictions_lookup(date):
    path = PRED_DIR / f"{date}_MLB.csv"
    lookup = {}

    if not path.exists():
        log(f"PREDICTION FILE MISSING FOR FINAL-SCORE GAME_ID LOOKUP: {path}")
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (
                r.get("home_team", "").strip(),
                r.get("away_team", "").strip(),
            )
            lookup.setdefault(key, []).append({
                "game_id": r.get("game_id", ""),
                "game_time": r.get("game_time", ""),
                "home_team": r.get("home_team", ""),
                "away_team": r.get("away_team", ""),
            })

    return lookup


def load_sportsbook_lookup(date):
    path = SPORTSBOOK_DIR / f"{date}_MLB.csv"
    lookup = {}

    if not path.exists():
        log(f"SPORTSBOOK FILE MISSING FOR FINAL-SCORE MARKET-LINE LOOKUP: {path}")
        return lookup

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (
                r.get("home_team", "").strip(),
                r.get("away_team", "").strip(),
            )
            lookup.setdefault(key, []).append({
                "game_time": r.get("game_time", ""),
                "away_run_line": r.get("away_run_line"),
                "home_run_line": r.get("home_run_line"),
                "total": r.get("total"),
            })

    return lookup


SUMMARY_ROW_PREFIXES = {"Sportsbooks", "DRatings"}

FINAL_HEADER = [
    "sport",
    "league",
    "game_id",
    "gamePk",
    "gameNumber",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "final_away_score",
    "final_home_score",
    "final_total",
    "away_run_line",
    "home_run_line",
    "total",
    "game_status",
    "final_scores_generated_at",
]


def is_summary_row(row):
    return row and isinstance(row, list) and str(row[0]).strip() in SUMMARY_ROW_PREFIXES


def write_csv(path, header, rows, files_written, label):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    files_written.append((str(path), len(rows)))
    log(f"WROTE {label} -> {path} ({len(rows)} rows)")


def write_audit_csv(path, header, rows, label):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})

    log(f"WROTE {label} -> {path} ({len(rows)} rows)")


def raw_row_text(row):
    try:
        return json.dumps(row, ensure_ascii=False, default=str)
    except Exception:
        return repr(row)


def make_parse_error_row(*, source_file, row_index, stage, error, row):
    return {
        "source_file": source_file,
        "row_index": row_index,
        "stage": stage,
        "error": str(error),
        "raw_row": raw_row_text(row),
    }


def log_review_rows(parse_error_rows, blank_game_id_rows):
    log("--- PARSE ERROR ROWS FOR REVIEW ---")
    if not parse_error_rows:
        log("None")
    else:
        for item in parse_error_rows:
            log(
                "PARSE_ERROR | "
                f"source_file={item.get('source_file', '')} | "
                f"row_index={item.get('row_index', '')} | "
                f"stage={item.get('stage', '')} | "
                f"error={item.get('error', '')} | "
                f"raw_row={item.get('raw_row', '')}"
            )

    log("--- BLANK GAME_ID ROWS FOR REVIEW ---")
    if not blank_game_id_rows:
        log("None")
    else:
        for item in blank_game_id_rows:
            log(
                "BLANK_GAME_ID | "
                f"source_file={item.get('source_file', '')} | "
                f"row_index={item.get('row_index', '')} | "
                f"game_date={item.get('game_date', '')} | "
                f"game_time={item.get('game_time', '')} | "
                f"away_team={item.get('away_team', '')} | "
                f"home_team={item.get('home_team', '')} | "
                f"final_away_score={item.get('final_away_score', '')} | "
                f"final_home_score={item.get('final_home_score', '')} | "
                f"gamePk={item.get('gamePk', '')} | "
                f"gameNumber={item.get('gameNumber', '')} | "
                f"raw_row={item.get('raw_row', '')}"
            )


def final_row_signature(record):
    return (
        record.get("sport", ""),
        record.get("league", ""),
        record.get("game_id", ""),
        record.get("gamePk", ""),
        record.get("gameNumber", ""),
        record.get("game_date", ""),
        record.get("home_team", ""),
        record.get("away_team", ""),
        record.get("final_away_score", ""),
        record.get("final_home_score", ""),
        record.get("final_total", ""),
        record.get("away_run_line", ""),
        record.get("home_run_line", ""),
        record.get("total", ""),
        record.get("game_status", ""),
    )


def make_key_audit_row(
    *,
    game_date,
    game_id,
    gamePk,
    gameNumber,
    away_team,
    home_team,
    duplicate_count,
    status,
    notes,
):
    return {
        "game_date": game_date,
        "game_id": game_id,
        "gamePk": gamePk,
        "gameNumber": gameNumber,
        "away_team": away_team,
        "home_team": home_team,
        "duplicate_count": duplicate_count,
        "status": status,
        "notes": notes,
    }


def add_final_record(
    *,
    record,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    key_audit_rows,
    use_game_time_for_fallback,
):
    game_id = str(record.get("game_id", "")).strip()
    game_date = record.get("game_date", "")
    game_time = record.get("game_time", "")
    home_team = record.get("home_team", "")
    away_team = record.get("away_team", "")

    if game_id:
        existing = seen_by_game_id.get(game_id)

        if existing is None:
            seen_by_game_id[game_id] = record
            final_records_by_date.setdefault(game_date, []).append(record)

            key_audit_rows.append(make_key_audit_row(
                game_date=game_date,
                game_id=game_id,
                gamePk=record.get("gamePk", ""),
                gameNumber=record.get("gameNumber", ""),
                away_team=away_team,
                home_team=home_team,
                duplicate_count=1,
                status="unique_game_id",
                notes="accepted; primary key game_id",
            ))
            return "accepted"

        if final_row_signature(existing) == final_row_signature(record):
            key_audit_rows.append(make_key_audit_row(
                game_date=game_date,
                game_id=game_id,
                gamePk=record.get("gamePk", ""),
                gameNumber=record.get("gameNumber", ""),
                away_team=away_team,
                home_team=home_team,
                duplicate_count=2,
                status="identical_duplicate_collapsed",
                notes="duplicate game_id row was identical and was not written twice",
            ))
            return "duplicate_collapsed"

        key_audit_rows.append(make_key_audit_row(
            game_date=game_date,
            game_id=game_id,
            gamePk=record.get("gamePk", ""),
            gameNumber=record.get("gameNumber", ""),
            away_team=away_team,
            home_team=home_team,
            duplicate_count=2,
            status="conflicting_duplicate_game_id",
            notes="same game_id had conflicting final-score fields",
        ))

        fail(
            "Conflicting final-score duplicate game_id found: "
            f"game_id={game_id} date={game_date} away={away_team} home={home_team}"
        )

    if use_game_time_for_fallback:
        fallback_key = (game_date, home_team, away_team, game_time)
        fallback_notes = "game_id missing; fallback date/team/time key used for doubleheader identification"
    else:
        fallback_key = (game_date, home_team, away_team)
        fallback_notes = "game_id missing; fallback date/team key used to avoid exact duplicate raw writes"

    existing_fallback = seen_by_fallback_key.get(fallback_key)

    if existing_fallback is None:
        seen_by_fallback_key[fallback_key] = record
        final_records_by_date.setdefault(game_date, []).append(record)

        key_audit_rows.append(make_key_audit_row(
            game_date=game_date,
            game_id="",
            gamePk=record.get("gamePk", ""),
            gameNumber=record.get("gameNumber", ""),
            away_team=away_team,
            home_team=home_team,
            duplicate_count=1,
            status="blank_game_id_written_for_downstream_audit",
            notes=fallback_notes,
        ))
        return "accepted_blank_game_id"

    if final_row_signature(existing_fallback) == final_row_signature(record):
        key_audit_rows.append(make_key_audit_row(
            game_date=game_date,
            game_id="",
            gamePk=record.get("gamePk", ""),
            gameNumber=record.get("gameNumber", ""),
            away_team=away_team,
            home_team=home_team,
            duplicate_count=2,
            status="blank_game_id_identical_duplicate_collapsed",
            notes="blank-game_id duplicate was identical and was not written twice",
        ))
        return "blank_game_id_duplicate_collapsed"

    key_audit_rows.append(make_key_audit_row(
        game_date=game_date,
        game_id="",
        gamePk=record.get("gamePk", ""),
        gameNumber=record.get("gameNumber", ""),
        away_team=away_team,
        home_team=home_team,
        duplicate_count=2,
        status="blank_game_id_conflicting_duplicate",
        notes="blank-game_id duplicate fallback key had conflicting fields",
    ))

    fail(
        "Conflicting blank-game_id final-score duplicate found: "
        f"date={game_date} time={game_time} away={away_team} home={home_team}"
    )

    return "failed"


def process_file(
    file_path,
    final_records_by_date,
    seen_by_game_id,
    seen_by_fallback_key,
    selected_dates,
    status_audit_rows,
    key_audit_rows,
    parse_error_rows,
    blank_game_id_rows,
):
    log(f"Processing {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    games_lookup_cache = {}
    predictions_lookup_cache = {}
    sportsbook_lookup_cache = {}

    parse_errors = 0
    skipped_summary = 0
    skipped_duplicate = 0
    skipped_no_selected_file = 0
    skipped_not_completed = 0
    completed_rows_seen = 0
    accepted_rows = 0
    accepted_blank_game_id_rows = 0

    if not isinstance(data, list):
        parse_errors += 1
        parse_error_rows.append(make_parse_error_row(
            source_file=file_path.name,
            row_index="",
            stage="validate_json_structure",
            error=f"expected top-level JSON list, found {type(data).__name__}",
            row=data,
        ))

        log(
            f"  completed_rows_seen={completed_rows_seen}, "
            f"accepted_rows={accepted_rows}, "
            f"accepted_blank_game_id_rows={accepted_blank_game_id_rows}, "
            f"parse_errors={parse_errors}, "
            f"skipped_summary={skipped_summary}, "
            f"skipped_duplicate={skipped_duplicate}, "
            f"skipped_not_completed={skipped_not_completed}, "
            f"skipped_no_selected_file={skipped_no_selected_file}, "
            f"final_score_dates_accumulated={len(final_records_by_date)}"
        )
        return

    for row_index, row in enumerate(data, start=1):
        if not isinstance(row, list):
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="validate_row_structure",
                error=f"expected row list, found {type(row).__name__}",
                row=row,
            ))
            continue

        if not row:
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="validate_row_structure",
                error="empty row",
                row=row,
            ))
            continue

        if is_summary_row(row):
            skipped_summary += 1
            continue

        if len(row) < 2:
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="validate_row_structure",
                error=f"expected at least 2 fields, found {len(row)}",
                row=row,
            ))
            continue

        status_norm, raw_status, status_source, status_available = infer_game_status(row)

        if not is_completed_game(row):
            skipped_not_completed += 1
            status_audit_rows.append({
                "game_date": "",
                "game_id": "",
                "gamePk": "",
                "gameNumber": "",
                "away_team": "",
                "home_team": "",
                "final_away_score": "",
                "final_home_score": "",
                "game_status": status_norm,
                "status_source": status_source,
                "status_available": str(status_available),
                "status_notes": "non-final row not written to final-score output",
            })
            continue

        completed_rows_seen += 1

        try:
            _dt, game_date, game_time = parse_datetime(row[0])
        except Exception as exc:
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="parse_datetime",
                error=exc,
                row=row,
            ))
            continue

        if game_date not in selected_dates:
            skipped_no_selected_file += 1
            continue

        try:
            team_value = row[1]
            if not isinstance(team_value, str):
                raise TypeError(
                    f"expected team field to be str, found {type(team_value).__name__}"
                )

            teams = team_value.split("\n")
            if len(teams) < 2:
                raise ValueError("expected at least two team names")

            away_team = clean_team(teams[0])
            home_team = clean_team(teams[1])

            if not away_team or not home_team:
                raise ValueError("away or home team is blank")

        except Exception as exc:
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="parse_teams",
                error=exc,
                row=row,
            ))
            continue

        key = (home_team, away_team)

        try:
            score_value = row[5]
            if not isinstance(score_value, str):
                raise TypeError(
                    f"expected score field to be str, found {type(score_value).__name__}"
                )

            scores = score_value.split("\n")
            away_score = int(scores[0].strip())
            home_score = int(scores[1].strip()) if len(scores) > 1 else 0
            final_total = str(away_score + home_score)

        except Exception as exc:
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="parse_scores",
                error=exc,
                row=row,
            ))
            continue

        try:
            if game_date not in games_lookup_cache:
                games_lookup_cache[game_date] = load_games_lookup(game_date)
            if game_date not in predictions_lookup_cache:
                predictions_lookup_cache[game_date] = load_predictions_lookup(game_date)
            if game_date not in sportsbook_lookup_cache:
                sportsbook_lookup_cache[game_date] = load_sportsbook_lookup(game_date)

            games_lookup = games_lookup_cache[game_date]
            pred_lookup = predictions_lookup_cache[game_date]
            book_lookup = sportsbook_lookup_cache[game_date]

            games_candidates = games_lookup.get(key, [])
            games_match = closest_time_record_match(games_candidates, game_time)

            pred_candidates = pred_lookup.get(key, [])
            pred_game_id = closest_time_match(pred_candidates, game_time, "game_id")

            game_id = str(games_match.get("game_id", "") or pred_game_id or "").strip()
            gamePk = str(games_match.get("gamePk", "") or "").strip()
            gameNumber = str(games_match.get("gameNumber", "") or "").strip()

            book_candidates = book_lookup.get(key, [])
            book = closest_time_book_match(book_candidates, game_time)

            record = {
                "sport": "baseball",
                "league": "mlb",
                "game_id": game_id,
                "gamePk": gamePk,
                "gameNumber": gameNumber,
                "game_date": game_date,
                "game_time": game_time,
                "home_team": home_team,
                "away_team": away_team,
                "final_away_score": str(away_score),
                "final_home_score": str(home_score),
                "final_total": final_total,
                "away_run_line": book.get("away_run_line"),
                "home_run_line": book.get("home_run_line"),
                "total": book.get("total"),
                "game_status": status_norm,
                "final_scores_generated_at": RUN_TS,
            }

            if not game_id:
                blank_game_id_rows.append({
                    "source_file": file_path.name,
                    "row_index": row_index,
                    "game_date": game_date,
                    "game_time": game_time,
                    "away_team": away_team,
                    "home_team": home_team,
                    "final_away_score": str(away_score),
                    "final_home_score": str(home_score),
                    "gamePk": gamePk,
                    "gameNumber": gameNumber,
                    "raw_row": raw_row_text(row),
                })

            action = add_final_record(
                record=record,
                final_records_by_date=final_records_by_date,
                seen_by_game_id=seen_by_game_id,
                seen_by_fallback_key=seen_by_fallback_key,
                key_audit_rows=key_audit_rows,
                use_game_time_for_fallback=(
                    len(games_candidates) > 1 or len(pred_candidates) > 1
                ),
            )

            if action in {"duplicate_collapsed", "blank_game_id_duplicate_collapsed"}:
                skipped_duplicate += 1
            else:
                accepted_rows += 1
                if not game_id:
                    accepted_blank_game_id_rows += 1

            status_audit_rows.append({
                "game_date": game_date,
                "game_id": game_id,
                "gamePk": gamePk,
                "gameNumber": gameNumber,
                "away_team": away_team,
                "home_team": home_team,
                "final_away_score": str(away_score),
                "final_home_score": str(home_score),
                "game_status": status_norm,
                "status_source": status_source,
                "status_available": str(status_available),
                "status_notes": (
                    "explicit source status available"
                    if status_available
                    else "status inferred as final from completed DRatings row shape"
                ),
            })

        except Exception as exc:
            parse_errors += 1
            parse_error_rows.append(make_parse_error_row(
                source_file=file_path.name,
                row_index=row_index,
                stage="build_final_record",
                error=exc,
                row=row,
            ))
            continue

    log(
        f"  completed_rows_seen={completed_rows_seen}, "
        f"accepted_rows={accepted_rows}, "
        f"accepted_blank_game_id_rows={accepted_blank_game_id_rows}, "
        f"parse_errors={parse_errors}, "
        f"skipped_summary={skipped_summary}, "
        f"skipped_duplicate={skipped_duplicate}, "
        f"skipped_not_completed={skipped_not_completed}, "
        f"skipped_no_selected_file={skipped_no_selected_file}, "
        f"final_score_dates_accumulated={len(final_records_by_date)}"
    )

def main():
    files_written = []
    final_records_by_date = {}
    seen_by_game_id = {}
    seen_by_fallback_key = {}
    status_audit_rows = []
    key_audit_rows = []
    parse_error_rows = []
    blank_game_id_rows = []

    status_audit_header = [
        "game_date",
        "game_id",
        "gamePk",
        "gameNumber",
        "away_team",
        "home_team",
        "final_away_score",
        "final_home_score",
        "game_status",
        "status_source",
        "status_available",
        "status_notes",
    ]

    key_audit_header = [
        "game_date",
        "game_id",
        "gamePk",
        "gameNumber",
        "away_team",
        "home_team",
        "duplicate_count",
        "status",
        "notes",
    ]

    try:
        selected_dates = assert_selected_files_exist()
        raw_files = sorted(RAW_DIR.glob("*_mlb_raw.json"))
        log(f"Raw files found: {len(raw_files)}")
        log(f"Post-selection final-score build timestamp: {RUN_TS}")

        for file in raw_files:
            process_file(
                file_path=file,
                final_records_by_date=final_records_by_date,
                seen_by_game_id=seen_by_game_id,
                seen_by_fallback_key=seen_by_fallback_key,
                selected_dates=selected_dates,
                status_audit_rows=status_audit_rows,
                key_audit_rows=key_audit_rows,
                parse_error_rows=parse_error_rows,
                blank_game_id_rows=blank_game_id_rows,
            )

        for date in sorted(final_records_by_date):
            records = final_records_by_date[date]
            out = FINAL_DIR / f"{date}_final_scores_MLB.csv"
            rows = [[record.get(col, "") for col in FINAL_HEADER] for record in records]
            write_csv(out, FINAL_HEADER, rows, files_written, "final scores")

        write_audit_csv(
            STATUS_AUDIT_FILE,
            status_audit_header,
            status_audit_rows,
            "final-score status audit",
        )

        write_audit_csv(
            KEY_AUDIT_FILE,
            key_audit_header,
            key_audit_rows,
            "final-score key audit",
        )

        blank_game_id_count = sum(
            1 for row in key_audit_rows
            if str(row.get("game_id", "")).strip() == ""
            and str(row.get("status", "")).startswith("blank_game_id")
        )

        unknown_status_count = sum(
            1 for row in status_audit_rows
            if str(row.get("game_status", "")).strip().lower() == "unknown"
        )

        total_parse_errors = len(parse_error_rows)
        selected_blank_game_id_rows = len(blank_game_id_rows)

        log("--- SUMMARY ---")
        log(f"Raw files processed: {len(raw_files)}")
        log(f"Files written: {len(files_written)}")
        log(f"Final-score dates written once: {len(final_records_by_date)}")
        log(f"Final-score game_id primary-key rows: {len(seen_by_game_id)}")
        log(f"Final-score blank game_id key-audit rows: {blank_game_id_count}")
        log(f"Selected completed rows with blank game_id: {selected_blank_game_id_rows}")
        log(f"Parse errors encountered: {total_parse_errors}")
        log(f"Unknown status audit rows: {unknown_status_count}")
        log(f"Status audit: {STATUS_AUDIT_FILE}")
        log(f"Key audit: {KEY_AUDIT_FILE}")

        if total_parse_errors:
            log(
                "WARNING: Parse errors were encountered. "
                "See PARSE ERROR ROWS FOR REVIEW below."
            )
        else:
            log("Parse-error review: no parse errors encountered.")

        if selected_blank_game_id_rows:
            log(
                "WARNING: Selected completed games with blank game_id were written. "
                "See BLANK GAME_ID ROWS FOR REVIEW below."
            )
        else:
            log("Blank-game_id review: no selected completed rows had blank game_id.")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")

        log_review_rows(parse_error_rows, blank_game_id_rows)

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("MLB final-score build complete.")


if __name__ == "__main__":
    main()


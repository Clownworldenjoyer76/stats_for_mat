#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/05_final_scores/01_mlb_results_grade.py

from datetime import datetime, UTC
from pathlib import Path
import csv
import sys

import pandas as pd

SELECT_DIR = Path("docs/win/baseball/mlb/04_select")
SCORE_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/final_scores")
OUTPUT_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/graded")
DAILY_DIR = OUTPUT_DIR / "daily"
UNMATCHED_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/unmatched")
AUDIT_DIR = Path("docs/win/baseball/mlb/05_final_scores/results/audit")
ERROR_DIR = Path("docs/win/baseball/mlb/errors/05_final_scores")

for directory in [ERROR_DIR, OUTPUT_DIR, DAILY_DIR, UNMATCHED_DIR, AUDIT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG = ERROR_DIR / "mlb_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "mlb_results_grade_summary.txt"

UNMATCHED_SELECTED_FILE = UNMATCHED_DIR / "MLB_unmatched_selected_bets.csv"
NOT_FINAL_SELECTED_FILE = UNMATCHED_DIR / "MLB_not_final_selected_bets.csv"
POSTPONED_CANCELED_FILE = UNMATCHED_DIR / "MLB_postponed_canceled_games.csv"
BLANK_SCORE_GAME_ID_FILE = UNMATCHED_DIR / "blank_final_score_game_ids_MLB.csv"

RECONCILIATION_AUDIT_FILE = AUDIT_DIR / "selected_vs_graded_reconciliation.csv"
DUPLICATE_AUDIT_FILE = AUDIT_DIR / "grading_duplicate_audit.csv"
VALIDATION_AUDIT_FILE = AUDIT_DIR / "graded_output_validation_audit.csv"
RESULT_COUNTS_FILE = AUDIT_DIR / "grading_result_counts.csv"
SPOT_CHECK_FILE = AUDIT_DIR / "grading_spot_check.csv"

OUTPUT_COLS = [
    "game_id", "sport", "league", "game_date", "game_time",
    "home_team", "away_team", "market_type", "bet_side", "line",
    "take_bet", "dk_odds_american", "model_prob", "ev", "kelly",
    "low_confidence", "gamePk", "gameNumber", "game_status",
    "final_scores_generated_at", "final_home_score", "final_away_score",
    "final_total", "home_run_line", "away_run_line", "total", "bet_result",
]

UNMATCHED_COLS = [
    "unmatched_reason", "game_id", "sport", "league", "game_date",
    "game_time", "home_team", "away_team", "market_type", "bet_side",
    "line", "take_bet", "dk_odds_american", "model_prob", "ev",
    "kelly", "low_confidence", "source_file",
]

RECONCILIATION_COLS = [
    "game_date", "selected_rows", "graded_rows", "unmatched_rows",
    "missing_final_score_rows", "missing_game_id_rows", "future_game_rows",
    "postponed_rows", "canceled_rows", "game_not_final_rows",
    "unknown_game_status_rows", "other_unmatched_rows", "status",
]

DUPLICATE_AUDIT_COLS = [
    "duplicate_scope", "game_date", "game_id", "market_type", "bet_side",
    "line", "duplicate_count", "identical_duplicate", "action_taken",
    "failure_reason", "source_files",
]

REQUIRED_SELECTED_COLUMNS = [
    "game_id", "game_date", "market_type", "bet_side",
    "line", "dk_odds_american",
]

REQUIRED_SCORE_COLUMNS = [
    "game_id", "game_date", "final_home_score", "final_away_score",
]

SELECTED_DUP_KEY = ["game_id", "market_type", "bet_side", "line"]
SCORE_DUP_KEY = ["game_id"]
VALID_RESULTS = {"Win", "Loss", "Push"}
KNOWN_UNMATCHED_REASONS = {
    "missing_final_score",
    "missing_game_id",
    "future_game",
    "postponed",
    "canceled",
    "game_not_final",
    "unknown_game_status",
}


def now_utc():
    return datetime.now(UTC).isoformat()


def reset_logs():
    GRADE_ERROR_LOG.write_text("", encoding="utf-8")
    GRADE_SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(message):
    with GRADE_ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def log_summary(message):
    with GRADE_SUMMARY_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def duplicate_columns(columns):
    seen = set()
    duplicates = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def read_header_columns(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def validate_no_duplicate_header(path, label):
    duplicates = duplicate_columns(read_header_columns(path))
    if duplicates:
        raise ValueError(f"{label} has duplicate header columns: {duplicates}")


def validate_no_duplicate_columns(frame, label):
    duplicates = duplicate_columns(list(frame.columns))
    if duplicates:
        raise ValueError(f"{label} has duplicate columns: {duplicates}")


def validate_required_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def write_csv_checked(frame, path, label):
    validate_no_duplicate_columns(frame, label)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_empty_csv(path, columns, label):
    write_csv_checked(pd.DataFrame(columns=columns), path, label)


def safe_read(path, required_columns=None, label=None):
    path = Path(path)
    read_label = label or str(path)
    try:
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()

        validate_no_duplicate_header(path, read_label)
        frame = pd.read_csv(path, dtype=str)

        if frame is None or frame.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()

        validate_no_duplicate_columns(frame, read_label)

        if required_columns:
            validate_required_columns(frame, required_columns, read_label)

        return frame.apply(
            lambda column: column.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
        )
    except Exception as error:
        log_error(f"READ/SCHEMA ERROR | {path} | {error}")
        return pd.DataFrame()


def normalize_date(value):
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.lower() in {"", "nan", "none", "nat"}:
        return ""
    return raw.replace("-", "_")


def clean_game_id(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def blank_mask(series):
    values = series.fillna("").astype(str).str.strip().str.lower()
    return values.isin({"", "nan", "none", "nat"})


def normalize_game_status(value):
    raw = "" if pd.isna(value) else str(value).strip().lower()

    if raw in {"final", "game over", "completed", "complete"}:
        return "final"
    if raw in {"postponed", "ppd"}:
        return "postponed"
    if raw in {"canceled", "cancelled"}:
        return "canceled"
    if raw == "suspended":
        return "suspended"
    if raw == "delayed":
        return "delayed"
    if raw in {"in progress", "in_progress", "live", "active"}:
        return "in_progress"
    if raw in {"scheduled", "pre-game", "pregame", "preview"}:
        return "scheduled"
    if raw in {"", "nan", "none", "nat"}:
        return "unknown"

    return raw.replace(" ", "_")


def enforce_columns(frame, columns):
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    return output[columns].copy()


def enforce_output_cols(frame):
    return enforce_columns(frame, OUTPUT_COLS)


def enforce_unmatched_cols(frame):
    return enforce_columns(frame, UNMATCHED_COLS)


def normalize_unmatched_selected_rows(frame):
    if frame.empty:
        return frame.copy()

    output = frame.copy()
    selected_preferred = [
        "sport", "league", "game_date", "game_time",
        "home_team", "away_team", "source_file", "take_bet",
    ]

    for base in selected_preferred:
        bet_column = f"{base}_bet"
        score_column = f"{base}_score"

        if bet_column in output.columns:
            bet_values = output[bet_column]
            if base in output.columns:
                use_bet = ~blank_mask(bet_values)
                output.loc[use_bet, base] = bet_values.loc[use_bet]
            else:
                output[base] = bet_values
        elif base not in output.columns and score_column in output.columns:
            output[base] = output[score_column]

    if "game_date" in output.columns:
        output["game_date"] = output["game_date"].apply(normalize_date)

    return output


def write_unmatched(frame):
    if frame.empty:
        make_empty_csv(
            UNMATCHED_SELECTED_FILE,
            UNMATCHED_COLS,
            "empty unmatched selected bets output",
        )
        return None

    output = normalize_unmatched_selected_rows(frame)
    output = enforce_unmatched_cols(output)
    write_csv_checked(
        output,
        UNMATCHED_SELECTED_FILE,
        "unmatched selected bets output",
    )
    return UNMATCHED_SELECTED_FILE


def duplicate_audit_row(
    scope,
    group,
    identical,
    action_taken,
    failure_reason,
    source_files,
):
    first = group.iloc[0]
    return {
        "duplicate_scope": scope,
        "game_date": first.get("game_date", ""),
        "game_id": first.get("game_id", ""),
        "market_type": first.get("market_type", ""),
        "bet_side": first.get("bet_side", ""),
        "line": first.get("line", ""),
        "duplicate_count": len(group),
        "identical_duplicate": str(bool(identical)),
        "action_taken": action_taken,
        "failure_reason": failure_reason,
        "source_files": source_files,
    }


def validate_and_collapse_duplicates(frame, key_columns, scope, compare_columns=None):
    if frame.empty:
        return frame.copy(), [], True

    missing_keys = [
        column for column in key_columns if column not in frame.columns
    ]
    if missing_keys:
        log_error(
            f"DUPLICATE VALIDATION FAILED | "
            f"scope={scope} missing_key_cols={missing_keys}"
        )
        return frame.copy(), [], False

    duplicate_rows = frame[
        frame.duplicated(subset=key_columns, keep=False)
    ].copy()

    if duplicate_rows.empty:
        return frame.copy(), [], True

    clean_parts = [
        frame[~frame.duplicated(subset=key_columns, keep=False)].copy()
    ]
    audit_rows = []
    success = True

    for _, group in duplicate_rows.groupby(key_columns, dropna=False):
        source_series = group.get(
            "source_file",
            pd.Series("", index=group.index, dtype=str),
        )
        source_files = ",".join(
            sorted(
                {
                    value
                    for value in source_series.fillna("").astype(str)
                    if value
                }
            )
        )

        available_compare_columns = [
            column
            for column in (compare_columns or list(group.columns))
            if column in group.columns
        ]

        comparable = (
            group[available_compare_columns]
            .fillna("")
            .astype(str)
        )
        identical = comparable.drop_duplicates().shape[0] == 1

        if identical:
            clean_parts.append(group.head(1).copy())
            audit_rows.append(
                duplicate_audit_row(
                    scope,
                    group,
                    True,
                    "collapsed_identical_duplicate",
                    "",
                    source_files,
                )
            )
        else:
            success = False
            audit_rows.append(
                duplicate_audit_row(
                    scope,
                    group,
                    False,
                    "hard_fail",
                    "conflicting_duplicate_rows",
                    source_files,
                )
            )

    cleaned = pd.concat(clean_parts, ignore_index=True)
    return cleaned, audit_rows, success


def write_duplicate_audit(rows):
    output = pd.DataFrame(rows, columns=DUPLICATE_AUDIT_COLS)
    write_csv_checked(
        output,
        DUPLICATE_AUDIT_FILE,
        "grading duplicate audit",
    )


def audit_and_drop_blank_score_game_ids(scores):
    blank = blank_mask(scores["game_id"])
    blank_scores = scores[blank].copy()
    clean_scores = scores[~blank].copy()

    if blank_scores.empty:
        make_empty_csv(
            BLANK_SCORE_GAME_ID_FILE,
            list(scores.columns),
            "empty blank final-score game_id audit",
        )
    else:
        sort_columns = [
            column
            for column in ["game_date", "away_team", "home_team"]
            if column in blank_scores.columns
        ]
        if sort_columns:
            blank_scores = blank_scores.sort_values(
                sort_columns,
                na_position="last",
            )

        write_csv_checked(
            blank_scores,
            BLANK_SCORE_GAME_ID_FILE,
            "blank final-score game_id audit",
        )
        log_summary(
            f"FINAL SCORE BLANK GAME_ID ROWS DROPPED | "
            f"rows={len(blank_scores)} | audit={BLANK_SCORE_GAME_ID_FILE}"
        )

    return clean_scores, len(blank_scores)


def determine_outcome(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side = str(row.get("bet_side", "")).strip().lower()
        away_score = float(row["final_away_score"])
        home_score = float(row["final_home_score"])

        if market == "moneyline":
            if away_score == home_score:
                return "Push"
            if side == "home":
                return "Win" if home_score > away_score else "Loss"
            if side == "away":
                return "Win" if away_score > home_score else "Loss"

        if market == "run_line":
            line = float(row.get("line", ""))
            if side == "home":
                difference = home_score + line - away_score
            elif side == "away":
                difference = away_score + line - home_score
            else:
                return ""

            if abs(difference) < 1e-9:
                return "Push"
            return "Win" if difference > 0 else "Loss"

        if market == "total":
            line = float(row.get("line", ""))
            final_total = away_score + home_score

            if abs(final_total - line) < 1e-9:
                return "Push"
            if side == "over":
                return "Win" if final_total > line else "Loss"
            if side == "under":
                return "Win" if final_total < line else "Loss"

    except Exception as error:
        log_error(
            f"DETERMINE OUTCOME ERROR | "
            f"game_id={row.get('game_id', '')} "
            f"market_type={row.get('market_type', '')} "
            f"bet_side={row.get('bet_side', '')} | {error}"
        )

    return ""


def build_calculation(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side = str(row.get("bet_side", "")).strip().lower()
        away_score = float(row.get("final_away_score", ""))
        home_score = float(row.get("final_home_score", ""))
        result = str(row.get("bet_result", "")).strip().lower()

        if market == "moneyline":
            return (
                f"moneyline {side}: away_score={away_score:g}, "
                f"home_score={home_score:g} => {result}"
            )

        if market == "run_line":
            line = float(row.get("line", ""))
            selected_score = home_score if side == "home" else away_score
            opposing_score = away_score if side == "home" else home_score
            adjusted_score = selected_score + line
            return (
                f"run_line {side} {line:g}: "
                f"selected_score={selected_score:g}, "
                f"opposing_score={opposing_score:g}, "
                f"adjusted_score={adjusted_score:g} => {result}"
            )

        if market == "total":
            line = float(row.get("line", ""))
            final_total = away_score + home_score
            return (
                f"total {side} {line:g}: "
                f"final_total={final_total:g} vs line={line:g} => {result}"
            )

    except Exception as error:
        return f"calculation_error: {error}"

    return ""


def resolve_merge_columns(frame):
    output = frame.copy()

    score_fields = {
        "game_date", "game_time", "home_team", "away_team",
        "sport", "league", "final_home_score", "final_away_score",
        "final_total", "home_run_line", "away_run_line", "total",
        "gamePk", "gameNumber", "game_status",
        "final_scores_generated_at",
    }
    selected_fields = {
        "sport", "league", "game_date", "game_time",
        "home_team", "away_team", "source_file",
    }

    for base in score_fields:
        score_column = f"{base}_score"
        bet_column = f"{base}_bet"

        if score_column in output.columns:
            output[base] = output[score_column]
        elif base not in output.columns and bet_column in output.columns:
            output[base] = output[bet_column]

    for base in selected_fields:
        bet_column = f"{base}_bet"
        score_column = f"{base}_score"

        if base not in output.columns and bet_column in output.columns:
            output[base] = output[bet_column]
        elif base not in output.columns and score_column in output.columns:
            output[base] = output[score_column]

    columns_to_drop = []
    for column in output.columns:
        if column == "take_bet":
            continue
        if column.endswith("_bet"):
            base = column[:-4]
        elif column.endswith("_score"):
            base = column[:-6]
        else:
            continue

        if base in selected_fields or base in score_fields:
            columns_to_drop.append(column)

    output = output.drop(columns=columns_to_drop, errors="ignore")
    validate_no_duplicate_columns(output, "post-resolve graded rows")
    return output


def load_selected_bets():
    files = sorted(SELECT_DIR.glob("*MLB*.csv"))
    if not files:
        log_error(f"NO SELECT FILES FOUND IN {SELECT_DIR}")
        return pd.DataFrame(), []

    parts = []
    audit_rows = []

    for path in files:
        frame = safe_read(
            path,
            REQUIRED_SELECTED_COLUMNS,
            f"selected file {path.name}",
        )
        if frame.empty:
            continue

        frame["source_file"] = path.name
        frame["game_date"] = frame["game_date"].apply(normalize_date)
        frame["game_id"] = clean_game_id(frame["game_id"])

        compare_columns = [
            column
            for column in frame.columns
            if column != "selected_row_id"
        ]

        frame, duplicate_rows, success = validate_and_collapse_duplicates(
            frame,
            SELECTED_DUP_KEY,
            "daily_selected_bet_key",
            compare_columns,
        )
        audit_rows.extend(duplicate_rows)

        if not success:
            return pd.DataFrame(), audit_rows

        parts.append(frame)

    if not parts:
        log_error("ALL SELECT FILES EMPTY, UNREADABLE, OR SCHEMA-INVALID")
        return pd.DataFrame(), audit_rows

    bets = pd.concat(parts, ignore_index=True)
    validate_no_duplicate_columns(bets, "combined selected bets")

    bets["game_id"] = clean_game_id(bets["game_id"])
    bets["selected_row_id"] = range(len(bets))

    compare_columns = [
        "game_id", "game_date", "market_type", "bet_side", "line",
        "take_bet", "dk_odds_american", "model_prob", "ev", "kelly",
        "prob_for_ev", "prob_for_kelly",
    ]

    bets, duplicate_rows, success = validate_and_collapse_duplicates(
        bets,
        SELECTED_DUP_KEY,
        "combined_selected_bet_key",
        compare_columns,
    )
    audit_rows.extend(duplicate_rows)

    if not success:
        return pd.DataFrame(), audit_rows

    bets["selected_row_id"] = range(len(bets))
    return bets, audit_rows


def load_final_scores():
    files = sorted(SCORE_DIR.glob("*_final_scores_MLB.csv"))
    if not files:
        log_error(f"NO SCORE FILES FOUND IN {SCORE_DIR}")
        return pd.DataFrame(), []

    parts = []
    audit_rows = []

    for path in files:
        frame = safe_read(
            path,
            REQUIRED_SCORE_COLUMNS,
            f"score file {path.name}",
        )
        if frame.empty:
            continue

        frame["source_file"] = path.name
        frame["game_date"] = frame["game_date"].apply(normalize_date)
        parts.append(frame)

    if not parts:
        log_error("ALL SCORE FILES EMPTY, UNREADABLE, OR SCHEMA-INVALID")
        return pd.DataFrame(), audit_rows

    scores = pd.concat(parts, ignore_index=True)
    validate_no_duplicate_columns(scores, "combined final scores")

    scores["game_id"] = clean_game_id(scores["game_id"])

    if "game_status" not in scores.columns:
        scores["game_status"] = "unknown"

    scores["game_status"] = scores["game_status"].apply(
        normalize_game_status
    )

    scores, blank_count = audit_and_drop_blank_score_game_ids(scores)

    compare_columns = [
        "game_id", "game_date", "home_team", "away_team",
        "final_home_score", "final_away_score", "final_total",
        "gamePk", "gameNumber", "game_status",
    ]

    scores, duplicate_rows, success = validate_and_collapse_duplicates(
        scores,
        SCORE_DUP_KEY,
        "final_score_game_id",
        compare_columns,
    )
    audit_rows.extend(duplicate_rows)

    if not success:
        log_error(
            f"FINAL SCORE CONFLICTING DUPLICATES | "
            f"audit={DUPLICATE_AUDIT_FILE}"
        )
        return pd.DataFrame(), audit_rows

    log_summary(f"SCORE BLANK GAME_ID ROWS DROPPED: {blank_count}")
    return scores, audit_rows


def reason_for_status(status):
    normalized = normalize_game_status(status)

    if normalized == "postponed":
        return "postponed"
    if normalized == "canceled":
        return "canceled"
    if normalized == "unknown":
        return "unknown_game_status"

    return "game_not_final"


def build_non_final_reports(matched_rows):
    output_columns = [
        "game_date", "game_id", "gamePk", "gameNumber",
        "away_team", "home_team", "market_type", "bet_side",
        "line", "game_status", "unmatched_reason",
    ]
    game_columns = [
        "game_date", "game_id", "gamePk", "gameNumber",
        "away_team", "home_team", "game_status",
        "selected_rows_affected",
    ]

    if matched_rows.empty or "game_status" not in matched_rows.columns:
        make_empty_csv(
            NOT_FINAL_SELECTED_FILE,
            output_columns,
            "empty not-final selected bets output",
        )
        make_empty_csv(
            POSTPONED_CANCELED_FILE,
            game_columns,
            "empty postponed/canceled games output",
        )
        return pd.DataFrame()

    non_final = matched_rows[
        matched_rows["game_status"].apply(normalize_game_status) != "final"
    ].copy()

    if non_final.empty:
        make_empty_csv(
            NOT_FINAL_SELECTED_FILE,
            output_columns,
            "empty not-final selected bets output",
        )
        make_empty_csv(
            POSTPONED_CANCELED_FILE,
            game_columns,
            "empty postponed/canceled games output",
        )
        return pd.DataFrame()

    non_final = non_final.drop(columns=["_merge"], errors="ignore")

    selected_values = {}
    for base in [
        "sport", "league", "game_date", "game_time",
        "home_team", "away_team", "source_file", "take_bet",
    ]:
        bet_column = f"{base}_bet"
        if bet_column in non_final.columns:
            selected_values[base] = non_final[bet_column].copy()

    non_final = resolve_merge_columns(non_final)

    for base, values in selected_values.items():
        non_final[base] = values

    non_final["game_date"] = non_final["game_date"].apply(normalize_date)
    non_final["game_status"] = non_final["game_status"].apply(
        normalize_game_status
    )
    non_final["unmatched_reason"] = non_final["game_status"].apply(
        reason_for_status
    )

    report = enforce_columns(non_final, output_columns)
    write_csv_checked(
        report,
        NOT_FINAL_SELECTED_FILE,
        "not-final selected bets output",
    )

    postponed_canceled = non_final[
        non_final["game_status"].isin(["postponed", "canceled"])
    ].copy()

    if postponed_canceled.empty:
        make_empty_csv(
            POSTPONED_CANCELED_FILE,
            game_columns,
            "empty postponed/canceled games output",
        )
    else:
        group_columns = [
            "game_date", "game_id", "gamePk", "gameNumber",
            "away_team", "home_team", "game_status",
        ]
        grouped = (
            postponed_canceled.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name="selected_rows_affected")
        )
        write_csv_checked(
            grouped[game_columns],
            POSTPONED_CANCELED_FILE,
            "postponed/canceled games output",
        )

    return non_final


def build_unmatched(all_bets, blank_id_rows, missing_score_rows, non_final_rows):
    reason_parts = []

    for frame in [blank_id_rows, missing_score_rows, non_final_rows]:
        if (
            not frame.empty
            and "selected_row_id" in frame.columns
            and "unmatched_reason" in frame.columns
        ):
            reason_parts.append(
                frame[["selected_row_id", "unmatched_reason"]].copy()
            )

    if not reason_parts:
        return pd.DataFrame()

    reasons = pd.concat(reason_parts, ignore_index=True)

    duplicate_ids = reasons[
        reasons.duplicated(subset=["selected_row_id"], keep=False)
    ]

    if not duplicate_ids.empty:
        values = duplicate_ids["selected_row_id"].astype(str).tolist()
        raise ValueError(
            f"duplicate unmatched selected_row_id values: {values}"
        )

    return pd.merge(
        all_bets,
        reasons,
        on="selected_row_id",
        how="inner",
        validate="one_to_one",
    )


def write_reconciliation(all_bets, final, unmatched):
    dates = set()

    for frame in [all_bets, final, unmatched]:
        if not frame.empty and "game_date" in frame.columns:
            dates.update(
                value
                for value in frame["game_date"]
                .fillna("")
                .astype(str)
                .map(normalize_date)
                if value
            )

    rows = []

    for game_date in sorted(dates):
        selected_date = all_bets[
            all_bets["game_date"].astype(str) == game_date
        ]
        graded_date = final[
            final["game_date"].astype(str) == game_date
        ]

        if unmatched.empty or "game_date" not in unmatched.columns:
            unmatched_date = pd.DataFrame()
        else:
            unmatched_date = unmatched[
                unmatched["game_date"].astype(str) == game_date
            ]

        if unmatched_date.empty:
            reasons = pd.Series(dtype=str)
        else:
            reasons = (
                unmatched_date["unmatched_reason"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        def count_reason(reason):
            return int((reasons == reason).sum())

        selected_count = len(selected_date)
        graded_count = len(graded_date)
        unmatched_count = len(unmatched_date)
        other_count = int((~reasons.isin(KNOWN_UNMATCHED_REASONS)).sum())
        accounted_count = graded_count + unmatched_count

        rows.append({
            "game_date": game_date,
            "selected_rows": selected_count,
            "graded_rows": graded_count,
            "unmatched_rows": unmatched_count,
            "missing_final_score_rows": count_reason("missing_final_score"),
            "missing_game_id_rows": count_reason("missing_game_id"),
            "future_game_rows": count_reason("future_game"),
            "postponed_rows": count_reason("postponed"),
            "canceled_rows": count_reason("canceled"),
            "game_not_final_rows": count_reason("game_not_final"),
            "unknown_game_status_rows": count_reason("unknown_game_status"),
            "other_unmatched_rows": other_count,
            "status": (
                "ok"
                if selected_count == accounted_count and other_count == 0
                else "review"
            ),
        })

    reconciliation = pd.DataFrame(rows, columns=RECONCILIATION_COLS)
    write_csv_checked(
        reconciliation,
        RECONCILIATION_AUDIT_FILE,
        "selected vs graded reconciliation audit",
    )
    return reconciliation


def write_result_counts(final):
    columns = [
        "market_type", "wins", "losses",
        "pushes", "blank_results", "total_rows",
    ]

    if final.empty:
        make_empty_csv(
            RESULT_COUNTS_FILE,
            columns,
            "empty grading result counts",
        )
        return pd.DataFrame(columns=columns)

    rows = []

    for market, group in final.groupby("market_type", dropna=False):
        results = group["bet_result"].fillna("").astype(str).str.strip()
        rows.append({
            "market_type": market,
            "wins": int((results == "Win").sum()),
            "losses": int((results == "Loss").sum()),
            "pushes": int((results == "Push").sum()),
            "blank_results": int((results == "").sum()),
            "total_rows": len(group),
        })

    output = pd.DataFrame(rows, columns=columns)
    write_csv_checked(
        output,
        RESULT_COUNTS_FILE,
        "grading result counts",
    )
    return output


def write_spot_check(final):
    columns = [
        "game_date", "game_id", "market_type", "bet_side", "line",
        "final_away_score", "final_home_score", "final_total",
        "result", "calculation",
    ]

    if final.empty:
        make_empty_csv(
            SPOT_CHECK_FILE,
            columns,
            "empty grading spot check",
        )
        return

    output = final.copy()
    output["result"] = output["bet_result"]
    output["calculation"] = output.apply(build_calculation, axis=1)
    output = enforce_columns(output, columns)

    write_csv_checked(
        output,
        SPOT_CHECK_FILE,
        "grading spot check",
    )


def validate_graded_output(final):
    audit_rows = []

    def add_validation(validation, column, bad_rows, status, notes):
        audit_rows.append({
            "validation": validation,
            "column": column,
            "bad_rows": bad_rows,
            "status": status,
            "notes": notes,
        })

    if "take_bet" not in final.columns:
        add_validation(
            "required_column_exists",
            "take_bet",
            "",
            "fail",
            "take_bet column missing after suffix cleanup",
        )
    else:
        add_validation(
            "required_column_exists",
            "take_bet",
            0,
            "ok",
            "take_bet column preserved",
        )

    required_nonblank = [
        "game_id", "market_type", "bet_side",
        "dk_odds_american", "bet_result",
    ]

    for column in required_nonblank:
        if column not in final.columns:
            add_validation(
                "required_nonblank",
                column,
                "",
                "fail",
                "column missing",
            )
            continue

        bad_rows = int(blank_mask(final[column]).sum())
        add_validation(
            "required_nonblank",
            column,
            bad_rows,
            "fail" if bad_rows else "ok",
            "blank values are not allowed",
        )

    if "bet_result" not in final.columns:
        invalid_count = len(final)
    else:
        invalid_count = int(
            (
                ~final["bet_result"]
                .fillna("")
                .astype(str)
                .str.strip()
                .isin(VALID_RESULTS)
            ).sum()
        )

    add_validation(
        "valid_bet_result",
        "bet_result",
        invalid_count,
        "fail" if invalid_count else "ok",
        "allowed values are Win, Loss, Push",
    )

    score_columns = [
        "final_home_score", "final_away_score", "final_total",
    ]

    for column in score_columns:
        if column not in final.columns:
            add_validation(
                "numeric_score",
                column,
                "",
                "fail",
                "column missing",
            )
            continue

        numeric = pd.to_numeric(final[column], errors="coerce")
        bad_rows = int(numeric.isna().sum())
        add_validation(
            "numeric_score",
            column,
            bad_rows,
            "fail" if bad_rows else "ok",
            "score values must be numeric",
        )

    if all(column in final.columns for column in score_columns):
        home = pd.to_numeric(final["final_home_score"], errors="coerce")
        away = pd.to_numeric(final["final_away_score"], errors="coerce")
        total = pd.to_numeric(final["final_total"], errors="coerce")
        mismatch = (
            home.notna()
            & away.notna()
            & total.notna()
            & ((home + away - total).abs() > 1e-9)
        )
        mismatch_count = int(mismatch.sum())
    else:
        mismatch_count = len(final)

    add_validation(
        "final_total_matches_scores",
        "final_total",
        mismatch_count,
        "fail" if mismatch_count else "ok",
        "final_total must equal final_home_score plus final_away_score",
    )

    audit = pd.DataFrame(audit_rows)
    write_csv_checked(
        audit,
        VALIDATION_AUDIT_FILE,
        "graded output validation audit",
    )

    failures = audit[audit["status"] == "fail"]

    if failures.empty:
        return True

    log_error(
        f"GRADED OUTPUT VALIDATION FAILED | "
        f"audit={VALIDATION_AUDIT_FILE}"
    )

    for _, row in failures.iterrows():
        log_error(
            f"VALIDATION FAILURE | "
            f"validation={row.get('validation')} "
            f"column={row.get('column')} "
            f"bad_rows={row.get('bad_rows')} "
            f"notes={row.get('notes')}"
        )

    return False


def grade_league():
    duplicate_audit_rows = []

    all_bets, selected_duplicates = load_selected_bets()
    duplicate_audit_rows.extend(selected_duplicates)

    if all_bets.empty:
        write_duplicate_audit(duplicate_audit_rows)
        return False

    all_scores, score_duplicates = load_final_scores()
    duplicate_audit_rows.extend(score_duplicates)
    write_duplicate_audit(duplicate_audit_rows)

    if all_scores.empty:
        return False

    blank_selected_ids = all_bets[blank_mask(all_bets["game_id"])].copy()
    valid_selected = all_bets[~blank_mask(all_bets["game_id"])].copy()

    if not blank_selected_ids.empty:
        blank_selected_ids["unmatched_reason"] = "missing_game_id"

    log_summary(f"BET cols: {list(all_bets.columns)}")
    log_summary(f"SCORE cols: {list(all_scores.columns)}")
    log_summary(f"SELECTED ROWS: {len(all_bets)}")
    log_summary(
        f"SELECTED BLANK GAME_ID ROWS: {len(blank_selected_ids)}"
    )
    log_summary(
        f"SCORE ROWS AFTER BLANK GAME_ID DROP: {len(all_scores)}"
    )

    merged_all = pd.merge(
        valid_selected,
        all_scores,
        on="game_id",
        how="left",
        suffixes=("_bet", "_score"),
        indicator=True,
        validate="many_to_one",
    )

    matched_rows = merged_all[merged_all["_merge"] == "both"].copy()
    missing_scores = merged_all[
        merged_all["_merge"] == "left_only"
    ].copy()

    if not missing_scores.empty:
        missing_scores["unmatched_reason"] = "missing_final_score"

    non_final = build_non_final_reports(matched_rows)

    if "game_status" in matched_rows.columns:
        matched_final = matched_rows[
            matched_rows["game_status"].apply(normalize_game_status)
            == "final"
        ].copy()
    else:
        matched_final = pd.DataFrame()

    try:
        unmatched = build_unmatched(
            all_bets,
            blank_selected_ids,
            missing_scores,
            non_final,
        )
    except Exception as error:
        log_error(f"UNMATCHED BUILD FAILED | {error}")
        return False

    unmatched_path = write_unmatched(unmatched)

    merged = matched_final.drop(
        columns=["_merge"],
        errors="ignore",
    ).copy()

    log_summary(f"MERGED ON game_id | rows={len(merged)}")
    log_summary(f"UNMATCHED SELECTED ROWS: {len(unmatched)}")

    if unmatched_path:
        log_summary(
            f"UNMATCHED SELECTED BETS WRITTEN | "
            f"rows={len(unmatched)} | out={unmatched_path}"
        )

    if merged.empty:
        log_error("MERGE EMPTY")
        write_reconciliation(all_bets, pd.DataFrame(), unmatched)
        return False

    merged = resolve_merge_columns(merged)
    merged["game_date"] = merged["game_date"].apply(normalize_date)
    merged["game_status"] = merged["game_status"].apply(
        normalize_game_status
    )
    merged["bet_result"] = merged.apply(determine_outcome, axis=1)

    final = enforce_output_cols(merged)

    if not validate_graded_output(final):
        return False

    result_counts = write_result_counts(final)
    write_spot_check(final)

    master_path = OUTPUT_DIR / "MLB_final.csv"
    write_csv_checked(final, master_path, "MLB graded master output")

    selected_count = len(all_bets)
    graded_count = len(final)
    unmatched_count = len(unmatched)

    log_summary(
        f"MLB MASTER BUILT | ROWS={graded_count} | OUT={master_path}"
    )
    log_summary(
        f"SELECTED VS GRADED | selected={selected_count} "
        f"graded={graded_count} unmatched={unmatched_count}"
    )

    reconciliation = write_reconciliation(
        all_bets,
        final,
        unmatched,
    )

    review_rows = reconciliation[
        reconciliation["status"] == "review"
    ]

    if not review_rows.empty:
        log_error(
            f"SELECTED VS GRADED RECONCILIATION HAS REVIEW ROWS | "
            f"audit={RECONCILIATION_AUDIT_FILE}"
        )
        return False

    if selected_count != graded_count + unmatched_count:
        log_error(
            f"GLOBAL RECONCILIATION FAILED | "
            f"selected={selected_count} graded={graded_count} "
            f"unmatched={unmatched_count}"
        )
        return False

    blank_results = int(
        pd.to_numeric(
            result_counts["blank_results"],
            errors="coerce",
        ).fillna(0).sum()
    )

    if blank_results:
        log_error(
            f"BLANK BET RESULTS FOUND | rows={blank_results} | "
            f"audit={RESULT_COUNTS_FILE}"
        )
        return False

    for game_date, group in final.groupby("game_date"):
        date_string = normalize_date(game_date)
        daily_output = enforce_output_cols(group.copy())
        daily_path = DAILY_DIR / f"{date_string}_MLB_final.csv"

        write_csv_checked(
            daily_output,
            daily_path,
            f"MLB graded daily output {date_string}",
        )

        counts = group["bet_result"].value_counts().to_dict()
        log_summary(
            f"MLB DAILY | DATE={date_string} | "
            f"ROWS={len(daily_output)} | RESULTS={counts}"
        )

    return True


def main():
    reset_logs()
    log_summary("START 01_mlb_results_grade.py")

    try:
        success = grade_league()
    except Exception as error:
        log_error(f"UNHANDLED ERROR | {type(error).__name__}: {error}")
        success = False

    log_summary("END 01_mlb_results_grade.py")

    if not success:
        print("MLB grading completed with errors. Check logs.")
        sys.exit(1)

    print("MLB grading complete.")


if __name__ == "__main__":
    main()

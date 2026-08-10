#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/clean_basketball_inputs.py
#
# Reads originals from:
#   docs/win/basketball/00_intake/predictions/{league}/
#   docs/win/basketball/00_intake/sportsbook/{league}/
#
# Writes cleaned copies to:
#   docs/win/basketball/00_intake/predictions/predictions_cleaned/{league}/
#   docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/{league}/
#
# Originals are never mutated. Filenames are preserved.
#
# Logs:
#   docs/win/basketball/errors/00_intake/clean_basketball_inputs.txt
#   docs/win/basketball/errors/00_intake/clean_basketball_inputs_nba.txt
#   docs/win/basketball/errors/00_intake/clean_basketball_inputs_ncaam.txt
#   docs/win/basketball/errors/00_intake/clean_basketball_inputs_wnba.txt

import csv
import traceback
from pathlib import Path
from datetime import datetime

# =========================
# PATHS
# =========================

PREDICTION_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/predictions/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/predictions/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/predictions/wnba"),
}

SPORTSBOOK_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/sportsbook/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/sportsbook/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/sportsbook/wnba"),
}

CLEANED_PREDICTION_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/wnba"),
}

CLEANED_SPORTSBOOK_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/wnba"),
}

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

MASTER_LOG_FILE = ERROR_DIR / "clean_basketball_inputs.txt"

LEAGUE_LOG_FILES = {
    "NBA": ERROR_DIR / "clean_basketball_inputs_nba.txt",
    "NCAAM": ERROR_DIR / "clean_basketball_inputs_ncaam.txt",
    "WNBA": ERROR_DIR / "clean_basketball_inputs_wnba.txt",
}

# =========================
# SETTINGS
# =========================

ODDS_CHECKS = [
    ("home_dk_moneyline_decimal", "away_dk_moneyline_decimal", "ML"),
    ("dk_total_over_decimal",     "dk_total_under_decimal",    "TOTAL"),
    ("home_dk_spread_decimal",    "away_dk_spread_decimal",    "SPREAD"),
]

MARKET_FIELDS = {
    "ML": [
        "home_dk_moneyline_american",
        "away_dk_moneyline_american",
        "home_dk_moneyline_decimal",
        "away_dk_moneyline_decimal",
        "home_decimal",
        "away_decimal",
    ],
    "TOTAL": [
        "total",
        "dk_total_over_american",
        "dk_total_under_american",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ],
    "SPREAD": [
        "home_spread",
        "away_spread",
        "home_dk_spread_american",
        "away_dk_spread_american",
        "home_dk_spread_decimal",
        "away_dk_spread_decimal",
    ],
}

SPREAD_OUTLIER_MAX = 25.0
TOTAL_OUTLIER_MAX = 40.0

MARGIN_BIAS = {
    "NBA": 0.4,
    "NCAAM": 0.6,
    "WNBA": 0.5,
}

TOTAL_BIAS = {
    "NBA": 0.4,
    "NCAAM": 1.2,
    "WNBA": 0.0,
}

BIAS_FLAG_COLUMN = "bias_applied"
BIAS_FLAG_VALUE = "1"

# =========================
# LOGGING
# =========================

def init_logs() -> None:
    run_line = f"=== clean_basketball_inputs RUN {datetime.now().isoformat()} ===\n"

    with open(MASTER_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(run_line)

    for league, path in LEAGUE_LOG_FILES.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(run_line)
            f.write(f"=== LEAGUE LOG: {league} ===\n")


def log_master(msg: str) -> None:
    with open(MASTER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def log_league(league: str, msg: str) -> None:
    path = LEAGUE_LOG_FILES.get(league)
    if path is None:
        log_master(f"{league} | {msg}")
        return

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# HELPERS
# =========================

def to_float(value):
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def odds_malformed_reason(raw_a, raw_b):
    raw_a_str = "" if raw_a is None else str(raw_a).strip()
    raw_b_str = "" if raw_b is None else str(raw_b).strip()

    if raw_a_str == "" and raw_b_str == "":
        return "BOTH_BLANK"

    if raw_a_str == "" or raw_b_str == "":
        return "MISSING_PAIRED_ODDS"

    dec_a = to_float(raw_a)
    dec_b = to_float(raw_b)

    if dec_a is None or dec_b is None:
        return "NON_NUMERIC"

    if dec_a <= 1 or dec_b <= 1:
        return "DECIMAL_LESS_THAN_OR_EQUAL_1"

    return None


def row_key(row):
    game_id = str(row.get("game_id", "")).strip()
    if game_id:
        return game_id

    return "|".join([
        str(row.get("game_date", "")).strip(),
        str(row.get("home_team", "")).strip(),
        str(row.get("away_team", "")).strip(),
    ])


def read_csv(path: Path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_files(folder: Path, league: str):
    if not folder.exists():
        log_league(league, f"WARN | Missing folder: {folder}")
        return []
    return sorted(folder.rglob("*.csv"))


def book_home_spread_to_model_margin(home_spread):
    """
    Converts sportsbook home_spread into the same convention as:
      model_spread = home_projected_points - away_projected_points

    Example:
      sportsbook home_spread = -17.0 means home favored by 17
      model-margin equivalent = +17.0

      sportsbook home_spread = +6.5 means home underdog by 6.5
      model-margin equivalent = -6.5
    """
    if home_spread is None:
        return None
    return -home_spread


def blank_market_fields(row, fieldnames, market):
    blanked = []

    for col in MARKET_FIELDS.get(market, []):
        if col in fieldnames:
            old = row.get(col, "")
            if str(old).strip() != "":
                blanked.append(col)
            row[col] = ""

    return blanked


def add_market_action(actions, league, path, key, market, reason):
    actions.setdefault(league, {})
    actions[league].setdefault(path, {})
    actions[league][path].setdefault(key, {})
    actions[league][path][key][market] = reason


# =========================
# LOAD ALL ORIGINALS INTO MEMORY
# =========================

def load_all(dir_map):
    loaded = {}

    for league, folder in dir_map.items():
        loaded[league] = {}

        for path in csv_files(folder, league):
            fieldnames, rows = read_csv(path)
            loaded[league][path] = [fieldnames, rows]

        log_league(
            league,
            f"LOADED | folder={folder} files={len(loaded[league])}"
        )

    return loaded


# =========================
# STEP 1: BLANK MALFORMED MARKET ODDS ONLY
# =========================

def blank_bad_market_odds(book_files):
    blanked_by_market = {
        league: {market: 0 for _, _, market in ODDS_CHECKS}
        for league in book_files
    }

    for league, files in book_files.items():
        for path, data in files.items():
            fieldnames, rows = data
            fieldset = set(fieldnames)

            file_blanked = 0

            for row in rows:
                key = row_key(row)

                for col_a, col_b, market in ODDS_CHECKS:
                    if col_a not in fieldset or col_b not in fieldset:
                        continue

                    raw_a = row.get(col_a)
                    raw_b = row.get(col_b)
                    reason = odds_malformed_reason(raw_a, raw_b)

                    if reason is None:
                        continue

                    blanked_cols = blank_market_fields(row, fieldnames, market)
                    blanked_by_market[league][market] += 1
                    file_blanked += 1

                    log_league(
                        league,
                        f"BLANK_BAD_{market}_MALFORMED | {path} | {key} | "
                        f"reason={reason} a={raw_a!r} b={raw_b!r} "
                        f"blanked_cols={blanked_cols}"
                    )

            if file_blanked:
                log_league(
                    league,
                    f"MARKET_BLANKED | {path} | bad_market_odds_blank_events={file_blanked}"
                )

    return blanked_by_market


# =========================
# STEP 2: BLANK BAD MARKET LINES ONLY
# =========================

def blank_bad_market_lines(book_files):
    blanked_by_market = {
        league: {"TOTAL": 0, "SPREAD": 0}
        for league in book_files
    }

    for league, files in book_files.items():
        for path, data in files.items():
            fieldnames, rows = data
            fieldset = set(fieldnames)

            file_blanked = 0

            for row in rows:
                key = row_key(row)

                if "total" in fieldset:
                    book_total = to_float(row.get("total"))

                    if book_total is not None and book_total <= 0:
                        blanked_cols = blank_market_fields(row, fieldnames, "TOTAL")
                        blanked_by_market[league]["TOTAL"] += 1
                        file_blanked += 1

                        log_league(
                            league,
                            f"BLANK_BAD_TOTAL_LINE | {path} | {key} | "
                            f"reason=TOTAL_LESS_THAN_OR_EQUAL_0 total={book_total} "
                            f"blanked_cols={blanked_cols}"
                        )

                if "home_spread" in fieldset and "away_spread" in fieldset:
                    home_spread = to_float(row.get("home_spread"))
                    away_spread = to_float(row.get("away_spread"))

                    if home_spread is None and away_spread is None:
                        continue

                    if home_spread is None or away_spread is None:
                        blanked_cols = blank_market_fields(row, fieldnames, "SPREAD")
                        blanked_by_market[league]["SPREAD"] += 1
                        file_blanked += 1

                        log_league(
                            league,
                            f"BLANK_BAD_SPREAD_LINE | {path} | {key} | "
                            f"reason=MISSING_PAIRED_SPREAD_LINE "
                            f"home_spread={row.get('home_spread')!r} away_spread={row.get('away_spread')!r} "
                            f"blanked_cols={blanked_cols}"
                        )

            if file_blanked:
                log_league(
                    league,
                    f"MARKET_BLANKED | {path} | bad_market_line_blank_events={file_blanked}"
                )

    return blanked_by_market


# =========================
# STEP 3: BUILD INDEXES
# =========================

def build_pred_index(pred_files):
    index = {}

    for league, files in pred_files.items():
        for path, (fieldnames, rows) in files.items():
            for row in rows:
                key = row_key(row)
                if key:
                    index[key] = {
                        "league": league,
                        "path": path,
                        "row": row,
                    }

    return index


def build_book_index(book_files):
    index = {}

    for league, files in book_files.items():
        for path, (fieldnames, rows) in files.items():
            for row in rows:
                key = row_key(row)
                if key:
                    index.setdefault(key, []).append({
                        "league": league,
                        "path": path,
                        "fieldnames": fieldnames,
                        "row": row,
                    })

    return index


# =========================
# STEP 4: FIND MODEL VS BOOK MARKET OUTLIERS
# =========================

def find_market_outlier_actions(pred_index, book_index):
    actions = {}

    outlier_counts = {
        "NBA": {"spread": 0, "total": 0},
        "NCAAM": {"spread": 0, "total": 0},
        "WNBA": {"spread": 0, "total": 0},
    }

    for key, pred_item in pred_index.items():
        if key not in book_index:
            continue

        league = pred_item["league"]
        pred = pred_item["row"]

        home_proj = to_float(pred.get("home_projected_points"))
        away_proj = to_float(pred.get("away_projected_points"))
        model_total = to_float(pred.get("total_projected_points"))

        for book_item in book_index[key]:
            book = book_item["row"]
            book_path = book_item["path"]

            book_home_spread = to_float(book.get("home_spread"))
            book_total = to_float(book.get("total"))

            if home_proj is not None and away_proj is not None and book_home_spread is not None:
                model_spread = home_proj - away_proj
                book_model_margin = book_home_spread_to_model_margin(book_home_spread)
                spread_diff = abs(model_spread - book_model_margin)

                if spread_diff > SPREAD_OUTLIER_MAX:
                    add_market_action(
                        actions,
                        league,
                        book_path,
                        key,
                        "SPREAD",
                        f"SPREAD_OUTLIER_DIFF_{round(spread_diff, 4)}",
                    )

                    outlier_counts[league]["spread"] += 1

                    log_league(
                        league,
                        f"BLANK_OUTLIER_SPREAD | {key} | {book_path} | "
                        f"model_spread_home_minus_away={round(model_spread, 4)} "
                        f"book_home_spread={book_home_spread} "
                        f"book_spread_converted_to_home_margin={round(book_model_margin, 4)} "
                        f"diff={round(spread_diff, 4)}"
                    )

            if model_total is not None and book_total is not None:
                total_diff = abs(model_total - book_total)

                if total_diff > TOTAL_OUTLIER_MAX:
                    add_market_action(
                        actions,
                        league,
                        book_path,
                        key,
                        "TOTAL",
                        f"TOTAL_OUTLIER_DIFF_{round(total_diff, 4)}",
                    )

                    outlier_counts[league]["total"] += 1

                    log_league(
                        league,
                        f"BLANK_OUTLIER_TOTAL | {key} | {book_path} | "
                        f"model_total={model_total} book_total={book_total} "
                        f"diff={round(total_diff, 4)}"
                    )

    for league, counts in outlier_counts.items():
        log_league(
            league,
            f"OUTLIER SUMMARY | spread_outliers={counts['spread']} total_outliers={counts['total']}"
        )

    return actions, outlier_counts


# =========================
# STEP 5: APPLY MARKET OUTLIER BLANKS
# =========================

def apply_market_outlier_actions(book_files, actions):
    blanked_by_league = {league: 0 for league in book_files}

    for league, files in book_files.items():
        league_actions = actions.get(league, {})

        for path, data in files.items():
            path_actions = league_actions.get(path, {})
            if not path_actions:
                continue

            fieldnames, rows = data
            file_blanked = 0

            for row in rows:
                key = row_key(row)
                market_actions = path_actions.get(key, {})

                if not market_actions:
                    continue

                for market, reason in market_actions.items():
                    blanked_cols = blank_market_fields(row, fieldnames, market)
                    blanked_by_league[league] += 1
                    file_blanked += 1

                    log_league(
                        league,
                        f"MARKET_OUTLIER_BLANKED | SPORTSBOOK | {path} | {key} | "
                        f"market={market} reason={reason} blanked_cols={blanked_cols}"
                    )

            if file_blanked:
                log_league(
                    league,
                    f"MARKET_BLANKED | SPORTSBOOK | {path} | outlier_market_blank_events={file_blanked}"
                )

    return blanked_by_league


# =========================
# STEP 6: APPLY PREDICTION BIASES
# =========================

def apply_prediction_biases(pred_files):
    stats = {
        league: {
            "files_with_biased_rows": 0,
            "rows_adjusted": 0,
            "rows_skipped_already_flagged": 0,
        }
        for league in pred_files
    }

    required = {
        "home_projected_points",
        "away_projected_points",
        "total_projected_points",
    }

    for league, files in pred_files.items():
        margin_bias = MARGIN_BIAS[league]
        total_bias = TOTAL_BIAS[league]
        margin_half = margin_bias / 2.0
        total_half = total_bias / 2.0

        for path, data in files.items():
            fieldnames, rows = data

            if not required.issubset(set(fieldnames)):
                log_league(league, f"WARN | Missing prediction columns, skipped: {path}")
                continue

            if BIAS_FLAG_COLUMN not in fieldnames:
                fieldnames = list(fieldnames) + [BIAS_FLAG_COLUMN]
                data[0] = fieldnames

            file_adjusted = 0
            file_skipped = 0

            for row in rows:
                if str(row.get(BIAS_FLAG_COLUMN, "")).strip() == BIAS_FLAG_VALUE:
                    file_skipped += 1
                    stats[league]["rows_skipped_already_flagged"] += 1
                    continue

                home = to_float(row.get("home_projected_points"))
                away = to_float(row.get("away_projected_points"))
                total = to_float(row.get("total_projected_points"))

                if home is None or away is None or total is None:
                    continue

                new_home = round(home - margin_half - total_half, 2)
                new_away = round(away + margin_half - total_half, 2)
                new_total = round(total - total_bias, 2)

                row["home_projected_points"] = f"{new_home:.2f}"
                row["away_projected_points"] = f"{new_away:.2f}"
                row["total_projected_points"] = f"{new_total:.2f}"
                row[BIAS_FLAG_COLUMN] = BIAS_FLAG_VALUE

                file_adjusted += 1
                stats[league]["rows_adjusted"] += 1

            if file_adjusted:
                stats[league]["files_with_biased_rows"] += 1
                log_league(
                    league,
                    f"BIASED | {path} | league={league} "
                    f"margin_bias={margin_bias} total_bias={total_bias} "
                    f"adjusted={file_adjusted} skipped_already_flagged={file_skipped}"
                )

    return stats


# =========================
# STEP 7: WRITE CLEANED FILES
# =========================

def write_cleaned(loaded_files, cleaned_dirs, label):
    stats = {
        league: {
            "files_written": 0,
            "rows_written": 0,
        }
        for league in loaded_files
    }

    for league, files in loaded_files.items():
        cleaned_root = cleaned_dirs[league]

        for path, (fieldnames, rows) in files.items():
            new_path = cleaned_root / path.name
            write_csv(new_path, fieldnames, rows)

            stats[league]["files_written"] += 1
            stats[league]["rows_written"] += len(rows)

            log_league(
                league,
                f"WROTE | {label} | {new_path} | rows={len(rows)}"
            )

    return stats


# =========================
# SUMMARY
# =========================

def sum_nested(stats, league, key, default=0):
    return stats.get(league, {}).get(key, default)


def write_league_summaries(
    pred_files,
    book_files,
    bad_odds_blanked,
    bad_lines_blanked,
    outlier_counts,
    sportsbook_outlier_market_blanked,
    bias_stats,
    pred_write_stats,
    book_write_stats,
):
    for league in ("NBA", "NCAAM", "WNBA"):
        bad_ml = bad_odds_blanked.get(league, {}).get("ML", 0)
        bad_total = bad_odds_blanked.get(league, {}).get("TOTAL", 0)
        bad_spread = bad_odds_blanked.get(league, {}).get("SPREAD", 0)
        bad_odds_total_all = bad_ml + bad_total + bad_spread

        bad_total_lines = bad_lines_blanked.get(league, {}).get("TOTAL", 0)
        bad_spread_lines = bad_lines_blanked.get(league, {}).get("SPREAD", 0)
        bad_lines_total_all = bad_total_lines + bad_spread_lines

        spread_outliers = outlier_counts.get(league, {}).get("spread", 0)
        total_outliers = outlier_counts.get(league, {}).get("total", 0)

        log_league(league, "")
        log_league(league, "============================================================")
        log_league(league, "SUMMARY")
        log_league(league, "============================================================")
        log_league(league, f"prediction_files_loaded              : {len(pred_files.get(league, {}))}")
        log_league(league, f"sportsbook_files_loaded              : {len(book_files.get(league, {}))}")
        log_league(league, f"bad_ml_market_rows_blanked           : {bad_ml}")
        log_league(league, f"bad_total_market_rows_blanked        : {bad_total}")
        log_league(league, f"bad_spread_market_rows_blanked       : {bad_spread}")
        log_league(league, f"bad_odds_market_rows_blanked_total   : {bad_odds_total_all}")
        log_league(league, f"bad_total_line_rows_blanked          : {bad_total_lines}")
        log_league(league, f"bad_spread_line_rows_blanked         : {bad_spread_lines}")
        log_league(league, f"bad_line_rows_blanked_total          : {bad_lines_total_all}")
        log_league(league, f"spread_outlier_events_found          : {spread_outliers}")
        log_league(league, f"total_outlier_events_found           : {total_outliers}")
        log_league(league, f"sportsbook_outlier_market_rows_blanked: {sportsbook_outlier_market_blanked.get(league, 0)}")
        log_league(league, f"prediction_rows_removed              : 0")
        log_league(league, f"sportsbook_rows_removed              : 0")
        log_league(league, f"bias_files_with_adjusted_rows        : {sum_nested(bias_stats, league, 'files_with_biased_rows')}")
        log_league(league, f"bias_rows_adjusted                   : {sum_nested(bias_stats, league, 'rows_adjusted')}")
        log_league(league, f"bias_rows_skipped_already_flagged    : {sum_nested(bias_stats, league, 'rows_skipped_already_flagged')}")
        log_league(league, f"prediction_files_written             : {sum_nested(pred_write_stats, league, 'files_written')}")
        log_league(league, f"prediction_rows_written              : {sum_nested(pred_write_stats, league, 'rows_written')}")
        log_league(league, f"sportsbook_files_written             : {sum_nested(book_write_stats, league, 'files_written')}")
        log_league(league, f"sportsbook_rows_written              : {sum_nested(book_write_stats, league, 'rows_written')}")
        log_league(league, "STATUS: SUCCESS")
        log_league(league, "============================================================")


def write_master_summary(
    pred_files,
    book_files,
    bad_odds_blanked,
    bad_lines_blanked,
    outlier_counts,
    sportsbook_outlier_market_blanked,
    bias_stats,
    pred_write_stats,
    book_write_stats,
):
    log_master("")
    log_master("============================================================")
    log_master("SUMMARY")
    log_master("============================================================")

    for league in ("NBA", "NCAAM", "WNBA"):
        bad_ml = bad_odds_blanked.get(league, {}).get("ML", 0)
        bad_total = bad_odds_blanked.get(league, {}).get("TOTAL", 0)
        bad_spread = bad_odds_blanked.get(league, {}).get("SPREAD", 0)
        bad_odds_total_all = bad_ml + bad_total + bad_spread

        bad_total_lines = bad_lines_blanked.get(league, {}).get("TOTAL", 0)
        bad_spread_lines = bad_lines_blanked.get(league, {}).get("SPREAD", 0)
        bad_lines_total_all = bad_total_lines + bad_spread_lines

        spread_outliers = outlier_counts.get(league, {}).get("spread", 0)
        total_outliers = outlier_counts.get(league, {}).get("total", 0)

        log_master("")
        log_master(f"--- {league} ---")
        log_master(f"prediction_files_loaded              : {len(pred_files.get(league, {}))}")
        log_master(f"sportsbook_files_loaded              : {len(book_files.get(league, {}))}")
        log_master(f"bad_ml_market_rows_blanked           : {bad_ml}")
        log_master(f"bad_total_market_rows_blanked        : {bad_total}")
        log_master(f"bad_spread_market_rows_blanked       : {bad_spread}")
        log_master(f"bad_odds_market_rows_blanked_total   : {bad_odds_total_all}")
        log_master(f"bad_total_line_rows_blanked          : {bad_total_lines}")
        log_master(f"bad_spread_line_rows_blanked         : {bad_spread_lines}")
        log_master(f"bad_line_rows_blanked_total          : {bad_lines_total_all}")
        log_master(f"spread_outlier_events_found          : {spread_outliers}")
        log_master(f"total_outlier_events_found           : {total_outliers}")
        log_master(f"sportsbook_outlier_market_rows_blanked: {sportsbook_outlier_market_blanked.get(league, 0)}")
        log_master(f"prediction_rows_removed              : 0")
        log_master(f"sportsbook_rows_removed              : 0")
        log_master(f"bias_files_with_adjusted_rows        : {sum_nested(bias_stats, league, 'files_with_biased_rows')}")
        log_master(f"bias_rows_adjusted                   : {sum_nested(bias_stats, league, 'rows_adjusted')}")
        log_master(f"prediction_files_written             : {sum_nested(pred_write_stats, league, 'files_written')}")
        log_master(f"prediction_rows_written              : {sum_nested(pred_write_stats, league, 'rows_written')}")
        log_master(f"sportsbook_files_written             : {sum_nested(book_write_stats, league, 'files_written')}")
        log_master(f"sportsbook_rows_written              : {sum_nested(book_write_stats, league, 'rows_written')}")
        log_master(f"league_detail_log                    : {LEAGUE_LOG_FILES[league]}")

    log_master("")
    log_master("STATUS: SUCCESS")
    log_master("============================================================")


# =========================
# MAIN
# =========================

def main():
    init_logs()

    log_master("INFO | Starting basketball input cleanup")
    log_master("INFO | Odds cleanup active: blanks only the affected market; does not drop entire games")
    log_master("INFO | Missing market odds are treated as unavailable market, not bad game")
    log_master("INFO | Hold-based odds filtering removed")
    log_master("INFO | Spread outlier fix active: sportsbook home_spread is converted to model-margin convention using -home_spread")
    log_master("INFO | Outlier cleanup active: blanks only affected sportsbook market fields; prediction rows are retained")
    log_master(f"INFO | League logs: {LEAGUE_LOG_FILES}")

    for league in ("NBA", "NCAAM", "WNBA"):
        log_league(league, "INFO | Starting league cleanup")
        log_league(league, "INFO | Odds cleanup active: blanks only the affected market; does not drop entire games")
        log_league(league, "INFO | Missing market odds are treated as unavailable market, not bad game")
        log_league(league, "INFO | Hold-based odds filtering removed")
        log_league(league, "INFO | Spread outlier fix active: compare model home-away margin to -book_home_spread")
        log_league(league, "INFO | Outlier cleanup active: blanks only affected sportsbook market fields; prediction rows are retained")

    pred_files = load_all(PREDICTION_DIRS)
    book_files = load_all(SPORTSBOOK_DIRS)

    pred_loaded = sum(len(v) for v in pred_files.values())
    book_loaded = sum(len(v) for v in book_files.values())
    log_master(f"LOADED | predictions_files={pred_loaded} sportsbook_files={book_loaded}")

    bad_odds_blanked = blank_bad_market_odds(book_files)
    bad_lines_blanked = blank_bad_market_lines(book_files)

    pred_index = build_pred_index(pred_files)
    book_index = build_book_index(book_files)

    market_outlier_actions, outlier_counts = find_market_outlier_actions(pred_index, book_index)
    sportsbook_outlier_market_blanked = apply_market_outlier_actions(book_files, market_outlier_actions)

    bias_stats = apply_prediction_biases(pred_files)

    pred_write_stats = write_cleaned(
        pred_files,
        CLEANED_PREDICTION_DIRS,
        "PREDICTIONS",
    )

    book_write_stats = write_cleaned(
        book_files,
        CLEANED_SPORTSBOOK_DIRS,
        "SPORTSBOOK",
    )

    write_league_summaries(
        pred_files=pred_files,
        book_files=book_files,
        bad_odds_blanked=bad_odds_blanked,
        bad_lines_blanked=bad_lines_blanked,
        outlier_counts=outlier_counts,
        sportsbook_outlier_market_blanked=sportsbook_outlier_market_blanked,
        bias_stats=bias_stats,
        pred_write_stats=pred_write_stats,
        book_write_stats=book_write_stats,
    )

    write_master_summary(
        pred_files=pred_files,
        book_files=book_files,
        bad_odds_blanked=bad_odds_blanked,
        bad_lines_blanked=bad_lines_blanked,
        outlier_counts=outlier_counts,
        sportsbook_outlier_market_blanked=sportsbook_outlier_market_blanked,
        bias_stats=bias_stats,
        pred_write_stats=pred_write_stats,
        book_write_stats=book_write_stats,
    )

    print("STATUS: SUCCESS")
    print(f"log_file_master                  : {MASTER_LOG_FILE}")
    print(f"log_file_nba                     : {LEAGUE_LOG_FILES['NBA']}")
    print(f"log_file_ncaam                   : {LEAGUE_LOG_FILES['NCAAM']}")
    print(f"log_file_wnba                    : {LEAGUE_LOG_FILES['WNBA']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            log_master("STATUS: FAILED")
            log_master(traceback.format_exc())

            for league in ("NBA", "NCAAM", "WNBA"):
                log_league(league, "STATUS: FAILED")
                log_league(league, traceback.format_exc())
        except Exception:
            pass

        print("STATUS: FAILED")
        print(f"See log: {MASTER_LOG_FILE}")
        raise

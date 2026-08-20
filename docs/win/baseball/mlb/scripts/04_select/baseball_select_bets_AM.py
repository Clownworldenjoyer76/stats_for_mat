#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/04_select/baseball_select_bets_AM.py
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

INPUT_DIR = Path("docs/win/baseball/mlb/03_edges/ev_kelly")
OUTPUT_DIR = Path("docs/win/baseball/mlb/04_select/morning")
CONFIG_PATH = Path("docs/win/baseball/mlb/config/markets_AM.yaml")

AUDIT_DIR = OUTPUT_DIR / "audit"
ERROR_DIR = Path("docs/win/baseball/mlb/errors/04_select")
LOG_FILE = ERROR_DIR / "select_bets_AM.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_CODE = "MLB"
PROB_TOLERANCE = 1e-9

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _yaml = yaml.safe_load(f)["markets"]["mlb"]
    CONFIG = _yaml
    FILTERS = _yaml


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_slate: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  run_mode                          : {summary['run_mode']}",
        f"  slates_found                      : {summary['slates_found']}",
        f"  slates_processed                  : {summary['slates_processed']}",
        f"  slates_written                    : {summary['slates_written']}",
        f"  total_bets                        : {summary['total_bets']}",
        f"  moneyline_bets                    : {summary['moneyline_bets']}",
        f"  run_line_bets                     : {summary['run_line_bets']}",
        f"  total_mkt_bets                    : {summary['total_mkt_bets']}",
        f"  skipped_slates                    : {summary['skipped_slates']}",
        f"  missing_moneyline                 : {summary['missing_moneyline']}",
        f"  missing_run_line                  : {summary['missing_run_line']}",
        f"  missing_total                     : {summary['missing_total']}",
        f"  row_count_warnings                : {summary['row_count_warnings']}",
        f"  duplicate_game_id_errors          : {summary['duplicate_game_id_errors']}",
        f"  selected_nonpositive_kelly        : {summary['selected_nonpositive_kelly']}",
        f"  selected_blank_probability_source : {summary['selected_blank_probability_source']}",
        f"  selected_probability_source_mismatch: {summary['selected_probability_source_mismatch']}",
        f"  schema_errors                     : {summary['schema_errors']}",
        f"  rain_excluded                     : {summary['rain_excluded']}",
        f"  rain_excluded_will_it_rain        : {summary['rain_excluded_will_it_rain']}",
        f"  rain_excluded_symbol_code         : {summary['rain_excluded_symbol_code']}",
        f"  sp_sample_excluded                : {summary['sp_sample_excluded']}",
        f"  low_confidence                    : {summary['low_confidence']}",
        f"  rejection_audit_rows              : {summary['rejection_audit_rows']}",
        f"  selected_audit_rows               : {summary['selected_audit_rows']}",
        f"  errors                            : {summary['errors']}",
        "",
        "--- Filter Breakdown ---",
    ]

    for market, sides in summary.get("counters", {}).items():
        for side, c in sides.items():
            lines.append(
                f"  {market}-{side:<10} passed={c['passed']:>4} "
                f"ev_fail={c['ev_fail']:>4} kelly_fail={c['kelly_fail']:>4} "
                f"odds_fail={c['odds_fail']:>4} line_fail={c['line_fail']:>4} "
                f"prob_fail={c['prob_fail']:>4} source_fail={c['source_fail']:>4} "
                f"missing={c['missing']:>4} "
                f"excluded={c['excluded']:>4}"
            )

    lines += [
        "",
        f"  {'slate':<30} {'bets':>5} {'ml':>5} {'rl':>5} {'tot':>5} {'status':>14}",
    ]

    for ps in per_slate:
        lines.append(
            f"  {ps['slate']:<30} {ps['bets']:>5} {ps['ml']:>5} "
            f"{ps['rl']:>5} {ps['tot']:>5} {ps['status']:>14}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA VALIDATION
# =========================

REQUIRED_BASE_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]

REQUIRED_CONTEXT_COLUMNS = [
    "home_batters_found",
    "away_batters_found",
    "home_sp_found",
    "away_sp_found",
]

REQUIRED_MONEYLINE_COLUMNS = REQUIRED_BASE_COLUMNS + REQUIRED_CONTEXT_COLUMNS + [
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_decimal_moneyline",
    "away_dk_decimal_moneyline",
    "home_normalized_prob_moneyline",
    "away_normalized_prob_moneyline",
    "home_prob_for_ev",
    "away_prob_for_ev",
    "home_prob_for_kelly",
    "away_prob_for_kelly",
    "home_ev_probability_source",
    "away_ev_probability_source",
    "home_kelly_probability_source",
    "away_kelly_probability_source",
    "home_ml_raw_ev",
    "away_ml_raw_ev",
    "home_ml_adjusted_ev",
    "away_ml_adjusted_ev",
    "home_ml_ev",
    "away_ml_ev",
    "home_ml_kelly",
    "away_ml_kelly",
]

REQUIRED_RUN_LINE_COLUMNS = REQUIRED_BASE_COLUMNS + REQUIRED_CONTEXT_COLUMNS + [
    "home_run_line",
    "away_run_line",
    "home_dk_run_line_american",
    "away_dk_run_line_american",
    "home_dk_run_line_decimal",
    "away_dk_run_line_decimal",
    "home_normalized_prob_run_line",
    "away_normalized_prob_run_line",
    "home_prob_for_ev",
    "away_prob_for_ev",
    "home_prob_for_kelly",
    "away_prob_for_kelly",
    "home_ev_probability_source",
    "away_ev_probability_source",
    "home_kelly_probability_source",
    "away_kelly_probability_source",
    "home_rl_raw_ev",
    "away_rl_raw_ev",
    "home_rl_adjusted_ev",
    "away_rl_adjusted_ev",
    "home_rl_ev",
    "away_rl_ev",
    "home_rl_kelly",
    "away_rl_kelly",
]

REQUIRED_TOTAL_COLUMNS = REQUIRED_BASE_COLUMNS + REQUIRED_CONTEXT_COLUMNS + [
    "total",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "over_normalized_prob_total",
    "under_normalized_prob_total",
    "over_prob_for_ev",
    "under_prob_for_ev",
    "over_prob_for_kelly",
    "under_prob_for_kelly",
    "over_ev_probability_source",
    "under_ev_probability_source",
    "over_kelly_probability_source",
    "under_kelly_probability_source",
    "over_raw_ev",
    "under_raw_ev",
    "over_adjusted_ev",
    "under_adjusted_ev",
    "over_ev",
    "under_ev",
    "over_kelly",
    "under_kelly",
]

FORBIDDEN_RUN_LINE_COLUMNS = [
    "home_run_line_prob",
    "away_run_line_prob",
]

SELECTED_AUDIT_COLUMNS = [
    "date",
    "game_id",
    "market",
    "side",
    "prob_used_for_selection",
    "prob_used_for_ev",
    "prob_used_for_kelly",
    "ev",
    "kelly",
    "odds",
    "line",
    "selection_reason",
]

REJECTION_AUDIT_COLUMNS = [
    "date",
    "game_id",
    "market",
    "side",
    "fail_reason",
    "fail_detail",
    "prob_used_for_selection",
    "prob_used_for_ev",
    "prob_used_for_kelly",
    "ev",
    "kelly",
    "odds",
    "line",
    "raw_ev",
    "adjusted_ev",
    "ev_probability_source",
    "kelly_probability_source",
]


def duplicate_columns(columns):
    seen = set()
    duplicates = []

    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    return duplicates


def validate_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_required_columns(df: pd.DataFrame, required_columns: list, label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_forbidden_columns(df: pd.DataFrame, forbidden_columns: list, label: str) -> None:
    present = [col for col in forbidden_columns if col in df.columns]

    if present:
        raise ValueError(
            f"{label} contains obsolete forbidden columns: {present}. "
            f"Use home_normalized_prob_run_line / away_normalized_prob_run_line."
        )


def validate_unique_game_id(df: pd.DataFrame, label: str) -> None:
    game_ids = df["game_id"].astype(str).str.strip()

    if (game_ids == "").any() or (game_ids.str.lower() == "nan").any():
        blank_rows = df.index[(game_ids == "") | (game_ids.str.lower() == "nan")].tolist()
        raise ValueError(f"{label} has blank game_id rows: {blank_rows[:20]}")

    dupes = game_ids[game_ids.duplicated(keep=False)]

    if not dupes.empty:
        counts = dupes.value_counts().to_dict()
        raise ValueError(f"{label} has multiple rows for one game_id: {counts}")


def read_market_csv(path: Path, required_columns: list, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    validate_no_duplicate_columns(df, label)
    validate_required_columns(df, required_columns, label)
    validate_unique_game_id(df, label)

    return df


def validate_selected_output(df: pd.DataFrame, label: str) -> dict:
    counts = {
        "selected_nonpositive_kelly": 0,
        "selected_blank_probability_source": 0,
        "selected_probability_source_mismatch": 0,
    }

    if df.empty:
        return counts

    kelly = pd.to_numeric(df["kelly"], errors="coerce")
    counts["selected_nonpositive_kelly"] = int((kelly.isna() | (kelly <= 0)).sum())

    ev_src = df["ev_probability_source"].astype(str).str.strip()
    kelly_src = df["kelly_probability_source"].astype(str).str.strip()
    counts["selected_blank_probability_source"] = int(
        ((ev_src == "") | (kelly_src == "") | (ev_src == "nan") | (kelly_src == "nan")).sum()
    )
    counts["selected_probability_source_mismatch"] = int((ev_src != kelly_src).sum())

    prob_ev = pd.to_numeric(df["prob_for_ev"], errors="coerce")
    prob_kelly = pd.to_numeric(df["prob_for_kelly"], errors="coerce")
    prob_mismatch = int(((prob_ev - prob_kelly).abs() > PROB_TOLERANCE).sum())
    counts["selected_probability_source_mismatch"] += prob_mismatch

    failures = {
        k: v for k, v in counts.items()
        if v > 0
    }

    if failures:
        raise ValueError(f"{label} failed selected-bet validation: {failures}")

    return counts


def write_output_csv(df: pd.DataFrame, path: Path, label: str) -> dict:
    validate_no_duplicate_columns(df, label)
    counts = validate_selected_output(df, label)
    df.to_csv(path, index=False)
    return counts


def row_count_check(slate: str, market_frames: dict, summary: dict) -> None:
    counts = {
        market: len(df)
        for market, df in market_frames.items()
        if df is not None
    }

    if len(counts) <= 1:
        return

    unique_counts = sorted(set(counts.values()))

    if len(unique_counts) > 1:
        summary["row_count_warnings"] += 1
        _log(f"{slate} market row-count mismatch before selection: {counts}", "WARN")
    else:
        _log(f"{slate} market row-count check OK: {counts}")


def validate_config() -> None:
    run_line_preference = CONFIG.get("run_line", {}).get("pick_preference", "best_ev")
    if run_line_preference not in {"best_ev", "best_prob"}:
        raise ValueError(
            "run_line.pick_preference must be best_ev or best_prob. "
            "Use of all is blocked to prevent both run-line sides on one game."
        )

    for market in ["moneyline", "run_line", "total"]:
        market_cfg = CONFIG.get(market, {})
        for side, rules in market_cfg.items():
            if not isinstance(rules, dict):
                continue

            for key in ["ev_bands", "kelly_bands", "odds_bands", "line_bands", "prob_bands"]:
                if key not in rules:
                    continue
                for band in rules[key]:
                    if len(band) != 2:
                        raise ValueError(f"{market}.{side}.{key} contains invalid band: {band}")
                    if band[0] > band[1]:
                        raise ValueError(f"{market}.{side}.{key} contains inverted band: {band}")


# =========================
# HELPERS
# =========================

def fv(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def iv(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def sv(x):
    try:
        if pd.isna(x):
            return None
        return str(x).strip()
    except Exception:
        return None


def in_range(val, ranges):
    if val is None:
        return False
    return any(lo <= val <= hi for lo, hi in ranges)


def matched_band(val, ranges):
    if val is None:
        return None

    for lo, hi in ranges:
        if lo <= val <= hi:
            return f"[{lo},{hi}]"

    return None



def check_probability_basis(prob_for_ev, prob_for_kelly, ev_source, kelly_source, counters):
    if not ev_source or not kelly_source:
        counters["source_fail"] += 1
        return False, "source_fail", "blank_probability_source"

    if str(ev_source).strip() != str(kelly_source).strip():
        counters["source_fail"] += 1
        return False, "source_fail", "ev_kelly_source_mismatch"

    if prob_for_ev is None or prob_for_kelly is None:
        counters["source_fail"] += 1
        return False, "source_fail", "missing_probability_basis"

    if abs(prob_for_ev - prob_for_kelly) > PROB_TOLERANCE:
        counters["source_fail"] += 1
        return False, "source_fail", "prob_for_ev_prob_for_kelly_mismatch"

    return True, "", ""


def violates_exclude_rules(ev, kelly, odds, line, prob, rules):
    for r in rules.get("exclude_rules", []):
        if "ev_min" in r and (ev is None or ev < r["ev_min"]):
            continue
        if "ev_max" in r and (ev is None or ev > r["ev_max"]):
            continue
        if "kelly_min" in r and (kelly is None or kelly < r["kelly_min"]):
            continue
        if "kelly_max" in r and (kelly is None or kelly > r["kelly_max"]):
            continue
        if "odds_min" in r and (odds is None or odds < r["odds_min"]):
            continue
        if "odds_max" in r and (odds is None or odds > r["odds_max"]):
            continue
        if "line_min" in r and (line is None or line < r["line_min"]):
            continue
        if "line_max" in r and (line is None or line > r["line_max"]):
            continue
        if "prob_min" in r and (prob is None or prob < r["prob_min"]):
            continue
        if "prob_max" in r and (prob is None or prob > r["prob_max"]):
            continue

        if "prob_bands" in r:
            if prob is None or not in_range(prob, r["prob_bands"]):
                continue

        return True

    return False


def check_rules(ev, kelly, odds, line, prob, rules, counters):
    reason_parts = []

    if ev is None or kelly is None:
        counters["missing"] += 1
        return False, "missing", "missing_ev_or_kelly"

    if kelly <= 0:
        counters["kelly_fail"] += 1
        return False, "kelly_fail", "kelly<=0"

    ev_band = matched_band(ev, rules.get("ev_bands", []))
    if ev_band is None:
        counters["ev_fail"] += 1
        return False, "ev_fail", "outside_ev_bands"
    reason_parts.append(f"ev_band={ev_band}")

    kelly_band = matched_band(kelly, rules.get("kelly_bands", []))
    if kelly_band is None:
        counters["kelly_fail"] += 1
        return False, "kelly_fail", "outside_kelly_bands"
    reason_parts.append(f"kelly_band={kelly_band}")

    if "odds_bands" in rules:
        odds_band = matched_band(odds, rules["odds_bands"])
        if odds_band is None:
            counters["odds_fail"] += 1
            return False, "odds_fail", "outside_odds_bands"
        reason_parts.append(f"odds_band={odds_band}")

    if "line_bands" in rules:
        line_band = matched_band(line, rules["line_bands"])
        if line_band is None:
            counters["line_fail"] += 1
            return False, "line_fail", "outside_line_bands"
        reason_parts.append(f"line_band={line_band}")

    if "prob_bands" in rules:
        prob_band = matched_band(prob, rules["prob_bands"])
        if prob_band is None:
            counters["prob_fail"] += 1
            return False, "prob_fail", "outside_prob_bands"
        reason_parts.append(f"prob_band={prob_band}")

    if "prob_min" in rules and (prob is None or prob < rules["prob_min"]):
        counters["prob_fail"] += 1
        return False, "prob_fail", "below_prob_min"

    if "prob_max" in rules and (prob is None or prob > rules["prob_max"]):
        counters["prob_fail"] += 1
        return False, "prob_fail", "above_prob_max"

    if violates_exclude_rules(ev, kelly, odds, line, prob, rules):
        counters["excluded"] += 1
        return False, "excluded", "matched_exclude_rule"

    counters["passed"] += 1
    return True, "", ";".join(reason_parts)


def init_counter():
    return {
        "passed": 0,
        "ev_fail": 0,
        "kelly_fail": 0,
        "odds_fail": 0,
        "line_fail": 0,
        "prob_fail": 0,
        "source_fail": 0,
        "excluded": 0,
        "missing": 0,
    }


def select_candidate(candidates, preference, market_name=None, game_id=None):
    if not candidates:
        return []

    if market_name == "run_line" and preference == "all":
        raise ValueError(f"{game_id} run_line pick_preference=all is not allowed")

    if preference == "all":
        return candidates

    if preference == "best_prob":
        return [max(candidates, key=lambda x: x["model_prob"] if x["model_prob"] is not None else -999)]

    return [max(candidates, key=lambda x: x["ev"] if x["ev"] is not None else -999)]


def get_market_row(df: pd.DataFrame, game_id: str):
    if df is None or df.empty:
        return None

    matched = df[df["game_id"].astype(str).str.strip() == str(game_id).strip()]

    if matched.empty:
        return None

    if len(matched) > 1:
        raise ValueError(f"Multiple rows matched game_id={game_id}")

    return matched.iloc[0]


def build_base_games(market_frames: dict) -> pd.DataFrame:
    pieces = []

    for market_name in ["moneyline", "run_line", "total"]:
        df = market_frames.get(market_name)

        if df is None or df.empty:
            continue

        piece = df[REQUIRED_BASE_COLUMNS].copy()
        piece["source_market"] = market_name
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=REQUIRED_BASE_COLUMNS + ["source_market"])

    combined = pd.concat(pieces, ignore_index=True)
    rows = []

    for game_id, group in combined.groupby(combined["game_id"].astype(str).str.strip(), sort=True):
        base = group.iloc[0].copy()

        for col in ["game_date", "home_team", "away_team"]:
            values = sorted(set(group[col].astype(str).str.strip()))
            if len(values) > 1:
                raise ValueError(f"game_id={game_id} has conflicting {col} values across market files: {values}")

        sources = sorted(set(group["source_market"].astype(str)))
        base["source_market"] = ";".join(sources)
        rows.append(base)

    return pd.DataFrame(rows)


def base_candidate_audit(row, candidate, fail_reason="", fail_detail=""):
    return {
        "date": row.get("game_date"),
        "game_id": row.get("game_id"),
        "market": candidate.get("market"),
        "side": candidate.get("side"),
        "fail_reason": fail_reason,
        "fail_detail": fail_detail,
        "prob_used_for_selection": candidate.get("prob_used_for_selection"),
        "prob_used_for_ev": candidate.get("prob_for_ev"),
        "prob_used_for_kelly": candidate.get("prob_for_kelly"),
        "ev": candidate.get("ev"),
        "kelly": candidate.get("kelly"),
        "odds": candidate.get("dk_odds_american"),
        "line": candidate.get("line"),
        "raw_ev": candidate.get("raw_ev"),
        "adjusted_ev": candidate.get("adjusted_ev"),
        "ev_probability_source": candidate.get("ev_probability_source"),
        "kelly_probability_source": candidate.get("kelly_probability_source"),
    }


def selected_audit_row(row):
    return {
        "date": row.get("game_date"),
        "game_id": row.get("game_id"),
        "market": row.get("market"),
        "side": row.get("side"),
        "prob_used_for_selection": row.get("prob_used_for_selection"),
        "prob_used_for_ev": row.get("prob_for_ev"),
        "prob_used_for_kelly": row.get("prob_for_kelly"),
        "ev": row.get("ev"),
        "kelly": row.get("kelly"),
        "odds": row.get("dk_odds_american"),
        "line": row.get("line"),
        "selection_reason": row.get("selection_reason"),
    }


# =========================
# CONTEXT FILTERS
# =========================

def rain_exclusion_reason(row):
    if row is None:
        return None

    weather_applicable = iv(row.get("weather_applicable"))

    if weather_applicable == 0:
        return None

    will_it_rain = iv(row.get("will_it_rain"))
    symbol_code = sv(row.get("symbol_code"))

    if FILTERS.get("rain_exclude_on_will_it_rain", True) and will_it_rain == 1:
        return "will_it_rain"

    if not FILTERS.get("rain_exclude_on_symbol_code", False):
        return None

    rain_terms = FILTERS.get("rain_symbol_terms", [
        "rain",
        "heavyrain",
        "lightrain",
        "sleet",
        "snow",
        "thunder",
    ])

    if symbol_code:
        symbol = symbol_code.lower()
        if any(str(term).lower() in symbol for term in rain_terms):
            return "symbol_code"

    return None


def sp_sample_excluded_for_total(row) -> bool:
    if row is None:
        return False

    if not FILTERS.get("sp_sample_exclude_totals", True):
        return False

    home_flag = sv(row.get("home_sp_sample_flag"))
    away_flag = sv(row.get("away_sp_sample_flag"))

    return home_flag == "low" or away_flag == "low"


def is_low_confidence(row) -> int:
    if row is None:
        return 0

    warn = FILTERS.get("lineup_low_sample_warn", 3)
    home_low = fv(row.get("home_low_sample_count"))
    away_low = fv(row.get("away_low_sample_count"))

    if home_low is not None and home_low > warn:
        return 1
    if away_low is not None and away_low > warn:
        return 1
    return 0


# =========================
# MARKET PROCESSORS
# =========================

def evaluate_candidate(row, candidate, rules, side_counter, rejection_rows):
    basis_ok, fail_reason, fail_detail = check_probability_basis(
        candidate["prob_for_ev"],
        candidate["prob_for_kelly"],
        candidate["ev_probability_source"],
        candidate["kelly_probability_source"],
        side_counter,
    )

    if not basis_ok:
        rejection_rows.append(base_candidate_audit(row, candidate, fail_reason, fail_detail))
        return None

    passed, fail_reason, detail = check_rules(
        candidate["ev"],
        candidate["kelly"],
        candidate["dk_odds_american"],
        candidate["line"] if candidate["line"] != "" else None,
        candidate["prob_used_for_selection"],
        rules,
        side_counter,
    )

    if not passed:
        rejection_rows.append(base_candidate_audit(row, candidate, fail_reason, detail))
        return None

    candidate["selection_reason"] = detail
    return candidate


def process_moneyline(row, counters, rejection_rows):
    candidates = []

    odds_by_side = {
        "home": fv(row.get("home_dk_moneyline_american")),
        "away": fv(row.get("away_dk_moneyline_american")),
    }

    for side in ["home", "away"]:
        rules = CONFIG["moneyline"][side]
        side_counter = counters["moneyline"][side]

        if not rules["enabled"]:
            continue

        prob_for_ev = fv(row.get(f"{side}_prob_for_ev"))
        prob_for_kelly = fv(row.get(f"{side}_prob_for_kelly"))

        candidate = {
            "market_type": "moneyline",
            "bet_side": side,
            "market": "moneyline",
            "side": side,
            "line": "",
            "take_bet": f"{side}_moneyline",
            "dk_odds_american": odds_by_side[side],
            "dk_odds_decimal": fv(row.get(f"{side}_dk_decimal_moneyline")),
            "model_prob": prob_for_ev,
            "prob_used_for_selection": prob_for_ev,
            "prob_for_ev": prob_for_ev,
            "prob_for_kelly": prob_for_kelly,
            "ev_probability_source": sv(row.get(f"{side}_ev_probability_source")),
            "kelly_probability_source": sv(row.get(f"{side}_kelly_probability_source")),
            "raw_ev": fv(row.get(f"{side}_ml_raw_ev")),
            "adjusted_ev": fv(row.get(f"{side}_ml_adjusted_ev")),
            "ev": fv(row.get(f"{side}_ml_ev")),
            "kelly": fv(row.get(f"{side}_ml_kelly")),
        }

        selected = evaluate_candidate(row, candidate, rules, side_counter, rejection_rows)
        if selected is not None:
            candidates.append(selected)

    return select_candidate(
        candidates,
        CONFIG["moneyline"].get("pick_preference", "best_ev"),
        "moneyline",
        row.get("game_id"),
    )


def process_run_line(row, counters, rejection_rows):
    candidates = []

    odds_by_side = {
        "home": fv(row.get("home_dk_run_line_american")),
        "away": fv(row.get("away_dk_run_line_american")),
    }

    for side in ["home", "away"]:
        rules = CONFIG["run_line"][side]
        side_counter = counters["run_line"][side]

        if not rules["enabled"]:
            continue

        prob_for_ev = fv(row.get(f"{side}_prob_for_ev"))
        prob_for_kelly = fv(row.get(f"{side}_prob_for_kelly"))

        candidate = {
            "market_type": "run_line",
            "bet_side": side,
            "market": "run_line",
            "side": side,
            "line": fv(row.get(f"{side}_run_line")),
            "take_bet": f"{side}_run_line",
            "dk_odds_american": odds_by_side[side],
            "dk_odds_decimal": fv(row.get(f"{side}_dk_run_line_decimal")),
            "model_prob": prob_for_ev,
            "prob_used_for_selection": prob_for_ev,
            "prob_for_ev": prob_for_ev,
            "prob_for_kelly": prob_for_kelly,
            "ev_probability_source": sv(row.get(f"{side}_ev_probability_source")),
            "kelly_probability_source": sv(row.get(f"{side}_kelly_probability_source")),
            "raw_ev": fv(row.get(f"{side}_rl_raw_ev")),
            "adjusted_ev": fv(row.get(f"{side}_rl_adjusted_ev")),
            "ev": fv(row.get(f"{side}_rl_ev")),
            "kelly": fv(row.get(f"{side}_rl_kelly")),
        }

        selected = evaluate_candidate(row, candidate, rules, side_counter, rejection_rows)
        if selected is not None:
            candidates.append(selected)

    return select_candidate(
        candidates,
        CONFIG["run_line"].get("pick_preference", "best_ev"),
        "run_line",
        row.get("game_id"),
    )


def process_total(row, counters, rejection_rows):
    candidates = []

    odds_by_side = {
        "over": fv(row.get("dk_total_over_american")),
        "under": fv(row.get("dk_total_under_american")),
    }

    for side in ["over", "under"]:
        rules = CONFIG["total"][side]
        side_counter = counters["total"][side]

        if not rules["enabled"]:
            continue

        prob_for_ev = fv(row.get(f"{side}_prob_for_ev"))
        prob_for_kelly = fv(row.get(f"{side}_prob_for_kelly"))

        candidate = {
            "market_type": "total",
            "bet_side": side,
            "market": "total",
            "side": side,
            "line": fv(row.get("total")),
            "take_bet": f"{side}_total",
            "dk_odds_american": odds_by_side[side],
            "dk_odds_decimal": fv(row.get(f"dk_total_{side}_decimal")),
            "model_prob": prob_for_ev,
            "prob_used_for_selection": prob_for_ev,
            "prob_for_ev": prob_for_ev,
            "prob_for_kelly": prob_for_kelly,
            "ev_probability_source": sv(row.get(f"{side}_ev_probability_source")),
            "kelly_probability_source": sv(row.get(f"{side}_kelly_probability_source")),
            "raw_ev": fv(row.get(f"{side}_raw_ev")),
            "adjusted_ev": fv(row.get(f"{side}_adjusted_ev")),
            "ev": fv(row.get(f"{side}_ev")),
            "kelly": fv(row.get(f"{side}_kelly")),
        }

        selected = evaluate_candidate(row, candidate, rules, side_counter, rejection_rows)
        if selected is not None:
            candidates.append(selected)

    return select_candidate(
        candidates,
        CONFIG["total"].get("pick_preference", "best_ev"),
        "total",
        row.get("game_id"),
    )


# =========================
# RUN MODE
# =========================

def choose_slates(slates: dict) -> tuple[list, str]:
    available = sorted(slates)
    return available, "rebuild_all"


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== MLB select_bets RUN {_now()} ===\n")

    summary = {
        "run_mode": "unresolved",
        "slates_found": 0,
        "slates_processed": 0,
        "slates_written": 0,
        "total_bets": 0,
        "moneyline_bets": 0,
        "run_line_bets": 0,
        "total_mkt_bets": 0,
        "skipped_slates": 0,
        "missing_moneyline": 0,
        "missing_run_line": 0,
        "missing_total": 0,
        "row_count_warnings": 0,
        "duplicate_game_id_errors": 0,
        "selected_nonpositive_kelly": 0,
        "selected_blank_probability_source": 0,
        "selected_probability_source_mismatch": 0,
        "schema_errors": 0,
        "rain_excluded": 0,
        "rain_excluded_will_it_rain": 0,
        "rain_excluded_symbol_code": 0,
        "sp_sample_excluded": 0,
        "low_confidence": 0,
        "rejection_audit_rows": 0,
        "selected_audit_rows": 0,
        "errors": 0,
        "counters": {},
    }

    per_slate = []
    selected_audit_rows = []
    rejection_rows = []

    for old in OUTPUT_DIR.glob("*.csv"):
        old.unlink()
    for old in AUDIT_DIR.glob("*.csv"):
        old.unlink()

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    _log(
        f"Rain filter: will_it_rain={FILTERS.get('rain_exclude_on_will_it_rain', True)} "
        f"symbol_code={FILTERS.get('rain_exclude_on_symbol_code', False)} "
        f"symbol_terms={FILTERS.get('rain_symbol_terms')} | "
        f"SP sample exclude totals: {FILTERS.get('sp_sample_exclude_totals')} | "
        f"Lineup low sample warn: {FILTERS.get('lineup_low_sample_warn')}"
    )
    _log("Selection requires kelly > 0 and matching EV/Kelly probability source per selected row.")
    _log("Selection matches market rows by game_id only. Team/date fallback matching is disabled.")

    try:
        validate_config()

        files = sorted(INPUT_DIR.glob("*_mlb_*.csv"))
        _log(f"Files found: {len(files)}")

        if not files:
            _log("No input files found", "WARN")
            _write_summary(summary, per_slate)
            return

        slates = {}

        for fp in files:
            key = fp.name.split("_mlb_")[0]
            slates.setdefault(key, []).append(fp)

        summary["slates_found"] = len(slates)

        slate_keys, run_mode = choose_slates(slates)
        summary["run_mode"] = run_mode
        _log(f"Selected run mode: {run_mode}")
        _log(f"Slate keys to process: {slate_keys}")

        global_counters = {
            "moneyline": {
                "home": init_counter(),
                "away": init_counter(),
            },
            "run_line": {
                "home": init_counter(),
                "away": init_counter(),
            },
            "total": {
                "over": init_counter(),
                "under": init_counter(),
            },
        }

        for slate in slate_keys:
            ps = {
                "slate": slate,
                "bets": 0,
                "ml": 0,
                "rl": 0,
                "tot": 0,
                "status": "ok",
            }

            _log(f"--- SLATE: {slate}")

            try:
                summary["slates_processed"] += 1

                ml_path = INPUT_DIR / f"{slate}_mlb_moneyline.csv"
                rl_path = INPUT_DIR / f"{slate}_mlb_run_line.csv"
                tt_path = INPUT_DIR / f"{slate}_mlb_total.csv"

                ml_df = None
                rl_df = None
                tt_df = None

                if ml_path.exists():
                    ml_df = read_market_csv(
                        ml_path,
                        REQUIRED_MONEYLINE_COLUMNS,
                        f"{slate} moneyline input",
                    )
                else:
                    summary["missing_moneyline"] += 1
                    _log(f"{slate} missing moneyline file — continuing without moneyline", "WARN")

                if rl_path.exists():
                    rl_df = read_market_csv(
                        rl_path,
                        REQUIRED_RUN_LINE_COLUMNS,
                        f"{slate} run-line input",
                    )
                    validate_forbidden_columns(
                        rl_df,
                        FORBIDDEN_RUN_LINE_COLUMNS,
                        f"{slate} run-line input",
                    )
                else:
                    summary["missing_run_line"] += 1
                    _log(f"{slate} missing run_line file — continuing without run_line", "WARN")

                if tt_path.exists():
                    tt_df = read_market_csv(
                        tt_path,
                        REQUIRED_TOTAL_COLUMNS,
                        f"{slate} total input",
                    )
                else:
                    summary["missing_total"] += 1
                    _log(f"{slate} missing total file — continuing without total", "WARN")

                market_frames = {
                    "moneyline": ml_df,
                    "run_line": rl_df,
                    "total": tt_df,
                }

                if all(df is None or df.empty for df in market_frames.values()):
                    _log(f"{slate} no usable market files — skipping slate", "WARN")
                    ps["status"] = "skipped"
                    summary["skipped_slates"] += 1
                    per_slate.append(ps)
                    continue

                row_count_check(slate, market_frames, summary)

                base_games = build_base_games(market_frames)

                if base_games.empty:
                    _log(f"{slate} no base games after market load — skipping slate", "WARN")
                    ps["status"] = "skipped"
                    summary["skipped_slates"] += 1
                    per_slate.append(ps)
                    continue

                final = []
                seen = set()

                for _, base_row in base_games.iterrows():
                    game_id = str(base_row["game_id"]).strip()
                    sport = base_row.get("sport", "")
                    game_date = base_row["game_date"]
                    game_time = base_row.get("game_time", "")
                    away = base_row["away_team"]
                    home = base_row["home_team"]

                    base = {
                        "game_id": game_id,
                        "sport": sport,
                        "game_date": game_date,
                        "game_time": game_time,
                        "league": LEAGUE_CODE,
                        "away_team": away,
                        "home_team": home,
                    }

                    rl_row = get_market_row(rl_df, game_id)
                    ml_row = get_market_row(ml_df, game_id)
                    tt_row = get_market_row(tt_df, game_id)

                    context_row = rl_row if rl_row is not None else (tt_row if tt_row is not None else ml_row)

                    base.update({
                        "home_batters_found": iv(context_row.get("home_batters_found")) if context_row is not None else None,
                        "away_batters_found": iv(context_row.get("away_batters_found")) if context_row is not None else None,
                        "home_sp_found": iv(context_row.get("home_sp_found")) if context_row is not None else None,
                        "away_sp_found": iv(context_row.get("away_sp_found")) if context_row is not None else None,
                    })
                    low_conf = is_low_confidence(context_row)

                    rain_reason = rain_exclusion_reason(context_row)

                    if rain_reason:
                        summary["rain_excluded"] += 1
                        if rain_reason == "will_it_rain":
                            summary["rain_excluded_will_it_rain"] += 1
                        elif rain_reason == "symbol_code":
                            summary["rain_excluded_symbol_code"] += 1

                        _log(
                            f"  {game_id} rain excluded "
                            f"(reason={rain_reason} will_it_rain={iv(context_row.get('will_it_rain'))} "
                            f"symbol_code={sv(context_row.get('symbol_code'))})",
                            "WARN",
                        )

                        rejection_rows.append({
                            "date": game_date,
                            "game_id": game_id,
                            "market": "all",
                            "side": "all",
                            "fail_reason": "rain_excluded",
                            "fail_detail": rain_reason,
                            "prob_used_for_selection": None,
                            "prob_used_for_ev": None,
                            "prob_used_for_kelly": None,
                            "ev": None,
                            "kelly": None,
                            "odds": None,
                            "line": None,
                            "raw_ev": None,
                            "adjusted_ev": None,
                            "ev_probability_source": None,
                            "kelly_probability_source": None,
                        })
                        continue

                    if rl_row is not None:
                        for r in process_run_line(rl_row, global_counters, rejection_rows):
                            k = f"{game_id}_{r['market_type']}"

                            if k not in seen:
                                if low_conf:
                                    summary["low_confidence"] += 1

                                selected = {**base, **r, "low_confidence": low_conf}
                                final.append(selected)
                                selected_audit_rows.append(selected_audit_row(selected))
                                seen.add(k)
                                ps["rl"] += 1

                    if tt_row is not None:
                        if sp_sample_excluded_for_total(tt_row):
                            summary["sp_sample_excluded"] += 1
                            _log(
                                f"  {game_id} total SP sample excluded "
                                f"(home={sv(tt_row.get('home_sp_sample_flag'))} "
                                f"away={sv(tt_row.get('away_sp_sample_flag'))})",
                                "WARN",
                            )
                            rejection_rows.append({
                                "date": game_date,
                                "game_id": game_id,
                                "market": "total",
                                "side": "all",
                                "fail_reason": "sp_sample_excluded",
                                "fail_detail": (
                                    f"home={sv(tt_row.get('home_sp_sample_flag'))};"
                                    f"away={sv(tt_row.get('away_sp_sample_flag'))}"
                                ),
                                "prob_used_for_selection": None,
                                "prob_used_for_ev": None,
                                "prob_used_for_kelly": None,
                                "ev": None,
                                "kelly": None,
                                "odds": None,
                                "line": fv(tt_row.get("total")),
                                "raw_ev": None,
                                "adjusted_ev": None,
                                "ev_probability_source": None,
                                "kelly_probability_source": None,
                                })
                        else:
                            for r in process_total(tt_row, global_counters, rejection_rows):
                                k = f"{game_id}_{r['market_type']}_{r['bet_side']}_{r['line']}"

                                if k not in seen:
                                    if low_conf:
                                        summary["low_confidence"] += 1

                                    selected = {**base, **r, "low_confidence": low_conf}
                                    final.append(selected)
                                    selected_audit_rows.append(selected_audit_row(selected))
                                    seen.add(k)
                                    ps["tot"] += 1

                    if ml_row is not None:
                        for r in process_moneyline(ml_row, global_counters, rejection_rows):
                            k = f"{game_id}_{r['market_type']}_{r['bet_side']}_{r['line']}"

                            if k not in seen:
                                if low_conf:
                                    summary["low_confidence"] += 1

                                selected = {**base, **r, "low_confidence": low_conf}
                                final.append(selected)
                                selected_audit_rows.append(selected_audit_row(selected))
                                seen.add(k)
                                ps["ml"] += 1

                ps["bets"] = len(final)

                if final:
                    out = OUTPUT_DIR / f"{slate}_MLB.csv"
                    out_df = pd.DataFrame(final)
                    validation_counts = write_output_csv(out_df, out, f"{slate} selected output")

                    for key, value in validation_counts.items():
                        summary[key] += value

                    summary["slates_written"] += 1
                    summary["total_bets"] += len(final)
                    summary["moneyline_bets"] += ps["ml"]
                    summary["run_line_bets"] += ps["rl"]
                    summary["total_mkt_bets"] += ps["tot"]

                    _log(
                        f"WROTE: {out.name} "
                        f"({len(final)} bets | ml={ps['ml']} rl={ps['rl']} tot={ps['tot']})"
                    )
                else:
                    _log(f"{slate} no bets passed filters", "WARN")
                    ps["status"] = "no_bets"

            except ValueError as e:
                message = str(e)
                if "multiple rows for one game_id" in message or "Multiple rows matched game_id" in message:
                    summary["duplicate_game_id_errors"] += 1
                _log(f"{slate} SCHEMA FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                ps["status"] = "schema_error"
                summary["schema_errors"] += 1
                summary["errors"] += 1

            except Exception as e:
                _log(f"{slate} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                ps["status"] = "error"
                summary["errors"] += 1

            per_slate.append(ps)

        summary["counters"] = global_counters

        rejection_df = pd.DataFrame(rejection_rows, columns=REJECTION_AUDIT_COLUMNS)
        selected_audit_df = pd.DataFrame(selected_audit_rows, columns=SELECTED_AUDIT_COLUMNS)

        rejection_audit_path = AUDIT_DIR / "selection_rejection_audit.csv"
        selected_audit_path = AUDIT_DIR / "selected_bet_audit.csv"

        rejection_df.to_csv(rejection_audit_path, index=False)
        selected_audit_df.to_csv(selected_audit_path, index=False)

        summary["rejection_audit_rows"] = len(rejection_df)
        summary["selected_audit_rows"] = len(selected_audit_df)

        _log(f"WROTE AUDIT: {rejection_audit_path} rows={len(rejection_df)}")
        _log(f"WROTE AUDIT: {selected_audit_path} rows={len(selected_audit_df)}")

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    _write_summary(summary, per_slate)

    if summary["errors"] > 0 or summary["schema_errors"] > 0:
        print(
            f"baseball select_bets completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        raise SystemExit(1)

    print(
        f"baseball select_bets complete. "
        f"run_mode={summary['run_mode']} "
        f"slates_written={summary['slates_written']} "
        f"total_bets={summary['total_bets']} "
        f"moneyline_bets={summary['moneyline_bets']} "
        f"run_line_bets={summary['run_line_bets']} "
        f"total_mkt_bets={summary['total_mkt_bets']} "
        f"selected_nonpositive_kelly={summary['selected_nonpositive_kelly']} "
        f"selected_probability_source_mismatch={summary['selected_probability_source_mismatch']} "
        f"rain_excluded_will_it_rain={summary['rain_excluded_will_it_rain']} "
        f"rain_excluded_symbol_code={summary['rain_excluded_symbol_code']} "
        f"rejection_audit_rows={summary['rejection_audit_rows']} "
        f"selected_audit_rows={summary['selected_audit_rows']} "
        f"row_count_warnings={summary['row_count_warnings']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/03_edges/compute_ev_kelly.py

import traceback
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("docs/win/baseball/mlb/03_edges")
OUTPUT_DIR = Path("docs/win/baseball/mlb/03_edges/ev_kelly")
AUDIT_DIR = OUTPUT_DIR / "audit"
ERROR_DIR = Path("docs/win/baseball/mlb/errors/03_edges")
LOG_FILE = ERROR_DIR / "compute_ev_kelly.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LEAKAGE_AUDIT_DIR = Path("docs/win/baseball/mlb/audit")
LEAKAGE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LEAKAGE_AUDIT_FILE = LEAKAGE_AUDIT_DIR / "leakage_audit.csv"
FORBIDDEN_READ_TOKENS = [
    "05_" + "final_scores",
    "final" + "_scores",
    "graded",
    "results",
    "reports",
]  # LEAKAGE_GUARD_ALLOWED_REFERENCE
SCRIPT_NAME = "compute_ev_kelly.py"
STAGE_NAME = "03_edges_ev_kelly"


def record_file_read(path: Path, path_allowed: bool, reason: str) -> None:
    path = Path(path)
    new_file = not LEAKAGE_AUDIT_FILE.exists()

    with open(LEAKAGE_AUDIT_FILE, "a", encoding="utf-8", newline="") as f:
        if new_file:
            f.write("script,file_read,path_allowed,reason,stage,timestamp\n")

        safe_path = str(path).replace('"', "''")

        f.write(
            f'{SCRIPT_NAME},"{safe_path}",{1 if path_allowed else 0},'
            f'"{reason}",{STAGE_NAME},{datetime.now(UTC).isoformat()}\n'
        )


def assert_read_path_allowed(path: Path) -> None:
    path = Path(path)
    lower_path = str(path).replace("\\", "/").lower()
    matched = [token for token in FORBIDDEN_READ_TOKENS if token in lower_path]

    if matched:
        reason = "forbidden_pre_selection_read:" + ";".join(matched)
        record_file_read(path, False, reason)
        raise RuntimeError(
            f"Blocked forbidden pre-selection read path: {path} ({reason})"
        )

    record_file_read(path, True, "allowed")


def read_csv_guarded(path: Path) -> pd.DataFrame:
    assert_read_path_allowed(path)
    return pd.read_csv(path)


PROB_TOLERANCE = 1e-6

PROBABILITY_SOURCES = {
    "moneyline": {
        "home": "home_normalized_prob_moneyline",
        "away": "away_normalized_prob_moneyline",
    },
    "run_line": {
        "home": "home_normalized_prob_run_line",
        "away": "away_normalized_prob_run_line",
    },
    "total": {
        "over": "over_normalized_prob_total",
        "under": "under_normalized_prob_total",
    },
}

MONEYLINE_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_normalized_prob_moneyline",
    "away_normalized_prob_moneyline",
    "home_dk_decimal_moneyline",
    "away_dk_decimal_moneyline",
    "home_edge_decimal_moneyline",
    "away_edge_decimal_moneyline",
]

RUN_LINE_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_normalized_prob_run_line",
    "away_normalized_prob_run_line",
    "home_dk_run_line_decimal",
    "away_dk_run_line_decimal",
    "home_edge_decimal_run_line",
    "away_edge_decimal_run_line",
]

TOTAL_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "over_normalized_prob_total",
    "under_normalized_prob_total",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "over_edge_decimal_total",
    "under_edge_decimal_total",
]

FORBIDDEN_RUN_LINE_COLUMNS = [
    "home_run_line_prob",
    "away_run_line_prob",
]


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_processed              : {summary['files_processed']}",
        f"  rows_processed               : {summary['rows_processed']}",
        f"  moneyline_files              : {summary['moneyline_files']}",
        f"  run_line_files               : {summary['run_line_files']}",
        f"  total_files                  : {summary['total_files']}",
        f"  skipped                      : {summary['skipped']}",
        f"  schema_errors                : {summary['schema_errors']}",
        f"  neg_kelly_clipped            : {summary['neg_kelly_clipped']}",
        f"  missing_adj_ev               : {summary['missing_adj_ev']}",
        f"  probability_source_mismatches: "
        f"{summary['probability_source_mismatches']}",
        f"  positive_ev_zero_kelly       : "
        f"{summary['positive_ev_zero_kelly']}",
        f"  raw_adjusted_ev_sign_flips   : "
        f"{summary['raw_adjusted_ev_sign_flips']}",
        f"  adjusted_only_positive       : "
        f"{summary['adjusted_only_positive']}",
        f"  audit_rows                   : {summary['audit_rows']}",
        f"  errors                       : {summary['errors']}",
        "",
        f"  {'file':<48} {'market':<12} {'rows':>5} "
        f"{'neg_kelly':>10} {'missing_adj':>12} {'status':>14}",
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<48} {pf['market']:<12} {pf['rows']:>5} "
            f"{pf['neg_kelly']:>10} {pf['missing_adj']:>12} "
            f"{pf['status']:>14}"
        )

    status = (
        "SUCCESS"
        if summary["errors"] == 0 and summary["schema_errors"] == 0
        else "COMPLETED WITH ERRORS"
    )
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA GUARDS
# =========================

def duplicate_columns(columns) -> list:
    seen = set()
    dupes = []

    for col in columns:
        if col in seen and col not in dupes:
            dupes.append(col)

        seen.add(col)

    return dupes


def assert_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def assert_required_columns(
    df: pd.DataFrame,
    required_columns: list,
    label: str,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def assert_forbidden_columns_absent(
    df: pd.DataFrame,
    forbidden_columns: list,
    label: str,
) -> None:
    present = [col for col in forbidden_columns if col in df.columns]

    if present:
        raise ValueError(
            f"{label} contains obsolete forbidden columns: {present}. "
            f"Use home_normalized_prob_run_line / "
            f"away_normalized_prob_run_line."
        )


def validate_probability_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label: str,
) -> None:
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    both_blank = a.isna() & b.isna()
    incomplete = a.isna() ^ b.isna()
    invalid_sum = (
        (~both_blank)
        & (~incomplete)
        & ((a + b - 1.0).abs() > PROB_TOLERANCE)
    )
    bad = incomplete | invalid_sum

    if bad.any():
        sample = (
            df.loc[bad, ["game_id", col_a, col_b]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{label} official probability columns contain an incomplete "
            f"pair or do not sum to 1.0 within tolerance; "
            f"bad_rows={int(bad.sum())}; sample={sample}"
        )


def validate_input_schema(
    df: pd.DataFrame,
    market: str,
    file_name: str,
) -> None:
    assert_no_duplicate_columns(df, f"{file_name} input")

    if market == "moneyline":
        assert_required_columns(
            df,
            MONEYLINE_REQUIRED_COLUMNS,
            f"{file_name} moneyline input",
        )
        validate_probability_pair(
            df,
            "home_normalized_prob_moneyline",
            "away_normalized_prob_moneyline",
            f"{file_name} moneyline input",
        )

    elif market == "run_line":
        assert_required_columns(
            df,
            RUN_LINE_REQUIRED_COLUMNS,
            f"{file_name} run_line input",
        )
        assert_forbidden_columns_absent(
            df,
            FORBIDDEN_RUN_LINE_COLUMNS,
            f"{file_name} run_line input",
        )
        validate_probability_pair(
            df,
            "home_normalized_prob_run_line",
            "away_normalized_prob_run_line",
            f"{file_name} run_line input",
        )

    elif market == "total":
        assert_required_columns(
            df,
            TOTAL_REQUIRED_COLUMNS,
            f"{file_name} total input",
        )
        validate_probability_pair(
            df,
            "over_normalized_prob_total",
            "under_normalized_prob_total",
            f"{file_name} total input",
        )

    else:
        raise ValueError(
            f"{file_name} unknown market for schema validation: {market}"
        )


def write_csv_checked(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    assert_no_duplicate_columns(df, f"{output_path} output")
    df.to_csv(output_path, index=False)


# =========================
# HELPERS
# =========================

def compute_ev(p, dec):
    p = pd.to_numeric(p, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")
    return (p * dec) - 1


def compute_kelly(p, dec, file_name=""):
    p = pd.to_numeric(p, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")

    b = dec - 1
    q = 1 - p

    k = pd.Series(np.nan, index=p.index, dtype="float64")
    valid = (
        b.notna()
        & (b != 0)
        & p.notna()
        & np.isfinite(b)
        & np.isfinite(p)
    )

    k.loc[valid] = (
        (b.loc[valid] * p.loc[valid]) - q.loc[valid]
    ) / b.loc[valid]

    neg = k[k.notna() & (k < 0)]

    if not neg.empty:
        _log(
            f"{file_name} | {len(neg)} negative Kelly values clipped to 0 "
            f"(min={neg.min():.4f})",
            "WARN",
        )

    k = k.clip(lower=0)
    return k, len(neg)


def adjusted_ev(df, adjusted_col, fallback_ev, file_name):
    """
    Contextual adjustments affect EV only.

    The official probability source is used for raw EV and Kelly. The existing
    adjusted edge column is preserved as adjusted EV. Missing adjusted values
    fall back to raw EV and are counted.
    """
    raw = pd.to_numeric(fallback_ev, errors="coerce")
    adj = pd.to_numeric(df[adjusted_col], errors="coerce")

    missing = int(adj.isna().sum())

    if missing > 0:
        _log(
            f"{file_name} | {missing} rows missing adjusted EV in "
            f"{adjusted_col}; using raw EV fallback for those rows",
            "WARN",
        )

    return adj.where(adj.notna(), raw), missing


def sign_flip_count(raw_ev, adjusted):
    raw = pd.to_numeric(raw_ev, errors="coerce")
    adj = pd.to_numeric(adjusted, errors="coerce")

    return int(
        (
            ((raw > 0) & (adj < 0))
            | ((raw < 0) & (adj > 0))
        ).sum()
    )


def positive_ev_zero_kelly_count(ev, kelly):
    ev = pd.to_numeric(ev, errors="coerce")
    kelly = pd.to_numeric(kelly, errors="coerce")

    return int(((ev > 0) & (kelly <= 0)).sum())


def adjusted_only_positive_count(raw_ev, adjusted):
    raw = pd.to_numeric(raw_ev, errors="coerce")
    adj = pd.to_numeric(adjusted, errors="coerce")

    return int(((raw <= 0) & (adj > 0)).sum())


def probability_source_mismatch_count(
    df: pd.DataFrame,
    sides: list,
) -> int:
    mismatch = pd.Series(False, index=df.index)

    for side in sides:
        mismatch = mismatch | (
            df[f"{side}_ev_probability_source"].astype(str)
            != df[f"{side}_kelly_probability_source"].astype(str)
        )

        prob_ev = pd.to_numeric(
            df[f"{side}_prob_for_ev"],
            errors="coerce",
        )
        prob_kelly = pd.to_numeric(
            df[f"{side}_prob_for_kelly"],
            errors="coerce",
        )

        mismatch = mismatch | (
            (prob_ev - prob_kelly).abs() > PROB_TOLERANCE
        )

    return int(mismatch.sum())


def probability_basis_columns(
    df: pd.DataFrame,
    side: str,
    source_col: str,
) -> tuple[pd.Series, dict]:
    probability = pd.to_numeric(df[source_col], errors="coerce")

    columns = {
        f"{side}_prob_for_ev": probability,
        f"{side}_prob_for_kelly": probability,
        f"{side}_ev_probability_source": source_col,
        f"{side}_kelly_probability_source": source_col,
    }

    return probability, columns


def add_columns_at_once(
    df: pd.DataFrame,
    columns: dict,
) -> pd.DataFrame:
    """
    Add or replace calculated columns in one operation.

    Repeated single-column assignments fragment wide DataFrames and trigger
    pandas PerformanceWarning messages. Building one calculated-column frame
    and concatenating it once keeps the table compact.
    """
    calculated = pd.DataFrame(columns, index=df.index)
    overlapping = [
        col for col in calculated.columns if col in df.columns
    ]

    if overlapping:
        df = df.drop(columns=overlapping)

    return pd.concat([df, calculated], axis=1).copy()


def status_for_row(
    raw_ev,
    adjusted,
    kelly,
    ev_source,
    kelly_source,
) -> str:
    statuses = []

    try:
        raw_ev = float(raw_ev)
    except Exception:
        raw_ev = np.nan

    try:
        adjusted = float(adjusted)
    except Exception:
        adjusted = np.nan

    try:
        kelly = float(kelly)
    except Exception:
        kelly = np.nan

    if ev_source != kelly_source:
        statuses.append("probability_source_mismatch")

    if (
        pd.notna(adjusted)
        and adjusted > 0
        and pd.notna(kelly)
        and kelly <= 0
    ):
        statuses.append("positive_ev_zero_kelly")

    if pd.notna(raw_ev) and pd.notna(adjusted):
        if (
            (raw_ev > 0 and adjusted < 0)
            or (raw_ev < 0 and adjusted > 0)
        ):
            statuses.append("raw_adjusted_ev_sign_flip")

        if raw_ev <= 0 and adjusted > 0:
            statuses.append("adjusted_only_positive")

    if not statuses:
        statuses.append("ok")

    return ";".join(statuses)


def audit_rows(
    df: pd.DataFrame,
    market: str,
    sides: list,
    decimal_cols: dict,
    ev_cols: dict,
    kelly_cols: dict,
) -> list:
    records = []

    for side in sides:
        for _, row in df.iterrows():
            raw_col = ev_cols[side]["raw"]
            adjusted_col = ev_cols[side]["adjusted"]
            final_col = ev_cols[side]["final"]
            kelly_col = kelly_cols[side]
            dec_col = decimal_cols[side]

            records.append(
                {
                    "date": row.get("game_date"),
                    "game_id": row.get("game_id"),
                    "market": market,
                    "side": side,
                    "prob_for_ev": row.get(
                        f"{side}_prob_for_ev"
                    ),
                    "prob_for_kelly": row.get(
                        f"{side}_prob_for_kelly"
                    ),
                    "dk_decimal": row.get(dec_col),
                    "raw_ev": row.get(raw_col),
                    "adjusted_ev": row.get(adjusted_col),
                    "ev": row.get(final_col),
                    "kelly": row.get(kelly_col),
                    "ev_probability_source": row.get(
                        f"{side}_ev_probability_source"
                    ),
                    "kelly_probability_source": row.get(
                        f"{side}_kelly_probability_source"
                    ),
                    "status": status_for_row(
                        row.get(raw_col),
                        row.get(adjusted_col),
                        row.get(kelly_col),
                        row.get(
                            f"{side}_ev_probability_source"
                        ),
                        row.get(
                            f"{side}_kelly_probability_source"
                        ),
                    ),
                }
            )

    return records


# =========================
# MARKET PROCESSORS
# =========================

def process_moneyline(df, file_name):
    home_prob, home_basis = probability_basis_columns(
        df,
        "home",
        PROBABILITY_SOURCES["moneyline"]["home"],
    )
    away_prob, away_basis = probability_basis_columns(
        df,
        "away",
        PROBABILITY_SOURCES["moneyline"]["away"],
    )

    home_raw_ev = compute_ev(
        home_prob,
        df["home_dk_decimal_moneyline"],
    )
    away_raw_ev = compute_ev(
        away_prob,
        df["away_dk_decimal_moneyline"],
    )

    home_adjusted_ev, h_missing = adjusted_ev(
        df,
        "home_edge_decimal_moneyline",
        home_raw_ev,
        file_name,
    )
    away_adjusted_ev, a_missing = adjusted_ev(
        df,
        "away_edge_decimal_moneyline",
        away_raw_ev,
        file_name,
    )

    home_kelly, h_neg = compute_kelly(
        home_prob,
        df["home_dk_decimal_moneyline"],
        file_name,
    )
    away_kelly, a_neg = compute_kelly(
        away_prob,
        df["away_dk_decimal_moneyline"],
        file_name,
    )

    calculated_columns = {
        **home_basis,
        **away_basis,
        "home_ml_raw_ev": home_raw_ev,
        "away_ml_raw_ev": away_raw_ev,
        "home_ml_adjusted_ev": home_adjusted_ev,
        "away_ml_adjusted_ev": away_adjusted_ev,
        "home_ml_ev": home_adjusted_ev,
        "away_ml_ev": away_adjusted_ev,
        "home_ml_kelly": home_kelly,
        "away_ml_kelly": away_kelly,
    }

    df = add_columns_at_once(df, calculated_columns)

    counts = {
        "probability_source_mismatches": (
            probability_source_mismatch_count(
                df,
                ["home", "away"],
            )
        ),
        "positive_ev_zero_kelly": (
            positive_ev_zero_kelly_count(
                df["home_ml_ev"],
                df["home_ml_kelly"],
            )
            + positive_ev_zero_kelly_count(
                df["away_ml_ev"],
                df["away_ml_kelly"],
            )
        ),
        "raw_adjusted_ev_sign_flips": (
            sign_flip_count(
                df["home_ml_raw_ev"],
                df["home_ml_adjusted_ev"],
            )
            + sign_flip_count(
                df["away_ml_raw_ev"],
                df["away_ml_adjusted_ev"],
            )
        ),
        "adjusted_only_positive": (
            adjusted_only_positive_count(
                df["home_ml_raw_ev"],
                df["home_ml_adjusted_ev"],
            )
            + adjusted_only_positive_count(
                df["away_ml_raw_ev"],
                df["away_ml_adjusted_ev"],
            )
        ),
    }

    audit = audit_rows(
        df,
        "moneyline",
        ["home", "away"],
        {
            "home": "home_dk_decimal_moneyline",
            "away": "away_dk_decimal_moneyline",
        },
        {
            "home": {
                "raw": "home_ml_raw_ev",
                "adjusted": "home_ml_adjusted_ev",
                "final": "home_ml_ev",
            },
            "away": {
                "raw": "away_ml_raw_ev",
                "adjusted": "away_ml_adjusted_ev",
                "final": "away_ml_ev",
            },
        },
        {
            "home": "home_ml_kelly",
            "away": "away_ml_kelly",
        },
    )

    return (
        df,
        h_neg + a_neg,
        h_missing + a_missing,
        counts,
        audit,
    )


def process_run_line(df, file_name):
    home_prob, home_basis = probability_basis_columns(
        df,
        "home",
        PROBABILITY_SOURCES["run_line"]["home"],
    )
    away_prob, away_basis = probability_basis_columns(
        df,
        "away",
        PROBABILITY_SOURCES["run_line"]["away"],
    )

    home_raw_ev = compute_ev(
        home_prob,
        df["home_dk_run_line_decimal"],
    )
    away_raw_ev = compute_ev(
        away_prob,
        df["away_dk_run_line_decimal"],
    )

    home_adjusted_ev, h_missing = adjusted_ev(
        df,
        "home_edge_decimal_run_line",
        home_raw_ev,
        file_name,
    )
    away_adjusted_ev, a_missing = adjusted_ev(
        df,
        "away_edge_decimal_run_line",
        away_raw_ev,
        file_name,
    )

    home_kelly, h_neg = compute_kelly(
        home_prob,
        df["home_dk_run_line_decimal"],
        file_name,
    )
    away_kelly, a_neg = compute_kelly(
        away_prob,
        df["away_dk_run_line_decimal"],
        file_name,
    )

    calculated_columns = {
        **home_basis,
        **away_basis,
        "home_rl_raw_ev": home_raw_ev,
        "away_rl_raw_ev": away_raw_ev,
        "home_rl_adjusted_ev": home_adjusted_ev,
        "away_rl_adjusted_ev": away_adjusted_ev,
        "home_rl_ev": home_adjusted_ev,
        "away_rl_ev": away_adjusted_ev,
        "home_rl_kelly": home_kelly,
        "away_rl_kelly": away_kelly,
    }

    df = add_columns_at_once(df, calculated_columns)

    counts = {
        "probability_source_mismatches": (
            probability_source_mismatch_count(
                df,
                ["home", "away"],
            )
        ),
        "positive_ev_zero_kelly": (
            positive_ev_zero_kelly_count(
                df["home_rl_ev"],
                df["home_rl_kelly"],
            )
            + positive_ev_zero_kelly_count(
                df["away_rl_ev"],
                df["away_rl_kelly"],
            )
        ),
        "raw_adjusted_ev_sign_flips": (
            sign_flip_count(
                df["home_rl_raw_ev"],
                df["home_rl_adjusted_ev"],
            )
            + sign_flip_count(
                df["away_rl_raw_ev"],
                df["away_rl_adjusted_ev"],
            )
        ),
        "adjusted_only_positive": (
            adjusted_only_positive_count(
                df["home_rl_raw_ev"],
                df["home_rl_adjusted_ev"],
            )
            + adjusted_only_positive_count(
                df["away_rl_raw_ev"],
                df["away_rl_adjusted_ev"],
            )
        ),
    }

    audit = audit_rows(
        df,
        "run_line",
        ["home", "away"],
        {
            "home": "home_dk_run_line_decimal",
            "away": "away_dk_run_line_decimal",
        },
        {
            "home": {
                "raw": "home_rl_raw_ev",
                "adjusted": "home_rl_adjusted_ev",
                "final": "home_rl_ev",
            },
            "away": {
                "raw": "away_rl_raw_ev",
                "adjusted": "away_rl_adjusted_ev",
                "final": "away_rl_ev",
            },
        },
        {
            "home": "home_rl_kelly",
            "away": "away_rl_kelly",
        },
    )

    return (
        df,
        h_neg + a_neg,
        h_missing + a_missing,
        counts,
        audit,
    )


def process_total(df, file_name):
    over_prob, over_basis = probability_basis_columns(
        df,
        "over",
        PROBABILITY_SOURCES["total"]["over"],
    )
    under_prob, under_basis = probability_basis_columns(
        df,
        "under",
        PROBABILITY_SOURCES["total"]["under"],
    )

    over_raw_ev = compute_ev(
        over_prob,
        df["dk_total_over_decimal"],
    )
    under_raw_ev = compute_ev(
        under_prob,
        df["dk_total_under_decimal"],
    )

    over_adjusted_ev, o_missing = adjusted_ev(
        df,
        "over_edge_decimal_total",
        over_raw_ev,
        file_name,
    )
    under_adjusted_ev, u_missing = adjusted_ev(
        df,
        "under_edge_decimal_total",
        under_raw_ev,
        file_name,
    )

    over_kelly, o_neg = compute_kelly(
        over_prob,
        df["dk_total_over_decimal"],
        file_name,
    )
    under_kelly, u_neg = compute_kelly(
        under_prob,
        df["dk_total_under_decimal"],
        file_name,
    )

    calculated_columns = {
        **over_basis,
        **under_basis,
        "over_raw_ev": over_raw_ev,
        "under_raw_ev": under_raw_ev,
        "over_adjusted_ev": over_adjusted_ev,
        "under_adjusted_ev": under_adjusted_ev,
        "over_ev": over_adjusted_ev,
        "under_ev": under_adjusted_ev,
        "over_kelly": over_kelly,
        "under_kelly": under_kelly,
    }

    df = add_columns_at_once(df, calculated_columns)

    counts = {
        "probability_source_mismatches": (
            probability_source_mismatch_count(
                df,
                ["over", "under"],
            )
        ),
        "positive_ev_zero_kelly": (
            positive_ev_zero_kelly_count(
                df["over_ev"],
                df["over_kelly"],
            )
            + positive_ev_zero_kelly_count(
                df["under_ev"],
                df["under_kelly"],
            )
        ),
        "raw_adjusted_ev_sign_flips": (
            sign_flip_count(
                df["over_raw_ev"],
                df["over_adjusted_ev"],
            )
            + sign_flip_count(
                df["under_raw_ev"],
                df["under_adjusted_ev"],
            )
        ),
        "adjusted_only_positive": (
            adjusted_only_positive_count(
                df["over_raw_ev"],
                df["over_adjusted_ev"],
            )
            + adjusted_only_positive_count(
                df["under_raw_ev"],
                df["under_adjusted_ev"],
            )
        ),
    }

    audit = audit_rows(
        df,
        "total",
        ["over", "under"],
        {
            "over": "dk_total_over_decimal",
            "under": "dk_total_under_decimal",
        },
        {
            "over": {
                "raw": "over_raw_ev",
                "adjusted": "over_adjusted_ev",
                "final": "over_ev",
            },
            "under": {
                "raw": "under_raw_ev",
                "adjusted": "under_adjusted_ev",
                "final": "under_ev",
            },
        },
        {
            "over": "over_kelly",
            "under": "under_kelly",
        },
    )

    return (
        df,
        o_neg + u_neg,
        o_missing + u_missing,
        counts,
        audit,
    )


def log_validation_counts(
    file_name: str,
    counts: dict,
) -> None:
    if counts["probability_source_mismatches"]:
        _log(
            f"{file_name} | probability source mismatches="
            f"{counts['probability_source_mismatches']}",
            "ERROR",
        )

    if counts["positive_ev_zero_kelly"]:
        _log(
            f"{file_name} | positive EV but zero Kelly rows="
            f"{counts['positive_ev_zero_kelly']}",
            "WARN",
        )

    if counts["raw_adjusted_ev_sign_flips"]:
        _log(
            f"{file_name} | raw EV and adjusted EV opposite-sign rows="
            f"{counts['raw_adjusted_ev_sign_flips']}",
            "WARN",
        )

    if counts["adjusted_only_positive"]:
        _log(
            f"{file_name} | adjusted EV positive while raw EV is "
            f"zero/negative rows="
            f"{counts['adjusted_only_positive']}",
            "WARN",
        )


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as log_f:
        log_f.write(
            f"=== compute_ev_kelly RUN {_now()} ===\n"
        )

    summary = {
        "files_processed": 0,
        "rows_processed": 0,
        "moneyline_files": 0,
        "run_line_files": 0,
        "total_files": 0,
        "skipped": 0,
        "schema_errors": 0,
        "neg_kelly_clipped": 0,
        "missing_adj_ev": 0,
        "probability_source_mismatches": 0,
        "positive_ev_zero_kelly": 0,
        "raw_adjusted_ev_sign_flips": 0,
        "adjusted_only_positive": 0,
        "audit_rows": 0,
        "errors": 0,
    }

    per_file = []
    all_audit_rows = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    _log(
        "Official probability basis: "
        "moneyline=normalized moneyline, "
        "run_line=normalized run_line, "
        "total=normalized total"
    )
    _log(
        "Contextual adjustments affect EV only. "
        "Kelly uses official normalized probability without "
        "contextual adjustment."
    )

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")

    for out_file in OUTPUT_DIR.glob("*.csv"):
        out_file.unlink()

    for old_audit in AUDIT_DIR.glob("*.csv"):
        old_audit.unlink()

    for input_file in input_files:
        name = input_file.name.lower()
        market = None

        pf = {
            "name": input_file.name,
            "market": "unknown",
            "rows": 0,
            "neg_kelly": 0,
            "missing_adj": 0,
            "status": "ok",
        }

        if "moneyline" in name:
            market = "moneyline"

        elif "run_line" in name:
            market = "run_line"

        elif "total" in name:
            market = "total"

        else:
            _log(
                f"SKIP unrecognized file: {input_file.name}"
            )
            pf["status"] = "skipped"
            summary["skipped"] += 1
            per_file.append(pf)
            continue

        pf["market"] = market

        _log(
            f"--- FILE: {input_file.name}  market={market}"
        )

        try:
            df = read_csv_guarded(input_file)

            if df.empty:
                _log(
                    f"{input_file.name} empty — skipping"
                )
                pf["status"] = "empty"
                summary["skipped"] += 1
                per_file.append(pf)
                continue

            try:
                validate_input_schema(
                    df,
                    market,
                    input_file.name,
                )

            except Exception as schema_error:
                _log(
                    f"{input_file.name} SCHEMA FAILED: "
                    f"{schema_error}",
                    "ERROR",
                )
                pf["status"] = "schema_error"
                summary["schema_errors"] += 1
                per_file.append(pf)
                continue

            pf["rows"] = len(df)
            summary["rows_processed"] += len(df)

            if market == "moneyline":
                (
                    df,
                    neg_kelly,
                    missing_adj,
                    counts,
                    audit,
                ) = process_moneyline(
                    df,
                    input_file.name,
                )
                summary["moneyline_files"] += 1

            elif market == "run_line":
                (
                    df,
                    neg_kelly,
                    missing_adj,
                    counts,
                    audit,
                ) = process_run_line(
                    df,
                    input_file.name,
                )
                summary["run_line_files"] += 1

            else:
                (
                    df,
                    neg_kelly,
                    missing_adj,
                    counts,
                    audit,
                ) = process_total(
                    df,
                    input_file.name,
                )
                summary["total_files"] += 1

            log_validation_counts(
                input_file.name,
                counts,
            )

            pf["neg_kelly"] = neg_kelly
            pf["missing_adj"] = missing_adj

            summary["neg_kelly_clipped"] += neg_kelly
            summary["missing_adj_ev"] += missing_adj
            summary["probability_source_mismatches"] += (
                counts["probability_source_mismatches"]
            )
            summary["positive_ev_zero_kelly"] += (
                counts["positive_ev_zero_kelly"]
            )
            summary["raw_adjusted_ev_sign_flips"] += (
                counts["raw_adjusted_ev_sign_flips"]
            )
            summary["adjusted_only_positive"] += (
                counts["adjusted_only_positive"]
            )
            summary["audit_rows"] += len(audit)
            all_audit_rows.extend(audit)

            if counts["probability_source_mismatches"] > 0:
                raise ValueError(
                    f"{input_file.name} EV/Kelly probability "
                    f"basis mismatch rows="
                    f"{counts['probability_source_mismatches']}"
                )

            output_path = OUTPUT_DIR / input_file.name

            write_csv_checked(
                df,
                output_path,
            )

            summary["files_processed"] += 1

            _log(
                f"WROTE: {output_path} "
                f"({len(df)} rows, "
                f"{neg_kelly} kelly clipped, "
                f"{missing_adj} adjusted EV fallback)"
            )

        except Exception as e:
            _log(
                f"{input_file.name} FAILED: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            pf["status"] = "error"
            summary["errors"] += 1

        per_file.append(pf)

    audit_path = (
        AUDIT_DIR / "post_ev_kelly_audit.csv"
    )

    audit_columns = [
        "date",
        "game_id",
        "market",
        "side",
        "prob_for_ev",
        "prob_for_kelly",
        "dk_decimal",
        "raw_ev",
        "adjusted_ev",
        "ev",
        "kelly",
        "ev_probability_source",
        "kelly_probability_source",
        "status",
    ]

    audit_df = pd.DataFrame(
        all_audit_rows,
        columns=audit_columns,
    )

    write_csv_checked(
        audit_df,
        audit_path,
    )

    _log(
        f"WROTE AUDIT: {audit_path} "
        f"rows={len(audit_df)}"
    )

    _write_summary(
        summary,
        per_file,
    )

    if (
        summary["errors"] > 0
        or summary["schema_errors"] > 0
    ):
        print(
            f"compute_ev_kelly completed with errors. "
            f"errors={summary['errors']} "
            f"schema_errors={summary['schema_errors']}"
        )
        raise SystemExit(1)

    print(
        f"compute_ev_kelly complete. "
        f"files_processed={summary['files_processed']} "
        f"rows_processed={summary['rows_processed']} "
        f"probability_source_mismatches="
        f"{summary['probability_source_mismatches']} "
        f"positive_ev_zero_kelly="
        f"{summary['positive_ev_zero_kelly']} "
        f"raw_adjusted_ev_sign_flips="
        f"{summary['raw_adjusted_ev_sign_flips']} "
        f"adjusted_only_positive="
        f"{summary['adjusted_only_positive']} "
        f"schema_errors={summary['schema_errors']} "
        f"errors={summary['errors']}"
    )


if __name__ == "__main__":
    main()

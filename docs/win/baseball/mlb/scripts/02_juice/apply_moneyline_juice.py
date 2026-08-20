#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/02_juice/apply_moneyline_juice.py
import glob
import math
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("docs/win/baseball/mlb/01_merge/01_merguiced")
SOURCE_MERGE_DIR = INPUT_DIR.parent
OUTPUT_DIR = Path("docs/win/baseball/mlb/02_juice")
AUDIT_DIR = OUTPUT_DIR / "audit"
JUICE_FILE = Path("docs/win/baseball/mlb/config/juice/mlb_ml_juice.csv")

ERROR_DIR = Path("docs/win/baseball/mlb/errors/02_juice")
LOG_FILE = ERROR_DIR / "apply_moneyline_juice.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZATION_TOLERANCE = 0.000001

REQUIRED_INPUT_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_fair_decimal_moneyline",
    "away_fair_decimal_moneyline",
]

DK_ODDS_COLUMNS = [
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
]

REQUIRED_JUICE_COLUMNS = [
    "band_min",
    "band_max",
    "extra_juice",
    "fav_ud",
    "venue",
]

OUTPUT_ADDED_COLUMNS = [
    "home_juiced_prob_moneyline",
    "away_juiced_prob_moneyline",
    "home_juiced_decimal_moneyline",
    "away_juiced_decimal_moneyline",
    "home_normalized_prob_moneyline",
    "away_normalized_prob_moneyline",
]

AUDIT_COLUMNS = [
    "date",
    "game_id",
    "market",
    "side",
    "dk_american",
    "dk_decimal",
    "fair_decimal",
    "juiced_decimal",
    "juiced_prob",
    "normalized_prob",
    "status",
]


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_found                       : {summary['files_found']}",
        f"  files_written                     : {summary['files_written']}",
        f"  total_rows                        : {summary['total_rows']}",
        f"  applied                           : {summary['applied']}",
        f"  skipped_bad                       : {summary['skipped_bad']}",
        f"  skipped_noband                    : {summary['skipped_noband']}",
        f"  missing_home_moneyline_dk         : {summary['missing_home_moneyline_dk']}",
        f"  missing_away_moneyline_dk         : {summary['missing_away_moneyline_dk']}",
        f"  missing_any_moneyline_dk          : {summary['missing_any_moneyline_dk']}",
        f"  stale_input_errors                : {summary['stale_input_errors']}",
        f"  normalization_errors              : {summary['normalization_errors']}",
        f"  schema_errors                     : {summary['schema_errors']}",
        f"  errors                            : {summary['errors']}",
        "",
        f"  {'file':<45} {'rows':>5} {'applied':>8} {'bad':>5} {'noband':>7} {'miss_home':>10} {'miss_away':>10} {'miss_any':>9} {'schema':>7}",
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<45} {pf['rows']:>5} {pf['applied']:>8} "
            f"{pf['skipped_bad']:>5} {pf['skipped_noband']:>7} "
            f"{pf['missing_home_moneyline_dk']:>10} {pf['missing_away_moneyline_dk']:>10} "
            f"{pf['missing_any_moneyline_dk']:>9} {pf['schema_errors']:>7}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA GUARDS
# =========================

def duplicate_columns(columns) -> list:
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
        raise ValueError(f"{label} duplicate columns: {dupes}")


def validate_required_columns(df: pd.DataFrame, required_columns: list, label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_csv_validated(path: Path, required_columns: list, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_no_duplicate_columns(df, label)
    validate_required_columns(df, required_columns, label)
    return df


def write_csv_validated(df: pd.DataFrame, path: Path, label: str) -> None:
    validate_no_duplicate_columns(df, label)
    df.to_csv(path, index=False)


def require_nonempty_columns(df: pd.DataFrame, columns: list, label: str) -> None:
    fully_empty = []
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"{label} missing required DK odds column: {col}")
        if df[col].isna().all() or df[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).isna().all():
            fully_empty.append(col)

    if fully_empty:
        raise ValueError(f"{label} has fully empty DK odds columns: {fully_empty}")


def validate_stale_input(input_path: Path) -> None:
    source_path = SOURCE_MERGE_DIR / input_path.name
    if not source_path.exists():
        _log(f"stale_check source_missing source={source_path} input={input_path}; continuing", "WARN")
        return

    if input_path.stat().st_mtime < source_path.stat().st_mtime:
        raise ValueError(
            f"stale 01_merguiced input: {input_path} is older than source merge file {source_path}"
        )


def validate_normalized_pair(df: pd.DataFrame, left_col: str, right_col: str, label: str) -> int:
    bad = 0
    for idx, row in df.iterrows():
        left = pd.to_numeric(pd.Series([row[left_col]]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([row[right_col]]), errors="coerce").iloc[0]

        if pd.isna(left) and pd.isna(right):
            continue

        if pd.isna(left) or pd.isna(right):
            bad += 1
            _log(f"{label} row={idx} reason=incomplete_normalized_pair {left_col}={left} {right_col}={right}", "ERROR")
            continue

        total = float(left) + float(right)
        if not math.isfinite(total) or abs(total - 1.0) > NORMALIZATION_TOLERANCE:
            bad += 1
            _log(f"{label} row={idx} reason=normalized_sum_invalid total={total}", "ERROR")

    return bad


# =========================
# JUICE CONFIG VALIDATION
# =========================

def validate_juice_config(juice_df: pd.DataFrame) -> None:
    invalid = juice_df[
        juice_df["band_min"].isna() |
        juice_df["band_max"].isna() |
        juice_df["extra_juice"].isna() |
        (juice_df["band_min"] >= juice_df["band_max"]) |
        (~juice_df["fav_ud"].isin(["favorite", "underdog"])) |
        (~juice_df["venue"].isin(["home", "away"]))
    ]
    if not invalid.empty:
        raise ValueError(f"moneyline juice config contains invalid rows: {len(invalid)}")

    dupes = juice_df.duplicated(subset=["band_min", "band_max", "fav_ud", "venue"], keep=False)
    if dupes.any():
        raise ValueError(f"moneyline juice config contains duplicate bands: {int(dupes.sum())}")

    required_combos = {(fav_ud, venue) for fav_ud in ["favorite", "underdog"] for venue in ["home", "away"]}
    present_combos = set(zip(juice_df["fav_ud"], juice_df["venue"]))
    missing_combos = sorted(required_combos - present_combos)
    if missing_combos:
        raise ValueError(f"moneyline juice config missing fav_ud/venue combinations: {missing_combos}")

    overlap_count = 0
    for (fav_ud, venue), group in juice_df.groupby(["fav_ud", "venue"]):
        group = group.sort_values(["band_min", "band_max"])
        prev_max = None
        for _, row in group.iterrows():
            if prev_max is not None and float(row["band_min"]) < prev_max:
                overlap_count += 1
            prev_max = max(prev_max, float(row["band_max"])) if prev_max is not None else float(row["band_max"])

    if overlap_count:
        raise ValueError(f"moneyline juice config contains overlapping bands: {overlap_count}")


# =========================
# JUICE LOOKUP
# =========================

def find_band_row(juice_df, american, fav_ud, venue):
    band = juice_df[
        (juice_df["band_min"] <= american) &
        (american < juice_df["band_max"]) &
        (juice_df["fav_ud"] == fav_ud) &
        (juice_df["venue"] == venue)
    ]

    if len(band) != 1:
        return None

    return float(band.iloc[0]["extra_juice"])


# =========================
# AUDIT
# =========================

def append_audit_rows(audit_rows: list, row: pd.Series, side: str, status: str, values: dict | None = None) -> None:
    values = values or {}
    audit_rows.append({
        "date": row.get("game_date", pd.NA),
        "game_id": row.get("game_id", pd.NA),
        "market": "moneyline",
        "side": side,
        "dk_american": values.get("dk_american", pd.NA),
        "dk_decimal": values.get("dk_decimal", pd.NA),
        "fair_decimal": values.get("fair_decimal", pd.NA),
        "juiced_decimal": values.get("juiced_decimal", pd.NA),
        "juiced_prob": values.get("juiced_prob", pd.NA),
        "normalized_prob": values.get("normalized_prob", pd.NA),
        "status": status,
    })


# =========================
# ROW PROCESSOR
# =========================

def process_row(df, juice_df, idx, row, audit_rows):
    missing_dk = False
    for col in DK_ODDS_COLUMNS:
        if pd.isna(row[col]):
            missing_dk = True

    if missing_dk:
        append_audit_rows(audit_rows, row, "home", "missing_dk_odds")
        append_audit_rows(audit_rows, row, "away", "missing_dk_odds")
        _log(f"row={idx} reason=missing_dk_odds", "SKIP")
        return df, "bad"

    try:
        home_american = float(row["home_dk_moneyline_american"])
        away_american = float(row["away_dk_moneyline_american"])
        home_dk_decimal = float(row["home_dk_moneyline_decimal"])
        away_dk_decimal = float(row["away_dk_moneyline_decimal"])
        home_fair = float(row["home_fair_decimal_moneyline"])
        away_fair = float(row["away_fair_decimal_moneyline"])
    except Exception:
        append_audit_rows(audit_rows, row, "home", "bad_parse")
        append_audit_rows(audit_rows, row, "away", "bad_parse")
        _log(f"row={idx} reason=conversion_failed", "SKIP")
        return df, "bad"

    for label, value in [
        ("home_american", home_american),
        ("away_american", away_american),
        ("home_dk_decimal", home_dk_decimal),
        ("away_dk_decimal", away_dk_decimal),
        ("home_fair", home_fair),
        ("away_fair", away_fair),
    ]:
        if not math.isfinite(value):
            append_audit_rows(audit_rows, row, "home", "invalid_numeric")
            append_audit_rows(audit_rows, row, "away", "invalid_numeric")
            _log(f"row={idx} reason=invalid_numeric {label}={value}", "SKIP")
            return df, "bad"

    if home_fair <= 1 or away_fair <= 1 or home_dk_decimal <= 1 or away_dk_decimal <= 1:
        append_audit_rows(audit_rows, row, "home", "invalid_decimal")
        append_audit_rows(audit_rows, row, "away", "invalid_decimal")
        _log(f"row={idx} reason=invalid_decimal", "SKIP")
        return df, "bad"

    home_fav_ud = "favorite" if home_american < 0 else "underdog"
    away_fav_ud = "favorite" if away_american < 0 else "underdog"

    home_extra = find_band_row(juice_df, home_american, home_fav_ud, "home")
    away_extra = find_band_row(juice_df, away_american, away_fav_ud, "away")

    if home_extra is None or away_extra is None:
        append_audit_rows(audit_rows, row, "home", "missing_band", {
            "dk_american": home_american,
            "dk_decimal": home_dk_decimal,
            "fair_decimal": home_fair,
        })
        append_audit_rows(audit_rows, row, "away", "missing_band", {
            "dk_american": away_american,
            "dk_decimal": away_dk_decimal,
            "fair_decimal": away_fair,
        })
        _log(f"row={idx} reason=band_lookup_failed home={home_american} away={away_american}", "SKIP")
        return df, "noband"

    home_juiced_decimal = home_fair * (1 - home_extra)
    away_juiced_decimal = away_fair * (1 - away_extra)

    if not math.isfinite(home_juiced_decimal) or not math.isfinite(away_juiced_decimal):
        append_audit_rows(audit_rows, row, "home", "invalid_juiced_decimal")
        append_audit_rows(audit_rows, row, "away", "invalid_juiced_decimal")
        _log(f"row={idx} reason=nonfinite_juiced_decimal home={home_juiced_decimal} away={away_juiced_decimal}", "SKIP")
        return df, "bad"

    if home_juiced_decimal <= 1 or away_juiced_decimal <= 1:
        append_audit_rows(audit_rows, row, "home", "invalid_juiced_decimal")
        append_audit_rows(audit_rows, row, "away", "invalid_juiced_decimal")
        _log(f"row={idx} reason=invalid_juiced_decimal home={home_juiced_decimal} away={away_juiced_decimal}", "SKIP")
        return df, "bad"

    home_juiced_prob = 1 / home_juiced_decimal
    away_juiced_prob = 1 / away_juiced_decimal
    total = home_juiced_prob + away_juiced_prob

    if not math.isfinite(total) or total <= 0:
        append_audit_rows(audit_rows, row, "home", "invalid_normalization_total")
        append_audit_rows(audit_rows, row, "away", "invalid_normalization_total")
        _log(f"row={idx} reason=invalid_normalization_total val={total}", "SKIP")
        return df, "bad"

    home_normalized = home_juiced_prob / total
    away_normalized = away_juiced_prob / total

    df.at[idx, "home_juiced_decimal_moneyline"] = home_juiced_decimal
    df.at[idx, "away_juiced_decimal_moneyline"] = away_juiced_decimal
    df.at[idx, "home_juiced_prob_moneyline"] = home_juiced_prob
    df.at[idx, "away_juiced_prob_moneyline"] = away_juiced_prob
    df.at[idx, "home_normalized_prob_moneyline"] = home_normalized
    df.at[idx, "away_normalized_prob_moneyline"] = away_normalized

    append_audit_rows(audit_rows, row, "home", "juiced", {
        "dk_american": home_american,
        "dk_decimal": home_dk_decimal,
        "fair_decimal": home_fair,
        "juiced_decimal": home_juiced_decimal,
        "juiced_prob": home_juiced_prob,
        "normalized_prob": home_normalized,
    })
    append_audit_rows(audit_rows, row, "away", "juiced", {
        "dk_american": away_american,
        "dk_decimal": away_dk_decimal,
        "fair_decimal": away_fair,
        "juiced_decimal": away_juiced_decimal,
        "juiced_prob": away_juiced_prob,
        "normalized_prob": away_normalized,
    })

    return df, "ok"


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_moneyline_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0,
        "files_written": 0,
        "total_rows": 0,
        "applied": 0,
        "skipped_bad": 0,
        "skipped_noband": 0,
        "missing_home_moneyline_dk": 0,
        "missing_away_moneyline_dk": 0,
        "missing_any_moneyline_dk": 0,
        "stale_input_errors": 0,
        "normalization_errors": 0,
        "schema_errors": 0,
        "errors": 0,
    }

    per_file = []
    audit_rows = []

    for f in OUTPUT_DIR.glob("*moneyline.csv"):
        f.unlink()
    for f in AUDIT_DIR.glob("*moneyline*post_juice_audit.csv"):
        f.unlink()

    try:
        _log(f"INPUT_DIR : {INPUT_DIR}")
        _log(f"SOURCE_MERGE_DIR: {SOURCE_MERGE_DIR}")
        _log(f"JUICE_FILE: {JUICE_FILE}")

        juice_df = read_csv_validated(JUICE_FILE, REQUIRED_JUICE_COLUMNS, f"juice file {JUICE_FILE}")
        juice_df["band_min"] = pd.to_numeric(juice_df["band_min"], errors="coerce")
        juice_df["band_max"] = pd.to_numeric(juice_df["band_max"], errors="coerce")
        juice_df["extra_juice"] = pd.to_numeric(juice_df["extra_juice"], errors="coerce")
        juice_df["fav_ud"] = juice_df["fav_ud"].astype(str).str.strip().str.lower()
        juice_df["venue"] = juice_df["venue"].astype(str).str.strip().str.lower()
        validate_juice_config(juice_df)

        files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_moneyline.csv")))
        summary["files_found"] = len(files)
        _log(f"Files found: {len(files)}")

        if not files:
            _log("No moneyline files found — exiting", "WARN")
            _write_summary(summary, per_file)
            return

        for file_path in files:
            in_path = Path(file_path)
            out_path = OUTPUT_DIR / in_path.name

            pf = {
                "name": in_path.name,
                "rows": 0,
                "applied": 0,
                "skipped_bad": 0,
                "skipped_noband": 0,
                "missing_home_moneyline_dk": 0,
                "missing_away_moneyline_dk": 0,
                "missing_any_moneyline_dk": 0,
                "schema_errors": 0,
            }

            _log(f"--- FILE: {in_path.name}")

            try:
                validate_stale_input(in_path)

                df = read_csv_validated(in_path, REQUIRED_INPUT_COLUMNS, f"{in_path.name} input")
                require_nonempty_columns(df, DK_ODDS_COLUMNS, f"{in_path.name} input")

                if df.empty:
                    _log(f"{in_path.name} empty — skipping")
                    per_file.append(pf)
                    continue

                for c in [
                    "home_dk_moneyline_american",
                    "away_dk_moneyline_american",
                    "home_dk_moneyline_decimal",
                    "away_dk_moneyline_decimal",
                    "home_fair_decimal_moneyline",
                    "away_fair_decimal_moneyline",
                    "home_prob",
                    "away_prob",
                ]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                pf["missing_home_moneyline_dk"] = int(df[["home_dk_moneyline_american", "home_dk_moneyline_decimal"]].isna().any(axis=1).sum())
                pf["missing_away_moneyline_dk"] = int(df[["away_dk_moneyline_american", "away_dk_moneyline_decimal"]].isna().any(axis=1).sum())
                pf["missing_any_moneyline_dk"] = int(df[DK_ODDS_COLUMNS].isna().any(axis=1).sum())

                for c in OUTPUT_ADDED_COLUMNS:
                    df[c] = pd.NA

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)
                summary["missing_home_moneyline_dk"] += pf["missing_home_moneyline_dk"]
                summary["missing_away_moneyline_dk"] += pf["missing_away_moneyline_dk"]
                summary["missing_any_moneyline_dk"] += pf["missing_any_moneyline_dk"]

                for idx, row in df.iterrows():
                    df, result = process_row(df, juice_df, idx, row, audit_rows)

                    if result == "ok":
                        pf["applied"] += 1
                    elif result == "noband":
                        pf["skipped_noband"] += 1
                    else:
                        pf["skipped_bad"] += 1

                norm_bad = validate_normalized_pair(
                    df,
                    "home_normalized_prob_moneyline",
                    "away_normalized_prob_moneyline",
                    f"{in_path.name} moneyline",
                )
                if norm_bad:
                    summary["normalization_errors"] += norm_bad
                    raise ValueError(f"{in_path.name} has {norm_bad} invalid normalized moneyline probability rows")

                write_csv_validated(df, out_path, f"{in_path.name} output")

                summary["files_written"] += 1
                summary["applied"] += pf["applied"]
                summary["skipped_bad"] += pf["skipped_bad"]
                summary["skipped_noband"] += pf["skipped_noband"]

                _log(
                    f"{in_path.name} | rows={pf['rows']} applied={pf['applied']} "
                    f"skipped_bad={pf['skipped_bad']} skipped_noband={pf['skipped_noband']} "
                    f"missing_any_moneyline_dk={pf['missing_any_moneyline_dk']}"
                )
                _log(f"WROTE: {out_path}")

            except ValueError as e:
                if "stale 01_merguiced input" in str(e):
                    summary["stale_input_errors"] += 1
                pf["schema_errors"] += 1
                summary["schema_errors"] += 1
                summary["errors"] += 1
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")

            except Exception as e:
                summary["errors"] += 1
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1
        _write_summary(summary, per_file)
        sys.exit(1)

    audit_path = AUDIT_DIR / "moneyline_post_juice_audit.csv"
    pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    _log(f"WROTE AUDIT: {audit_path}")

    _write_summary(summary, per_file)

    if summary["schema_errors"] > 0 or summary["errors"] > 0:
        print(
            f"apply_moneyline_juice completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        sys.exit(1)

    print(
        f"apply_moneyline_juice complete. "
        f"files_written={summary['files_written']} "
        f"applied={summary['applied']} "
        f"missing_any_moneyline_dk={summary['missing_any_moneyline_dk']} "
        f"skipped_bad={summary['skipped_bad']} "
        f"skipped_noband={summary['skipped_noband']} "
        f"schema_errors={summary['schema_errors']} "
        f"errors={summary['errors']} "
        f"STATUS: SUCCESS"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/02_juice/apply_total_juice.py

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
JUICE_FILE = Path("docs/win/baseball/mlb/config/juice/mlb_totals_juice.csv")

ERROR_DIR = Path("docs/win/baseball/mlb/errors/02_juice")
LOG_FILE = ERROR_DIR / "apply_total_juice.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

MIN_JUICED_DECIMAL = 1.01
NORMALIZATION_TOLERANCE = 0.000001

REQUIRED_INPUT_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "total",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "fair_total_over_decimal",
    "fair_total_under_decimal",
]

DK_ODDS_COLUMNS = [
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

REQUIRED_JUICE_COLUMNS = [
    "band_min",
    "band_max",
    "side",
    "extra_juice",
]

OPTIONAL_ODDS_BAND_COLUMNS = [
    "odds_min",
    "odds_max",
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
        f"  files_found                 : {summary['files_found']}",
        f"  files_written               : {summary['files_written']}",
        f"  total_rows                  : {summary['total_rows']}",
        f"  over_applied                : {summary['over_applied']}",
        f"  under_applied               : {summary['under_applied']}",
        f"  clamped_decimal             : {summary['clamped_decimal']}",
        f"  skipped_bad                 : {summary['skipped_bad']}",
        f"  skipped_noband              : {summary['skipped_noband']}",
        f"  missing_total_dk            : {summary['missing_total_dk']}",
        f"  missing_over_price_dk       : {summary['missing_over_price_dk']}",
        f"  missing_under_price_dk      : {summary['missing_under_price_dk']}",
        f"  missing_any_total_dk        : {summary['missing_any_total_dk']}",
        f"  stale_input_errors          : {summary['stale_input_errors']}",
        f"  normalization_errors        : {summary['normalization_errors']}",
        f"  schema_errors               : {summary['schema_errors']}",
        f"  errors                      : {summary['errors']}",
        "",
        f"  {'file':<45} {'rows':>5} {'o_app':>7} {'u_app':>7} {'clamp':>6} {'bad':>5} {'noband':>7} {'miss_any':>9} {'schema':>7}",
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<45} {pf['rows']:>5} {pf['over_applied']:>7} "
            f"{pf['under_applied']:>7} {pf['clamped_decimal']:>6} "
            f"{pf['skipped_bad']:>5} {pf['skipped_noband']:>7} "
            f"{pf['missing_any_total_dk']:>9} {pf['schema_errors']:>7}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA GUARDS
# =========================

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

def config_uses_odds_bands(juice_df: pd.DataFrame) -> bool:
    return all(col in juice_df.columns for col in OPTIONAL_ODDS_BAND_COLUMNS)


def validate_juice_config(juice_df: pd.DataFrame) -> bool:
    uses_odds_bands = config_uses_odds_bands(juice_df)

    required_check_cols = ["band_min", "band_max", "extra_juice"]
    if uses_odds_bands:
        required_check_cols += OPTIONAL_ODDS_BAND_COLUMNS

    invalid_mask = (
        juice_df[required_check_cols].isna().any(axis=1) |
        (juice_df["band_min"] >= juice_df["band_max"]) |
        (~juice_df["side"].isin(["over", "under"]))
    )

    if uses_odds_bands:
        invalid_mask = invalid_mask | (juice_df["odds_min"] >= juice_df["odds_max"])

    invalid = juice_df[invalid_mask]
    if not invalid.empty:
        raise ValueError(f"total juice config contains invalid rows: {len(invalid)}")

    duplicate_subset = ["band_min", "band_max", "side"]
    if uses_odds_bands:
        duplicate_subset += OPTIONAL_ODDS_BAND_COLUMNS

    dupes = juice_df.duplicated(subset=duplicate_subset, keep=False)
    if dupes.any():
        raise ValueError(f"total juice config contains duplicate bands: {int(dupes.sum())}")

    required_sides = {"over", "under"}
    present_sides = set(juice_df["side"])
    missing_sides = sorted(required_sides - present_sides)
    if missing_sides:
        raise ValueError(f"total juice config missing side combinations: {missing_sides}")

    overlap_count = 0
    if uses_odds_bands:
        group_cols = ["side"]
        for side, group in juice_df.groupby(group_cols):
            rows = list(group.sort_values(["band_min", "band_max", "odds_min", "odds_max"]).to_dict("records"))
            for i, left in enumerate(rows):
                for right in rows[i + 1:]:
                    total_overlap = float(left["band_min"]) < float(right["band_max"]) and float(right["band_min"]) < float(left["band_max"])
                    odds_overlap = float(left["odds_min"]) < float(right["odds_max"]) and float(right["odds_min"]) < float(left["odds_max"])
                    if total_overlap and odds_overlap:
                        overlap_count += 1
    else:
        for side, group in juice_df.groupby("side"):
            group = group.sort_values(["band_min", "band_max"])
            prev_max = None
            for _, row in group.iterrows():
                if prev_max is not None and float(row["band_min"]) < prev_max:
                    overlap_count += 1
                prev_max = max(prev_max, float(row["band_max"])) if prev_max is not None else float(row["band_max"])

    if overlap_count:
        raise ValueError(f"total juice config contains overlapping bands: {overlap_count}")

    if uses_odds_bands:
        _log("total juice config uses odds-price bands via odds_min/odds_max")
    else:
        _log("total juice config does not use odds-price bands; applying total/side bands only", "WARN")

    return uses_odds_bands


# =========================
# JUICE LOOKUP
# =========================

def find_band_row(juice_df, total, side, dk_american, uses_odds_bands):
    band = juice_df[
        (juice_df["band_min"] <= total) &
        (total < juice_df["band_max"]) &
        (juice_df["side"] == side)
    ]

    if uses_odds_bands:
        band = band[
            (band["odds_min"] <= dk_american) &
            (dk_american < band["odds_max"])
        ]

    if len(band) != 1:
        return None

    return float(band.iloc[0]["extra_juice"])


# =========================
# AUDIT
# =========================

def append_audit_row(audit_rows: list, row: pd.Series, side: str, status: str, values: dict | None = None) -> None:
    values = values or {}
    audit_rows.append({
        "date": row.get("game_date", pd.NA),
        "game_id": row.get("game_id", pd.NA),
        "market": "total",
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
# SIDE PROCESSOR
# =========================

def process_side(df, juice_df, side, uses_odds_bands, audit_rows):
    fair_col = f"fair_total_{side}_decimal"
    american_col = f"dk_total_{side}_american"
    dk_decimal_col = f"dk_total_{side}_decimal"
    juiced_dec_col = f"{side}_juiced_decimal_total"
    juiced_prob_col = f"{side}_juiced_prob_total"

    df[juiced_dec_col] = pd.NA
    df[juiced_prob_col] = pd.NA

    applied = 0
    skipped_noband = 0
    skipped_bad = 0
    clamped_decimal = 0

    for idx, row in df.iterrows():
        if pd.isna(row[american_col]) or pd.isna(row[dk_decimal_col]):
            append_audit_row(audit_rows, row, side, "missing_dk_odds")
            _log(f"row={idx} side={side} reason=missing_dk_odds", "SKIP")
            skipped_bad += 1
            continue

        try:
            total = round(float(row["total"]), 1)
            fair_decimal = float(row[fair_col])
            dk_american = float(row[american_col])
            dk_decimal = float(row[dk_decimal_col])
        except Exception:
            append_audit_row(audit_rows, row, side, "bad_parse")
            _log(f"row={idx} side={side} reason=bad_parse", "SKIP")
            skipped_bad += 1
            continue

        if not math.isfinite(fair_decimal) or fair_decimal <= 1:
            append_audit_row(audit_rows, row, side, "invalid_fair_decimal", {
                "dk_american": dk_american,
                "dk_decimal": dk_decimal,
                "fair_decimal": fair_decimal,
            })
            _log(f"row={idx} side={side} reason=bad_fair_decimal val={fair_decimal}", "SKIP")
            skipped_bad += 1
            continue

        if not math.isfinite(dk_american) or not math.isfinite(dk_decimal) or dk_decimal <= 1:
            append_audit_row(audit_rows, row, side, "invalid_dk_odds", {
                "dk_american": dk_american,
                "dk_decimal": dk_decimal,
                "fair_decimal": fair_decimal,
            })
            _log(f"row={idx} side={side} reason=bad_dk_odds american={dk_american} decimal={dk_decimal}", "SKIP")
            skipped_bad += 1
            continue

        extra = find_band_row(juice_df, total, side, dk_american, uses_odds_bands)

        if extra is None:
            append_audit_row(audit_rows, row, side, "missing_band", {
                "dk_american": dk_american,
                "dk_decimal": dk_decimal,
                "fair_decimal": fair_decimal,
            })
            _log(f"row={idx} side={side} reason=no_band total={total} dk_american={dk_american}", "SKIP")
            skipped_noband += 1
            continue

        try:
            juiced_decimal = fair_decimal * (1 - extra)
            status = "juiced"

            if not math.isfinite(juiced_decimal):
                append_audit_row(audit_rows, row, side, "invalid_juiced_decimal", {
                    "dk_american": dk_american,
                    "dk_decimal": dk_decimal,
                    "fair_decimal": fair_decimal,
                })
                _log(f"row={idx} side={side} reason=invalid_juiced_decimal val={juiced_decimal} fair={fair_decimal} extra={extra}", "SKIP")
                skipped_bad += 1
                continue

            if juiced_decimal <= 1:
                _log(
                    f"row={idx} side={side} reason=clamped_juiced_decimal "
                    f"original={juiced_decimal} clamped={MIN_JUICED_DECIMAL} fair={fair_decimal} extra={extra}",
                    "WARN",
                )
                juiced_decimal = MIN_JUICED_DECIMAL
                clamped_decimal += 1
                status = "invalid_decimal_clamped"

            juiced_prob = 1 / juiced_decimal

            df.at[idx, juiced_dec_col] = juiced_decimal
            df.at[idx, juiced_prob_col] = juiced_prob
            append_audit_row(audit_rows, row, side, status, {
                "dk_american": dk_american,
                "dk_decimal": dk_decimal,
                "fair_decimal": fair_decimal,
                "juiced_decimal": juiced_decimal,
                "juiced_prob": juiced_prob,
            })
            applied += 1

        except Exception:
            append_audit_row(audit_rows, row, side, "calc_error")
            _log(f"row={idx} side={side} reason=calc_error", "SKIP")
            skipped_bad += 1

    return df, applied, skipped_noband, skipped_bad, clamped_decimal


# =========================
# NORMALIZATION
# =========================

def apply_normalization(df, audit_rows):
    df["over_normalized_prob_total"] = pd.NA
    df["under_normalized_prob_total"] = pd.NA

    for idx, row in df.iterrows():
        try:
            op = float(row["over_juiced_prob_total"])
            up = float(row["under_juiced_prob_total"])

            if not math.isfinite(op) or not math.isfinite(up):
                continue

            total = op + up

            if total <= 0:
                continue

            over_norm = op / total
            under_norm = up / total

            df.at[idx, "over_normalized_prob_total"] = over_norm
            df.at[idx, "under_normalized_prob_total"] = under_norm

            for audit_row in reversed(audit_rows):
                if audit_row["game_id"] == row.get("game_id") and audit_row["market"] == "total" and audit_row["side"] == "over" and pd.isna(audit_row["normalized_prob"]):
                    audit_row["normalized_prob"] = over_norm
                    break
            for audit_row in reversed(audit_rows):
                if audit_row["game_id"] == row.get("game_id") and audit_row["market"] == "total" and audit_row["side"] == "under" and pd.isna(audit_row["normalized_prob"]):
                    audit_row["normalized_prob"] = under_norm
                    break

        except Exception:
            continue

    return df


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_total_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0,
        "files_written": 0,
        "total_rows": 0,
        "over_applied": 0,
        "under_applied": 0,
        "clamped_decimal": 0,
        "skipped_bad": 0,
        "skipped_noband": 0,
        "missing_total_dk": 0,
        "missing_over_price_dk": 0,
        "missing_under_price_dk": 0,
        "missing_any_total_dk": 0,
        "stale_input_errors": 0,
        "normalization_errors": 0,
        "schema_errors": 0,
        "errors": 0,
    }

    per_file = []
    audit_rows = []

    for f in OUTPUT_DIR.glob("*total.csv"):
        f.unlink()
    for f in AUDIT_DIR.glob("*total*post_juice_audit.csv"):
        f.unlink()

    try:
        _log(f"INPUT_DIR : {INPUT_DIR}")
        _log(f"SOURCE_MERGE_DIR: {SOURCE_MERGE_DIR}")
        _log(f"JUICE_FILE: {JUICE_FILE}")

        juice_df = read_csv_validated(JUICE_FILE, REQUIRED_JUICE_COLUMNS, f"juice file {JUICE_FILE}")
        juice_df["band_min"] = pd.to_numeric(juice_df["band_min"], errors="coerce")
        juice_df["band_max"] = pd.to_numeric(juice_df["band_max"], errors="coerce")
        juice_df["side"] = juice_df["side"].astype(str).str.strip().str.lower()
        juice_df["extra_juice"] = pd.to_numeric(juice_df["extra_juice"], errors="coerce")

        if config_uses_odds_bands(juice_df):
            juice_df["odds_min"] = pd.to_numeric(juice_df["odds_min"], errors="coerce")
            juice_df["odds_max"] = pd.to_numeric(juice_df["odds_max"], errors="coerce")

        uses_odds_bands = validate_juice_config(juice_df)

        files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_total.csv")))
        summary["files_found"] = len(files)

        _log(f"Files found: {len(files)}")

        if not files:
            _log("No total files found — exiting", "WARN")
            _write_summary(summary, per_file)
            return

        for file_path in files:
            in_path = Path(file_path)
            out_path = OUTPUT_DIR / in_path.name

            pf = {
                "name": in_path.name,
                "rows": 0,
                "over_applied": 0,
                "under_applied": 0,
                "clamped_decimal": 0,
                "skipped_bad": 0,
                "skipped_noband": 0,
                "missing_total_dk": 0,
                "missing_over_price_dk": 0,
                "missing_under_price_dk": 0,
                "missing_any_total_dk": 0,
                "schema_errors": 0,
            }

            _log(f"--- FILE: {in_path.name}")

            try:
                validate_stale_input(in_path)

                df = read_csv_validated(in_path, REQUIRED_INPUT_COLUMNS, f"total input {in_path.name}")
                require_nonempty_columns(df, DK_ODDS_COLUMNS, f"total input {in_path.name}")

                if df.empty:
                    _log(f"{in_path.name} empty — skipping")
                    per_file.append(pf)
                    continue

                for c in [
                    "total",
                    "dk_total_over_american",
                    "dk_total_under_american",
                    "dk_total_over_decimal",
                    "dk_total_under_decimal",
                    "fair_total_over_decimal",
                    "fair_total_under_decimal",
                ]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                pf["missing_total_dk"] = int(df[["dk_total_over_american", "dk_total_under_american"]].isna().any(axis=1).sum())
                pf["missing_over_price_dk"] = int(df["dk_total_over_decimal"].isna().sum())
                pf["missing_under_price_dk"] = int(df["dk_total_under_decimal"].isna().sum())
                pf["missing_any_total_dk"] = int(df[DK_ODDS_COLUMNS].isna().any(axis=1).sum())

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)
                summary["missing_total_dk"] += pf["missing_total_dk"]
                summary["missing_over_price_dk"] += pf["missing_over_price_dk"]
                summary["missing_under_price_dk"] += pf["missing_under_price_dk"]
                summary["missing_any_total_dk"] += pf["missing_any_total_dk"]

                df, o_applied, o_noband, o_bad, o_clamped = process_side(df, juice_df, "over", uses_odds_bands, audit_rows)
                df, u_applied, u_noband, u_bad, u_clamped = process_side(df, juice_df, "under", uses_odds_bands, audit_rows)
                df = apply_normalization(df, audit_rows)

                pf["over_applied"] = o_applied
                pf["under_applied"] = u_applied
                pf["clamped_decimal"] = o_clamped + u_clamped
                pf["skipped_bad"] = o_bad + u_bad
                pf["skipped_noband"] = o_noband + u_noband

                norm_bad = validate_normalized_pair(
                    df,
                    "over_normalized_prob_total",
                    "under_normalized_prob_total",
                    f"{in_path.name} total",
                )
                if norm_bad:
                    summary["normalization_errors"] += norm_bad
                    raise ValueError(f"{in_path.name} has {norm_bad} invalid normalized total probability rows")

                write_csv_validated(df, out_path, f"total output {out_path.name}")

                summary["files_written"] += 1
                summary["over_applied"] += o_applied
                summary["under_applied"] += u_applied
                summary["clamped_decimal"] += pf["clamped_decimal"]
                summary["skipped_bad"] += pf["skipped_bad"]
                summary["skipped_noband"] += pf["skipped_noband"]

                _log(
                    f"{in_path.name} | rows={pf['rows']} over_applied={o_applied} "
                    f"under_applied={u_applied} clamped_decimal={pf['clamped_decimal']} "
                    f"skipped_bad={pf['skipped_bad']} skipped_noband={pf['skipped_noband']} "
                    f"missing_any_total_dk={pf['missing_any_total_dk']}"
                )

                _log(f"WROTE: {out_path}")

            except ValueError as e:
                if "stale 01_merguiced input" in str(e):
                    summary["stale_input_errors"] += 1
                pf["schema_errors"] += 1
                summary["schema_errors"] += 1
                summary["errors"] += 1
                _log(f"{in_path.name} SCHEMA FAILED: {e}\n{traceback.format_exc()}", "ERROR")

            except Exception as e:
                summary["errors"] += 1
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1
        _write_summary(summary, per_file)
        sys.exit(1)

    audit_path = AUDIT_DIR / "total_post_juice_audit.csv"
    pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    _log(f"WROTE AUDIT: {audit_path}")

    _write_summary(summary, per_file)

    if summary["errors"] > 0 or summary["schema_errors"] > 0:
        print(
            f"apply_total_juice completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        sys.exit(1)

    print(
        f"apply_total_juice complete. "
        f"files_written={summary['files_written']} "
        f"total_rows={summary['total_rows']} "
        f"missing_any_total_dk={summary['missing_any_total_dk']} "
        f"clamped_decimal={summary['clamped_decimal']} "
        f"skipped_bad={summary['skipped_bad']} "
        f"skipped_noband={summary['skipped_noband']} "
        f"schema_errors={summary['schema_errors']} "
        f"errors={summary['errors']} "
        f"STATUS: SUCCESS"
    )


if __name__ == "__main__":
    main()

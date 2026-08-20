#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/03_edges/compute_edges.py

import traceback
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INPUT_DIR   = Path("docs/win/baseball/mlb/02_juice")
OUTPUT_DIR  = Path("docs/win/baseball/mlb/03_edges")
ERROR_DIR   = Path("docs/win/baseball/mlb/errors/03_edges")
LOG_FILE    = ERROR_DIR / "compute_edges.txt"
CONFIG_PATH = Path("docs/win/baseball/mlb/config/edge_adjustments.yaml")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = Path("docs/win/baseball/mlb/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LEAKAGE_AUDIT_FILE = AUDIT_DIR / "leakage_audit.csv"
FORBIDDEN_READ_TOKENS = ["05_" + "final_scores", "final" + "_scores", "graded", "results", "reports"]  # LEAKAGE_GUARD_ALLOWED_REFERENCE
SCRIPT_NAME = "compute_edges.py"
STAGE_NAME = "03_edges"


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
        raise RuntimeError(f"Blocked forbidden pre-selection read path: {path} ({reason})")
    record_file_read(path, True, "allowed")


def read_csv_guarded(path: Path) -> pd.DataFrame:
    assert_read_path_allowed(path)
    return pd.read_csv(path)


def open_text_guarded(path: Path, mode="r", encoding="utf-8"):
    assert_read_path_allowed(path)
    return open(path, mode, encoding=encoding)



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
]

TOTAL_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "fair_total_over_decimal",
    "fair_total_under_decimal",
    "over_normalized_prob_total",
    "under_normalized_prob_total",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
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
        f"  files_processed  : {summary['files_processed']}",
        f"  rows_processed   : {summary['rows_processed']}",
        f"  moneyline_files  : {summary['moneyline_files']}",
        f"  run_line_files   : {summary['run_line_files']}",
        f"  total_files      : {summary['total_files']}",
        f"  skipped          : {summary['skipped']}",
        f"  null_edges       : {summary['null_edges']}",
        f"  schema_errors    : {summary['schema_errors']}",
        f"  errors           : {summary['errors']}",
        "",
        f"  {'file':<48} {'market':<12} {'rows':>5} {'null_edges':>10} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<48} {pf['market']:<12} {pf['rows']:>5} "
            f"{pf['null_edges']:>10} {pf['status']:>10}"
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
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_required_columns(df: pd.DataFrame, required_columns: list, label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_input_schema(df: pd.DataFrame, market: str, file_name: str) -> None:
    validate_no_duplicate_columns(df, f"{file_name} input")

    if market == "moneyline":
        validate_required_columns(df, MONEYLINE_REQUIRED_COLUMNS, f"{file_name} moneyline input")
    elif market == "run_line":
        validate_required_columns(df, RUN_LINE_REQUIRED_COLUMNS, f"{file_name} run_line input")
    elif market == "total":
        validate_required_columns(df, TOTAL_REQUIRED_COLUMNS, f"{file_name} total input")
    else:
        raise ValueError(f"{file_name} unknown market for schema validation: {market}")


def write_csv_checked(df: pd.DataFrame, output_path: Path) -> None:
    validate_no_duplicate_columns(df, f"{output_path} output")
    df.to_csv(output_path, index=False)


# =========================
# HELPERS
# =========================

def safe_edge(dk, p):
    dk  = pd.to_numeric(dk, errors="coerce")
    p   = pd.to_numeric(p,  errors="coerce")
    out = pd.Series(np.nan, index=dk.index, dtype="float64")
    valid = (dk > 1) & (p > 0) & (p < 1) & np.isfinite(dk) & np.isfinite(p)
    out.loc[valid] = p.loc[valid] * dk.loc[valid] - 1
    return out


def count_null_edges(df, cols):
    return sum(df[col].isna().sum() for col in cols if col in df.columns)


def _col(df, name):
    """Return column as float series, NaN if missing."""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


# =========================
# BASE EDGE COMPUTATION
# =========================

def compute_moneyline(df):
    df["home_edge_decimal_moneyline"] = safe_edge(
        df["home_dk_decimal_moneyline"], df["home_normalized_prob_moneyline"]
    )
    df["away_edge_decimal_moneyline"] = safe_edge(
        df["away_dk_decimal_moneyline"], df["away_normalized_prob_moneyline"]
    )
    null_edges = count_null_edges(df, [
        "home_edge_decimal_moneyline",
        "away_edge_decimal_moneyline",
    ])
    return df, null_edges


def compute_run_line(df):
    df["home_edge_decimal_run_line"] = safe_edge(
        df["home_dk_run_line_decimal"], df["home_normalized_prob_run_line"]
    )
    df["away_edge_decimal_run_line"] = safe_edge(
        df["away_dk_run_line_decimal"], df["away_normalized_prob_run_line"]
    )
    null_edges = count_null_edges(df, [
        "home_edge_decimal_run_line",
        "away_edge_decimal_run_line",
    ])
    return df, null_edges


def compute_total(df):
    df["over_prob"]  = 1 / pd.to_numeric(df["fair_total_over_decimal"],  errors="coerce")
    df["under_prob"] = 1 / pd.to_numeric(df["fair_total_under_decimal"], errors="coerce")

    df["over_edge_decimal_total"]  = safe_edge(df["dk_total_over_decimal"],  df["over_normalized_prob_total"])
    df["under_edge_decimal_total"] = safe_edge(df["dk_total_under_decimal"], df["under_normalized_prob_total"])

    null_edges = count_null_edges(df, [
        "over_edge_decimal_total",
        "under_edge_decimal_total",
    ])
    return df, null_edges


# =========================
# CONTEXTUAL ADJUSTMENTS
# =========================

def adjust_moneyline_run_line(df: pd.DataFrame, cfg: dict, market: str) -> pd.DataFrame:
    """
    Apply SP xwOBA, lineup xwOBA, and park factor adjustments
    to moneyline or run line edges.
    """
    park_scale    = cfg.get("park_edge_scale",    0.002)
    sp_scale      = cfg.get("sp_xwoba_scale",     0.5)
    lineup_scale  = cfg.get("lineup_xwoba_scale", 0.3)

    if market == "moneyline":
        home_col = "home_edge_decimal_moneyline"
        away_col = "away_edge_decimal_moneyline"
    else:
        home_col = "home_edge_decimal_run_line"
        away_col = "away_edge_decimal_run_line"

    home_edge = _col(df, home_col)
    away_edge = _col(df, away_col)

    # ── SP xwOBA adjustment ──────────────────────────────────
    # sp_xwoba_diff > 0 = home pitcher weaker → away up, home down
    home_sp_xwoba = _col(df, "home_sp_xwoba")
    away_sp_xwoba = _col(df, "away_sp_xwoba")
    sp_diff       = home_sp_xwoba - away_sp_xwoba
    sp_valid      = sp_diff.notna() & np.isfinite(sp_diff)

    sp_nudge = sp_diff * sp_scale
    home_edge = home_edge.where(~sp_valid, home_edge - sp_nudge)
    away_edge = away_edge.where(~sp_valid, away_edge + sp_nudge)

    # ── Lineup xwOBA adjustment ──────────────────────────────
    # lineup_diff > 0 = home lineup stronger → home up, away down
    home_lu = _col(df, "home_lineup_xwoba")
    away_lu = _col(df, "away_lineup_xwoba")
    lu_diff  = home_lu - away_lu
    lu_valid = lu_diff.notna() & np.isfinite(lu_diff)

    lu_nudge  = lu_diff * lineup_scale
    home_edge = home_edge.where(~lu_valid, home_edge + lu_nudge)
    away_edge = away_edge.where(~lu_valid, away_edge - lu_nudge)

    # ── Park factor adjustment ───────────────────────────────
    # Direction determined by sign of lineup_xwoba_diff
    # park_nudge = (park_factor - 100) * park_scale * sign(lineup_diff)
    park_factor = _col(df, "park_factor")
    park_valid  = park_factor.notna() & np.isfinite(park_factor) & lu_valid

    sign        = np.sign(lu_diff)
    park_nudge  = (park_factor - 100) * park_scale * sign
    home_edge   = home_edge.where(~park_valid, home_edge + park_nudge)
    away_edge   = away_edge.where(~park_valid, away_edge - park_nudge)

    df[home_col] = home_edge
    df[away_col] = away_edge
    return df


def adjust_totals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply park factor and weather adjustments to over/under total edges.

    Current weather columns used:
      weather_applicable
      temp_f
      wind_mph
      wind_blowing_out
      precip_in
      humidity
      air_pressure_at_sea_level
      dew_point_f
    """
    park_scale = cfg.get("park_edge_scale", 0.002)

    wind_thresh = cfg.get("wind_nudge_threshold", 12)
    wind_scale  = cfg.get("wind_nudge_scale", 0.005)

    cold_thresh = cfg.get("temp_cold_threshold", 50)
    hot_thresh  = cfg.get("temp_hot_threshold", 80)
    temp_scale  = cfg.get("temp_nudge_scale", 0.003)

    pressure_low_thresh  = cfg.get("pressure_low_threshold", 1010)
    pressure_high_thresh = cfg.get("pressure_high_threshold", 1020)
    pressure_scale       = cfg.get("pressure_nudge_scale", 0.003)

    dew_low_thresh  = cfg.get("dew_point_low_threshold", 45)
    dew_high_thresh = cfg.get("dew_point_high_threshold", 65)
    dew_scale       = cfg.get("dew_point_nudge_scale", 0.003)

    precip_thresh = cfg.get("precip_nudge_threshold", 0.02)
    precip_scale  = cfg.get("precip_nudge_scale", 0.004)

    humidity_low_thresh  = cfg.get("humidity_low_threshold", 35)
    humidity_high_thresh = cfg.get("humidity_high_threshold", 70)
    humidity_scale       = cfg.get("humidity_nudge_scale", 0.0015)

    over_edge  = _col(df, "over_edge_decimal_total")
    under_edge = _col(df, "under_edge_decimal_total")

    # ── Park factor (park_R) ─────────────────────────────────
    park_R      = _col(df, "park_R")
    park_valid  = park_R.notna() & np.isfinite(park_R)

    park_over_mult  = 1 + (park_R - 100) * park_scale
    park_under_mult = 1 - (park_R - 100) * park_scale

    over_edge  = over_edge.where(~park_valid,  over_edge  * park_over_mult)
    under_edge = under_edge.where(~park_valid, under_edge * park_under_mult)

    # ── Weather base columns ─────────────────────────────────
    applicable = _col(df, "weather_applicable")
    weather_on = applicable == 1

    wind_mph = _col(df, "wind_mph")
    wind_out = _col(df, "wind_blowing_out")
    temp_f   = _col(df, "temp_f")

    pressure  = _col(df, "air_pressure_at_sea_level")
    dew_point = _col(df, "dew_point_f")
    precip    = _col(df, "precip_in")
    humidity  = _col(df, "humidity")

    # ── Wind ─────────────────────────────────────────────────
    # Blowing out: over up, under down.
    # Blowing in: over down, under up.
    wind_active = weather_on & wind_mph.notna() & np.isfinite(wind_mph) & (wind_mph > wind_thresh)
    wind_nudge  = (wind_mph / 20) * wind_scale

    wind_out_cond = wind_active & (wind_out == 1)
    wind_in_cond  = wind_active & (wind_out == 0)

    over_edge  = over_edge.where(~wind_out_cond,  over_edge  + wind_nudge)
    under_edge = under_edge.where(~wind_out_cond, under_edge - wind_nudge)
    over_edge  = over_edge.where(~wind_in_cond,   over_edge  - wind_nudge)
    under_edge = under_edge.where(~wind_in_cond,  under_edge + wind_nudge)

    # ── Temperature ──────────────────────────────────────────
    # Cold: over down.
    # Hot: over up.
    temp_valid = weather_on & temp_f.notna() & np.isfinite(temp_f)

    cold_cond  = temp_valid & (temp_f < cold_thresh)
    hot_cond   = temp_valid & (temp_f > hot_thresh)

    cold_nudge = (cold_thresh - temp_f) / cold_thresh * temp_scale
    hot_nudge  = (temp_f - hot_thresh) / hot_thresh * temp_scale

    over_edge  = over_edge.where(~cold_cond, over_edge - cold_nudge)
    under_edge = under_edge.where(~cold_cond, under_edge + cold_nudge)
    over_edge  = over_edge.where(~hot_cond,  over_edge + hot_nudge)
    under_edge = under_edge.where(~hot_cond,  under_edge - hot_nudge)

    # ── Air pressure ─────────────────────────────────────────
    # Low pressure: over up, under down.
    # High pressure: over down, under up.
    pressure_valid = weather_on & pressure.notna() & np.isfinite(pressure)

    low_pressure_cond  = pressure_valid & (pressure < pressure_low_thresh)
    high_pressure_cond = pressure_valid & (pressure > pressure_high_thresh)

    low_pressure_nudge = (
        (pressure_low_thresh - pressure) / pressure_low_thresh * pressure_scale
    )
    high_pressure_nudge = (
        (pressure - pressure_high_thresh) / pressure_high_thresh * pressure_scale
    )

    over_edge  = over_edge.where(~low_pressure_cond,  over_edge  + low_pressure_nudge)
    under_edge = under_edge.where(~low_pressure_cond, under_edge - low_pressure_nudge)
    over_edge  = over_edge.where(~high_pressure_cond, over_edge  - high_pressure_nudge)
    under_edge = under_edge.where(~high_pressure_cond, under_edge + high_pressure_nudge)

    # ── Dew point ────────────────────────────────────────────
    # High dew point: over up.
    # Low dew point: over down.
    dew_valid = weather_on & dew_point.notna() & np.isfinite(dew_point)

    low_dew_cond  = dew_valid & (dew_point < dew_low_thresh)
    high_dew_cond = dew_valid & (dew_point > dew_high_thresh)

    low_dew_nudge = (
        (dew_low_thresh - dew_point) / dew_low_thresh * dew_scale
    )
    high_dew_nudge = (
        (dew_point - dew_high_thresh) / dew_high_thresh * dew_scale
    )

    over_edge  = over_edge.where(~low_dew_cond,  over_edge  - low_dew_nudge)
    under_edge = under_edge.where(~low_dew_cond, under_edge + low_dew_nudge)
    over_edge  = over_edge.where(~high_dew_cond, over_edge  + high_dew_nudge)
    under_edge = under_edge.where(~high_dew_cond, under_edge - high_dew_nudge)

    # ── Precipitation ────────────────────────────────────────
    # Measurable precipitation: over down, under up.
    precip_valid = weather_on & precip.notna() & np.isfinite(precip)
    precip_cond  = precip_valid & (precip > precip_thresh)

    if precip_thresh and precip_thresh > 0:
        precip_nudge = (precip / precip_thresh) * precip_scale
    else:
        precip_nudge = precip * precip_scale

    over_edge  = over_edge.where(~precip_cond,  over_edge  - precip_nudge)
    under_edge = under_edge.where(~precip_cond, under_edge + precip_nudge)

    # ── Humidity ─────────────────────────────────────────────
    # Intentionally weaker than dew point.
    # High humidity: over up slightly.
    # Low humidity: over down slightly.
    humidity_valid = weather_on & humidity.notna() & np.isfinite(humidity)

    low_humidity_cond  = humidity_valid & (humidity < humidity_low_thresh)
    high_humidity_cond = humidity_valid & (humidity > humidity_high_thresh)

    low_humidity_nudge = (
        (humidity_low_thresh - humidity) / humidity_low_thresh * humidity_scale
    )
    high_humidity_nudge = (
        (humidity - humidity_high_thresh) / humidity_high_thresh * humidity_scale
    )

    over_edge  = over_edge.where(~low_humidity_cond,  over_edge  - low_humidity_nudge)
    under_edge = under_edge.where(~low_humidity_cond, under_edge + low_humidity_nudge)
    over_edge  = over_edge.where(~high_humidity_cond, over_edge  + high_humidity_nudge)
    under_edge = under_edge.where(~high_humidity_cond, under_edge - high_humidity_nudge)

    df["over_edge_decimal_total"]  = over_edge
    df["under_edge_decimal_total"] = under_edge
    return df


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== compute_edges RUN {_now()} ===\n")

    # Load config
    try:
        with open_text_guarded(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)["mlb"]
        _log(f"Config loaded: {CONFIG_PATH}")
    except Exception as e:
        _log(f"FATAL: could not load config {CONFIG_PATH}: {e}", "ERROR")
        return

    summary = {
        "files_processed": 0,
        "rows_processed":  0,
        "moneyline_files": 0,
        "run_line_files":  0,
        "total_files":     0,
        "skipped":         0,
        "null_edges":      0,
        "schema_errors":   0,
        "errors":          0,
    }
    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")

    for out_file in OUTPUT_DIR.glob("*.csv"):
        out_file.unlink()

    for input_file in input_files:
        name   = input_file.name.lower()
        market = None
        pf = {
            "name":       input_file.name,
            "market":     "unknown",
            "rows":       0,
            "null_edges": 0,
            "status":     "ok",
        }

        if "moneyline" in name:
            market = "moneyline"
        elif "run_line" in name:
            market = "run_line"
        elif "total" in name:
            market = "total"
        else:
            _log(f"SKIP unrecognized file: {input_file.name}")
            pf["status"] = "skipped"
            summary["skipped"] += 1
            per_file.append(pf)
            continue

        pf["market"] = market
        _log(f"--- FILE: {input_file.name}  market={market}")

        try:
            df = read_csv_guarded(input_file)

            if df.empty:
                _log(f"{input_file.name} empty — skipping")
                pf["status"] = "empty"
                summary["skipped"] += 1
                per_file.append(pf)
                continue

            try:
                validate_input_schema(df, market, input_file.name)
            except Exception as schema_error:
                _log(f"{input_file.name} SCHEMA FAILED: {schema_error}", "ERROR")
                pf["status"] = "schema_error"
                summary["schema_errors"] += 1
                per_file.append(pf)
                continue

            pf["rows"] = len(df)
            summary["rows_processed"] += len(df)

            # Base edge computation
            if market == "moneyline":
                df, null_edges = compute_moneyline(df)
                summary["moneyline_files"] += 1
            elif market == "run_line":
                df, null_edges = compute_run_line(df)
                summary["run_line_files"] += 1
            else:
                df, null_edges = compute_total(df)
                summary["total_files"] += 1

            # Contextual adjustments
            if market in ("moneyline", "run_line"):
                if "sp_data_available" in df.columns and "lineup_data_available" in df.columns:
                    missing_sp     = int((df["sp_data_available"]     == 0).sum())
                    missing_lineup = int((df["lineup_data_available"] == 0).sum())
                    if missing_sp > 0:
                        _log(f"{input_file.name} | {missing_sp} rows missing SP data — SP adjustment will use 0 diff", "WARN")
                    if missing_lineup > 0:
                        _log(f"{input_file.name} | {missing_lineup} rows missing lineup data — lineup/park adjustments will use 0 diff", "WARN")
                else:
                    _log(f"{input_file.name} | data availability columns not present — adjustments applied without indicators", "WARN")
                df = adjust_moneyline_run_line(df, cfg, market)
            else:
                df = adjust_totals(df, cfg)

            pf["null_edges"] = null_edges
            summary["null_edges"] += null_edges

            if null_edges > 0:
                _log(f"{input_file.name} | {null_edges} null edges", "WARN")

            output_path = OUTPUT_DIR / input_file.name
            write_csv_checked(df, output_path)

            summary["files_processed"] += 1
            _log(f"WROTE: {output_path} ({len(df)} rows, {null_edges} null edges)")

        except Exception as e:
            _log(f"{input_file.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
            pf["status"] = "error"
            summary["errors"] += 1

        per_file.append(pf)

    _write_summary(summary, per_file)

    if summary["errors"] > 0 or summary["schema_errors"] > 0:
        print(
            f"compute_edges completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        raise SystemExit(1)

    print("compute_edges complete.")


if __name__ == "__main__":
    main()

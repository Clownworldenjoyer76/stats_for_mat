#!/usr/bin/env python3
# docs/win/basketball/scripts/pipeline_health.py
"""Current-run basketball pipeline health contract.

Writes docs/win/basketball/pipeline_health.json and exits non-zero on current,
in-season integrity failures. Historical/offseason mismatch noise is reported but is
not allowed to make an otherwise valid live run fail.

Operational season dates are loaded from:
    docs/win/basketball/config/season_dates.yaml
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


BASE = Path("docs/win/basketball")
ERRORS = BASE / "errors"
OUTPUT = BASE / "pipeline_health.json"
LOG = ERRORS / "pipeline_health.txt"

SEASON_CONFIG = BASE / "config/season_dates.yaml"

WNBA_DRIFT_REPORT = ERRORS / "99_validation/wnba_bias_drift.csv"

NY = ZoneInfo("America/New_York")

LEAGUES = ["nba", "ncaam", "wnba"]

LABEL = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

DRIFT_WINDOWS = [25, 50, 100]

DRIFT_WARN = float(
    os.getenv("WNBA_BIAS_DRIFT_WARN", "2.0")
)

STAGE_LOGS = [
    ERRORS / "00_intake/basketball_odds.txt",
    ERRORS / "00_intake/basketball_drat_scraper.txt",
    ERRORS / "00_intake/transform_basketball_nba.txt",
    ERRORS / "00_intake/transform_basketball_ncaam.txt",
    ERRORS / "00_intake/transform_basketball_wnba.txt",
    ERRORS / "00_intake/basketball_daily_games.txt",
    ERRORS / "00_intake/basketball_game_id.txt",
    ERRORS / "00_intake/clean_basketball_inputs.txt",
    ERRORS / "01_merge/merge_intake.txt",
    ERRORS / "01_merge/build_juice_files.txt",
    ERRORS / "03_edges/compute_ev_kelly.txt",
    ERRORS / "04_select/select_bets.txt",
    ERRORS / "04_select/daily_slate.txt",
    ERRORS / "05_final_scores/01_basketball_results_grade.txt",
]

BAD_STATUS = (
    "STATUS: FAILED",
    "STATUS: PARTIAL",
    "STATUS: COMPLETED WITH ERRORS",
)


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def fnum(v):
    try:
        if v is None or clean(v) == "":
            return None

        x = float(v)

        return x if math.isfinite(x) else None

    except Exception:
        return None


def comp(row: dict) -> tuple[str, str, str]:
    return (
        clean(row.get("game_date")),
        clean(row.get("home_team")).casefold(),
        clean(row.get("away_team")).casefold(),
    )


def validate_month_day(
    league: str,
    label: str,
    month: int,
    day: int,
) -> None:
    """Validate a month/day pair using a leap year."""
    try:
        datetime(2000, month, day)

    except ValueError as exc:
        raise ValueError(
            f"Invalid {league}.{label}: month={month}, day={day}"
        ) from exc


def load_season_config() -> dict[str, dict[str, int]]:
    """Load and validate operational season dates."""
    if not SEASON_CONFIG.exists():
        raise FileNotFoundError(
            f"Season config not found: {SEASON_CONFIG}"
        )

    with open(
        SEASON_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"{SEASON_CONFIG} must contain a top-level mapping"
        )

    required_fields = (
        "start_month",
        "start_day",
        "end_month",
        "end_day",
    )

    config: dict[str, dict[str, int]] = {}

    for league in LEAGUES:
        row = raw.get(league)

        if not isinstance(row, dict):
            raise ValueError(
                f"Missing season configuration for league={league}"
            )

        values: dict[str, int] = {}

        for field in required_fields:
            if field not in row:
                raise ValueError(
                    f"Missing {league}.{field} in {SEASON_CONFIG}"
                )

            try:
                values[field] = int(row[field])

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {league}.{field}: {row[field]!r}"
                ) from exc

        validate_month_day(
            league,
            "start",
            values["start_month"],
            values["start_day"],
        )

        validate_month_day(
            league,
            "end",
            values["end_month"],
            values["end_day"],
        )

        config[league] = values

    return config


def in_season(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> bool:
    """Return True when the current date is inside the league's season."""
    league = league.strip().lower()

    if league not in season_config:
        raise KeyError(
            f"No season configuration found for league={league}"
        )

    cfg = season_config[league]

    current_mmdd = (
        now.month,
        now.day,
    )

    start_mmdd = (
        cfg["start_month"],
        cfg["start_day"],
    )

    end_mmdd = (
        cfg["end_month"],
        cfg["end_day"],
    )

    # Normal season contained within one calendar year.
    # Example: May 1 through October 31.
    if start_mmdd <= end_mmdd:
        return start_mmdd <= current_mmdd <= end_mmdd

    # Season crosses New Year.
    # Example: October 15 through July 1.
    return (
        current_mmdd >= start_mmdd
        or current_mmdd <= end_mmdd
    )


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def unique_by_comp(rows: list[dict]) -> dict[tuple, dict]:
    out = {}

    for row in rows:
        key = comp(row)

        if all(key):
            out[key] = row

    return out


def duplicate_integrity(
    rows: list[dict],
) -> tuple[list[str], list[str]]:
    duplicate_composites = []
    duplicate_ids = []

    seen_comp = {}
    seen_id = {}

    for row in rows:
        key = comp(row)
        gid = clean(row.get("game_id"))

        if all(key):
            if key in seen_comp:
                duplicate_composites.append(
                    "|".join(key)
                )
            else:
                seen_comp[key] = row

        if gid:
            prior = seen_id.get(gid)

            if (
                prior is not None
                and comp(prior) != key
            ):
                duplicate_ids.append(gid)
            else:
                seen_id[gid] = row

    return (
        sorted(set(duplicate_composites)),
        sorted(set(duplicate_ids)),
    )


def last_status(
    path: Path,
) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    statuses = [
        line.strip()
        for line in text.splitlines()
        if "STATUS:" in line
    ]

    return (
        statuses[-1]
        if statuses
        else None
    )


def stage_health() -> tuple[list[dict], list[str]]:
    rows = []
    fatals = []

    for path in STAGE_LOGS:
        status = last_status(path)

        item = {
            "path": str(path),
            "exists": path.exists(),
            "status": status,
        }

        rows.append(item)

        if (
            status
            and any(
                bad in status
                for bad in BAD_STATUS
            )
        ):
            fatals.append(
                f"stage failure: {path} -> {status}"
            )

    return rows, fatals


def current_league_health(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> tuple[dict, list[str]]:
    upper = LABEL[league]
    date = now.strftime("%Y_%m_%d")

    active = in_season(
        league,
        now,
        season_config,
    )

    daily_path = (
        BASE
        / f"daily_games/{league}/{date}_{upper}.csv"
    )

    pred_path = (
        BASE
        / (
            f"00_intake/predictions/predictions_cleaned/"
            f"{league}/{date}_{upper}_predictions.csv"
        )
    )

    book_path = (
        BASE
        / (
            f"00_intake/sportsbook/sportsbook_cleaned/"
            f"{league}/{date}_{upper}_odds.csv"
        )
    )

    merge_path = (
        BASE
        / (
            f"01_merge/{league}/moneyline/"
            f"{date}_{upper}_moneyline.csv"
        )
    )

    picks_path = (
        BASE
        / (
            f"04_select/{league}/daily_picks/"
            f"{date}_{league}_selected.csv"
        )
    )

    locked_path = (
        BASE
        / (
            f"04_select/{league}/locked_picks/"
            f"{date}_{league}_selected.csv"
        )
    )

    daily = read_rows(daily_path)
    preds = read_rows(pred_path)
    books = read_rows(book_path)
    merged = read_rows(merge_path)
    picks = read_rows(picks_path)
    locked = read_rows(locked_path)

    daily_map = unique_by_comp(daily)
    pred_map = unique_by_comp(preds)
    book_map = unique_by_comp(books)
    merge_map = unique_by_comp(merged)

    dup_comp, dup_ids = duplicate_integrity(
        daily
    )

    pred_dup_comp, pred_dup_ids = duplicate_integrity(
        preds
    )

    missing_predictions = sorted(
        "|".join(k)
        for k in set(daily_map) - set(pred_map)
    )

    missing_sportsbook = sorted(
        "|".join(k)
        for k in set(daily_map) - set(book_map)
    )

    predicted_not_merged = sorted(
        "|".join(k)
        for k in set(pred_map) - set(merge_map)
    )

    blank_pred_ids = sum(
        1
        for r in preds
        if not clean(r.get("game_id"))
    )

    blank_daily_ids = sum(
        1
        for r in daily
        if not clean(r.get("game_id"))
    )

    item = {
        "in_season": active,
        "season_config": {
            "start_month": season_config[league]["start_month"],
            "start_day": season_config[league]["start_day"],
            "end_month": season_config[league]["end_month"],
            "end_day": season_config[league]["end_day"],
        },
        "paths": {
            "daily_games": str(daily_path),
            "predictions": str(pred_path),
            "sportsbook": str(book_path),
            "merged": str(merge_path),
            "selected": str(picks_path),
            "locked": str(locked_path),
        },
        "counts": {
            "scheduled_games": len(daily_map),
            "prediction_games": len(pred_map),
            "sportsbook_games": len(book_map),
            "merged_games": len(merge_map),
            "selected_bets": len(picks),
            "locked_bets": len(locked),
        },
        "identity": {
            "daily_duplicate_composites": dup_comp,
            "daily_conflicting_game_ids": dup_ids,
            "prediction_duplicate_composites": pred_dup_comp,
            "prediction_conflicting_game_ids": pred_dup_ids,
            "blank_daily_game_ids": blank_daily_ids,
            "blank_prediction_game_ids": blank_pred_ids,
        },
        "coverage": {
            "scheduled_missing_predictions": missing_predictions,
            "scheduled_missing_sportsbook": missing_sportsbook,
            "predictions_not_merged": predicted_not_merged,
        },
    }

    fatals: list[str] = []

    # Only current in-season slates are strict.
    # A true no-games day where all counts are zero is valid.
    if active:
        if (
            dup_comp
            or dup_ids
            or pred_dup_comp
            or pred_dup_ids
        ):
            fatals.append(
                f"{upper}: duplicate/conflicting current game identity"
            )

        if blank_daily_ids:
            fatals.append(
                f"{upper}: {blank_daily_ids} current daily games "
                "missing game_id"
            )

        if preds and blank_pred_ids:
            fatals.append(
                f"{upper}: {blank_pred_ids} current predictions "
                "missing game_id"
            )

        if (
            daily_map
            and missing_predictions
        ):
            fatals.append(
                f"{upper}: {len(missing_predictions)} scheduled games "
                "missing predictions"
            )

        if (
            daily_map
            and missing_sportsbook
        ):
            fatals.append(
                f"{upper}: {len(missing_sportsbook)} scheduled games "
                "missing sportsbook rows"
            )

        if (
            pred_map
            and predicted_not_merged
        ):
            fatals.append(
                f"{upper}: {len(predicted_not_merged)} prediction games "
                "did not merge"
            )

    return item, fatals


def load_all_csv(
    folder: Path,
    pattern: str = "*.csv",
) -> list[dict]:
    rows = []

    if not folder.exists():
        return rows

    for path in sorted(
        folder.glob(pattern)
    ):
        try:
            rows.extend(
                read_rows(path)
            )
        except Exception:
            continue

    return rows


def wnba_bias_drift() -> dict:
    preds = load_all_csv(
        BASE
        / "00_intake/predictions/predictions_cleaned/wnba",
        "*_WNBA_predictions.csv",
    )

    finals = load_all_csv(
        BASE
        / "05_final_scores/results/wnba",
        "*_final_scores_WNBA.csv",
    )

    finals_by_id = {}
    finals_by_comp = {}

    for row in finals:
        if (
            fnum(row.get("home_score")) is None
            or fnum(row.get("away_score")) is None
        ):
            continue

        gid = clean(
            row.get("game_id")
        )

        key = comp(row)

        if gid:
            finals_by_id[gid] = row

        if all(key):
            finals_by_comp[key] = row

    residuals = []

    for p in preds:
        gid = clean(
            p.get("game_id")
        )

        final = (
            finals_by_id.get(gid)
            if gid
            else None
        )

        if final is None:
            final = finals_by_comp.get(
                comp(p)
            )

        if final is None:
            continue

        hp = fnum(
            p.get("home_projected_points")
        )

        ap = fnum(
            p.get("away_projected_points")
        )

        tp = fnum(
            p.get("total_projected_points")
        )

        hs = fnum(
            final.get("home_score")
        )

        aws = fnum(
            final.get("away_score")
        )

        if None in (
            hp,
            ap,
            tp,
            hs,
            aws,
        ):
            continue

        residuals.append({
            "game_date": clean(
                p.get("game_date")
            ),
            "game_id": (
                gid
                or clean(final.get("game_id"))
            ),
            "margin_residual_projected_minus_actual": (
                (hp - ap)
                - (hs - aws)
            ),
            "total_residual_projected_minus_actual": (
                tp
                - (hs + aws)
            ),
        })

    residuals.sort(
        key=lambda r: (
            r["game_date"],
            r["game_id"],
        )
    )

    windows = {}
    warnings = []

    for n in DRIFT_WINDOWS:
        sample = residuals[-n:]

        if not sample:
            windows[str(n)] = {
                "games": 0,
                "margin_mean_residual": None,
                "total_mean_residual": None,
                "warning": False,
            }
            continue

        m = (
            sum(
                r["margin_residual_projected_minus_actual"]
                for r in sample
            )
            / len(sample)
        )

        t = (
            sum(
                r["total_residual_projected_minus_actual"]
                for r in sample
            )
            / len(sample)
        )

        warn = (
            len(sample) >= min(n, 25)
            and (
                abs(m) >= DRIFT_WARN
                or abs(t) >= DRIFT_WARN
            )
        )

        windows[str(n)] = {
            "games": len(sample),
            "margin_mean_residual": round(m, 4),
            "total_mean_residual": round(t, 4),
            "warning": warn,
        }

        if warn:
            warnings.append(
                f"WNBA {len(sample)}-game residual drift exceeds "
                f"{DRIFT_WARN}: margin={m:.3f}, total={t:.3f}"
            )

    return {
        "definition": "adjusted_projected_minus_actual",
        "warning_threshold_abs_points": DRIFT_WARN,
        "matched_games": len(residuals),
        "windows": windows,
        "warnings": warnings,
    }


def write_wnba_drift_report(
    drift: dict,
    generated_at_utc: str,
) -> None:
    """Write a stable, machine-readable WNBA calibration drift report each run."""
    WNBA_DRIFT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "generated_at_utc",
        "matched_games",
        "window_games",
        "sample_games",
        "margin_mean_residual",
        "total_mean_residual",
        "threshold_points",
        "warning",
    ]

    tmp = WNBA_DRIFT_REPORT.with_suffix(
        WNBA_DRIFT_REPORT.suffix + ".tmp"
    )

    with open(
        tmp,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for n in DRIFT_WINDOWS:
            vals = (
                drift
                .get("windows", {})
                .get(str(n), {})
            )

            writer.writerow({
                "generated_at_utc": generated_at_utc,
                "matched_games": drift.get(
                    "matched_games",
                    0,
                ),
                "window_games": n,
                "sample_games": vals.get(
                    "games",
                    0,
                ),
                "margin_mean_residual": vals.get(
                    "margin_mean_residual"
                ),
                "total_mean_residual": vals.get(
                    "total_mean_residual"
                ),
                "threshold_points": drift.get(
                    "warning_threshold_abs_points"
                ),
                "warning": bool(
                    vals.get(
                        "warning",
                        False,
                    )
                ),
            })

    tmp.replace(
        WNBA_DRIFT_REPORT
    )


def main() -> None:
    now = datetime.now(NY)

    try:
        season_config = load_season_config()

    except Exception as exc:
        generated_at_utc = datetime.now(UTC).isoformat()

        fatal = (
            f"season config failure: {exc}"
        )

        payload = {
            "schema_version": 1,
            "generated_at_utc": generated_at_utc,
            "game_date_new_york": now.strftime("%Y_%m_%d"),
            "season_config_path": str(SEASON_CONFIG),
            "status": "failed",
            "fatal_errors": [fatal],
            "stage_status": [],
            "leagues": {},
            "wnba_bias_drift": {},
        }

        OUTPUT.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        LOG.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        lines = [
            f"=== basketball pipeline health {generated_at_utc} ===",
            f"game_date_new_york: {payload['game_date_new_york']}",
            f"season_config: {SEASON_CONFIG}",
            "status: failed",
            f"FATAL: {fatal}",
        ]

        LOG.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print(
            "\n".join(lines)
        )

        sys.exit(1)

    stages, fatals = stage_health()

    leagues = {}

    for league in LEAGUES:
        item, errs = current_league_health(
            league,
            now,
            season_config,
        )

        leagues[league] = item

        fatals.extend(
            errs
        )

    drift = wnba_bias_drift()

    generated_at_utc = datetime.now(
        UTC
    ).isoformat()

    write_wnba_drift_report(
        drift,
        generated_at_utc,
    )

    drift["report_path"] = str(
        WNBA_DRIFT_REPORT
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "game_date_new_york": now.strftime("%Y_%m_%d"),
        "season_config_path": str(SEASON_CONFIG),
        "status": (
            "failed"
            if fatals
            else "healthy"
        ),
        "fatal_errors": fatals,
        "stage_status": stages,
        "leagues": leagues,
        "wnba_bias_drift": drift,
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        f"=== basketball pipeline health {payload['generated_at_utc']} ===",
        f"game_date_new_york: {payload['game_date_new_york']}",
        f"season_config: {SEASON_CONFIG}",
        f"status: {payload['status']}",
    ]

    for league, item in leagues.items():
        c = item["counts"]

        season = item["season_config"]

        lines.append(
            f"{league.upper()}: "
            f"in_season={item['in_season']} "
            f"season="
            f"{season['start_month']:02d}/{season['start_day']:02d}-"
            f"{season['end_month']:02d}/{season['end_day']:02d} "
            f"scheduled={c['scheduled_games']} "
            f"predictions={c['prediction_games']} "
            f"sportsbook={c['sportsbook_games']} "
            f"merged={c['merged_games']} "
            f"selected={c['selected_bets']} "
            f"locked={c['locked_bets']}"
        )

    lines.append(
        f"WNBA drift report: {WNBA_DRIFT_REPORT}"
    )

    for n, vals in drift["windows"].items():
        lines.append(
            f"WNBA drift {n}: "
            f"games={vals['games']} "
            f"margin={vals['margin_mean_residual']} "
            f"total={vals['total_mean_residual']} "
            f"warning={vals['warning']}"
        )

    for warning in drift["warnings"]:
        lines.append(
            f"WARN: {warning}"
        )

    for error in fatals:
        lines.append(
            f"FATAL: {error}"
        )

    LOG.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    if fatals:
        sys.exit(1)


if __name__ == "__main__":
    main()

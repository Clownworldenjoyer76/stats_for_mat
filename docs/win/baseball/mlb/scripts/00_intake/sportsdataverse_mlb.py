#!/usr/bin/env python3
"""Build SportsDataverse pregame features for existing MLB game files.

Reads the authoritative ``00_intake/games/{date}_games.csv`` spine, pulls
SportsDataverse/Statcast history for the probable starters, and writes one
clean row per game to:

    00_intake/sportsdataverse/{date}_sportsdataverse.csv

All features are as-of the day before the game date. Same-day pitches are
never included.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

from sportsdataverse.mlb import (
    mlb_command_plus,
    mlb_statcast_search,
    mlb_stuff_plus,
    x_era,
)
from sportsdataverse.mlb.mlb_pitch_features import pitch_features


BASE_DIR = Path("docs/win/baseball/mlb")
GAMES_DIR = BASE_DIR / "00_intake/games"
OUT_DIR = BASE_DIR / "00_intake/sportsdataverse"
ERROR_DIR = BASE_DIR / "errors/00_intake"

OUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "sportsdataverse_mlb.txt"

DEFAULT_LOOKBACK_DAYS = 30

PITCHER_FEATURE_COLUMNS = [
    "pitcher_id",
    "sp_pitches",
    "sp_games",
    "sp_pitch_types",
    "sp_avg_velo",
    "sp_avg_spin",
    "sp_stuff_plus",
    "sp_stuff_scored_pitches",
    "sp_command_plus",
    "sp_command_scored_pitches",
    "sp_xwoba",
    "sp_xera",
    "sp_pitches_30d",
    "sp_games_30d",
    "sp_avg_velo_30d",
    "sp_avg_spin_30d",
    "sp_xwoba_30d",
    "sp_xera_30d",
    "sp_velo_delta_30d",
    "sp_last_game_date",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _log(message: str, level: str = "INFO") -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {message.rstrip()}\n")


def normalize_date(value: str) -> str:
    return str(value or "").strip().replace("-", "_")


def parse_date(value: str) -> date:
    text = str(value or "").strip().replace("_", "-")
    return datetime.strptime(text, "%Y-%m-%d").date()


def _empty_pitcher_features(pitcher_ids: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame({"pitcher_id": pitcher_ids})
    for col in PITCHER_FEATURE_COLUMNS[1:]:
        frame[col] = pd.NA
    return frame[PITCHER_FEATURE_COLUMNS]


def _safe_ints(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none", "<na>"}:
                continue
            parsed = int(float(text))
        except (TypeError, ValueError):
            continue
        if parsed not in seen:
            seen.add(parsed)
            out.append(parsed)
    return out


def _game_date_for_file(date_str: str, games: pd.DataFrame) -> date:
    if "game_date" in games.columns:
        for value in games["game_date"].tolist():
            try:
                return parse_date(value)
            except (TypeError, ValueError):
                continue
    return parse_date(date_str)


def _filter_since(pitches: pl.DataFrame, start_date: date) -> pl.DataFrame:
    if pitches is None or pitches.height == 0 or "game_date" not in pitches.columns:
        return pitches

    parsed_col = (
        pl.col("game_date")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("_sdv_game_date")
    )
    return (
        pitches.with_columns(parsed_col)
        .filter(pl.col("_sdv_game_date") >= pl.lit(start_date))
        .drop("_sdv_game_date")
    )


def _base_pitcher_stats(pitches: pl.DataFrame, suffix: str = "") -> pl.DataFrame:
    if pitches is None or pitches.height == 0 or "pitcher" not in pitches.columns:
        return pl.DataFrame()

    aggs: list[pl.Expr] = [pl.len().alias(f"sp_pitches{suffix}")]

    if "game_pk" in pitches.columns:
        aggs.append(pl.col("game_pk").n_unique().alias(f"sp_games{suffix}"))
    if "pitch_type" in pitches.columns and not suffix:
        aggs.append(pl.col("pitch_type").drop_nulls().n_unique().alias("sp_pitch_types"))
    if "release_speed" in pitches.columns:
        aggs.append(pl.col("release_speed").mean().alias(f"sp_avg_velo{suffix}"))
    if "release_spin_rate" in pitches.columns:
        aggs.append(pl.col("release_spin_rate").mean().alias(f"sp_avg_spin{suffix}"))
    if "game_date" in pitches.columns and not suffix:
        aggs.append(
            pl.col("game_date")
            .cast(pl.Utf8)
            .max()
            .alias("sp_last_game_date")
        )

    return pitches.group_by("pitcher").agg(aggs)


def _safe_model(label: str, callback) -> pl.DataFrame:
    try:
        result = callback()
        if result is None:
            return pl.DataFrame()
        return result
    except Exception as exc:
        _log(f"{label} failed: {exc}", "WARN")
        return pl.DataFrame()


def _merge_polars_feature(
    base: pd.DataFrame,
    frame: pl.DataFrame,
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    if frame is None or frame.height == 0 or "pitcher" not in frame.columns:
        return base

    pdf = frame.to_pandas()
    pdf["pitcher_id"] = pd.to_numeric(pdf["pitcher"], errors="coerce").astype("Int64")
    pdf = pdf.drop(columns=["pitcher"])

    if rename:
        pdf = pdf.rename(columns=rename)

    return base.merge(pdf, on="pitcher_id", how="left")


def build_pitcher_features(
    raw_pitches: pl.DataFrame,
    pitcher_ids: list[int],
    season: int,
    cutoff_date: date,
    lookback_days: int,
) -> pd.DataFrame:
    features = pd.DataFrame({"pitcher_id": pd.Series(pitcher_ids, dtype="Int64")})

    if raw_pitches is None or raw_pitches.height == 0:
        return _empty_pitcher_features(pitcher_ids)

    if "pitcher" in raw_pitches.columns:
        raw_pitches = raw_pitches.with_columns(
            pl.col("pitcher").cast(pl.Int64, strict=False)
        )

    season_base = _base_pitcher_stats(raw_pitches)
    features = _merge_polars_feature(features, season_base)

    feats = _safe_model("pitch_features", lambda: pitch_features(raw_pitches))

    stuff_pitch = _safe_model(
        "mlb_stuff_plus",
        lambda: mlb_stuff_plus(feats, level="pitch"),
    )
    if stuff_pitch.height:
        stuff = stuff_pitch.group_by("pitcher").agg(
            pl.col("stuff_plus").mean().alias("sp_stuff_plus"),
            pl.len().alias("sp_stuff_scored_pitches"),
        )
        features = _merge_polars_feature(features, stuff)

    command_pitch = _safe_model(
        "mlb_command_plus",
        lambda: mlb_command_plus(feats, level="pitch"),
    )
    if command_pitch.height:
        command = command_pitch.group_by("pitcher").agg(
            pl.col("command_plus").mean().alias("sp_command_plus"),
            pl.len().alias("sp_command_scored_pitches"),
        )
        features = _merge_polars_feature(features, command)

    xera = _safe_model("x_era", lambda: x_era(raw_pitches, season))
    if xera.height:
        features = _merge_polars_feature(
            features,
            xera.select("pitcher", "x_woba", "x_era"),
            rename={"x_woba": "sp_xwoba", "x_era": "sp_xera"},
        )

    recent_start = cutoff_date - timedelta(days=lookback_days)
    recent = _filter_since(raw_pitches, recent_start)

    recent_base = _base_pitcher_stats(recent, suffix="_30d")
    features = _merge_polars_feature(features, recent_base)

    recent_xera = _safe_model("x_era_30d", lambda: x_era(recent, season))
    if recent_xera.height:
        features = _merge_polars_feature(
            features,
            recent_xera.select("pitcher", "x_woba", "x_era"),
            rename={"x_woba": "sp_xwoba_30d", "x_era": "sp_xera_30d"},
        )

    if "sp_avg_velo" in features.columns and "sp_avg_velo_30d" in features.columns:
        features["sp_velo_delta_30d"] = (
            pd.to_numeric(features["sp_avg_velo_30d"], errors="coerce")
            - pd.to_numeric(features["sp_avg_velo"], errors="coerce")
        )

    for col in PITCHER_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = pd.NA

    return features[PITCHER_FEATURE_COLUMNS]


def attach_side_features(
    games: pd.DataFrame,
    pitcher_features: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    key = f"{side}_pitcher_id"
    if key not in games.columns:
        games[key] = ""

    side_features = pitcher_features.copy()
    side_features["_pitcher_key"] = side_features["pitcher_id"].astype("Int64").astype("string")
    side_features = side_features.drop(columns=["pitcher_id"])

    rename = {
        col: f"sdv_{side}_{col}"
        for col in side_features.columns
        if col != "_pitcher_key"
    }
    side_features = side_features.rename(columns=rename)

    games[key] = games[key].astype("string").str.strip()
    games = games.merge(
        side_features,
        left_on=key,
        right_on="_pitcher_key",
        how="left",
    ).drop(columns=["_pitcher_key"])

    pitches_col = f"sdv_{side}_sp_pitches"
    games[f"sdv_{side}_sp_found"] = games[pitches_col].notna().astype(int)
    return games


def write_base_output(
    games: pd.DataFrame,
    out_path: Path,
    game_date: date,
    lookback_days: int,
    status: str,
) -> None:
    out = games.copy()
    out["sdv_as_of_date"] = (game_date - timedelta(days=1)).isoformat()
    out["sdv_season"] = game_date.year
    out["sdv_lookback_days"] = lookback_days
    out["sdv_status"] = status

    for side in ("home", "away"):
        for feature in PITCHER_FEATURE_COLUMNS[1:]:
            out[f"sdv_{side}_{feature}"] = pd.NA
        out[f"sdv_{side}_sp_found"] = 0

    out.to_csv(out_path, index=False)


def process_date(date_str: str, lookback_days: int, summary: dict) -> None:
    games_path = GAMES_DIR / f"{date_str}_games.csv"
    out_path = OUT_DIR / f"{date_str}_sportsdataverse.csv"

    if not games_path.exists():
        _log(f"MISSING games file: {games_path}", "ERROR")
        summary["errors"] += 1
        return

    games = pd.read_csv(games_path, dtype=str, encoding="utf-8-sig")
    if games.empty:
        _log(f"{date_str} | games file is empty", "ERROR")
        summary["errors"] += 1
        return

    game_date = _game_date_for_file(date_str, games)
    cutoff_date = game_date
    statcast_end = cutoff_date - timedelta(days=1)
    season = game_date.year
    season_start = date(season, 3, 1)

    pitcher_ids = _safe_ints(
        list(games.get("home_pitcher_id", pd.Series(dtype=str)))
        + list(games.get("away_pitcher_id", pd.Series(dtype=str)))
    )

    _log(
        f"{date_str} | games={len(games)} pitchers={len(pitcher_ids)} "
        f"as_of={statcast_end.isoformat()}"
    )

    if not pitcher_ids:
        write_base_output(
            games,
            out_path,
            game_date,
            lookback_days,
            "no_probable_pitchers",
        )
        summary["files_written"] += 1
        summary["rows_written"] += len(games)
        return

    if statcast_end < season_start:
        write_base_output(
            games,
            out_path,
            game_date,
            lookback_days,
            "no_prior_regular_season_data",
        )
        summary["files_written"] += 1
        summary["rows_written"] += len(games)
        return

    try:
        raw = mlb_statcast_search(
            season_start.isoformat(),
            statcast_end.isoformat(),
            player_type="pitcher",
            game_type="R",
            pitchers_lookup=pitcher_ids,
        )
    except Exception as exc:
        _log(f"{date_str} | Statcast pull failed: {exc}", "ERROR")
        write_base_output(
            games,
            out_path,
            game_date,
            lookback_days,
            "statcast_pull_error",
        )
        summary["files_written"] += 1
        summary["rows_written"] += len(games)
        summary["errors"] += 1
        return

    if raw is None or raw.height == 0:
        write_base_output(
            games,
            out_path,
            game_date,
            lookback_days,
            "no_statcast_rows",
        )
        _log(f"{date_str} | Statcast returned zero rows", "WARN")
        summary["files_written"] += 1
        summary["rows_written"] += len(games)
        return

    _log(f"{date_str} | Statcast pitches pulled={raw.height}")

    pitcher_features = build_pitcher_features(
        raw,
        pitcher_ids,
        season,
        cutoff_date,
        lookback_days,
    )

    output = games.copy()
    output = attach_side_features(output, pitcher_features, "home")
    output = attach_side_features(output, pitcher_features, "away")

    output["sdv_as_of_date"] = statcast_end.isoformat()
    output["sdv_season"] = season
    output["sdv_lookback_days"] = lookback_days
    output["sdv_status"] = "ok"

    output.to_csv(out_path, index=False)

    found_home = int(output["sdv_home_sp_found"].sum())
    found_away = int(output["sdv_away_sp_found"].sum())
    _log(
        f"{date_str} | WROTE {out_path} rows={len(output)} "
        f"home_sp_found={found_home}/{len(output)} "
        f"away_sp_found={found_away}/{len(output)}"
    )

    summary["files_written"] += 1
    summary["rows_written"] += len(output)
    summary["statcast_pitches"] += raw.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dates",
        nargs="*",
        help=(
            "Optional date(s) to process (YYYY_MM_DD or YYYY-MM-DD). "
            "If omitted, processes the latest *_games.csv file."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Recent-form window in calendar days (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with LOG_FILE.open("w", encoding="utf-8") as f:
        f.write(f"=== sportsdataverse_mlb RUN {_now()} ===\n")

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "statcast_pitches": 0,
        "errors": 0,
    }

    if args.lookback_days < 1:
        _log("--lookback-days must be >= 1", "ERROR")
        sys.exit(2)

    if args.dates:
        dates = [normalize_date(value) for value in args.dates]
    else:
        game_files = sorted(GAMES_DIR.glob("*_games.csv"))
        if not game_files:
            _log(f"No *_games.csv files found in {GAMES_DIR}", "ERROR")
            sys.exit(1)
        dates = [game_files[-1].stem.replace("_games", "")]

    _log(f"dates={dates} lookback_days={args.lookback_days}")

    for date_str in dates:
        try:
            process_date(date_str, args.lookback_days, summary)
        except Exception as exc:
            _log(
                f"{date_str} FAILED: {exc}\n{traceback.format_exc()}",
                "ERROR",
            )
            summary["errors"] += 1

    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    _log(
        f"SUMMARY files_written={summary['files_written']} "
        f"rows_written={summary['rows_written']} "
        f"statcast_pitches={summary['statcast_pitches']} "
        f"errors={summary['errors']} status={status}"
    )

    print(
        "sportsdataverse_mlb complete. "
        f"{summary['files_written']} files written, "
        f"{summary['rows_written']} rows. Status: {status}"
    )

    if summary["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

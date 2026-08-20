#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_historical_storage.py
"""
Build the normalized SportsDataVerse historical basketball warehouse.

Reads:
  docs/win/basketball/config/sdv_storage.yaml
  docs/win/basketball/config/sdv_seasons.yaml
  docs/win/basketball/scripts/00_intake/sdv_season_mapping.py

Writes season partitions:
  docs/win/basketball/00_intake/sdv/history/{league}/{internal_season}/
    games.parquet
    team_game.parquet
    player_game.parquet
    rosters.parquet
    pbp.parquet
    possessions.parquet
    lineups.parquet
    shots.parquet
    manifest.json

Sportsbook odds are deliberately not copied. Existing immutable snapshots remain:
  docs/win/basketball/00_intake/sportsbook_snapshots/

Source availability in SportsDataVerse 0.0.75:
- NBA: all eight requested tables have release loaders.
- WNBA: all eight requested tables have release loaders.
- NCAAM: games, team_game, player_game, rosters, pbp and shots have release
  loaders. SportsDataVerse 0.0.75 does not publish equivalent season-release
  loaders for NCAAM possessions or lineups. Those two partitions are written
  as explicit zero-row Parquet tables and marked unavailable in manifest.json
  rather than fabricating possession/lineup data from a different source.

The warehouse keeps source columns, normalizes column names to snake_case,
adds stable canonical aliases where possible, and adds provenance columns:
  league
  internal_season
  sdv_season
  source_loader
  ingested_at_utc
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sdv_season_mapping import sdv_season_id


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_storage.yaml"
SEASON_CONFIG_PATH = BASE / "config/sdv_seasons.yaml"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_historical_storage.txt"

VALID_LEAGUES = ("nba", "ncaam", "wnba")
VALID_TABLES = (
    "games",
    "team_game",
    "player_game",
    "rosters",
    "pbp",
    "possessions",
    "lineups",
    "shots",
)

# Exact SportsDataVerse 0.0.75 historical release-loader mapping.
# A None loader means the requested partition exists in the warehouse contract
# but SDV does not publish an equivalent season-release dataset for that league.
LOADER_REGISTRY: dict[str, dict[str, tuple[str, str] | None]] = {
    "nba": {
        "games": ("sportsdataverse.nba", "load_nba_schedule"),
        "team_game": ("sportsdataverse.nba", "load_nba_team_boxscore"),
        "player_game": ("sportsdataverse.nba", "load_nba_player_boxscore"),
        "rosters": ("sportsdataverse.nba", "load_nba_rosters"),
        "pbp": ("sportsdataverse.nba", "load_nba_pbp"),
        "possessions": ("sportsdataverse.nba", "load_nba_stats_possessions"),
        "lineups": ("sportsdataverse.nba", "load_nba_stats_game_lineups"),
        "shots": ("sportsdataverse.nba", "load_nba_shots"),
    },
    "ncaam": {
        "games": ("sportsdataverse.mbb", "load_mbb_schedule"),
        "team_game": ("sportsdataverse.mbb", "load_mbb_team_boxscore"),
        "player_game": ("sportsdataverse.mbb", "load_mbb_player_boxscore"),
        "rosters": ("sportsdataverse.mbb", "load_mbb_rosters"),
        "pbp": ("sportsdataverse.mbb", "load_mbb_pbp"),
        "possessions": None,
        "lineups": None,
        "shots": ("sportsdataverse.mbb", "load_mbb_shots"),
    },
    "wnba": {
        "games": ("sportsdataverse.wnba", "load_wnba_schedule"),
        "team_game": ("sportsdataverse.wnba", "load_wnba_team_boxscore"),
        "player_game": ("sportsdataverse.wnba", "load_wnba_player_boxscore"),
        "rosters": ("sportsdataverse.wnba", "load_wnba_rosters"),
        "pbp": ("sportsdataverse.wnba", "load_wnba_pbp"),
        "possessions": ("sportsdataverse.wnba", "load_wnba_stats_possessions"),
        "lineups": ("sportsdataverse.wnba", "load_wnba_stats_game_lineups"),
        "shots": ("sportsdataverse.wnba", "load_wnba_shots"),
    },
}

CANONICAL_ALIAS_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "games": {
        "game_id": ("game_id", "id"),
        "game_date": ("game_date", "date"),
        "home_team_id": ("home_team_id", "home_id"),
        "away_team_id": ("away_team_id", "away_id"),
        "home_team": (
            "home_team",
            "home_display_name",
            "home_short_display_name",
            "home_name",
            "home_location",
        ),
        "away_team": (
            "away_team",
            "away_display_name",
            "away_short_display_name",
            "away_name",
            "away_location",
        ),
        "venue_name": ("venue_name", "venue_full_name"),
    },
    "team_game": {
        "game_id": ("game_id",),
        "team_id": ("team_id",),
        "team": (
            "team",
            "team_display_name",
            "team_name",
            "team_short_display_name",
            "team_location",
        ),
        "opponent_team_id": ("opponent_team_id",),
        "opponent_team": (
            "opponent_team",
            "opponent_team_display_name",
            "opponent_team_name",
            "opponent_team_location",
        ),
    },
    "player_game": {
        "game_id": ("game_id",),
        "player_id": ("player_id", "athlete_id"),
        "player_name": (
            "player_name",
            "athlete_display_name",
            "athlete_name",
            "athlete_short_name",
        ),
        "team_id": ("team_id",),
    },
    "rosters": {
        "player_id": ("player_id", "athlete_id"),
        "player_name": (
            "player_name",
            "athlete_display_name",
            "athlete_name",
            "athlete_full_name",
            "athlete_short_name",
        ),
        "team_id": ("team_id",),
    },
    "pbp": {
        "game_id": ("game_id",),
        "play_id": ("play_id", "id"),
        "event_text": ("event_text", "text", "short_description"),
        "team_id": ("team_id",),
    },
    "possessions": {
        "game_id": ("game_id",),
        "team_id": ("team_id",),
    },
    "lineups": {
        "game_id": ("game_id",),
        "team_id": ("team_id",),
    },
    "shots": {
        "game_id": ("game_id",),
        "shot_id": ("shot_id", "id"),
        "player_id": ("player_id", "athlete_id", "person_id"),
        "team_id": ("team_id",),
    },
}

ID_COLUMNS = {
    "game_id",
    "play_id",
    "shot_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "home_team_id",
    "away_team_id",
    "athlete_id",
    "person_id",
    "venue_id",
}


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def snake_case(value: str) -> str:
    text = clean(value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def get_polars():
    try:
        return importlib.import_module("polars")
    except ImportError as exc:
        raise RuntimeError(
            "polars is required for SDV Parquet storage. "
            "It is a dependency of sportsdataverse==0.0.75."
        ) from exc


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SDV storage config: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a top-level mapping")

    if config.get("schema_version") != 1:
        raise ValueError(f"{path} schema_version must be 1")

    sdv_cfg = config.get("sportsdataverse")
    if not isinstance(sdv_cfg, dict):
        raise ValueError(f"{path} missing sportsdataverse mapping")

    expected_version = clean(sdv_cfg.get("expected_version"))
    if not expected_version:
        raise ValueError(f"{path} sportsdataverse.expected_version is required")

    storage = config.get("storage")
    if not isinstance(storage, dict):
        raise ValueError(f"{path} missing storage mapping")

    if clean(storage.get("format")).lower() != "parquet":
        raise ValueError(f"{path} storage.format must be parquet")

    storage_root = clean(storage.get("root"))
    if not storage_root:
        raise ValueError(f"{path} storage.root is required")

    snapshots = config.get("sportsbook_snapshots")
    if not isinstance(snapshots, dict):
        raise ValueError(f"{path} missing sportsbook_snapshots mapping")

    if snapshots.get("reuse_existing") is not True:
        raise ValueError(
            f"{path} sportsbook_snapshots.reuse_existing must be true"
        )

    if snapshots.get("duplicate_into_sdv_storage") is not False:
        raise ValueError(
            f"{path} sportsbook_snapshots.duplicate_into_sdv_storage "
            "must be false"
        )

    tables = config.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError(f"{path} tables must be a non-empty list")

    normalized_tables = [clean(item).lower() for item in tables]
    if normalized_tables != list(VALID_TABLES):
        raise ValueError(
            f"{path} tables must exactly equal {list(VALID_TABLES)}"
        )

    if "odds" in normalized_tables or "sportsbook" in normalized_tables:
        raise ValueError(
            "Sportsbook odds must not be duplicated into SDV historical storage"
        )

    seasons = config.get("historical_internal_seasons")
    if not isinstance(seasons, dict):
        raise ValueError(
            f"{path} historical_internal_seasons must be a mapping"
        )

    for league in VALID_LEAGUES:
        values = seasons.get(league)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"{path} historical_internal_seasons.{league} "
                "must be a non-empty list"
            )
        for value in values:
            text = clean(value)
            if not re.fullmatch(r"\d{4}", text):
                raise ValueError(
                    f"{path} invalid {league} internal season: {value!r}"
                )

    return config


def verify_sdv_version(config: dict[str, Any]) -> str:
    expected = clean(
        config["sportsdataverse"]["expected_version"]
    )
    try:
        installed = importlib.metadata.version("sportsdataverse")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "sportsdataverse is not installed"
        ) from exc

    if installed != expected:
        raise RuntimeError(
            f"SportsDataVerse version mismatch: installed={installed}, "
            f"expected={expected}"
        )
    return installed


def configured_storage_root(config: dict[str, Any]) -> Path:
    return Path(clean(config["storage"]["root"]))


def configured_snapshot_root(config: dict[str, Any]) -> Path:
    return Path(clean(config["sportsbook_snapshots"]["root"]))


def configured_compression(config: dict[str, Any]) -> str:
    value = clean(config["storage"].get("compression") or "zstd").lower()
    if value not in {"zstd", "snappy", "gzip", "lz4", "uncompressed"}:
        raise ValueError(f"Unsupported Parquet compression: {value}")
    return value


def resolve_jobs(
    config: dict[str, Any],
    *,
    leagues: list[str] | None = None,
    internal_seasons: list[int] | None = None,
) -> list[tuple[str, int]]:
    selected_leagues = leagues or list(VALID_LEAGUES)

    for league in selected_leagues:
        if league not in VALID_LEAGUES:
            raise ValueError(f"Unsupported league: {league}")

    jobs: list[tuple[str, int]] = []
    configured = config["historical_internal_seasons"]

    for league in selected_leagues:
        league_seasons = [int(v) for v in configured[league]]

        if internal_seasons:
            requested = {int(v) for v in internal_seasons}
            league_seasons = [
                season for season in league_seasons
                if season in requested
            ]

        for season in sorted(set(league_seasons)):
            jobs.append((league, season))

    if internal_seasons and not jobs:
        raise ValueError(
            "Requested internal season is not configured as historical"
        )

    return jobs


def loader_spec(
    league: str,
    table: str,
) -> tuple[str, str] | None:
    if league not in LOADER_REGISTRY:
        raise ValueError(f"Unsupported league: {league}")
    if table not in VALID_TABLES:
        raise ValueError(f"Unsupported table: {table}")
    return LOADER_REGISTRY[league][table]


def call_loader(
    league: str,
    table: str,
    sdv_season: int,
):
    spec = loader_spec(league, table)
    if spec is None:
        return None, None

    module_name, function_name = spec
    module = importlib.import_module(module_name)

    try:
        loader = getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"SportsDataVerse loader missing: "
            f"{module_name}.{function_name}"
        ) from exc

    frame = loader(
        seasons=[int(sdv_season)],
        return_as_pandas=False,
    )
    return frame, f"{module_name}.{function_name}"


def to_polars(frame):
    pl = get_polars()

    if frame is None:
        return pl.DataFrame()

    if isinstance(frame, pl.DataFrame):
        return frame

    if hasattr(frame, "collect"):
        collected = frame.collect()
        if isinstance(collected, pl.DataFrame):
            return collected

    if hasattr(frame, "to_pandas"):
        return pl.from_pandas(frame.to_pandas())

    try:
        return pl.DataFrame(frame)
    except Exception as exc:
        raise TypeError(
            f"Cannot convert loader result to polars DataFrame: "
            f"{type(frame).__name__}"
        ) from exc


def normalize_column_names(df):
    current = list(df.columns)
    renamed = [snake_case(col) for col in current]

    if len(set(renamed)) != len(renamed):
        collisions: dict[str, list[str]] = {}
        for old, new in zip(current, renamed):
            collisions.setdefault(new, []).append(old)
        bad = {
            key: values
            for key, values in collisions.items()
            if len(values) > 1
        }
        raise ValueError(
            f"Column-name collision after snake_case normalization: {bad}"
        )

    mapping = {
        old: new
        for old, new in zip(current, renamed)
        if old != new
    }
    return df.rename(mapping) if mapping else df


def first_existing_column(
    df,
    candidates: tuple[str, ...],
) -> str | None:
    available = set(df.columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def add_canonical_aliases(df, table: str):
    pl = get_polars()
    aliases = CANONICAL_ALIAS_CANDIDATES[table]

    expressions = []
    for canonical, candidates in aliases.items():
        source = first_existing_column(df, candidates)
        if source is None:
            continue
        if canonical == source:
            continue
        expressions.append(
            pl.col(source).alias(canonical)
        )

    if expressions:
        df = df.with_columns(expressions)

    return df


def cast_identifier_columns(df):
    pl = get_polars()
    expressions = []

    for column in df.columns:
        if (
            column in ID_COLUMNS
            or column.endswith("_game_id")
            or column.endswith("_team_id")
            or column.endswith("_player_id")
            or column.endswith("_athlete_id")
        ):
            expressions.append(
                pl.col(column)
                .cast(pl.Utf8, strict=False)
                .alias(column)
            )

    if expressions:
        df = df.with_columns(expressions)

    return df


def add_metadata(
    df,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
    source_loader: str,
    ingested_at_utc: str,
):
    pl = get_polars()

    df = df.with_columns(
        pl.lit(league.upper()).alias("league"),
        pl.lit(int(internal_season))
        .cast(pl.Int32)
        .alias("internal_season"),
        pl.lit(int(sdv_season))
        .cast(pl.Int32)
        .alias("sdv_season"),
        pl.lit(source_loader).alias("source_loader"),
        pl.lit(ingested_at_utc).alias("ingested_at_utc"),
    )

    leading = [
        "league",
        "internal_season",
        "sdv_season",
        "source_loader",
        "ingested_at_utc",
    ]
    rest = [column for column in df.columns if column not in leading]
    return df.select(leading + rest)


def normalize_table(
    frame,
    *,
    table: str,
    league: str,
    internal_season: int,
    sdv_season: int,
    source_loader: str,
    ingested_at_utc: str,
):
    df = to_polars(frame)
    df = normalize_column_names(df)
    df = add_canonical_aliases(df, table)
    df = cast_identifier_columns(df)
    df = add_metadata(
        df,
        league=league,
        internal_season=internal_season,
        sdv_season=sdv_season,
        source_loader=source_loader,
        ingested_at_utc=ingested_at_utc,
    )
    return df


def empty_unavailable_table(
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
    reason: str,
    ingested_at_utc: str,
):
    pl = get_polars()

    return pl.DataFrame(
        schema={
            "league": pl.Utf8,
            "internal_season": pl.Int32,
            "sdv_season": pl.Int32,
            "source_loader": pl.Utf8,
            "ingested_at_utc": pl.Utf8,
            "availability_reason": pl.Utf8,
        }
    ).with_columns(
        pl.lit(league.upper()).alias("league"),
        pl.lit(int(internal_season))
        .cast(pl.Int32)
        .alias("internal_season"),
        pl.lit(int(sdv_season))
        .cast(pl.Int32)
        .alias("sdv_season"),
        pl.lit("").alias("source_loader"),
        pl.lit(ingested_at_utc).alias("ingested_at_utc"),
        pl.lit(reason).alias("availability_reason"),
    ).head(0)


def write_parquet_atomic(
    df,
    path: Path,
    *,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    if tmp.exists():
        tmp.unlink()

    df.write_parquet(
        tmp,
        compression=compression,
        statistics=True,
    )
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    tmp.replace(path)


def season_partition_path(
    storage_root: Path,
    league: str,
    internal_season: int,
) -> Path:
    return storage_root / league / str(internal_season)


def table_path(
    storage_root: Path,
    league: str,
    internal_season: int,
    table: str,
) -> Path:
    return (
        season_partition_path(
            storage_root,
            league,
            internal_season,
        )
        / f"{table}.parquet"
    )


def validate_game_id_presence(df, table: str) -> None:
    if table in {
        "games",
        "team_game",
        "player_game",
        "pbp",
        "possessions",
        "lineups",
        "shots",
    }:
        if df.height > 0 and "game_id" not in df.columns:
            raise ValueError(
                f"{table}: non-empty normalized table is missing game_id"
            )


def build_table(
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
    table: str,
    storage_root: Path,
    compression: str,
    ingested_at_utc: str,
    force: bool,
) -> dict[str, Any]:
    output = table_path(
        storage_root,
        league,
        internal_season,
        table,
    )

    spec = loader_spec(league, table)

    if spec is None:
        reason = (
            "SportsDataVerse 0.0.75 does not publish an equivalent "
            f"historical season-release loader for {league}.{table}"
        )
        df = empty_unavailable_table(
            league=league,
            internal_season=internal_season,
            sdv_season=sdv_season,
            reason=reason,
            ingested_at_utc=ingested_at_utc,
        )
        write_parquet_atomic(
            df,
            output,
            compression=compression,
        )
        log(
            f"UNAVAILABLE | {league} {internal_season} {table} | "
            f"{reason}"
        )
        return {
            "status": "unavailable_from_sdv_release",
            "rows": 0,
            "columns": list(df.columns),
            "file": str(output),
            "source_loader": None,
            "reason": reason,
        }

    module_name, function_name = spec
    source_loader = f"{module_name}.{function_name}"

    if output.exists() and not force:
        log(
            f"SKIP EXISTING | {league} {internal_season} {table} | "
            f"{output}"
        )
        return {
            "status": "existing_not_rebuilt",
            "rows": None,
            "columns": None,
            "file": str(output),
            "source_loader": source_loader,
        }

    log(
        f"LOAD START | league={league} internal={internal_season} "
        f"sdv={sdv_season} table={table} loader={source_loader}"
    )

    frame, actual_loader = call_loader(
        league,
        table,
        sdv_season,
    )

    df = normalize_table(
        frame,
        table=table,
        league=league,
        internal_season=internal_season,
        sdv_season=sdv_season,
        source_loader=actual_loader or source_loader,
        ingested_at_utc=ingested_at_utc,
    )

    validate_game_id_presence(df, table)

    write_parquet_atomic(
        df,
        output,
        compression=compression,
    )

    status = "ready" if df.height > 0 else "no_data_published"
    log(
        f"LOAD COMPLETE | league={league} internal={internal_season} "
        f"table={table} rows={df.height} status={status} file={output}"
    )

    return {
        "status": status,
        "rows": int(df.height),
        "columns": list(df.columns),
        "file": str(output),
        "source_loader": actual_loader or source_loader,
    }


def build_season(
    *,
    config: dict[str, Any],
    league: str,
    internal_season: int,
    tables: list[str] | None = None,
    force: bool = False,
) -> Path:
    storage_root = configured_storage_root(config)
    snapshot_root = configured_snapshot_root(config)
    compression = configured_compression(config)

    mapped_sdv_season = sdv_season_id(
        league,
        internal_season,
        config_path=SEASON_CONFIG_PATH,
    )

    selected_tables = tables or list(VALID_TABLES)
    for table in selected_tables:
        if table not in VALID_TABLES:
            raise ValueError(f"Unsupported table: {table}")

    ingested_at_utc = datetime.now(timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": ingested_at_utc,
        "sportsdataverse_version": clean(
            config["sportsdataverse"]["expected_version"]
        ),
        "league": league.upper(),
        "internal_season": int(internal_season),
        "sdv_season": int(mapped_sdv_season),
        "storage_format": "parquet",
        "compression": compression,
        "sportsbook_snapshots": {
            "path": str(snapshot_root),
            "reused_external_to_sdv_storage": True,
            "copied_into_sdv_storage": False,
        },
        "tables": {},
    }

    for table in selected_tables:
        manifest["tables"][table] = build_table(
            league=league,
            internal_season=internal_season,
            sdv_season=mapped_sdv_season,
            table=table,
            storage_root=storage_root,
            compression=compression,
            ingested_at_utc=ingested_at_utc,
            force=force,
        )

    manifest_path = (
        season_partition_path(
            storage_root,
            league,
            internal_season,
        )
        / "manifest.json"
    )
    write_json_atomic(manifest_path, manifest)
    log(
        f"MANIFEST | league={league} internal={internal_season} "
        f"path={manifest_path}"
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized SportsDataVerse historical basketball "
            "Parquet storage."
        )
    )
    parser.add_argument(
        "--league",
        action="append",
        choices=VALID_LEAGUES,
        help="League to build; may be repeated. Default: all configured.",
    )
    parser.add_argument(
        "--internal-season",
        type=int,
        action="append",
        help=(
            "Historical internal season to build; may be repeated. "
            "Must exist in sdv_storage.yaml."
        ),
    )
    parser.add_argument(
        "--table",
        action="append",
        choices=VALID_TABLES,
        help="Table to build; may be repeated. Default: all eight tables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing Parquet files.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config/version/mappings without downloading datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Storage config path (default: {CONFIG_PATH})",
    )
    return parser


def validate_mappings(
    config: dict[str, Any],
    jobs: list[tuple[str, int]],
) -> dict[str, int]:
    mapped: dict[str, int] = {}

    for league, internal_season in jobs:
        sdv_season = sdv_season_id(
            league,
            internal_season,
            config_path=SEASON_CONFIG_PATH,
        )
        mapped[f"{league}:{internal_season}"] = int(sdv_season)

    return mapped


def main() -> None:
    args = build_parser().parse_args()

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== SDV HISTORICAL STORAGE "
        f"{datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )

    try:
        config = load_config(args.config)
        installed_version = verify_sdv_version(config)

        jobs = resolve_jobs(
            config,
            leagues=args.league,
            internal_seasons=args.internal_season,
        )
        mappings = validate_mappings(config, jobs)

        log(
            f"CONFIG VALID | sportsdataverse={installed_version} "
            f"jobs={jobs} mappings={mappings}"
        )

        if args.validate_config:
            print("SDV historical storage config valid.")
            return

        manifests: list[Path] = []

        for league, internal_season in jobs:
            manifests.append(
                build_season(
                    config=config,
                    league=league,
                    internal_season=internal_season,
                    tables=args.table,
                    force=args.force,
                )
            )

        log(f"STATUS: SUCCESS | manifests={len(manifests)}")
        print("SDV historical basketball storage complete.")

    except Exception as exc:
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

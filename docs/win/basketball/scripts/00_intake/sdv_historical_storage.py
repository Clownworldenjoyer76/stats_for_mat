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

NCAAM possessions and lineups use the separate stats.ncaa.org/bigballR path
provided by SportsDataVerse 0.0.75. The release-loader ESPN game id remains
the warehouse game_id. The stats.ncaa.org contest id is preserved separately
as ncaa_game_id in derived possession/lineup rows.

NCAA contest ids are resolved deterministically:
  ESPN team ids from games.parquet
    -> sportsdataverse.mbb.ncaa_espn_team_crosswalk()
    -> season-specific NCAA team ids
    -> both teams' sportsdataverse.mbb.ncaa_mbb_team_schedule()
    -> shared stats.ncaa.org contest id for that matchup/date.

No fuzzy matching is used.

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
from datetime import date, datetime, timezone
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

NCAAM_DERIVED_TABLES = ("possessions", "lineups")

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

NCAAM_SOURCE_FUNCTIONS = {
    "possessions": (
        "sportsdataverse.mbb.ncaa_mbb_game_pbp -> "
        "sportsdataverse.mbb.ncaa_mbb_possessions"
    ),
    "lineups": (
        "sportsdataverse.mbb.ncaa_mbb_game_pbp -> "
        "sportsdataverse.mbb.ncaa_mbb_lineups"
    ),
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
    "ncaa_game_id",
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
            "Install it explicitly in requirements.txt."
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
        raise ValueError(f"{path} sportsbook_snapshots.reuse_existing must be true")

    if snapshots.get("duplicate_into_sdv_storage") is not False:
        raise ValueError(
            f"{path} sportsbook_snapshots.duplicate_into_sdv_storage must be false"
        )

    tables = config.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError(f"{path} tables must be a non-empty list")

    normalized_tables = [clean(item).lower() for item in tables]
    if normalized_tables != list(VALID_TABLES):
        raise ValueError(f"{path} tables must exactly equal {list(VALID_TABLES)}")

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
    expected = clean(config["sportsdataverse"]["expected_version"])
    try:
        installed = importlib.metadata.version("sportsdataverse")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("sportsdataverse is not installed") from exc

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
                season for season in league_seasons if season in requested
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
        raise RuntimeError(
            f"{league}.{table} does not use a season-release loader"
        )

    module_name, function_name = spec
    module = importlib.import_module(module_name)

    try:
        loader = getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"SportsDataVerse loader missing: {module_name}.{function_name}"
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
            "Cannot convert loader result to polars DataFrame: "
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
        if source is None or canonical == source:
            continue
        expressions.append(pl.col(source).alias(canonical))

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


def manifest_entry(
    *,
    df,
    output: Path,
    source_loader: str,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "rows": int(df.height),
        "columns": list(df.columns),
        "file": str(output),
        "output_filename": output.name,
        "source_loader": source_loader,
        "source_function": source_loader,
    }


def existing_manifest_entry(
    *,
    output: Path,
    source_loader: str,
) -> dict[str, Any]:
    pl = get_polars()
    df = pl.read_parquet(output)
    return manifest_entry(
        df=df,
        output=output,
        source_loader=source_loader,
        status="existing_not_rebuilt",
    )


def build_release_table(
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
        raise RuntimeError(
            f"{league}.{table} requires its dedicated build path"
        )

    module_name, function_name = spec
    source_loader = f"{module_name}.{function_name}"

    if output.exists() and not force:
        log(
            f"SKIP EXISTING | {league} {internal_season} {table} | "
            f"{output}"
        )
        return existing_manifest_entry(
            output=output,
            source_loader=source_loader,
        )

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

    return manifest_entry(
        df=df,
        output=output,
        source_loader=actual_loader or source_loader,
        status=status,
    )


def ncaa_season_label(sdv_season: int) -> str:
    start = int(sdv_season) - 1
    return f"{start}-{str(int(sdv_season))[-2:]}"


def normalize_date(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""

    if "T" in text and re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        return text[:10]

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text

    for fmt in (
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)

    return text


def ncaam_team_crosswalk(sdv_season: int) -> dict[str, int]:
    mbb = importlib.import_module("sportsdataverse.mbb")
    pl = get_polars()

    crosswalk = to_polars(
        mbb.ncaa_espn_team_crosswalk(
            league="mbb",
            return_as_pandas=False,
        )
    )

    required = {"season", "ncaa_team_id", "espn_team_id"}
    missing = required - set(crosswalk.columns)
    if missing:
        raise RuntimeError(
            "NCAA/ESPN team crosswalk missing columns: "
            f"{sorted(missing)}"
        )

    season = ncaa_season_label(sdv_season)
    crosswalk = crosswalk.filter(
        (pl.col("season") == season)
        & pl.col("espn_team_id").is_not_null()
        & pl.col("ncaa_team_id").is_not_null()
    )

    mapping: dict[str, int] = {}
    duplicates: set[str] = set()

    for row in crosswalk.select(
        "espn_team_id",
        "ncaa_team_id",
    ).iter_rows(named=True):
        espn_id = clean(row["espn_team_id"])
        ncaa_id = int(row["ncaa_team_id"])
        if espn_id in mapping and mapping[espn_id] != ncaa_id:
            duplicates.add(espn_id)
        mapping[espn_id] = ncaa_id

    if duplicates:
        raise RuntimeError(
            "NCAA/ESPN team crosswalk has conflicting ESPN ids: "
            f"{sorted(duplicates)}"
        )

    if not mapping:
        raise RuntimeError(
            f"NCAA/ESPN team crosswalk has no rows for season {season}"
        )

    return mapping


def schedule_index(schedule_df) -> dict[str, str]:
    if "game_id" not in schedule_df.columns:
        raise RuntimeError("NCAA team schedule is missing game_id")

    has_date = "game_date" in schedule_df.columns
    out: dict[str, str] = {}

    for row in schedule_df.iter_rows(named=True):
        game_id = clean(row.get("game_id"))
        if not game_id:
            continue
        out[game_id] = (
            normalize_date(row.get("game_date"))
            if has_date
            else ""
        )

    return out


def resolve_ncaa_contest_id(
    *,
    espn_game_id: str,
    game_date: str,
    home_espn_team_id: str,
    away_espn_team_id: str,
    team_map: dict[str, int],
    schedule_cache: dict[int, dict[str, str]],
    mbb,
    fetcher,
) -> str:
    try:
        home_ncaa_team_id = team_map[home_espn_team_id]
        away_ncaa_team_id = team_map[away_espn_team_id]
    except KeyError as exc:
        missing_team = clean(exc.args[0])
        raise RuntimeError(
            f"game_id={espn_game_id}: no NCAA team mapping for "
            f"ESPN team_id={missing_team}"
        ) from exc

    def get_schedule(team_id: int) -> dict[str, str]:
        if team_id not in schedule_cache:
            frame = mbb.ncaa_mbb_team_schedule(
                team_id=int(team_id),
                fetcher=fetcher,
                return_as_pandas=False,
            )
            schedule_cache[team_id] = schedule_index(to_polars(frame))
        return schedule_cache[team_id]

    home_schedule = get_schedule(home_ncaa_team_id)
    away_schedule = get_schedule(away_ncaa_team_id)

    common = sorted(
        set(home_schedule).intersection(away_schedule)
    )

    if not common:
        raise RuntimeError(
            f"game_id={espn_game_id}: teams have no shared NCAA contest id"
        )

    target_date = normalize_date(game_date)

    if len(common) == 1:
        return common[0]

    dated = [
        game_id
        for game_id in common
        if target_date
        and (
            home_schedule.get(game_id) == target_date
            or away_schedule.get(game_id) == target_date
        )
    ]

    if len(dated) == 1:
        return dated[0]

    raise RuntimeError(
        f"game_id={espn_game_id}: NCAA contest id is ambiguous; "
        f"candidates={common} date={target_date or 'unknown'}"
    )


def prepare_ncaam_games(
    games_df,
    *,
    sdv_season: int,
    mbb,
    fetcher,
):
    pl = get_polars()

    required = {
        "game_id",
        "game_date",
        "home_team_id",
        "away_team_id",
    }
    missing = required - set(games_df.columns)
    if missing:
        raise RuntimeError(
            "NCAAM games.parquet missing required columns: "
            f"{sorted(missing)}"
        )

    team_map = ncaam_team_crosswalk(sdv_season)
    schedule_cache: dict[int, dict[str, str]] = {}
    resolved: dict[str, str] = {}

    games = games_df.select(
        "game_id",
        "game_date",
        "home_team_id",
        "away_team_id",
    ).unique(
        subset=["game_id"],
        maintain_order=True,
    )

    for row in games.iter_rows(named=True):
        espn_game_id = clean(row["game_id"])
        if not espn_game_id:
            raise RuntimeError("NCAAM games.parquet contains a blank game_id")

        try:
            resolved[espn_game_id] = resolve_ncaa_contest_id(
                espn_game_id=espn_game_id,
                game_date=clean(row["game_date"]),
                home_espn_team_id=clean(row["home_team_id"]),
                away_espn_team_id=clean(row["away_team_id"]),
                team_map=team_map,
                schedule_cache=schedule_cache,
                mbb=mbb,
                fetcher=fetcher,
            )
        except Exception as exc:
            log(
                f"NCAAM GAME FAILED | game_id={espn_game_id} | "
                f"stage=resolve_ncaa_contest_id | error={exc}"
            )
            raise

    return games.with_columns(
        pl.col("game_id")
        .replace(resolved)
        .alias("ncaa_game_id")
    ).select(
        "game_id",
        "ncaa_game_id",
    )


def remap_ncaam_derived_game_id(
    frame,
    *,
    espn_game_id: str,
    ncaa_game_id: str,
):
    pl = get_polars()
    df = to_polars(frame)

    if df.height == 0:
        return df

    if "game_id" in df.columns:
        df = df.with_columns(
            pl.col("game_id")
            .cast(pl.Utf8, strict=False)
            .alias("ncaa_game_id"),
            pl.lit(espn_game_id).alias("game_id"),
        )
    else:
        df = df.with_columns(
            pl.lit(espn_game_id).alias("game_id"),
            pl.lit(ncaa_game_id).alias("ncaa_game_id"),
        )

    return df


def build_ncaam_derived_tables(
    *,
    internal_season: int,
    sdv_season: int,
    selected_tables: list[str],
    storage_root: Path,
    compression: str,
    ingested_at_utc: str,
    force: bool,
) -> dict[str, dict[str, Any]]:
    pl = get_polars()
    mbb = importlib.import_module("sportsdataverse.mbb")
    fetch_module = importlib.import_module(
        "sportsdataverse.mbb.mbb_ncaa_fetch"
    )
    NcaaFetcher = getattr(fetch_module, "NcaaFetcher")

    requested = [
        table
        for table in NCAAM_DERIVED_TABLES
        if table in selected_tables
    ]
    if not requested:
        return {}

    outputs = {
        table: table_path(
            storage_root,
            "ncaam",
            internal_season,
            table,
        )
        for table in requested
    }

    entries: dict[str, dict[str, Any]] = {}
    needs_build: list[str] = []

    for table in requested:
        output = outputs[table]
        source = NCAAM_SOURCE_FUNCTIONS[table]
        if output.exists() and not force:
            entries[table] = existing_manifest_entry(
                output=output,
                source_loader=source,
            )
            log(
                f"SKIP EXISTING | ncaam {internal_season} {table} | "
                f"{output}"
            )
        else:
            needs_build.append(table)

    if not needs_build:
        return entries

    games_path = table_path(
        storage_root,
        "ncaam",
        internal_season,
        "games",
    )
    if not games_path.exists():
        raise RuntimeError(
            "NCAAM games.parquet must exist before possessions/lineups: "
            f"{games_path}"
        )

    games_df = pl.read_parquet(games_path)
    if games_df.height == 0:
        raise RuntimeError(
            f"NCAAM games.parquet has zero rows: {games_path}"
        )

    frames: dict[str, list[Any]] = {
        table: []
        for table in needs_build
    }

    with NcaaFetcher.with_browser() as fetcher:
        id_map = prepare_ncaam_games(
            games_df,
            sdv_season=sdv_season,
            mbb=mbb,
            fetcher=fetcher,
        )

        for row in id_map.iter_rows(named=True):
            espn_game_id = clean(row["game_id"])
            ncaa_game_id = clean(row["ncaa_game_id"])

            log(
                f"NCAAM GAME START | game_id={espn_game_id} | "
                f"ncaa_game_id={ncaa_game_id}"
            )

            try:
                pbp = mbb.ncaa_mbb_game_pbp(
                    ncaa_game_id,
                    fetcher=fetcher,
                    return_as_pandas=False,
                )
                pbp = to_polars(pbp)

                if pbp.height == 0:
                    raise RuntimeError(
                        "ncaa_mbb_game_pbp returned zero rows"
                    )

                if "possessions" in needs_build:
                    possessions = mbb.ncaa_mbb_possessions(
                        pbp,
                        simple=False,
                        fix_cross_game_leak=True,
                        return_as_pandas=False,
                    )
                    possessions = remap_ncaam_derived_game_id(
                        possessions,
                        espn_game_id=espn_game_id,
                        ncaa_game_id=ncaa_game_id,
                    )
                    frames["possessions"].append(possessions)

                if "lineups" in needs_build:
                    lineups = mbb.ncaa_mbb_lineups(
                        pbp,
                        include_transition=False,
                        fix_tip_in=True,
                        return_as_pandas=False,
                    )
                    lineups = remap_ncaam_derived_game_id(
                        lineups,
                        espn_game_id=espn_game_id,
                        ncaa_game_id=ncaa_game_id,
                    )
                    frames["lineups"].append(lineups)

            except Exception as exc:
                log(
                    f"NCAAM GAME FAILED | game_id={espn_game_id} | "
                    f"ncaa_game_id={ncaa_game_id} | "
                    f"stage=pbp_or_transform | error={exc}"
                )
                raise RuntimeError(
                    f"NCAAM game failed: game_id={espn_game_id}, "
                    f"ncaa_game_id={ncaa_game_id}: {exc}"
                ) from exc

            log(
                f"NCAAM GAME COMPLETE | game_id={espn_game_id} | "
                f"ncaa_game_id={ncaa_game_id}"
            )

    for table in needs_build:
        if not frames[table]:
            raise RuntimeError(
                f"NCAAM {table}: no game outputs were produced"
            )

        combined = pl.concat(
            [to_polars(frame) for frame in frames[table]],
            how="diagonal_relaxed",
        )

        if combined.height == 0:
            raise RuntimeError(
                f"NCAAM {table}: concatenated season output has zero rows"
            )

        source = NCAAM_SOURCE_FUNCTIONS[table]
        df = normalize_table(
            combined,
            table=table,
            league="ncaam",
            internal_season=internal_season,
            sdv_season=sdv_season,
            source_loader=source,
            ingested_at_utc=ingested_at_utc,
        )

        validate_game_id_presence(df, table)

        output = outputs[table]
        write_parquet_atomic(
            df,
            output,
            compression=compression,
        )

        log(
            f"LOAD COMPLETE | league=ncaam internal={internal_season} "
            f"table={table} rows={df.height} status=ready file={output}"
        )

        entries[table] = manifest_entry(
            df=df,
            output=output,
            source_loader=source,
            status="ready",
        )

    return entries


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

    # Build every release-backed table first. NCAAM possessions/lineups need
    # games.parquet and are built together afterward so NCAA PBP is fetched once.
    for table in selected_tables:
        if league == "ncaam" and table in NCAAM_DERIVED_TABLES:
            continue

        manifest["tables"][table] = build_release_table(
            league=league,
            internal_season=internal_season,
            sdv_season=mapped_sdv_season,
            table=table,
            storage_root=storage_root,
            compression=compression,
            ingested_at_utc=ingested_at_utc,
            force=force,
        )

    if league == "ncaam":
        requested_derived = [
            table
            for table in NCAAM_DERIVED_TABLES
            if table in selected_tables
        ]

        if requested_derived:
            games_path = table_path(
                storage_root,
                "ncaam",
                internal_season,
                "games",
            )

            # A partial --table possessions/lineups run still needs games.parquet.
            if not games_path.exists():
                games_entry = build_release_table(
                    league="ncaam",
                    internal_season=internal_season,
                    sdv_season=mapped_sdv_season,
                    table="games",
                    storage_root=storage_root,
                    compression=compression,
                    ingested_at_utc=ingested_at_utc,
                    force=force,
                )
                if "games" in selected_tables:
                    manifest["tables"]["games"] = games_entry

            derived_entries = build_ncaam_derived_tables(
                internal_season=internal_season,
                sdv_season=mapped_sdv_season,
                selected_tables=selected_tables,
                storage_root=storage_root,
                compression=compression,
                ingested_at_utc=ingested_at_utc,
                force=force,
            )
            manifest["tables"].update(derived_entries)

    # Preserve configured table order in manifest.
    manifest["tables"] = {
        table: manifest["tables"][table]
        for table in selected_tables
        if table in manifest["tables"]
    }

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

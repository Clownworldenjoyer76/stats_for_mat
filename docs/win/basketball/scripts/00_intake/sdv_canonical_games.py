#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_canonical_games.py
"""
Build the SportsDataVerse-backed canonical basketball game/ID layer.

Primary source:
  SportsDataVerse 0.0.75 ESPN schedule surfaces:
    NBA   -> sportsdataverse.nba.espn_nba_schedule
    NCAAM -> sportsdataverse.mbb.espn_mbb_schedule
    WNBA  -> sportsdataverse.wnba.espn_wnba_schedule

Season translation:
  docs/win/basketball/scripts/00_intake/sdv_season_mapping.py
  docs/win/basketball/config/sdv_seasons.yaml

Writes one season-level canonical schedule per league:
  docs/win/basketball/00_intake/sdv/canonical_games/nba/{internal}_NBA_games.csv
  docs/win/basketball/00_intake/sdv/canonical_games/ncaam/{internal}_NCAAM_games.csv
  docs/win/basketball/00_intake/sdv/canonical_games/wnba/{internal}_WNBA_games.csv

The canonical SDV/ESPN game_id becomes the primary identity used downstream.
Sportsbook rows remain a fallback/compatibility source in basketball_daily_games.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from sdv_season_mapping import sdv_season_id


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_seasons.yaml"
OUTPUT_ROOT = BASE / "00_intake/sdv/canonical_games"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_canonical_games.txt"
NY_TZ = ZoneInfo("America/New_York")

TEAM_MAPS = {
    "nba": BASE / "maps/team_map_nba.csv",
    "ncaam": BASE / "maps/team_map_ncaam.csv",
    "wnba": BASE / "maps/team_map_wnba.csv",
}

LEAGUES = {
    "nba": {
        "label": "NBA",
        "sport": "basketball",
        "limit": 5000,
    },
    "ncaam": {
        "label": "NCAAM",
        "sport": "basketball",
        "limit": 15000,
    },
    "wnba": {
        "label": "WNBA",
        "sport": "basketball",
        "limit": 2000,
    },
}

FIELDNAMES = [
    "sport",
    "league",
    "internal_season",
    "sdv_season",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "game_id",
    "home_team_id",
    "away_team_id",
    "neutral_site",
    "venue_id",
    "venue_name",
    "status",
    "source",
    "fetched_at_utc",
]

REQUIRED_SDV_COLUMNS = {
    "game_id",
    "date",
    "home_display_name",
    "away_display_name",
    "season",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def clean_id(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def current_internal_season(league: str, now: datetime | None = None) -> int:
    """
    Internal labels:
      NBA/NCAAM -> season START year
      WNBA      -> calendar year
    """
    if league not in LEAGUES:
        raise ValueError(f"Unsupported league: {league}")

    local_now = now or datetime.now(NY_TZ)
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=NY_TZ)
    local_now = local_now.astimezone(NY_TZ)

    if league in {"nba", "ncaam"}:
        return local_now.year if local_now.month >= 10 else local_now.year - 1
    return local_now.year


def load_team_map(league: str) -> dict[str, str]:
    path = TEAM_MAPS[league]
    mapping: dict[str, str] = {}

    if not path.exists():
        log(f"TEAM MAP MISSING: {path}; SDV display names will be used unchanged")
        return mapping

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_league = clean(row.get("league")).lower()
            alias = clean(row.get("alias")).casefold()
            canonical = clean(row.get("canonical_team"))
            if row_league == league and alias and canonical:
                mapping[alias] = canonical

    return mapping


def normalize_team_name(value: Any, team_map: dict[str, str]) -> str:
    text = clean(value)
    if not text:
        return ""
    return team_map.get(text.casefold(), text)


def parse_event_datetime(value: Any) -> datetime:
    text = clean(value)
    if not text:
        raise ValueError("SDV schedule row has blank date")
    parsed = pd.to_datetime(text, utc=True, errors="raise")
    return parsed.to_pydatetime().astimezone(NY_TZ)


def bool_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return text


def fetch_schedule(league: str, sdv_season: int, limit: int) -> pd.DataFrame:
    if league == "nba":
        from sportsdataverse.nba import espn_nba_schedule
        df = espn_nba_schedule(
            dates=sdv_season,
            season_type=None,
            limit=limit,
            return_as_pandas=True,
        )
    elif league == "ncaam":
        from sportsdataverse.mbb import espn_mbb_schedule
        df = espn_mbb_schedule(
            dates=sdv_season,
            groups=50,
            season_type=None,
            limit=limit,
            return_as_pandas=True,
        )
    elif league == "wnba":
        from sportsdataverse.wnba import espn_wnba_schedule
        df = espn_wnba_schedule(
            dates=sdv_season,
            season_type=None,
            limit=limit,
            return_as_pandas=True,
        )
    else:
        raise ValueError(f"Unsupported league: {league}")

    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def validate_source_schema(df: pd.DataFrame, league: str, limit: int) -> None:
    missing = sorted(REQUIRED_SDV_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{league}: SDV schedule schema missing required columns: {missing}"
        )

    if len(df) >= limit:
        raise ValueError(
            f"{league}: SDV schedule returned {len(df)} rows at configured "
            f"limit={limit}; refusing possible truncated season"
        )


def source_team_name(row: pd.Series, side: str) -> str:
    for column in (
        f"{side}_display_name",
        f"{side}_short_display_name",
        f"{side}_location",
        f"{side}_name",
    ):
        value = clean(row.get(column))
        if value:
            return value
    return ""


def canonicalize_schedule(
    df: pd.DataFrame,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
    fetched_at_utc: str,
) -> list[dict[str, str]]:
    cfg = LEAGUES[league]
    team_map = load_team_map(league)
    rows: list[dict[str, str]] = []

    seen_by_id: dict[str, tuple[str, str, str]] = {}
    seen_by_identity: dict[tuple[str, str, str], str] = {}

    for _, source in df.iterrows():
        source_season = clean_id(source.get("season"))
        if source_season and int(source_season) != int(sdv_season):
            raise ValueError(
                f"{league}: SDV returned season={source_season} while "
                f"explicit mapping requires season={sdv_season}"
            )

        game_id = clean_id(source.get("game_id") or source.get("id"))
        if not game_id:
            raise ValueError(f"{league}: SDV schedule row has blank game_id")

        event_value = source.get("date")
        if clean(event_value) == "":
            event_value = source.get("start_date")
        event_dt = parse_event_datetime(event_value)

        game_date = event_dt.strftime("%Y_%m_%d")
        game_time = event_dt.strftime("%I:%M %p")

        home_team = normalize_team_name(
            source_team_name(source, "home"),
            team_map,
        )
        away_team = normalize_team_name(
            source_team_name(source, "away"),
            team_map,
        )

        if not home_team or not away_team:
            raise ValueError(
                f"{league}: SDV schedule row game_id={game_id} "
                f"has blank home/away team"
            )

        identity = (
            game_date,
            home_team.casefold(),
            away_team.casefold(),
        )

        old_identity = seen_by_id.get(game_id)
        if old_identity is not None and old_identity != identity:
            raise ValueError(
                f"{league}: SDV game_id={game_id} maps to multiple identities: "
                f"{old_identity} vs {identity}"
            )

        old_game_id = seen_by_identity.get(identity)
        if old_game_id is not None and old_game_id != game_id:
            raise ValueError(
                f"{league}: one SDV identity has multiple game_ids: "
                f"{identity} -> {old_game_id} vs {game_id}"
            )

        seen_by_id[game_id] = identity
        seen_by_identity[identity] = game_id

        rows.append(
            {
                "sport": cfg["sport"],
                "league": cfg["label"],
                "internal_season": str(internal_season),
                "sdv_season": str(sdv_season),
                "game_date": game_date,
                "game_time": game_time,
                "home_team": home_team,
                "away_team": away_team,
                "game_id": game_id,
                "home_team_id": clean_id(source.get("home_id")),
                "away_team_id": clean_id(source.get("away_id")),
                "neutral_site": bool_text(source.get("neutral_site")),
                "venue_id": clean_id(source.get("venue_id")),
                "venue_name": clean(source.get("venue_full_name")),
                "status": clean(source.get("status_type_description")),
                "source": "sportsdataverse_espn",
                "fetched_at_utc": fetched_at_utc,
            }
        )

    rows.sort(
        key=lambda row: (
            row["game_date"],
            row["game_time"],
            row["home_team"],
            row["away_team"],
            row["game_id"],
        )
    )
    return rows


def output_path(league: str, internal_season: int) -> Path:
    label = LEAGUES[league]["label"]
    return (
        OUTPUT_ROOT
        / league
        / f"{internal_season}_{label}_games.csv"
    )


def write_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    tmp.replace(path)


def existing_file_is_valid(
    path: Path,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
) -> bool:
    if not path.exists():
        return False

    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if not set(FIELDNAMES).issubset(fieldnames):
                return False

            for row in reader:
                if clean(row.get("league")).upper() != LEAGUES[league]["label"]:
                    return False
                if clean_id(row.get("internal_season")) != str(internal_season):
                    return False
                if clean_id(row.get("sdv_season")) != str(sdv_season):
                    return False
        return True
    except Exception:
        return False


def build_league(
    league: str,
    *,
    internal_season: int | None = None,
    allow_cached_on_fetch_error: bool = True,
) -> Path:
    if league not in LEAGUES:
        raise ValueError(f"Unsupported league: {league}")

    internal = (
        int(internal_season)
        if internal_season is not None
        else current_internal_season(league)
    )
    mapped_sdv_season = sdv_season_id(
        league,
        internal,
        config_path=CONFIG_PATH,
    )
    cfg = LEAGUES[league]
    path = output_path(league, internal)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()

    log(
        f"FETCH START | league={cfg['label']} "
        f"internal_season={internal} sdv_season={mapped_sdv_season}"
    )

    try:
        df = fetch_schedule(
            league,
            mapped_sdv_season,
            cfg["limit"],
        )
        validate_source_schema(
            df,
            league,
            cfg["limit"],
        )
        rows = canonicalize_schedule(
            df,
            league=league,
            internal_season=internal,
            sdv_season=mapped_sdv_season,
            fetched_at_utc=fetched_at_utc,
        )
        write_atomic(path, rows)
        log(
            f"FETCH SUCCESS | league={cfg['label']} rows={len(rows)} "
            f"path={path}"
        )
        return path

    except Exception as exc:
        if (
            allow_cached_on_fetch_error
            and existing_file_is_valid(
                path,
                league=league,
                internal_season=internal,
                sdv_season=mapped_sdv_season,
            )
        ):
            log(
                f"FETCH WARNING | league={cfg['label']} using valid cached "
                f"canonical schedule {path} because refresh failed: {exc}"
            )
            return path
        raise


def build_current_canonical_games(
    leagues: list[str] | None = None,
) -> list[Path]:
    selected = leagues or ["nba", "ncaam", "wnba"]
    paths: list[Path] = []

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== SDV CANONICAL GAMES {datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )

    for league in selected:
        paths.append(build_league(league))

    log(f"STATUS: SUCCESS | files={len(paths)}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SDV-backed canonical basketball schedules/game IDs."
    )
    parser.add_argument(
        "--league",
        choices=sorted(LEAGUES),
        action="append",
        help="League to build; may be repeated. Default: all leagues.",
    )
    parser.add_argument(
        "--internal-season",
        type=int,
        help=(
            "Explicit internal season. Only valid when exactly one --league "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--no-cache-fallback",
        action="store_true",
        help="Fail if the SDV refresh fails even when a valid cached file exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = args.league or ["nba", "ncaam", "wnba"]

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== SDV CANONICAL GAMES {datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )

    try:
        if args.internal_season is not None:
            if len(selected) != 1:
                raise ValueError(
                    "--internal-season requires exactly one --league"
                )
            build_league(
                selected[0],
                internal_season=args.internal_season,
                allow_cached_on_fetch_error=not args.no_cache_fallback,
            )
        else:
            for league in selected:
                build_league(
                    league,
                    allow_cached_on_fetch_error=not args.no_cache_fallback,
                )

        log("STATUS: SUCCESS")
        print("SDV canonical basketball games complete.")

    except Exception as exc:
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

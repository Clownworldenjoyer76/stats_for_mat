#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_wnba_historical_storage.py
"""Build SportsDataVerse historical storage for WNBA seasons only.

This script intentionally leaves sdv_historical_storage.py unchanged.

It reuses the stable shared storage/load/normalization code from that module,
but replaces only the historical WNBA Stats -> ESPN schedule crosswalk inside
this process.

Historical WNBA Stats schedules may arrive as either:
- one explicit game row with home_* / away_* columns; or
- two team rows per game.

For both schemas, matching is performed by:

    game_date + unordered pair of teams

The canonical ESPN games.parquet supplies home/away orientation.
Scores are used only to disambiguate multiple canonical candidates.
"""
from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sdv_historical_storage as core


LOG_FILE = (
    core.ERROR_DIR
    / "sdv_wnba_historical_storage.txt"
)


def clean(value: Any) -> str:
    return core.clean(value)


def side_payload(
    *,
    team_name: Any,
    team_abbreviation: Any,
    pts: Any,
) -> dict[str, Any]:
    value = None

    if (
        pts is not None
        and clean(pts)
    ):
        try:
            value = float(pts)
        except (
            TypeError,
            ValueError,
        ):
            value = None

    return {
        "team_name": clean(
            team_name
        ),
        "team_abbreviation": clean(
            team_abbreviation
        ),
        "pts": value,
    }


def side_identity(
    side: dict[str, Any],
) -> tuple[
    str,
    str,
    float | None,
]:
    return (
        clean(
            side.get(
                "team_name"
            )
        ).lower(),
        clean(
            side.get(
                "team_abbreviation"
            )
        ).lower(),
        side.get(
            "pts"
        ),
    )


def side_team_identity(
    side: dict[str, Any],
) -> tuple[
    str,
    str,
]:
    return (
        clean(
            side.get(
                "team_name"
            )
        ).lower(),
        clean(
            side.get(
                "team_abbreviation"
            )
        ).lower(),
    )


def side_variants(
    side: dict[str, Any],
) -> set[str]:
    return (
        core.team_variants(
            side.get(
                "team_name"
            )
        )
        | core.team_variants(
            side.get(
                "team_abbreviation"
            )
        )
    )


def build_wnba_legacy_schedule_crosswalk(
    cfg: dict[str, Any],
    internal_season: int,
    sdv_season: int,
):
    """Map historical WNBA Stats IDs to canonical ESPN game IDs."""
    P = core.pl()

    (
        stats_schedule,
        stats_source,
    ) = core.load_wnba_stats_schedule(
        sdv_season
    )

    columns = set(
        stats_schedule.columns
    )

    game_row_required = {
        "game_id",
        "game_date",
        "home_team_name",
        "home_team_abbreviation",
        "away_team_name",
        "away_team_abbreviation",
    }

    team_row_required = {
        "game_id",
        "game_date",
        "team_name",
        "team_abbreviation",
    }

    has_game_rows = (
        game_row_required
        <= columns
    )

    has_team_rows = (
        team_row_required
        <= columns
    )

    if not (
        has_game_rows
        or has_team_rows
    ):
        raise RuntimeError(
            "WNBA Stats schedule has "
            "unsupported schema; "
            f"columns={stats_schedule.columns}"
        )

    stats_games: dict[
        str,
        dict[str, Any],
    ] = {}

    bad_date_ids: set[str] = set()
    conflicting_ids: set[str] = set()
    incomplete_ids: set[str] = set()

    exact_duplicate_rows = 0

    if has_game_rows:
        schema_used = "game_rows"

        selected = [
            "game_id",
            "game_date",
            "home_team_name",
            "home_team_abbreviation",
            "away_team_name",
            "away_team_abbreviation",
        ]

        if (
            "home_pts"
            in columns
        ):
            selected.append(
                "home_pts"
            )

        if (
            "away_pts"
            in columns
        ):
            selected.append(
                "away_pts"
            )

        for row in (
            stats_schedule
            .select(
                selected
            )
            .iter_rows(
                named=True
            )
        ):
            native_game_id = clean(
                row.get(
                    "game_id"
                )
            )

            if not native_game_id:
                continue

            date_key = (
                core
                .normalize_legacy_schedule_date(
                    row.get(
                        "game_date"
                    )
                )
            )

            if not date_key:
                bad_date_ids.add(
                    native_game_id
                )
                continue

            team_a = side_payload(
                team_name=row.get(
                    "home_team_name"
                ),
                team_abbreviation=row.get(
                    "home_team_abbreviation"
                ),
                pts=row.get(
                    "home_pts"
                ),
            )

            team_b = side_payload(
                team_name=row.get(
                    "away_team_name"
                ),
                team_abbreviation=row.get(
                    "away_team_abbreviation"
                ),
                pts=row.get(
                    "away_pts"
                ),
            )

            if (
                not side_variants(
                    team_a
                )
                or not side_variants(
                    team_b
                )
                or side_team_identity(
                    team_a
                )
                == side_team_identity(
                    team_b
                )
            ):
                incomplete_ids.add(
                    native_game_id
                )
                continue

            game = {
                "game_date_key": (
                    date_key
                ),
                "team_a": team_a,
                "team_b": team_b,
            }

            existing = (
                stats_games.get(
                    native_game_id
                )
            )

            if existing is None:
                stats_games[
                    native_game_id
                ] = game

                continue

            same_order = (
                existing[
                    "game_date_key"
                ]
                == date_key
                and side_identity(
                    existing[
                        "team_a"
                    ]
                )
                == side_identity(
                    team_a
                )
                and side_identity(
                    existing[
                        "team_b"
                    ]
                )
                == side_identity(
                    team_b
                )
            )

            reverse_order = (
                existing[
                    "game_date_key"
                ]
                == date_key
                and side_identity(
                    existing[
                        "team_a"
                    ]
                )
                == side_identity(
                    team_b
                )
                and side_identity(
                    existing[
                        "team_b"
                    ]
                )
                == side_identity(
                    team_a
                )
            )

            if (
                same_order
                or reverse_order
            ):
                exact_duplicate_rows += 1
            else:
                conflicting_ids.add(
                    native_game_id
                )

    else:
        schema_used = "team_rows"

        selected = [
            "game_id",
            "game_date",
            "team_name",
            "team_abbreviation",
        ]

        if (
            "pts"
            in columns
        ):
            selected.append(
                "pts"
            )

        raw_games: dict[
            str,
            dict[str, Any],
        ] = {}

        for row in (
            stats_schedule
            .select(
                selected
            )
            .iter_rows(
                named=True
            )
        ):
            native_game_id = clean(
                row.get(
                    "game_id"
                )
            )

            if not native_game_id:
                continue

            date_key = (
                core
                .normalize_legacy_schedule_date(
                    row.get(
                        "game_date"
                    )
                )
            )

            if not date_key:
                bad_date_ids.add(
                    native_game_id
                )
                continue

            side = side_payload(
                team_name=row.get(
                    "team_name"
                ),
                team_abbreviation=row.get(
                    "team_abbreviation"
                ),
                pts=row.get(
                    "pts"
                ),
            )

            if not side_variants(
                side
            ):
                incomplete_ids.add(
                    native_game_id
                )
                continue

            raw_game = (
                raw_games.setdefault(
                    native_game_id,
                    {
                        "game_date_key": (
                            date_key
                        ),
                        "teams": [],
                    },
                )
            )

            if (
                raw_game[
                    "game_date_key"
                ]
                != date_key
            ):
                conflicting_ids.add(
                    native_game_id
                )
                continue

            duplicate_found = False
            team_conflict_found = False

            for existing_side in (
                raw_game[
                    "teams"
                ]
            ):
                if (
                    side_team_identity(
                        existing_side
                    )
                    != side_team_identity(
                        side
                    )
                ):
                    continue

                if (
                    side_identity(
                        existing_side
                    )
                    == side_identity(
                        side
                    )
                ):
                    duplicate_found = True
                else:
                    team_conflict_found = True

                break

            if duplicate_found:
                exact_duplicate_rows += 1
                continue

            if team_conflict_found:
                conflicting_ids.add(
                    native_game_id
                )
                continue

            raw_game[
                "teams"
            ].append(
                side
            )

        for (
            native_game_id,
            raw_game,
        ) in raw_games.items():
            if (
                native_game_id
                in conflicting_ids
            ):
                continue

            teams = raw_game[
                "teams"
            ]

            if (
                len(
                    teams
                )
                != 2
                or side_team_identity(
                    teams[0]
                )
                == side_team_identity(
                    teams[1]
                )
            ):
                incomplete_ids.add(
                    native_game_id
                )
                continue

            stats_games[
                native_game_id
            ] = {
                "game_date_key": (
                    raw_game[
                        "game_date_key"
                    ]
                ),
                "team_a": teams[0],
                "team_b": teams[1],
            }

    if bad_date_ids:
        raise RuntimeError(
            "WNBA Stats schedule has "
            "unparseable game_date values "
            "for game_ids="
            f"{sorted(bad_date_ids)[:20]}"
        )

    if conflicting_ids:
        raise RuntimeError(
            "WNBA Stats schedule has "
            "conflicting duplicate rows "
            "for game_ids="
            f"{sorted(conflicting_ids)[:20]}"
        )

    if incomplete_ids:
        raise RuntimeError(
            "WNBA Stats schedule cannot "
            "identify exactly two unique teams "
            "for game_ids="
            f"{sorted(incomplete_ids)[:20]}"
        )

    if not stats_games:
        raise RuntimeError(
            "WNBA Stats schedule produced "
            "zero usable games for "
            f"season={sdv_season}"
        )

    root = core.storage_root(
        cfg
    )

    games_file = core.table_path(
        root,
        "wnba",
        internal_season,
        "games",
    )

    if not games_file.exists():
        raise RuntimeError(
            "WNBA canonical games file "
            f"missing: {games_file}"
        )

    games = P.read_parquet(
        games_file
    )

    required_games = {
        "game_id",
        "game_date",
    }

    missing_games = sorted(
        required_games
        - set(
            games.columns
        )
    )

    if missing_games:
        raise RuntimeError(
            "WNBA games.parquet "
            f"missing columns={missing_games}"
        )

    home_name_columns = [
        column
        for column
        in (
            "home_display_name",
            "home_name",
            "home_short_display_name",
            "home_location",
            "home_abbreviation",
        )
        if column
        in games.columns
    ]

    away_name_columns = [
        column
        for column
        in (
            "away_display_name",
            "away_name",
            "away_short_display_name",
            "away_location",
            "away_abbreviation",
        )
        if column
        in games.columns
    ]

    if (
        not home_name_columns
        or not away_name_columns
    ):
        raise RuntimeError(
            "WNBA games.parquet does not "
            "expose usable home/away "
            "team-name columns"
        )

    canonical_columns = [
        "game_id",
        "game_date",
        *home_name_columns,
        *away_name_columns,
    ]

    if (
        "home_score"
        in games.columns
    ):
        canonical_columns.append(
            "home_score"
        )

    if (
        "away_score"
        in games.columns
    ):
        canonical_columns.append(
            "away_score"
        )

    by_date: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = {}

    for row in (
        games
        .select(
            canonical_columns
        )
        .iter_rows(
            named=True
        )
    ):
        date_key = (
            core
            .normalize_legacy_schedule_date(
                row.get(
                    "game_date"
                )
            )
        )

        if not date_key:
            continue

        by_date.setdefault(
            date_key,
            [],
        ).append(
            row
        )

    def candidate_variants(
        candidate: dict[str, Any],
        name_columns: list[str],
    ) -> set[str]:
        variants: set[str] = set()

        for column in name_columns:
            variants.update(
                core.team_variants(
                    candidate.get(
                        column
                    )
                )
            )

        return variants

    def orientations(
        game: dict[str, Any],
        candidate: dict[str, Any],
    ) -> set[str]:
        team_a_variants = (
            side_variants(
                game[
                    "team_a"
                ]
            )
        )

        team_b_variants = (
            side_variants(
                game[
                    "team_b"
                ]
            )
        )

        home_variants = (
            candidate_variants(
                candidate,
                home_name_columns,
            )
        )

        away_variants = (
            candidate_variants(
                candidate,
                away_name_columns,
            )
        )

        result: set[str] = set()

        if (
            team_a_variants
            & home_variants
            and team_b_variants
            & away_variants
        ):
            result.add(
                "a_home"
            )

        if (
            team_a_variants
            & away_variants
            and team_b_variants
            & home_variants
        ):
            result.add(
                "a_away"
            )

        return result

    mappings: list[
        dict[str, str]
    ] = []

    unmatched: list[str] = []
    ambiguous: list[str] = []

    for native_game_id in sorted(
        stats_games
    ):
        game = stats_games[
            native_game_id
        ]

        match_records = []

        for candidate in (
            by_date.get(
                game[
                    "game_date_key"
                ],
                [],
            )
        ):
            candidate_orientations = (
                orientations(
                    game,
                    candidate,
                )
            )

            if candidate_orientations:
                match_records.append(
                    {
                        "candidate": (
                            candidate
                        ),
                        "orientations": (
                            candidate_orientations
                        ),
                    }
                )

        chosen = None
        method = ""

        if (
            len(
                match_records
            )
            == 1
        ):
            chosen = (
                match_records[
                    0
                ][
                    "candidate"
                ]
            )

            method = (
                "date_unordered_team_pair_"
                "canonical_home_away"
            )

        elif (
            len(
                match_records
            )
            > 1
        ):
            score_matches = []

            team_a_pts = (
                game[
                    "team_a"
                ].get(
                    "pts"
                )
            )

            team_b_pts = (
                game[
                    "team_b"
                ].get(
                    "pts"
                )
            )

            if (
                team_a_pts is not None
                and team_b_pts is not None
            ):
                for record in (
                    match_records
                ):
                    candidate = (
                        record[
                            "candidate"
                        ]
                    )

                    try:
                        home_score = float(
                            candidate.get(
                                "home_score"
                            )
                        )

                        away_score = float(
                            candidate.get(
                                "away_score"
                            )
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    score_match = False

                    if (
                        "a_home"
                        in record[
                            "orientations"
                        ]
                        and home_score
                        == team_a_pts
                        and away_score
                        == team_b_pts
                    ):
                        score_match = True

                    if (
                        "a_away"
                        in record[
                            "orientations"
                        ]
                        and home_score
                        == team_b_pts
                        and away_score
                        == team_a_pts
                    ):
                        score_match = True

                    if score_match:
                        score_matches.append(
                            record
                        )

            if (
                len(
                    score_matches
                )
                == 1
            ):
                chosen = (
                    score_matches[
                        0
                    ][
                        "candidate"
                    ]
                )

                method = (
                    "date_unordered_team_pair_"
                    "score_canonical_home_away"
                )

        if chosen is None:
            if (
                len(
                    match_records
                )
                > 1
            ):
                ambiguous.append(
                    native_game_id
                )
            else:
                unmatched.append(
                    native_game_id
                )

            continue

        espn_game_id = clean(
            chosen.get(
                "game_id"
            )
        )

        if not espn_game_id:
            unmatched.append(
                native_game_id
            )
            continue

        mappings.append(
            {
                "wnba_game_id": (
                    native_game_id
                ),
                "espn_game_id": (
                    espn_game_id
                ),
                "match_method": (
                    method
                ),
            }
        )

    if not mappings:
        raise RuntimeError(
            "WNBA historical Stats->ESPN "
            "crosswalk produced zero "
            "mapped games"
        )

    xwalk = P.DataFrame(
        mappings
    )

    duplicate_native = (
        xwalk
        .group_by(
            "wnba_game_id"
        )
        .agg(
            P.col(
                "espn_game_id"
            )
            .n_unique()
            .alias(
                "n"
            )
        )
        .filter(
            P.col(
                "n"
            )
            > 1
        )
    )

    duplicate_espn = (
        xwalk
        .group_by(
            "espn_game_id"
        )
        .agg(
            P.col(
                "wnba_game_id"
            )
            .n_unique()
            .alias(
                "n"
            )
        )
        .filter(
            P.col(
                "n"
            )
            > 1
        )
    )

    if (
        duplicate_native.height
        or duplicate_espn.height
    ):
        raise RuntimeError(
            "WNBA historical Stats->ESPN "
            "crosswalk is not one-to-one"
        )

    for game_id in sorted(
        unmatched
    ):
        core.log(
            "WNBA GAME UNMAPPED | "
            f"wnba_game_id={game_id} "
            "reason=no_unique_date_"
            "unordered_team_match"
        )

    for game_id in sorted(
        ambiguous
    ):
        core.log(
            "WNBA GAME AMBIGUOUS | "
            f"wnba_game_id={game_id} "
            "reason=multiple_date_"
            "unordered_team_matches"
        )

    core.log(
        "WNBA LEGACY GAME CROSSWALK | "
        f"internal={internal_season} "
        f"sdv={sdv_season} "
        f"schema={schema_used} "
        f"stats_rows={stats_schedule.height} "
        f"stats_games={len(stats_games)} "
        f"mapped={xwalk.height} "
        f"unmatched={len(unmatched)} "
        f"ambiguous={len(ambiguous)} "
        f"exact_duplicate_rows_ignored="
        f"{exact_duplicate_rows} "
        f"source={stats_source}"
    )

    return (
        xwalk.select(
            "wnba_game_id",
            "espn_game_id",
        ),
        (
            f"{stats_source} -> "
            "deterministic game_date/"
            "unordered-team-pair match "
            "with canonical ESPN "
            "home-away orientation"
        ),
        "wnba_game_id",
    )


def resolve_seasons(
    cfg: dict[str, Any],
    requested: list[int] | None,
) -> list[int]:
    configured = sorted(
        {
            int(
                value
            )
            for value
            in cfg[
                "historical_internal_seasons"
            ][
                "wnba"
            ]
        }
    )

    if not requested:
        return configured

    requested_set = {
        int(
            value
        )
        for value
        in requested
    }

    invalid = sorted(
        requested_set
        - set(
            configured
        )
    )

    if invalid:
        raise ValueError(
            "Requested WNBA internal season "
            "is not configured as historical: "
            f"{invalid}"
        )

    return [
        season
        for season
        in configured
        if season
        in requested_set
    ]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--internal-season",
        action="append",
        type=int,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--validate-config",
        action="store_true",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=core.CONFIG_PATH,
    )

    args = parser.parse_args()

    core.ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Redirect all shared logging for this process
    # into the WNBA-specific log.
    core.LOG_FILE = LOG_FILE

    LOG_FILE.write_text(
        (
            "=== SDV WNBA HISTORICAL STORAGE "
            f"{datetime.now(timezone.utc).isoformat()} "
            "===\n"
        ),
        encoding="utf-8",
    )

    try:
        cfg = core.load_config(
            args.config
        )

        version = core.verify_version(
            cfg
        )

        seasons = resolve_seasons(
            cfg,
            args.internal_season,
        )

        mappings = {
            season: int(
                core.sdv_season_id(
                    "wnba",
                    season,
                    config_path=(
                        core.SEASON_CONFIG_PATH
                    ),
                )
            )
            for season
            in seasons
        }

        core.log(
            "CONFIG VALID | "
            f"sportsdataverse={version} "
            "league=wnba "
            f"internal_seasons={seasons} "
            f"mappings={mappings}"
        )

        if args.validate_config:
            print(
                "SDV WNBA historical storage "
                "config valid."
            )
            return

        # Process-local replacement only.
        # sdv_historical_storage.py on disk is untouched.
        core.build_wnba_legacy_schedule_crosswalk = (
            build_wnba_legacy_schedule_crosswalk
        )

        manifests = [
            core.build_season(
                cfg,
                "wnba",
                season,
                args.force,
            )
            for season
            in seasons
        ]

        core.log(
            "STATUS: SUCCESS | "
            f"manifests={len(manifests)}"
        )

        print(
            "SDV WNBA historical basketball "
            "storage complete."
        )

    except Exception as exc:
        core.log(
            f"FATAL: {exc}"
        )

        core.log(
            traceback
            .format_exc()
            .rstrip()
        )

        core.log(
            "STATUS: FAILED"
        )

        print(
            "SDV WNBA historical storage failed: "
            f"{exc}"
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
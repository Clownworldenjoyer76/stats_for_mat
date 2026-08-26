#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_feature_generation.py
"""Build strict point-in-time SportsDataVerse basketball features."""
from __future__ import annotations

import argparse
import csv
import math
import re
import traceback
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import yaml


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_model.yaml"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_feature_generation.txt"
NY_TZ = ZoneInfo("America/New_York")

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

TEAM_METRICS = {
    "points_for": (
        "team_score",
    ),
    "points_against": (
        "opponent_team_score",
    ),
    "fg_pct": (
        "field_goal_pct",
    ),
    "three_pct": (
        "three_point_field_goal_pct",
    ),
    "ft_pct": (
        "free_throw_pct",
    ),
    "rebounds": (
        "total_rebounds",
        "rebounds",
    ),
    "assists": (
        "assists",
    ),
    "turnovers": (
        "total_turnovers",
        "turnovers",
    ),
}

PLAYER_STAT_COLUMNS = {
    "points": (
        "points",
    ),
    "rebounds": (
        "rebounds",
    ),
    "assists": (
        "assists",
    ),
    "steals": (
        "steals",
    ),
    "blocks": (
        "blocks",
    ),
    "turnovers": (
        "turnovers",
    ),
    "minutes": (
        "minutes",
    ),
}


def clean(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(
        value
    ).strip()


def clean_id(
    value: Any,
) -> str:
    text = clean(
        value
    )

    if not text:
        return ""

    try:
        number = float(
            text
        )

        if (
            math.isfinite(
                number
            )
            and number.is_integer()
        ):
            return str(
                int(
                    number
                )
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return text


def to_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return float(
            value
        )

    text = clean(
        value
    )

    if not text:
        return None

    try:
        number = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def log(
    message: str,
) -> None:
    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{message}\n"
        )


def read_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    payload = (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "YAML root must be a mapping: "
            f"{path}"
        )

    return payload


def required_mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = parent.get(
        key
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "sdv_model.yaml missing mapping: "
            f"{key}"
        )

    return value


def configured_paths(
    cfg: dict[str, Any],
) -> dict[str, Path]:
    section = required_mapping(
        cfg,
        "paths",
    )

    required = (
        "history_input_root",
        "history_output_root",
        "current_output_root",
        "canonical_current_root",
    )

    result: dict[
        str,
        Path,
    ] = {}

    for key in required:
        value = clean(
            section.get(
                key
            )
        )

        if not value:
            raise ValueError(
                "sdv_model.yaml paths."
                f"{key} is blank"
            )

        result[
            key
        ] = Path(
            value
        )

    return result


def validate_config(
    cfg: dict[str, Any],
) -> None:
    if (
        int(
            cfg.get(
                "schema_version",
                0,
            )
        )
        != 1
    ):
        raise ValueError(
            "sdv_model.yaml schema_version must be 1"
        )

    if not clean(
        cfg.get(
            "feature_version"
        )
    ):
        raise ValueError(
            "sdv_model.yaml feature_version is blank"
        )

    windows = required_mapping(
        cfg,
        "feature_windows",
    )

    team_windows = windows.get(
        "team_games"
    )

    if (
        not isinstance(
            team_windows,
            list,
        )
        or not team_windows
    ):
        raise ValueError(
            "feature_windows.team_games "
            "must be a non-empty list"
        )

    normalized = [
        int(
            value
        )
        for value
        in team_windows
    ]

    if (
        any(
            value <= 0
            for value
            in normalized
        )
        or len(
            set(
                normalized
            )
        )
        != len(
            normalized
        )
    ):
        raise ValueError(
            "feature_windows.team_games "
            "must contain unique positive integers"
        )

    if (
        int(
            windows.get(
                "player_team_games",
                0,
            )
        )
        <= 0
    ):
        raise ValueError(
            "feature_windows.player_team_games "
            "must be positive"
        )

    if (
        int(
            windows.get(
                "player_top_n",
                0,
            )
        )
        <= 0
    ):
        raise ValueError(
            "feature_windows.player_top_n "
            "must be positive"
        )

    shrinkage = required_mapping(
        cfg,
        "shrinkage",
    )

    if (
        float(
            shrinkage.get(
                "team_pseudo_games",
                -1,
            )
        )
        < 0
    ):
        raise ValueError(
            "shrinkage.team_pseudo_games "
            "must be >= 0"
        )

    if (
        float(
            shrinkage.get(
                "player_pseudo_games",
                -1,
            )
        )
        < 0
    ):
        raise ValueError(
            "shrinkage.player_pseudo_games "
            "must be >= 0"
        )

    required_mapping(
        cfg,
        "player_strength",
    )

    required_mapping(
        cfg,
        "model_inputs",
    )

    configured_paths(
        cfg
    )


def parse_date(
    value: Any,
) -> date | None:
    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    text = clean(
        value
    )

    if not text:
        return None

    text = (
        text[
            :10
        ]
        .replace(
            "_",
            "-",
        )
    )

    try:
        return date.fromisoformat(
            text
        )

    except ValueError:
        return None


def parse_datetime(
    value: Any,
) -> datetime | None:
    if isinstance(
        value,
        datetime,
    ):
        parsed = value

    else:
        text = clean(
            value
        )

        if not text:
            return None

        try:
            parsed = datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            )

        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def canonical_datetime(
    game_date: Any,
    game_time: Any,
) -> datetime | None:
    parsed_date = parse_date(
        game_date
    )

    time_text = clean(
        game_time
    )

    if (
        parsed_date is None
        or not time_text
    ):
        return None

    for fmt in (
        "%I:%M %p",
        "%H:%M",
        "%I:%M:%S %p",
        "%H:%M:%S",
    ):
        try:
            parsed_time = (
                datetime.strptime(
                    time_text,
                    fmt,
                )
                .time()
            )

            local = datetime.combine(
                parsed_date,
                parsed_time,
                tzinfo=NY_TZ,
            )

            return local.astimezone(
                timezone.utc
            )

        except ValueError:
            continue

    return None


def normalize_game_date(
    value: Any,
) -> str:
    parsed = parse_date(
        value
    )

    if parsed is None:
        return ""

    return parsed.strftime(
        "%Y_%m_%d"
    )


def normalize_neutral(
    value: Any,
) -> int | None:
    if isinstance(
        value,
        bool,
    ):
        return (
            1
            if value
            else 0
        )

    text = clean(
        value
    ).lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return 1

    if text in {
        "false",
        "0",
        "no",
        "n",
    }:
        return 0

    return None


def first_value(
    row: dict[str, Any],
    columns: tuple[str, ...],
) -> Any:
    for column in columns:
        if (
            column
            in row
            and clean(
                row.get(
                    column
                )
            )
            != ""
        ):
            return row.get(
                column
            )

    return None


def row_game_id(
    row: dict[str, Any],
) -> str:
    return clean_id(
        row.get(
            "game_id"
        )
        or row.get(
            "id"
        )
    )


def row_team_id(
    row: dict[str, Any],
) -> str:
    return clean_id(
        row.get(
            "team_id"
        )
    )


def row_game_datetime(
    row: dict[str, Any],
) -> datetime | None:
    value = row.get(
        "game_date_time"
    )

    parsed = parse_datetime(
        value
    )

    if parsed is not None:
        return parsed

    for column in (
        "date",
        "start_date",
    ):
        text = clean(
            row.get(
                column
            )
        )

        if (
            not text
            or not (
                "T"
                in text
                or re.search(
                    r"\d{1,2}:\d{2}",
                    text,
                )
            )
        ):
            continue

        parsed = parse_datetime(
            text
        )

        if parsed is not None:
            return parsed

    return None


def row_game_date(
    row: dict[str, Any],
) -> date | None:
    for column in (
        "game_date",
        "game_date_time",
        "date",
        "start_date",
    ):
        parsed = parse_date(
            row.get(
                column
            )
        )

        if parsed is not None:
            return parsed

    return None


def target_time(
    row: dict[str, Any],
    *,
    canonical: bool,
) -> tuple[
    datetime | None,
    date,
]:
    target_date = row_game_date(
        row
    )

    if target_date is None:
        raise ValueError(
            "Target game_id="
            f"{row_game_id(row)} "
            "has no parseable game date"
        )

    if canonical:
        target_dt = canonical_datetime(
            row.get(
                "game_date"
            ),
            row.get(
                "game_time"
            ),
        )

    else:
        target_dt = row_game_datetime(
            row
        )

    return (
        target_dt,
        target_date,
    )


def source_is_prior(
    source: dict[str, Any],
    target_dt: datetime | None,
    target_date: date,
) -> bool:
    source_date = source.get(
        "_date"
    )

    source_dt = source.get(
        "_dt"
    )

    if not isinstance(
        source_date,
        date,
    ):
        return False

    if (
        target_dt is not None
        and isinstance(
            source_dt,
            datetime,
        )
    ):
        return (
            source_dt
            < target_dt
        )

    return (
        source_date
        < target_date
    )


def assert_target_absent(
    rows: list[
        dict[str, Any]
    ],
    target_game_id: str,
    source_name: str,
) -> None:
    leaked = [
        row_game_id(
            row
        )
        for row
        in rows
        if row_game_id(
            row
        )
        == target_game_id
    ]

    if leaked:
        raise RuntimeError(
            "LEAKAGE CHECK FAILED | "
            f"target_game_id={target_game_id} "
            f"source={source_name} "
            f"rows={len(leaked)}"
        )


def history_seasons(
    root: Path,
    league: str,
) -> list[int]:
    league_root = (
        root
        / league
    )

    if not league_root.exists():
        raise FileNotFoundError(
            league_root
        )

    seasons = sorted(
        int(
            path.name
        )
        for path
        in league_root.iterdir()
        if (
            path.is_dir()
            and path.name.isdigit()
        )
    )

    if not seasons:
        raise RuntimeError(
            "No historical seasons found "
            f"for {league}: {league_root}"
        )

    return seasons


def read_history_table(
    root: Path,
    league: str,
    seasons: list[int],
    table: str,
) -> list[
    dict[str, Any]
]:
    rows: list[
        dict[str, Any]
    ] = []

    for season in seasons:
        path = (
            root
            / league
            / str(
                season
            )
            / f"{table}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        rows.extend(
            pl.read_parquet(
                path
            )
            .to_dicts()
        )

    return rows


def prepare_source_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    prepared = dict(
        row
    )

    prepared[
        "_dt"
    ] = row_game_datetime(
        prepared
    )

    prepared[
        "_date"
    ] = row_game_date(
        prepared
    )

    return prepared


def source_sort_key(
    row: dict[str, Any],
) -> tuple[
    date,
    datetime,
]:
    source_date = row.get(
        "_date"
    )

    source_dt = row.get(
        "_dt"
    )

    fallback_date = (
        source_date
        if isinstance(
            source_date,
            date,
        )
        else date.min
    )

    fallback_dt = (
        source_dt
        if isinstance(
            source_dt,
            datetime,
        )
        else datetime.min.replace(
            tzinfo=timezone.utc
        )
    )

    return (
        fallback_date,
        fallback_dt,
    )


def build_team_index(
    rows: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for raw in rows:
        team_id = row_team_id(
            raw
        )

        game_id = row_game_id(
            raw
        )

        if (
            not team_id
            or not game_id
        ):
            continue

        row = prepare_source_row(
            raw
        )

        if row[
            "_date"
        ] is None:
            continue

        index[
            team_id
        ].append(
            row
        )

    for team_rows in index.values():
        team_rows.sort(
            key=source_sort_key
        )

    return dict(
        index
    )


def build_player_game_index(
    rows: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    grouped: dict[
        tuple[
            str,
            str,
        ],
        dict[str, Any],
    ] = {}

    for raw in rows:
        team_id = row_team_id(
            raw
        )

        game_id = row_game_id(
            raw
        )

        if (
            not team_id
            or not game_id
        ):
            continue

        key = (
            team_id,
            game_id,
        )

        group = grouped.get(
            key
        )

        if group is None:
            prepared = prepare_source_row(
                raw
            )

            if prepared[
                "_date"
            ] is None:
                continue

            group = {
                "team_id": (
                    team_id
                ),
                "game_id": (
                    game_id
                ),
                "_dt": prepared[
                    "_dt"
                ],
                "_date": prepared[
                    "_date"
                ],
                "players": [],
            }

            grouped[
                key
            ] = group

        group[
            "players"
        ].append(
            raw
        )

    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for group in grouped.values():
        index[
            group[
                "team_id"
            ]
        ].append(
            group
        )

    for games in index.values():
        games.sort(
            key=source_sort_key
        )

    return dict(
        index
    )


def build_home_venue_index(
    rows: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for raw in rows:
        home_team_id = clean_id(
            raw.get(
                "home_team_id"
            )
            or raw.get(
                "home_id"
            )
        )

        game_id = row_game_id(
            raw
        )

        if (
            not home_team_id
            or not game_id
        ):
            continue

        row = prepare_source_row(
            raw
        )

        if row[
            "_date"
        ] is None:
            continue

        index[
            home_team_id
        ].append(
            row
        )

    for games in index.values():
        games.sort(
            key=source_sort_key
        )

    return dict(
        index
    )


def prior_rows(
    rows: list[
        dict[str, Any]
    ],
    target_dt: datetime | None,
    target_date: date,
) -> list[
    dict[str, Any]
]:
    return [
        row
        for row
        in rows
        if source_is_prior(
            row,
            target_dt,
            target_date,
        )
    ]


def mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return (
        sum(
            values
        )
        / len(
            values
        )
    )


def shrunk_mean(
    window_values: list[float],
    baseline_values: list[float],
    pseudo_games: float,
) -> float | None:
    window_mean = mean(
        window_values
    )

    if window_mean is None:
        return None

    baseline = mean(
        baseline_values
    )

    if (
        baseline is None
        or pseudo_games <= 0
    ):
        return window_mean

    n = float(
        len(
            window_values
        )
    )

    return (
        (
            n
            * window_mean
        )
        + (
            pseudo_games
            * baseline
        )
    ) / (
        n
        + pseudo_games
    )


def team_metric_value(
    row: dict[str, Any],
    metric: str,
) -> float | None:
    if metric == "margin":
        points_for = to_float(
            first_value(
                row,
                TEAM_METRICS[
                    "points_for"
                ],
            )
        )

        points_against = to_float(
            first_value(
                row,
                TEAM_METRICS[
                    "points_against"
                ],
            )
        )

        if (
            points_for is None
            or points_against is None
        ):
            return None

        return (
            points_for
            - points_against
        )

    columns = TEAM_METRICS[
        metric
    ]

    return to_float(
        first_value(
            row,
            columns,
        )
    )


def team_features(
    team_id: str,
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    windows: list[int],
    pseudo_games: float,
    side: str,
) -> dict[str, Any]:
    eligible = prior_rows(
        index.get(
            team_id,
            [],
        ),
        target_dt,
        target_date,
    )

    assert_target_absent(
        eligible,
        target_game_id,
        f"team_game:{side}",
    )

    features: dict[
        str,
        Any,
    ] = {
        f"{side}_games_prior": (
            len(
                eligible
            )
        ),
        f"{side}_days_since_last_game": (
            None
        ),
    }

    if eligible:
        last_date = eligible[
            -1
        ].get(
            "_date"
        )

        if isinstance(
            last_date,
            date,
        ):
            features[
                f"{side}_days_since_last_game"
            ] = (
                target_date
                - last_date
            ).days

    metrics = [
        *TEAM_METRICS.keys(),
        "margin",
    ]

    for window in windows:
        recent = eligible[
            -window:
        ]

        features[
            f"{side}_games_used_{window}"
        ] = len(
            recent
        )

        for metric in metrics:
            recent_values: list[
                float
            ] = []

            for row in recent:
                value = team_metric_value(
                    row,
                    metric,
                )

                if value is not None:
                    recent_values.append(
                        value
                    )

            baseline_values: list[
                float
            ] = []

            for row in eligible:
                value = team_metric_value(
                    row,
                    metric,
                )

                if value is not None:
                    baseline_values.append(
                        value
                    )

            features[
                f"{side}_{metric}_{window}"
            ] = shrunk_mean(
                recent_values,
                baseline_values,
                pseudo_games,
            )

    return features


def player_contribution(
    row: dict[str, Any],
    weights: dict[
        str,
        float,
    ],
) -> float:
    total = 0.0

    for (
        stat,
        weight,
    ) in weights.items():
        if stat == "minutes":
            continue

        columns = PLAYER_STAT_COLUMNS.get(
            stat
        )

        if columns is None:
            raise ValueError(
                "Unsupported player_strength "
                f"weight: {stat}"
            )

        value = (
            to_float(
                first_value(
                    row,
                    columns,
                )
            )
            or 0.0
        )

        total += (
            float(
                weight
            )
            * value
        )

    return total


def player_features(
    team_id: str,
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    team_game_window: int,
    top_n: int,
    pseudo_games: float,
    weights: dict[
        str,
        float,
    ],
    side: str,
) -> dict[str, Any]:
    eligible_games = prior_rows(
        index.get(
            team_id,
            [],
        ),
        target_dt,
        target_date,
    )

    assert_target_absent(
        eligible_games,
        target_game_id,
        f"player_game:{side}",
    )

    recent_games = eligible_games[
        -team_game_window:
    ]

    recent_rows = [
        player
        for game
        in recent_games
        for player
        in game[
            "players"
        ]
    ]

    assert_target_absent(
        recent_rows,
        target_game_id,
        f"player_game_rows:{side}",
    )

    player_rows: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in recent_rows:
        player_id = clean_id(
            row.get(
                "player_id"
            )
            or row.get(
                "athlete_id"
            )
        )

        if player_id:
            player_rows[
                player_id
            ].append(
                row
            )

    all_contributions = [
        player_contribution(
            row,
            weights,
        )
        for row
        in recent_rows
    ]

    baseline = mean(
        all_contributions
    )

    summaries: list[
        dict[
            str,
            float,
        ]
    ] = []

    for rows in player_rows.values():
        contributions = [
            player_contribution(
                row,
                weights,
            )
            for row
            in rows
        ]

        minutes: list[
            float
        ] = []

        for row in rows:
            value = to_float(
                first_value(
                    row,
                    PLAYER_STAT_COLUMNS[
                        "minutes"
                    ],
                )
            )

            if value is not None:
                minutes.append(
                    value
                )

        raw_strength = (
            mean(
                contributions
            )
            or 0.0
        )

        games = len(
            rows
        )

        if (
            baseline is not None
            and pseudo_games > 0
        ):
            strength = (
                (
                    games
                    * raw_strength
                )
                + (
                    pseudo_games
                    * baseline
                )
            ) / (
                games
                + pseudo_games
            )

        else:
            strength = (
                raw_strength
            )

        summaries.append(
            {
                "strength": (
                    strength
                ),
                "minutes": (
                    mean(
                        minutes
                    )
                    or 0.0
                ),
                "recent_minutes": (
                    sum(
                        minutes
                    )
                ),
            }
        )

    summaries.sort(
        key=lambda item: (
            item[
                "recent_minutes"
            ],
            item[
                "strength"
            ],
        ),
        reverse=True,
    )

    top = summaries[
        :top_n
    ]

    return {
        f"{side}_player_games_used": (
            len(
                recent_games
            )
        ),
        f"{side}_player_recent_count": (
            len(
                summaries
            )
        ),
        f"{side}_player_strength": (
            sum(
                item[
                    "strength"
                ]
                for item
                in top
            )
            if top
            else None
        ),
        f"{side}_player_minutes": (
            sum(
                item[
                    "minutes"
                ]
                for item
                in top
            )
            if top
            else None
        ),
    }


def home_court_indicator(
    target: dict[str, Any],
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    home_venue_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
) -> int | None:
    neutral = normalize_neutral(
        target.get(
            "neutral_site"
        )
    )

    if neutral == 1:
        return 0

    if neutral is None:
        return None

    venue_id = clean_id(
        target.get(
            "venue_id"
        )
    )

    if not venue_id:
        return None

    direct_home_venue_id = clean_id(
        target.get(
            "home_venue_id"
        )
    )

    if direct_home_venue_id:
        return (
            1
            if (
                venue_id
                == direct_home_venue_id
            )
            else 0
        )

    home_team_id = clean_id(
        target.get(
            "home_team_id"
        )
        or target.get(
            "home_id"
        )
    )

    if not home_team_id:
        return None

    eligible = prior_rows(
        home_venue_index.get(
            home_team_id,
            [],
        ),
        target_dt,
        target_date,
    )

    assert_target_absent(
        eligible,
        target_game_id,
        "games:home_venue",
    )

    prior_home_venues = {
        clean_id(
            row.get(
                "venue_id"
            )
        )
        for row
        in eligible
        if (
            normalize_neutral(
                row.get(
                    "neutral_site"
                )
            )
            == 0
            and clean_id(
                row.get(
                    "venue_id"
                )
            )
        )
    }

    if not prior_home_venues:
        return None

    return (
        1
        if venue_id
        in prior_home_venues
        else 0
    )


def venue_name(
    target: dict[str, Any],
) -> str:
    text = clean(
        target.get(
            "venue_name"
        )
        or target.get(
            "venue_full_name"
        )
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def feature_row(
    target: dict[str, Any],
    *,
    canonical: bool,
    league: str,
    feature_version: str,
    generated_at: str,
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    player_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    home_venue_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    team_windows: list[int],
    player_game_window: int,
    player_top_n: int,
    team_pseudo_games: float,
    player_pseudo_games: float,
    player_weights: dict[
        str,
        float,
    ],
) -> dict[str, Any]:
    target_game_id = row_game_id(
        target
    )

    if not target_game_id:
        raise ValueError(
            "Target row has blank game_id"
        )

    (
        target_dt,
        target_date,
    ) = target_time(
        target,
        canonical=canonical,
    )

    home_team_id = clean_id(
        target.get(
            "home_team_id"
        )
        or target.get(
            "home_id"
        )
    )

    away_team_id = clean_id(
        target.get(
            "away_team_id"
        )
        or target.get(
            "away_id"
        )
    )

    if (
        not home_team_id
        or not away_team_id
    ):
        raise ValueError(
            "Target game_id="
            f"{target_game_id} "
            "has blank home/away team id"
        )

    is_neutral = normalize_neutral(
        target.get(
            "neutral_site"
        )
    )

    indicator = home_court_indicator(
        target,
        target_game_id,
        target_dt,
        target_date,
        home_venue_index,
    )

    if (
        is_neutral == 1
        and indicator != 0
    ):
        raise RuntimeError(
            "HOME COURT ASSERTION FAILED | "
            f"target_game_id={target_game_id} "
            "neutral_site=1 "
            "home_court_indicator="
            f"{indicator}"
        )

    result: dict[
        str,
        Any,
    ] = {
        "league": (
            LEAGUE_LABELS[
                league
            ]
        ),
        "internal_season": (
            int(
                clean_id(
                    target.get(
                        "internal_season"
                    )
                )
                or 0
            )
        ),
        "sdv_season": (
            int(
                clean_id(
                    target.get(
                        "sdv_season"
                    )
                    or target.get(
                        "season"
                    )
                )
                or 0
            )
        ),
        "game_id": (
            target_game_id
        ),
        "game_date": (
            target_date.strftime(
                "%Y_%m_%d"
            )
        ),
        "game_date_time_utc": (
            target_dt.isoformat()
            if target_dt
            else None
        ),
        "home_team_id": (
            home_team_id
        ),
        "away_team_id": (
            away_team_id
        ),
        "is_neutral_site": (
            is_neutral
        ),
        "home_court_indicator": (
            indicator
        ),
        "venue_id": (
            clean_id(
                target.get(
                    "venue_id"
                )
            )
        ),
        "venue_name": (
            venue_name(
                target
            )
        ),
        "feature_version": (
            feature_version
        ),
        "feature_generated_at_utc": (
            generated_at
        ),
    }

    result.update(
        team_features(
            home_team_id,
            target_game_id,
            target_dt,
            target_date,
            team_index,
            team_windows,
            team_pseudo_games,
            "home",
        )
    )

    result.update(
        team_features(
            away_team_id,
            target_game_id,
            target_dt,
            target_date,
            team_index,
            team_windows,
            team_pseudo_games,
            "away",
        )
    )

    result.update(
        player_features(
            home_team_id,
            target_game_id,
            target_dt,
            target_date,
            player_index,
            player_game_window,
            player_top_n,
            player_pseudo_games,
            player_weights,
            "home",
        )
    )

    result.update(
        player_features(
            away_team_id,
            target_game_id,
            target_dt,
            target_date,
            player_index,
            player_game_window,
            player_top_n,
            player_pseudo_games,
            player_weights,
            "away",
        )
    )

    return result


def validate_model_inputs(
    rows: list[
        dict[str, Any]
    ],
    cfg: dict[str, Any],
) -> None:
    if not rows:
        raise RuntimeError(
            "Feature generation produced zero rows"
        )

    inputs = required_mapping(
        cfg,
        "model_inputs",
    )

    configured: set[
        str
    ] = set()

    for key in (
        "categorical",
        "numeric",
    ):
        values = inputs.get(
            key,
            [],
        )

        if not isinstance(
            values,
            list,
        ):
            raise ValueError(
                "model_inputs."
                f"{key} must be a list"
            )

        configured.update(
            clean(
                value
            )
            for value
            in values
            if clean(
                value
            )
        )

    missing = sorted(
        configured
        - set(
            rows[
                0
            ]
        )
    )

    if missing:
        raise RuntimeError(
            "Configured model inputs "
            "missing from feature output: "
            f"{missing}"
        )


def write_features(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
    cfg: dict[str, Any],
) -> None:
    validate_model_inputs(
        rows,
        cfg,
    )

    game_ids = [
        clean_id(
            row.get(
                "game_id"
            )
        )
        for row
        in rows
    ]

    if any(
        not game_id
        for game_id
        in game_ids
    ):
        raise RuntimeError(
            "Feature output has blank game_id: "
            f"{path}"
        )

    if (
        len(
            game_ids
        )
        != len(
            set(
                game_ids
            )
        )
    ):
        raise RuntimeError(
            "Feature output has duplicate game_id: "
            f"{path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    if tmp.exists():
        tmp.unlink()

    try:
        frame = pl.DataFrame(
            rows,
            infer_schema_length=None,
            strict=False,
        )

        frame.write_parquet(
            tmp,
            compression="zstd",
        )

        tmp.replace(
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def canonical_target_rows(
    path: Path,
    internal_season: int,
) -> list[
    dict[str, Any]
]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = [
            dict(
                row
            )
            for row
            in csv.DictReader(
                handle
            )
        ]

    for row in rows:
        row[
            "internal_season"
        ] = (
            clean_id(
                row.get(
                    "internal_season"
                )
            )
            or str(
                internal_season
            )
        )

    return rows


def historical_target_rows(
    root: Path,
    league: str,
    internal_season: int,
) -> list[
    dict[str, Any]
]:
    path = (
        root
        / league
        / str(
            internal_season
        )
        / "games.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return (
        pl.read_parquet(
            path
        )
        .to_dicts()
    )


def build_indexes(
    history_root: Path,
    league: str,
) -> tuple[
    list[int],
    dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
]:
    seasons = history_seasons(
        history_root,
        league,
    )

    games = read_history_table(
        history_root,
        league,
        seasons,
        "games",
    )

    team_game = read_history_table(
        history_root,
        league,
        seasons,
        "team_game",
    )

    player_game = read_history_table(
        history_root,
        league,
        seasons,
        "player_game",
    )

    return (
        seasons,
        build_team_index(
            team_game
        ),
        build_player_game_index(
            player_game
        ),
        build_home_venue_index(
            games
        ),
    )


def generation_settings(
    cfg: dict[str, Any],
) -> tuple[
    list[int],
    int,
    int,
    float,
    float,
    dict[
        str,
        float,
    ],
]:
    windows_cfg = required_mapping(
        cfg,
        "feature_windows",
    )

    shrinkage = required_mapping(
        cfg,
        "shrinkage",
    )

    player_cfg = required_mapping(
        cfg,
        "player_strength",
    )

    weights_raw = player_cfg.get(
        "weights"
    )

    if (
        not isinstance(
            weights_raw,
            dict,
        )
        or not weights_raw
    ):
        raise ValueError(
            "player_strength.weights "
            "must be a non-empty mapping"
        )

    weights = {
        clean(
            key
        ): float(
            value
        )
        for (
            key,
            value,
        )
        in weights_raw.items()
    }

    return (
        sorted(
            int(
                value
            )
            for value
            in windows_cfg[
                "team_games"
            ]
        ),
        int(
            windows_cfg[
                "player_team_games"
            ]
        ),
        int(
            windows_cfg[
                "player_top_n"
            ]
        ),
        float(
            shrinkage[
                "team_pseudo_games"
            ]
        ),
        float(
            shrinkage[
                "player_pseudo_games"
            ]
        ),
        weights,
    )


def generate_rows(
    targets: list[
        dict[str, Any]
    ],
    *,
    canonical: bool,
    league: str,
    cfg: dict[str, Any],
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    player_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    home_venue_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    generated_at: str,
) -> list[
    dict[str, Any]
]:
    (
        team_windows,
        player_game_window,
        player_top_n,
        team_pseudo_games,
        player_pseudo_games,
        player_weights,
    ) = generation_settings(
        cfg
    )

    feature_version = clean(
        cfg[
            "feature_version"
        ]
    )

    rows = [
        feature_row(
            target,
            canonical=canonical,
            league=league,
            feature_version=(
                feature_version
            ),
            generated_at=(
                generated_at
            ),
            team_index=(
                team_index
            ),
            player_index=(
                player_index
            ),
            home_venue_index=(
                home_venue_index
            ),
            team_windows=(
                team_windows
            ),
            player_game_window=(
                player_game_window
            ),
            player_top_n=(
                player_top_n
            ),
            team_pseudo_games=(
                team_pseudo_games
            ),
            player_pseudo_games=(
                player_pseudo_games
            ),
            player_weights=(
                player_weights
            ),
        )
        for target
        in targets
    ]

    rows.sort(
        key=lambda row: (
            row[
                "game_date"
            ],
            row[
                "game_date_time_utc"
            ]
            or "",
            row[
                "game_id"
            ],
        )
    )

    return rows


def build_historical(
    cfg: dict[str, Any],
    league: str,
    internal_seasons: list[int] | None,
) -> list[Path]:
    paths = configured_paths(
        cfg
    )

    history_root = paths[
        "history_input_root"
    ]

    (
        available,
        team_index,
        player_index,
        home_venue_index,
    ) = build_indexes(
        history_root,
        league,
    )

    selected = sorted(
        set(
            internal_seasons
            or available
        )
    )

    invalid = sorted(
        set(
            selected
        )
        - set(
            available
        )
    )

    if invalid:
        raise ValueError(
            "Historical seasons unavailable "
            f"for {league}: {invalid}"
        )

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    outputs: list[
        Path
    ] = []

    label = LEAGUE_LABELS[
        league
    ]

    for season in selected:
        targets = historical_target_rows(
            history_root,
            league,
            season,
        )

        rows = generate_rows(
            targets,
            canonical=False,
            league=league,
            cfg=cfg,
            team_index=(
                team_index
            ),
            player_index=(
                player_index
            ),
            home_venue_index=(
                home_venue_index
            ),
            generated_at=(
                generated_at
            ),
        )

        output = (
            paths[
                "history_output_root"
            ]
            / league
            / (
                f"{season}_"
                f"{label}_features.parquet"
            )
        )

        write_features(
            output,
            rows,
            cfg,
        )

        outputs.append(
            output
        )

        log(
            "HISTORY READY | "
            f"league={label} "
            f"internal_season={season} "
            f"rows={len(rows)} "
            f"path={output}"
        )

    return outputs


def build_current(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
    game_dates: list[str] | None,
) -> list[Path]:
    paths = configured_paths(
        cfg
    )

    history_root = paths[
        "history_input_root"
    ]

    (
        _,
        team_index,
        player_index,
        home_venue_index,
    ) = build_indexes(
        history_root,
        league,
    )

    label = LEAGUE_LABELS[
        league
    ]

    canonical_path = (
        paths[
            "canonical_current_root"
        ]
        / league
        / (
            f"{internal_season}_"
            f"{label}_games.csv"
        )
    )

    targets = canonical_target_rows(
        canonical_path,
        internal_season,
    )

    requested_dates = {
        normalize_game_date(
            value
        )
        for value
        in (
            game_dates
            or []
        )
    }

    requested_dates.discard(
        ""
    )

    if requested_dates:
        targets = [
            row
            for row
            in targets
            if normalize_game_date(
                row.get(
                    "game_date"
                )
            )
            in requested_dates
        ]

    if not targets:
        raise RuntimeError(
            "No current/upcoming target games "
            f"selected from {canonical_path}"
        )

    grouped: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in targets:
        game_date = normalize_game_date(
            row.get(
                "game_date"
            )
        )

        if not game_date:
            raise ValueError(
                "Canonical target game_id="
                f"{row_game_id(row)} "
                "has invalid game_date"
            )

        grouped[
            game_date
        ].append(
            row
        )

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    outputs: list[
        Path
    ] = []

    for game_date in sorted(
        grouped
    ):
        rows = generate_rows(
            grouped[
                game_date
            ],
            canonical=True,
            league=league,
            cfg=cfg,
            team_index=(
                team_index
            ),
            player_index=(
                player_index
            ),
            home_venue_index=(
                home_venue_index
            ),
            generated_at=(
                generated_at
            ),
        )

        output = (
            paths[
                "current_output_root"
            ]
            / league
            / (
                f"{game_date}_"
                f"{label}_features.parquet"
            )
        )

        write_features(
            output,
            rows,
            cfg,
        )

        outputs.append(
            output
        )

        log(
            "CURRENT READY | "
            f"league={label} "
            f"internal_season={internal_season} "
            f"game_date={game_date} "
            f"rows={len(rows)} "
            f"path={output}"
        )

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict point-in-time "
            "SDV basketball features."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "history",
            "current",
        ),
        required=True,
    )

    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(
            LEAGUE_LABELS
        ),
    )

    parser.add_argument(
        "--internal-season",
        action="append",
        type=int,
    )

    parser.add_argument(
        "--game-date",
        action="append",
        help=(
            "Current mode only; "
            "YYYY_MM_DD or YYYY-MM-DD. "
            "May be repeated."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
    )

    return parser


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        (
            "=== SDV FEATURE GENERATION "
            f"{datetime.now(timezone.utc).isoformat()} "
            "===\n"
        ),
        encoding="utf-8",
    )

    try:
        cfg = read_yaml(
            args.config
        )

        validate_config(
            cfg
        )

        leagues = (
            args.league
            or list(
                LEAGUE_LABELS
            )
        )

        outputs: list[
            Path
        ] = []

        if args.mode == "history":
            if args.game_date:
                raise ValueError(
                    "--game-date is only valid "
                    "with --mode current"
                )

            for league in leagues:
                outputs.extend(
                    build_historical(
                        cfg,
                        league,
                        args.internal_season,
                    )
                )

        else:
            if len(
                leagues
            ) != 1:
                raise ValueError(
                    "--mode current requires "
                    "exactly one --league"
                )

            if (
                not args.internal_season
                or len(
                    args.internal_season
                )
                != 1
            ):
                raise ValueError(
                    "--mode current requires "
                    "exactly one --internal-season"
                )

            outputs.extend(
                build_current(
                    cfg,
                    leagues[
                        0
                    ],
                    args.internal_season[
                        0
                    ],
                    args.game_date,
                )
            )

        log(
            "STATUS: SUCCESS | "
            f"files={len(outputs)} "
            "feature_version="
            f"{cfg['feature_version']}"
        )

        print(
            "SDV feature generation "
            "complete: SUCCESS. "
            f"files={len(outputs)}"
        )

        return 0

    except Exception as exc:
        log(
            f"FATAL: {exc}"
        )

        log(
            traceback
            .format_exc()
            .rstrip()
        )

        log(
            "STATUS: FAILED"
        )

        print(
            "SDV feature generation "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
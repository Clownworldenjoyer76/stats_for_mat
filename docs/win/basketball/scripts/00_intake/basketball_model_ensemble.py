#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_model_ensemble.py
"""Learn and apply the DRatings + SDV basketball ensemble.

Training inputs
---------------
SDV chronological/OOS predictions:
    docs/win/basketball/errors/99_validation/sdv_model_v1/{league}/
        {LEAGUE}_sdv_model_v1_oos_predictions.parquet

SDV training report / lockbox definition:
    docs/win/basketball/errors/99_validation/sdv_model_v1/{league}/
        {LEAGUE}_sdv_model_v1_training_report.json

DRatings predictions:
    docs/win/basketball/00_intake/predictions/{league}/
        *_predictions.csv

Weights:
    docs/win/basketball/models/ensemble/{league}/weights.json

Current inference inputs
------------------------
DRatings:
    docs/win/basketball/00_intake/predictions/{league}/
        {game_date}_{LEAGUE}_predictions.csv

SDV:
    docs/win/basketball/00_intake/predictions_sdv/{league}/
        {game_date}_{LEAGUE}_predictions.csv

Canonical slate:
    docs/win/basketball/daily_games/{league}/
        {game_date}_{LEAGUE}.csv

Output:
    docs/win/basketball/00_intake/predictions_ensemble/{league}/
        {game_date}_{LEAGUE}_predictions.csv

Rules
-----
- DRatings and SDV are joined strictly by canonical game_id.
- No team/date composite fallback is permitted.
- Ensemble margin and total weights are learned separately.
- Weight learning uses SDV OOS/development rows only.
- The untouched SDV lockbox season is forbidden.
- Current inference requires exact daily-slate coverage from both components.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


BASE = Path("docs/win/basketball")

DRATINGS_ROOT = (
    BASE
    / "00_intake/predictions"
)

SDV_PREDICTIONS_ROOT = (
    BASE
    / "00_intake/predictions_sdv"
)

DAILY_GAMES_ROOT = (
    BASE
    / "daily_games"
)

SDV_VALIDATION_ROOT = (
    BASE
    / "errors/99_validation/sdv_model_v1"
)

SDV_MODEL_ROOT = (
    BASE
    / "models/sdv"
)

ENSEMBLE_MODEL_ROOT = (
    BASE
    / "models/ensemble"
)

ENSEMBLE_OUTPUT_ROOT = (
    BASE
    / "00_intake/predictions_ensemble"
)

ERROR_DIR = (
    BASE
    / "errors/00_intake"
)

LOG_FILE = (
    ERROR_DIR
    / "basketball_model_ensemble.txt"
)

DRATINGS_SCRAPER_PATH = (
    BASE
    / "scripts/00_intake/"
    "basketball_drat_scraper.py"
)

DRATINGS_TRANSFORM_PATH = (
    BASE
    / "scripts/00_intake/"
    "transform_basketball.py"
)

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

ENSEMBLE_VERSION = (
    "dratings_sdv_ensemble_v1"
)

DRATINGS_MODEL_VERSION = (
    "dratings_external_unversioned"
)

MIN_TRAINING_ROWS = 20
MIN_UNIQUE_DATES = 4
OOS_FOLDS = 5
OOS_INITIAL_DATE_FRACTION = 0.50

OUTPUT_FIELDS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",

    "model_source",
    "model_version",
    "ensemble_version",

    "dratings_model_version",
    "dratings_pipeline_version",
    "sdv_model_version",
    "sdv_feature_version",

    "margin_weight_dratings",
    "margin_weight_sdv",
    "total_weight_dratings",
    "total_weight_sdv",

    "home_prob",
    "away_prob",
    "raw_home_ml_prob",
    "raw_away_ml_prob",

    "home_projected_points",
    "away_projected_points",
    "total_projected_points",
    "expected_margin",
    "expected_total",

    "dratings_expected_margin",
    "sdv_expected_margin",
    "dratings_expected_total",
    "sdv_expected_total",
    "dratings_home_prob",
    "sdv_home_prob",

    "prediction_generated_at_utc",
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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
            f"{utc_now()} | "
            f"{message}\n"
        )


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
            math.isfinite(number)
            and number.is_integer()
        ):
            return str(
                int(number)
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return text


def normalize_date(
    value: Any,
) -> str:
    text = clean(
        value
    )

    if not text:
        return ""

    text = (
        text[:10]
        .replace(
            "_",
            "-",
        )
        .replace(
            "/",
            "-",
        )
    )

    try:
        parsed = datetime.strptime(
            text,
            "%Y-%m-%d",
        )

    except ValueError:
        return ""

    return parsed.strftime(
        "%Y-%m-%d"
    )


def file_date(
    value: Any,
) -> str:
    normalized = normalize_date(
        value
    )

    if not normalized:
        return ""

    return normalized.replace(
        "-",
        "_",
    )


def team_key(
    value: Any,
) -> str:
    return " ".join(
        clean(value)
        .casefold()
        .split()
    )


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
        result = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def required_float(
    value: Any,
    label: str,
) -> float:
    result = to_float(
        value
    )

    if result is None:
        raise RuntimeError(
            f"{label} is missing "
            "or non-numeric"
        )

    return result


def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "JSON root must be object: "
            f"{path}"
        )

    return payload


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        tmp.replace(
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def read_csv_rows(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return [
            dict(row)
            for row
            in csv.DictReader(
                handle
            )
        ]


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            "Refusing to write zero-row "
            f"ensemble file: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    try:
        with tmp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    OUTPUT_FIELDS
                ),
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        tmp.replace(
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def sha256_file(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def dratings_pipeline_metadata() -> dict[str, Any]:
    scraper_sha = sha256_file(
        DRATINGS_SCRAPER_PATH
    )

    transform_sha = sha256_file(
        DRATINGS_TRANSFORM_PATH
    )

    digest = hashlib.sha256()

    digest.update(
        scraper_sha.encode(
            "utf-8"
        )
    )

    digest.update(
        transform_sha.encode(
            "utf-8"
        )
    )

    pipeline_sha = digest.hexdigest()

    return {
        "model_version": (
            DRATINGS_MODEL_VERSION
        ),
        "pipeline_version": (
            "dratings_pipeline_"
            f"{pipeline_sha[:16]}"
        ),
        "scraper_path": str(
            DRATINGS_SCRAPER_PATH
        ),
        "scraper_sha256": (
            scraper_sha
        ),
        "transform_path": str(
            DRATINGS_TRANSFORM_PATH
        ),
        "transform_sha256": (
            transform_sha
        ),
    }


def sdv_training_paths(
    league: str,
) -> tuple[Path, Path]:
    label = LEAGUE_LABELS[
        league
    ]

    root = (
        SDV_VALIDATION_ROOT
        / league
    )

    report = (
        root
        / (
            f"{label}_"
            "sdv_model_v1_"
            "training_report.json"
        )
    )

    oos = (
        root
        / (
            f"{label}_"
            "sdv_model_v1_"
            "oos_predictions.parquet"
        )
    )

    return (
        report,
        oos,
    )


def load_sdv_training_info(
    league: str,
) -> dict[str, Any]:
    (
        report_path,
        oos_path,
    ) = sdv_training_paths(
        league
    )

    report = read_json(
        report_path
    )

    if clean(
        report.get(
            "status"
        )
    ).upper() != "PASS":
        raise RuntimeError(
            f"{league}: SDV training "
            "report is not PASS"
        )

    development = report.get(
        "development"
    )

    lockbox = report.get(
        "lockbox"
    )

    if not isinstance(
        development,
        dict,
    ):
        raise RuntimeError(
            f"{league}: development "
            "metadata missing"
        )

    if not isinstance(
        lockbox,
        dict,
    ):
        raise RuntimeError(
            f"{league}: lockbox "
            "metadata missing"
        )

    development_seasons = [
        int(value)
        for value
        in development.get(
            "seasons",
            [],
        )
    ]

    lockbox_season = int(
        lockbox[
            "season"
        ]
    )

    if lockbox_season in (
        development_seasons
    ):
        raise RuntimeError(
            f"{league}: lockbox season "
            "appears in development seasons"
        )

    if bool(
        lockbox.get(
            "used_for_model_fit",
            True,
        )
    ):
        raise RuntimeError(
            f"{league}: SDV report says "
            "lockbox was used for fit"
        )

    if bool(
        lockbox.get(
            "used_for_model_selection",
            True,
        )
    ):
        raise RuntimeError(
            f"{league}: SDV report says "
            "lockbox was used for "
            "model selection"
        )

    if not oos_path.exists():
        raise FileNotFoundError(
            oos_path
        )

    return {
        "report_path": report_path,
        "oos_path": oos_path,
        "report": report,
        "development_seasons": (
            development_seasons
        ),
        "lockbox_season": (
            lockbox_season
        ),
        "sdv_model_version": clean(
            report.get(
                "model_version"
            )
        ),
        "sdv_feature_version": clean(
            report.get(
                "feature_version"
            )
        ),
    }


def load_sdv_oos_rows(
    league: str,
    info: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    path = info[
        "oos_path"
    ]

    frame = pl.read_parquet(
        path
    )

    required_columns = {
        "game_id",
        "game_date",
        "internal_season",
        "actual_margin",
        "expected_margin",
        "actual_total",
        "expected_total",
    }

    missing = sorted(
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{league}: SDV OOS file "
            f"missing columns={missing}"
        )

    development_seasons = set(
        info[
            "development_seasons"
        ]
    )

    lockbox_season = int(
        info[
            "lockbox_season"
        ]
    )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for raw in frame.to_dicts():
        game_id = clean_id(
            raw.get(
                "game_id"
            )
        )

        if not game_id:
            raise RuntimeError(
                f"{league}: SDV OOS row "
                "has blank game_id"
            )

        if game_id in result:
            raise RuntimeError(
                f"{league}: duplicate SDV "
                f"OOS game_id={game_id}"
            )

        season = int(
            raw[
                "internal_season"
            ]
        )

        if season == lockbox_season:
            raise RuntimeError(
                "LOCKBOX VIOLATION | "
                f"league={league} "
                f"game_id={game_id} "
                f"season={season}"
            )

        if season not in (
            development_seasons
        ):
            raise RuntimeError(
                f"{league}: SDV OOS "
                f"game_id={game_id} "
                f"season={season} is not "
                "a configured development "
                "season"
            )

        game_date = normalize_date(
            raw.get(
                "game_date"
            )
        )

        if not game_date:
            raise RuntimeError(
                f"{league}: SDV OOS "
                f"game_id={game_id} "
                "has invalid game_date"
            )

        result[
            game_id
        ] = {
            "game_id": game_id,
            "game_date": game_date,
            "internal_season": season,
            "actual_margin": (
                required_float(
                    raw.get(
                        "actual_margin"
                    ),
                    "actual_margin",
                )
            ),
            "sdv_expected_margin": (
                required_float(
                    raw.get(
                        "expected_margin"
                    ),
                    "expected_margin",
                )
            ),
            "actual_total": (
                required_float(
                    raw.get(
                        "actual_total"
                    ),
                    "actual_total",
                )
            ),
            "sdv_expected_total": (
                required_float(
                    raw.get(
                        "expected_total"
                    ),
                    "expected_total",
                )
            ),
        }

    if not result:
        raise RuntimeError(
            f"{league}: SDV OOS set "
            "is empty"
        )

    return result


def dratings_projection_values(
    row: dict[str, Any],
) -> tuple[
    float | None,
    float | None,
]:
    home = to_float(
        row.get(
            "home_projected_points"
        )
    )

    away = to_float(
        row.get(
            "away_projected_points"
        )
    )

    total = to_float(
        row.get(
            "total_projected_points"
        )
    )

    margin: float | None = None

    if (
        home is not None
        and away is not None
    ):
        margin = (
            home
            - away
        )

        if total is None:
            total = (
                home
                + away
            )

    return (
        margin,
        total,
    )


def load_dratings_history(
    league: str,
) -> dict[str, dict[str, Any]]:
    label = LEAGUE_LABELS[
        league
    ]

    folder = (
        DRATINGS_ROOT
        / league
    )

    if not folder.exists():
        return {}

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for path in sorted(
        folder.glob(
            f"*_{label}_predictions.csv"
        )
    ):
        rows = read_csv_rows(
            path
        )

        for raw in rows:
            game_id = clean_id(
                raw.get(
                    "game_id"
                )
            )

            if not game_id:
                continue

            (
                margin,
                total,
            ) = dratings_projection_values(
                raw
            )

            if (
                margin is None
                or total is None
            ):
                continue

            candidate = {
                "game_id": game_id,
                "game_date": (
                    normalize_date(
                        raw.get(
                            "game_date"
                        )
                    )
                ),
                "dratings_expected_margin": (
                    margin
                ),
                "dratings_expected_total": (
                    total
                ),
                "source_file": str(
                    path
                ),
            }

            existing = result.get(
                game_id
            )

            if existing is None:
                result[
                    game_id
                ] = candidate
                continue

            comparable_old = (
                existing[
                    "dratings_expected_margin"
                ],
                existing[
                    "dratings_expected_total"
                ],
            )

            comparable_new = (
                candidate[
                    "dratings_expected_margin"
                ],
                candidate[
                    "dratings_expected_total"
                ],
            )

            if comparable_old != comparable_new:
                raise RuntimeError(
                    f"{league}: conflicting "
                    "DRatings predictions for "
                    f"game_id={game_id}"
                )

    return result


def build_training_rows(
    league: str,
    sdv_rows: dict[
        str,
        dict[str, Any],
    ],
    dratings_rows: dict[
        str,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    matched_ids = sorted(
        set(
            sdv_rows
        )
        & set(
            dratings_rows
        )
    )

    if not matched_ids:
        sdv_examples = list(
            sorted(
                sdv_rows
            )
        )[:5]

        dratings_examples = list(
            sorted(
                dratings_rows
            )
        )[:5]

        raise RuntimeError(
            "NO NON-LOCKBOX DRATINGS/SDV "
            "TRAINING MATCHES | "
            f"league={LEAGUE_LABELS[league]} | "
            f"sdv_oos_rows={len(sdv_rows)} | "
            "dratings_history_rows="
            f"{len(dratings_rows)} | "
            f"sdv_example_ids={sdv_examples} | "
            "dratings_example_ids="
            f"{dratings_examples}. "
            "Weights cannot be learned "
            "without matching canonical "
            "game_id history. The lockbox "
            "will not be used as fallback."
        )

    rows: list[
        dict[str, Any]
    ] = []

    for game_id in matched_ids:
        sdv = sdv_rows[
            game_id
        ]

        dratings = dratings_rows[
            game_id
        ]

        dratings_date = clean(
            dratings.get(
                "game_date"
            )
        )

        if (
            dratings_date
            and dratings_date
            != sdv[
                "game_date"
            ]
        ):
            raise RuntimeError(
                f"{league}: game_date "
                "mismatch for canonical "
                f"game_id={game_id}: "
                f"sdv={sdv['game_date']} "
                f"dratings={dratings_date}"
            )

        rows.append(
            {
                **sdv,
                "dratings_expected_margin": (
                    dratings[
                        "dratings_expected_margin"
                    ]
                ),
                "dratings_expected_total": (
                    dratings[
                        "dratings_expected_total"
                    ]
                ),
                "dratings_source_file": (
                    dratings[
                        "source_file"
                    ]
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row[
                "game_date"
            ],
            row[
                "game_id"
            ],
        )
    )

    unique_dates = {
        row[
            "game_date"
        ]
        for row
        in rows
    }

    if len(
        rows
    ) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"{league}: only {len(rows)} "
            "matched development rows; "
            f"minimum={MIN_TRAINING_ROWS}"
        )

    if len(
        unique_dates
    ) < MIN_UNIQUE_DATES:
        raise RuntimeError(
            f"{league}: only "
            f"{len(unique_dates)} "
            "unique matched dates; "
            f"minimum={MIN_UNIQUE_DATES}"
        )

    return rows


def fit_weight(
    rows: list[dict[str, Any]],
    *,
    sdv_field: str,
    dratings_field: str,
    actual_field: str,
) -> float:
    if not rows:
        raise RuntimeError(
            "Cannot fit ensemble weight "
            "with zero rows"
        )

    sdv = np.asarray(
        [
            float(
                row[
                    sdv_field
                ]
            )
            for row
            in rows
        ],
        dtype=float,
    )

    dratings = np.asarray(
        [
            float(
                row[
                    dratings_field
                ]
            )
            for row
            in rows
        ],
        dtype=float,
    )

    actual = np.asarray(
        [
            float(
                row[
                    actual_field
                ]
            )
            for row
            in rows
        ],
        dtype=float,
    )

    delta = (
        sdv
        - dratings
    )

    denominator = float(
        np.dot(
            delta,
            delta,
        )
    )

    if denominator <= 1e-12:
        return 0.5

    numerator = float(
        np.dot(
            delta,
            (
                actual
                - dratings
            ),
        )
    )

    sdv_weight = (
        numerator
        / denominator
    )

    return float(
        min(
            max(
                sdv_weight,
                0.0,
            ),
            1.0,
        )
    )


def blend(
    sdv_value: float,
    dratings_value: float,
    sdv_weight: float,
) -> float:
    return (
        sdv_weight
        * sdv_value
        + (
            1.0
            - sdv_weight
        )
        * dratings_value
    )


def metrics(
    actual: list[float],
    predicted: list[float],
) -> dict[str, float]:
    if (
        not actual
        or len(actual)
        != len(predicted)
    ):
        raise RuntimeError(
            "Invalid metric vectors"
        )

    a = np.asarray(
        actual,
        dtype=float,
    )

    p = np.asarray(
        predicted,
        dtype=float,
    )

    residual = (
        a
        - p
    )

    return {
        "rows": int(
            len(a)
        ),
        "mae": float(
            np.mean(
                np.abs(
                    residual
                )
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        residual
                    )
                )
            )
        ),
        "mean_residual": float(
            np.mean(
                residual
            )
        ),
    }


def chronological_folds(
    rows: list[dict[str, Any]],
) -> list[
    tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
]:
    dates = sorted(
        {
            row[
                "game_date"
            ]
            for row
            in rows
        }
    )

    first_validation_index = max(
        1,
        int(
            math.ceil(
                len(dates)
                * OOS_INITIAL_DATE_FRACTION
            )
        ),
    )

    if (
        first_validation_index
        >= len(dates)
    ):
        raise RuntimeError(
            "Chronological OOS split "
            "has no validation dates"
        )

    validation_dates = dates[
        first_validation_index:
    ]

    fold_count = min(
        OOS_FOLDS,
        len(
            validation_dates
        ),
    )

    chunks = np.array_split(
        np.asarray(
            validation_dates,
            dtype=object,
        ),
        fold_count,
    )

    result = []

    for chunk in chunks:
        values = [
            str(value)
            for value
            in chunk.tolist()
        ]

        if not values:
            continue

        validation_set = set(
            values
        )

        validation_start = min(
            validation_set
        )

        train_rows = [
            row
            for row
            in rows
            if row[
                "game_date"
            ] < validation_start
        ]

        validation_rows = [
            row
            for row
            in rows
            if row[
                "game_date"
            ] in validation_set
        ]

        if (
            not train_rows
            or not validation_rows
        ):
            continue

        if max(
            row[
                "game_date"
            ]
            for row
            in train_rows
        ) >= min(
            row[
                "game_date"
            ]
            for row
            in validation_rows
        ):
            raise RuntimeError(
                "Chronological fold leakage "
                "detected"
            )

        result.append(
            (
                train_rows,
                validation_rows,
            )
        )

    if not result:
        raise RuntimeError(
            "No chronological ensemble "
            "folds created"
        )

    return result


def chronological_validation(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    folds = chronological_folds(
        rows
    )

    margin_actual: list[
        float
    ] = []

    margin_ensemble: list[
        float
    ] = []

    margin_sdv: list[
        float
    ] = []

    margin_dratings: list[
        float
    ] = []

    total_actual: list[
        float
    ] = []

    total_ensemble: list[
        float
    ] = []

    total_sdv: list[
        float
    ] = []

    total_dratings: list[
        float
    ] = []

    fold_reports = []

    for (
        fold_number,
        (
            train_rows,
            validation_rows,
        ),
    ) in enumerate(
        folds,
        start=1,
    ):
        margin_sdv_weight = (
            fit_weight(
                train_rows,
                sdv_field=(
                    "sdv_expected_margin"
                ),
                dratings_field=(
                    "dratings_expected_margin"
                ),
                actual_field=(
                    "actual_margin"
                ),
            )
        )

        total_sdv_weight = (
            fit_weight(
                train_rows,
                sdv_field=(
                    "sdv_expected_total"
                ),
                dratings_field=(
                    "dratings_expected_total"
                ),
                actual_field=(
                    "actual_total"
                ),
            )
        )

        for row in validation_rows:
            actual_margin = float(
                row[
                    "actual_margin"
                ]
            )

            actual_total = float(
                row[
                    "actual_total"
                ]
            )

            sdv_margin = float(
                row[
                    "sdv_expected_margin"
                ]
            )

            dratings_margin = float(
                row[
                    "dratings_expected_margin"
                ]
            )

            sdv_total_value = float(
                row[
                    "sdv_expected_total"
                ]
            )

            dratings_total_value = float(
                row[
                    "dratings_expected_total"
                ]
            )

            margin_actual.append(
                actual_margin
            )

            margin_sdv.append(
                sdv_margin
            )

            margin_dratings.append(
                dratings_margin
            )

            margin_ensemble.append(
                blend(
                    sdv_margin,
                    dratings_margin,
                    margin_sdv_weight,
                )
            )

            total_actual.append(
                actual_total
            )

            total_sdv.append(
                sdv_total_value
            )

            total_dratings.append(
                dratings_total_value
            )

            total_ensemble.append(
                blend(
                    sdv_total_value,
                    dratings_total_value,
                    total_sdv_weight,
                )
            )

        fold_reports.append(
            {
                "fold": fold_number,
                "training_rows": len(
                    train_rows
                ),
                "validation_rows": len(
                    validation_rows
                ),
                "training_first_date": (
                    min(
                        row[
                            "game_date"
                        ]
                        for row
                        in train_rows
                    )
                ),
                "training_last_date": (
                    max(
                        row[
                            "game_date"
                        ]
                        for row
                        in train_rows
                    )
                ),
                "validation_first_date": (
                    min(
                        row[
                            "game_date"
                        ]
                        for row
                        in validation_rows
                    )
                ),
                "validation_last_date": (
                    max(
                        row[
                            "game_date"
                        ]
                        for row
                        in validation_rows
                    )
                ),
                "margin_sdv_weight": (
                    margin_sdv_weight
                ),
                "margin_dratings_weight": (
                    1.0
                    - margin_sdv_weight
                ),
                "total_sdv_weight": (
                    total_sdv_weight
                ),
                "total_dratings_weight": (
                    1.0
                    - total_sdv_weight
                ),
            }
        )

    return {
        "method": (
            "expanding_window"
        ),
        "initial_training_date_fraction": (
            OOS_INITIAL_DATE_FRACTION
        ),
        "folds": fold_reports,
        "margin": {
            "ensemble": metrics(
                margin_actual,
                margin_ensemble,
            ),
            "sdv": metrics(
                margin_actual,
                margin_sdv,
            ),
            "dratings": metrics(
                margin_actual,
                margin_dratings,
            ),
        },
        "total": {
            "ensemble": metrics(
                total_actual,
                total_ensemble,
            ),
            "sdv": metrics(
                total_actual,
                total_sdv,
            ),
            "dratings": metrics(
                total_actual,
                total_dratings,
            ),
        },
    }


def train_league(
    league: str,
) -> dict[str, Any]:
    label = LEAGUE_LABELS[
        league
    ]

    info = load_sdv_training_info(
        league
    )

    sdv_rows = load_sdv_oos_rows(
        league,
        info,
    )

    dratings_rows = (
        load_dratings_history(
            league
        )
    )

    training_rows = (
        build_training_rows(
            league,
            sdv_rows,
            dratings_rows,
        )
    )

    lockbox = int(
        info[
            "lockbox_season"
        ]
    )

    if any(
        int(
            row[
                "internal_season"
            ]
        )
        == lockbox
        for row
        in training_rows
    ):
        raise RuntimeError(
            "LOCKBOX VIOLATION DURING "
            f"ENSEMBLE TRAINING | {label}"
        )

    validation = (
        chronological_validation(
            training_rows
        )
    )

    margin_sdv_weight = (
        fit_weight(
            training_rows,
            sdv_field=(
                "sdv_expected_margin"
            ),
            dratings_field=(
                "dratings_expected_margin"
            ),
            actual_field=(
                "actual_margin"
            ),
        )
    )

    total_sdv_weight = (
        fit_weight(
            training_rows,
            sdv_field=(
                "sdv_expected_total"
            ),
            dratings_field=(
                "dratings_expected_total"
            ),
            actual_field=(
                "actual_total"
            ),
        )
    )

    dratings_meta = (
        dratings_pipeline_metadata()
    )

    payload = {
        "schema_version": 1,
        "status": "trained",
        "ensemble_version": (
            ENSEMBLE_VERSION
        ),
        "league": label,
        "created_at_utc": utc_now(),

        "components": {
            "dratings": (
                dratings_meta
            ),
            "sdv": {
                "model_version": (
                    info[
                        "sdv_model_version"
                    ]
                ),
                "feature_version": (
                    info[
                        "sdv_feature_version"
                    ]
                ),
                "oos_predictions_path": str(
                    info[
                        "oos_path"
                    ]
                ),
                "training_report_path": str(
                    info[
                        "report_path"
                    ]
                ),
            },
        },

        "training": {
            "join_key": "game_id",
            "join_policy": (
                "strict_canonical_game_id_only"
            ),
            "sdv_source": (
                "chronological_oos_predictions"
            ),
            "dratings_source_root": str(
                DRATINGS_ROOT
                / league
            ),
            "development_internal_seasons": (
                info[
                    "development_seasons"
                ]
            ),
            "lockbox_internal_season": (
                lockbox
            ),
            "lockbox_used": False,
            "lockbox_tuning_forbidden": True,
            "training_rows": len(
                training_rows
            ),
            "first_training_game_date": (
                min(
                    row[
                        "game_date"
                    ]
                    for row
                    in training_rows
                )
            ),
            "last_training_game_date": (
                max(
                    row[
                        "game_date"
                    ]
                    for row
                    in training_rows
                )
            ),
            "unique_training_dates": len(
                {
                    row[
                        "game_date"
                    ]
                    for row
                    in training_rows
                }
            ),
            "chronological_validation": (
                validation
            ),
        },

        "margin": {
            "target_definition": (
                "actual_home_points - "
                "actual_away_points"
            ),
            "objective": (
                "minimum_squared_error_"
                "constrained_convex_blend"
            ),
            "sdv_weight": (
                margin_sdv_weight
            ),
            "dratings_weight": (
                1.0
                - margin_sdv_weight
            ),
        },

        "total": {
            "target_definition": (
                "actual_home_points + "
                "actual_away_points"
            ),
            "objective": (
                "minimum_squared_error_"
                "constrained_convex_blend"
            ),
            "sdv_weight": (
                total_sdv_weight
            ),
            "dratings_weight": (
                1.0
                - total_sdv_weight
            ),
        },
    }

    return payload


def weights_path(
    league: str,
) -> Path:
    return (
        ENSEMBLE_MODEL_ROOT
        / league
        / "weights.json"
    )


def train_selected(
    leagues: list[str],
) -> None:
    payloads: dict[
        str,
        dict[str, Any],
    ] = {}

    for league in leagues:
        log(
            "TRAIN START | "
            f"league="
            f"{LEAGUE_LABELS[league]}"
        )

        payloads[
            league
        ] = train_league(
            league
        )

        log(
            "TRAIN READY | "
            f"league="
            f"{LEAGUE_LABELS[league]} "
            f"rows="
            f"{payloads[league]['training']['training_rows']}"
        )

    for (
        league,
        payload,
    ) in payloads.items():
        path = weights_path(
            league
        )

        write_json_atomic(
            path,
            payload,
        )

        log(
            "WEIGHTS WRITTEN | "
            f"league="
            f"{LEAGUE_LABELS[league]} "
            f"path={path}"
        )


def load_weights(
    league: str,
) -> dict[str, Any]:
    path = weights_path(
        league
    )

    payload = read_json(
        path
    )

    if clean(
        payload.get(
            "status"
        )
    ) != "trained":
        raise RuntimeError(
            f"{league}: ensemble "
            "weights are not trained"
        )

    if clean(
        payload.get(
            "ensemble_version"
        )
    ) != ENSEMBLE_VERSION:
        raise RuntimeError(
            f"{league}: ensemble "
            "version mismatch"
        )

    training = payload.get(
        "training"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise RuntimeError(
            f"{league}: ensemble "
            "training metadata missing"
        )

    if bool(
        training.get(
            "lockbox_used",
            True,
        )
    ):
        raise RuntimeError(
            f"{league}: weight artifact "
            "reports lockbox usage"
        )

    for target in (
        "margin",
        "total",
    ):
        section = payload.get(
            target
        )

        if not isinstance(
            section,
            dict,
        ):
            raise RuntimeError(
                f"{league}: {target} "
                "weight section missing"
            )

        sdv_weight = required_float(
            section.get(
                "sdv_weight"
            ),
            f"{target}.sdv_weight",
        )

        dratings_weight = required_float(
            section.get(
                "dratings_weight"
            ),
            (
                f"{target}."
                "dratings_weight"
            ),
        )

        if not (
            0.0
            <= sdv_weight
            <= 1.0
        ):
            raise RuntimeError(
                f"{league}: invalid "
                f"{target} SDV weight"
            )

        if not (
            0.0
            <= dratings_weight
            <= 1.0
        ):
            raise RuntimeError(
                f"{league}: invalid "
                f"{target} DRatings weight"
            )

        if abs(
            (
                sdv_weight
                + dratings_weight
            )
            - 1.0
        ) > 1e-12:
            raise RuntimeError(
                f"{league}: {target} "
                "weights do not sum to 1"
            )

    current_dratings_meta = (
        dratings_pipeline_metadata()
    )

    components = payload.get(
        "components"
    )

    if not isinstance(
        components,
        dict,
    ):
        raise RuntimeError(
            f"{league}: component "
            "metadata missing"
        )

    dratings_component = (
        components.get(
            "dratings"
        )
    )

    if not isinstance(
        dratings_component,
        dict,
    ):
        raise RuntimeError(
            f"{league}: DRatings "
            "component metadata missing"
        )

    if clean(
        dratings_component.get(
            "pipeline_version"
        )
    ) != clean(
        current_dratings_meta.get(
            "pipeline_version"
        )
    ):
        raise RuntimeError(
            f"{league}: DRatings pipeline "
            "version changed after ensemble "
            "weight training; retrain weights"
        )

    return payload


def unique_rows_by_id(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:
        game_id = clean_id(
            row.get(
                "game_id"
            )
        )

        if not game_id:
            raise RuntimeError(
                f"{source}: blank "
                "canonical game_id"
            )

        if game_id in result:
            raise RuntimeError(
                f"{source}: duplicate "
                f"game_id={game_id}"
            )

        row[
            "game_id"
        ] = game_id

        result[
            game_id
        ] = row

    return result


def current_paths(
    league: str,
    game_date: str,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
]:
    label = LEAGUE_LABELS[
        league
    ]

    filename = (
        f"{game_date}_"
        f"{label}_predictions.csv"
    )

    daily_filename = (
        f"{game_date}_"
        f"{label}.csv"
    )

    dratings = (
        DRATINGS_ROOT
        / league
        / filename
    )

    sdv = (
        SDV_PREDICTIONS_ROOT
        / league
        / filename
    )

    daily = (
        DAILY_GAMES_ROOT
        / league
        / daily_filename
    )

    output = (
        ENSEMBLE_OUTPUT_ROOT
        / league
        / filename
    )

    return (
        dratings,
        sdv,
        daily,
        output,
    )


def assert_identity(
    game_id: str,
    daily: dict[str, Any],
    dratings: dict[str, Any],
    sdv: dict[str, Any],
) -> None:
    daily_date = normalize_date(
        daily.get(
            "game_date"
        )
    )

    for (
        source_name,
        row,
    ) in (
        (
            "DRatings",
            dratings,
        ),
        (
            "SDV",
            sdv,
        ),
    ):
        source_date = normalize_date(
            row.get(
                "game_date"
            )
        )

        if (
            source_date
            and source_date
            != daily_date
        ):
            raise RuntimeError(
                f"game_id={game_id}: "
                f"{source_name} game_date "
                "does not match daily slate"
            )

        if (
            team_key(
                row.get(
                    "home_team"
                )
            )
            != team_key(
                daily.get(
                    "home_team"
                )
            )
        ):
            raise RuntimeError(
                f"game_id={game_id}: "
                f"{source_name} home team "
                "does not match daily slate"
            )

        if (
            team_key(
                row.get(
                    "away_team"
                )
            )
            != team_key(
                daily.get(
                    "away_team"
                )
            )
        ):
            raise RuntimeError(
                f"game_id={game_id}: "
                f"{source_name} away team "
                "does not match daily slate"
            )


def current_dratings_values(
    row: dict[str, Any],
) -> dict[str, float]:
    (
        margin,
        total,
    ) = dratings_projection_values(
        row
    )

    if margin is None:
        raise RuntimeError(
            "Current DRatings row has "
            "no projected margin"
        )

    if total is None:
        raise RuntimeError(
            "Current DRatings row has "
            "no projected total"
        )

    home_prob = required_float(
        row.get(
            "home_prob"
        ),
        "DRatings home_prob",
    )

    if not (
        0.0
        <= home_prob
        <= 1.0
    ):
        raise RuntimeError(
            "DRatings home_prob "
            "outside [0,1]"
        )

    return {
        "expected_margin": margin,
        "expected_total": total,
        "home_prob": home_prob,
    }


def current_sdv_values(
    row: dict[str, Any],
) -> dict[str, float]:
    margin = to_float(
        row.get(
            "expected_margin"
        )
    )

    if margin is None:
        home = to_float(
            row.get(
                "home_projected_points"
            )
        )

        away = to_float(
            row.get(
                "away_projected_points"
            )
        )

        if (
            home is None
            or away is None
        ):
            raise RuntimeError(
                "Current SDV row has "
                "no expected margin"
            )

        margin = (
            home
            - away
        )

    total = to_float(
        row.get(
            "expected_total"
        )
    )

    if total is None:
        total = to_float(
            row.get(
                "total_projected_points"
            )
        )

    if total is None:
        raise RuntimeError(
            "Current SDV row has "
            "no expected total"
        )

    home_prob = to_float(
        row.get(
            "raw_home_ml_prob"
        )
    )

    if home_prob is None:
        home_prob = to_float(
            row.get(
                "home_prob"
            )
        )

    if home_prob is None:
        raise RuntimeError(
            "Current SDV row has "
            "no home probability"
        )

    if not (
        0.0
        <= home_prob
        <= 1.0
    ):
        raise RuntimeError(
            "SDV home probability "
            "outside [0,1]"
        )

    return {
        "expected_margin": margin,
        "expected_total": total,
        "home_prob": home_prob,
    }


def validate_sdv_versions(
    rows: dict[str, dict[str, Any]],
    weights: dict[str, Any],
) -> tuple[str, str]:
    components = weights[
        "components"
    ]

    expected_model_version = clean(
        components[
            "sdv"
        ].get(
            "model_version"
        )
    )

    expected_feature_version = clean(
        components[
            "sdv"
        ].get(
            "feature_version"
        )
    )

    model_versions = {
        clean(
            row.get(
                "model_version"
            )
        )
        for row
        in rows.values()
    }

    feature_versions = {
        clean(
            row.get(
                "feature_version"
            )
        )
        for row
        in rows.values()
    }

    sources = {
        clean(
            row.get(
                "model_source"
            )
        ).lower()
        for row
        in rows.values()
    }

    if sources != {
        "sdv"
    }:
        raise RuntimeError(
            "Current SDV prediction file "
            "contains invalid model_source"
        )

    if model_versions != {
        expected_model_version
    }:
        raise RuntimeError(
            "SDV model version does not "
            "match ensemble weights"
        )

    if feature_versions != {
        expected_feature_version
    }:
        raise RuntimeError(
            "SDV feature version does not "
            "match ensemble weights"
        )

    return (
        expected_model_version,
        expected_feature_version,
    )


def predict_league_date(
    league: str,
    game_date: str,
) -> Path:
    label = LEAGUE_LABELS[
        league
    ]

    weights = load_weights(
        league
    )

    (
        dratings_path,
        sdv_path,
        daily_path,
        output_path,
    ) = current_paths(
        league,
        game_date,
    )

    dratings_rows = (
        unique_rows_by_id(
            read_csv_rows(
                dratings_path
            ),
            source=(
                f"DRatings {dratings_path}"
            ),
        )
    )

    sdv_rows = unique_rows_by_id(
        read_csv_rows(
            sdv_path
        ),
        source=(
            f"SDV {sdv_path}"
        ),
    )

    daily_rows = (
        unique_rows_by_id(
            read_csv_rows(
                daily_path
            ),
            source=(
                f"daily_games {daily_path}"
            ),
        )
    )

    daily_ids = set(
        daily_rows
    )

    dratings_ids = set(
        dratings_rows
    )

    sdv_ids = set(
        sdv_rows
    )

    if dratings_ids != daily_ids:
        raise RuntimeError(
            "CURRENT DRATINGS COVERAGE "
            "FAILED | "
            f"league={label} "
            f"missing_from_dratings="
            f"{sorted(daily_ids - dratings_ids)} "
            f"extra_in_dratings="
            f"{sorted(dratings_ids - daily_ids)}"
        )

    if sdv_ids != daily_ids:
        raise RuntimeError(
            "CURRENT SDV COVERAGE "
            "FAILED | "
            f"league={label} "
            f"missing_from_sdv="
            f"{sorted(daily_ids - sdv_ids)} "
            f"extra_in_sdv="
            f"{sorted(sdv_ids - daily_ids)}"
        )

    (
        sdv_model_version,
        sdv_feature_version,
    ) = validate_sdv_versions(
        sdv_rows,
        weights,
    )

    margin_sdv_weight = float(
        weights[
            "margin"
        ][
            "sdv_weight"
        ]
    )

    margin_dratings_weight = float(
        weights[
            "margin"
        ][
            "dratings_weight"
        ]
    )

    total_sdv_weight = float(
        weights[
            "total"
        ][
            "sdv_weight"
        ]
    )

    total_dratings_weight = float(
        weights[
            "total"
        ][
            "dratings_weight"
        ]
    )

    dratings_component = (
        weights[
            "components"
        ][
            "dratings"
        ]
    )

    prediction_time = utc_now()

    output_rows = []

    for game_id in daily_rows:
        daily = daily_rows[
            game_id
        ]

        dratings = dratings_rows[
            game_id
        ]

        sdv = sdv_rows[
            game_id
        ]

        assert_identity(
            game_id,
            daily,
            dratings,
            sdv,
        )

        dratings_values = (
            current_dratings_values(
                dratings
            )
        )

        sdv_values = (
            current_sdv_values(
                sdv
            )
        )

        expected_margin = (
            margin_sdv_weight
            * sdv_values[
                "expected_margin"
            ]
            + margin_dratings_weight
            * dratings_values[
                "expected_margin"
            ]
        )

        expected_total = (
            total_sdv_weight
            * sdv_values[
                "expected_total"
            ]
            + total_dratings_weight
            * dratings_values[
                "expected_total"
            ]
        )

        home_projected_points = (
            expected_total
            + expected_margin
        ) / 2.0

        away_projected_points = (
            expected_total
            - expected_margin
        ) / 2.0

        home_prob = (
            margin_sdv_weight
            * sdv_values[
                "home_prob"
            ]
            + margin_dratings_weight
            * dratings_values[
                "home_prob"
            ]
        )

        home_prob = min(
            max(
                home_prob,
                0.0,
            ),
            1.0,
        )

        away_prob = (
            1.0
            - home_prob
        )

        if abs(
            (
                home_prob
                + away_prob
            )
            - 1.0
        ) > 1e-12:
            raise RuntimeError(
                "Ensemble moneyline "
                "complement check failed"
            )

        output_rows.append(
            {
                "sport": (
                    clean(
                        daily.get(
                            "sport"
                        )
                    )
                    or "Basketball"
                ),
                "league": (
                    clean(
                        daily.get(
                            "league"
                        )
                    )
                    or label
                ),
                "game_id": game_id,
                "game_date": (
                    file_date(
                        daily.get(
                            "game_date"
                        )
                    )
                ),
                "game_time": clean(
                    daily.get(
                        "game_time"
                    )
                ),
                "home_team": clean(
                    daily.get(
                        "home_team"
                    )
                ),
                "away_team": clean(
                    daily.get(
                        "away_team"
                    )
                ),

                "model_source": (
                    "ensemble"
                ),
                "model_version": (
                    ENSEMBLE_VERSION
                ),
                "ensemble_version": (
                    ENSEMBLE_VERSION
                ),

                "dratings_model_version": (
                    dratings_component[
                        "model_version"
                    ]
                ),
                "dratings_pipeline_version": (
                    dratings_component[
                        "pipeline_version"
                    ]
                ),
                "sdv_model_version": (
                    sdv_model_version
                ),
                "sdv_feature_version": (
                    sdv_feature_version
                ),

                "margin_weight_dratings": (
                    margin_dratings_weight
                ),
                "margin_weight_sdv": (
                    margin_sdv_weight
                ),
                "total_weight_dratings": (
                    total_dratings_weight
                ),
                "total_weight_sdv": (
                    total_sdv_weight
                ),

                "home_prob": home_prob,
                "away_prob": away_prob,
                "raw_home_ml_prob": (
                    home_prob
                ),
                "raw_away_ml_prob": (
                    away_prob
                ),

                "home_projected_points": (
                    home_projected_points
                ),
                "away_projected_points": (
                    away_projected_points
                ),
                "total_projected_points": (
                    expected_total
                ),
                "expected_margin": (
                    expected_margin
                ),
                "expected_total": (
                    expected_total
                ),

                "dratings_expected_margin": (
                    dratings_values[
                        "expected_margin"
                    ]
                ),
                "sdv_expected_margin": (
                    sdv_values[
                        "expected_margin"
                    ]
                ),
                "dratings_expected_total": (
                    dratings_values[
                        "expected_total"
                    ]
                ),
                "sdv_expected_total": (
                    sdv_values[
                        "expected_total"
                    ]
                ),
                "dratings_home_prob": (
                    dratings_values[
                        "home_prob"
                    ]
                ),
                "sdv_home_prob": (
                    sdv_values[
                        "home_prob"
                    ]
                ),

                "prediction_generated_at_utc": (
                    prediction_time
                ),
            }
        )

    if len(
        output_rows
    ) != len(
        daily_rows
    ):
        raise RuntimeError(
            "Ensemble output row count "
            "does not equal daily slate"
        )

    output_ids = {
        row[
            "game_id"
        ]
        for row
        in output_rows
    }

    if output_ids != daily_ids:
        raise RuntimeError(
            "Ensemble output game_id "
            "coverage does not exactly "
            "match daily slate"
        )

    write_csv_atomic(
        output_path,
        output_rows,
    )

    log(
        "PREDICT SUCCESS | "
        f"league={label} "
        f"game_date={game_date} "
        f"rows={len(output_rows)} "
        f"path={output_path}"
    )

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn/apply the DRatings + "
            "SDV basketball ensemble."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "train",
            "predict",
        ),
    )

    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(
            LEAGUE_LABELS
        ),
    )

    parser.add_argument(
        "--game-date",
        help=(
            "Prediction date in "
            "YYYY_MM_DD or YYYY-MM-DD."
        ),
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
            "=== BASKETBALL MODEL ENSEMBLE "
            f"{utc_now()} ===\n"
        ),
        encoding="utf-8",
    )

    try:
        leagues = (
            args.league
            or list(
                LEAGUE_LABELS
            )
        )

        if args.mode == "train":
            if args.game_date:
                raise ValueError(
                    "--game-date is invalid "
                    "with --mode train"
                )

            train_selected(
                leagues
            )

            log(
                "STATUS: SUCCESS | "
                "mode=train | "
                f"leagues={len(leagues)}"
            )

            print(
                "Basketball ensemble training "
                "complete: SUCCESS. "
                f"leagues={len(leagues)}"
            )

            return 0

        if len(
            leagues
        ) != 1:
            raise ValueError(
                "--mode predict requires "
                "exactly one --league"
            )

        game_date = file_date(
            args.game_date
        )

        if not game_date:
            raise ValueError(
                "--mode predict requires "
                "valid --game-date"
            )

        output = predict_league_date(
            leagues[
                0
            ],
            game_date,
        )

        log(
            "STATUS: SUCCESS | "
            "mode=predict | "
            f"path={output}"
        )

        print(
            "Basketball ensemble prediction "
            "complete: SUCCESS. "
            f"league="
            f"{LEAGUE_LABELS[leagues[0]]} "
            f"game_date={game_date}"
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
            "Basketball model ensemble "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
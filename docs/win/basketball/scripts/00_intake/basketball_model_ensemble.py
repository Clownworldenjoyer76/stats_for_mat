#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_model_ensemble.py
"""Create fixed 50/50 DRatings + SDV basketball ensemble predictions.

Behavior
--------
- DRatings prediction files remain untouched.
- SDV prediction files remain untouched.
- The script scans prediction folders for matching DRatings + SDV files.
- Games are matched strictly by canonical game_id.
- Only game_ids present in both component files are combined.
- Unmatched files are skipped.
- Unmatched games are skipped.
- No game date is required on the command line.
- Margin uses 50% DRatings + 50% SDV.
- Total uses 50% DRatings + 50% SDV.
- Moneyline probability uses 50% DRatings + 50% SDV.
- Ensemble outputs are written under predictions_ensemble/{league}/.
- --mode train writes the fixed 50/50 weights.json artifacts.
- --mode train does not fit anything from historical results.

Inputs
------
DRatings:
    docs/win/basketball/00_intake/predictions/{league}/
        *_predictions.csv

SDV:
    docs/win/basketball/00_intake/predictions_sdv/{league}/
        *_predictions.csv

SDV metadata:
    docs/win/basketball/models/sdv/{league}/metadata.json

Weights:
    docs/win/basketball/models/ensemble/{league}/weights.json

Outputs
-------
    docs/win/basketball/00_intake/predictions_ensemble/{league}/
        *_predictions.csv
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


BASE = Path("docs/win/basketball")

DRATINGS_ROOT = (
    BASE
    / "00_intake/predictions"
)

SDV_PREDICTIONS_ROOT = (
    BASE
    / "00_intake/predictions_sdv"
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
    "dratings_sdv_ensemble_v1_50_50"
)

DRATINGS_MODEL_VERSION = (
    "dratings_external_unversioned"
)

FIXED_DRATINGS_WEIGHT = 0.50
FIXED_SDV_WEIGHT = 0.50

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
    "feature_version",
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
        return clean(
            value
        )

    return normalized.replace(
        "-",
        "_",
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


def first_nonblank(
    *values: Any,
) -> str:
    for value in values:
        text = clean(
            value
        )

        if text:
            return text

    return ""


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
        return

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
                fieldnames=OUTPUT_FIELDS,
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
        return ""

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

    return {
        "model_version": (
            DRATINGS_MODEL_VERSION
        ),
        "pipeline_version": (
            "dratings_pipeline_"
            f"{digest.hexdigest()[:16]}"
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


def sdv_metadata_path(
    league: str,
) -> Path:
    return (
        SDV_MODEL_ROOT
        / league
        / "metadata.json"
    )


def load_sdv_metadata(
    league: str,
) -> dict[str, Any]:
    path = sdv_metadata_path(
        league
    )

    payload = read_json(
        path
    )

    model_version = clean(
        payload.get(
            "model_version"
        )
    )

    feature_version = clean(
        payload.get(
            "feature_version"
        )
    )

    if not model_version:
        raise RuntimeError(
            f"{league}: SDV metadata "
            "missing model_version"
        )

    if not feature_version:
        raise RuntimeError(
            f"{league}: SDV metadata "
            "missing feature_version"
        )

    return {
        "path": path,
        "model_version": (
            model_version
        ),
        "feature_version": (
            feature_version
        ),
    }


def weights_path(
    league: str,
) -> Path:
    return (
        ENSEMBLE_MODEL_ROOT
        / league
        / "weights.json"
    )


def build_fixed_weights(
    league: str,
) -> dict[str, Any]:
    label = LEAGUE_LABELS[
        league
    ]

    dratings_meta = (
        dratings_pipeline_metadata()
    )

    sdv_meta = (
        load_sdv_metadata(
            league
        )
    )

    return {
        "schema_version": 1,
        "status": "fixed_50_50",
        "ensemble_version": (
            ENSEMBLE_VERSION
        ),
        "league": label,
        "created_at_utc": (
            utc_now()
        ),
        "weight_source": (
            "fixed_50_50"
        ),

        "components": {
            "dratings": (
                dratings_meta
            ),
            "sdv": {
                "model_version": (
                    sdv_meta[
                        "model_version"
                    ]
                ),
                "feature_version": (
                    sdv_meta[
                        "feature_version"
                    ]
                ),
                "metadata_path": str(
                    sdv_meta[
                        "path"
                    ]
                ),
            },
        },

        "training": {
            "performed": False,
            "training_rows": 0,
            "lockbox_used": False,
        },

        "margin": {
            "method": (
                "fixed_equal_weight"
            ),
            "dratings_weight": (
                FIXED_DRATINGS_WEIGHT
            ),
            "sdv_weight": (
                FIXED_SDV_WEIGHT
            ),
        },

        "total": {
            "method": (
                "fixed_equal_weight"
            ),
            "dratings_weight": (
                FIXED_DRATINGS_WEIGHT
            ),
            "sdv_weight": (
                FIXED_SDV_WEIGHT
            ),
        },

        "moneyline": {
            "method": (
                "fixed_equal_weight"
            ),
            "dratings_weight": (
                FIXED_DRATINGS_WEIGHT
            ),
            "sdv_weight": (
                FIXED_SDV_WEIGHT
            ),
        },
    }


def initialize_selected(
    leagues: list[str],
) -> None:
    for league in leagues:
        payload = build_fixed_weights(
            league
        )

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
            f"{LEAGUE_LABELS[league]} | "
            "dratings=0.50 | "
            "sdv=0.50 | "
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
    ) != "fixed_50_50":
        raise RuntimeError(
            f"{league}: weights.json "
            "is not fixed_50_50"
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

    for target in (
        "margin",
        "total",
        "moneyline",
    ):
        section = payload.get(
            target
        )

        if not isinstance(
            section,
            dict,
        ):
            raise RuntimeError(
                f"{league}: missing "
                f"{target} weights"
            )

        dratings_weight = (
            required_float(
                section.get(
                    "dratings_weight"
                ),
                (
                    f"{target}."
                    "dratings_weight"
                ),
            )
        )

        sdv_weight = (
            required_float(
                section.get(
                    "sdv_weight"
                ),
                (
                    f"{target}."
                    "sdv_weight"
                ),
            )
        )

        if (
            abs(
                dratings_weight
                - 0.50
            )
            > 1e-12
            or abs(
                sdv_weight
                - 0.50
            )
            > 1e-12
        ):
            raise RuntimeError(
                f"{league}: {target} "
                "weights are not 50/50"
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

    for (
        row_number,
        raw,
    ) in enumerate(
        rows,
        start=2,
    ):
        game_id = clean_id(
            raw.get(
                "game_id"
            )
        )

        if not game_id:
            log(
                "ROW SKIPPED WITHOUT GAME_ID | "
                f"source={source} | "
                f"row={row_number}"
            )
            continue

        if game_id in result:
            raise RuntimeError(
                f"{source}: duplicate "
                f"game_id={game_id}"
            )

        row = dict(
            raw
        )

        row[
            "game_id"
        ] = game_id

        result[
            game_id
        ] = row

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
            "DRatings row has "
            "no projected margin"
        )

    if total is None:
        raise RuntimeError(
            "DRatings row has "
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
        "expected_margin": (
            margin
        ),
        "expected_total": (
            total
        ),
        "home_prob": (
            home_prob
        ),
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
                "SDV row has "
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
            "SDV row has "
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
            "SDV row has "
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
        "expected_margin": (
            margin
        ),
        "expected_total": (
            total
        ),
        "home_prob": (
            home_prob
        ),
    }


def combine_game(
    *,
    league: str,
    game_id: str,
    dratings: dict[str, Any],
    sdv: dict[str, Any],
    weights: dict[str, Any],
    prediction_time: str,
) -> dict[str, Any]:
    label = LEAGUE_LABELS[
        league
    ]

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

    margin_dratings_weight = float(
        weights[
            "margin"
        ][
            "dratings_weight"
        ]
    )

    margin_sdv_weight = float(
        weights[
            "margin"
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

    total_sdv_weight = float(
        weights[
            "total"
        ][
            "sdv_weight"
        ]
    )

    moneyline_dratings_weight = float(
        weights[
            "moneyline"
        ][
            "dratings_weight"
        ]
    )

    moneyline_sdv_weight = float(
        weights[
            "moneyline"
        ][
            "sdv_weight"
        ]
    )

    expected_margin = (
        margin_dratings_weight
        * dratings_values[
            "expected_margin"
        ]
        + margin_sdv_weight
        * sdv_values[
            "expected_margin"
        ]
    )

    expected_total = (
        total_dratings_weight
        * dratings_values[
            "expected_total"
        ]
        + total_sdv_weight
        * sdv_values[
            "expected_total"
        ]
    )

    home_prob = (
        moneyline_dratings_weight
        * dratings_values[
            "home_prob"
        ]
        + moneyline_sdv_weight
        * sdv_values[
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

    home_projected_points = (
        expected_total
        + expected_margin
    ) / 2.0

    away_projected_points = (
        expected_total
        - expected_margin
    ) / 2.0

    dratings_component = (
        weights[
            "components"
        ][
            "dratings"
        ]
    )

    sdv_component = (
        weights[
            "components"
        ][
            "sdv"
        ]
    )

    sdv_model_version = (
        first_nonblank(
            sdv.get(
                "model_version"
            ),
            sdv_component.get(
                "model_version"
            ),
        )
    )

    sdv_feature_version = (
        first_nonblank(
            sdv.get(
                "feature_version"
            ),
            sdv_component.get(
                "feature_version"
            ),
        )
    )

    game_date_value = (
        first_nonblank(
            dratings.get(
                "game_date"
            ),
            sdv.get(
                "game_date"
            ),
        )
    )

    return {
        "sport": (
            first_nonblank(
                dratings.get(
                    "sport"
                ),
                sdv.get(
                    "sport"
                ),
            )
            or "Basketball"
        ),

        "league": (
            first_nonblank(
                dratings.get(
                    "league"
                ),
                sdv.get(
                    "league"
                ),
            )
            or label
        ),

        "game_id": (
            game_id
        ),

        "game_date": (
            file_date(
                game_date_value
            )
        ),

        "game_time": (
            first_nonblank(
                dratings.get(
                    "game_time"
                ),
                sdv.get(
                    "game_time"
                ),
            )
        ),

        "home_team": (
            first_nonblank(
                dratings.get(
                    "home_team"
                ),
                sdv.get(
                    "home_team"
                ),
            )
        ),

        "away_team": (
            first_nonblank(
                dratings.get(
                    "away_team"
                ),
                sdv.get(
                    "away_team"
                ),
            )
        ),

        "model_source": (
            "ensemble"
        ),

        "model_version": (
            ENSEMBLE_VERSION
        ),

        "feature_version": (
            sdv_feature_version
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

        "home_prob": (
            home_prob
        ),

        "away_prob": (
            away_prob
        ),

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


def prediction_files(
    folder: Path,
    league: str,
) -> dict[str, Path]:
    label = LEAGUE_LABELS[
        league
    ]

    if not folder.exists():
        return {}

    return {
        path.name: path
        for path
        in sorted(
            folder.glob(
                f"*_{label}_predictions.csv"
            )
        )
        if path.is_file()
    }


def process_prediction_file(
    *,
    league: str,
    filename: str,
    dratings_path: Path,
    sdv_path: Path,
    weights: dict[str, Any],
) -> tuple[
    int,
    int,
    int,
]:
    dratings_rows = (
        unique_rows_by_id(
            read_csv_rows(
                dratings_path
            ),
            source=str(
                dratings_path
            ),
        )
    )

    sdv_rows = (
        unique_rows_by_id(
            read_csv_rows(
                sdv_path
            ),
            source=str(
                sdv_path
            ),
        )
    )

    dratings_ids = set(
        dratings_rows
    )

    sdv_ids = set(
        sdv_rows
    )

    matched_ids = sorted(
        dratings_ids
        & sdv_ids
    )

    dratings_only = sorted(
        dratings_ids
        - sdv_ids
    )

    sdv_only = sorted(
        sdv_ids
        - dratings_ids
    )

    if dratings_only:
        log(
            "UNMATCHED DRATINGS GAME_IDS | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"file={filename} | "
            f"count={len(dratings_only)} | "
            f"game_ids={dratings_only}"
        )

    if sdv_only:
        log(
            "UNMATCHED SDV GAME_IDS | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"file={filename} | "
            f"count={len(sdv_only)} | "
            f"game_ids={sdv_only}"
        )

    if not matched_ids:
        log(
            "NO MATCHED GAME_IDS | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"file={filename} | "
            "skipped"
        )

        return (
            0,
            len(
                dratings_only
            ),
            len(
                sdv_only
            ),
        )

    prediction_time = utc_now()

    output_rows: list[
        dict[str, Any]
    ] = []

    for game_id in matched_ids:
        try:
            row = combine_game(
                league=league,
                game_id=game_id,
                dratings=(
                    dratings_rows[
                        game_id
                    ]
                ),
                sdv=(
                    sdv_rows[
                        game_id
                    ]
                ),
                weights=weights,
                prediction_time=(
                    prediction_time
                ),
            )

            output_rows.append(
                row
            )

        except Exception as exc:
            log(
                "GAME SKIPPED | "
                f"league="
                f"{LEAGUE_LABELS[league]} | "
                f"file={filename} | "
                f"game_id={game_id} | "
                f"error={exc}"
            )

    if not output_rows:
        log(
            "NO ENSEMBLE ROWS WRITTEN | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"file={filename}"
        )

        return (
            0,
            len(
                dratings_only
            ),
            len(
                sdv_only
            ),
        )

    output_path = (
        ENSEMBLE_OUTPUT_ROOT
        / league
        / filename
    )

    write_csv_atomic(
        output_path,
        output_rows,
    )

    log(
        "ENSEMBLE FILE WRITTEN | "
        f"league="
        f"{LEAGUE_LABELS[league]} | "
        f"file={filename} | "
        f"rows={len(output_rows)} | "
        f"path={output_path}"
    )

    return (
        len(
            output_rows
        ),
        len(
            dratings_only
        ),
        len(
            sdv_only
        ),
    )


def predict_league(
    league: str,
) -> dict[str, int]:
    weights = load_weights(
        league
    )

    dratings_files = prediction_files(
        DRATINGS_ROOT
        / league,
        league,
    )

    sdv_files = prediction_files(
        SDV_PREDICTIONS_ROOT
        / league,
        league,
    )

    dratings_names = set(
        dratings_files
    )

    sdv_names = set(
        sdv_files
    )

    matching_files = sorted(
        dratings_names
        & sdv_names
    )

    dratings_only_files = sorted(
        dratings_names
        - sdv_names
    )

    sdv_only_files = sorted(
        sdv_names
        - dratings_names
    )

    for filename in dratings_only_files:
        log(
            "DRATINGS FILE WITHOUT SDV MATCH | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"file={filename} | "
            "skipped"
        )

    for filename in sdv_only_files:
        log(
            "SDV FILE WITHOUT DRATINGS MATCH | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"file={filename} | "
            "skipped"
        )

    files_written = 0
    rows_written = 0

    dratings_only_games = 0
    sdv_only_games = 0

    file_errors = 0

    for filename in matching_files:
        try:
            (
                written,
                dratings_only,
                sdv_only,
            ) = process_prediction_file(
                league=league,
                filename=filename,
                dratings_path=(
                    dratings_files[
                        filename
                    ]
                ),
                sdv_path=(
                    sdv_files[
                        filename
                    ]
                ),
                weights=weights,
            )

            if written:
                files_written += 1

            rows_written += (
                written
            )

            dratings_only_games += (
                dratings_only
            )

            sdv_only_games += (
                sdv_only
            )

        except Exception as exc:
            file_errors += 1

            log(
                "FILE SKIPPED AFTER ERROR | "
                f"league="
                f"{LEAGUE_LABELS[league]} | "
                f"file={filename} | "
                f"error={exc}"
            )

            log(
                traceback
                .format_exc()
                .rstrip()
            )

    return {
        "matching_files": len(
            matching_files
        ),

        "files_written": (
            files_written
        ),

        "rows_written": (
            rows_written
        ),

        "dratings_only_files": len(
            dratings_only_files
        ),

        "sdv_only_files": len(
            sdv_only_files
        ),

        "dratings_only_games": (
            dratings_only_games
        ),

        "sdv_only_games": (
            sdv_only_games
        ),

        "file_errors": (
            file_errors
        ),
    }


def predict_selected(
    leagues: list[str],
) -> dict[
    str,
    dict[str, int],
]:
    results: dict[
        str,
        dict[str, int],
    ] = {}

    for league in leagues:
        results[
            league
        ] = predict_league(
            league
        )

        summary = results[
            league
        ]

        log(
            "LEAGUE SUMMARY | "
            f"league="
            f"{LEAGUE_LABELS[league]} | "
            f"matching_files="
            f"{summary['matching_files']} | "
            f"files_written="
            f"{summary['files_written']} | "
            f"rows_written="
            f"{summary['rows_written']} | "
            f"dratings_only_files="
            f"{summary['dratings_only_files']} | "
            f"sdv_only_files="
            f"{summary['sdv_only_files']} | "
            f"file_errors="
            f"{summary['file_errors']}"
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create/apply the fixed "
            "50/50 DRatings + SDV "
            "basketball ensemble."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "train",
            "predict",
        ),
        help=(
            "train writes fixed 50/50 "
            "weights; predict scans all "
            "matching DRatings and SDV "
            "prediction files"
        ),
    )

    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(
            LEAGUE_LABELS
        ),
        help=(
            "League to process. "
            "May be repeated. "
            "If omitted, all leagues "
            "are processed."
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
            initialize_selected(
                leagues
            )

            log(
                "STATUS: SUCCESS | "
                "mode=train | "
                f"leagues={len(leagues)}"
            )

            print(
                "Basketball ensemble 50/50 "
                "weights complete: SUCCESS. "
                f"leagues={len(leagues)}"
            )

            return 0

        results = predict_selected(
            leagues
        )

        total_matching_files = sum(
            row[
                "matching_files"
            ]
            for row
            in results.values()
        )

        total_files_written = sum(
            row[
                "files_written"
            ]
            for row
            in results.values()
        )

        total_rows_written = sum(
            row[
                "rows_written"
            ]
            for row
            in results.values()
        )

        total_file_errors = sum(
            row[
                "file_errors"
            ]
            for row
            in results.values()
        )

        log(
            "STATUS: SUCCESS | "
            "mode=predict | "
            f"matching_files="
            f"{total_matching_files} | "
            f"files_written="
            f"{total_files_written} | "
            f"rows_written="
            f"{total_rows_written} | "
            f"file_errors="
            f"{total_file_errors}"
        )

        print(
            "Basketball ensemble prediction "
            "complete: SUCCESS. "
            f"matching_files="
            f"{total_matching_files} "
            f"files_written="
            f"{total_files_written} "
            f"rows_written="
            f"{total_rows_written} "
            f"file_errors="
            f"{total_file_errors}"
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
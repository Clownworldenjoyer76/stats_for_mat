#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_model_ensemble.py
"""Apply a fixed 50/50 DRatings + SDV basketball ensemble.

The original plan was to learn ensemble weights from historical DRatings + SDV
predictions. Historical DRatings predictions are unavailable, so this version uses
an explicit fixed 50/50 split for both expected margin and expected total.

Running with --mode train does NOT train on game results. It writes the required
50/50 weights.json files for the selected leagues.

Current inputs
--------------
DRatings:
    docs/win/basketball/00_intake/predictions/{league}/
        {game_date}_{LEAGUE}_predictions.csv

SDV:
    docs/win/basketball/00_intake/predictions_sdv/{league}/
        {game_date}_{LEAGUE}_predictions.csv

Canonical slate:
    docs/win/basketball/daily_games/{league}/
        {game_date}_{LEAGUE}.csv

SDV production metadata:
    docs/win/basketball/models/sdv/{league}/metadata.json

Weights:
    docs/win/basketball/models/ensemble/{league}/weights.json

Output:
    docs/win/basketball/00_intake/predictions_ensemble/{league}/
        {game_date}_{LEAGUE}_predictions.csv

Rules
-----
- DRatings and SDV are joined strictly by canonical game_id.
- No team/date composite fallback is permitted.
- Both margin and total use 50% DRatings + 50% SDV.
- Moneyline probability uses 50% DRatings + 50% SDV.
- Current inference requires exact daily-slate coverage from both components.
- Existing DRatings-only and SDV-only files are never overwritten.
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

DRATINGS_ROOT = BASE / "00_intake/predictions"
SDV_PREDICTIONS_ROOT = BASE / "00_intake/predictions_sdv"
DAILY_GAMES_ROOT = BASE / "daily_games"
SDV_MODEL_ROOT = BASE / "models/sdv"
ENSEMBLE_MODEL_ROOT = BASE / "models/ensemble"
ENSEMBLE_OUTPUT_ROOT = BASE / "00_intake/predictions_ensemble"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "basketball_model_ensemble.txt"

DRATINGS_SCRAPER_PATH = (
    BASE / "scripts/00_intake/basketball_drat_scraper.py"
)

DRATINGS_TRANSFORM_PATH = (
    BASE / "scripts/00_intake/transform_basketball.py"
)

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

ENSEMBLE_VERSION = "dratings_sdv_ensemble_v1_50_50"
DRATINGS_MODEL_VERSION = "dratings_external_unversioned"

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

    if (
        clean(
            payload.get(
                "league"
            )
        ).upper()
        != LEAGUE_LABELS[
            league
        ]
    ):
        raise RuntimeError(
            f"{league}: SDV metadata league "
            "does not match expected league"
        )

    return {
        "path": path,
        "model_version": (
            model_version
        ),
        "feature_version": (
            feature_version
        ),
        "metadata": payload,
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
            "manual_fixed_50_50"
        ),
        "reason": (
            "Historical DRatings predictions "
            "required to learn reliable historical "
            "ensemble weights are unavailable. "
            "Production starts with equal weights."
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
            "historical_dratings_available": False,
            "lockbox_used": False,
            "join_key": "game_id",
            "join_policy": (
                "strict_canonical_game_id_only"
            ),
        },
        "margin": {
            "method": (
                "fixed_equal_weight"
            ),
            "sdv_weight": (
                FIXED_SDV_WEIGHT
            ),
            "dratings_weight": (
                FIXED_DRATINGS_WEIGHT
            ),
        },
        "total": {
            "method": (
                "fixed_equal_weight"
            ),
            "sdv_weight": (
                FIXED_SDV_WEIGHT
            ),
            "dratings_weight": (
                FIXED_DRATINGS_WEIGHT
            ),
        },
        "moneyline": {
            "method": (
                "fixed_equal_weight"
            ),
            "sdv_weight": (
                FIXED_SDV_WEIGHT
            ),
            "dratings_weight": (
                FIXED_DRATINGS_WEIGHT
            ),
        },
    }


def initialize_selected(
    leagues: list[str],
) -> None:
    payloads: dict[
        str,
        dict[str, Any],
    ] = {}

    for league in leagues:
        label = LEAGUE_LABELS[
            league
        ]

        log(
            "WEIGHT INIT START | "
            f"league={label}"
        )

        payloads[
            league
        ] = build_fixed_weights(
            league
        )

        log(
            "WEIGHT INIT READY | "
            f"league={label} | "
            "dratings=0.50 | sdv=0.50"
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
            f"{LEAGUE_LABELS[league]} | "
            f"path={path}"
        )


def validate_weight_pair(
    league: str,
    target: str,
    section: dict[str, Any],
) -> tuple[float, float]:
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

    return (
        sdv_weight,
        dratings_weight,
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
            f"{league}: expected fixed_50_50 "
            "ensemble weights; "
            "rerun --mode train"
        )

    if clean(
        payload.get(
            "ensemble_version"
        )
    ) != ENSEMBLE_VERSION:
        raise RuntimeError(
            f"{league}: ensemble version "
            "mismatch; rerun --mode train"
        )

    training = payload.get(
        "training"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise RuntimeError(
            f"{league}: ensemble weight "
            "metadata missing"
        )

    if bool(
        training.get(
            "performed",
            True,
        )
    ):
        raise RuntimeError(
            f"{league}: expected fixed "
            "weights, not trained weights"
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
                f"{league}: {target} "
                "weight section missing"
            )

        (
            sdv_weight,
            dratings_weight,
        ) = validate_weight_pair(
            league,
            target,
            section,
        )

        if (
            abs(
                sdv_weight
                - 0.50
            )
            > 1e-12
            or abs(
                dratings_weight
                - 0.50
            )
            > 1e-12
        ):
            raise RuntimeError(
                f"{league}: {target} "
                "is not 50/50; "
                "rerun --mode train"
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
            f"{league}: DRatings component "
            "metadata missing"
        )

    current_dratings_meta = (
        dratings_pipeline_metadata()
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
            "changed after weights were created; "
            "rerun --mode train"
        )

    sdv_component = components.get(
        "sdv"
    )

    if not isinstance(
        sdv_component,
        dict,
    ):
        raise RuntimeError(
            f"{league}: SDV component "
            "metadata missing"
        )

    current_sdv_meta = (
        load_sdv_metadata(
            league
        )
    )

    if clean(
        sdv_component.get(
            "model_version"
        )
    ) != current_sdv_meta[
        "model_version"
    ]:
        raise RuntimeError(
            f"{league}: SDV model version "
            "changed after weights were created; "
            "rerun --mode train"
        )

    if clean(
        sdv_component.get(
            "feature_version"
        )
    ) != current_sdv_meta[
        "feature_version"
    ]:
        raise RuntimeError(
            f"{league}: SDV feature version "
            "changed after weights were created; "
            "rerun --mode train"
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

    prediction_filename = (
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
        / prediction_filename
    )

    sdv = (
        SDV_PREDICTIONS_ROOT
        / league
        / prediction_filename
    )

    daily = (
        DAILY_GAMES_ROOT
        / league
        / daily_filename
    )

    output = (
        ENSEMBLE_OUTPUT_ROOT
        / league
        / prediction_filename
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

    if not daily_date:
        raise RuntimeError(
            f"game_id={game_id}: "
            "daily slate has invalid "
            "game_date"
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


def validate_sdv_versions(
    rows: dict[str, dict[str, Any]],
    weights: dict[str, Any],
) -> tuple[str, str]:
    components = weights.get(
        "components"
    )

    if not isinstance(
        components,
        dict,
    ):
        raise RuntimeError(
            "Ensemble component "
            "metadata missing"
        )

    sdv_component = components.get(
        "sdv"
    )

    if not isinstance(
        sdv_component,
        dict,
    ):
        raise RuntimeError(
            "Ensemble SDV component "
            "metadata missing"
        )

    expected_model_version = clean(
        sdv_component.get(
            "model_version"
        )
    )

    expected_feature_version = clean(
        sdv_component.get(
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
                f"DRatings "
                f"{dratings_path}"
            ),
        )
    )

    sdv_rows = (
        unique_rows_by_id(
            read_csv_rows(
                sdv_path
            ),
            source=(
                f"SDV "
                f"{sdv_path}"
            ),
        )
    )

    daily_rows = (
        unique_rows_by_id(
            read_csv_rows(
                daily_path
            ),
            source=(
                f"daily_games "
                f"{daily_path}"
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

    if not daily_ids:
        raise RuntimeError(
            "CURRENT SLATE IS EMPTY | "
            f"league={label} | "
            f"date={game_date}"
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

    moneyline_sdv_weight = float(
        weights[
            "moneyline"
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

    dratings_component = (
        weights[
            "components"
        ][
            "dratings"
        ]
    )

    prediction_time = utc_now()

    output_rows: list[
        dict[str, Any]
    ] = []

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
            moneyline_sdv_weight
            * sdv_values[
                "home_prob"
            ]
            + moneyline_dratings_weight
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
                "game_id": (
                    game_id
                ),
                "game_date": (
                    file_date(
                        daily.get(
                            "game_date"
                        )
                    )
                ),
                "game_time": (
                    clean(
                        daily.get(
                            "game_time"
                        )
                    )
                ),
                "home_team": (
                    clean(
                        daily.get(
                            "home_team"
                        )
                    )
                ),
                "away_team": (
                    clean(
                        daily.get(
                            "away_team"
                        )
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
        "weights=50/50 "
        f"path={output_path}"
    )

    return output_path


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
            "weights.json files; predict "
            "combines current DRatings "
            "and SDV predictions"
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

            initialize_selected(
                leagues
            )

            log(
                "STATUS: SUCCESS | "
                "mode=train | "
                f"leagues={len(leagues)} | "
                "fixed_weights=50/50"
            )

            print(
                "Basketball ensemble 50/50 "
                "weights complete: SUCCESS. "
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
            f"game_date={game_date} "
            "weights=50/50"
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
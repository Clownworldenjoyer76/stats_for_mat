#!/usr/bin/env python3
# docs/win/basketball/scripts/testing/basketball_backtest.py
#
# Historical replay/backtest runner for the basketball pipeline.
#
# Expected layout:
#   docs/win/basketball/BACKTEST/
#       configs/markets_test.yaml
#       input/2025_NBA.csv
#       input/2025_NCAAM.csv
#       input/2025_WNBA.csv
#       working/
#       selections/
#       graded/
#       reports/
#       runs/
#
# Permanent model settings:
#   docs/win/basketball/config/model_config.yaml
#
# This script intentionally DOES NOT run the live intake cleaner and DOES NOT
# re-apply or reverse historical prediction bias.
#
# The combined historical input files are treated as frozen historical model
# outputs. The script reproduces the downstream logic of:
#
#   build_juice_files.py
#   compute_ev_kelly.py
#   basketball_select_bets.py
#   01_basketball_results_grade.py
#
# All generated files remain inside BACKTEST/.
#
# Final-score columns are removed before probability, EV/Kelly, and selection
# processing. They are joined back by game_id only after bets have been
# selected. This prevents outcome leakage.
#
# Run from repository root:
#
#   python docs/win/basketball/scripts/testing/basketball_backtest.py
#
# Optional named run:
#
#   python docs/win/basketball/scripts/testing/basketball_backtest.py \
#       --run-name test_001

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import norm


# ============================================================
# CONSTANTS / PATHS
# ============================================================

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

BASKETBALL_ROOT = Path("docs/win/basketball")

DEFAULT_BACKTEST_DIR = BASKETBALL_ROOT / "BACKTEST"

DEFAULT_MODEL_CONFIG = (
    BASKETBALL_ROOT
    / "config"
    / "model_config.yaml"
)

RESULT_COLUMNS = [
    "home_score",
    "away_score",
    "actual_total",
    "actual_home_spread",
    "actual_away_spread",
]

REQUIRED_INPUT_COLUMNS = {
    "game_date",
    "game_id",
    "home_team",
    "away_team",

    "home_spread",
    "away_spread",
    "total",

    "home_dk_moneyline_american",
    "away_dk_moneyline_american",

    "home_dk_spread_american",
    "away_dk_spread_american",

    "dk_total_over_american",
    "dk_total_under_american",

    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",

    "home_dk_spread_decimal",
    "away_dk_spread_decimal",

    "dk_total_over_decimal",
    "dk_total_under_decimal",

    "home_prob",
    "away_prob",

    "home_projected_points",
    "away_projected_points",
    "total_projected_points",

    "home_score",
    "away_score",
}

DEBUG_COUNTS: Counter = Counter()


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_id() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%SZ")


def sanitize_run_name(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_." else "_"
        for ch in value.strip()
    )

    cleaned = cleaned.strip("._")

    if not cleaned:
        raise ValueError(
            "run name becomes empty after sanitization"
        )

    return cleaned


def ensure_mapping(
    value: Any,
    label: str,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must be a mapping"
        )

    return value


def require_number(
    value: Any,
    label: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{label} must be numeric"
        )

    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be numeric; got {value!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{label} must be finite"
        )

    return number


def fv(value: Any) -> float | None:
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        text = str(value).strip()

        if text == "":
            return None

        return float(text)

    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing YAML file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root must be a mapping: {path}"
        )

    return data


def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def clear_directory_contents(
    path: Path,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in path.iterdir():

        if child.is_dir():
            shutil.rmtree(child)

        else:
            child.unlink(
                missing_ok=True
            )


def copy_tree_contents(
    src: Path,
    dst: Path,
) -> None:
    if not src.exists():
        return

    dst.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in src.iterdir():

        target = dst / child.name

        if child.is_dir():
            shutil.copytree(
                child,
                target,
                dirs_exist_ok=True,
            )

        else:
            shutil.copy2(
                child,
                target,
            )


# ============================================================
# LOGGING
# ============================================================

class RunLogger:

    def __init__(
        self,
        path: Path,
    ):
        self.path = path

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                f"=== basketball_backtest RUN "
                f"{now_utc()} ===\n"
            )

    def log(
        self,
        msg: str,
        level: str = "INFO",
    ) -> None:

        line = (
            f"{now_utc()} | "
            f"{level:<5} | "
            f"{msg.rstrip()}"
        )

        print(
            line,
            flush=True,
        )

        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                line + "\n"
            )


# ============================================================
# MODEL CONFIG
# ============================================================

def calibration_cfg(
    league_cfg: dict,
    market: str,
    side: str,
) -> dict:

    market_cfg = (
        (
            league_cfg.get(
                "calibration"
            )
            or {}
        )
        .get(
            market
        )
        or {}
    )

    cfg = (
        market_cfg.get(side)
        or {"method": "none"}
    )

    if isinstance(
        cfg,
        str,
    ):
        cfg = {
            "method": cfg
        }

    if not isinstance(
        cfg,
        dict,
    ):
        raise ValueError(
            f"calibration."
            f"{market}."
            f"{side} "
            f"must be a mapping"
        )

    return cfg


def build_league_settings(
    model_cfg: dict,
) -> dict:

    leagues_cfg = ensure_mapping(
        model_cfg.get("leagues"),
        "model_config.leagues",
    )

    settings = {}

    for league in LEAGUES:

        league_cfg = ensure_mapping(
            leagues_cfg.get(league),
            f"model_config."
            f"leagues."
            f"{league}",
        )

        status = str(
            league_cfg.get(
                "status",
                "",
            )
        ).strip().lower()

        if status != "active":
            raise ValueError(
                f"League "
                f"{league.upper()} "
                f"is not active in "
                f"model_config.yaml"
            )

        edge_cfg = ensure_mapping(
            league_cfg.get(
                "edge"
            )
            or {},
            f"{league}.edge",
        )

        std_cfg = ensure_mapping(
            league_cfg.get(
                "std"
            )
            or {},
            f"{league}.std",
        )

        spread_std_cfg = (
            ensure_mapping(
                std_cfg.get(
                    "spread"
                )
                or {},
                f"{league}."
                f"std.spread",
            )
        )

        total_std_cfg = (
            ensure_mapping(
                std_cfg.get(
                    "total"
                )
                or {},
                f"{league}."
                f"std.total",
            )
        )

        spread_mode = str(
            spread_std_cfg.get(
                "mode",
                "",
            )
        ).strip().lower()

        total_mode = str(
            total_std_cfg.get(
                "mode",
                "",
            )
        ).strip().lower()

        if spread_mode != "fixed":
            raise ValueError(
                f"{league.upper()} "
                f"spread STD mode "
                f"must be fixed"
            )

        if total_mode != "fixed":
            raise ValueError(
                f"{league.upper()} "
                f"total STD mode "
                f"must be fixed"
            )

        settings[league] = {

            "ML_EDGE":
                require_number(
                    edge_cfg.get(
                        "moneyline"
                    ),
                    f"{league}."
                    f"edge.moneyline",
                ),

            "SPREAD_EDGE":
                require_number(
                    edge_cfg.get(
                        "spread"
                    ),
                    f"{league}."
                    f"edge.spread",
                ),

            "TOTAL_EDGE":
                require_number(
                    edge_cfg.get(
                        "total"
                    ),
                    f"{league}."
                    f"edge.total",
                ),

            "SPREAD_STD":
                require_number(
                    spread_std_cfg.get(
                        "value"
                    ),
                    f"{league}."
                    f"std.spread.value",
                ),

            "TOTAL_STD":
                require_number(
                    total_std_cfg.get(
                        "value"
                    ),
                    f"{league}."
                    f"std.total.value",
                ),

            "CALIBRATION": {

                "moneyline": {
                    "home":
                        calibration_cfg(
                            league_cfg,
                            "moneyline",
                            "home",
                        ),
                    "away":
                        calibration_cfg(
                            league_cfg,
                            "moneyline",
                            "away",
                        ),
                },

                "spread": {
                    "home":
                        calibration_cfg(
                            league_cfg,
                            "spread",
                            "home",
                        ),
                    "away":
                        calibration_cfg(
                            league_cfg,
                            "spread",
                            "away",
                        ),
                },

                "total": {
                    "over":
                        calibration_cfg(
                            league_cfg,
                            "total",
                            "over",
                        ),
                    "under":
                        calibration_cfg(
                            league_cfg,
                            "total",
                            "under",
                        ),
                },
            },
        }

    return settings


def apply_calibration(
    p: Any,
    cfg: dict,
) -> float | str:

    if p is None:
        return ""

    if pd.isna(p):
        return ""

    try:
        p = float(p)

    except (
        TypeError,
        ValueError,
    ):
        return ""

    method = str(
        (
            cfg
            or {}
        )
        .get(
            "method",
            "none",
        )
    ).strip().lower()

    if method in {
        "none",
        "raw",
        "",
    }:
        return p

    if method == "beta":

        p = min(
            max(
                p,
                1e-12,
            ),
            1.0 - 1e-12,
        )

        intercept = (
            require_number(
                cfg.get(
                    "intercept"
                ),
                "beta.intercept",
            )
        )

        coef_log_p = (
            require_number(
                cfg.get(
                    "coef_log_p"
                ),
                "beta.coef_log_p",
            )
        )

        coef_log_1mp = (
            require_number(
                cfg.get(
                    "coef_log_1mp"
                ),
                "beta.coef_log_1mp",
            )
        )

        z = (
            intercept
            + coef_log_p
            * math.log(p)
            + coef_log_1mp
            * math.log(
                1.0 - p
            )
        )

        if z >= 0:

            ez = math.exp(
                -z
            )

            return (
                1.0
                / (
                    1.0
                    + ez
                )
            )

        ez = math.exp(z)

        return (
            ez
            / (
                1.0
                + ez
            )
        )

    raise ValueError(
        f"Unsupported "
        f"calibration method: "
        f"{method!r}"
    )


# ============================================================
# ODDS / PROBABILITY HELPERS
# ============================================================

def american_to_decimal(
    odds: Any,
) -> float | str:

    if odds is None:
        return ""

    if pd.isna(odds):
        return ""

    if str(
        odds
    ).strip() == "":
        return ""

    try:
        a = float(odds)

    except (
        TypeError,
        ValueError,
    ):
        return ""

    if a == 0:
        return ""

    if a > 0:
        return (
            1.0
            + (
                a
                / 100.0
            )
        )

    return (
        1.0
        + (
            100.0
            / abs(a)
        )
    )


def american_to_decimal_or_none(
    odds: Any,
) -> float | None:

    value = american_to_decimal(
        odds
    )

    if value == "":
        return None

    return float(value)


def to_american(
    decimal_value: Any,
) -> str:

    if decimal_value is None:
        return ""

    if decimal_value == "":
        return ""

    if pd.isna(
        decimal_value
    ):
        return ""

    try:
        dec = float(
            decimal_value
        )

    except (
        TypeError,
        ValueError,
    ):
        return ""

    if dec <= 1:
        return ""

    if dec >= 2:
        return (
            f"+"
            f"{int((dec - 1) * 100)}"
        )

    return (
        f"-"
        f"{int(100 / (dec - 1))}"
    )


def clamp_probability(
    p: Any,
) -> float:

    return min(
        max(
            float(p),
            0.01,
        ),
        0.99,
    )


def safe_implied_prob(
    decimal_value: Any,
) -> float | str:

    if decimal_value is None:
        return ""

    if decimal_value == "":
        return ""

    if pd.isna(
        decimal_value
    ):
        return ""

    try:
        d = float(
            decimal_value
        )

    except (
        TypeError,
        ValueError,
    ):
        return ""

    if d <= 0:
        return ""

    return (
        1.0
        / d
    )


def devig_pair(
    p_a: Any,
    p_b: Any,
) -> tuple[
    float | str,
    float | str,
]:

    if p_a == "":
        return "", ""

    if p_b == "":
        return "", ""

    if pd.isna(p_a):
        return "", ""

    if pd.isna(p_b):
        return "", ""

    try:
        a = float(p_a)
        b = float(p_b)

    except (
        TypeError,
        ValueError,
    ):
        return "", ""

    total = (
        a
        + b
    )

    if total <= 0:
        return "", ""

    return (
        a / total,
        b / total,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_historical_input(
    df: pd.DataFrame,
    path: Path,
    expected_league: str,
) -> None:

    missing = sorted(
        REQUIRED_INPUT_COLUMNS
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path.name} "
            f"missing required "
            f"columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            f"{path.name} "
            f"is empty"
        )

    blank_ids = (
        df["game_id"]
        .isna()
        | (
            df["game_id"]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    if blank_ids.any():
        raise ValueError(
            f"{path.name} "
            f"contains blank "
            f"game_id values"
        )

    if "league" in df.columns:

        seen = {
            str(x)
            .strip()
            .lower()

            for x in (
                df["league"]
                .dropna()
                .unique()
            )

            if str(x)
            .strip()
        }

        if (
            seen
            and seen
            != {
                expected_league
            }
        ):
            raise ValueError(
                f"{path.name} "
                f"league values "
                f"{sorted(seen)} "
                f"do not match "
                f"expected "
                f"{expected_league}"
            )


def split_features_and_scores(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    score_cols = [
        c
        for c in RESULT_COLUMNS
        if c in df.columns
    ]

    score_df = df[
        [
            "game_id",
            *score_cols,
        ]
    ].copy()

    score_df = (
        score_df
        .drop_duplicates(
            subset=[
                "game_id"
            ],
            keep="last",
        )
    )

    feature_df = (
        df
        .drop(
            columns=score_cols,
            errors="ignore",
        )
        .copy()
    )

    return (
        feature_df,
        score_df,
    )


# ============================================================
# BUILD JUICE - MONEYLINE
# ============================================================

def process_moneyline_juice(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:

    edge = settings[
        "ML_EDGE"
    ]

    cal = settings[
        "CALIBRATION"
    ][
        "moneyline"
    ]

    out = df.copy()

    out[
        "away_decimal"
    ] = (
        out[
            "away_dk_moneyline_american"
        ]
        .apply(
            american_to_decimal
        )
    )

    out[
        "home_decimal"
    ] = (
        out[
            "home_dk_moneyline_american"
        ]
        .apply(
            american_to_decimal
        )
    )

    out[
        "away_implied_prob"
    ] = (
        out[
            "away_decimal"
        ]
        .apply(
            safe_implied_prob
        )
    )

    out[
        "home_implied_prob"
    ] = (
        out[
            "home_decimal"
        ]
        .apply(
            safe_implied_prob
        )
    )

    pairs = out.apply(
        lambda r: devig_pair(
            r[
                "away_implied_prob"
            ],
            r[
                "home_implied_prob"
            ],
        ),
        axis=1,
    )

    out[
        "away_market_prob"
    ] = pairs.apply(
        lambda x: x[0]
    )

    out[
        "home_market_prob"
    ] = pairs.apply(
        lambda x: x[1]
    )

    raw_home = pd.to_numeric(
        out["home_prob"],
        errors="coerce",
    )

    raw_away = pd.to_numeric(
        out["away_prob"],
        errors="coerce",
    )

    out[
        "home_model_prob"
    ] = raw_home.apply(
        lambda p:
        apply_calibration(
            p,
            cal["home"],
        )
    )

    out[
        "away_model_prob"
    ] = raw_away.apply(
        lambda p:
        apply_calibration(
            p,
            cal["away"],
        )
    )

    out[
        "away_fair"
    ] = (
        out[
            "away_model_prob"
        ]
        .apply(
            lambda x:
            (
                1.0
                / float(x)
            )
            if (
                x != ""
                and pd.notna(x)
                and float(x) > 0
            )
            else ""
        )
    )

    out[
        "home_fair"
    ] = (
        out[
            "home_model_prob"
        ]
        .apply(
            lambda x:
            (
                1.0
                / float(x)
            )
            if (
                x != ""
                and pd.notna(x)
                and float(x) > 0
            )
            else ""
        )
    )

    out[
        "away_acceptable_decimal_moneyline"
    ] = (
        out[
            "away_fair"
        ]
        .apply(
            lambda x:
            float(x)
            * (
                1.0
                + edge
            )
            if x != ""
            else ""
        )
    )

    out[
        "home_acceptable_decimal_moneyline"
    ] = (
        out[
            "home_fair"
        ]
        .apply(
            lambda x:
            float(x)
            * (
                1.0
                + edge
            )
            if x != ""
            else ""
        )
    )

    out[
        "away_acceptable_american_moneyline"
    ] = (
        out[
            "away_acceptable_decimal_moneyline"
        ]
        .apply(
            to_american
        )
    )

    out[
        "home_acceptable_american_moneyline"
    ] = (
        out[
            "home_acceptable_decimal_moneyline"
        ]
        .apply(
            to_american
        )
    )

    return out


# ============================================================
# BUILD JUICE - TOTAL
# ============================================================

def process_total_juice(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:

    edge = settings[
        "TOTAL_EDGE"
    ]

    std = settings[
        "TOTAL_STD"
    ]

    cal = settings[
        "CALIBRATION"
    ][
        "total"
    ]

    out = df.copy()

    over_model_prob = []
    under_model_prob = []

    fair_over = []
    fair_under = []

    acceptable_over = []
    acceptable_under = []

    for _, row in out.iterrows():

        line = fv(
            row.get(
                "total"
            )
        )

        mean = fv(
            row.get(
                "total_projected_points"
            )
        )

        if (
            line is None
            or mean is None
        ):

            over_model_prob.append("")
            under_model_prob.append("")

            fair_over.append("")
            fair_under.append("")

            acceptable_over.append("")
            acceptable_under.append("")

            continue

        z = (
            (
                line
                - mean
            )
            / std
        )

        raw_under = (
            clamp_probability(
                norm.cdf(z)
            )
        )

        raw_over = (
            1.0
            - raw_under
        )

        p_over = (
            apply_calibration(
                raw_over,
                cal["over"],
            )
        )

        p_under = (
            apply_calibration(
                raw_under,
                cal["under"],
            )
        )

        over_model_prob.append(
            p_over
        )

        under_model_prob.append(
            p_under
        )

        fair_over_dec = (
            1.0
            / float(
                p_over
            )
        )

        fair_under_dec = (
            1.0
            / float(
                p_under
            )
        )

        fair_over.append(
            fair_over_dec
        )

        fair_under.append(
            fair_under_dec
        )

        acceptable_over.append(
            fair_over_dec
            * (
                1.0
                + edge
            )
        )

        acceptable_under.append(
            fair_under_dec
            * (
                1.0
                + edge
            )
        )

    out[
        "over_model_prob"
    ] = over_model_prob

    out[
        "under_model_prob"
    ] = under_model_prob

    out[
        "fair_over"
    ] = fair_over

    out[
        "fair_under"
    ] = fair_under

    out[
        "acceptable_over"
    ] = acceptable_over

    out[
        "acceptable_under"
    ] = acceptable_under

    out[
        "over_implied_prob"
    ] = (
        out[
            "dk_total_over_decimal"
        ]
        .apply(
            safe_implied_prob
        )
    )

    out[
        "under_implied_prob"
    ] = (
        out[
            "dk_total_under_decimal"
        ]
        .apply(
            safe_implied_prob
        )
    )

    pairs = out.apply(
        lambda r:
        devig_pair(
            r[
                "over_implied_prob"
            ],
            r[
                "under_implied_prob"
            ],
        ),
        axis=1,
    )

    out[
        "over_market_prob"
    ] = pairs.apply(
        lambda x: x[0]
    )

    out[
        "under_market_prob"
    ] = pairs.apply(
        lambda x: x[1]
    )

    return out


# ============================================================
# BUILD JUICE - SPREAD
# ============================================================

def process_spread_juice(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:

    edge = settings[
        "SPREAD_EDGE"
    ]

    std = settings[
        "SPREAD_STD"
    ]

    cal = settings[
        "CALIBRATION"
    ][
        "spread"
    ]

    out = df.copy()

    home_model_prob = []
    away_model_prob = []

    fair_home = []
    fair_away = []

    acceptable_home = []
    acceptable_away = []

    for _, row in out.iterrows():

        home_proj = fv(
            row.get(
                "home_projected_points"
            )
        )

        away_proj = fv(
            row.get(
                "away_projected_points"
            )
        )

        home_line = fv(
            row.get(
                "home_spread"
            )
        )

        if (
            home_proj is None
            or away_proj is None
            or home_line is None
        ):

            home_model_prob.append("")
            away_model_prob.append("")

            fair_home.append("")
            fair_away.append("")

            acceptable_home.append("")
            acceptable_away.append("")

            continue

        mean_margin = (
            home_proj
            - away_proj
        )

        cover_threshold = (
            -home_line
        )

        raw_home = (
            1.0
            - norm.cdf(
                cover_threshold,
                loc=mean_margin,
                scale=std,
            )
        )

        raw_home = (
            clamp_probability(
                raw_home
            )
        )

        raw_away = (
            1.0
            - raw_home
        )

        p_home = (
            apply_calibration(
                raw_home,
                cal["home"],
            )
        )

        p_away = (
            apply_calibration(
                raw_away,
                cal["away"],
            )
        )

        home_model_prob.append(
            p_home
        )

        away_model_prob.append(
            p_away
        )

        fair_home_dec = (
            1.0
            / float(
                p_home
            )
        )

        fair_away_dec = (
            1.0
            / float(
                p_away
            )
        )

        fair_home.append(
            fair_home_dec
        )

        fair_away.append(
            fair_away_dec
        )

        acceptable_home.append(
            fair_home_dec
            * (
                1.0
                + edge
            )
        )

        acceptable_away.append(
            fair_away_dec
            * (
                1.0
                + edge
            )
        )

    out[
        "home_spread_model_prob"
    ] = home_model_prob

    out[
        "away_spread_model_prob"
    ] = away_model_prob

    out[
        "fair_home_spread_decimal"
    ] = fair_home

    out[
        "fair_away_spread_decimal"
    ] = fair_away

    out[
        "home_acceptable_spread_decimal"
    ] = acceptable_home

    out[
        "away_acceptable_spread_decimal"
    ] = acceptable_away

    out[
        "home_acceptable_spread_american"
    ] = (
        out[
            "home_acceptable_spread_decimal"
        ]
        .apply(
            to_american
        )
    )

    out[
        "away_acceptable_spread_american"
    ] = (
        out[
            "away_acceptable_spread_decimal"
        ]
        .apply(
            to_american
        )
    )

    out[
        "home_spread_implied_prob"
    ] = (
        out[
            "home_dk_spread_decimal"
        ]
        .apply(
            safe_implied_prob
        )
    )

    out[
        "away_spread_implied_prob"
    ] = (
        out[
            "away_dk_spread_decimal"
        ]
        .apply(
            safe_implied_prob
        )
    )

    pairs = out.apply(
        lambda r:
        devig_pair(
            r[
                "home_spread_implied_prob"
            ],
            r[
                "away_spread_implied_prob"
            ],
        ),
        axis=1,
    )

    out[
        "home_spread_market_prob"
    ] = pairs.apply(
        lambda x: x[0]
    )

    out[
        "away_spread_market_prob"
    ] = pairs.apply(
        lambda x: x[1]
    )

    return out


# ============================================================
# EV / KELLY HELPERS
# ============================================================

def compute_ev(
    model_prob: Any,
    book_decimal: Any,
) -> float | None:

    p = fv(
        model_prob
    )

    d = fv(
        book_decimal
    )

    if (
        p is None
        or d is None
    ):
        return None

    return (
        p
        * d
        - 1.0
    )


def compute_kelly(
    model_prob: Any,
    book_decimal: Any,
) -> float | None:

    p = fv(
        model_prob
    )

    d = fv(
        book_decimal
    )

    if (
        p is None
        or d is None
        or d <= 1.0
    ):
        return None

    b = (
        d
        - 1.0
    )

    q = (
        1.0
        - p
    )

    k = (
        (
            b
            * p
        )
        - q
    ) / b

    if not math.isfinite(k):
        return None

    return max(
        k,
        0.0,
    )


# ============================================================
# EV / KELLY - MONEYLINE
# ============================================================

def process_moneyline_ev(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out[
        "home_ml_ev"
    ] = out.apply(
        lambda r:
        compute_ev(
            r.get(
                "home_model_prob"
            ),
            r.get(
                "home_dk_moneyline_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "away_ml_ev"
    ] = out.apply(
        lambda r:
        compute_ev(
            r.get(
                "away_model_prob"
            ),
            r.get(
                "away_dk_moneyline_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "home_ml_edge_vs_market"
    ] = (
        pd.to_numeric(
            out[
                "home_model_prob"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            out[
                "home_market_prob"
            ],
            errors="coerce",
        )
    )

    out[
        "away_ml_edge_vs_market"
    ] = (
        pd.to_numeric(
            out[
                "away_model_prob"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            out[
                "away_market_prob"
            ],
            errors="coerce",
        )
    )

    out[
        "home_ml_kelly"
    ] = out.apply(
        lambda r:
        compute_kelly(
            r.get(
                "home_model_prob"
            ),
            r.get(
                "home_dk_moneyline_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "away_ml_kelly"
    ] = out.apply(
        lambda r:
        compute_kelly(
            r.get(
                "away_model_prob"
            ),
            r.get(
                "away_dk_moneyline_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "home_ml_ev_pct"
    ] = (
        out[
            "home_ml_ev"
        ]
        * 100.0
    )

    out[
        "away_ml_ev_pct"
    ] = (
        out[
            "away_ml_ev"
        ]
        * 100.0
    )

    out[
        "home_ml_edge_vs_market_pct"
    ] = (
        out[
            "home_ml_edge_vs_market"
        ]
        * 100.0
    )

    out[
        "away_ml_edge_vs_market_pct"
    ] = (
        out[
            "away_ml_edge_vs_market"
        ]
        * 100.0
    )

    return out


# ============================================================
# EV / KELLY - SPREAD
# ============================================================

def process_spread_ev(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out[
        "home_spread_ev"
    ] = out.apply(
        lambda r:
        compute_ev(
            r.get(
                "home_spread_model_prob"
            ),
            r.get(
                "home_dk_spread_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "away_spread_ev"
    ] = out.apply(
        lambda r:
        compute_ev(
            r.get(
                "away_spread_model_prob"
            ),
            r.get(
                "away_dk_spread_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "home_spread_edge_vs_market"
    ] = (
        pd.to_numeric(
            out[
                "home_spread_model_prob"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            out[
                "home_spread_market_prob"
            ],
            errors="coerce",
        )
    )

    out[
        "away_spread_edge_vs_market"
    ] = (
        pd.to_numeric(
            out[
                "away_spread_model_prob"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            out[
                "away_spread_market_prob"
            ],
            errors="coerce",
        )
    )

    out[
        "home_spread_kelly"
    ] = out.apply(
        lambda r:
        compute_kelly(
            r.get(
                "home_spread_model_prob"
            ),
            r.get(
                "home_dk_spread_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "away_spread_kelly"
    ] = out.apply(
        lambda r:
        compute_kelly(
            r.get(
                "away_spread_model_prob"
            ),
            r.get(
                "away_dk_spread_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "home_spread_ev_pct"
    ] = (
        out[
            "home_spread_ev"
        ]
        * 100.0
    )

    out[
        "away_spread_ev_pct"
    ] = (
        out[
            "away_spread_ev"
        ]
        * 100.0
    )

    out[
        "home_spread_edge_vs_market_pct"
    ] = (
        out[
            "home_spread_edge_vs_market"
        ]
        * 100.0
    )

    out[
        "away_spread_edge_vs_market_pct"
    ] = (
        out[
            "away_spread_edge_vs_market"
        ]
        * 100.0
    )

    return out


# ============================================================
# EV / KELLY - TOTAL
# ============================================================

def process_total_ev(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out[
        "over_ev"
    ] = out.apply(
        lambda r:
        compute_ev(
            r.get(
                "over_model_prob"
            ),
            r.get(
                "dk_total_over_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "under_ev"
    ] = out.apply(
        lambda r:
        compute_ev(
            r.get(
                "under_model_prob"
            ),
            r.get(
                "dk_total_under_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "over_edge_vs_market"
    ] = (
        pd.to_numeric(
            out[
                "over_model_prob"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            out[
                "over_market_prob"
            ],
            errors="coerce",
        )
    )

    out[
        "under_edge_vs_market"
    ] = (
        pd.to_numeric(
            out[
                "under_model_prob"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            out[
                "under_market_prob"
            ],
            errors="coerce",
        )
    )

    out[
        "over_kelly"
    ] = out.apply(
        lambda r:
        compute_kelly(
            r.get(
                "over_model_prob"
            ),
            r.get(
                "dk_total_over_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "under_kelly"
    ] = out.apply(
        lambda r:
        compute_kelly(
            r.get(
                "under_model_prob"
            ),
            r.get(
                "dk_total_under_decimal"
            ),
        ),
        axis=1,
    )

    out[
        "over_ev_pct"
    ] = (
        out[
            "over_ev"
        ]
        * 100.0
    )

    out[
        "under_ev_pct"
    ] = (
        out[
            "under_ev"
        ]
        * 100.0
    )

    out[
        "over_edge_vs_market_pct"
    ] = (
        out[
            "over_edge_vs_market"
        ]
        * 100.0
    )

    out[
        "under_edge_vs_market_pct"
    ] = (
        out[
            "under_edge_vs_market"
        ]
        * 100.0
    )

    return out


# ============================================================
# SELECTION FILTER HELPERS
# ============================================================

def in_any_band(
    value: float | None,
    bands: Any,
) -> bool:

    if value is None:
        return False

    if bands is None:
        return False

    try:
        return any(
            float(lo)
            <= value
            <= float(hi)

            for lo, hi
            in bands
        )

    except Exception:
        return False


def parse_game_date(
    value: Any,
) -> datetime | None:

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(
        value
    ).strip()

    for fmt in (
        "%Y_%m_%d",
        "%Y-%m-%d",
    ):

        try:
            return datetime.strptime(
                text,
                fmt,
            )

        except ValueError:
            continue

    return None


def date_ok(
    game_date: Any,
    months: list,
    exclude_dow: list,
) -> bool:

    if (
        not months
        and not exclude_dow
    ):
        return True

    dt = parse_game_date(
        game_date
    )

    if dt is None:
        return True

    if (
        months
        and dt.month
        not in months
    ):
        DEBUG_COUNTS[
            "fail_month"
        ] += 1

        return False

    if (
        exclude_dow
        and dt.weekday()
        in exclude_dow
    ):
        DEBUG_COUNTS[
            "fail_dow"
        ] += 1

        return False

    return True


def passes_filters(
    values: dict,
    side_cfg: dict,
    game_date: Any,
) -> bool:

    if "odds_bands" in side_cfg:

        if not in_any_band(
            values.get(
                "odds"
            ),
            side_cfg[
                "odds_bands"
            ],
        ):
            DEBUG_COUNTS[
                "fail_odds"
            ] += 1

            return False

    if (
        "line_bands"
        in side_cfg
        and values.get(
            "line"
        )
        is not None
    ):

        if not in_any_band(
            values.get(
                "line"
            ),
            side_cfg[
                "line_bands"
            ],
        ):
            DEBUG_COUNTS[
                "fail_line"
            ] += 1

            return False

    if "ev_bands" in side_cfg:

        if not in_any_band(
            values.get(
                "ev"
            ),
            side_cfg[
                "ev_bands"
            ],
        ):
            DEBUG_COUNTS[
                "fail_ev"
            ] += 1

            return False

    if "kelly_bands" in side_cfg:

        if not in_any_band(
            values.get(
                "kelly"
            ),
            side_cfg[
                "kelly_bands"
            ],
        ):
            DEBUG_COUNTS[
                "fail_kelly"
            ] += 1

            return False

    if (
        "model_prob_bands"
        in side_cfg
    ):

        if not in_any_band(
            values.get(
                "model_prob"
            ),
            side_cfg[
                "model_prob_bands"
            ],
        ):
            DEBUG_COUNTS[
                "fail_model_prob"
            ] += 1

            return False

    if (
        "edge_vs_market_bands"
        in side_cfg
    ):

        if not in_any_band(
            values.get(
                "edge_vs_market_pct"
            ),
            side_cfg[
                "edge_vs_market_bands"
            ],
        ):
            DEBUG_COUNTS[
                "fail_edge_vs_market"
            ] += 1

            return False

    if not date_ok(
        game_date,
        side_cfg.get(
            "months",
            [],
        )
        or [],
        side_cfg.get(
            "exclude_days_of_week",
            [],
        )
        or [],
    ):
        return False

    return True


def model_edge_threshold(
    settings: dict,
    league: str,
    market: str,
) -> float:

    mapping = {
        "moneyline":
            "ML_EDGE",
        "spread":
            "SPREAD_EDGE",
        "total":
            "TOTAL_EDGE",
    }

    return float(
        settings[
            league
        ][
            mapping[
                market
            ]
        ]
    )


def passes_model_edge(
    ev: float | None,
    settings: dict,
    league: str,
    market: str,
) -> bool:

    threshold = (
        model_edge_threshold(
            settings,
            league,
            market,
        )
    )

    if (
        ev is None
        or ev < threshold
    ):

        DEBUG_COUNTS[
            f"fail_model_edge_"
            f"{market}"
        ] += 1

        return False

    return True


def pick_one(
    qualifying: list[dict],
    preference: dict,
) -> dict | None:

    if not qualifying:
        return None

    # Mirrors current production behavior.
    #
    # Candidate keys:
    #   ev
    #   kelly
    #   model_prob
    #   edge_vs_market
    #
    # If markets_test.yaml uses a different metric, such as win_prob,
    # the value is missing and the first qualifying side wins the tie.
    # This is intentional production parity.

    metric = preference.get(
        "metric",
        "ev",
    )

    direction = preference.get(
        "direction",
        "max",
    )

    def key(
        candidate: dict,
    ) -> float:

        value = candidate.get(
            metric
        )

        if value is None:

            if direction == "max":
                return float(
                    "-inf"
                )

            return float(
                "inf"
            )

        return float(value)

    if direction == "max":

        return max(
            qualifying,
            key=key,
        )

    return min(
        qualifying,
        key=key,
    )


def stake_pct(
    kelly: float | None,
    kelly_fraction: float,
    kelly_cap: float,
) -> float | None:

    if (
        kelly is None
        or kelly <= 0
    ):
        return None

    raw = (
        kelly
        * kelly_fraction
    )

    return min(
        raw,
        kelly_cap,
    )


def market_config(
    filter_cfg: dict,
    league: str,
    market: str,
) -> dict:

    try:
        cfg = (
            filter_cfg[
                "markets"
            ][
                league
            ][
                market
            ]
        )

    except KeyError as exc:
        raise KeyError(
            f"No test config "
            f"for league={league} "
            f"market={market}"
        ) from exc

    if not isinstance(
        cfg,
        dict,
    ):
        raise ValueError(
            f"markets."
            f"{league}."
            f"{market} "
            f"must be a mapping"
        )

    return cfg


# ============================================================
# BUILD MONEYLINE SIDES
# ============================================================

def build_moneyline_sides(
    row: pd.Series,
    league: str,
    game_date: Any,
    cfg: dict,
    settings: dict,
) -> list[dict]:

    sides = []

    for side in (
        "home",
        "away",
    ):

        side_cfg = ensure_mapping(
            cfg.get(side),
            f"markets."
            f"{league}."
            f"moneyline."
            f"{side}",
        )

        if not side_cfg.get(
            "enabled",
            True,
        ):
            continue

        odds = fv(
            row.get(
                f"{side}_"
                f"dk_moneyline_american"
            )
        )

        ev = fv(
            row.get(
                f"{side}_"
                f"ml_ev"
            )
        )

        kelly = fv(
            row.get(
                f"{side}_"
                f"ml_kelly"
            )
        )

        model_prob = fv(
            row.get(
                f"{side}_"
                f"model_prob"
            )
        )

        if model_prob is None:

            model_prob = fv(
                row.get(
                    f"{side}_"
                    f"prob"
                )
            )

        edge_vs_market_pct = fv(
            row.get(
                f"{side}_"
                f"ml_edge_vs_market_pct"
            )
        )

        if not passes_model_edge(
            ev,
            settings,
            league,
            "moneyline",
        ):

            DEBUG_COUNTS[
                "rejected_ml"
            ] += 1

            continue

        values = {
            "odds":
                odds,
            "ev":
                ev,
            "kelly":
                kelly,
            "model_prob":
                model_prob,
            "edge_vs_market_pct":
                edge_vs_market_pct,
        }

        if passes_filters(
            values,
            side_cfg,
            game_date,
        ):

            sides.append(
                {
                    "side":
                        side,
                    "line":
                        odds,
                    "odds":
                        odds,
                    "ev":
                        ev,
                    "kelly":
                        kelly,
                    "model_prob":
                        model_prob,
                    "edge_vs_market":
                        edge_vs_market_pct,
                }
            )

        else:

            DEBUG_COUNTS[
                "rejected_ml"
            ] += 1

    return sides


# ============================================================
# BUILD SPREAD SIDES
# ============================================================

def build_spread_sides(
    row: pd.Series,
    league: str,
    game_date: Any,
    cfg: dict,
    settings: dict,
) -> list[dict]:

    sides = []

    for side in (
        "home",
        "away",
    ):

        side_cfg = ensure_mapping(
            cfg.get(side),
            f"markets."
            f"{league}."
            f"spread."
            f"{side}",
        )

        if not side_cfg.get(
            "enabled",
            True,
        ):
            continue

        line = fv(
            row.get(
                f"{side}_spread"
            )
        )

        odds = fv(
            row.get(
                f"{side}_"
                f"dk_spread_american"
            )
        )

        ev = fv(
            row.get(
                f"{side}_"
                f"spread_ev"
            )
        )

        kelly = fv(
            row.get(
                f"{side}_"
                f"spread_kelly"
            )
        )

        model_prob = fv(
            row.get(
                f"{side}_"
                f"spread_model_prob"
            )
        )

        edge_vs_market_pct = fv(
            row.get(
                f"{side}_"
                f"spread_edge_vs_market_pct"
            )
        )

        if not passes_model_edge(
            ev,
            settings,
            league,
            "spread",
        ):

            DEBUG_COUNTS[
                "rejected_spread"
            ] += 1

            continue

        values = {
            "odds":
                odds,
            "line":
                line,
            "ev":
                ev,
            "kelly":
                kelly,
            "model_prob":
                model_prob,
            "edge_vs_market_pct":
                edge_vs_market_pct,
        }

        if passes_filters(
            values,
            side_cfg,
            game_date,
        ):

            sides.append(
                {
                    "side":
                        side,
                    "line":
                        line,
                    "odds":
                        odds,
                    "ev":
                        ev,
                    "kelly":
                        kelly,
                    "model_prob":
                        model_prob,
                    "edge_vs_market":
                        edge_vs_market_pct,
                }
            )

        else:

            DEBUG_COUNTS[
                "rejected_spread"
            ] += 1

    return sides


# ============================================================
# BUILD TOTAL SIDES
# ============================================================

def build_total_sides(
    row: pd.Series,
    league: str,
    game_date: Any,
    cfg: dict,
    settings: dict,
) -> list[dict]:

    sides = []

    line = fv(
        row.get(
            "total"
        )
    )

    for side in (
        "over",
        "under",
    ):

        side_cfg = ensure_mapping(
            cfg.get(side),
            f"markets."
            f"{league}."
            f"total."
            f"{side}",
        )

        if not side_cfg.get(
            "enabled",
            True,
        ):
            continue

        odds = fv(
            row.get(
                f"dk_total_"
                f"{side}_american"
            )
        )

        ev = fv(
            row.get(
                f"{side}_ev"
            )
        )

        kelly = fv(
            row.get(
                f"{side}_kelly"
            )
        )

        model_prob = fv(
            row.get(
                f"{side}_"
                f"model_prob"
            )
        )

        edge_vs_market_pct = fv(
            row.get(
                f"{side}_"
                f"edge_vs_market_pct"
            )
        )

        if not passes_model_edge(
            ev,
            settings,
            league,
            "total",
        ):

            DEBUG_COUNTS[
                "rejected_total"
            ] += 1

            continue

        values = {
            "odds":
                odds,
            "line":
                line,
            "ev":
                ev,
            "kelly":
                kelly,
            "model_prob":
                model_prob,
            "edge_vs_market_pct":
                edge_vs_market_pct,
        }

        if passes_filters(
            values,
            side_cfg,
            game_date,
        ):

            sides.append(
                {
                    "side":
                        side,
                    "line":
                        line,
                    "odds":
                        odds,
                    "ev":
                        ev,
                    "kelly":
                        kelly,
                    "model_prob":
                        model_prob,
                    "edge_vs_market":
                        edge_vs_market_pct,
                }
            )

        else:

            DEBUG_COUNTS[
                "rejected_total"
            ] += 1

    return sides


SIDE_BUILDERS = {
    "moneyline":
        build_moneyline_sides,
    "spread":
        build_spread_sides,
    "total":
        build_total_sides,
}


# ============================================================
# SELECT BETS
# ============================================================

def select_bets_for_market(
    df: pd.DataFrame,
    league: str,
    market: str,
    filter_cfg: dict,
    settings: dict,
    kelly_fraction: float,
    kelly_cap: float,
) -> pd.DataFrame:

    cfg = market_config(
        filter_cfg,
        league,
        market,
    )

    if not cfg.get(
        "enabled",
        True,
    ):
        return pd.DataFrame()

    selection_mode = str(
        cfg.get(
            "selection_mode",
            "pick_one",
        )
    ).strip().lower()

    preference = (
        cfg.get(
            "pick_preference"
        )
        or {
            "metric":
                "ev",
            "direction":
                "max",
        }
    )

    builder = (
        SIDE_BUILDERS[
            market
        ]
    )

    out_rows = []

    for _, row in df.iterrows():

        game_date = row.get(
            "game_date"
        )

        sides = builder(
            row,
            league,
            game_date,
            cfg,
            settings,
        )

        if not sides:
            continue

        if (
            selection_mode
            == "all_qualifying"
        ):

            picks = sides

        else:

            selected = pick_one(
                sides,
                preference,
            )

            picks = (
                [selected]
                if selected
                else []
            )

        for selected in picks:

            DEBUG_COUNTS[
                "selected"
            ] += 1

            record = (
                row.to_dict()
            )

            record.update(
                {
                    "bet_side":
                        selected[
                            "side"
                        ],

                    "bet_line":
                        selected[
                            "line"
                        ],

                    "bet_odds_american":
                        selected[
                            "odds"
                        ],

                    "bet_ev":
                        selected[
                            "ev"
                        ],

                    "bet_kelly":
                        selected[
                            "kelly"
                        ],

                    "bet_model_prob":
                        selected[
                            "model_prob"
                        ],

                    "bet_edge_vs_market":
                        selected[
                            "edge_vs_market"
                        ],

                    "bet_stake_pct":
                        stake_pct(
                            selected[
                                "kelly"
                            ],
                            kelly_fraction,
                            kelly_cap,
                        ),

                    "market_type":
                        market,

                    "league_lower":
                        league,

                    "league":
                        league.upper(),

                    "game_date":
                        game_date,
                }
            )

            out_rows.append(
                record
            )

    return pd.DataFrame(
        out_rows
    )


# ============================================================
# GRADING
# ============================================================

def determine_outcome(
    row: pd.Series,
) -> str:

    market = str(
        row.get(
            "market_type",
            "",
        )
    ).lower()

    side = str(
        row.get(
            "bet_side",
            "",
        )
    ).lower()

    home = fv(
        row.get(
            "home_score"
        )
    )

    away = fv(
        row.get(
            "away_score"
        )
    )

    if (
        home is None
        or away is None
    ):
        return "Unknown"

    # MONEYLINE
    if market == "moneyline":

        if home == away:
            return "Push"

        home_won = (
            home
            > away
        )

        if (
            (
                side == "home"
                and home_won
            )
            or
            (
                side == "away"
                and not home_won
            )
        ):
            return "Win"

        return "Loss"

    # SPREAD
    if market == "spread":

        line = fv(
            row.get(
                "bet_line"
            )
        )

        if line is None:
            return "Unknown"

        if side == "home":

            diff = (
                home
                + line
                - away
            )

        elif side == "away":

            diff = (
                away
                + line
                - home
            )

        else:
            return "Unknown"

        if abs(
            diff
        ) < 1e-9:
            return "Push"

        if diff > 0:
            return "Win"

        return "Loss"

    # TOTAL
    if market == "total":

        line = fv(
            row.get(
                "bet_line"
            )
        )

        if line is None:
            return "Unknown"

        actual_total = (
            home
            + away
        )

        if abs(
            actual_total
            - line
        ) < 1e-9:

            return "Push"

        if (
            actual_total
            > line
            and side == "over"
        ):

            return "Win"

        if (
            actual_total
            < line
            and side == "under"
        ):

            return "Win"

        return "Loss"

    return "Unknown"


def compute_profits(
    row: pd.Series,
) -> tuple[
    float | None,
    float | None,
]:

    result = str(
        row.get(
            "bet_result",
            "",
        )
    ).strip()

    decimal = (
        american_to_decimal_or_none(
            row.get(
                "bet_odds_american"
            )
        )
    )

    if result == "Push":
        return (
            0.0,
            0.0,
        )

    if result not in (
        "Win",
        "Loss",
    ):
        return (
            None,
            None,
        )

    if (
        decimal is None
        or decimal <= 1.0
    ):
        return (
            None,
            None,
        )

    stake = fv(
        row.get(
            "bet_stake_pct"
        )
    )

    if result == "Win":

        profit_unit = (
            decimal
            - 1.0
        )

        profit_kelly = (
            stake
            * (
                decimal
                - 1.0
            )
            if stake is not None
            else None
        )

    else:

        profit_unit = -1.0

        profit_kelly = (
            -stake
            if stake is not None
            else None
        )

    return (
        profit_unit,
        profit_kelly,
    )


def grade_selections(
    selections: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:

    if selections.empty:
        return pd.DataFrame()

    merged = selections.merge(
        scores,
        on="game_id",
        how="left",
        suffixes=(
            "",
            "_score",
        ),
    )

    for col in RESULT_COLUMNS:

        score_col = (
            f"{col}_score"
        )

        if score_col in merged.columns:

            if col in merged.columns:

                merged[col] = (
                    merged[
                        score_col
                    ]
                    .combine_first(
                        merged[
                            col
                        ]
                    )
                )

            else:

                merged[col] = (
                    merged[
                        score_col
                    ]
                )

            merged = merged.drop(
                columns=[
                    score_col
                ]
            )

    merged[
        "bet_result"
    ] = merged.apply(
        determine_outcome,
        axis=1,
    )

    profits = merged.apply(
        compute_profits,
        axis=1,
        result_type="expand",
    )

    profits.columns = [
        "profit_unit",
        "profit_kelly",
    ]

    merged = pd.concat(
        [
            merged,
            profits,
        ],
        axis=1,
    )

    key_cols = [
        c
        for c in [
            "source_file",
            "game_id",
            "market_type",
            "bet_side",
        ]
        if c in merged.columns
    ]

    if key_cols:

        merged = (
            merged
            .drop_duplicates(
                subset=key_cols,
                keep="last",
            )
        )

    return merged


# ============================================================
# REPORTING
# ============================================================

def summarize_group(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:

    columns = [
        *group_cols,

        "bets",
        "wins",
        "losses",
        "pushes",
        "unknown",

        "win_rate",

        "profit_units",
        "roi_units",

        "profit_kelly",
        "kelly_staked",
        "roi_kelly",

        "avg_ev",
        "avg_kelly",
        "avg_model_prob",
        "avg_edge_vs_market",
        "avg_odds_american",
    ]

    if df.empty:
        return pd.DataFrame(
            columns=columns
        )

    records = []

    if group_cols:

        grouped = df.groupby(
            group_cols,
            dropna=False,
            sort=True,
        )

    else:

        grouped = [
            (
                (),
                df,
            )
        ]

    for keys, group in grouped:

        if (
            group_cols
            and not isinstance(
                keys,
                tuple,
            )
        ):

            keys = (
                keys,
            )

        elif not group_cols:

            keys = ()

        wins = int(
            (
                group[
                    "bet_result"
                ]
                == "Win"
            )
            .sum()
        )

        losses = int(
            (
                group[
                    "bet_result"
                ]
                == "Loss"
            )
            .sum()
        )

        pushes = int(
            (
                group[
                    "bet_result"
                ]
                == "Push"
            )
            .sum()
        )

        unknown = int(
            (
                group[
                    "bet_result"
                ]
                == "Unknown"
            )
            .sum()
        )

        bets = (
            wins
            + losses
            + pushes
            + unknown
        )

        decisions = (
            wins
            + losses
        )

        graded_stakes = (
            wins
            + losses
            + pushes
        )

        profit_units = float(
            pd.to_numeric(
                group[
                    "profit_unit"
                ],
                errors="coerce",
            )
            .sum(
                skipna=True
            )
        )

        profit_kelly = float(
            pd.to_numeric(
                group[
                    "profit_kelly"
                ],
                errors="coerce",
            )
            .sum(
                skipna=True
            )
        )

        stake_series = (
            pd.to_numeric(
                group[
                    "bet_stake_pct"
                ],
                errors="coerce",
            )
        )

        kelly_staked = float(
            stake_series
            .fillna(
                0.0
            )
            .sum()
        )

        record = {
            col:
                value

            for col, value
            in zip(
                group_cols,
                keys,
            )
        }

        record.update(
            {
                "bets":
                    bets,

                "wins":
                    wins,

                "losses":
                    losses,

                "pushes":
                    pushes,

                "unknown":
                    unknown,

                "win_rate":
                    (
                        wins
                        / decisions
                    )
                    if decisions
                    else None,

                "profit_units":
                    profit_units,

                "roi_units":
                    (
                        profit_units
                        / graded_stakes
                    )
                    if graded_stakes
                    else None,

                "profit_kelly":
                    profit_kelly,

                "kelly_staked":
                    kelly_staked,

                "roi_kelly":
                    (
                        profit_kelly
                        / kelly_staked
                    )
                    if kelly_staked > 0
                    else None,

                "avg_ev":
                    pd.to_numeric(
                        group[
                            "bet_ev"
                        ],
                        errors="coerce",
                    )
                    .mean(),

                "avg_kelly":
                    pd.to_numeric(
                        group[
                            "bet_kelly"
                        ],
                        errors="coerce",
                    )
                    .mean(),

                "avg_model_prob":
                    pd.to_numeric(
                        group[
                            "bet_model_prob"
                        ],
                        errors="coerce",
                    )
                    .mean(),

                "avg_edge_vs_market":
                    pd.to_numeric(
                        group[
                            "bet_edge_vs_market"
                        ],
                        errors="coerce",
                    )
                    .mean(),

                "avg_odds_american":
                    pd.to_numeric(
                        group[
                            "bet_odds_american"
                        ],
                        errors="coerce",
                    )
                    .mean(),
            }
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records,
        columns=columns,
    )


def build_reports(
    graded: pd.DataFrame,
    reports_dir: Path,
) -> dict[
    str,
    pd.DataFrame,
]:

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {

        "overall":
            summarize_group(
                graded,
                [],
            ),

        "by_source":
            summarize_group(
                graded,
                [
                    "source_file",
                    "league",
                ],
            ),

        "by_league":
            summarize_group(
                graded,
                [
                    "league",
                ],
            ),

        "by_market":
            summarize_group(
                graded,
                [
                    "league",
                    "market_type",
                ],
            ),

        "by_market_side":
            summarize_group(
                graded,
                [
                    "league",
                    "market_type",
                    "bet_side",
                ],
            ),
    }

    filenames = {
        "overall":
            "overall.csv",

        "by_source":
            "performance_by_source.csv",

        "by_league":
            "performance_by_league.csv",

        "by_market":
            "performance_by_market.csv",

        "by_market_side":
            "performance_by_market_side.csv",
    }

    for (
        name,
        report,
    ) in reports.items():

        atomic_write_csv(
            report,
            reports_dir
            / filenames[
                name
            ],
        )

    reject_df = pd.DataFrame(
        [
            {
                "reason":
                    key,
                "count":
                    value,
            }

            for key, value
            in sorted(
                DEBUG_COUNTS.items()
            )
        ]
    )

    atomic_write_csv(
        reject_df,
        reports_dir
        / "filter_counts.csv",
    )

    return reports


# ============================================================
# CONFIG DIAGNOSTICS
# ============================================================

def collect_config_warnings(
    filter_cfg: dict,
) -> list[str]:

    warnings = []

    supported_pick_metrics = {
        "ev",
        "kelly",
        "model_prob",
        "edge_vs_market",
    }

    markets = (
        filter_cfg.get(
            "markets"
        )
        or {}
    )

    for league in LEAGUES:

        league_cfg = (
            markets.get(
                league
            )
            or {}
        )

        for market in MARKETS:

            market_cfg = (
                league_cfg.get(
                    market
                )
                or {}
            )

            preference = (
                market_cfg.get(
                    "pick_preference"
                )
                or {}
            )

            metric = str(
                preference.get(
                    "metric",
                    "ev",
                )
            ).strip()

            if (
                metric
                not in supported_pick_metrics
            ):

                warnings.append(
                    f"markets."
                    f"{league}."
                    f"{market}."
                    f"pick_preference."
                    f"metric="
                    f"{metric!r} "
                    f"is not a production "
                    f"candidate key; "
                    f"production falls back "
                    f"to first qualifying "
                    f"side on ties"
                )

            for (
                side_name,
                side_cfg,
            ) in market_cfg.items():

                if side_name not in {
                    "home",
                    "away",
                    "over",
                    "under",
                }:
                    continue

                if not isinstance(
                    side_cfg,
                    dict,
                ):
                    continue

                for band_name in (
                    "odds_bands",
                    "line_bands",
                    "ev_bands",
                    "kelly_bands",
                    "model_prob_bands",
                    "edge_vs_market_bands",
                ):

                    bands = (
                        side_cfg.get(
                            band_name
                        )
                    )

                    if not isinstance(
                        bands,
                        list,
                    ):
                        continue

                    for (
                        idx,
                        band,
                    ) in enumerate(
                        bands
                    ):

                        if (
                            not isinstance(
                                band,
                                (
                                    list,
                                    tuple,
                                ),
                            )
                            or len(
                                band
                            )
                            != 2
                        ):

                            warnings.append(
                                f"markets."
                                f"{league}."
                                f"{market}."
                                f"{side_name}."
                                f"{band_name}"
                                f"[{idx}] "
                                f"is not "
                                f"[min, max]"
                            )

                            continue

                        lo = fv(
                            band[0]
                        )

                        hi = fv(
                            band[1]
                        )

                        if (
                            lo is not None
                            and hi is not None
                            and lo > hi
                        ):

                            warnings.append(
                                f"markets."
                                f"{league}."
                                f"{market}."
                                f"{side_name}."
                                f"{band_name}"
                                f"[{idx}] "
                                f"has min > max "
                                f"({lo} > {hi}); "
                                f"it will match "
                                f"nothing"
                            )

    return warnings


# ============================================================
# RUN MANIFEST
# ============================================================

def write_manifest(
    path: Path,
    run_id: str,
    model_config_path: Path,
    filter_config_path: Path,
    input_files: list[Path],
    settings: dict,
    config_warnings: list[str],
    total_rows: int,
    total_selected: int,
    total_graded: int,
) -> None:

    manifest = {

        "schema_version":
            1,

        "run_id":
            run_id,

        "generated_at_utc":
            now_utc(),

        "backtest_method":
            (
                "frozen_historical_predictions_"
                "current_downstream_logic"
            ),

        "historical_bias_handling":
            (
                "preserve_stored_predictions_"
                "no_rebias"
            ),

        "outcome_leakage_prevention":
            (
                "final_score_columns_removed_"
                "before_selection_and_rejoined_"
                "by_game_id_after_selection"
            ),

        "ml_vs_spread_reconciliation":
            (
                "not_applied_to_match_current_"
                "production_selector"
            ),

        "model_config": {
            "path":
                str(
                    model_config_path
                ),

            "sha256":
                sha256_file(
                    model_config_path
                ),
        },

        "filter_config": {
            "path":
                str(
                    filter_config_path
                ),

            "sha256":
                sha256_file(
                    filter_config_path
                ),
        },

        "input_files": [
            {
                "path":
                    str(p),

                "sha256":
                    sha256_file(p),

                "size_bytes":
                    p.stat().st_size,
            }

            for p
            in input_files
        ],

        "model_settings":
            settings,

        "config_warnings":
            config_warnings,

        "counts": {
            "historical_rows":
                total_rows,

            "selected_bets":
                total_selected,

            "graded_bets":
                total_graded,
        },

        "filter_counts":
            dict(
                sorted(
                    DEBUG_COUNTS.items()
                )
            ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


# ============================================================
# RUN INDEX
# ============================================================

def append_run_index(
    index_path: Path,
    run_id: str,
    overall: pd.DataFrame,
) -> None:

    row = {
        "run_id":
            run_id,

        "generated_at_utc":
            now_utc(),

        "bets":
            0,

        "wins":
            0,

        "losses":
            0,

        "pushes":
            0,

        "unknown":
            0,

        "win_rate":
            None,

        "profit_units":
            0.0,

        "roi_units":
            None,

        "profit_kelly":
            0.0,

        "roi_kelly":
            None,
    }

    if not overall.empty:

        first = (
            overall.iloc[0]
        )

        for key in row.keys():

            if key in {
                "run_id",
                "generated_at_utc",
            }:
                continue

            if key in first.index:
                row[key] = (
                    first[key]
                )

    new = pd.DataFrame(
        [row]
    )

    if index_path.exists():

        old = pd.read_csv(
            index_path
        )

        combined = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    else:

        combined = new

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "run_id"
            ],
            keep="last",
        )
    )

    atomic_write_csv(
        combined,
        index_path,
    )


# ============================================================
# PROCESS ONE HISTORICAL FILE
# ============================================================

def process_historical_file(
    path: Path,
    league: str,
    settings: dict,
    filter_cfg: dict,
    working_dir: Path,
    selections_dir: Path,
    graded_dir: Path,
    kelly_fraction: float,
    kelly_cap: float,
    logger: RunLogger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:

    source_file = (
        path.stem
    )

    logger.log(
        f"[{league.upper()}] "
        f"reading {path}"
    )

    raw = pd.read_csv(
        path,
        dtype={
            "game_id":
                str,

            "game_date":
                str,
        },
    )

    validate_historical_input(
        raw,
        path,
        league,
    )

    raw[
        "source_file"
    ] = source_file

    (
        feature_df,
        scores,
    ) = split_features_and_scores(
        raw
    )

    feature_df[
        "source_file"
    ] = source_file

    scores[
        "source_file"
    ] = source_file

    market_frames = {}

    # MONEYLINE

    ml = (
        process_moneyline_juice(
            feature_df,
            settings[
                league
            ],
        )
    )

    ml = (
        process_moneyline_ev(
            ml
        )
    )

    market_frames[
        "moneyline"
    ] = ml

    # SPREAD

    spread = (
        process_spread_juice(
            feature_df,
            settings[
                league
            ],
        )
    )

    spread = (
        process_spread_ev(
            spread
        )
    )

    market_frames[
        "spread"
    ] = spread

    # TOTAL

    total = (
        process_total_juice(
            feature_df,
            settings[
                league
            ],
        )
    )

    total = (
        process_total_ev(
            total
        )
    )

    market_frames[
        "total"
    ] = total

    selected_parts = []

    for (
        market,
        market_df,
    ) in market_frames.items():

        work_path = (
            working_dir
            / league
            / market
            / (
                f"{source_file}_"
                f"{market}.csv"
            )
        )

        atomic_write_csv(
            market_df,
            work_path,
        )

        selected = (
            select_bets_for_market(
                market_df,
                league,
                market,
                filter_cfg,
                settings,
                kelly_fraction,
                kelly_cap,
            )
        )

        if not selected.empty:

            selected_parts.append(
                selected
            )

        logger.log(
            f"[{league.upper()}] "
            f"{source_file} "
            f"{market}: "
            f"rows={len(market_df)} "
            f"selected={len(selected)}"
        )

    if selected_parts:

        selections = pd.concat(
            selected_parts,
            ignore_index=True,
        )

    else:

        selections = (
            pd.DataFrame()
        )

    selection_path = (
        selections_dir
        / league
        / (
            f"{source_file}_"
            f"selected.csv"
        )
    )

    atomic_write_csv(
        selections,
        selection_path,
    )

    if selections.empty:

        graded = (
            pd.DataFrame()
        )

    else:

        score_cols = [
            c
            for c in scores.columns
            if c != "source_file"
        ]

        graded = (
            grade_selections(
                selections,
                scores[
                    score_cols
                ],
            )
        )

    graded_path = (
        graded_dir
        / league
        / (
            f"{source_file}_"
            f"graded.csv"
        )
    )

    atomic_write_csv(
        graded,
        graded_path,
    )

    logger.log(
        f"[{league.upper()}] "
        f"{source_file}: "
        f"historical_rows="
        f"{len(raw)} "
        f"selected="
        f"{len(selections)} "
        f"graded="
        f"{len(graded)}"
    )

    return (
        selections,
        graded,
        len(raw),
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Replay combined historical "
                "basketball files through "
                "current downstream model "
                "and selection logic."
            )
        )
    )

    parser.add_argument(
        "--backtest-dir",
        default=str(
            DEFAULT_BACKTEST_DIR
        ),
        help=(
            f"Backtest root directory "
            f"(default: "
            f"{DEFAULT_BACKTEST_DIR})"
        ),
    )

    parser.add_argument(
        "--model-config",
        default=str(
            DEFAULT_MODEL_CONFIG
        ),
        help=(
            f"Production model config "
            f"(default: "
            f"{DEFAULT_MODEL_CONFIG})"
        ),
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional immutable run "
            "snapshot name. "
            "Default is UTC timestamp."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    backtest_dir = Path(
        args.backtest_dir
    )

    input_dir = (
        backtest_dir
        / "input"
    )

    configs_dir = (
        backtest_dir
        / "configs"
    )

    working_dir = (
        backtest_dir
        / "working"
    )

    selections_dir = (
        backtest_dir
        / "selections"
    )

    graded_dir = (
        backtest_dir
        / "graded"
    )

    reports_dir = (
        backtest_dir
        / "reports"
    )

    runs_dir = (
        backtest_dir
        / "runs"
    )

    model_config_path = Path(
        args.model_config
    )

    filter_config_path = (
        configs_dir
        / "markets_test.yaml"
    )

    for folder in (
        input_dir,
        configs_dir,
        working_dir,
        selections_dir,
        graded_dir,
        reports_dir,
        runs_dir,
    ):

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    if args.run_name:

        run_id = (
            sanitize_run_name(
                args.run_name
            )
        )

    else:

        run_id = (
            timestamp_id()
        )

    run_dir = (
        runs_dir
        / run_id
    )

    if run_dir.exists():

        raise FileExistsError(
            f"Run snapshot "
            f"already exists: "
            f"{run_dir}"
        )

    # Current-view outputs are replaced each run.
    # Prior immutable run snapshots remain untouched.

    clear_directory_contents(
        working_dir
    )

    clear_directory_contents(
        selections_dir
    )

    clear_directory_contents(
        graded_dir
    )

    clear_directory_contents(
        reports_dir
    )

    logger = RunLogger(
        reports_dir
        / "basketball_backtest.txt"
    )

    logger.log(
        f"run_id={run_id}"
    )

    logger.log(
        f"backtest_dir="
        f"{backtest_dir}"
    )

    logger.log(
        f"model_config="
        f"{model_config_path}"
    )

    logger.log(
        f"filter_config="
        f"{filter_config_path}"
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    model_cfg = read_yaml(
        model_config_path
    )

    filter_cfg = read_yaml(
        filter_config_path
    )

    settings = (
        build_league_settings(
            model_cfg
        )
    )

    stake_cfg = (
        filter_cfg.get(
            "stake_sizing"
        )
        or {}
    )

    kelly_fraction = (
        require_number(
            stake_cfg.get(
                "kelly_fraction",
                1.0,
            ),
            "stake_sizing."
            "kelly_fraction",
        )
    )

    kelly_cap = (
        require_number(
            stake_cfg.get(
                "kelly_cap",
                1.0,
            ),
            "stake_sizing."
            "kelly_cap",
        )
    )

    # --------------------------------------------------------
    # CONFIG WARNINGS
    # --------------------------------------------------------

    config_warnings = (
        collect_config_warnings(
            filter_cfg
        )
    )

    for warning in config_warnings:

        logger.log(
            warning,
            "WARN",
        )

    # --------------------------------------------------------
    # INPUT DISCOVERY
    # --------------------------------------------------------

    input_files = []

    for league in LEAGUES:

        league_files = sorted(
            input_dir.glob(
                f"*_{league.upper()}.csv"
            )
        )

        if not league_files:

            raise FileNotFoundError(
                f"No historical input "
                f"files found for "
                f"{league.upper()} "
                f"in {input_dir}"
            )

        input_files.extend(
            league_files
        )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    all_selections = []
    all_graded = []

    total_rows = 0

    for league in LEAGUES:

        files = sorted(
            input_dir.glob(
                f"*_{league.upper()}.csv"
            )
        )

        for path in files:

            (
                selections,
                graded,
                row_count,
            ) = process_historical_file(
                path=path,
                league=league,
                settings=settings,
                filter_cfg=filter_cfg,
                working_dir=working_dir,
                selections_dir=selections_dir,
                graded_dir=graded_dir,
                kelly_fraction=kelly_fraction,
                kelly_cap=kelly_cap,
                logger=logger,
            )

            total_rows += (
                row_count
            )

            if not selections.empty:

                all_selections.append(
                    selections
                )

            if not graded.empty:

                all_graded.append(
                    graded
                )

    # --------------------------------------------------------
    # COMBINED OUTPUTS
    # --------------------------------------------------------

    if all_selections:

        combined_selected = (
            pd.concat(
                all_selections,
                ignore_index=True,
            )
        )

    else:

        combined_selected = (
            pd.DataFrame()
        )

    if all_graded:

        combined_graded = (
            pd.concat(
                all_graded,
                ignore_index=True,
            )
        )

    else:

        combined_graded = (
            pd.DataFrame()
        )

    atomic_write_csv(
        combined_selected,
        selections_dir
        / "all_selected.csv",
    )

    atomic_write_csv(
        combined_graded,
        graded_dir
        / "all_graded.csv",
    )

    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------

    reports = build_reports(
        combined_graded,
        reports_dir,
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    write_manifest(
        path=(
            reports_dir
            / "run_manifest.yaml"
        ),
        run_id=run_id,
        model_config_path=model_config_path,
        filter_config_path=filter_config_path,
        input_files=input_files,
        settings=settings,
        config_warnings=config_warnings,
        total_rows=total_rows,
        total_selected=len(
            combined_selected
        ),
        total_graded=len(
            combined_graded
        ),
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    logger.log(
        "--- FINAL SUMMARY ---"
    )

    logger.log(
        f"historical_rows="
        f"{total_rows}"
    )

    logger.log(
        f"selected_bets="
        f"{len(combined_selected)}"
    )

    logger.log(
        f"graded_bets="
        f"{len(combined_graded)}"
    )

    if not reports[
        "overall"
    ].empty:

        row = reports[
            "overall"
        ].iloc[0]

        wins = int(
            row["wins"]
        )

        losses = int(
            row["losses"]
        )

        pushes = int(
            row["pushes"]
        )

        unknown = int(
            row["unknown"]
        )

        profit_units = float(
            row[
                "profit_units"
            ]
        )

        roi_units = row[
            "roi_units"
        ]

        if pd.notna(
            roi_units
        ):

            logger.log(
                f"W/L/P/U="
                f"{wins}/"
                f"{losses}/"
                f"{pushes}/"
                f"{unknown} "
                f"profit_units="
                f"{profit_units:+.4f} "
                f"roi_units="
                f"{float(roi_units):+.4%}"
            )

        else:

            logger.log(
                f"W/L/P/U="
                f"{wins}/"
                f"{losses}/"
                f"{pushes}/"
                f"{unknown} "
                f"profit_units="
                f"{profit_units:+.4f} "
                f"roi_units=N/A"
            )

    logger.log(
        f"run_snapshot="
        f"{run_dir}"
    )

    logger.log(
        "STATUS: SUCCESS"
    )

    # --------------------------------------------------------
    # IMMUTABLE RUN SNAPSHOT
    # --------------------------------------------------------

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        filter_config_path,
        run_dir
        / "markets_test.yaml",
    )

    shutil.copy2(
        model_config_path,
        run_dir
        / "model_config.yaml",
    )

    copy_tree_contents(
        reports_dir,
        run_dir
        / "reports",
    )

    copy_tree_contents(
        selections_dir,
        run_dir
        / "selections",
    )

    copy_tree_contents(
        graded_dir,
        run_dir
        / "graded",
    )

    # --------------------------------------------------------
    # RUN COMPARISON INDEX
    # --------------------------------------------------------

    append_run_index(
        runs_dir
        / "index.csv",
        run_id,
        reports[
            "overall"
        ],
    )

    print(
        "basketball_backtest complete."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"STATUS: FAILED | "
            f"{exc}",
            file=sys.stderr,
        )

        traceback.print_exc()

        sys.exit(1)

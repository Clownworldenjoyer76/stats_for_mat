#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[5]

BASE = (
    REPO_ROOT
    / "docs/win/basketball"
)

BACKTEST = (
    BASE
    / "scripts/testing/basketball_backtest.py"
)

PRODUCTION_MODEL_CONFIG = (
    BASE
    / "config/model_config.yaml"
)

BACKTEST_ROOT = (
    BASE
    / "backtest"
)

CONFIG_DIR = (
    BACKTEST_ROOT
    / "configs"
)

RUNS_DIR = (
    BACKTEST_ROOT
    / "runs"
)

OUTPUT_ROOT = (
    BACKTEST_ROOT
    / "nba_margin_bias_150_end_to_end"
)

METRICS = {
    "model_probability":
        "bet_model_prob",

    "raw_ev":
        "bet_raw_ev",

    "adjusted_probability":
        "bet_adjusted_model_prob",

    "adjusted_ev":
        "bet_uncertainty_adjusted_ev",
}


def utc_id():
    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%SZ"
    )


def load_yaml(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = (
            yaml.safe_load(f)
            or {}
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"Invalid YAML root: {path}"
        )

    return data


def write_yaml(
    data,
    path,
):
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
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def nba_margin_rule(cfg):
    try:
        return (
            cfg[
                "leagues"
            ][
                "nba"
            ][
                "bias"
            ][
                "margin"
            ]
        )

    except KeyError as exc:
        raise ValueError(
            "model_config.yaml missing "
            "leagues.nba.bias.margin"
        ) from exc


def build_test_config(
    source_cfg,
    window,
    output_path,
):

    cfg = load_yaml(
        source_cfg
    )

    rule = nba_margin_rule(
        cfg
    )

    method = str(
        rule.get(
            "method",
            "",
        )
    ).strip().lower()

    if method != "rolling":
        raise ValueError(
            "NBA margin bias is not "
            f"rolling; found {method!r}"
        )

    rule[
        "window_games"
    ] = int(
        window
    )

    write_yaml(
        cfg,
        output_path,
    )

    return cfg


def run_backtest(
    config_path,
    run_name,
):

    run_dir = (
        RUNS_DIR
        / run_name
    )

    if run_dir.exists():
        raise FileExistsError(
            run_dir
        )

    command = [
        sys.executable,
        str(BACKTEST),
        "--model-config",
        str(config_path),
        "--run-name",
        run_name,
    ]

    print()
    print(
        "=" * 100
    )

    print(
        f"RUNNING {run_name}"
    )

    print(
        "=" * 100
    )

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Backtest failed: "
            f"{run_name}"
        )

    if not run_dir.exists():
        raise RuntimeError(
            "Backtest completed but "
            f"run snapshot is missing: "
            f"{run_dir}"
        )

    return run_dir


def load_nba_spread(
    run_dir,
):

    path = (
        run_dir
        / "graded/nba/"
        "2025_NBA_graded.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:
        raise RuntimeError(
            f"Empty graded file: {path}"
        )

    required = [
        "market_type",
        "bet_result",
        "profit_unit",
        "game_id",
        "bet_side",
        *METRICS.values(),
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{missing}"
        )

    df = df[
        df[
            "market_type"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "spread"
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No NBA spread rows in {path}"
        )

    return df


def ranking_test(
    df,
    column,
):

    work = df[
        df[
            "bet_result"
        ].isin(
            [
                "Win",
                "Loss",
            ]
        )
    ].copy()

    work[column] = (
        pd.to_numeric(
            work[column],
            errors="coerce",
        )
    )

    work = work[
        work[column].notna()
    ].copy()

    if len(work) < 25:
        return {
            "works": "TOO_SMALL",
            "low_win_rate": np.nan,
            "high_win_rate": np.nan,
            "spearman": np.nan,
            "n": len(work),
        }

    work = (
        work.sort_values(
            column,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    work[
        "bucket"
    ] = pd.qcut(
        np.arange(
            len(work)
        ),
        q=5,
        labels=[
            1,
            2,
            3,
            4,
            5,
        ],
    )

    rates = []

    for _bucket, group in (
        work.groupby(
            "bucket",
            observed=True,
            sort=True,
        )
    ):
        wins = int(
            (
                group[
                    "bet_result"
                ]
                == "Win"
            ).sum()
        )

        losses = int(
            (
                group[
                    "bet_result"
                ]
                == "Loss"
            ).sum()
        )

        rates.append(
            wins
            / (
                wins
                + losses
            )
        )

    low = rates[0]
    high = rates[-1]

    corr = (
        pd.Series(
            rates
        )
        .corr(
            pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            ),
            method="spearman",
        )
    )

    works = (
        "YES"
        if (
            pd.notna(
                corr
            )
            and corr > 0
            and high > low
        )
        else "NO"
    )

    return {
        "works": works,
        "low_win_rate": low,
        "high_win_rate": high,
        "spearman": corr,
        "n": len(work),
    }


def performance_summary(
    df,
    label,
):

    wins = int(
        (
            df[
                "bet_result"
            ]
            == "Win"
        ).sum()
    )

    losses = int(
        (
            df[
                "bet_result"
            ]
            == "Loss"
        ).sum()
    )

    pushes = int(
        (
            df[
                "bet_result"
            ]
            == "Push"
        ).sum()
    )

    unknown = int(
        (
            df[
                "bet_result"
            ]
            == "Unknown"
        ).sum()
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

    profit = (
        pd.to_numeric(
            df[
                "profit_unit"
            ],
            errors="coerce",
        )
        .sum(
            skipna=True
        )
    )

    win_rate = (
        wins
        / decisions
        if decisions
        else np.nan
    )

    roi = (
        profit
        / graded_stakes
        if graded_stakes
        else np.nan
    )

    result = {
        "version": label,
        "bets": len(df),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "unknown": unknown,
        "win_rate": win_rate,
        "profit_units": profit,
        "roi": roi,
    }

    ranking_rows = []

    for (
        metric_name,
        column,
    ) in METRICS.items():

        test = ranking_test(
            df,
            column,
        )

        ranking_rows.append(
            {
                "version":
                    label,

                "metric":
                    metric_name,

                **test,
            }
        )

        result[
            f"{metric_name}_works"
        ] = test[
            "works"
        ]

    return (
        result,
        ranking_rows,
    )


def selection_keys(df):

    return set(
        zip(
            df[
                "game_id"
            ].astype(str),

            df[
                "bet_side"
            ]
            .astype(str)
            .str.lower(),
        )
    )


def main():

    if not PRODUCTION_MODEL_CONFIG.exists():
        raise FileNotFoundError(
            PRODUCTION_MODEL_CONFIG
        )

    current_cfg = load_yaml(
        PRODUCTION_MODEL_CONFIG
    )

    current_rule = (
        nba_margin_rule(
            current_cfg
        )
    )

    current_window = int(
        current_rule[
            "window_games"
        ]
    )

    if current_window != 100:
        raise RuntimeError(
            "Expected current NBA "
            "margin window to be 100, "
            f"but found {current_window}. "
            "No test was run."
        )

    stamp = utc_id()

    config_100 = (
        CONFIG_DIR
        / (
            "model_config_"
            f"nba_margin_100_{stamp}.yaml"
        )
    )

    config_150 = (
        CONFIG_DIR
        / (
            "model_config_"
            f"nba_margin_150_{stamp}.yaml"
        )
    )

    build_test_config(
        PRODUCTION_MODEL_CONFIG,
        100,
        config_100,
    )

    build_test_config(
        PRODUCTION_MODEL_CONFIG,
        150,
        config_150,
    )

    run_100_name = (
        "nba_margin_100_e2e_"
        + stamp
    )

    run_150_name = (
        "nba_margin_150_e2e_"
        + stamp
    )

    # Run exact current production setup first.
    run_100 = run_backtest(
        config_100,
        run_100_name,
    )

    # Run identical pipeline with ONLY
    # NBA margin rolling window changed
    # from 100 to 150.
    run_150 = run_backtest(
        config_150,
        run_150_name,
    )

    nba_100 = load_nba_spread(
        run_100
    )

    nba_150 = load_nba_spread(
        run_150
    )

    (
        summary_100,
        rankings_100,
    ) = performance_summary(
        nba_100,
        "WINDOW_100",
    )

    (
        summary_150,
        rankings_150,
    ) = performance_summary(
        nba_150,
        "WINDOW_150",
    )

    summary = pd.DataFrame(
        [
            summary_100,
            summary_150,
        ]
    )

    rankings = pd.DataFrame(
        rankings_100
        + rankings_150
    )

    keys_100 = selection_keys(
        nba_100
    )

    keys_150 = selection_keys(
        nba_150
    )

    overlap = pd.DataFrame(
        [
            {
                "window_100_bets":
                    len(
                        keys_100
                    ),

                "window_150_bets":
                    len(
                        keys_150
                    ),

                "shared":
                    len(
                        keys_100
                        & keys_150
                    ),

                "only_window_100":
                    len(
                        keys_100
                        - keys_150
                    ),

                "only_window_150":
                    len(
                        keys_150
                        - keys_100
                    ),
            }
        ]
    )

    output_dir = (
        OUTPUT_ROOT
        / stamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "nba_spread_100_vs_150_summary.csv"
    )

    rankings_path = (
        output_dir
        / "nba_spread_100_vs_150_rankings.csv"
    )

    overlap_path = (
        output_dir
        / "nba_spread_100_vs_150_selection_overlap.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    rankings.to_csv(
        rankings_path,
        index=False,
    )

    overlap.to_csv(
        overlap_path,
        index=False,
    )

    display_cols = [
        "version",
        "bets",
        "wins",
        "losses",
        "pushes",
        "win_rate",
        "profit_units",
        "roi",
        "model_probability_works",
        "raw_ev_works",
        "adjusted_probability_works",
        "adjusted_ev_works",
    ]

    print()
    print(
        "=" * 130
    )

    print(
        "NBA SPREAD — END-TO-END "
        "100 VS 150 MARGIN BIAS WINDOW"
    )

    print(
        "=" * 130
    )

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "DETAILED RANKING"
    )

    print(
        rankings[
            [
                "version",
                "metric",
                "n",
                "low_win_rate",
                "high_win_rate",
                "spearman",
                "works",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "SELECTION CHANGES"
    )

    print(
        overlap.to_string(
            index=False
        )
    )

    print()
    print(
        f"100 RUN: {run_100}"
    )

    print(
        f"150 RUN: {run_150}"
    )

    print()
    print(
        f"WROTE: {summary_path}"
    )

    print(
        f"WROTE: {rankings_path}"
    )

    print(
        f"WROTE: {overlap_path}"
    )

    print()
    print(
        "PRODUCTION model_config.yaml "
        "WAS NOT MODIFIED."
    )


if __name__ == "__main__":
    main()
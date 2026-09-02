#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


INPUT = Path(
    "docs/win/basketball/backtest/"
    "probability_ev_trace/probability_ev_trace_rows.csv"
)

OUTPUT_DIR = Path(
    "docs/win/basketball/backtest/"
    "nba_spread_threshold_test"
)

# None = absolutely no raw-EV eligibility filter.
THRESHOLDS = [
    None,
    -0.100,
    -0.075,
    -0.050,
    -0.025,
    -0.010,
    0.000,
    0.005,
    0.010,
    0.015,
]

METRICS = [
    "model_probability",
    "raw_ev",
    "adjusted_probability",
    "adjusted_ev",
]


def metric_test(df, metric):
    work = df[
        df["bet_result"].isin(["W", "L"])
        & df[metric].notna()
    ].copy()

    if len(work) < 25:
        return "TOO_SMALL", np.nan, np.nan, np.nan

    work = work.sort_values(
        metric,
        kind="stable",
    ).reset_index(drop=True)

    work["bucket"] = pd.qcut(
        np.arange(len(work)),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    bucket_rows = []

    for bucket, bdf in work.groupby(
        "bucket",
        observed=True,
        sort=True,
    ):
        wins = int(
            (bdf["bet_result"] == "W").sum()
        )

        losses = int(
            (bdf["bet_result"] == "L").sum()
        )

        bucket_rows.append(
            {
                "bucket": int(bucket),
                "win_rate": wins / (wins + losses),
            }
        )

    buckets = pd.DataFrame(bucket_rows)

    low = float(
        buckets.iloc[0]["win_rate"]
    )

    high = float(
        buckets.iloc[-1]["win_rate"]
    )

    corr = (
        buckets[
            ["bucket", "win_rate"]
        ]
        .corr(method="spearman")
        .iloc[0, 1]
    )

    result = (
        "YES"
        if (
            pd.notna(corr)
            and corr > 0
            and high > low
        )
        else "NO"
    )

    return result, low, high, corr


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df = df[
        (df["league"] == "NBA")
        & (df["market_type"] == "spread")
    ].copy()

    for col in METRICS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Rebuild sportsbook decimal odds from:
    # EV = probability * decimal_odds - 1
    df["decimal_odds"] = np.where(
        df["model_probability"] > 0,
        (1.0 + df["raw_ev"])
        / df["model_probability"],
        np.nan,
    )

    results = []

    for threshold in THRESHOLDS:

        if threshold is None:
            test = df.copy()
            threshold_label = "NO_FILTER"
        else:
            test = df[
                df["raw_ev"] >= threshold
            ].copy()

            threshold_label = f"{threshold:.3f}"

        decisions = test[
            test["bet_result"].isin(["W", "L"])
        ]

        wins = int(
            (decisions["bet_result"] == "W").sum()
        )

        losses = int(
            (decisions["bet_result"] == "L").sum()
        )

        pushes = int(
            (test["bet_result"] == "P").sum()
        )

        win_rate = (
            wins / (wins + losses)
            if wins + losses
            else np.nan
        )

        test["profit_unit"] = np.where(
            test["bet_result"] == "W",
            test["decimal_odds"] - 1.0,
            np.where(
                test["bet_result"] == "L",
                -1.0,
                0.0,
            ),
        )

        profit = test[
            "profit_unit"
        ].sum()

        roi = (
            profit / len(test)
            if len(test)
            else np.nan
        )

        row = {
            "threshold":
                threshold_label,

            "bets":
                len(test),

            "wins":
                wins,

            "losses":
                losses,

            "pushes":
                pushes,

            "win_rate":
                win_rate,

            "profit_units":
                profit,

            "roi":
                roi,
        }

        for metric in METRICS:

            result, low, high, corr = (
                metric_test(
                    test,
                    metric,
                )
            )

            prefix = {
                "model_probability":
                    "prob",

                "raw_ev":
                    "raw_ev",

                "adjusted_probability":
                    "adj_prob",

                "adjusted_ev":
                    "adj_ev",
            }[metric]

            row[
                f"{prefix}_works"
            ] = result

            row[
                f"{prefix}_low_win_rate"
            ] = low

            row[
                f"{prefix}_high_win_rate"
            ] = high

            row[
                f"{prefix}_rank_corr"
            ] = corr

        results.append(row)

    out = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "nba_spread_negative_ev_thresholds.csv"
    )

    out.to_csv(
        output_path,
        index=False,
    )

    display = out[
        [
            "threshold",
            "bets",
            "win_rate",
            "profit_units",
            "roi",
            "prob_works",
            "raw_ev_works",
            "adj_prob_works",
            "adj_ev_works",
        ]
    ]

    print()
    print("=" * 110)
    print("NBA SPREAD — NEGATIVE RAW-EV THRESHOLD TEST")
    print("=" * 110)

    print(
        display.to_string(
            index=False,
        )
    )

    print()
    print(f"WROTE: {output_path}")


if __name__ == "__main__":
    main()
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
    "nba_spread_model_side_test"
)

THRESHOLDS = [
    None,
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
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

    rates = []

    for _, bucket in work.groupby(
        "bucket",
        observed=True,
        sort=True,
    ):
        wins = (
            bucket["bet_result"] == "W"
        ).sum()

        losses = (
            bucket["bet_result"] == "L"
        ).sum()

        rates.append(
            wins / (wins + losses)
        )

    low = rates[0]
    high = rates[-1]

    corr = pd.Series(rates).corr(
        pd.Series([1, 2, 3, 4, 5]),
        method="spearman",
    )

    works = (
        "YES"
        if (
            pd.notna(corr)
            and corr > 0
            and high > low
        )
        else "NO"
    )

    return works, low, high, corr


def evaluate(df, mode, threshold):
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

    profit = test["profit_unit"].sum()

    roi = (
        profit / len(test)
        if len(test)
        else np.nan
    )

    result = {
        "mode": mode,
        "threshold": threshold_label,
        "bets": len(test),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "profit_units": profit,
        "roi": roi,
    }

    for metric in METRICS:
        works, low, high, corr = metric_test(
            test,
            metric,
        )

        prefix = {
            "model_probability": "prob",
            "raw_ev": "raw_ev",
            "adjusted_probability": "adj_prob",
            "adjusted_ev": "adj_ev",
        }[metric]

        result[f"{prefix}_works"] = works
        result[f"{prefix}_low"] = low
        result[f"{prefix}_high"] = high
        result[f"{prefix}_corr"] = corr

    return result


def main():
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

    # Recover sportsbook decimal odds.
    df["decimal_odds"] = np.where(
        df["model_probability"] > 0,
        (1.0 + df["raw_ev"])
        / df["model_probability"],
        np.nan,
    )

    # -------------------------------------------------
    # MODE 1:
    # Current-style candidate pool: either side allowed.
    # -------------------------------------------------

    any_side = df.copy()

    # -------------------------------------------------
    # MODE 2:
    # Keep exactly ONE side per game:
    # whichever side has the higher model probability.
    # -------------------------------------------------

    model_side = (
        df.sort_values(
            [
                "game_id",
                "model_probability",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=["game_id"],
            keep="first",
        )
        .copy()
    )

    print(
        f"NBA spread games: "
        f"{model_side['game_id'].nunique()}"
    )

    print(
        f"All candidate sides: "
        f"{len(any_side)}"
    )

    print(
        f"Model-side-only candidates: "
        f"{len(model_side)}"
    )

    rows = []

    for threshold in THRESHOLDS:
        rows.append(
            evaluate(
                any_side,
                "ANY_SIDE",
                threshold,
            )
        )

        rows.append(
            evaluate(
                model_side,
                "MODEL_SIDE_ONLY",
                threshold,
            )
        )

    out = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "nba_spread_model_side_comparison.csv"
    )

    out.to_csv(
        output_path,
        index=False,
    )

    display = out[
        [
            "mode",
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
    print("=" * 125)
    print(
        "NBA SPREAD — ANY SIDE VS MODEL'S MORE-LIKELY SIDE"
    )
    print("=" * 125)

    print(
        display.to_string(
            index=False,
        )
    )

    print()
    print(
        f"WROTE: {output_path}"
    )


if __name__ == "__main__":
    main()
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
    "nba_spread_point_difference_test"
)


def summarize(df, label):
    decisions = df[
        df["bet_result"].isin(["W", "L"])
    ]

    wins = int(
        (decisions["bet_result"] == "W").sum()
    )
    losses = int(
        (decisions["bet_result"] == "L").sum()
    )
    pushes = int(
        (df["bet_result"] == "P").sum()
    )

    win_rate = (
        wins / (wins + losses)
        if wins + losses
        else np.nan
    )

    profit = df["profit_unit"].sum()

    roi = (
        profit / len(df)
        if len(df)
        else np.nan
    )

    return {
        "point_difference": label,
        "bets": len(df),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "profit_units": profit,
        "roi": roi,
        "avg_model_probability":
            df["model_probability"].mean(),
        "avg_raw_ev":
            df["raw_ev"].mean(),
    }


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

    for col in [
        "model_probability",
        "raw_ev",
        "signal_points",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Keep exactly one side per game:
    # whichever side the model considers more likely.
    df = (
        df.sort_values(
            [
                "game_id",
                "model_probability",
            ],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates(
            subset=["game_id"],
            keep="first",
        )
        .copy()
    )

    # Recover sportsbook decimal odds:
    # raw_ev = probability * decimal_odds - 1
    df["decimal_odds"] = np.where(
        df["model_probability"] > 0,
        (1.0 + df["raw_ev"])
        / df["model_probability"],
        np.nan,
    )

    df["profit_unit"] = np.where(
        df["bet_result"] == "W",
        df["decimal_odds"] - 1.0,
        np.where(
            df["bet_result"] == "L",
            -1.0,
            0.0,
        ),
    )

    df = df[
        df["signal_points"].notna()
    ].copy()

    # -----------------------------------------
    # Exact point-difference ranges
    # -----------------------------------------

    ranges = [
        (0.0, 1.0, "0 to <1"),
        (1.0, 2.0, "1 to <2"),
        (2.0, 3.0, "2 to <3"),
        (3.0, 4.0, "3 to <4"),
        (4.0, 5.0, "4 to <5"),
        (5.0, 6.0, "5 to <6"),
        (6.0, np.inf, "6+"),
    ]

    range_rows = []

    for low, high, label in ranges:

        test = df[
            (df["signal_points"] >= low)
            & (df["signal_points"] < high)
        ].copy()

        range_rows.append(
            summarize(
                test,
                label,
            )
        )

    range_df = pd.DataFrame(
        range_rows
    )

    # -----------------------------------------
    # Cumulative minimum disagreement
    # -----------------------------------------

    thresholds = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
    ]

    cumulative_rows = []

    for threshold in thresholds:

        test = df[
            df["signal_points"]
            >= threshold
        ].copy()

        cumulative_rows.append(
            summarize(
                test,
                f">= {threshold:g}",
            )
        )

    cumulative_df = pd.DataFrame(
        cumulative_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    range_path = (
        OUTPUT_DIR
        / "nba_spread_point_ranges.csv"
    )

    cumulative_path = (
        OUTPUT_DIR
        / "nba_spread_point_thresholds.csv"
    )

    range_df.to_csv(
        range_path,
        index=False,
    )

    cumulative_df.to_csv(
        cumulative_path,
        index=False,
    )

    print()
    print("=" * 85)
    print("NBA SPREAD — MODEL VS SPORTSBOOK POINT DIFFERENCE")
    print("=" * 85)

    print()
    print("EXACT RANGES")
    print(
        range_df[
            [
                "point_difference",
                "bets",
                "win_rate",
                "profit_units",
                "roi",
            ]
        ].to_string(index=False)
    )

    print()
    print("CUMULATIVE MINIMUM")
    print(
        cumulative_df[
            [
                "point_difference",
                "bets",
                "win_rate",
                "profit_units",
                "roi",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"WROTE: {range_path}")
    print(f"WROTE: {cumulative_path}")


if __name__ == "__main__":
    main()
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
    "nba_spread_1_to_4_chronological"
)


def summarize(df, period):
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
        "period": period,
        "first_date": (
            df["game_date"].min().date()
            if len(df)
            else None
        ),
        "last_date": (
            df["game_date"].max().date()
            if len(df)
            else None
        ),
        "bets": len(df),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "profit_units": profit,
        "roi": roi,
        "avg_signal_points":
            df["signal_points"].mean(),
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

    df["game_date"] = pd.to_datetime(
        df["game_date"]
        .astype(str)
        .str.replace("_", "-", regex=False),
        errors="coerce",
    )

    df = df[
        df["game_date"].notna()
    ].copy()

    # Keep exactly the model's more-likely side.
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

    # Exact candidate rule being tested:
    # model-favored side AND disagreement >=1 and <4 points.
    df = df[
        (df["signal_points"] >= 1.0)
        & (df["signal_points"] < 4.0)
    ].copy()

    # Recover decimal odds from:
    # EV = probability * decimal_odds - 1
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

    df = df.sort_values(
        ["game_date", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    unique_dates = sorted(
        df["game_date"].dropna().unique()
    )

    if len(unique_dates) < 10:
        raise RuntimeError(
            "Not enough unique dates for chronological test"
        )

    # Chronological 70/30 split by GAME DATE.
    split_70 = int(
        len(unique_dates) * 0.70
    )

    early_dates = set(
        unique_dates[:split_70]
    )

    late_dates = set(
        unique_dates[split_70:]
    )

    early_70 = df[
        df["game_date"].isin(
            early_dates
        )
    ].copy()

    late_30 = df[
        df["game_date"].isin(
            late_dates
        )
    ].copy()

    # Also divide the season into three chronological
    # date blocks to check consistency.
    date_groups = np.array_split(
        np.array(unique_dates),
        3,
    )

    rows = [
        summarize(
            df,
            "FULL_SEASON",
        ),
        summarize(
            early_70,
            "EARLY_70_PERCENT_DATES",
        ),
        summarize(
            late_30,
            "LATE_30_PERCENT_DATES",
        ),
    ]

    for i, dates in enumerate(
        date_groups,
        start=1,
    ):
        part = df[
            df["game_date"].isin(
                set(dates)
            )
        ].copy()

        rows.append(
            summarize(
                part,
                f"CHRONOLOGICAL_THIRD_{i}",
            )
        )

    out = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "nba_spread_1_to_4_chronological.csv"
    )

    out.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 105)
    print(
        "NBA SPREAD — 1 TO <4 POINT BAND — "
        "CHRONOLOGICAL TEST"
    )
    print("=" * 105)

    print(
        out[
            [
                "period",
                "first_date",
                "last_date",
                "bets",
                "win_rate",
                "profit_units",
                "roi",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"WROTE: {output_path}")


if __name__ == "__main__":
    main()
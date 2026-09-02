#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd


BASE = Path("docs/win/basketball")

COMBINED_PATH = (
    BASE
    / "00_intake/final_combined_files/combined/2025_NBA.csv"
)

TRACE_PATH = (
    BASE
    / "backtest/probability_ev_trace/probability_ev_trace_rows.csv"
)

OUTPUT_DIR = (
    BASE
    / "backtest/nba_spread_chronological_accuracy"
)


PERIODS = [
    (
        "THIRD_1",
        "2025-11-03",
        "2026-01-08",
    ),
    (
        "THIRD_2",
        "2026-01-09",
        "2026-03-18",
    ),
    (
        "THIRD_3",
        "2026-03-19",
        "2026-06-08",
    ),
]


def fv(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def canonical_id(value):
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        x = float(text)

        if x.is_integer():
            return str(int(x))

    except Exception:
        pass

    return text


def load_combined():
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(COMBINED_PATH)

    df = pd.read_csv(
        COMBINED_PATH,
        low_memory=False,
    )

    required = [
        "game_id",
        "game_date",
        "home_spread",
        "home_projected_points",
        "away_projected_points",
        "home_score",
        "away_score",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{COMBINED_PATH} missing columns: {missing}"
        )

    df["game_id"] = df[
        "game_id"
    ].map(canonical_id)

    df["game_date"] = pd.to_datetime(
        df["game_date"]
        .astype(str)
        .str.replace(
            "_",
            "-",
            regex=False,
        ),
        errors="coerce",
    )

    numeric_cols = [
        "home_spread",
        "home_projected_points",
        "away_projected_points",
        "home_score",
        "away_score",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["predicted_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    df["actual_margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    # Same sign used by calculate_rolling_bias.py:
    #
    # positive = model projected home margin too high
    # negative = model projected home margin too low
    df["projected_minus_actual"] = (
        df["predicted_margin"]
        - df["actual_margin"]
    )

    df["absolute_margin_error"] = (
        df["projected_minus_actual"].abs()
    )

    # Positive = model favors HOME against sportsbook.
    # Negative = model favors AWAY.
    df["signed_model_vs_book_points"] = (
        df["predicted_margin"]
        + df["home_spread"]
    )

    df["signal_points"] = (
        df["signed_model_vs_book_points"].abs()
    )

    return df


def load_trace():
    if not TRACE_PATH.exists():
        raise FileNotFoundError(TRACE_PATH)

    df = pd.read_csv(
        TRACE_PATH,
        low_memory=False,
    )

    df = df[
        (df["league"] == "NBA")
        & (df["market_type"] == "spread")
    ].copy()

    required = [
        "game_id",
        "bet_side",
        "model_probability",
        "raw_ev",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{TRACE_PATH} missing columns: {missing}"
        )

    df["game_id"] = df[
        "game_id"
    ].map(canonical_id)

    for col in [
        "model_probability",
        "raw_ev",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Keep exactly one side per game:
    # the side with the higher model probability.
    df = (
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

    df = df[
        [
            "game_id",
            "bet_side",
            "model_probability",
            "raw_ev",
        ]
    ]

    return df


def grade_ats(row):
    actual_with_spread = (
        row["actual_margin"]
        + row["home_spread"]
    )

    if abs(actual_with_spread) < 1e-12:
        return "P"

    home_cover = (
        actual_with_spread > 0
    )

    side = str(
        row["bet_side"]
    ).lower()

    if side == "home":
        return (
            "W"
            if home_cover
            else "L"
        )

    if side == "away":
        return (
            "L"
            if home_cover
            else "W"
        )

    return ""


def add_profit(df):
    out = df.copy()

    out["decimal_odds"] = np.where(
        out["model_probability"] > 0,
        (
            1.0
            + out["raw_ev"]
        )
        / out["model_probability"],
        np.nan,
    )

    out["profit_unit"] = np.where(
        out["ats_result"] == "W",
        out["decimal_odds"] - 1.0,
        np.where(
            out["ats_result"] == "L",
            -1.0,
            0.0,
        ),
    )

    return out


def summarize(
    df,
    period,
    sample,
):
    decisions = df[
        df["ats_result"].isin(
            ["W", "L"]
        )
    ]

    wins = int(
        (
            decisions["ats_result"]
            == "W"
        ).sum()
    )

    losses = int(
        (
            decisions["ats_result"]
            == "L"
        ).sum()
    )

    pushes = int(
        (
            df["ats_result"]
            == "P"
        ).sum()
    )

    ats_win_rate = (
        wins / (wins + losses)
        if wins + losses
        else np.nan
    )

    profit = (
        df["profit_unit"].sum()
        if "profit_unit" in df.columns
        else np.nan
    )

    roi = (
        profit / len(df)
        if len(df)
        and pd.notna(profit)
        else np.nan
    )

    return {
        "period":
            period,

        "sample":
            sample,

        "games":
            len(df),

        "margin_mae":
            df[
                "absolute_margin_error"
            ].mean(),

        # projected minus actual
        "signed_margin_bias":
            df[
                "projected_minus_actual"
            ].mean(),

        "avg_abs_model_vs_book_points":
            df[
                "signal_points"
            ].mean(),

        "median_abs_model_vs_book_points":
            df[
                "signal_points"
            ].median(),

        "avg_model_probability":
            df[
                "model_probability"
            ].mean(),

        "ats_wins":
            wins,

        "ats_losses":
            losses,

        "ats_pushes":
            pushes,

        "ats_win_rate":
            ats_win_rate,

        "profit_units":
            profit,

        "roi":
            roi,
    }


def main():
    combined = load_combined()
    trace = load_trace()

    df = combined.merge(
        trace,
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    df = df[
        df["game_date"].notna()
        & df["predicted_margin"].notna()
        & df["actual_margin"].notna()
        & df["home_spread"].notna()
    ].copy()

    df["ats_result"] = df.apply(
        grade_ats,
        axis=1,
    )

    df = add_profit(df)

    rows = []

    for (
        period,
        start,
        end,
    ) in PERIODS:

        start_date = pd.Timestamp(
            start
        )

        end_date = pd.Timestamp(
            end
        )

        period_df = df[
            (
                df["game_date"]
                >= start_date
            )
            & (
                df["game_date"]
                <= end_date
            )
        ].copy()

        rows.append(
            summarize(
                period_df,
                period,
                "ALL_MODEL_FAVORED_SIDES",
            )
        )

        band_df = period_df[
            (
                period_df[
                    "signal_points"
                ]
                >= 1.0
            )
            & (
                period_df[
                    "signal_points"
                ]
                < 4.0
            )
        ].copy()

        rows.append(
            summarize(
                band_df,
                period,
                "SIGNAL_1_TO_LT4",
            )
        )

    out = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "nba_spread_chronological_accuracy.csv"
    )

    out.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 135)
    print(
        "NBA SPREAD — CHRONOLOGICAL "
        "PREDICTION ACCURACY DIAGNOSTIC"
    )
    print("=" * 135)

    display = out[
        [
            "period",
            "sample",
            "games",
            "margin_mae",
            "signed_margin_bias",
            "avg_abs_model_vs_book_points",
            "ats_win_rate",
            "profit_units",
            "roi",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "signed_margin_bias = "
        "projected margin minus actual margin"
    )

    print()
    print(
        f"WROTE: {output_path}"
    )


if __name__ == "__main__":
    main()
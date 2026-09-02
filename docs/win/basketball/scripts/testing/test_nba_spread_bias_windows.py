#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path("docs/win/basketball")

STAGE3 = (
    BASE
    / "backtest/working/nba/spread/2025_NBA_spread.csv"
)

OUTCOMES = (
    BASE
    / "00_intake/final_combined_files/combined/2025_NBA.csv"
)

OUTPUT = (
    BASE
    / "backtest/nba_spread_bias_window_test/"
    "nba_spread_bias_windows.csv"
)

WINDOWS = [25, 50, 75, 100, 150]

PERIODS = [
    ("FULL_SEASON", "2025-11-03", "2026-06-08"),
    ("THIRD_1", "2025-11-03", "2026-01-08"),
    ("THIRD_2", "2026-01-09", "2026-03-18"),
    ("THIRD_3", "2026-03-19", "2026-06-08"),
]


def canonical_id(value):
    text = str(value).strip()

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass

    return text


def grade(df, margin_col):

    out = df.copy()

    out["margin_error"] = (
        out[margin_col]
        - out["actual_margin"]
    )

    out["abs_margin_error"] = (
        out["margin_error"].abs()
    )

    out["model_vs_book"] = (
        out[margin_col]
        + out["home_spread"]
    )

    out["signal_points"] = (
        out["model_vs_book"].abs()
    )

    out["model_side"] = np.where(
        out["model_vs_book"] >= 0,
        "home",
        "away",
    )

    spread_result = (
        out["actual_margin"]
        + out["home_spread"]
    )

    out["ats_result"] = np.where(
        spread_result == 0,
        "P",
        np.where(
            (
                (out["model_side"] == "home")
                & (spread_result > 0)
            )
            |
            (
                (out["model_side"] == "away")
                & (spread_result < 0)
            ),
            "W",
            "L",
        ),
    )

    return out


def summarize(
    df,
    period,
    version,
    margin_col,
    bias_col=None,
):

    work = grade(
        df,
        margin_col,
    )

    decisions = work[
        work["ats_result"].isin(
            ["W", "L"]
        )
    ]

    wins = int(
        (decisions["ats_result"] == "W").sum()
    )

    losses = int(
        (decisions["ats_result"] == "L").sum()
    )

    pushes = int(
        (work["ats_result"] == "P").sum()
    )

    return {
        "period": period,
        "version": version,
        "games": len(work),

        "avg_applied_bias": (
            work[bias_col].mean()
            if bias_col
            else 0.0
        ),

        "margin_mae":
            work["abs_margin_error"].mean(),

        "signed_margin_bias":
            work["margin_error"].mean(),

        "avg_model_vs_book_points":
            work["signal_points"].mean(),

        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": pushes,

        "ats_win_rate": (
            wins / (wins + losses)
            if wins + losses
            else np.nan
        ),
    }


def main():

    if not STAGE3.exists():
        raise FileNotFoundError(STAGE3)

    if not OUTCOMES.exists():
        raise FileNotFoundError(OUTCOMES)

    stage3 = pd.read_csv(
        STAGE3,
        low_memory=False,
    )

    outcomes = pd.read_csv(
        OUTCOMES,
        low_memory=False,
    )

    required_stage3 = [
        "game_id",
        "game_date",
        "home_projected_points",
        "away_projected_points",
        "home_spread",
        "margin_bias",
    ]

    missing = [
        col
        for col in required_stage3
        if col not in stage3.columns
    ]

    if missing:
        raise ValueError(
            f"Stage-3 missing columns: {missing}"
        )

    required_outcomes = [
        "game_id",
        "home_score",
        "away_score",
    ]

    missing = [
        col
        for col in required_outcomes
        if col not in outcomes.columns
    ]

    if missing:
        raise ValueError(
            f"Outcome file missing columns: {missing}"
        )

    stage3["game_id"] = (
        stage3["game_id"]
        .map(canonical_id)
    )

    outcomes["game_id"] = (
        outcomes["game_id"]
        .map(canonical_id)
    )

    outcomes = (
        outcomes[
            [
                "game_id",
                "home_score",
                "away_score",
            ]
        ]
        .drop_duplicates("game_id")
    )

    df = stage3.merge(
        outcomes,
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    numeric = [
        "home_projected_points",
        "away_projected_points",
        "home_spread",
        "margin_bias",
        "home_score",
        "away_score",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

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

    df = df.dropna(
        subset=[
            "game_date",
            "home_projected_points",
            "away_projected_points",
            "home_spread",
            "margin_bias",
            "home_score",
            "away_score",
        ]
    ).copy()

    # Exact Stage-3 margin after the historical
    # rolling bias was applied.
    df["stage3_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    # Production does:
    #
    # adjusted_margin = raw_margin - margin_bias
    #
    # therefore reconstruct raw:
    df["raw_margin"] = (
        df["stage3_margin"]
        + df["margin_bias"]
    )

    df["actual_margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    df["raw_error"] = (
        df["raw_margin"]
        - df["actual_margin"]
    )

    # Stable chronological ordering.
    sort_cols = ["game_date"]

    if "game_time" in df.columns:
        sort_cols.append(
            "game_time"
        )

    sort_cols.append(
        "game_id"
    )

    df = df.sort_values(
        sort_cols,
        kind="stable",
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Rebuild alternative rolling biases.
    #
    # IMPORTANT:
    # Every game on a date uses ONLY games from
    # earlier dates. No same-day final score can leak
    # into that day's prediction.
    # -------------------------------------------------

    for window in WINDOWS:
        df[f"bias_{window}"] = np.nan

    history = []

    for game_date in sorted(
        df["game_date"].unique()
    ):

        day_mask = (
            df["game_date"]
            == game_date
        )

        day_indexes = df.index[
            day_mask
        ].tolist()

        for window in WINDOWS:

            if len(history) >= window:

                value = float(
                    np.mean(
                        history[-window:]
                    )
                )

                df.loc[
                    day_indexes,
                    f"bias_{window}",
                ] = value

        # Only AFTER all biases for this date
        # have been calculated do today's
        # completed-game errors enter history.
        day_errors = (
            df.loc[
                day_indexes,
                "raw_error",
            ]
            .dropna()
            .tolist()
        )

        history.extend(
            day_errors
        )

    for window in WINDOWS:

        df[
            f"margin_{window}"
        ] = (
            df["raw_margin"]
            - df[f"bias_{window}"]
        )

    # -------------------------------------------------
    # Fair comparison:
    # use only games where all five windows had enough
    # prior games. Therefore every version is evaluated
    # on exactly the same games.
    # -------------------------------------------------

    common = df.dropna(
        subset=[
            f"bias_{window}"
            for window in WINDOWS
        ]
    ).copy()

    if common.empty:
        raise RuntimeError(
            "No games have enough history "
            "for all rolling windows."
        )

    # -------------------------------------------------
    # Verify our reconstructed 100-game rolling bias
    # against the exact Stage-3 recorded bias.
    # -------------------------------------------------

    common[
        "bias_100_difference"
    ] = (
        common["bias_100"]
        - common["margin_bias"]
    )

    mean_abs_diff = (
        common[
            "bias_100_difference"
        ]
        .abs()
        .mean()
    )

    max_abs_diff = (
        common[
            "bias_100_difference"
        ]
        .abs()
        .max()
    )

    rows = []

    for (
        period,
        start,
        end,
    ) in PERIODS:

        part = common[
            (
                common["game_date"]
                >= pd.Timestamp(start)
            )
            &
            (
                common["game_date"]
                <= pd.Timestamp(end)
            )
        ].copy()

        # No bias at all.
        rows.append(
            summarize(
                part,
                period,
                "RAW_NO_BIAS",
                "raw_margin",
            )
        )

        # Exact margin that actually reached Stage 3.
        rows.append(
            summarize(
                part,
                period,
                "RECORDED_STAGE3_100",
                "stage3_margin",
                "margin_bias",
            )
        )

        # Replayed alternative rolling windows.
        for window in WINDOWS:

            rows.append(
                summarize(
                    part,
                    period,
                    f"ROLLING_{window}",
                    f"margin_{window}",
                    f"bias_{window}",
                )
            )

    out = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 125)
    print(
        "NBA SPREAD — ROLLING MARGIN BIAS "
        "WINDOW COMPARISON"
    )
    print("=" * 125)

    print()
    print(
        f"TOTAL STAGE3 GAMES: {len(df)}"
    )

    print(
        f"COMMON COMPARISON GAMES: {len(common)}"
    )

    print()
    print(
        "100-GAME REPLAY VS RECORDED "
        "STAGE3 BIAS"
    )

    print(
        f"mean absolute difference: "
        f"{mean_abs_diff:.6f}"
    )

    print(
        f"max absolute difference:  "
        f"{max_abs_diff:.6f}"
    )

    print()
    print(
        out[
            [
                "period",
                "version",
                "games",
                "avg_applied_bias",
                "margin_mae",
                "signed_margin_bias",
                "ats_win_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"WROTE: {OUTPUT}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path("docs/win/basketball")

COMBINED = (
    BASE
    / "00_intake/final_combined_files/combined/2025_NBA.csv"
)

STAGE3 = (
    BASE
    / "backtest/working/nba/spread/2025_NBA_spread.csv"
)

OUTPUT = (
    BASE
    / "backtest/nba_spread_bias_window_exact_test/"
    "nba_spread_bias_windows_exact.csv"
)

WINDOWS = [
    25,
    50,
    75,
    100,
    150,
]

# Exact historical fallback used for NBA internal season 2025.
LEGACY_MARGIN_BIAS = 0.4

PERIODS = [
    (
        "FULL_SEASON",
        "2025-10-21",
        "2026-06-08",
    ),
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


def canonical_id(value):
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def reconstruct_all_completed_games():

    df = pd.read_csv(
        COMBINED,
        low_memory=False,
    )

    required = [
        "game_id",
        "game_date",
        "home_projected_points",
        "away_projected_points",
        "bias_applied",
        "home_score",
        "away_score",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Combined file missing columns: {missing}"
        )

    df["game_id"] = (
        df["game_id"]
        .map(canonical_id)
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

    for col in [
        "home_projected_points",
        "away_projected_points",
        "bias_applied",
        "home_score",
        "away_score",
    ]:
        df[col] = numeric(
            df[col]
        )

    df = df.dropna(
        subset=[
            "game_id",
            "game_date",
            "home_projected_points",
            "away_projected_points",
            "bias_applied",
            "home_score",
            "away_score",
        ]
    ).copy()

    if df["game_id"].duplicated().any():
        dupes = (
            df.loc[
                df["game_id"].duplicated(
                    keep=False
                ),
                "game_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate combined game_id values: "
            f"{dupes[:10]}"
        )

    invalid_flags = df[
        ~df["bias_applied"].isin(
            [0.0, 1.0]
        )
    ]

    if len(invalid_flags):
        raise ValueError(
            "Invalid bias_applied values found"
        )

    df["stored_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    # ---------------------------------------------
    # Reverse historical stored bias to RAW margin.
    #
    # If per-game margin_bias exists, use it.
    # Otherwise NBA 2025 uses the exact legacy
    # fallback of 0.4.
    # ---------------------------------------------

    if "margin_bias" in df.columns:

        df["_stored_margin_bias"] = numeric(
            df["margin_bias"]
        )

    else:

        df["_stored_margin_bias"] = np.nan

    def reversal_bias(row):

        if row["bias_applied"] == 0:
            return 0.0

        value = row["_stored_margin_bias"]

        if pd.notna(value):
            return float(value)

        return LEGACY_MARGIN_BIAS

    df["reversal_margin_bias"] = df.apply(
        reversal_bias,
        axis=1,
    )

    df["raw_margin"] = (
        df["stored_margin"]
        + df["reversal_margin_bias"]
    )

    df["actual_margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    # Exact sign used by production:
    # projected minus actual.
    df["raw_margin_error"] = (
        df["raw_margin"]
        - df["actual_margin"]
    )

    # basketball_backtest.py sorts by:
    # game_date, game_id
    df = (
        df.sort_values(
            [
                "game_date",
                "game_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------
    # Exact point-in-time replay.
    #
    # For each game:
    # 1. calculate bias from PRIOR games
    # 2. apply bias to current prediction
    # 3. append current RAW error to history
    #
    # This intentionally mirrors basketball_backtest.py
    # row-by-row behavior.
    # ---------------------------------------------

    history = []

    for window in WINDOWS:
        df[f"bias_{window}"] = np.nan
        df[f"margin_{window}"] = np.nan

    for index, row in df.iterrows():

        raw_margin = float(
            row["raw_margin"]
        )

        for window in WINDOWS:

            if len(history) < window:
                continue

            bias = round(
                float(
                    sum(
                        history[-window:]
                    )
                    / window
                ),
                3,
            )

            df.at[
                index,
                f"bias_{window}",
            ] = bias

            df.at[
                index,
                f"margin_{window}",
            ] = (
                raw_margin
                - bias
            )

        history.append(
            float(
                row["raw_margin_error"]
            )
        )

    return df


def load_stage3():

    df = pd.read_csv(
        STAGE3,
        low_memory=False,
    )

    required = [
        "game_id",
        "game_date",
        "home_projected_points",
        "away_projected_points",
        "home_spread",
        "margin_bias",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Stage-3 file missing columns: {missing}"
        )

    df["game_id"] = (
        df["game_id"]
        .map(canonical_id)
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

    for col in [
        "home_projected_points",
        "away_projected_points",
        "home_spread",
        "margin_bias",
    ]:
        df[col] = numeric(
            df[col]
        )

    df = df.dropna(
        subset=[
            "game_id",
            "game_date",
            "home_projected_points",
            "away_projected_points",
            "home_spread",
            "margin_bias",
        ]
    ).copy()

    if df["game_id"].duplicated().any():
        raise ValueError(
            "Duplicate Stage-3 game IDs found"
        )

    df["recorded_stage3_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    return df[
        [
            "game_id",
            "game_date",
            "home_spread",
            "margin_bias",
            "recorded_stage3_margin",
        ]
    ].copy()


def grade(
    frame,
    margin_col,
):

    df = frame.copy()

    df["margin_error"] = (
        df[margin_col]
        - df["actual_margin"]
    )

    df["abs_margin_error"] = (
        df["margin_error"].abs()
    )

    df["model_vs_book"] = (
        df[margin_col]
        + df["home_spread"]
    )

    df["signal_points"] = (
        df["model_vs_book"].abs()
    )

    df["model_side"] = np.where(
        df["model_vs_book"] >= 0,
        "home",
        "away",
    )

    sportsbook_result = (
        df["actual_margin"]
        + df["home_spread"]
    )

    df["ats_result"] = np.where(
        sportsbook_result == 0,
        "P",
        np.where(
            (
                (df["model_side"] == "home")
                & (sportsbook_result > 0)
            )
            |
            (
                (df["model_side"] == "away")
                & (sportsbook_result < 0)
            ),
            "W",
            "L",
        ),
    )

    return df


def summarize(
    frame,
    period,
    version,
    margin_col,
    bias_col=None,
):

    df = grade(
        frame,
        margin_col,
    )

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

    return {
        "period": period,
        "version": version,
        "games": len(df),

        "avg_applied_bias": (
            df[bias_col].mean()
            if bias_col is not None
            else 0.0
        ),

        "margin_mae":
            df[
                "abs_margin_error"
            ].mean(),

        "signed_margin_bias":
            df[
                "margin_error"
            ].mean(),

        "avg_model_vs_book_points":
            df[
                "signal_points"
            ].mean(),

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

    if not COMBINED.exists():
        raise FileNotFoundError(
            COMBINED
        )

    if not STAGE3.exists():
        raise FileNotFoundError(
            STAGE3
        )

    completed = (
        reconstruct_all_completed_games()
    )

    stage3 = load_stage3()

    # Stage-3 sportsbook games get their rolling
    # values from ALL completed NBA games.
    joined = stage3.merge(
        completed[
            [
                "game_id",
                "raw_margin",
                "actual_margin",
                *[
                    f"bias_{window}"
                    for window in WINDOWS
                ],
                *[
                    f"margin_{window}"
                    for window in WINDOWS
                ],
            ]
        ],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    # ---------------------------------------------
    # CRITICAL VALIDATION:
    # replayed 100-game bias must reproduce the
    # bias already recorded in the Stage-3 backtest.
    # ---------------------------------------------

    replay_check = joined.dropna(
        subset=[
            "margin_bias",
            "bias_100",
        ]
    ).copy()

    replay_check[
        "bias_difference"
    ] = (
        replay_check["bias_100"]
        - replay_check["margin_bias"]
    )

    mean_abs_diff = (
        replay_check[
            "bias_difference"
        ]
        .abs()
        .mean()
    )

    max_abs_diff = (
        replay_check[
            "bias_difference"
        ]
        .abs()
        .max()
    )

    exact_001 = int(
        (
            replay_check[
                "bias_difference"
            ].abs()
            <= 0.001
        ).sum()
    )

    exact_005 = int(
        (
            replay_check[
                "bias_difference"
            ].abs()
            <= 0.005
        ).sum()
    )

    # ---------------------------------------------
    # Fair comparison:
    # every window evaluated on exactly the same
    # sportsbook games where even 150 is available.
    # ---------------------------------------------

    common = joined.dropna(
        subset=[
            *[
                f"bias_{window}"
                for window in WINDOWS
            ],
            "margin_bias",
            "recorded_stage3_margin",
            "home_spread",
            "actual_margin",
        ]
    ).copy()

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

        rows.append(
            summarize(
                part,
                period,
                "RAW_NO_BIAS",
                "raw_margin",
            )
        )

        rows.append(
            summarize(
                part,
                period,
                "RECORDED_STAGE3_100",
                "recorded_stage3_margin",
                "margin_bias",
            )
        )

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
    print("=" * 130)
    print(
        "NBA SPREAD — EXACT ALL-COMPLETED-GAME "
        "ROLLING BIAS WINDOW TEST"
    )
    print("=" * 130)

    print()
    print(
        f"ALL COMPLETED NBA GAMES: "
        f"{len(completed)}"
    )

    print(
        f"STAGE3 SPORTSBOOK GAMES: "
        f"{len(stage3)}"
    )

    print(
        f"STAGE3 GAMES MATCHED TO HISTORY: "
        f"{len(joined)}"
    )

    print(
        f"COMMON 25/50/75/100/150 SAMPLE: "
        f"{len(common)}"
    )

    print()
    print(
        "100-GAME EXACT REPLAY "
        "VS RECORDED STAGE3"
    )

    print(
        f"rows compared:             "
        f"{len(replay_check)}"
    )

    print(
        f"mean absolute difference: "
        f"{mean_abs_diff:.6f}"
    )

    print(
        f"max absolute difference:  "
        f"{max_abs_diff:.6f}"
    )

    print(
        f"within 0.001:              "
        f"{exact_001} / "
        f"{len(replay_check)}"
    )

    print(
        f"within 0.005:              "
        f"{exact_005} / "
        f"{len(replay_check)}"
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
    print(
        f"WROTE: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
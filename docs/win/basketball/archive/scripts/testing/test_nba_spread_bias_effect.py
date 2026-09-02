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
    / "backtest/nba_spread_bias_effect/"
    "nba_spread_bias_effect.csv"
)

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


def evaluate(df, period, version, margin_col):

    part = df.copy()

    part["margin_error"] = (
        part[margin_col]
        - part["actual_margin"]
    )

    part["abs_margin_error"] = (
        part["margin_error"].abs()
    )

    part["model_vs_book"] = (
        part[margin_col]
        + part["home_spread"]
    )

    part["signal_points"] = (
        part["model_vs_book"].abs()
    )

    part["model_side"] = np.where(
        part["model_vs_book"] >= 0,
        "home",
        "away",
    )

    spread_result = (
        part["actual_margin"]
        + part["home_spread"]
    )

    part["ats_result"] = np.where(
        spread_result == 0,
        "P",
        np.where(
            (
                (part["model_side"] == "home")
                & (spread_result > 0)
            )
            |
            (
                (part["model_side"] == "away")
                & (spread_result < 0)
            ),
            "W",
            "L",
        ),
    )

    decisions = part[
        part["ats_result"].isin(["W", "L"])
    ]

    wins = int(
        (decisions["ats_result"] == "W").sum()
    )

    losses = int(
        (decisions["ats_result"] == "L").sum()
    )

    pushes = int(
        (part["ats_result"] == "P").sum()
    )

    return {
        "period": period,
        "version": version,
        "games": len(part),
        "margin_mae":
            part["abs_margin_error"].mean(),
        "signed_margin_bias":
            part["margin_error"].mean(),
        "avg_model_vs_book_points":
            part["signal_points"].mean(),
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": pushes,
        "ats_win_rate":
            wins / (wins + losses)
            if wins + losses
            else np.nan,
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
        stage3["game_id"].map(canonical_id)
    )

    outcomes["game_id"] = (
        outcomes["game_id"].map(canonical_id)
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
        .str.replace("_", "-", regex=False),
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

    # Actual Stage-3 adjusted margin.
    df["adjusted_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    # clean_basketball_inputs.py applies:
    #
    # adjusted_margin = raw_margin - margin_bias
    #
    # Therefore:
    df["raw_margin"] = (
        df["adjusted_margin"]
        + df["margin_bias"]
    )

    df["actual_margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    # Check how often the rolling bias actually
    # changes which spread side the model favors.
    adjusted_signal = (
        df["adjusted_margin"]
        + df["home_spread"]
    )

    raw_signal = (
        df["raw_margin"]
        + df["home_spread"]
    )

    df["adjusted_side"] = np.where(
        adjusted_signal >= 0,
        "home",
        "away",
    )

    df["raw_side"] = np.where(
        raw_signal >= 0,
        "home",
        "away",
    )

    df["side_flipped"] = (
        df["adjusted_side"]
        != df["raw_side"]
    )

    rows = []

    for name, start, end in PERIODS:

        part = df[
            (df["game_date"] >= pd.Timestamp(start))
            & (df["game_date"] <= pd.Timestamp(end))
        ].copy()

        rows.append(
            evaluate(
                part,
                name,
                "RAW_PRE_BIAS",
                "raw_margin",
            )
        )

        rows.append(
            evaluate(
                part,
                name,
                "STAGE3_BIAS_ADJUSTED",
                "adjusted_margin",
            )
        )

    out = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "NBA SPREAD — RAW PROJECTION VS "
        "ROLLING-BIAS-ADJUSTED PROJECTION"
    )
    print("=" * 120)

    print(
        out[
            [
                "period",
                "version",
                "games",
                "margin_mae",
                "signed_margin_bias",
                "avg_model_vs_book_points",
                "ats_win_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("SIDE FLIPS CAUSED BY BIAS")
    print(
        f"{int(df['side_flipped'].sum())} / {len(df)} "
        f"({df['side_flipped'].mean():.2%})"
    )

    print()
    print(f"WROTE: {OUTPUT}")


if __name__ == "__main__":
    main()
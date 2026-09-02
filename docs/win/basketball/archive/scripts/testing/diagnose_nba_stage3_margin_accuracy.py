#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path("docs/win/basketball")

WORKING = (
    BASE
    / "backtest/working/nba/spread/2025_NBA_spread.csv"
)

OUTCOMES = (
    BASE
    / "00_intake/final_combined_files/combined/2025_NBA.csv"
)

OUTPUT = (
    BASE
    / "backtest/nba_spread_chronological_accuracy/"
    "nba_stage3_margin_accuracy.csv"
)

PERIODS = [
    ("THIRD_1", "2025-11-03", "2026-01-08"),
    ("THIRD_2", "2026-01-09", "2026-03-18"),
    ("THIRD_3", "2026-03-19", "2026-06-08"),
]


def canonical_id(x):
    s = str(x).strip()

    try:
        n = float(s)
        if n.is_integer():
            return str(int(n))
    except Exception:
        pass

    return s


def summarize(df, name):

    decisions = df[
        df["ats_result"].isin(["W", "L"])
    ]

    wins = int(
        (decisions["ats_result"] == "W").sum()
    )

    losses = int(
        (decisions["ats_result"] == "L").sum()
    )

    return {
        "period": name,
        "games": len(df),
        "margin_mae":
            df["abs_margin_error"].mean(),
        "signed_margin_bias":
            df["margin_error"].mean(),
        "avg_model_vs_book_points":
            df["signal_points"].mean(),
        "ats_win_rate":
            wins / (wins + losses)
            if wins + losses
            else np.nan,
    }


def main():

    working = pd.read_csv(
        WORKING,
        low_memory=False,
    )

    outcomes = pd.read_csv(
        OUTCOMES,
        low_memory=False,
    )

    required_working = [
        "game_id",
        "game_date",
        "home_projected_points",
        "away_projected_points",
        "home_spread",
    ]

    missing = [
        x for x in required_working
        if x not in working.columns
    ]

    if missing:
        raise ValueError(
            f"Stage-3 file missing columns: {missing}"
        )

    required_outcomes = [
        "game_id",
        "home_score",
        "away_score",
    ]

    missing = [
        x for x in required_outcomes
        if x not in outcomes.columns
    ]

    if missing:
        raise ValueError(
            f"Outcome file missing columns: {missing}"
        )

    working["game_id"] = (
        working["game_id"]
        .map(canonical_id)
    )

    outcomes["game_id"] = (
        outcomes["game_id"]
        .map(canonical_id)
    )

    outcomes = outcomes[
        [
            "game_id",
            "home_score",
            "away_score",
        ]
    ].drop_duplicates(
        "game_id"
    )

    df = working.merge(
        outcomes,
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    numeric = [
        "home_projected_points",
        "away_projected_points",
        "home_spread",
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
            "home_score",
            "away_score",
        ]
    ).copy()

    df["predicted_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    df["actual_margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    # projected minus actual
    df["margin_error"] = (
        df["predicted_margin"]
        - df["actual_margin"]
    )

    df["abs_margin_error"] = (
        df["margin_error"].abs()
    )

    df["signed_model_vs_book"] = (
        df["predicted_margin"]
        + df["home_spread"]
    )

    df["signal_points"] = (
        df["signed_model_vs_book"].abs()
    )

    # Model-favored side.
    df["bet_side"] = np.where(
        df["signed_model_vs_book"] >= 0,
        "home",
        "away",
    )

    spread_result = (
        df["actual_margin"]
        + df["home_spread"]
    )

    df["ats_result"] = np.where(
        spread_result == 0,
        "P",
        np.where(
            (
                (df["bet_side"] == "home")
                & (spread_result > 0)
            )
            |
            (
                (df["bet_side"] == "away")
                & (spread_result < 0)
            ),
            "W",
            "L",
        ),
    )

    rows = []

    for name, start, end in PERIODS:

        part = df[
            (df["game_date"] >= pd.Timestamp(start))
            & (df["game_date"] <= pd.Timestamp(end))
        ].copy()

        rows.append(
            summarize(
                part,
                name,
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
    print("=" * 100)
    print("NBA SPREAD — EXACT STAGE-3 MARGIN ACCURACY")
    print("=" * 100)

    print(
        out.to_string(
            index=False
        )
    )

    if "margin_bias" in working.columns:
        print()
        print("Stage-3 file contains margin_bias.")
        print(
            pd.to_numeric(
                working["margin_bias"],
                errors="coerce",
            ).describe().to_string()
        )
    else:
        print()
        print(
            "Stage-3 file does NOT contain margin_bias."
        )

    print()
    print(f"WROTE: {OUTPUT}")


if __name__ == "__main__":
    main()
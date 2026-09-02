#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path("docs/win/basketball/backtest")

RUN_100 = (
    BASE
    / "runs/nba_margin_100_e2e_20260902_102536Z"
)

RUN_150 = (
    BASE
    / "runs/nba_margin_150_e2e_20260902_102536Z"
)

# The last backtest executed was WINDOW_150, so this
# working file contains its exact 1,118 NBA game population.
COMMON_WORKING = (
    BASE
    / "working/nba/spread/2025_NBA_spread.csv"
)

OUTPUT_DIR = (
    BASE
    / "nba_margin_bias_150_common_game_test"
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


def load_common_games():

    if not COMMON_WORKING.exists():
        raise FileNotFoundError(
            COMMON_WORKING
        )

    df = pd.read_csv(
        COMMON_WORKING,
        low_memory=False,
    )

    if "game_id" not in df.columns:
        raise ValueError(
            f"{COMMON_WORKING} missing game_id"
        )

    ids = set(
        df["game_id"]
        .astype(str)
        .str.strip()
    )

    if len(ids) != 1118:
        raise RuntimeError(
            "Expected exactly 1118 games from "
            "the WINDOW_150 working file, "
            f"but found {len(ids)}. "
            "Do not use this result."
        )

    return ids


def load_selected_spreads(
    run_dir,
    common_ids,
):

    path = (
        run_dir
        / "graded/nba/2025_NBA_graded.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "game_id",
        "market_type",
        "bet_side",
        "bet_result",
        "profit_unit",
        *METRICS.values(),
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    df["game_id"] = (
        df["game_id"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["market_type"]
        .astype(str)
        .str.lower()
        .eq("spread")
        &
        df["game_id"].isin(
            common_ids
        )
    ].copy()

    return df


def ranking_test(
    df,
    column,
):

    work = df[
        df["bet_result"].isin(
            ["Win", "Loss"]
        )
    ].copy()

    work[column] = pd.to_numeric(
        work[column],
        errors="coerce",
    )

    work = work[
        work[column].notna()
    ].copy()

    if len(work) < 25:
        return {
            "n": len(work),
            "low_win_rate": np.nan,
            "high_win_rate": np.nan,
            "spearman": np.nan,
            "works": "TOO_SMALL",
        }

    work = (
        work.sort_values(
            column,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    work["bucket"] = pd.qcut(
        np.arange(len(work)),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    rates = []

    for _, group in work.groupby(
        "bucket",
        observed=True,
        sort=True,
    ):
        wins = int(
            (
                group["bet_result"]
                == "Win"
            ).sum()
        )

        losses = int(
            (
                group["bet_result"]
                == "Loss"
            ).sum()
        )

        rates.append(
            wins / (wins + losses)
        )

    low = rates[0]
    high = rates[-1]

    corr = pd.Series(
        rates
    ).corr(
        pd.Series(
            [1, 2, 3, 4, 5]
        ),
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

    return {
        "n": len(work),
        "low_win_rate": low,
        "high_win_rate": high,
        "spearman": corr,
        "works": works,
    }


def summarize(
    df,
    version,
):

    wins = int(
        (
            df["bet_result"]
            == "Win"
        ).sum()
    )

    losses = int(
        (
            df["bet_result"]
            == "Loss"
        ).sum()
    )

    pushes = int(
        (
            df["bet_result"]
            == "Push"
        ).sum()
    )

    unknown = int(
        (
            df["bet_result"]
            == "Unknown"
        ).sum()
    )

    decisions = (
        wins + losses
    )

    graded = (
        wins
        + losses
        + pushes
    )

    profit = pd.to_numeric(
        df["profit_unit"],
        errors="coerce",
    ).sum()

    row = {
        "version": version,
        "common_game_population": 1118,
        "selected_bets": len(df),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "unknown": unknown,
        "win_rate": (
            wins / decisions
            if decisions
            else np.nan
        ),
        "profit_units": profit,
        "roi": (
            profit / graded
            if graded
            else np.nan
        ),
    }

    rankings = []

    for name, column in METRICS.items():

        result = ranking_test(
            df,
            column,
        )

        row[
            f"{name}_works"
        ] = result["works"]

        rankings.append(
            {
                "version": version,
                "metric": name,
                **result,
            }
        )

    return row, rankings


def bet_keys(df):

    return set(
        zip(
            df["game_id"].astype(str),
            df["bet_side"]
            .astype(str)
            .str.lower(),
        )
    )


def main():

    common_ids = load_common_games()

    df100 = load_selected_spreads(
        RUN_100,
        common_ids,
    )

    df150 = load_selected_spreads(
        RUN_150,
        common_ids,
    )

    summary100, ranking100 = summarize(
        df100,
        "WINDOW_100",
    )

    summary150, ranking150 = summarize(
        df150,
        "WINDOW_150",
    )

    summary = pd.DataFrame(
        [
            summary100,
            summary150,
        ]
    )

    rankings = pd.DataFrame(
        ranking100
        + ranking150
    )

    keys100 = bet_keys(
        df100
    )

    keys150 = bet_keys(
        df150
    )

    overlap = pd.DataFrame(
        [
            {
                "common_game_population":
                    len(common_ids),

                "window_100_selected":
                    len(keys100),

                "window_150_selected":
                    len(keys150),

                "same_selected_bets":
                    len(
                        keys100
                        & keys150
                    ),

                "only_window_100":
                    len(
                        keys100
                        - keys150
                    ),

                "only_window_150":
                    len(
                        keys150
                        - keys100
                    ),
            }
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "nba_spread_common_games_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "nba_spread_common_games_rankings.csv"
    )

    overlap_path = (
        OUTPUT_DIR
        / "nba_spread_common_games_overlap.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    rankings.to_csv(
        ranking_path,
        index=False,
    )

    overlap.to_csv(
        overlap_path,
        index=False,
    )

    print()
    print("=" * 130)
    print(
        "NBA SPREAD — 100 VS 150 — "
        "IDENTICAL 1,118-GAME POPULATION"
    )
    print("=" * 130)

    print()
    print(
        summary[
            [
                "version",
                "common_game_population",
                "selected_bets",
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
        ].to_string(
            index=False
        )
    )

    print()
    print("DETAILED RANKING")

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
    print("SELECTION OVERLAP")

    print(
        overlap.to_string(
            index=False
        )
    )

    print()
    print(
        f"WROTE: {summary_path}"
    )

    print(
        f"WROTE: {ranking_path}"
    )

    print(
        f"WROTE: {overlap_path}"
    )


if __name__ == "__main__":
    main()
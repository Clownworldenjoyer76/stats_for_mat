#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd


BASE = Path("docs/win/basketball")

COMBINED_DIR = BASE / "00_intake/final_combined_files/combined"
ERROR_DIR = BASE / "backtest/error_history"
OUTPUT_DIR = BASE / "backtest/empirical_probability_test"

# Test-only safeguard. Probability rows are not evaluated until the league
# has at least this many PRIOR completed games available.
MIN_PRIOR_GAMES = 100

LEAGUES = {
    "NBA": {
        "combined": COMBINED_DIR / "2025_NBA.csv",
        "errors": ERROR_DIR / "2025_NBA_error_history.csv",
    },
    "NCAAM": {
        "combined": COMBINED_DIR / "2025_NCAAM.csv",
        "errors": ERROR_DIR / "2025_NCAAM_error_history.csv",
    },
    "WNBA": {
        "combined": COMBINED_DIR / "2025_WNBA.csv",
        "errors": ERROR_DIR / "2025_WNBA_error_history.csv",
    },
}


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def empirical_binary_probability(errors, threshold):
    """
    Event being measured:
        actual_error > threshold

    Exact equality represents a push and is excluded from the
    binary win/loss probability.
    """
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    wins = np.sum(arr > threshold)
    losses = np.sum(arr < threshold)

    decisions = wins + losses

    if decisions == 0:
        return np.nan

    return wins / decisions


def profit_unit(result, decimal_odds):
    if not math.isfinite(decimal_odds) or decimal_odds <= 1:
        return np.nan

    if result == "W":
        return decimal_odds - 1.0

    if result == "L":
        return -1.0

    if result == "P":
        return 0.0

    return np.nan


def grade_spread(actual_margin, home_spread, side):
    value = actual_margin + home_spread

    if abs(value) < 1e-12:
        return "P"

    home_wins = value > 0

    if side == "home":
        return "W" if home_wins else "L"

    return "L" if home_wins else "W"


def grade_total(actual_total, line, side):
    value = actual_total - line

    if abs(value) < 1e-12:
        return "P"

    over_wins = value > 0

    if side == "over":
        return "W" if over_wins else "L"

    return "L" if over_wins else "W"


def prepare_combined(path):
    df = pd.read_csv(path)

    required = [
        "game_id",
        "game_date",
        "home_spread",
        "total",
        "home_dk_spread_decimal",
        "away_dk_spread_decimal",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    df["game_date"] = pd.to_datetime(
        df["game_date"].astype(str).str.replace("_", "-"),
        errors="coerce",
    )

    for col in [
        "home_spread",
        "total",
        "home_dk_spread_decimal",
        "away_dk_spread_decimal",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ]:
        df[col] = numeric(df[col])

    return df[
        [
            "game_id",
            "game_date",
            "home_spread",
            "total",
            "home_dk_spread_decimal",
            "away_dk_spread_decimal",
            "dk_total_over_decimal",
            "dk_total_under_decimal",
        ]
    ].copy()


def prepare_errors(path):
    df = pd.read_csv(path)

    required = [
        "game_id",
        "game_date",
        "predicted_margin",
        "actual_margin",
        "margin_error",
        "predicted_total",
        "actual_total",
        "total_error",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    for col in [
        "predicted_margin",
        "actual_margin",
        "margin_error",
        "predicted_total",
        "actual_total",
        "total_error",
    ]:
        df[col] = numeric(df[col])

    return df


def build_league_rows(league, cfg):
    combined = prepare_combined(cfg["combined"])
    errors = prepare_errors(cfg["errors"])

    df = errors.merge(
        combined,
        on="game_id",
        how="inner",
        suffixes=("_error", "_combined"),
        validate="one_to_one",
    )

    # Use the date from the error-history file.
    df["game_date"] = df["game_date_error"]

    df = df.sort_values(
        ["game_date", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    output_rows = []

    # Important:
    # all games on the SAME date use exactly the same history.
    # Nothing from the current date can leak into another game that day.
    for game_date, current_games in df.groupby(
        "game_date",
        sort=True,
    ):
        prior = df[df["game_date"] < game_date]

        prior_margin_errors = (
            prior["margin_error"]
            .dropna()
            .to_numpy(dtype=float)
        )

        prior_total_errors = (
            prior["total_error"]
            .dropna()
            .to_numpy(dtype=float)
        )

        for _, row in current_games.iterrows():

            # ==========================================
            # SPREAD
            # ==========================================

            if (
                len(prior_margin_errors) >= MIN_PRIOR_GAMES
                and pd.notna(row["predicted_margin"])
                and pd.notna(row["actual_margin"])
                and pd.notna(row["home_spread"])
            ):
                # Positive means model likes HOME relative to sportsbook.
                spread_signal = (
                    float(row["predicted_margin"])
                    + float(row["home_spread"])
                )

                # home cover occurs when:
                #
                # actual_margin + home_spread > 0
                #
                # actual_margin =
                # predicted_margin + margin_error
                #
                # therefore:
                # margin_error > -(predicted_margin + home_spread)

                threshold = -spread_signal

                p_home = empirical_binary_probability(
                    prior_margin_errors,
                    threshold,
                )

                if pd.notna(p_home):
                    if spread_signal >= 0:
                        side = "home"
                        probability = p_home
                        decimal_odds = row[
                            "home_dk_spread_decimal"
                        ]
                    else:
                        side = "away"
                        probability = 1.0 - p_home
                        decimal_odds = row[
                            "away_dk_spread_decimal"
                        ]

                    result = grade_spread(
                        float(row["actual_margin"]),
                        float(row["home_spread"]),
                        side,
                    )

                    ev = (
                        probability * decimal_odds - 1.0
                        if pd.notna(decimal_odds)
                        else np.nan
                    )

                    output_rows.append(
                        {
                            "league": league,
                            "game_date": game_date,
                            "game_id": row["game_id"],
                            "market_type": "spread",
                            "bet_side": side,
                            "sportsbook_line": row["home_spread"],
                            "predicted_value": row["predicted_margin"],
                            "signal_points": abs(spread_signal),
                            "signed_signal_points": spread_signal,
                            "prior_games": len(prior_margin_errors),
                            "empirical_probability": probability,
                            "decimal_odds": decimal_odds,
                            "empirical_ev": ev,
                            "bet_result": result,
                            "profit_unit": profit_unit(
                                result,
                                decimal_odds,
                            ),
                        }
                    )

            # ==========================================
            # TOTAL
            # ==========================================

            if (
                len(prior_total_errors) >= MIN_PRIOR_GAMES
                and pd.notna(row["predicted_total"])
                and pd.notna(row["actual_total"])
                and pd.notna(row["total"])
            ):
                # Positive means model likes OVER relative to sportsbook.
                total_signal = (
                    float(row["predicted_total"])
                    - float(row["total"])
                )

                # over occurs when:
                #
                # actual_total > sportsbook_total
                #
                # actual_total =
                # predicted_total + total_error
                #
                # therefore:
                # total_error > -(predicted_total - sportsbook_total)

                threshold = -total_signal

                p_over = empirical_binary_probability(
                    prior_total_errors,
                    threshold,
                )

                if pd.notna(p_over):
                    if total_signal >= 0:
                        side = "over"
                        probability = p_over
                        decimal_odds = row[
                            "dk_total_over_decimal"
                        ]
                    else:
                        side = "under"
                        probability = 1.0 - p_over
                        decimal_odds = row[
                            "dk_total_under_decimal"
                        ]

                    result = grade_total(
                        float(row["actual_total"]),
                        float(row["total"]),
                        side,
                    )

                    ev = (
                        probability * decimal_odds - 1.0
                        if pd.notna(decimal_odds)
                        else np.nan
                    )

                    output_rows.append(
                        {
                            "league": league,
                            "game_date": game_date,
                            "game_id": row["game_id"],
                            "market_type": "total",
                            "bet_side": side,
                            "sportsbook_line": row["total"],
                            "predicted_value": row["predicted_total"],
                            "signal_points": abs(total_signal),
                            "signed_signal_points": total_signal,
                            "prior_games": len(prior_total_errors),
                            "empirical_probability": probability,
                            "decimal_odds": decimal_odds,
                            "empirical_ev": ev,
                            "bet_result": result,
                            "profit_unit": profit_unit(
                                result,
                                decimal_odds,
                            ),
                        }
                    )

    return pd.DataFrame(output_rows)


def add_bucket(df, column, buckets=5):
    work = df.copy()

    valid = work[column].notna()

    work.loc[valid, "_rank"] = (
        work.loc[valid, column]
        .rank(method="first")
    )

    n = int(valid.sum())

    if n == 0:
        work["bucket"] = np.nan
        return work

    work.loc[valid, "bucket"] = (
        (
            (work.loc[valid, "_rank"] - 1)
            * buckets
            / n
        )
        .astype(int)
        .clip(0, buckets - 1)
        + 1
    )

    return work.drop(
        columns=["_rank"],
        errors="ignore",
    )


def summarize_metric(all_rows, metric):
    summaries = []

    for (league, market), group in all_rows.groupby(
        ["league", "market_type"],
        sort=True,
    ):
        work = group[
            group[metric].notna()
        ].copy()

        if work.empty:
            continue

        work = add_bucket(
            work,
            metric,
            buckets=5,
        )

        for bucket, bucket_df in work.groupby(
            "bucket",
            sort=True,
        ):
            decisions = bucket_df[
                bucket_df["bet_result"].isin(["W", "L"])
            ]

            wins = int(
                (decisions["bet_result"] == "W").sum()
            )

            losses = int(
                (decisions["bet_result"] == "L").sum()
            )

            pushes = int(
                (bucket_df["bet_result"] == "P").sum()
            )

            win_rate = (
                wins / (wins + losses)
                if wins + losses > 0
                else np.nan
            )

            profit = bucket_df[
                "profit_unit"
            ].sum(min_count=1)

            roi = (
                profit / len(bucket_df)
                if len(bucket_df) > 0
                and pd.notna(profit)
                else np.nan
            )

            summaries.append(
                {
                    "league": league,
                    "market_type": market,
                    "metric": metric,
                    "bucket": int(bucket),
                    "bets": len(bucket_df),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "average_metric": bucket_df[
                        metric
                    ].mean(),
                    "average_probability": bucket_df[
                        "empirical_probability"
                    ].mean(),
                    "win_rate": win_rate,
                    "profit_units": profit,
                    "roi": roi,
                }
            )

    return pd.DataFrame(summaries)


def build_overall_summary(bucket_df):
    rows = []

    for (league, market, metric), group in bucket_df.groupby(
        ["league", "market_type", "metric"],
        sort=True,
    ):
        group = group.sort_values("bucket")

        lowest = group.iloc[0]
        highest = group.iloc[-1]

        corr = group[
            ["bucket", "win_rate"]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        rows.append(
            {
                "league": league,
                "market_type": market,
                "metric": metric,
                "lowest_bucket_win_rate": lowest[
                    "win_rate"
                ],
                "highest_bucket_win_rate": highest[
                    "win_rate"
                ],
                "win_rate_change": (
                    highest["win_rate"]
                    - lowest["win_rate"]
                ),
                "win_rate_rank_correlation": corr,
                "stronger_metric_better": (
                    "YES"
                    if (
                        pd.notna(corr)
                        and corr > 0
                        and highest["win_rate"]
                        > lowest["win_rate"]
                    )
                    else "NO"
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    league_frames = []

    for league, cfg in LEAGUES.items():
        rows = build_league_rows(
            league,
            cfg,
        )

        league_frames.append(rows)

        print(
            f"{league}: "
            f"{len(rows)} spread/total test rows"
        )

    all_rows = pd.concat(
        league_frames,
        ignore_index=True,
    )

    all_rows["game_date"] = pd.to_datetime(
        all_rows["game_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    rows_path = (
        OUTPUT_DIR
        / "empirical_probability_rows.csv"
    )

    all_rows.to_csv(
        rows_path,
        index=False,
    )

    probability_summary = summarize_metric(
        all_rows,
        "empirical_probability",
    )

    ev_summary = summarize_metric(
        all_rows,
        "empirical_ev",
    )

    buckets = pd.concat(
        [
            probability_summary,
            ev_summary,
        ],
        ignore_index=True,
    )

    buckets_path = (
        OUTPUT_DIR
        / "empirical_probability_buckets.csv"
    )

    buckets.to_csv(
        buckets_path,
        index=False,
    )

    overall = build_overall_summary(
        buckets
    )

    overall_path = (
        OUTPUT_DIR
        / "empirical_probability_summary.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    print()
    print("=" * 75)
    print("EMPIRICAL PROBABILITY TEST")
    print("=" * 75)
    print(
        f"Minimum prior games: {MIN_PRIOR_GAMES}"
    )
    print()

    print(
        overall.to_string(
            index=False
        )
    )

    print()
    print(f"WROTE: {rows_path}")
    print(f"WROTE: {buckets_path}")
    print(f"WROTE: {overall_path}")


if __name__ == "__main__":
    main()
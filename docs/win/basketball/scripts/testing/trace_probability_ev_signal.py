#!/usr/bin/env python3

from pathlib import Path
import math
import numpy as np
import pandas as pd
import yaml


BASE = Path("docs/win/basketball")

WORKING_DIR = BASE / "backtest/working"
COMBINED_DIR = BASE / "00_intake/final_combined_files/combined"
SELECTED_PATH = BASE / "backtest/selections/all_selected.csv"
MODEL_CONFIG_PATH = BASE / "config/model_config.yaml"
OUTPUT_DIR = BASE / "backtest/probability_ev_trace"

LEAGUES = ("nba", "ncaam", "wnba")
MARKETS = ("moneyline", "spread", "total")

METRICS = (
    "model_probability",
    "raw_ev",
    "adjusted_probability",
    "adjusted_ev",
    "signal_points",
)

STAGES = (
    "01_STAGE3_ALL",
    "02_MODEL_EDGE_PASS",
    "03_FINAL_SELECTED",
)

MIN_BETS = 20


def fv(value):
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return np.nan
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


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")

    return data


MODEL_CONFIG = read_yaml(MODEL_CONFIG_PATH)


def model_edge(league, market):
    return float(
        MODEL_CONFIG["leagues"][league]["edge"][market]
    )


def load_outcomes():
    result = {}

    for league in LEAGUES:
        path = (
            COMBINED_DIR
            / f"2025_{league.upper()}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        required = [
            "game_id",
            "home_score",
            "away_score",
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{path} missing {missing}"
            )

        league_map = {}

        for _, row in df.iterrows():
            gid = canonical_id(
                row["game_id"]
            )

            league_map[gid] = {
                "home_score":
                    fv(row["home_score"]),
                "away_score":
                    fv(row["away_score"]),
            }

        result[league] = league_map

        print(
            f"{league.upper()}: "
            f"loaded {len(league_map)} outcomes"
        )

    return result


def load_selected_keys():
    df = pd.read_csv(
        SELECTED_PATH,
        low_memory=False,
    )

    league_col = (
        "league_lower"
        if "league_lower" in df.columns
        else "league"
    )

    required = [
        league_col,
        "game_id",
        "market_type",
        "bet_side",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{SELECTED_PATH} missing {missing}"
        )

    keys = set()

    for _, row in df.iterrows():
        keys.add(
            (
                str(
                    row[league_col]
                ).strip().lower(),

                canonical_id(
                    row["game_id"]
                ),

                str(
                    row["market_type"]
                ).strip().lower(),

                str(
                    row["bet_side"]
                ).strip().lower(),
            )
        )

    return keys, len(df)


def grade_moneyline(
    home_score,
    away_score,
    side,
):
    if not (
        math.isfinite(home_score)
        and math.isfinite(away_score)
    ):
        return ""

    if home_score == away_score:
        return "P"

    home_win = (
        home_score > away_score
    )

    if side == "home":
        return (
            "W"
            if home_win
            else "L"
        )

    return (
        "L"
        if home_win
        else "W"
    )


def grade_spread(
    home_score,
    away_score,
    home_spread,
    side,
):
    if not all(
        math.isfinite(x)
        for x in (
            home_score,
            away_score,
            home_spread,
        )
    ):
        return ""

    value = (
        home_score
        - away_score
        + home_spread
    )

    if abs(value) < 1e-12:
        return "P"

    home_cover = value > 0

    if side == "home":
        return (
            "W"
            if home_cover
            else "L"
        )

    return (
        "L"
        if home_cover
        else "W"
    )


def grade_total(
    home_score,
    away_score,
    total,
    side,
):
    if not all(
        math.isfinite(x)
        for x in (
            home_score,
            away_score,
            total,
        )
    ):
        return ""

    value = (
        home_score
        + away_score
        - total
    )

    if abs(value) < 1e-12:
        return "P"

    over_win = value > 0

    if side == "over":
        return (
            "W"
            if over_win
            else "L"
        )

    return (
        "L"
        if over_win
        else "W"
    )


def side_specs(market):

    if market == "moneyline":
        return {
            "home": {
                "prob":
                    "home_model_prob",
                "raw_ev":
                    "home_ml_ev",
                "adj_prob":
                    "home_ml_adjusted_model_prob",
                "adj_ev":
                    "home_ml_uncertainty_adjusted_ev",
                "signal":
                    "home_ml_signal_points",
            },
            "away": {
                "prob":
                    "away_model_prob",
                "raw_ev":
                    "away_ml_ev",
                "adj_prob":
                    "away_ml_adjusted_model_prob",
                "adj_ev":
                    "away_ml_uncertainty_adjusted_ev",
                "signal":
                    "away_ml_signal_points",
            },
        }

    if market == "spread":
        return {
            "home": {
                "prob":
                    "home_spread_model_prob",
                "raw_ev":
                    "home_spread_ev",
                "adj_prob":
                    "home_spread_adjusted_model_prob",
                "adj_ev":
                    "home_spread_uncertainty_adjusted_ev",
                "signal":
                    "home_spread_signal_points",
            },
            "away": {
                "prob":
                    "away_spread_model_prob",
                "raw_ev":
                    "away_spread_ev",
                "adj_prob":
                    "away_spread_adjusted_model_prob",
                "adj_ev":
                    "away_spread_uncertainty_adjusted_ev",
                "signal":
                    "away_spread_signal_points",
            },
        }

    if market == "total":
        return {
            "over": {
                "prob":
                    "over_model_prob",
                "raw_ev":
                    "over_ev",
                "adj_prob":
                    "over_adjusted_model_prob",
                "adj_ev":
                    "over_uncertainty_adjusted_ev",
                "signal":
                    "over_signal_points",
            },
            "under": {
                "prob":
                    "under_model_prob",
                "raw_ev":
                    "under_ev",
                "adj_prob":
                    "under_adjusted_model_prob",
                "adj_ev":
                    "under_uncertainty_adjusted_ev",
                "signal":
                    "under_signal_points",
            },
        }

    raise ValueError(market)


def build_rows(
    selected_keys,
    outcomes,
):
    rows = []

    unmatched_outcomes = 0

    for league in LEAGUES:

        for market in MARKETS:

            folder = (
                WORKING_DIR
                / league
                / market
            )

            files = sorted(
                folder.glob("*.csv")
            )

            if not files:
                print(
                    f"WARNING: "
                    f"{league.upper()} "
                    f"{market}: no working files"
                )
                continue

            threshold = model_edge(
                league,
                market,
            )

            specs = side_specs(
                market
            )

            count = 0

            for path in files:

                df = pd.read_csv(
                    path,
                    low_memory=False,
                )

                if "game_id" not in df.columns:
                    raise ValueError(
                        f"{path} missing game_id"
                    )

                for _, row in df.iterrows():

                    gid = canonical_id(
                        row["game_id"]
                    )

                    outcome = (
                        outcomes[league]
                        .get(gid)
                    )

                    if outcome is None:
                        unmatched_outcomes += 1
                        continue

                    home_score = outcome[
                        "home_score"
                    ]

                    away_score = outcome[
                        "away_score"
                    ]

                    for side, spec in specs.items():

                        raw_ev = fv(
                            row.get(
                                spec["raw_ev"]
                            )
                        )

                        if market == "moneyline":

                            result = (
                                grade_moneyline(
                                    home_score,
                                    away_score,
                                    side,
                                )
                            )

                        elif market == "spread":

                            result = (
                                grade_spread(
                                    home_score,
                                    away_score,
                                    fv(
                                        row.get(
                                            "home_spread"
                                        )
                                    ),
                                    side,
                                )
                            )

                        else:

                            result = (
                                grade_total(
                                    home_score,
                                    away_score,
                                    fv(
                                        row.get(
                                            "total"
                                        )
                                    ),
                                    side,
                                )
                            )

                        key = (
                            league,
                            gid,
                            market,
                            side,
                        )

                        rows.append(
                            {
                                "league":
                                    league.upper(),

                                "league_lower":
                                    league,

                                "game_id":
                                    gid,

                                "game_date":
                                    str(
                                        row.get(
                                            "game_date",
                                            "",
                                        )
                                    ),

                                "market_type":
                                    market,

                                "bet_side":
                                    side,

                                "model_probability":
                                    fv(
                                        row.get(
                                            spec["prob"]
                                        )
                                    ),

                                "raw_ev":
                                    raw_ev,

                                "adjusted_probability":
                                    fv(
                                        row.get(
                                            spec["adj_prob"]
                                        )
                                    ),

                                "adjusted_ev":
                                    fv(
                                        row.get(
                                            spec["adj_ev"]
                                        )
                                    ),

                                "signal_points":
                                    fv(
                                        row.get(
                                            spec["signal"]
                                        )
                                    ),

                                "model_edge_threshold":
                                    threshold,

                                "model_edge_pass":
                                    (
                                        math.isfinite(
                                            raw_ev
                                        )
                                        and raw_ev
                                        >= threshold
                                    ),

                                "final_selected":
                                    (
                                        key
                                        in selected_keys
                                    ),

                                "bet_result":
                                    result,
                            }
                        )

                        count += 1

            print(
                f"{league.upper()} "
                f"{market}: "
                f"{count} candidate sides"
            )

    print(
        f"Unmatched outcome rows: "
        f"{unmatched_outcomes}"
    )

    return pd.DataFrame(rows)


def stage_data(
    rows,
    stage,
):
    if stage == "01_STAGE3_ALL":
        return rows

    if stage == "02_MODEL_EDGE_PASS":
        return rows[
            rows[
                "model_edge_pass"
            ]
        ]

    if stage == "03_FINAL_SELECTED":
        return rows[
            rows[
                "final_selected"
            ]
        ]

    raise ValueError(stage)


def test_metric(
    df,
    league,
    market,
    stage,
    metric,
):
    work = df[
        (df["league"] == league)
        & (
            df["market_type"]
            == market
        )
        & (
            df["bet_result"]
            .isin(["W", "L"])
        )
    ].copy()

    work[metric] = pd.to_numeric(
        work[metric],
        errors="coerce",
    )

    work = work[
        work[metric].notna()
    ].copy()

    n = len(work)

    if n < MIN_BETS:
        return {
            "league": league,
            "market_type": market,
            "stage": stage,
            "metric": metric,
            "bets": n,
            "lowest_bucket_win_rate":
                np.nan,
            "highest_bucket_win_rate":
                np.nan,
            "win_rate_change":
                np.nan,
            "win_rate_rank_correlation":
                np.nan,
            "stronger_metric_better":
                "TOO_SMALL",
        }

    work = work.sort_values(
        metric,
        kind="stable",
    ).reset_index(
        drop=True
    )

    work["bucket"] = pd.qcut(
        np.arange(n),
        q=5,
        labels=[
            1, 2, 3, 4, 5
        ],
    )

    bucket_rows = []

    for bucket, bdf in work.groupby(
        "bucket",
        observed=True,
        sort=True,
    ):

        wins = int(
            (
                bdf["bet_result"]
                == "W"
            ).sum()
        )

        losses = int(
            (
                bdf["bet_result"]
                == "L"
            ).sum()
        )

        win_rate = (
            wins
            / (wins + losses)
        )

        bucket_rows.append(
            {
                "bucket":
                    int(bucket),

                "win_rate":
                    win_rate,
            }
        )

    bucket_df = pd.DataFrame(
        bucket_rows
    )

    low = float(
        bucket_df.iloc[0][
            "win_rate"
        ]
    )

    high = float(
        bucket_df.iloc[-1][
            "win_rate"
        ]
    )

    corr = (
        bucket_df[
            [
                "bucket",
                "win_rate",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )

    stronger = (
        "YES"
        if (
            pd.notna(corr)
            and corr > 0
            and high > low
        )
        else "NO"
    )

    return {
        "league":
            league,

        "market_type":
            market,

        "stage":
            stage,

        "metric":
            metric,

        "bets":
            n,

        "lowest_bucket_win_rate":
            low,

        "highest_bucket_win_rate":
            high,

        "win_rate_change":
            high - low,

        "win_rate_rank_correlation":
            corr,

        "stronger_metric_better":
            stronger,
    }


def build_summary(rows):
    results = []

    for stage in STAGES:

        current = stage_data(
            rows,
            stage,
        )

        for league in (
            "NBA",
            "NCAAM",
            "WNBA",
        ):

            for market in MARKETS:

                for metric in METRICS:

                    results.append(
                        test_metric(
                            current,
                            league,
                            market,
                            stage,
                            metric,
                        )
                    )

    return pd.DataFrame(results)


def print_summary(summary):

    labels = {
        "model_probability":
            "PROB",

        "raw_ev":
            "RAW_EV",

        "adjusted_probability":
            "ADJ_PROB",

        "adjusted_ev":
            "ADJ_EV",

        "signal_points":
            "POINTS",
    }

    print()
    print("=" * 110)
    print("PROBABILITY / EV SIGNAL TRACE")
    print("=" * 110)

    for league in (
        "NBA",
        "NCAAM",
        "WNBA",
    ):

        print()
        print(league)

        for market in MARKETS:

            print(
                f"  {market.upper()}"
            )

            for stage in STAGES:

                group = summary[
                    (summary["league"] == league)
                    & (
                        summary[
                            "market_type"
                        ]
                        == market
                    )
                    & (
                        summary["stage"]
                        == stage
                    )
                ]

                if group.empty:
                    continue

                parts = []

                for metric in METRICS:

                    r = group[
                        group["metric"]
                        == metric
                    ]

                    if r.empty:
                        continue

                    parts.append(
                        f"{labels[metric]}="
                        f"{r.iloc[0]['stronger_metric_better']}"
                    )

                n = int(
                    group.iloc[0][
                        "bets"
                    ]
                )

                print(
                    f"    "
                    f"{stage:<22} "
                    f"N={n:<5} "
                    + "  ".join(parts)
                )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_keys, selected_count = (
        load_selected_keys()
    )

    print(
        f"Loaded final selected rows: "
        f"{selected_count}"
    )

    outcomes = load_outcomes()

    rows = build_rows(
        selected_keys,
        outcomes,
    )

    if rows.empty:
        raise RuntimeError(
            "ZERO TRACE ROWS BUILT"
        )

    matched = int(
        rows["final_selected"].sum()
    )

    print(
        f"Stage-3 candidate-side rows: "
        f"{len(rows)}"
    )

    print(
        f"Final selections matched: "
        f"{matched} / "
        f"{selected_count}"
    )

    summary = build_summary(
        rows
    )

    rows_path = (
        OUTPUT_DIR
        / "probability_ev_trace_rows.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "probability_ev_trace_summary.csv"
    )

    rows.to_csv(
        rows_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_summary(
        summary
    )

    print()
    print(
        f"WROTE: {rows_path}"
    )

    print(
        f"WROTE: {summary_path}"
    )


if __name__ == "__main__":
    main()
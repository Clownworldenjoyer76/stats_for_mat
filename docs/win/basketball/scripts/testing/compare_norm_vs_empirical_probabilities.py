#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm


BASE = Path("docs/win/basketball")

COMBINED_DIR = BASE / "00_intake/final_combined_files/combined"
ERROR_DIR = BASE / "backtest/error_history"
OUTPUT_DIR = BASE / "backtest/norm_vs_empirical"
MODEL_CONFIG_PATH = BASE / "config/model_config.yaml"

MIN_PRIOR_GAMES = 100

LEAGUES = {
    "NBA": {
        "key": "nba",
        "combined": COMBINED_DIR / "2025_NBA.csv",
        "errors": ERROR_DIR / "2025_NBA_error_history.csv",
    },
    "NCAAM": {
        "key": "ncaam",
        "combined": COMBINED_DIR / "2025_NCAAM.csv",
        "errors": ERROR_DIR / "2025_NCAAM_error_history.csv",
    },
    "WNBA": {
        "key": "wnba",
        "combined": COMBINED_DIR / "2025_WNBA.csv",
        "errors": ERROR_DIR / "2025_WNBA_error_history.csv",
    },
}


def load_model_config():
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


MODEL_CONFIG = load_model_config()


def fv(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def clamp_probability(p):
    return min(max(float(p), 0.01), 0.99)


def apply_calibration(p, cfg):
    method = str((cfg or {}).get("method", "none")).strip().lower()

    if method in {"none", "raw", ""}:
        return float(p)

    if method == "beta":
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)

        intercept = float(cfg["intercept"])
        coef_log_p = float(cfg["coef_log_p"])
        coef_log_1mp = float(cfg["coef_log_1mp"])

        z = (
            intercept
            + coef_log_p * math.log(p)
            + coef_log_1mp * math.log(1.0 - p)
        )

        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)

        ez = math.exp(z)
        return ez / (1.0 + ez)

    raise ValueError(f"Unsupported calibration method: {method}")


def complementary_calibration(raw_first, raw_second, cfg, first_side, second_side):
    canonical = str(
        cfg.get("canonical_side", first_side)
    ).strip().lower()

    side_cfg = cfg.get(canonical) or {"method": "none"}

    raw_canonical = (
        raw_first
        if canonical == first_side
        else raw_second
    )

    calibrated = clamp_probability(
        apply_calibration(raw_canonical, side_cfg)
    )

    opposite = 1.0 - calibrated

    if canonical == first_side:
        return calibrated, opposite

    return opposite, calibrated


def empirical_probability(errors, threshold):
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]

    wins = np.sum(arr > threshold)
    losses = np.sum(arr < threshold)

    decisions = wins + losses

    if decisions == 0:
        return np.nan

    return wins / decisions


def current_spread_probability(league_key, predicted_margin, home_spread):
    cfg = MODEL_CONFIG["leagues"][league_key]

    std = float(
        cfg["std"]["spread"]["value"]
    )

    raw_home = 1.0 - norm.cdf(
        -home_spread,
        loc=predicted_margin,
        scale=std,
    )

    raw_home = clamp_probability(raw_home)
    raw_away = 1.0 - raw_home

    cal = (
        cfg.get("calibration", {})
        .get("spread", {})
    )

    return complementary_calibration(
        raw_home,
        raw_away,
        cal,
        "home",
        "away",
    )


def current_total_probability(league_key, predicted_total, sportsbook_total):
    cfg = MODEL_CONFIG["leagues"][league_key]

    std = float(
        cfg["std"]["total"]["value"]
    )

    z = (
        sportsbook_total
        - predicted_total
    ) / std

    raw_under = clamp_probability(
        norm.cdf(z)
    )

    raw_over = 1.0 - raw_under

    cal = (
        cfg.get("calibration", {})
        .get("total", {})
    )

    return complementary_calibration(
        raw_over,
        raw_under,
        cal,
        "over",
        "under",
    )


def grade_spread(actual_margin, home_spread, side):
    result = actual_margin + home_spread

    if abs(result) < 1e-12:
        return "P"

    home_cover = result > 0

    if side == "home":
        return "W" if home_cover else "L"

    return "L" if home_cover else "W"


def grade_total(actual_total, sportsbook_total, side):
    result = actual_total - sportsbook_total

    if abs(result) < 1e-12:
        return "P"

    over = result > 0

    if side == "over":
        return "W" if over else "L"

    return "L" if over else "W"


def profit(result, decimal):
    if not math.isfinite(decimal) or decimal <= 1:
        return np.nan

    if result == "W":
        return decimal - 1.0

    if result == "L":
        return -1.0

    if result == "P":
        return 0.0

    return np.nan


def load_data(cfg):
    combined = pd.read_csv(cfg["combined"])
    errors = pd.read_csv(cfg["errors"])

    combined["game_date"] = pd.to_datetime(
        combined["game_date"].astype(str).str.replace("_", "-"),
        errors="coerce",
    )

    errors["game_date"] = pd.to_datetime(
        errors["game_date"],
        errors="coerce",
    )

    combined_cols = [
        "game_id",
        "game_date",
        "home_spread",
        "total",
        "home_dk_spread_decimal",
        "away_dk_spread_decimal",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ]

    df = errors.merge(
        combined[combined_cols],
        on="game_id",
        how="inner",
        suffixes=("_error", "_combined"),
        validate="one_to_one",
    )

    df["game_date"] = df["game_date_error"]

    numeric_cols = [
        "predicted_margin",
        "actual_margin",
        "margin_error",
        "predicted_total",
        "actual_total",
        "total_error",
        "home_spread",
        "total",
        "home_dk_spread_decimal",
        "away_dk_spread_decimal",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df.sort_values(
        ["game_date", "game_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_rows(league, cfg):
    df = load_data(cfg)
    league_key = cfg["key"]

    rows = []

    for game_date, current in df.groupby(
        "game_date",
        sort=True,
    ):
        prior = df[
            df["game_date"] < game_date
        ]

        margin_errors = (
            prior["margin_error"]
            .dropna()
            .to_numpy(float)
        )

        total_errors = (
            prior["total_error"]
            .dropna()
            .to_numpy(float)
        )

        for _, row in current.iterrows():

            # =================================================
            # SPREAD
            # =================================================

            if (
                len(margin_errors) >= MIN_PRIOR_GAMES
                and pd.notna(row["predicted_margin"])
                and pd.notna(row["actual_margin"])
                and pd.notna(row["home_spread"])
            ):
                predicted_margin = float(
                    row["predicted_margin"]
                )

                home_spread = float(
                    row["home_spread"]
                )

                signal = (
                    predicted_margin
                    + home_spread
                )

                # Use SAME SIDE for both methods.
                side = (
                    "home"
                    if signal >= 0
                    else "away"
                )

                current_home, current_away = (
                    current_spread_probability(
                        league_key,
                        predicted_margin,
                        home_spread,
                    )
                )

                empirical_home = empirical_probability(
                    margin_errors,
                    -signal,
                )

                if side == "home":
                    current_probability = current_home
                    empirical_prob = empirical_home
                    decimal = fv(
                        row["home_dk_spread_decimal"]
                    )
                else:
                    current_probability = current_away
                    empirical_prob = 1.0 - empirical_home
                    decimal = fv(
                        row["away_dk_spread_decimal"]
                    )

                result = grade_spread(
                    float(row["actual_margin"]),
                    home_spread,
                    side,
                )

                rows.append({
                    "league": league,
                    "game_date": game_date,
                    "game_id": row["game_id"],
                    "market_type": "spread",
                    "bet_side": side,
                    "signal_points": abs(signal),
                    "current_probability": current_probability,
                    "empirical_probability": empirical_prob,
                    "current_ev": (
                        current_probability * decimal - 1.0
                        if math.isfinite(decimal)
                        else np.nan
                    ),
                    "empirical_ev": (
                        empirical_prob * decimal - 1.0
                        if math.isfinite(decimal)
                        else np.nan
                    ),
                    "decimal_odds": decimal,
                    "bet_result": result,
                    "profit_unit": profit(
                        result,
                        decimal,
                    ),
                    "prior_games": len(
                        margin_errors
                    ),
                })

            # =================================================
            # TOTAL
            # =================================================

            if (
                len(total_errors) >= MIN_PRIOR_GAMES
                and pd.notna(row["predicted_total"])
                and pd.notna(row["actual_total"])
                and pd.notna(row["total"])
            ):
                predicted_total = float(
                    row["predicted_total"]
                )

                sportsbook_total = float(
                    row["total"]
                )

                signal = (
                    predicted_total
                    - sportsbook_total
                )

                side = (
                    "over"
                    if signal >= 0
                    else "under"
                )

                current_over, current_under = (
                    current_total_probability(
                        league_key,
                        predicted_total,
                        sportsbook_total,
                    )
                )

                empirical_over = empirical_probability(
                    total_errors,
                    -signal,
                )

                if side == "over":
                    current_probability = current_over
                    empirical_prob = empirical_over
                    decimal = fv(
                        row["dk_total_over_decimal"]
                    )
                else:
                    current_probability = current_under
                    empirical_prob = 1.0 - empirical_over
                    decimal = fv(
                        row["dk_total_under_decimal"]
                    )

                result = grade_total(
                    float(row["actual_total"]),
                    sportsbook_total,
                    side,
                )

                rows.append({
                    "league": league,
                    "game_date": game_date,
                    "game_id": row["game_id"],
                    "market_type": "total",
                    "bet_side": side,
                    "signal_points": abs(signal),
                    "current_probability": current_probability,
                    "empirical_probability": empirical_prob,
                    "current_ev": (
                        current_probability * decimal - 1.0
                        if math.isfinite(decimal)
                        else np.nan
                    ),
                    "empirical_ev": (
                        empirical_prob * decimal - 1.0
                        if math.isfinite(decimal)
                        else np.nan
                    ),
                    "decimal_odds": decimal,
                    "bet_result": result,
                    "profit_unit": profit(
                        result,
                        decimal,
                    ),
                    "prior_games": len(
                        total_errors
                    ),
                })

    return pd.DataFrame(rows)


def assign_buckets(group, metric):
    work = group[
        group[metric].notna()
    ].copy()

    work = work.sort_values(
        metric,
        kind="stable",
    ).reset_index(drop=True)

    if len(work) < 5:
        return pd.DataFrame()

    work["bucket"] = pd.qcut(
        np.arange(len(work)),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    return work


def summarize(rows):
    output = []

    metrics = [
        "current_probability",
        "empirical_probability",
        "current_ev",
        "empirical_ev",
    ]

    for (
        league,
        market
    ), group in rows.groupby(
        ["league", "market_type"],
        sort=True,
    ):

        for metric in metrics:
            work = assign_buckets(
                group,
                metric,
            )

            if work.empty:
                continue

            bucket_results = []

            for bucket, bdf in work.groupby(
                "bucket",
                observed=True,
                sort=True,
            ):
                decisions = bdf[
                    bdf["bet_result"].isin(
                        ["W", "L"]
                    )
                ]

                wins = int(
                    (
                        decisions["bet_result"]
                        == "W"
                    ).sum()
                )

                losses = int(
                    (
                        decisions["bet_result"]
                        == "L"
                    ).sum()
                )

                win_rate = (
                    wins / (wins + losses)
                    if wins + losses
                    else np.nan
                )

                bucket_results.append({
                    "bucket": int(bucket),
                    "win_rate": win_rate,
                })

            b = pd.DataFrame(
                bucket_results
            ).sort_values("bucket")

            corr = b[
                ["bucket", "win_rate"]
            ].corr(
                method="spearman"
            ).iloc[0, 1]

            low = b.iloc[0]["win_rate"]
            high = b.iloc[-1]["win_rate"]

            output.append({
                "league": league,
                "market_type": market,
                "metric": metric,
                "bets": len(work),
                "lowest_bucket_win_rate": low,
                "highest_bucket_win_rate": high,
                "win_rate_change": high - low,
                "win_rate_rank_correlation": corr,
                "stronger_metric_better": (
                    "YES"
                    if (
                        pd.notna(corr)
                        and corr > 0
                        and high > low
                    )
                    else "NO"
                ),
            })

    return pd.DataFrame(output)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    for league, cfg in LEAGUES.items():
        rows = build_rows(
            league,
            cfg,
        )

        frames.append(rows)

        print(
            f"{league}: {len(rows)} "
            f"matched spread/total rows"
        )

    all_rows = pd.concat(
        frames,
        ignore_index=True,
    )

    all_rows["game_date"] = pd.to_datetime(
        all_rows["game_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    summary = summarize(
        all_rows
    )

    rows_path = (
        OUTPUT_DIR
        / "norm_vs_empirical_rows.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "norm_vs_empirical_summary.csv"
    )

    all_rows.to_csv(
        rows_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("CURRENT NORMAL-CURVE VS EMPIRICAL HISTORICAL-ERROR PROBABILITY")
    print("=" * 110)
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(f"WROTE: {rows_path}")
    print(f"WROTE: {summary_path}")


if __name__ == "__main__":
    main()
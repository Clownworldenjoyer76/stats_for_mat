#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


BASE = Path("docs/win/basketball")

INPUT_DIR = (
    BASE
    / "00_intake/final_combined_files/combined"
)

OUTPUT_DIR = (
    BASE
    / "backtest/error_history"
)

FILES = {
    "NBA": INPUT_DIR / "2025_NBA.csv",
    "NCAAM": INPUT_DIR / "2025_NCAAM.csv",
    "WNBA": INPUT_DIR / "2025_WNBA.csv",
}

REQUIRED_COLUMNS = [
    "game_date",
    "game_id",
    "home_team",
    "away_team",
    "home_projected_points",
    "away_projected_points",
    "total_projected_points",
    "home_score",
    "away_score",
    "actual_total",
]


def build_history(league: str, input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{input_path} missing columns: {missing}"
        )

    numeric_columns = [
        "home_projected_points",
        "away_projected_points",
        "total_projected_points",
        "home_score",
        "away_score",
        "actual_total",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["game_date"] = pd.to_datetime(
        df["game_date"].astype(str).str.replace("_", "-"),
        errors="coerce",
    )

    valid = df[
        ["game_date", *numeric_columns]
    ].notna().all(axis=1)

    skipped = int((~valid).sum())
    df = df.loc[valid].copy()

    # -------------------------------------------------
    # Requested error definitions
    # positive error = actual result was ABOVE prediction
    # negative error = actual result was BELOW prediction
    # -------------------------------------------------

    df["predicted_margin"] = (
        df["home_projected_points"]
        - df["away_projected_points"]
    )

    df["actual_margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    df["margin_error"] = (
        df["actual_margin"]
        - df["predicted_margin"]
    )

    df["predicted_total"] = (
        df["total_projected_points"]
    )

    df["total_error"] = (
        df["actual_total"]
        - df["predicted_total"]
    )

    out = df[
        [
            "game_date",
            "game_id",
            "home_team",
            "away_team",
            "predicted_margin",
            "actual_margin",
            "margin_error",
            "predicted_total",
            "actual_total",
            "total_error",
        ]
    ].copy()

    out.insert(0, "league", league)

    out = out.sort_values(
        ["game_date", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    out["game_date"] = out[
        "game_date"
    ].dt.strftime("%Y-%m-%d")

    print(
        f"{league}: "
        f"{len(out)} valid games, "
        f"{skipped} skipped"
    )

    return out


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for league, input_path in FILES.items():
        history = build_history(
            league,
            input_path,
        )

        output_path = (
            OUTPUT_DIR
            / f"2025_{league}_error_history.csv"
        )

        history.to_csv(
            output_path,
            index=False,
        )

        print(f"WROTE: {output_path}")

    print("\nERROR HISTORY BUILD COMPLETE")


if __name__ == "__main__":
    main()
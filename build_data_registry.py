#!/usr/bin/env python3
"""Catalog every raw Retrosheet-derived field and its allowed modeling use.

Scans season files such as 2022batting.csv ... 2025teamstats.csv and writes:
  model/raw_data_registry.csv

The registry distinguishes:
  current_game   known before first pitch and safe directly
  historical_only usable only from games strictly before the prediction date
  never          identifiers/targets/postgame administration not model inputs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

FILE_TYPES = (
    "allplayers", "batting", "fielding", "gameinfo",
    "pitching", "plays", "teamstats",
)

# Exact current-game fields that can be known before first pitch.
CURRENT_SAFE = {
    "gameinfo": {
        "gid", "visteam", "hometeam", "site", "date", "number", "starttime",
        "daynight", "tiebreaker", "usedh", "fieldcond", "precip", "sky",
        "temp", "winddir", "windspeed", "umphome", "ump1b", "ump2b",
        "ump3b", "umplf", "umprf", "gametype", "season",
    },
    "teamstats": {
        "gid", "team", "date", "number", "site", "vishome", "opp",
        "gametype", "start_l1", "start_l2", "start_l3", "start_l4",
        "start_l5", "start_l6", "start_l7", "start_l8", "start_l9",
        "start_f1", "start_f2", "start_f3", "start_f4", "start_f5",
        "start_f6", "start_f7", "start_f8", "start_f9", "start_f10",
    },
    "allplayers": {"id", "last", "first", "bat", "throw", "team"},
}

# Postgame fields can be targets and/or shifted historical features, but never direct current-game inputs.
TARGET_OR_HISTORY = {
    "win", "loss", "tie", "wteam", "lteam", "vruns", "hruns", "wp",
    "lp", "save", "timeofgame", "attendance", "innings", "htbf",
}
ADMIN_NEVER = {"forfeit", "suspend", "oscorer", "box", "pbp", "line", "batteries", "lineups"}
FORBIDDEN_PREFIXES = ("score_", "run_b", "run1", "run2", "run3", "prun_", "ur_", "rbi_")

IDENTIFIER_FIELDS = {
    "gid", "id", "team", "opp", "visteam", "hometeam", "site", "date",
    "number", "event", "batter", "pitcher", "batteam", "pitteam",
}

DESCRIPTIONS = {
    "gid": "Retrosheet game identifier",
    "id": "Retrosheet player identifier",
    "date": "Game date in YYYYMMDD form",
    "site": "Ballpark identifier",
    "team": "Team identifier",
    "opp": "Opponent team identifier",
    "visteam": "Visiting team identifier",
    "hometeam": "Home team identifier",
    "vishome": "Visitor/home indicator",
    "starttime": "Scheduled game start time",
    "daynight": "Day or night game indicator",
    "tiebreaker": "Extra-inning tiebreaker rule",
    "usedh": "Designated hitter rule in use",
    "fieldcond": "Field condition at game time",
    "precip": "Precipitation condition",
    "sky": "Sky condition",
    "temp": "Temperature in Fahrenheit",
    "winddir": "Wind direction",
    "windspeed": "Wind speed",
    "umphome": "Home-plate umpire identifier",
    "bat": "Player batting handedness",
    "throw": "Player throwing handedness",
    "b_pa": "Plate appearances",
    "b_ab": "At-bats",
    "b_r": "Runs scored",
    "b_h": "Hits",
    "b_d": "Doubles",
    "b_t": "Triples",
    "b_hr": "Home runs",
    "b_w": "Walks",
    "b_k": "Strikeouts",
    "b_hbp": "Hit by pitch",
    "b_sb": "Stolen bases",
    "b_cs": "Caught stealing",
    "b_gdp": "Grounded into double plays",
    "p_ipouts": "Pitching outs recorded",
    "p_bfp": "Batters faced",
    "p_h": "Hits allowed",
    "p_d": "Doubles allowed",
    "p_t": "Triples allowed",
    "p_hr": "Home runs allowed",
    "p_r": "Runs allowed",
    "p_er": "Earned runs allowed",
    "p_w": "Walks allowed",
    "p_k": "Strikeouts",
    "p_hbp": "Hit batters",
    "p_wp": "Wild pitches",
    "p_bk": "Balks",
    "d_ifouts": "Defensive innings expressed as outs",
    "d_po": "Putouts",
    "d_a": "Assists",
    "d_e": "Errors",
    "d_dp": "Double plays",
    "d_pb": "Passed balls",
    "nump": "Pitches in plate appearance",
    "bathand": "Batter handedness for the plate appearance",
    "pithand": "Pitcher handedness for the plate appearance",
    "outs_pre": "Outs before the play",
    "outs_post": "Outs after the play",
    "bip": "Ball put in play indicator",
    "ground": "Ground-ball indicator",
    "fly": "Fly-ball indicator",
    "line": "Line-drive indicator",
    "lob": "Team runners left on base",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="baseball_projector")
    p.add_argument("--out", default="model/raw_data_registry.csv")
    p.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--sample-rows", type=int, default=5000)
    return p.parse_args()


def file_type(path: Path) -> str | None:
    name = path.name.lower()
    for kind in FILE_TYPES:
        if name.endswith(f"{kind}.csv"):
            return kind
    return None


def discover(data_dir: Path, seasons: Iterable[int]) -> list[tuple[int, str, Path]]:
    found: list[tuple[int, str, Path]] = []
    for season in seasons:
        for path in sorted(data_dir.rglob(f"{season}*.csv")):
            kind = file_type(path)
            if kind:
                found.append((season, kind, path))
    return found


def generic_description(kind: str, col: str) -> str:
    if col in DESCRIPTIONS:
        return DESCRIPTIONS[col]
    if re.fullmatch(r"inn\d+", col):
        return f"Runs scored in inning {col[3:]}"
    if re.fullmatch(r"start_l\d+", col):
        return f"Starting lineup player in batting slot {col[7:]}"
    if re.fullmatch(r"start_f\d+", col):
        return f"Starting defensive player at fielding position {col[7:]}"
    if re.fullmatch(r"g(_[a-z0-9]+)?", col):
        return "Season games played total or positional games total"
    if col.startswith("b_"):
        return "Batting game statistic"
    if col.startswith("p_"):
        return "Pitching game statistic"
    if col.startswith("d_"):
        return "Fielding game statistic"
    if col.startswith(("po", "a", "e")) and col[-1:].isdigit():
        return "Play-level fielding involvement indicator"
    return f"Raw {kind} field"


def classify(kind: str, col: str) -> tuple[str, str, str, str, str]:
    """Return availability, leakage, transformation, engineered feature, enabled."""
    if col in ADMIN_NEVER:
        return "never", "administrative_or_quality_flag", "exclude", "", "no"
    if col in TARGET_OR_HISTORY or col.startswith(FORBIDDEN_PREFIXES):
        engineered = f"{kind}_{col}_rolling"
        return "historical_only", "current_game_target_or_postgame; safe_only_if_shifted", \
               "target_and_or_shift_then_rolling", engineered, "yes"

    if col in CURRENT_SAFE.get(kind, set()):
        if kind == "allplayers" and col in {"team"}:
            return "historical_only", "season_summary_leak_if_current", "prior_season_only", "prior_player_team", "yes"
        if col in IDENTIFIER_FIELDS:
            engineered = {
                "site": "park_id", "date": "calendar_features", "team": "team_id",
                "opp": "opponent_id", "gid": "game_key",
            }.get(col, col)
            return "current_game", "safe_context_or_key", "direct_or_join_key", engineered, "yes"
        return "current_game", "safe_current", "direct_current", col, "yes"

    if kind == "allplayers":
        if col in {"g", "g_p", "g_sp", "g_rp", "g_c", "g_1b", "g_2b", "g_3b", "g_ss", "g_lf", "g_cf", "g_rf", "g_of", "g_dh", "g_ph", "g_pr"}:
            return "historical_only", "full_season_leak_if_same_season", "prior_season_only", f"prior_{col}", "yes"
        return "never", "administrative_or_posthoc", "exclude", "", "no"

    if kind in {"batting", "pitching", "fielding", "plays", "teamstats"}:
        # Outcome/result flags and starting identities are handled above.
        if col in {"stattype", "gametype"}:
            return "current_game", "safe_filter", "filter_only", "", "yes"
        if col in IDENTIFIER_FIELDS:
            return "historical_only", "safe_join_key", "join_key", "", "yes"
        engineered = f"{kind}_{col}_rolling"
        return "historical_only", "safe_only_if_strictly_shifted", "shift_then_7d_14d_30d_std_prior", engineered, "yes"

    return "never", "unclassified", "manual_review", "", "review"


def infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    parsed = pd.to_numeric(series, errors="coerce")
    if parsed.notna().mean() > 0.95:
        return "numeric_text"
    return "string"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = discover(data_dir, args.seasons)
    if not files:
        raise SystemExit(f"No season CSVs found under {data_dir.resolve()}")

    rows: list[dict[str, object]] = []
    for season, kind, path in files:
        sample = pd.read_csv(path, nrows=args.sample_rows, low_memory=False)
        for col in sample.columns:
            s = sample[col]
            availability, leakage, transform, engineered, enabled = classify(kind, col)
            vals = s.dropna().astype(str).head(5).tolist()
            rows.append({
                "season": season,
                "source_file": path.name,
                "source_type": kind,
                "column_name": col,
                "description": generic_description(kind, col),
                "data_type": infer_dtype(s),
                "available_pregame": availability,
                "leakage_status": leakage,
                "transformation": transform,
                "engineered_feature": engineered,
                "enabled_status": enabled,
                "sample_values": json.dumps(vals),
                "sample_missing_pct": round(float(s.isna().mean() * 100), 3),
                "sample_unique": int(s.nunique(dropna=True)),
            })

    out = pd.DataFrame(rows).sort_values(["source_type", "column_name", "season"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} registry rows for {len(files)} files -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

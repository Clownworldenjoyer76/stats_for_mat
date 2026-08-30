#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/build_closing_lines.py
#
# Builds deterministic closing-line files from immutable sportsbook snapshots.
#
# Input:
#   docs/win/basketball/00_intake/sportsbook_snapshots/{league}/*_odds_snapshot.csv
#
# For each game + market, selects the latest captured sportsbook observation whose
# scraped_at_utc is strictly before scheduled tipoff. Tipoff is reconstructed from
# game_date + game_time in America/New_York, matching the sportsbook intake stage.
#
# Output:
#   docs/win/basketball/05_final_scores/closing_lines/{league}/{game_date}_closing_lines.csv
#
# Historical dates with no immutable pre-tip snapshot are not fabricated from
# mutable/latest-state sportsbook files.

from __future__ import annotations

import math
import re
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

LEAGUES = ["nba", "ncaam", "wnba"]
NY = ZoneInfo("America/New_York")

BASE = Path("docs/win/basketball")
SNAPSHOT_ROOT = BASE / "00_intake/sportsbook_snapshots"
OUTPUT_ROOT = BASE / "05_final_scores/closing_lines"
ERROR_DIR = BASE / "errors/05_final_scores"
LOG_FILE = ERROR_DIR / "build_closing_lines.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COLUMNS = [
    "league", "game_id", "game_date", "game_time", "home_team", "away_team",
    "market_type", "sportsbook_provider", "snapshot_file",
    "closing_observed_at_utc", "scheduled_tipoff_utc", "minutes_before_tipoff",
    "closing_home_spread", "closing_away_spread", "closing_total",
    "closing_home_ml_american", "closing_away_ml_american",
    "closing_home_spread_american", "closing_away_spread_american",
    "closing_over_american", "closing_under_american",
    "closing_home_ml_decimal", "closing_away_ml_decimal",
    "closing_home_spread_decimal", "closing_away_spread_decimal",
    "closing_over_decimal", "closing_under_decimal",
    "closing_home_market_prob", "closing_away_market_prob",
    "closing_home_spread_market_prob", "closing_away_spread_market_prob",
    "closing_over_market_prob", "closing_under_market_prob",
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def log(level: str, message: str) -> None:
    line = f"{now_iso()} | {level:<5} | {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def number(value):
    try:
        if value is None or pd.isna(value) or clean(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def american_to_decimal(value):
    a = number(value)
    if a is None or a == 0:
        return None
    return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))


def decimal_value(row: pd.Series, decimal_col: str, american_col: str):
    dec = number(row.get(decimal_col))
    if dec is not None and dec > 1.0:
        return dec
    return american_to_decimal(row.get(american_col))


def no_vig_pair(decimal_a, decimal_b) -> tuple[float | None, float | None]:
    a = number(decimal_a)
    b = number(decimal_b)
    if a is None or b is None or a <= 1.0 or b <= 1.0:
        return None, None
    pa = 1.0 / a
    pb = 1.0 / b
    total = pa + pb
    if total <= 0:
        return None, None
    return pa / total, pb / total


def parse_tipoff(game_date, game_time):
    d = clean(game_date)
    t = clean(game_time)
    if not d or not t:
        return None
    for date_fmt in ("%Y_%m_%d", "%Y-%m-%d", "%Y/%m/%d"):
        for time_fmt in ("%I:%M %p", "%H:%M", "%I:%M:%S %p", "%H:%M:%S"):
            try:
                dt = datetime.strptime(f"{d} {t}", f"{date_fmt} {time_fmt}")
                return dt.replace(tzinfo=NY).astimezone(UTC)
            except ValueError:
                pass
    return None


def parse_utc(value):
    s = clean(value)
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


SNAPSHOT_RE = re.compile(r"^(?P<stamp>\d{8}T\d{12}Z)_")


def timestamp_from_filename(path: Path):
    match = SNAPSHOT_RE.match(path.name)
    if not match:
        return None
    stamp = match.group("stamp")
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def market_has_data(row: pd.Series, market: str) -> bool:
    if market == "moneyline":
        cols = [
            "home_dk_moneyline_american", "away_dk_moneyline_american",
            "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
        ]
        return any(number(row.get(c)) is not None for c in cols)
    if market == "spread":
        return (
            number(row.get("home_spread")) is not None
            or number(row.get("away_spread")) is not None
        ) and any(
            number(row.get(c)) is not None
            for c in [
                "home_dk_spread_american", "away_dk_spread_american",
                "home_dk_spread_decimal", "away_dk_spread_decimal",
            ]
        )
    if market == "total":
        return number(row.get("total")) is not None and any(
            number(row.get(c)) is not None
            for c in [
                "dk_total_over_american", "dk_total_under_american",
                "dk_total_over_decimal", "dk_total_under_decimal",
            ]
        )
    return False


def candidate_from_snapshot(row: pd.Series, source_file: Path, market: str):
    tipoff = parse_tipoff(row.get("game_date"), row.get("game_time"))
    observed = parse_utc(row.get("scraped_at_utc")) or timestamp_from_filename(source_file)
    if tipoff is None or observed is None:
        return None
    if observed >= tipoff:
        return None
    if not market_has_data(row, market):
        return None

    home_ml_dec = decimal_value(row, "home_dk_moneyline_decimal", "home_dk_moneyline_american")
    away_ml_dec = decimal_value(row, "away_dk_moneyline_decimal", "away_dk_moneyline_american")
    home_sp_dec = decimal_value(row, "home_dk_spread_decimal", "home_dk_spread_american")
    away_sp_dec = decimal_value(row, "away_dk_spread_decimal", "away_dk_spread_american")
    over_dec = decimal_value(row, "dk_total_over_decimal", "dk_total_over_american")
    under_dec = decimal_value(row, "dk_total_under_decimal", "dk_total_under_american")

    home_ml_p, away_ml_p = no_vig_pair(home_ml_dec, away_ml_dec)
    home_sp_p, away_sp_p = no_vig_pair(home_sp_dec, away_sp_dec)
    over_p, under_p = no_vig_pair(over_dec, under_dec)

    return {
        "league": clean(row.get("league")).lower() or "",
        "game_id": clean(row.get("game_id")),
        "game_date": clean(row.get("game_date")),
        "game_time": clean(row.get("game_time")),
        "home_team": clean(row.get("home_team")),
        "away_team": clean(row.get("away_team")),
        "market_type": market,
        "sportsbook_provider": clean(row.get("sportsbook_provider")),
        "snapshot_file": source_file.name,
        "closing_observed_at_utc": observed.isoformat(),
        "scheduled_tipoff_utc": tipoff.isoformat(),
        "minutes_before_tipoff": round((tipoff - observed).total_seconds() / 60.0, 3),
        "closing_home_spread": number(row.get("home_spread")),
        "closing_away_spread": number(row.get("away_spread")),
        "closing_total": number(row.get("total")),
        "closing_home_ml_american": number(row.get("home_dk_moneyline_american")),
        "closing_away_ml_american": number(row.get("away_dk_moneyline_american")),
        "closing_home_spread_american": number(row.get("home_dk_spread_american")),
        "closing_away_spread_american": number(row.get("away_dk_spread_american")),
        "closing_over_american": number(row.get("dk_total_over_american")),
        "closing_under_american": number(row.get("dk_total_under_american")),
        "closing_home_ml_decimal": home_ml_dec,
        "closing_away_ml_decimal": away_ml_dec,
        "closing_home_spread_decimal": home_sp_dec,
        "closing_away_spread_decimal": away_sp_dec,
        "closing_over_decimal": over_dec,
        "closing_under_decimal": under_dec,
        "closing_home_market_prob": home_ml_p,
        "closing_away_market_prob": away_ml_p,
        "closing_home_spread_market_prob": home_sp_p,
        "closing_away_spread_market_prob": away_sp_p,
        "closing_over_market_prob": over_p,
        "closing_under_market_prob": under_p,
        "_observed_dt": observed,
    }


def build_league(league: str) -> tuple[int, int]:
    snapshot_dir = SNAPSHOT_ROOT / league
    output_dir = OUTPUT_ROOT / league
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(snapshot_dir.glob("*_odds_snapshot.csv")) if snapshot_dir.exists() else []
    if not files:
        log("WARN", f"[{league}] no immutable sportsbook snapshots in {snapshot_dir}")
        return 0, 0

    candidates = []
    read_errors = 0

    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            read_errors += 1
            log("ERROR", f"[{league}] failed reading {path.name}: {type(exc).__name__}: {exc}")
            continue

        for _, row in frame.iterrows():
            for market in ("moneyline", "spread", "total"):
                item = candidate_from_snapshot(row, path, market)
                if item is not None:
                    if not item["league"]:
                        item["league"] = league
                    candidates.append(item)

    if not candidates:
        log("WARN", f"[{league}] snapshots contained no usable pre-tip market observations")
        return 0, read_errors

    cand = pd.DataFrame(candidates)
    cand["_game_key"] = cand["game_id"].astype(str).str.strip()
    blank_id = cand["_game_key"].eq("")
    cand.loc[blank_id, "_game_key"] = (
        cand.loc[blank_id, "game_date"].astype(str)
        + "|" + cand.loc[blank_id, "home_team"].astype(str).str.casefold()
        + "|" + cand.loc[blank_id, "away_team"].astype(str).str.casefold()
    )

    cand = cand.sort_values("_observed_dt")
    selected = (
        cand.groupby(["_game_key", "market_type"], as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    selected = selected.drop(columns=["_game_key", "_observed_dt"], errors="ignore")
    selected = selected.reindex(columns=OUTPUT_COLUMNS)

    written = 0
    for game_date, date_df in selected.groupby("game_date", dropna=False):
        if not clean(game_date):
            log("WARN", f"[{league}] skipping {len(date_df)} closing rows with blank game_date")
            continue
        out_path = output_dir / f"{game_date}_closing_lines.csv"
        date_df = date_df.sort_values(["game_id", "market_type"], kind="stable")
        date_df.to_csv(out_path, index=False)
        written += len(date_df)
        log("INFO", f"[{league}] wrote {len(date_df)} closing rows -> {out_path}")

    return written, read_errors


def main() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== build_closing_lines RUN {now_iso()} ===\n")

    total_rows = 0
    errors = 0
    try:
        for league in LEAGUES:
            rows, league_errors = build_league(league)
            total_rows += rows
            errors += league_errors

        log("INFO", f"TOTAL CLOSING ROWS WRITTEN: {total_rows}")
        log("INFO", f"READ ERRORS: {errors}")
        status = "SUCCESS" if errors == 0 else "COMPLETED WITH ERRORS"
        log("INFO", f"STATUS: {status}")
        if errors:
            sys.exit(1)
    except Exception as exc:
        log("ERROR", f"FATAL: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()

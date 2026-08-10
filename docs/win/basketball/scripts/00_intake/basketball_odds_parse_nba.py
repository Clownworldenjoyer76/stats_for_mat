#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_odds_parse_nba.py

import csv
import json
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# =========================
# PATHS
# =========================

ODDS_DIR = Path("docs/win/basketball/odds/nba")
OUTPUT_DIR = Path("docs/win/basketball/00_intake/sportsbook/nba")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

BOOKMAKER_NAME = "FanDuel"

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_odds_parse_nba.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== basketball_odds_parse_nba RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# DECIMAL TO AMERICAN
# =========================

def decimal_to_american(dec):
    if dec is None:
        return None

    try:
        dec = float(dec)
    except Exception:
        return None

    if dec >= 2.0:
        return round((dec - 1) * 100)

    return round(-100 / (dec - 1))


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


# =========================
# MARKET HELPERS
# =========================

def get_market(markets: list[dict], market_name: str) -> dict | None:
    for market in markets:
        if market.get("name") == market_name:
            return market

    return None


def latest_updated_at(markets: list[dict]) -> str | None:
    updates = []

    for market in markets:
        if market.get("name") not in {"ML", "Spread", "Totals"}:
            continue

        updated_at = market.get("updatedAt")
        if updated_at:
            updates.append(updated_at)

    if not updates:
        return None

    return max(updates)


def select_main_total(total_rows: list[dict]) -> dict | None:
    candidates = []

    for row in total_rows:
        total = safe_float(row.get("hdp"))
        over = safe_float(row.get("over"))
        under = safe_float(row.get("under"))

        if total is None or over is None or under is None:
            continue

        balance_score = abs(over - under)
        even_score = abs(over - 1.91) + abs(under - 1.91)

        candidates.append(
            {
                "row": row,
                "balance_score": balance_score,
                "even_score": even_score,
                "total": total,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            round(x["balance_score"], 4),
            round(x["even_score"], 4),
            abs(x["total"] - 216.5),
        )
    )

    return candidates[0]["row"]


# =========================
# PARSE ONE GAME
# =========================

def parse_game(item: dict):
    event = item.get("event") or {}
    odds_by_source = item.get("odds") or {}

    bookmaker_payload = odds_by_source.get(BOOKMAKER_NAME)
    if not isinstance(bookmaker_payload, dict):
        return None, None

    bookmaker_markets = (
        bookmaker_payload
        .get("bookmakers", {})
        .get(BOOKMAKER_NAME, [])
    )

    if not isinstance(bookmaker_markets, list):
        return None, None

    game_id = event.get("id")
    home_team = event.get("home")
    away_team = event.get("away")
    commence_time = event.get("date")

    if not game_id or not home_team or not away_team or not commence_time:
        return None, None

    commence_utc = datetime.fromisoformat(
        commence_time.replace("Z", "+00:00")
    ).astimezone(UTC_TZ)

    commence_ny = commence_utc.astimezone(NY_TZ)
    game_date = commence_ny.strftime("%Y_%m_%d")
    game_time = commence_ny.strftime("%I:%M %p")

    odds_last_update = latest_updated_at(bookmaker_markets)

    away_ml_dec = home_ml_dec = None
    away_spread_dec = home_spread_dec = None
    away_spread = home_spread = None
    over_dec = under_dec = total = None

    ml_market = get_market(bookmaker_markets, "ML")
    if ml_market:
        ml_rows = ml_market.get("odds") or []
        if ml_rows:
            ml = ml_rows[0]
            home_ml_dec = ml.get("home")
            away_ml_dec = ml.get("away")

    spread_market = get_market(bookmaker_markets, "Spread")
    if spread_market:
        spread_rows = spread_market.get("odds") or []
        if spread_rows:
            spread = spread_rows[0]
            hdp = safe_float(spread.get("hdp"))

            if hdp is not None:
                home_spread = hdp
                away_spread = -hdp

            home_spread_dec = spread.get("home")
            away_spread_dec = spread.get("away")

    totals_market = get_market(bookmaker_markets, "Totals")
    if totals_market:
        total_rows = totals_market.get("odds") or []
        main_total = select_main_total(total_rows)

        if main_total:
            total = main_total.get("hdp")
            over_dec = main_total.get("over")
            under_dec = main_total.get("under")

    row = {
        "sport": "Basketball",
        "league": "NBA",
        "game_date": game_date,
        "game_id": game_id,
        "odds_last_update": odds_last_update,
        "game_time": game_time,
        "home_team": home_team,
        "away_team": away_team,
        "home_spread": home_spread,
        "away_spread": away_spread,
        "total": total,
        "home_dk_moneyline_american": decimal_to_american(home_ml_dec),
        "away_dk_moneyline_american": decimal_to_american(away_ml_dec),
        "home_dk_spread_american": decimal_to_american(home_spread_dec),
        "away_dk_spread_american": decimal_to_american(away_spread_dec),
        "dk_total_over_american": decimal_to_american(over_dec),
        "dk_total_under_american": decimal_to_american(under_dec),
        "home_dk_moneyline_decimal": home_ml_dec,
        "away_dk_moneyline_decimal": away_ml_dec,
        "home_dk_spread_decimal": home_spread_dec,
        "away_dk_spread_decimal": away_spread_dec,
        "dk_total_over_decimal": over_dec,
        "dk_total_under_decimal": under_dec,
    }

    return game_date, row


# =========================
# MAIN
# =========================

def main():
    files_written = []
    games_parsed = 0
    games_skipped = 0

    fieldnames = [
        "sport", "league", "game_date", "game_id", "odds_last_update",
        "game_time", "home_team", "away_team",
        "home_spread", "away_spread", "total",
        "home_dk_moneyline_american", "away_dk_moneyline_american",
        "home_dk_spread_american", "away_dk_spread_american",
        "dk_total_over_american", "dk_total_under_american",
        "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
        "home_dk_spread_decimal", "away_dk_spread_decimal",
        "dk_total_over_decimal", "dk_total_under_decimal",
    ]

    try:
        json_files = sorted(ODDS_DIR.glob("*.json"))

        if not json_files:
            log(f"No JSON files found in {ODDS_DIR}")
            log("STATUS: SUCCESS (nothing to do)")
            return

        for json_path in json_files:
            log(f"Processing {json_path.name}")

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                if not isinstance(payload, dict):
                    log(f"ERROR processing {json_path.name}: expected top-level object, got {type(payload)}")
                    continue

                events = payload.get("events")

                if not isinstance(events, list):
                    log(f"ERROR processing {json_path.name}: expected payload['events'] list")
                    continue

                by_date = defaultdict(list)

                for item in events:
                    try:
                        game_date, row = parse_game(item)

                        if row is not None:
                            by_date[game_date].append(row)
                            games_parsed += 1
                        else:
                            games_skipped += 1

                    except Exception as e:
                        games_skipped += 1

                        event = item.get("event") if isinstance(item, dict) else {}
                        event_id = event.get("id", "?") if isinstance(event, dict) else "?"

                        log(
                            f"  ERROR parsing game {event_id}: {e}\n"
                            f"{traceback.format_exc()}"
                        )

                for game_date, rows in by_date.items():
                    out_path = OUTPUT_DIR / f"{game_date}_NBA_odds.csv"

                    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)

                    files_written.append((str(out_path), len(rows)))
                    log(f"  WROTE {out_path.name} ({len(rows)} games)")

            except Exception as e:
                log(f"ERROR processing {json_path.name}: {e}\n{traceback.format_exc()}")

        log("--- SUMMARY ---")
        log(f"Games parsed: {games_parsed}")
        log(f"Games skipped: {games_skipped}")
        log(f"Files written: {len(files_written)}")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} games)")

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()

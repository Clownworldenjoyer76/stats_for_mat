#!/usr/bin/env python3

# docs/win/basketball/scripts/00_intake/basketball_odds_parse_wnba.py

import csv
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# =========================
# PATHS
# =========================

INPUT_DIR = Path("docs/win/basketball/odds/wnba")
OUTPUT_DIR = Path("docs/win/basketball/00_intake/sportsbook/wnba")

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
LOG_FILE = ERROR_DIR / "basketball_odds_parse_wnba.txt"

# =========================
# CONSTANTS
# =========================

SPORT = "Basketball"
LEAGUE = "WNBA"
BOOKMAKER = "FanDuel"
NY_TZ = ZoneInfo("America/New_York")

OUTPUT_HEADERS = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "odds_last_update",
    "game_time",
    "home_team",
    "away_team",
    "home_spread",
    "away_spread",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

# =========================
# LOGGING
# =========================


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================
# HELPERS
# =========================


def parse_utc_datetime(value: str) -> datetime | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def local_game_date(value: str) -> str:
    dt = parse_utc_datetime(value)
    if not dt:
        return ""

    return dt.astimezone(NY_TZ).strftime("%Y_%m_%d")


def local_game_time(value: str) -> str:
    dt = parse_utc_datetime(value)
    if not dt:
        return ""

    return dt.astimezone(NY_TZ).strftime("%I:%M %p")


def clean_decimal(value) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return ""

    return text


def decimal_to_american(value) -> str:
    text = clean_decimal(value)
    if not text:
        return ""

    try:
        dec = float(text)
    except Exception:
        return ""

    if dec <= 1:
        return ""

    if dec >= 2:
        american = round((dec - 1) * 100)
    else:
        american = round(-100 / (dec - 1))

    return str(int(american))


def number_to_text(value) -> str:
    if value is None:
        return ""

    try:
        f = float(value)
        if math.isfinite(f):
            if f.is_integer():
                return str(int(f))
            return str(f)
    except Exception:
        pass

    return str(value).strip()


def flip_spread(value) -> str:
    if value is None:
        return ""

    try:
        f = float(value)
        flipped = -f
        if flipped.is_integer():
            return str(int(flipped))
        return str(flipped)
    except Exception:
        return ""


def safe_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(f):
        return None

    return f


def get_market(markets: list[dict], market_name: str) -> dict | None:
    target = market_name.strip().lower()

    for market in markets:
        name = str(market.get("name", "")).strip().lower()
        if name == target:
            return market

    return None


def first_odds(market: dict | None) -> dict:
    if not market:
        return {}

    odds = market.get("odds")
    if isinstance(odds, list) and odds:
        if isinstance(odds[0], dict):
            return odds[0]

    return {}


def select_main_total(market: dict | None) -> dict:
    if not market:
        return {}

    odds = market.get("odds")
    if not isinstance(odds, list) or not odds:
        return {}

    candidates = []

    for row in odds:
        if not isinstance(row, dict):
            continue

        total = safe_float(row.get("hdp"))
        over = safe_float(row.get("over"))
        under = safe_float(row.get("under"))

        if total is None or over is None or under is None:
            continue

        if over <= 1 or under <= 1:
            continue

        balance_score = abs(over - under)
        market_price_score = abs(over - 1.91) + abs(under - 1.91)

        candidates.append(
            (
                balance_score,
                market_price_score,
                row,
            )
        )

    if not candidates:
        return {}

    candidates.sort(key=lambda item: (item[0], item[1]))

    return candidates[0][2]


def latest_updated_at(markets: list[dict]) -> str:
    latest_raw = ""
    latest_dt = None

    for market in markets:
        updated = market.get("updatedAt")
        if not updated:
            continue

        dt = parse_utc_datetime(updated)
        if dt and (latest_dt is None or dt > latest_dt):
            latest_dt = dt
            latest_raw = updated

    return latest_raw


def parse_event_row(wrapper: dict) -> dict | None:
    event = wrapper.get("event") or {}
    odds_block = wrapper.get("odds") or {}

    fd = odds_block.get(BOOKMAKER) or {}
    bookmakers = fd.get("bookmakers") or {}
    markets = bookmakers.get(BOOKMAKER) or []

    if not event or not isinstance(markets, list):
        return None

    event_id = event.get("id") or fd.get("id")
    event_date = event.get("date") or fd.get("date")

    ml_market = get_market(markets, "ML")
    spread_market = get_market(markets, "Spread")
    totals_market = get_market(markets, "Totals")

    ml_odds = first_odds(ml_market)
    spread_odds = first_odds(spread_market)
    totals_odds = select_main_total(totals_market)

    home_spread_value = spread_odds.get("hdp")
    total_value = totals_odds.get("hdp")

    home_ml_decimal = clean_decimal(ml_odds.get("home"))
    away_ml_decimal = clean_decimal(ml_odds.get("away"))

    home_spread_decimal = clean_decimal(spread_odds.get("home"))
    away_spread_decimal = clean_decimal(spread_odds.get("away"))

    total_over_decimal = clean_decimal(totals_odds.get("over"))
    total_under_decimal = clean_decimal(totals_odds.get("under"))

    return {
        "sport": SPORT,
        "league": LEAGUE,
        "game_date": local_game_date(event_date),
        "game_id": f"WNBA_{event_id}" if event_id else "",
        "odds_last_update": latest_updated_at(markets),
        "game_time": local_game_time(event_date),
        "home_team": event.get("home") or fd.get("home") or "",
        "away_team": event.get("away") or fd.get("away") or "",
        "home_spread": number_to_text(home_spread_value),
        "away_spread": flip_spread(home_spread_value),
        "total": number_to_text(total_value),
        "home_dk_moneyline_american": decimal_to_american(home_ml_decimal),
        "away_dk_moneyline_american": decimal_to_american(away_ml_decimal),
        "home_dk_spread_american": decimal_to_american(home_spread_decimal),
        "away_dk_spread_american": decimal_to_american(away_spread_decimal),
        "dk_total_over_american": decimal_to_american(total_over_decimal),
        "dk_total_under_american": decimal_to_american(total_under_decimal),
        "home_dk_moneyline_decimal": home_ml_decimal,
        "away_dk_moneyline_decimal": away_ml_decimal,
        "home_dk_spread_decimal": home_spread_decimal,
        "away_dk_spread_decimal": away_spread_decimal,
        "dk_total_over_decimal": total_over_decimal,
        "dk_total_under_decimal": total_under_decimal,
    }


def parse_file(input_file: Path) -> int:
    with input_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    events = payload.get("events") or []
    if not isinstance(events, list):
        log(f"WARN: Skipping {input_file}; events is not a list")
        return 0

    rows = []

    for wrapper in events:
        try:
            row = parse_event_row(wrapper)
            if row:
                rows.append(row)
            else:
                log(
                    f"WARN: Skipped event in {input_file}; "
                    "no usable FanDuel markets"
                )
        except Exception:
            log(f"WARN: Failed parsing event in {input_file}")
            log(traceback.format_exc())

    if not rows:
        log(f"WARN: No rows parsed from {input_file}")
        return 0

    output_date = (
        payload.get("game_date")
        or rows[0].get("game_date")
        or input_file.stem.replace("_wnba", "")
    )
    output_file = OUTPUT_DIR / f"{output_date}_WNBA_odds.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    log(f"WROTE: {output_file} ({len(rows)} rows)")
    return len(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE.write_text(
        (
            "=== basketball_odds_parse_wnba.py RUN "
            f"{datetime.now().isoformat(timespec='seconds')} ===\n"
        ),
        encoding="utf-8",
    )

    try:
        files = sorted(INPUT_DIR.glob("*_wnba.json"))

        if not files:
            log(f"WARN: No input files found in {INPUT_DIR}")
            return 0

        total_rows = 0
        parsed_files = 0

        for input_file in files:
            log(f"READING: {input_file}")
            row_count = parse_file(input_file)
            if row_count > 0:
                parsed_files += 1
                total_rows += row_count

        log(
            f"SUCCESS: Parsed {parsed_files} file(s), "
            f"wrote {total_rows} row(s)"
        )
        return 0

    except Exception:
        log("ERROR: Script failed")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

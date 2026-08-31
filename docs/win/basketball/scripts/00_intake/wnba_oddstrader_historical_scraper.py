#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


OUTPUT_DIR = Path(r"C:\Users\Mat\Downloads\chatgpt_is_retarded")
BASE_URL = "https://www.oddstrader.com/wnba/"
ET = ZoneInfo("America/New_York")

FIELDNAMES = [
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

MARKET_URL_VALUE = {
    "spread": "spread",
    "moneyline": "money",
    "total": "total",
}

AMERICAN_RE = re.compile(r"^[+-]\d+$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")


def clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def normalize_team(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).casefold()


def normalize_odds_text(value: str) -> str:
    text = clean_text(value)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("½", ".5")
    text = text.replace("PK", "0").replace("Pk", "0").replace("pk", "0")
    return text.strip()


def parse_american(value: str) -> str:
    text = normalize_odds_text(value).upper()
    if text in {"EV", "EVEN", "EVENS"}:
        return "+100"
    if AMERICAN_RE.fullmatch(text):
        return text
    return ""


def parse_number(value: str) -> str:
    text = normalize_odds_text(value)
    if NUMBER_RE.fullmatch(text):
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return format(number, ".15g")
    return ""


def american_to_decimal(value: str) -> str:
    american = parse_american(value)
    if not american:
        return ""
    number = int(american)
    if number > 0:
        decimal = 1.0 + number / 100.0
    elif number < 0:
        decimal = 1.0 + 100.0 / abs(number)
    else:
        return ""
    return (f"{decimal:.6f}").rstrip("0").rstrip(".")


def parse_market_button(top: str, bottom: str, market: str) -> dict[str, str] | None:
    top = normalize_odds_text(top)
    bottom = normalize_odds_text(bottom)

    if not top and not bottom:
        return None

    if market == "moneyline":
        american = parse_american(bottom) or parse_american(top)
        if not american:
            return None
        return {"american": american}

    american = parse_american(bottom)
    if not american:
        return None

    if market == "spread":
        line = parse_number(top)
        if not line:
            return None
        return {"line": line, "american": american}

    if market == "total":
        lowered = top.casefold()
        if not lowered or lowered[0] not in {"o", "u"}:
            return None
        side = "over" if lowered[0] == "o" else "under"
        line = parse_number(top[1:])
        if not line:
            return None
        return {"side": side, "line": line, "american": american}

    return None


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def url_for(day: date, market: str) -> str:
    return (
        f"{BASE_URL}?date={day.strftime('%Y%m%d')}"
        f"&g=game&m={MARKET_URL_VALUE[market]}"
    )


def goto(page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    try:
        page.wait_for_selector("table", timeout=15000)
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(2500)


def get_state(page) -> dict:
    state = page.evaluate("() => window.__INITIAL_STATE__ || null")
    return state if isinstance(state, dict) else {}


def team_name_from_participant(participant: dict) -> str:
    source = participant.get("source") or {}
    return clean_text(
        source.get("nam")
        or source.get("sn")
        or source.get("abbr")
    )


def events_for_day(state: dict, requested_day: date) -> list[dict]:
    events_state = state.get("events") or {}
    raw_events = events_state.get("events") or {}

    if not isinstance(raw_events, dict):
        return []

    events: list[dict] = []

    for event in raw_events.values():
        if not isinstance(event, dict):
            continue
        if str(event.get("lid")) != "15":
            continue

        try:
            event_dt = datetime.fromtimestamp(
                float(event["dt"]) / 1000.0,
                tz=timezone.utc,
            ).astimezone(ET)
        except (KeyError, TypeError, ValueError, OSError):
            continue

        if event_dt.date() != requested_day:
            continue

        participants = event.get("participants") or {}
        if not isinstance(participants, dict):
            continue

        home = None
        away = None
        for participant in participants.values():
            if not isinstance(participant, dict):
                continue
            if participant.get("ih") is True:
                home = participant
            elif participant.get("ih") is False:
                away = participant

        if home is None or away is None:
            continue

        home_name = team_name_from_participant(home)
        away_name = team_name_from_participant(away)
        if not home_name or not away_name:
            continue

        events.append({
            "event_id": clean_text(event.get("eid")),
            "game_time": event_dt.strftime("%I:%M %p"),
            "home_team": home_name,
            "away_team": away_name,
        })

    events.sort(key=lambda row: (row["game_time"], row["away_team"], row["home_team"]))
    return events


def read_grid_column(grid, column_index: int, market: str) -> dict[str, str] | None:
    columns = grid.locator('div[class*="gridColumn-"]')
    if column_index < 0 or column_index >= columns.count():
        return None

    column = columns.nth(column_index)
    buttons = column.locator("button.optionBox")
    if buttons.count() == 0:
        return None

    button = buttons.first
    top_locator = button.locator(".odds-top")
    bottom_locator = button.locator(".odds-bottom")

    top = top_locator.first.inner_text().strip() if top_locator.count() else ""
    bottom = bottom_locator.first.inner_text().strip() if bottom_locator.count() else ""
    return parse_market_button(top, bottom, market)


def betonline_grid_index(state: dict) -> int:
    # Grid column 0 is the synthetic OddsTrader "Opener" column.
    # Sportsbook columns follow it. Determine BetOnline's position from
    # the sportsbook list instead of hard-coding the sportsbook index.
    sportsbook_state = state.get("sportsbooks") or {}
    books = sportsbook_state.get("sportsbooks") or []
    if not isinstance(books, list):
        return 1

    enabled = [book for book in books if isinstance(book, dict) and book.get("enabled", True)]
    enabled.sort(key=lambda book: (book.get("ord", 999999), clean_text(book.get("nam"))))

    for position, book in enumerate(enabled):
        if clean_text(book.get("nam")).casefold() == "betonline":
            return 1 + position

    return 1


def scrape_market(page, day: date, market: str, expected_teams: set[str]) -> dict[str, dict[str, str] | None]:
    goto(page, url_for(day, market))

    team_locator = page.locator(".teamName")
    team_count = team_locator.count()
    if team_count == 0:
        return {}

    # OddsTrader renders the line grids beside the team rows rather than as
    # reliable descendants of each <tr>. Pair the team-name list and line-grid
    # list by their rendered order.
    grid_locator = page.locator('div[class*="linesGrid-"]')
    grid_count = grid_locator.count()

    if grid_count < team_count:
        page.wait_for_timeout(2500)
        grid_count = grid_locator.count()

    state = get_state(page)
    betonline_index = betonline_grid_index(state)

    result: dict[str, dict[str, str] | None] = {}

    pair_count = min(team_count, grid_count)
    for i in range(pair_count):
        team_name = clean_text(team_locator.nth(i).inner_text())
        key = normalize_team(team_name)
        if expected_teams and key not in expected_teams:
            continue

        grid = grid_locator.nth(i)
        opener = read_grid_column(grid, 0, market)
        betonline = read_grid_column(grid, betonline_index, market)

        # User-specified priority: Opener first, BetOnline fallback.
        result[key] = opener if opener is not None else betonline

    return result


def build_rows(day: date, events: list[dict], market_data: dict[str, dict[str, dict[str, str] | None]]) -> list[dict]:
    rows: list[dict] = []

    for event in events:
        home_key = normalize_team(event["home_team"])
        away_key = normalize_team(event["away_team"])

        spread_home = market_data.get("spread", {}).get(home_key)
        spread_away = market_data.get("spread", {}).get(away_key)
        money_home = market_data.get("moneyline", {}).get(home_key)
        money_away = market_data.get("moneyline", {}).get(away_key)
        total_home = market_data.get("total", {}).get(home_key)
        total_away = market_data.get("total", {}).get(away_key)

        over = None
        under = None
        for candidate in (total_home, total_away):
            if not candidate:
                continue
            if candidate.get("side") == "over":
                over = candidate
            elif candidate.get("side") == "under":
                under = candidate

        total = ""
        if over and under and over.get("line") == under.get("line"):
            total = clean_text(over.get("line"))
        elif over and not under:
            total = clean_text(over.get("line"))
        elif under and not over:
            total = clean_text(under.get("line"))

        home_ml_am = clean_text((money_home or {}).get("american"))
        away_ml_am = clean_text((money_away or {}).get("american"))
        home_spread_am = clean_text((spread_home or {}).get("american"))
        away_spread_am = clean_text((spread_away or {}).get("american"))
        over_am = clean_text((over or {}).get("american"))
        under_am = clean_text((under or {}).get("american"))

        row = {
            "sport": "Basketball",
            "league": "WNBA",
            "game_date": day.strftime("%Y_%m_%d"),
            "game_id": "",
            "odds_last_update": "",
            "game_time": event["game_time"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "home_spread": clean_text((spread_home or {}).get("line")),
            "away_spread": clean_text((spread_away or {}).get("line")),
            "total": total,
            "home_dk_moneyline_american": home_ml_am,
            "away_dk_moneyline_american": away_ml_am,
            "home_dk_spread_american": home_spread_am,
            "away_dk_spread_american": away_spread_am,
            "dk_total_over_american": over_am,
            "dk_total_under_american": under_am,
            "home_dk_moneyline_decimal": american_to_decimal(home_ml_am),
            "away_dk_moneyline_decimal": american_to_decimal(away_ml_am),
            "home_dk_spread_decimal": american_to_decimal(home_spread_am),
            "away_dk_spread_decimal": american_to_decimal(away_spread_am),
            "dk_total_over_decimal": american_to_decimal(over_am),
            "dk_total_under_decimal": american_to_decimal(under_am),
        }
        rows.append(row)

    return rows


def write_csv(day: date, rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{day.strftime('%Y_%m_%d')}_WNBA_odds.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def scrape_day(page, day: date) -> tuple[Path | None, int]:
    # Load one rendered WNBA page to get the actual events, home/away identity,
    # and scheduled time from OddsTrader's own page state.
    goto(page, url_for(day, "spread"))
    state = get_state(page)
    events = events_for_day(state, day)

    if not events:
        return None, 0

    expected_teams = {
        normalize_team(event["home_team"])
        for event in events
    } | {
        normalize_team(event["away_team"])
        for event in events
    }

    market_data = {
        "spread": scrape_market(page, day, "spread", expected_teams),
        "moneyline": scrape_market(page, day, "moneyline", expected_teams),
        "total": scrape_market(page, day, "total", expected_teams),
    }

    rows = build_rows(day, events, market_data)
    path = write_csv(day, rows)
    return path, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape historical WNBA odds from rendered OddsTrader pages."
    )
    parser.add_argument("--date", help="One date only, YYYY-MM-DD")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite CSVs that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        days = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    else:
        if args.end_year < args.start_year:
            raise ValueError("--end-year must be >= --start-year")
        days = []
        for year in range(args.start_year, args.end_year + 1):
            # Covers WNBA regular season and playoffs for the requested years.
            days.extend(daterange(date(year, 5, 1), date(year, 10, 31)))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 2400, "height": 1400},
            timezone_id="America/New_York",
            locale="en-US",
        )
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0.0.0 Safari/537.36"
            )
        })

        written = 0
        failed = 0

        try:
            for day in days:
                output_path = OUTPUT_DIR / f"{day.strftime('%Y_%m_%d')}_WNBA_odds.csv"
                if output_path.exists() and not args.overwrite:
                    print(f"SKIP {day} | exists | {output_path}")
                    continue

                try:
                    path, row_count = scrape_day(page, day)
                    if path is None:
                        print(f"NO GAMES {day}")
                    else:
                        written += 1
                        print(f"WROTE {day} | games={row_count} | {path}")
                except Exception as exc:
                    failed += 1
                    print(f"FAILED {day} | {type(exc).__name__}: {exc}")

                time.sleep(0.15)
        finally:
            browser.close()

    print(f"DONE | files_written={written} | failed_dates={failed}")


if __name__ == "__main__":
    main()

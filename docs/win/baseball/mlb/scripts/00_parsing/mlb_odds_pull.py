#!/usr/bin/env python3
# docs/win/baseball/scripts/00_parsing/mlb_odds_pull.py

import requests
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


API_KEY = os.getenv("API_ODDS")

if not API_KEY:
    raise RuntimeError("API_ODDS environment variable is not set")


BASE_URL = "https://api.odds-api.io/v3"

SPORT = "baseball"
LEAGUE = "usa-mlb"

PRIMARY_BOOKMAKER = "DraftKings"
FALLBACK_BOOKMAKER = "FanDuel"
BOOKMAKERS = [PRIMARY_BOOKMAKER, FALLBACK_BOOKMAKER]

EVENT_STATUS = "pending"

ET = ZoneInfo("America/New_York")

NOW_UTC = datetime.now(timezone.utc)
NOW_ET = NOW_UTC.astimezone(ET)

TARGET_ET_DATE = NOW_ET.date()
today = TARGET_ET_DATE.strftime("%Y_%m_%d")

PRIMARY_OUTPUT_PATH = Path(f"docs/win/baseball/mlb/odds/{today}.json")
OUTPUT_PATHS = [PRIMARY_OUTPUT_PATH]


def get_json(endpoint, params):
    response = requests.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        print(f"{endpoint} error: {response.status_code}")
        print(response.text)
        raise SystemExit(1)

    return response.json()


def parse_event_utc_datetime(value):
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_event_et_date(value):
    dt = parse_event_utc_datetime(value)

    if dt is None:
        return None

    return dt.astimezone(ET).date()


def event_commence_value(event):
    return event.get("date") or event.get("commence_time")


def is_target_et_date(event):
    event_date = parse_event_et_date(event_commence_value(event))
    return event_date == TARGET_ET_DATE


def has_started(event):
    dt = parse_event_utc_datetime(event_commence_value(event))

    if dt is None:
        return False

    return dt <= NOW_UTC


def fetch_events_for_bookmaker(bookmaker):
    events = []
    skip = 0
    limit = 100

    while True:
        batch = get_json(
            "/events",
            {
                "apiKey": API_KEY,
                "sport": SPORT,
                "league": LEAGUE,
                "status": EVENT_STATUS,
                "bookmaker": bookmaker,
                "limit": limit,
                "skip": skip,
            },
        )

        if not isinstance(batch, list):
            print(f"Unexpected /events response for {bookmaker}:")
            print(json.dumps(batch, indent=2))
            raise SystemExit(1)

        events.extend(batch)

        if len(batch) < limit:
            break

        skip += limit

    return events


def fetch_events():
    by_id = {}
    counts = {}
    skipped_non_target = {}
    skipped_started = {}

    for bookmaker in BOOKMAKERS:
        events = fetch_events_for_bookmaker(bookmaker)
        counts[bookmaker] = len(events)
        skipped_non_target[bookmaker] = 0
        skipped_started[bookmaker] = 0

        for event in events:
            if not is_target_et_date(event):
                skipped_non_target[bookmaker] += 1
                continue

            if has_started(event):
                skipped_started[bookmaker] += 1
                continue

            event_id = event.get("id")
            if event_id is None:
                continue

            event_id = str(event_id)

            if event_id not in by_id:
                by_id[event_id] = event

            if bookmaker == PRIMARY_BOOKMAKER:
                by_id[event_id] = event

    return list(by_id.values()), counts, skipped_non_target, skipped_started


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_odds_for_bookmaker(event_ids, bookmaker):
    odds = []

    for batch_ids in chunks(event_ids, 10):
        batch = get_json(
            "/odds/multi",
            {
                "apiKey": API_KEY,
                "eventIds": ",".join(str(event_id) for event_id in batch_ids),
                "bookmakers": bookmaker,
            },
        )

        if not isinstance(batch, list):
            print(f"Unexpected /odds/multi response for {bookmaker}:")
            print(json.dumps(batch, indent=2))
            raise SystemExit(1)

        odds.extend(batch)

    return odds


def fetch_odds(event_ids):
    by_id = {}

    for bookmaker in BOOKMAKERS:
        bookmaker_odds = fetch_odds_for_bookmaker(event_ids, bookmaker)

        for event in bookmaker_odds:
            if not is_target_et_date(event):
                continue

            if has_started(event):
                continue

            event_id = event.get("id")
            if event_id is None:
                continue

            event_id = str(event_id)

            if event_id not in by_id:
                by_id[event_id] = {}

            by_id[event_id][bookmaker] = event

    return by_id


def to_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def market_key(name):
    normalized = str(name or "").strip().lower()

    if normalized == "ml":
        return "h2h"

    if normalized in {"spread", "spreads", "asian handicap", "handicap"}:
        return "spreads"

    if normalized in {"totals", "total", "over/under", "over under"}:
        return "totals"

    return None


def get_bookmaker_markets(event, bookmaker):
    bookmakers = event.get("bookmakers") or {}

    if isinstance(bookmakers, dict):
        return bookmakers.get(bookmaker, []) or []

    if isinstance(bookmakers, list):
        for item in bookmakers:
            title = str(item.get("title") or item.get("key") or "").strip().lower()
            if title == bookmaker.lower():
                return item.get("markets") or []

    return []


def price_normal_score(*prices):
    valid = [p for p in prices if p is not None]

    if not valid:
        return 999999.0

    return sum(abs(p - 2.0) for p in valid)


def select_main_spread_row(odds_rows):
    candidates = []

    for row in odds_rows:
        point = to_float(row.get("hdp"))
        home_price = to_float(row.get("home"))
        away_price = to_float(row.get("away"))

        if point is None:
            continue

        if home_price is None and away_price is None:
            continue

        candidates.append(
            (
                abs(abs(point) - 1.5),
                price_normal_score(home_price, away_price),
                abs(point),
                row,
            )
        )

    if not candidates:
        return None

    return min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]


def select_main_total_row(odds_rows):
    candidates = []

    for row in odds_rows:
        point = to_float(row.get("hdp"))
        over_price = to_float(row.get("over"))
        under_price = to_float(row.get("under"))

        if point is None:
            continue

        if over_price is None or under_price is None:
            continue

        candidates.append(
            (
                price_normal_score(over_price, under_price),
                abs(point - 8.5),
                point,
                row,
            )
        )

    if not candidates:
        return None

    return min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]


def select_main_spread_outcomes(outcomes):
    by_abs_point = {}

    for outcome in outcomes:
        point = to_float(outcome.get("point"))
        price = to_float(outcome.get("price"))

        if point is None or price is None:
            continue

        key = abs(point)

        if key not in by_abs_point:
            by_abs_point[key] = []

        by_abs_point[key].append(outcome)

    candidates = []

    for abs_point, rows in by_abs_point.items():
        if len(rows) < 2:
            continue

        prices = [to_float(row.get("price")) for row in rows]
        candidates.append(
            (
                abs(abs_point - 1.5),
                price_normal_score(*prices),
                abs_point,
                rows[:2],
            )
        )

    if not candidates:
        return []

    return min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]


def select_main_total_outcomes(outcomes):
    by_point = {}

    for outcome in outcomes:
        point = to_float(outcome.get("point"))
        price = to_float(outcome.get("price"))

        if point is None or price is None:
            continue

        if point not in by_point:
            by_point[point] = []

        by_point[point].append(outcome)

    candidates = []

    for point, rows in by_point.items():
        over_rows = [
            row
            for row in rows
            if str(row.get("name", "")).strip().lower() == "over"
        ]
        under_rows = [
            row
            for row in rows
            if str(row.get("name", "")).strip().lower() == "under"
        ]

        if not over_rows or not under_rows:
            continue

        over = over_rows[0]
        under = under_rows[0]

        over_price = to_float(over.get("price"))
        under_price = to_float(under.get("price"))

        candidates.append(
            (
                price_normal_score(over_price, under_price),
                abs(point - 8.5),
                point,
                [over, under],
            )
        )

    if not candidates:
        return []

    return min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]


def convert_market(event, market):
    key = market_key(market.get("name") or market.get("key"))

    if key not in {"h2h", "spreads", "totals"}:
        return None

    converted = {
        "key": key,
        "last_update": market.get("updatedAt") or market.get("last_update"),
        "outcomes": [],
    }

    odds_rows = market.get("odds") or []

    if odds_rows:
        if key == "h2h":
            for row in odds_rows:
                home_price = to_float(row.get("home"))
                away_price = to_float(row.get("away"))

                if home_price is not None:
                    converted["outcomes"].append(
                        {
                            "name": event.get("home"),
                            "price": home_price,
                        }
                    )

                if away_price is not None:
                    converted["outcomes"].append(
                        {
                            "name": event.get("away"),
                            "price": away_price,
                        }
                    )

                break

        elif key == "spreads":
            row = select_main_spread_row(odds_rows)

            if row:
                point = to_float(row.get("hdp"))
                home_price = to_float(row.get("home"))
                away_price = to_float(row.get("away"))

                if home_price is not None:
                    converted["outcomes"].append(
                        {
                            "name": event.get("home"),
                            "price": home_price,
                            "point": point,
                        }
                    )

                if away_price is not None:
                    converted["outcomes"].append(
                        {
                            "name": event.get("away"),
                            "price": away_price,
                            "point": -point if point is not None else None,
                        }
                    )

        elif key == "totals":
            row = select_main_total_row(odds_rows)

            if row:
                point = to_float(row.get("hdp"))
                over_price = to_float(row.get("over"))
                under_price = to_float(row.get("under"))

                if over_price is not None:
                    converted["outcomes"].append(
                        {
                            "name": "Over",
                            "price": over_price,
                            "point": point,
                        }
                    )

                if under_price is not None:
                    converted["outcomes"].append(
                        {
                            "name": "Under",
                            "price": under_price,
                            "point": point,
                        }
                    )

    else:
        outcomes = market.get("outcomes") or []

        if key == "h2h":
            for outcome in outcomes:
                name = outcome.get("name")
                price = to_float(outcome.get("price"))

                if price is None:
                    continue

                converted["outcomes"].append(
                    {
                        "name": name,
                        "price": price,
                    }
                )

        elif key == "spreads":
            selected = select_main_spread_outcomes(outcomes)

            for outcome in selected:
                name = outcome.get("name")
                price = to_float(outcome.get("price"))
                point = to_float(outcome.get("point"))

                if price is None:
                    continue

                converted["outcomes"].append(
                    {
                        "name": name,
                        "price": price,
                        "point": point,
                    }
                )

        elif key == "totals":
            selected = select_main_total_outcomes(outcomes)

            for outcome in selected:
                name = outcome.get("name")
                price = to_float(outcome.get("price"))
                point = to_float(outcome.get("point"))

                if price is None:
                    continue

                converted["outcomes"].append(
                    {
                        "name": name,
                        "price": price,
                        "point": point,
                    }
                )

    if not converted["outcomes"]:
        return None

    return converted


def converted_markets_for_bookmaker(event, bookmaker):
    bookmaker_markets = get_bookmaker_markets(event, bookmaker)
    converted_markets = []

    for market in bookmaker_markets:
        converted_market = convert_market(event, market)

        if converted_market:
            converted_markets.append(converted_market)

    return converted_markets


def convert_event(event_by_bookmaker, event_fallback):
    selected_bookmaker = None
    selected_event = None
    converted_markets = []

    for bookmaker in BOOKMAKERS:
        candidate_event = event_by_bookmaker.get(bookmaker)

        if not candidate_event:
            continue

        candidate_markets = converted_markets_for_bookmaker(candidate_event, bookmaker)

        if candidate_markets:
            selected_bookmaker = bookmaker
            selected_event = candidate_event
            converted_markets = candidate_markets
            break

    if not selected_event or not converted_markets:
        return None, None

    last_update_values = [
        market.get("last_update")
        for market in converted_markets
        if market.get("last_update")
    ]

    bookmaker = {
        "key": "draftkings",
        "title": PRIMARY_BOOKMAKER,
        "last_update": max(last_update_values) if last_update_values else None,
        "markets": converted_markets,
    }

    return {
        "id": str(selected_event.get("id") or event_fallback.get("id")),
        "sport_key": "baseball_mlb",
        "sport_title": "MLB",
        "commence_time": selected_event.get("date") or event_fallback.get("date"),
        "home_team": selected_event.get("home") or event_fallback.get("home"),
        "away_team": selected_event.get("away") or event_fallback.get("away"),
        "bookmakers": [bookmaker],
    }, selected_bookmaker


def read_json_list(input_path):
    if not input_path.exists():
        return []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in existing file: {input_path}")
        raise SystemExit(1)

    if not isinstance(loaded, list):
        print(f"Expected JSON list in existing file: {input_path}")
        raise SystemExit(1)

    return loaded


def event_id(event):
    value = event.get("id")

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def filter_target_date_events(events):
    filtered = []

    for event in events:
        if isinstance(event, dict) and is_target_et_date(event):
            filtered.append(event)

    return filtered


def sort_events(events):
    return sorted(
        events,
        key=lambda event: (
            event.get("commence_time", ""),
            event.get("home_team", ""),
            event.get("away_team", ""),
            event.get("id", ""),
        ),
    )


def merge_with_existing(existing_events, pulled_events):
    existing_by_id = {}
    order = []

    for event in filter_target_date_events(existing_events):
        eid = event_id(event)

        if eid is None:
            continue

        if eid not in existing_by_id:
            order.append(eid)

        existing_by_id[eid] = event

    added = 0
    updated = 0
    preserved_started = 0

    for event in pulled_events:
        eid = event_id(event)

        if eid is None:
            continue

        if eid in existing_by_id and has_started(existing_by_id[eid]):
            preserved_started += 1
            continue

        if eid in existing_by_id:
            updated += 1
        else:
            order.append(eid)
            added += 1

        existing_by_id[eid] = event

    merged = [existing_by_id[eid] for eid in order]
    return sort_events(merged), updated, added, preserved_started


def write_json(output_path, data):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    temp_path.replace(output_path)


events, event_counts, skipped_non_target_counts, skipped_started_counts = fetch_events()

if not events:
    print("No pending MLB events found with DraftKings or FanDuel odds.")
    data = []
    source_counts = {
        PRIMARY_BOOKMAKER: 0,
        FALLBACK_BOOKMAKER: 0,
    }
else:
    events_by_id = {
        str(event["id"]): event
        for event in events
        if event.get("id") is not None
    }

    event_ids = sorted(events_by_id.keys())
    raw_odds_by_id = fetch_odds(event_ids)

    data = []
    source_counts = {
        PRIMARY_BOOKMAKER: 0,
        FALLBACK_BOOKMAKER: 0,
    }

    for event_id_value in event_ids:
        converted, source_bookmaker = convert_event(
            raw_odds_by_id.get(event_id_value, {}),
            events_by_id[event_id_value],
        )

        if converted:
            data.append(converted)
            source_counts[source_bookmaker] = source_counts.get(source_bookmaker, 0) + 1

data = sort_events(data)

existing_data = read_json_list(PRIMARY_OUTPUT_PATH)
output_data, updated_existing, added_new, preserved_started_existing = merge_with_existing(
    existing_data,
    data,
)

for output_path in OUTPUT_PATHS:
    write_json(output_path, output_data)
    print(f"Saved {output_path}")

print(f"Target ET date: {TARGET_ET_DATE.isoformat()}")
print(f"Target date string: {today}")
print(f"Current UTC time: {NOW_UTC.isoformat()}")
print(f"Current ET time: {NOW_ET.isoformat()}")
print(f"API event status: {EVENT_STATUS}")
print(f"{PRIMARY_BOOKMAKER} events found: {event_counts.get(PRIMARY_BOOKMAKER, 0)}")
print(f"{FALLBACK_BOOKMAKER} events found: {event_counts.get(FALLBACK_BOOKMAKER, 0)}")
print(f"{PRIMARY_BOOKMAKER} non-target ET date events skipped: {skipped_non_target_counts.get(PRIMARY_BOOKMAKER, 0)}")
print(f"{FALLBACK_BOOKMAKER} non-target ET date events skipped: {skipped_non_target_counts.get(FALLBACK_BOOKMAKER, 0)}")
print(f"{PRIMARY_BOOKMAKER} started events skipped: {skipped_started_counts.get(PRIMARY_BOOKMAKER, 0)}")
print(f"{FALLBACK_BOOKMAKER} started events skipped: {skipped_started_counts.get(FALLBACK_BOOKMAKER, 0)}")
print(f"Unique pending target-date events found: {len(events)}")
print(f"Events with converted odds this pull: {len(data)}")
print(f"Events using {PRIMARY_BOOKMAKER}: {source_counts.get(PRIMARY_BOOKMAKER, 0)}")
print(f"Events using {FALLBACK_BOOKMAKER}: {source_counts.get(FALLBACK_BOOKMAKER, 0)}")
print(f"Existing output seed count: {len(existing_data)}")
print(f"Updated existing not-started events: {updated_existing}")
print(f"Added new pending events: {added_new}")
print(f"Preserved existing started events from overwrite: {preserved_started_existing}")
print(f"Final output event count: {len(output_data)}")

#!/usr/bin/env python3
# docs/win/basketball/scripts/00_parsing/wnba_odds_api.py

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API_KEY = os.getenv("API_ODDS")

BASE_URL = "https://api.odds-api.io/v3"
SPORT_SLUG = "basketball"
LEAGUE_SLUG = "usa-wnba"
BOOKMAKERS = ["FanDuel"]

GAME_TZ = ZoneInfo("America/New_York")

OUT_DIR = Path("docs/win/basketball/odds/wnba")
ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
LOG_FILE = ERROR_DIR / "wnba_odds_api.txt"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


class RateLimitError(RuntimeError):
    pass


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def api_get(endpoint: str, params: dict) -> object:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None

            if status_code == 429:
                raise RateLimitError(f"API rate limit hit: 429 Too Many Requests endpoint={endpoint}") from exc

            if status_code in {500, 502, 503, 504}:
                last_exc = exc
                log(
                    f"WARN: API retryable HTTP error "
                    f"endpoint={endpoint} status={status_code} "
                    f"attempt={attempt}/{MAX_RETRIES}: {exc}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SECONDS * attempt)
                    continue

            raise

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            last_exc = exc
            log(
                f"WARN: API transient read/connection failure "
                f"endpoint={endpoint} attempt={attempt}/{MAX_RETRIES}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
                continue

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            log(
                f"WARN: API request failure "
                f"endpoint={endpoint} attempt={attempt}/{MAX_RETRIES}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
                continue

    raise last_exc


def game_date_yyyy_mm_dd(utc_date_text: str) -> str:
    dt_utc = datetime.fromisoformat(utc_date_text.replace("Z", "+00:00"))
    dt_local = dt_utc.astimezone(GAME_TZ)
    return dt_local.strftime("%Y_%m_%d")


def today_yyyy_mm_dd() -> str:
    return datetime.now(GAME_TZ).strftime("%Y_%m_%d")


def get_wnba_events() -> list[dict]:
    data = api_get(
        "events",
        {
            "apiKey": API_KEY,
            "sport": SPORT_SLUG,
            "limit": 1000,
        },
    )

    if not isinstance(data, list):
        raise RuntimeError(f"Expected events response to be a list, got {type(data)}")

    events = []
    for e in data:
        league = e.get("league") or {}
        if league.get("slug") == LEAGUE_SLUG:
            events.append(e)

    return events


def get_event_odds(event_id: int | str, bookmaker: str) -> dict:
    return api_get(
        "odds",
        {
            "apiKey": API_KEY,
            "eventId": event_id,
            "bookmakers": bookmaker,
        },
    )


def main() -> int:
    ensure_dirs()

    LOG_FILE.write_text(
        f"=== wnba_odds_api.py RUN {datetime.now().isoformat(timespec='seconds')} ===\n",
        encoding="utf-8",
    )

    if not API_KEY:
        log("ERROR: Missing required environment secret/API key: API_ODDS")
        return 1

    try:
        today = today_yyyy_mm_dd()

        log("Fetching WNBA events from odds-api.io")
        log(f"Today cutoff date: {today}")

        events = get_wnba_events()
        log(f"Found {len(events)} WNBA events")

        grouped: dict[str, list[dict]] = defaultdict(list)
        skipped_past = 0

        for event in events:
            event_id = event.get("id")
            if not event_id:
                log(f"WARN: Skipping event with missing id: {event}")
                continue

            game_date = game_date_yyyy_mm_dd(event.get("date", ""))

            if game_date < today:
                skipped_past += 1
                log(f"SKIP past event: game_date={game_date}, event_id={event_id}")
                continue

            odds_by_bookmaker = {}

            for bookmaker in BOOKMAKERS:
                log(f"Fetching odds: event_id={event_id}, bookmaker={bookmaker}")
                odds_by_bookmaker[bookmaker] = get_event_odds(event_id, bookmaker)
                time.sleep(0.25)

            grouped[game_date].append(
                {
                    "source": "odds-api.io",
                    "sport_slug": SPORT_SLUG,
                    "league_slug": LEAGUE_SLUG,
                    "game_date": game_date,
                    "event": event,
                    "odds": odds_by_bookmaker,
                }
            )

        written = 0

        for game_date, rows in sorted(grouped.items()):
            out_file = OUT_DIR / f"{game_date}_wnba.json"

            payload = {
                "source": "odds-api.io",
                "sport_slug": SPORT_SLUG,
                "league_slug": LEAGUE_SLUG,
                "bookmakers": BOOKMAKERS,
                "game_date": game_date,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "event_count": len(rows),
                "events": rows,
            }

            out_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            log(f"Wrote {out_file} ({len(rows)} events)")
            written += 1

        log(
            f"SUCCESS: Wrote {written} WNBA odds file(s) | "
            f"skipped_past_events={skipped_past}"
        )
        return 0

    except RateLimitError as exc:
        log(f"ERROR: {exc}")
        log("ERROR: Stopping without writing partial odds files")
        return 1

    except Exception:
        log("ERROR: Script failed")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

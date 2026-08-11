#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_odds.py

import argparse
import csv
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

LEAGUES = {
    "nba": {
        "label": "NBA",
        "espn_slug": "nba",
        "output_dir": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/nba"),
    },
    "wnba": {
        "label": "WNBA",
        "espn_slug": "wnba",
        "output_dir": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/wnba"),
    },
    "ncaam": {
        "label": "NCAAM",
        "espn_slug": "mens-college-basketball",
        "output_dir": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/ncaam"),
    },
}

FIELDNAMES = [
    "sport", "league", "game_date", "game_id", "odds_last_update",
    "game_time", "home_team", "away_team", "home_spread", "away_spread",
    "total", "home_dk_moneyline_american", "away_dk_moneyline_american",
    "home_dk_spread_american", "away_dk_spread_american",
    "dk_total_over_american", "dk_total_under_american",
    "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    "home_dk_spread_decimal", "away_dk_spread_decimal",
    "dk_total_over_decimal", "dk_total_under_decimal",
]

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_odds.txt"

for cfg in LEAGUES.values():
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== basketball_odds RUN {datetime.now().isoformat()} ===\n")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def get_json(url: str) -> dict:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json,text/plain,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            log(f"HTTP attempt {attempt}/{MAX_RETRIES} failed: {url} | {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(f"Failed to fetch ESPN JSON: {url}") from last_error


def nested_get(obj, *keys):
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def clean_number(value):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def clean_text(value):
    return "" if value is None else str(value).strip()


def parse_espn_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def current_run_timestamp() -> str:
    return datetime.now(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")


def scoreboard_url(espn_slug: str, date_yyyymmdd: str) -> str:
    return f"https://cdn.espn.com/core/{espn_slug}/scoreboard?xhr=1&date={date_yyyymmdd}"


def fetch_scoreboard(espn_slug: str, date_yyyymmdd: str) -> list[dict]:
    payload = get_json(scoreboard_url(espn_slug, date_yyyymmdd))
    events = ((payload.get("content") or {}).get("sbData") or {}).get("events") or []
    if not isinstance(events, list):
        raise RuntimeError(f"Unexpected ESPN scoreboard structure for {espn_slug} {date_yyyymmdd}")
    return events


def core_odds_url(espn_slug: str, event_id: str, competition_id: str) -> str:
    return (
        "https://sports.core.api.espn.com/v2/sports/basketball/"
        f"leagues/{espn_slug}/events/{event_id}/competitions/{competition_id}/odds"
    )


def fetch_current_odds(espn_slug: str, event_id: str, competition_id: str) -> dict | None:
    payload = get_json(core_odds_url(espn_slug, event_id, competition_id))
    items = payload.get("items") or []
    if not isinstance(items, list) or not items:
        return None
    for item in items:
        provider = item.get("provider") or {}
        if str(provider.get("name", "")).strip().lower() == "draftkings":
            return item
    return items[0]


def get_competition(event: dict) -> dict | None:
    competitions = event.get("competitions") or []
    return competitions[0] if isinstance(competitions, list) and competitions else None


def get_competitors(competition: dict) -> tuple[dict | None, dict | None]:
    home = None
    away = None
    for competitor in competition.get("competitors") or []:
        home_away = str(competitor.get("homeAway", "")).lower()
        if home_away == "home":
            home = competitor
        elif home_away == "away":
            away = competitor
    return home, away


def team_name(competitor: dict | None) -> str:
    if not isinstance(competitor, dict):
        return ""
    team = competitor.get("team") or {}
    return clean_text(team.get("displayName") or team.get("shortDisplayName") or team.get("name"))


def team_name_from_odds(odds: dict, side: str) -> str:
    return clean_text(nested_get(odds, f"{side}TeamOdds", "team", "displayName"))


def competition_id(competition: dict, event_id: str) -> str:
    value = competition.get("id")
    return str(value) if value not in (None, "") else event_id


def build_row(league_label: str, event: dict, competition: dict, odds: dict, run_timestamp: str) -> dict | None:
    event_id = clean_text(event.get("id"))
    event_date_raw = clean_text(event.get("date"))
    if not event_id or not event_date_raw:
        return None

    event_dt_ny = parse_espn_datetime(event_date_raw).astimezone(NY_TZ)
    game_date = event_dt_ny.strftime("%Y_%m_%d")
    game_time = event_dt_ny.strftime("%I:%M %p")

    home_competitor, away_competitor = get_competitors(competition)
    home_team = team_name(home_competitor) or team_name_from_odds(odds, "home")
    away_team = team_name(away_competitor) or team_name_from_odds(odds, "away")

    home_spread = nested_get(odds, "homeTeamOdds", "current", "pointSpread", "american")
    away_spread = nested_get(odds, "awayTeamOdds", "current", "pointSpread", "american")
    total = nested_get(odds, "current", "total", "american")

    home_ml_american = nested_get(odds, "homeTeamOdds", "current", "moneyLine", "american")
    away_ml_american = nested_get(odds, "awayTeamOdds", "current", "moneyLine", "american")
    home_spread_american = nested_get(odds, "homeTeamOdds", "current", "spread", "american")
    away_spread_american = nested_get(odds, "awayTeamOdds", "current", "spread", "american")
    over_american = nested_get(odds, "current", "over", "american")
    under_american = nested_get(odds, "current", "under", "american")

    home_ml_decimal = nested_get(odds, "homeTeamOdds", "current", "moneyLine", "decimal")
    away_ml_decimal = nested_get(odds, "awayTeamOdds", "current", "moneyLine", "decimal")
    home_spread_decimal = nested_get(odds, "homeTeamOdds", "current", "spread", "decimal")
    away_spread_decimal = nested_get(odds, "awayTeamOdds", "current", "spread", "decimal")
    over_decimal = nested_get(odds, "current", "over", "decimal")
    under_decimal = nested_get(odds, "current", "under", "decimal")

    if home_spread in (None, ""):
        home_spread = odds.get("spread")
    if away_spread in (None, "") and home_spread not in (None, ""):
        try:
            away_spread = -float(home_spread)
        except (TypeError, ValueError):
            pass
    if total in (None, ""):
        total = odds.get("overUnder")
    if home_ml_american in (None, ""):
        home_ml_american = nested_get(odds, "homeTeamOdds", "moneyLine")
    if away_ml_american in (None, ""):
        away_ml_american = nested_get(odds, "awayTeamOdds", "moneyLine")
    if home_spread_american in (None, ""):
        home_spread_american = nested_get(odds, "homeTeamOdds", "spreadOdds")
    if away_spread_american in (None, ""):
        away_spread_american = nested_get(odds, "awayTeamOdds", "spreadOdds")
    if over_american in (None, ""):
        over_american = odds.get("overOdds")
    if under_american in (None, ""):
        under_american = odds.get("underOdds")

    return {
        "sport": "Basketball",
        "league": league_label,
        "game_date": game_date,
        "game_id": event_id,
        "odds_last_update": run_timestamp,
        "game_time": game_time,
        "home_team": home_team,
        "away_team": away_team,
        "home_spread": clean_number(home_spread),
        "away_spread": clean_number(away_spread),
        "total": clean_number(total),
        "home_dk_moneyline_american": clean_number(home_ml_american),
        "away_dk_moneyline_american": clean_number(away_ml_american),
        "home_dk_spread_american": clean_number(home_spread_american),
        "away_dk_spread_american": clean_number(away_spread_american),
        "dk_total_over_american": clean_number(over_american),
        "dk_total_under_american": clean_number(under_american),
        "home_dk_moneyline_decimal": clean_number(home_ml_decimal),
        "away_dk_moneyline_decimal": clean_number(away_ml_decimal),
        "home_dk_spread_decimal": clean_number(home_spread_decimal),
        "away_dk_spread_decimal": clean_number(away_spread_decimal),
        "dk_total_over_decimal": clean_number(over_decimal),
        "dk_total_under_decimal": clean_number(under_decimal),
    }


def row_key(row: dict):
    return row.get("league"), row.get("game_date"), row.get("game_id")


def sort_key(row: dict):
    game_time = row.get("game_time") or ""
    try:
        parsed_time = datetime.strptime(game_time, "%I:%M %p").time()
    except ValueError:
        parsed_time = datetime.max.time()
    return row.get("game_date") or "", parsed_time, row.get("home_team") or "", row.get("away_team") or "", row.get("game_id") or ""


def write_csv(output_dir: Path, league_label: str, game_date: str, rows: list[dict]):
    out_path = output_dir / f"{game_date}_{league_label}_odds.csv"
    unique_rows = []
    seen = set()
    for row in sorted(rows, key=sort_key):
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique_rows)
    return out_path, len(unique_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch current ESPN basketball odds and write standardized cleaned sportsbook CSV files.")
    parser.add_argument("--league", choices=["all", "nba", "wnba", "ncaam"], default="all")
    parser.add_argument("--date", help="Start date in YYYY-MM-DD or YYYYMMDD format. Default: today in America/New_York.")
    parser.add_argument("--days", type=int, default=1, help="Number of calendar days to fetch starting at --date. Default: 1")
    return parser.parse_args()


def resolve_start_date(value: str | None):
    if not value:
        return datetime.now(NY_TZ).date()
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError("--date must be YYYY-MM-DD or YYYYMMDD")


def main():
    args = parse_args()
    if args.days < 1:
        raise ValueError("--days must be at least 1")

    start_date = resolve_start_date(args.date)
    selected_leagues = LEAGUES if args.league == "all" else {args.league: LEAGUES[args.league]}
    run_timestamp = current_run_timestamp()

    files_written = []
    total_events_found = 0
    total_odds_found = 0
    total_rows_written = 0
    total_skipped_no_odds = 0
    total_errors = 0

    try:
        for _, cfg in selected_leagues.items():
            league_label = cfg["label"]
            espn_slug = cfg["espn_slug"]
            output_dir = cfg["output_dir"]
            rows_by_game_date = {}

            for day_offset in range(args.days):
                fetch_date = start_date + timedelta(days=day_offset)
                date_yyyymmdd = fetch_date.strftime("%Y%m%d")
                log(f"FETCH SCOREBOARD: {league_label} {date_yyyymmdd}")

                try:
                    events = fetch_scoreboard(espn_slug, date_yyyymmdd)
                except Exception as exc:
                    total_errors += 1
                    log(f"ERROR fetching scoreboard {league_label} {date_yyyymmdd}: {exc}\n{traceback.format_exc()}")
                    continue

                total_events_found += len(events)
                log(f"SCOREBOARD: {league_label} {date_yyyymmdd} returned {len(events)} events")

                for event in events:
                    event_id = clean_text(event.get("id"))
                    event_name = clean_text(event.get("name"))
                    competition = get_competition(event)
                    if not event_id or competition is None:
                        total_errors += 1
                        log(f"SKIP malformed event: {league_label} {event_name or event_id}")
                        continue

                    comp_id = competition_id(competition, event_id)
                    try:
                        odds = fetch_current_odds(espn_slug, event_id, comp_id)
                    except Exception as exc:
                        total_errors += 1
                        log(f"ERROR fetching odds: {league_label} {event_id} {event_name}: {exc}")
                        continue

                    if odds is None:
                        total_skipped_no_odds += 1
                        log(f"NO ODDS: {league_label} {event_id} {event_name}")
                        continue

                    total_odds_found += 1
                    provider = clean_text(nested_get(odds, "provider", "name"))
                    log(f"ODDS: {league_label} {event_id} {event_name} | provider={provider or 'unknown'}")

                    try:
                        row = build_row(league_label, event, competition, odds, run_timestamp)
                    except Exception as exc:
                        total_errors += 1
                        log(f"ERROR building row: {league_label} {event_id} {event_name}: {exc}\n{traceback.format_exc()}")
                        continue

                    if row is None:
                        total_errors += 1
                        log(f"SKIP row build returned None: {league_label} {event_id}")
                        continue

                    rows_by_game_date.setdefault(row["game_date"], []).append(row)

            for game_date, rows in sorted(rows_by_game_date.items()):
                out_path, row_count = write_csv(output_dir, league_label, game_date, rows)
                files_written.append((str(out_path), row_count))
                total_rows_written += row_count
                log(f"WROTE {out_path} ({row_count} games)")

        log("--- SUMMARY ---")
        log(f"Events found: {total_events_found}")
        log(f"Events with odds: {total_odds_found}")
        log(f"Events skipped without odds: {total_skipped_no_odds}")
        log(f"Rows written: {total_rows_written}")
        log(f"Files written: {len(files_written)}")
        log(f"Errors: {total_errors}")
        for path, count in files_written:
            log(f"FILE: {path} ({count} games)")
        log("STATUS: SUCCESS")

        print("Basketball odds complete.")
        print(f"Events found: {total_events_found}")
        print(f"Events with odds: {total_odds_found}")
        print(f"Rows written: {total_rows_written}")
        print(f"Files written: {len(files_written)}")
        for path, count in files_written:
            print(f"  {path} ({count} games)")
        if total_skipped_no_odds:
            print(f"Events skipped without odds: {total_skipped_no_odds}")
        if total_errors:
            print(f"Errors: {total_errors}")
            print(f"See log: {LOG_FILE}")

    except Exception as exc:
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()

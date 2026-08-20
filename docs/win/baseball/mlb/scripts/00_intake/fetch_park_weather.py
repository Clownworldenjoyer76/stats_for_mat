#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/fetch_park_weather.py
#
# Reads docs/win/baseball/mlb/00_intake/games/{date}_games.csv
# Reads docs/win/baseball/mlb/maps/mlb_venue_ids.csv
# Calls MET Norway Locationforecast /complete for each outdoor game location
# Writes raw selected MET Norway API fields to:
# docs/win/baseball/mlb/data/weather/metno_raw/{date}_metno_raw.csv
#
# Only processes games files dated today or in the future.

import os
import re
import time
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path("docs/win/baseball/mlb")
GAMES_DIR = BASE_DIR / "00_intake/games"
MAPS_DIR = Path("docs/win/baseball/mlb/maps")
WEATHER_DIR = BASE_DIR / "data/weather"
RAW_OUT_DIR = WEATHER_DIR / "metno_raw"
ERROR_DIR = BASE_DIR / "errors/00_intake"

RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "fetch_park_weather.txt"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"

METNO_USER_AGENT = os.environ.get(
    "METNO_USER_AGENT",
    "MatsPicksWeather/1.0 local-dev",
)

REQUEST_TIMEOUT = 20
REQUEST_SLEEP_SECONDS = 1.25

RAW_WEATHER_COLUMNS = [
    "time",
    "air_pressure_at_sea_level",
    "air_temperature",
    "air_temperature_max",
    "air_temperature_min",
    "cloud_area_fraction",
    "cloud_area_fraction_high",
    "cloud_area_fraction_low",
    "cloud_area_fraction_medium",
    "dew_point_temperature",
    "fog_area_fraction",
    "relative_humidity",
    "ultraviolet_index_clear_sky",
    "wind_from_direction",
    "wind_speed",
    "precipitation_amount",
    "symbol_code",
]

OUTPUT_COLUMNS = [
    "gamePk",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "venue_id",
    "venue_name",
    "roof_type",
    "time_zone_id",
    "latitude",
    "longitude",
    "weather_applicable",
    *RAW_WEATHER_COLUMNS,
]

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_log_message(msg: str) -> str:
    sanitized = str(msg)

    sanitized = re.sub(
        r"(?i)\blat\s*=\s*[^,\s|]+",
        "lat=<redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\blon\s*=\s*[^,\s|]+",
        "lon=<redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"\b-?\d{1,3}(?:\.\d+)?\s*,\s*-?\d{1,3}(?:\.\d+)?\b",
        "<redacted-coordinates>",
        sanitized,
    )

    return sanitized


def _log(msg: str, level: str = "INFO") -> None:
    safe_msg = _sanitize_log_message(msg).rstrip()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {safe_msg}\n")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_float(value):
    try:
        s = _clean(value)
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _blank_raw_weather() -> dict:
    return {col: "" for col in RAW_WEATHER_COLUMNS}


def load_venue_map() -> dict:
    path = MAPS_DIR / "mlb_venue_ids.csv"

    df = pd.read_csv(
        path,
        dtype={"venue_id": str},
        encoding="utf-8-sig",
    )

    df["venue_id"] = df["venue_id"].astype(str).str.strip()

    return df.set_index("venue_id").to_dict("index")


def is_weather_applicable(roof_type: str) -> int:
    roof = _clean(roof_type).lower()
    return 0 if roof in ("dome", "indoor") else 1


def parse_game_datetime_utc(game_date: str, game_time: str, time_zone_id: str):
    date_clean = _clean(game_date).replace("_", "-")
    time_clean = _clean(game_time)

    if not date_clean or not time_clean or not time_zone_id:
        return None

    local_naive = datetime.strptime(
        f"{date_clean} {time_clean}",
        "%Y-%m-%d %H:%M:%S",
    )

    local_hour = local_naive.replace(minute=0, second=0, microsecond=0)
    local_aware = local_hour.replace(tzinfo=ZoneInfo(time_zone_id))

    return local_aware.astimezone(UTC)


def parse_metno_time_utc(value: str):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def parse_games_file_date(games_file: Path):
    try:
        date_part = games_file.stem.replace("_games", "")
        return datetime.strptime(date_part, "%Y_%m_%d").date()
    except Exception:
        return None


def get_today_and_future_games_files():
    today = datetime.now().date()

    games_files = []

    for games_file in sorted(GAMES_DIR.glob("*_games.csv")):
        file_date = parse_games_file_date(games_file)

        if file_date is None:
            _log(
                f"{games_file.name} | "
                f"could not parse date from filename — skipping",
                "WARN",
            )
            continue

        if file_date >= today:
            games_files.append(games_file)

    return games_files


def call_metno(lat: float, lon: float) -> dict:
    headers = {
        "User-Agent": METNO_USER_AGENT,
    }

    params = {
        "lat": f"{lat:.4f}",
        "lon": f"{lon:.4f}",
    }

    response = requests.get(
        METNO_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    return response.json()


def select_timeseries_item(payload: dict, target_utc):
    timeseries = (
        payload
        .get("properties", {})
        .get("timeseries", [])
    )

    if not timeseries or target_utc is None:
        return None

    exact = None

    for item in timeseries:
        item_time = parse_metno_time_utc(_clean(item.get("time")))
        if item_time == target_utc:
            exact = item
            break

    if exact is not None:
        return exact

    valid_items = []

    for item in timeseries:
        item_time = parse_metno_time_utc(_clean(item.get("time")))
        if item_time is not None:
            valid_items.append((item, item_time))

    if not valid_items:
        return None

    return min(
        valid_items,
        key=lambda pair: abs(pair[1] - target_utc),
    )[0]


def extract_raw_fields(item: dict) -> dict:
    raw = _blank_raw_weather()

    if not item:
        return raw

    raw["time"] = _clean(item.get("time"))

    data = (
        item.get("data", {})
        if isinstance(item.get("data"), dict)
        else {}
    )

    instant = (
        data.get("instant", {})
        if isinstance(data.get("instant"), dict)
        else {}
    )
    instant_details = (
        instant.get("details", {})
        if isinstance(instant.get("details"), dict)
        else {}
    )

    for field in [
        "air_pressure_at_sea_level",
        "air_temperature",
        "cloud_area_fraction",
        "cloud_area_fraction_high",
        "cloud_area_fraction_low",
        "cloud_area_fraction_medium",
        "dew_point_temperature",
        "fog_area_fraction",
        "relative_humidity",
        "ultraviolet_index_clear_sky",
        "wind_from_direction",
        "wind_speed",
    ]:
        raw[field] = instant_details.get(field, "")

    next_1_hours = (
        data.get("next_1_hours", {})
        if isinstance(data.get("next_1_hours"), dict)
        else {}
    )

    next_1_summary = (
        next_1_hours.get("summary", {})
        if isinstance(next_1_hours.get("summary"), dict)
        else {}
    )

    next_1_details = (
        next_1_hours.get("details", {})
        if isinstance(next_1_hours.get("details"), dict)
        else {}
    )

    raw["precipitation_amount"] = next_1_details.get(
        "precipitation_amount",
        "",
    )
    raw["symbol_code"] = next_1_summary.get("symbol_code", "")

    next_6_hours = (
        data.get("next_6_hours", {})
        if isinstance(data.get("next_6_hours"), dict)
        else {}
    )

    next_6_details = (
        next_6_hours.get("details", {})
        if isinstance(next_6_hours.get("details"), dict)
        else {}
    )

    raw["air_temperature_max"] = next_6_details.get(
        "air_temperature_max",
        "",
    )
    raw["air_temperature_min"] = next_6_details.get(
        "air_temperature_min",
        "",
    )

    return raw


def build_output_row(
    game_row: dict,
    venue_id: str,
    vinfo: dict,
    weather_applicable: int,
    raw_weather: dict,
) -> dict:
    row = {
        "gamePk": _clean(game_row.get("gamePk")),
        "game_id": _clean(game_row.get("game_id")),
        "game_date": _clean(game_row.get("game_date")),
        "game_time": _clean(game_row.get("game_time")),
        "home_team": _clean(game_row.get("home_team")),
        "away_team": _clean(game_row.get("away_team")),
        "venue_id": venue_id,
        "venue_name": _clean(vinfo.get("venue_name")),
        "roof_type": _clean(vinfo.get("roof_type")),
        "time_zone_id": _clean(vinfo.get("time_zone_id")),
        "latitude": _clean(vinfo.get("latitude")),
        "longitude": _clean(vinfo.get("longitude")),
        "weather_applicable": weather_applicable,
    }

    for col in RAW_WEATHER_COLUMNS:
        row[col] = raw_weather.get(col, "")

    return row


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(
    date_str: str,
    venue_map: dict,
    summary: dict,
) -> None:
    games_path = GAMES_DIR / f"{date_str}_games.csv"
    out_path = RAW_OUT_DIR / f"{date_str}_metno_raw.csv"

    if not games_path.exists():
        _log(
            f"{date_str} | games file not found — skipping",
            "WARN",
        )
        summary["skipped"] += 1
        return

    games_df = pd.read_csv(games_path, dtype=str)
    rows = []

    forecast_cache = {}

    _log(f"--- {date_str} | games={len(games_df)}")

    for _, game_row in games_df.iterrows():
        game = game_row.to_dict()

        game_pk = _clean(game.get("gamePk"))
        venue_id = _clean(game.get("venue_id"))

        vinfo = venue_map.get(venue_id, {})

        roof_type = _clean(vinfo.get("roof_type"))
        time_zone_id = _clean(vinfo.get("time_zone_id"))
        lat = _to_float(vinfo.get("latitude"))
        lon = _to_float(vinfo.get("longitude"))

        weather_applicable = is_weather_applicable(roof_type)
        raw_weather = _blank_raw_weather()

        if (
            weather_applicable
            and lat is not None
            and lon is not None
            and time_zone_id
        ):
            try:
                target_utc = parse_game_datetime_utc(
                    _clean(game.get("game_date")),
                    _clean(game.get("game_time")),
                    time_zone_id,
                )

                coord_key = (
                    f"{round(lat, 4):.4f}_"
                    f"{round(lon, 4):.4f}"
                )

                if coord_key in forecast_cache:
                    payload = forecast_cache[coord_key]
                    _log(
                        f"{game_pk} | "
                        f"reused MET Norway data "
                        f"venue={venue_id}"
                    )
                else:
                    payload = call_metno(
                        round(lat, 4),
                        round(lon, 4),
                    )
                    forecast_cache[coord_key] = payload
                    summary["api_calls"] += 1
                    _log(
                        f"{game_pk} | "
                        f"fetched MET Norway data "
                        f"venue={venue_id}"
                    )
                    time.sleep(REQUEST_SLEEP_SECONDS)

                selected_item = select_timeseries_item(
                    payload,
                    target_utc,
                )
                raw_weather = extract_raw_fields(
                    selected_item
                )

            except requests.HTTPError as e:
                status = getattr(
                    e.response,
                    "status_code",
                    "",
                )
                _log(
                    f"{game_pk} | "
                    f"MET Norway HTTP error "
                    f"status={status}",
                    "ERROR",
                )
                summary["errors"] += 1
            except Exception:
                _log(
                    f"{game_pk} | "
                    f"MET Norway fetch/parse failed",
                    "ERROR",
                )
                summary["errors"] += 1

        rows.append(
            build_output_row(
                game_row=game,
                venue_id=venue_id,
                vinfo=vinfo,
                weather_applicable=weather_applicable,
                raw_weather=raw_weather,
            )
        )

    pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    ).to_csv(
        out_path,
        index=False,
    )

    _log(f"WROTE: {out_path}")
    summary["files_written"] += 1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"=== fetch_park_weather RUN "
            f"{_now()} ===\n"
        )

    summary = {
        "files_written": 0,
        "api_calls": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        venue_map = load_venue_map()
        _log(
            f"Venue map loaded: "
            f"{len(venue_map)} entries"
        )

        games_files = get_today_and_future_games_files()
        _log(
            f"Today/future games files found: "
            f"{len(games_files)}"
        )

        for games_file in games_files:
            date_str = games_file.stem.replace(
                "_games",
                "",
            )
            process_date(
                date_str,
                venue_map,
                summary,
            )

    except Exception as e:
        _log(
            f"FATAL: {type(e).__name__}",
            "ERROR",
        )
        summary["errors"] += 1

    status = (
        "SUCCESS"
        if summary["errors"] == 0
        else "COMPLETED WITH ERRORS"
    )

    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        (
            f"  files_written  : "
            f"{summary['files_written']}"
        ),
        (
            f"  api_calls      : "
            f"{summary['api_calls']}"
        ),
        (
            f"  skipped        : "
            f"{summary['skipped']}"
        ),
        (
            f"  errors         : "
            f"{summary['errors']}"
        ),
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"fetch_park_weather complete. "
        f"{summary['files_written']} files written, "
        f"{summary['api_calls']} API calls. "
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()
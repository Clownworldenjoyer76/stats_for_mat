#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/fetch_weather.py
#
# Runs ONCE per day after build_games_list.py.
# Reads {date}_games.csv for venue/game_time info,
# calls weatherapi.com once per unique venue+hour combo,
# and writes results to data/weather/{date}_weather.csv.
#
# enrich_game_context.py reads from that cache — it never calls the API.

import os
import re
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR    = Path("docs/win/baseball")
GAMES_DIR   = BASE_DIR / "00_intake/games"
MAPS_DIR    = BASE_DIR / "maps"
WEATHER_DIR = BASE_DIR / "data/weather"
ERROR_DIR   = BASE_DIR / "errors/00_intake"

WEATHER_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "fetch_weather.txt"

WEATHER_API_KEY = os.environ.get("WEATHER_API", "")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def _now():
    return datetime.now(UTC).isoformat()


def _sanitize_log_message(msg: str) -> str:
    sanitized = str(msg)
    # Redact key-value coordinate fields that may contain private location data.
    sanitized = re.sub(r"(?i)\blat\s*=\s*[^,\s|]+", "lat=<redacted>", sanitized)
    sanitized = re.sub(r"(?i)\blon\s*=\s*[^,\s|]+", "lon=<redacted>", sanitized)
    # Redact raw coordinate pairs (e.g., "40.7128,-74.0060").
    sanitized = re.sub(
        r"\b-?\d{1,3}(?:\.\d+)?\s*,\s*-?\d{1,3}(?:\.\d+)?\b",
        "<redacted-coordinates>",
        sanitized,
    )
    return sanitized


def _log(msg: str, level: str = "INFO"):
    safe_msg = _sanitize_log_message(msg).rstrip()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {safe_msg}\n")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_venue_map() -> dict:
    df = pd.read_csv(
        MAPS_DIR / "mlb_venue_ids.csv",
        dtype={"venue_id": str},
        encoding="utf-8-sig"
    )
    df["venue_id"] = df["venue_id"].str.strip()
    return df.set_index("venue_id").to_dict("index")


def call_weather_api(lat: str, lon: str, local_date: str, local_hour: str) -> dict:
    """
    Call weatherapi.com over HTTPS.
    API key passed via params dict — never embedded in a string or logged.
    Matches returned hour by time string, not by array index.
    """
    if not WEATHER_API_KEY:
        _log("WEATHER_API not set — cannot fetch weather", "WARN")
        return {}
    try:
        r = requests.get(
            "https://api.weatherapi.com/v1/forecast.json",
            params={
                "key":  WEATHER_API_KEY,
                "q":    f"{lat},{lon}",
                "days": 1,
                "dt":   local_date,
                "hour": local_hour,
            },
            timeout=10,
        )
        r.raise_for_status()
        hours = r.json()["forecast"]["forecastday"][0]["hour"]
        # Match by time string (e.g. "2026-04-19 21:00") rather than assuming [0] is correct
        target_suffix = f"{int(local_hour):02d}:00"
        hour_data = next(
            (h for h in hours if str(h.get("time", "")).endswith(target_suffix)),
            {}
        )
        if not hour_data:
            _log(f"Weather API returned no matching hour for hr={local_hour} date={local_date}", "WARN")
            return {}
        return {
            "weather_time":   hour_data.get("time"),
            "temp_f":         hour_data.get("temp_f"),
            "wind_mph":       hour_data.get("wind_mph"),
            "wind_dir":       hour_data.get("wind_dir"),
            "gust_mph":       hour_data.get("gust_mph"),
            "precip_in":      hour_data.get("precip_in"),
            "humidity":       hour_data.get("humidity"),
            "chance_of_rain": hour_data.get("chance_of_rain"),
            "will_it_rain":   hour_data.get("will_it_rain"),
        }
    except requests.HTTPError as e:
        _log(f"Weather API HTTP error: status={e.response.status_code} date={local_date} hr={local_hour}", "ERROR")
        return {}
    except Exception:
        _log(f"Weather API failed: date={local_date} hr={local_hour}", "ERROR")
        return {}


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(date_str: str, venue_map: dict, summary: dict) -> None:
    games_path   = GAMES_DIR / f"{date_str}_games.csv"
    weather_path = WEATHER_DIR / f"{date_str}_weather.csv"

    if not games_path.exists():
        _log(f"{date_str} | games file not found — skipping", "WARN")
        summary["skipped"] += 1
        return

    games_df    = pd.read_csv(games_path, dtype=str)
    total_games = len(games_df)
    game_pks    = {str(pk).strip() for pk in games_df["gamePk"]}
    game_order  = {str(pk).strip(): i for i, pk in enumerate(games_df["gamePk"])}

    # Load existing weather rows keyed by gamePk
    existing_by_pk = {}
    if weather_path.exists():
        existing_df = pd.read_csv(weather_path, dtype=str)
        existing_by_pk = {
            str(r["gamePk"]).strip(): r
            for r in existing_df.to_dict("records")
        }

    # Check coverage by set comparison, not count
    missing_pks = game_pks - set(existing_by_pk.keys())
    if not missing_pks:
        _log(f"{date_str} | all {total_games} games already have weather — skipping")
        summary["skipped"] += 1
        return

    if existing_by_pk:
        _log(f"{date_str} | {len(existing_by_pk)}/{total_games} games have weather — fetching {len(missing_pks)} missing")

    _log(f"--- {date_str} | {total_games} games total")

    seen_keys = {}  # cache_key -> weather dict (within this run)

    for _, row in games_df.iterrows():
        game_pk = str(row.get("gamePk", "")).strip()

        if game_pk not in missing_pks:
            continue

        venue_id  = str(row.get("venue_id", "")).strip()
        game_time = row.get("game_time", "")
        game_date = row.get("game_date", "")

        vinfo     = venue_map.get(venue_id, {})
        roof_type = str(vinfo.get("roof_type", "")).strip().lower()
        lat       = str(vinfo.get("latitude", "")).strip()
        lon       = str(vinfo.get("longitude", "")).strip()
        wind_out  = str(vinfo.get("wind_out_direction", "")).strip()

        weather_applicable = 0 if roof_type in ("dome", "indoor") else 1
        weather            = {}
        wind_blowing_out   = None

        if weather_applicable and lat and lon and game_time:
            try:
                date_clean = game_date.replace("_", "-")
                dt_local   = datetime.strptime(f"{date_clean} {game_time}", "%Y-%m-%d %H:%M:%S")
                local_date = dt_local.strftime("%Y-%m-%d")
                local_hour = str(dt_local.hour)
            except Exception:
                _log(f"  {game_pk} time parse failed", "WARN")
                local_date = local_hour = None

            if local_date and local_hour:
                cache_key = f"{lat}_{lon}_{local_date}_{local_hour}"

                if cache_key in seen_keys:
                    weather = seen_keys[cache_key]
                    _log(f"  {game_pk} | reused weather venue={venue_id} hr={local_hour}")
                else:
                    weather = call_weather_api(lat, lon, local_date, local_hour)
                    if weather:
                        seen_keys[cache_key] = weather
                        summary["api_calls"] += 1
                        _log(f"  {game_pk} | fetched weather venue={venue_id} hr={local_hour}")
                    else:
                        _log(f"  {game_pk} | weather fetch failed venue={venue_id} hr={local_hour}", "WARN")

                if weather.get("wind_dir") and wind_out and wind_out not in ("NULL", ""):
                    wind_blowing_out = 1 if weather["wind_dir"].strip().upper() == wind_out.strip().upper() else 0

        existing_by_pk[game_pk] = {
            "gamePk":             game_pk,
            "venue_id":           venue_id,
            "weather_applicable": weather_applicable,
            "weather_time":       weather.get("weather_time"),
            "temp_f":             weather.get("temp_f"),
            "wind_mph":           weather.get("wind_mph"),
            "wind_dir":           weather.get("wind_dir"),
            "gust_mph":           weather.get("gust_mph"),
            "precip_in":          weather.get("precip_in"),
            "humidity":           weather.get("humidity"),
            "chance_of_rain":     weather.get("chance_of_rain"),
            "will_it_rain":       weather.get("will_it_rain"),
            "wind_blowing_out":   wind_blowing_out,
        }

    # Write in original game order from games_df
    ordered = sorted(
        existing_by_pk.values(),
        key=lambda r: game_order.get(str(r["gamePk"]).strip(), 999)
    )
    pd.DataFrame(ordered).to_csv(weather_path, index=False)
    _log(f"  WROTE: {weather_path.name} ({len(ordered)} rows)")
    summary["files_written"] += 1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== fetch_weather RUN {_now()} ===\n")

    summary = {
        "files_written": 0,
        "api_calls":     0,
        "skipped":       0,
        "errors":        0,
    }

    try:
        venue_map = load_venue_map()
        _log(f"Venue map loaded: {len(venue_map)} entries")

        games_files = sorted(GAMES_DIR.glob("*_games.csv"))
        _log(f"Games files found: {len(games_files)}")

        for gf in games_files:
            date_str = gf.stem.replace("_games", "")
            try:
                process_date(date_str, venue_map, summary)
            except Exception:
                _log(f"{date_str} FAILED", "ERROR")
                summary["errors"] += 1

    except Exception:
        _log("FATAL", "ERROR")
        summary["errors"] += 1

    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_written  : {summary['files_written']}",
        f"  api_calls      : {summary['api_calls']}",
        f"  skipped        : {summary['skipped']}",
        f"  errors         : {summary['errors']}",
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"fetch_weather complete. {summary['files_written']} files written, {summary['api_calls']} API calls. Status: {status}")


if __name__ == "__main__":
    main()

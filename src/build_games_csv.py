
#!/usr/bin/env python3
"""
Build the official games.csv at data/processed/games.csv

Inputs:
- data/raw/week_matchups_odds.csv
    Required columns: DAY, DATE, TIME, TEAM_A, TEAM_B, ODDS, weatherapi_datetime
- data/team_stadiums.csv
    Required columns: TEAM, STADIUM, LATITUDE, LONGITUDE

Environment:
- WEATHERAPI_KEY must be set to a valid WeatherAPI.com API key.

Notes:
- Day is converted from abbreviations (e.g., 'SUN' -> 'Sunday').
- Date is converted from like 'Oct 26' -> 'October 26'. Year is inferred from the weatherapi_datetime when possible,
  otherwise current year is used for parsing only (year is dropped in the final output).

Output columns (exact order):
Day, Date, Time, away_team, home_team, Stadium, temp_f, wind_mph, wind_dir, precip_in, chance_of_rain, Odds
"""
import os
import sys
import csv
import math
import json
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import requests
from dateutil import parser as dateparser
from calendar import day_name

# ---------- Config ----------
RAW_MATCHUPS = Path("data/raw/week_matchups_odds.csv")
TEAM_STADIUMS = Path("data/team_stadiums.csv")
OUTPUT = Path("data/processed/games.csv")

WEATHER_API_KEY = os.getenv("WEATHERAPI_KEY")
WEATHER_BASE = "https://api.weatherapi.com/v1/forecast.json"
# We'll request up to 3 days ahead/behind by selecting the date part of weatherapi_datetime.
# WeatherAPI forecast endpoint includes hourly data in forecastday[].hour[]

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------- Utilities ----------
DAY_ABBR_TO_FULL = {
    'SUN': 'Sunday', 'MON': 'Monday', 'TUE': 'Tuesday', 'WED': 'Wednesday',
    'THU': 'Thursday', 'FRI': 'Friday', 'SAT': 'Saturday'
}

MONTH_ABBR_TO_FULL = {
    'Jan': 'January','Feb':'February','Mar':'March','Apr':'April','May':'May','Jun':'June',
    'Jul':'July','Aug':'August','Sep':'September','Sept':'September','Oct':'October','Nov':'November','Dec':'December'
}

def to_full_day(day_abbr: str) -> str:
    if pd.isna(day_abbr):
        return ""
    d = day_abbr.strip().upper()
    return DAY_ABBR_TO_FULL.get(d, d.title())

def to_full_month_date(date_str: str, year_hint: Optional[int]) -> str:
    """
    Convert 'Oct 26' -> 'October 26'. If an unexpected format is present,
    return the input unchanged (but stripped).
    """
    if pd.isna(date_str) or not str(date_str).strip():
        return ""
    s = str(date_str).strip()
    # Split expected format "Oct 26"
    parts = s.replace(",", "").split()
    if len(parts) >= 2 and parts[0] in MONTH_ABBR_TO_FULL:
        try:
            month_full = MONTH_ABBR_TO_FULL[parts[0]]
            day_num = str(int(parts[1]))
            return f"{month_full} {day_num}"
        except Exception:
            pass
    # Fallback: try to parse with dateutil (adding a year if we have one)
    try:
        if year_hint is not None:
            tmp = f"{s} {year_hint}"
        else:
            tmp = s
        dt = dateparser.parse(tmp, fuzzy=True, default=None)
        if dt:
            month_full = dt.strftime("%B")
            day_num = dt.day
            return f"{month_full} {day_num}"
    except Exception:
        pass
    return s

def nearest_hour_record(hours, target_epoch):
    """
    Given a list of hourly objects (each with 'time_epoch'), return the one closest to target_epoch.
    """
    if not hours:
        return None
    best = None
    best_diff = 10**18
    for h in hours:
        te = h.get("time_epoch")
        if te is None:
            continue
        diff = abs(te - target_epoch)
        if diff < best_diff:
            best_diff = diff
            best = h
    return best

def fetch_hourly_weather(lat: float, lon: float, when_iso: str, session: requests.Session, cache: Dict[Tuple[str, str], dict]) -> dict:
    """
    Fetch hourly forecast from WeatherAPI and return the hour record nearest to `when_iso`.
    Caches by (q, date) to minimize requests. Returns a dict with keys used downstream.
    """
    if not WEATHER_API_KEY:
        raise RuntimeError("WEATHERAPI_KEY is not set in the environment. Please export it before running.")

    when_dt = dateparser.parse(when_iso)
    if when_dt is None:
        raise ValueError(f"Could not parse weatherapi_datetime value: {when_iso}")
    date_key = when_dt.strftime("%Y-%m-%d")
    q = f"{lat:.4f},{lon:.4f}"

    cache_key = (q, date_key)
    if cache_key not in cache:
        params = {
            "key": WEATHER_API_KEY,
            "q": q,
            "dt": date_key,           # request the specific date
            "aqi": "no",
            "alerts": "no"
        }
        # WeatherAPI's forecast.json supports a 'days' param for multiple days; with dt we pin a specific date.
        # We'll rely on hourly data within that day to select the closest hour.
        logging.info(f"Requesting WeatherAPI for q={q} dt={date_key}")
        resp = session.get(WEATHER_BASE, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"WeatherAPI error {resp.status_code}: {resp.text}")
        data = resp.json()
        cache[cache_key] = data
    else:
        data = cache[cache_key]

    # Extract hours list
    forecastdays = data.get("forecast", {}).get("forecastday", [])
    hours = []
    for fd in forecastdays:
        hours.extend(fd.get("hour", []))

    target_epoch = int(when_dt.timestamp())
    h = nearest_hour_record(hours, target_epoch)

    # Prepare a normalized result, with safe defaults if missing
    if not h:
        # return safe defaults if no hourly match
        return {
            "temp_f": None,
            "wind_mph": None,
            "wind_dir": None,
            "precip_in": None,
            "chance_of_rain": None
        }

    return {
        "temp_f": h.get("temp_f"),
        "wind_mph": h.get("wind_mph"),
        "wind_dir": h.get("wind_dir"),
        "precip_in": h.get("precip_in"),
        # hour objects typically have chance_of_rain (int 0-100); fallback to daily "day" chance_of_rain if missing
        "chance_of_rain": h.get("chance_of_rain")
    }

def main() -> int:
    # Validate inputs exist
    for p in [RAW_MATCHUPS, TEAM_STADIUMS]:
        if not p.exists():
            logging.error(f"Missing required input: {p}")
            return 2

    # Read inputs
    matchups = pd.read_csv(RAW_MATCHUPS)
    stadiums = pd.read_csv(TEAM_STADIUMS)

    # Validate required columns
    required_match_cols = {"DAY", "DATE", "TIME", "TEAM_A", "TEAM_B", "ODDS", "weatherapi_datetime"}
    missing_match = required_match_cols - set(matchups.columns)
    if missing_match:
        logging.error(f"Missing required columns in {RAW_MATCHUPS}: {sorted(missing_match)}")
        return 2

    required_stad_cols = {"TEAM", "STADIUM", "LATITUDE", "LONGITUDE"}
    missing_stad = required_stad_cols - set(stadiums.columns)
    if missing_stad:
        logging.error(f"Missing required columns in {TEAM_STADIUMS}: {sorted(missing_stad)}")
        return 2

    # Prepare stadium lookup on TEAM
    stad_lookup = stadiums.set_index("TEAM")[["STADIUM", "LATITUDE", "LONGITUDE"]]

    # Merge stadium info onto matchups, matching home team (TEAM_B)
    df = matchups.merge(stad_lookup, left_on="TEAM_B", right_index=True, how="left")

    # For year hint in date conversion, try to extract from weatherapi_datetime
    def extract_year(v):
        try:
            return dateparser.parse(str(v)).year
        except Exception:
            return None

    year_hints = df["weatherapi_datetime"].map(extract_year)

    # Convert Day and Date
    df["Day"] = df["DAY"].map(to_full_day)
    df["Date"] = [to_full_month_date(d, yh) for d, yh in zip(df["DATE"], year_hints)]

    # Rename teams
    df["away_team"] = df["TEAM_A"]
    df["home_team"] = df["TEAM_B"]

    # Stadium
    df["Stadium"] = df["STADIUM"]

    # Weather columns placeholders
    df["temp_f"] = None
    df["wind_mph"] = None
    df["wind_dir"] = None
    df["precip_in"] = None
    df["chance_of_rain"] = None

    # Fetch weather for each row using WeatherAPI hourly, nearest to weatherapi_datetime
    session = requests.Session()
    cache: Dict[Tuple[str, str], dict] = {}

    for idx, row in df.iterrows():
        lat = row.get("LATITUDE")
        lon = row.get("LONGITUDE")
        when_iso = row.get("weatherapi_datetime")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(when_iso):
            logging.warning(f"Skipping weather fetch for row {idx} due to missing lat/lon or time.")
            continue
        try:
            w = fetch_hourly_weather(float(lat), float(lon), str(when_iso), session, cache)
            df.at[idx, "temp_f"] = w.get("temp_f")
            df.at[idx, "wind_mph"] = w.get("wind_mph")
            df.at[idx, "wind_dir"] = w.get("wind_dir")
            df.at[idx, "precip_in"] = w.get("precip_in")
            df.at[idx, "chance_of_rain"] = w.get("chance_of_rain")
        except Exception as e:
            logging.error(f"Weather fetch failed for row {idx} ({row.get('TEAM_B')} @ {row.get('STADIUM')}): {e}")

    # Odds
    df["Odds"] = df["ODDS"]

    # Time (pass through)
    df["Time"] = df["TIME"]

    # Select columns in EXACT order
    out_cols = [
        "Day",
        "Date",
        "Time",
        "away_team",
        "home_team",
        "Stadium",
        "temp_f",
        "wind_mph",
        "wind_dir",
        "precip_in",
        "chance_of_rain",
        "Odds",
    ]

    out_df = df[out_cols].copy()

    # Ensure output directory
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT, index=False)
    logging.info(f"Wrote {OUTPUT.resolve()} with {len(out_df)} rows.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Build the official games.csv at data/processed/games.csv

Implements:
- Weather lookup via WeatherAPI (hourly; nearest to weatherapi_datetime)
- Stadium lookup (TEAM_B -> TEAM in team_stadiums.csv)
- Odds, favorite, total, game_id from data/raw/odds.csv using only rows where book == "CONSENSUS"
- Matching odds by TEAM_A (week_matchups_odds.csv) -> away_team (odds.csv)
- Odds = lower (more negative) of spread_home vs spread_away (numeric compare)
- favorite = home_team if chosen from spread_home, else away_team
- Output columns (EXACT order):
  Day, Date, Time, away_team, home_team, Stadium, temp_f, wind_mph, wind_dir, precip_in, chance_of_rain, Odds, favorite, total, game_id
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import requests
from dateutil import parser as dateparser

# ---------- Paths ----------
RAW_MATCHUPS = Path("data/raw/week_matchups_odds.csv")
TEAM_STADIUMS = Path("data/team_stadiums.csv")
RAW_ODDS = Path("data/raw/odds.csv")
OUTPUT = Path("data/processed/games.csv")

# ---------- WeatherAPI ----------
WEATHER_API_KEY = os.getenv("WEATHERAPI_KEY")
WEATHER_BASE = "https://api.weatherapi.com/v1/forecast.json"

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------- Helpers ----------
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
    d = str(day_abbr).strip().upper()
    return DAY_ABBR_TO_FULL.get(d, d.title())

def to_full_month_date(date_str: str, year_hint: Optional[int]) -> str:
    """Convert 'Oct 26' -> 'October 26'; fall back gracefully on unknown formats."""
    if pd.isna(date_str) or not str(date_str).strip():
        return ""
    s = str(date_str).strip().replace(",", "")
    parts = s.split()
    if len(parts) >= 2 and parts[0] in MONTH_ABBR_TO_FULL:
        try:
            month_full = MONTH_ABBR_TO_FULL[parts[0]]
            day_num = str(int(parts[1]))
            return f"{month_full} {day_num}"
        except Exception:
            pass
    try:
        tmp = f"{s} {year_hint}" if year_hint is not None else s
        dt = dateparser.parse(tmp, fuzzy=True, default=None)
        if dt:
            return f"{dt.strftime('%B')} {dt.day}"
    except Exception:
        pass
    return s

def nearest_hour_record(hours, target_epoch):
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
    if not WEATHER_API_KEY:
        raise RuntimeError("WEATHERAPI_KEY is not set in the environment. Please export it before running.")
    when_dt = dateparser.parse(str(when_iso))
    if when_dt is None:
        raise ValueError(f"Could not parse weatherapi_datetime value: {when_iso}")
    date_key = when_dt.strftime("%Y-%m-%d")
    q = f"{lat:.4f},{lon:.4f}"
    cache_key = (q, date_key)

    if cache_key not in cache:
        params = {"key": WEATHER_API_KEY, "q": q, "dt": date_key, "aqi": "no", "alerts": "no"}
        logging.info(f"Requesting WeatherAPI for q={q} dt={date_key}")
        resp = session.get(WEATHER_BASE, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"WeatherAPI error {resp.status_code}: {resp.text}")
        cache[cache_key] = resp.json()

    data = cache[cache_key]
    forecastdays = data.get("forecast", {}).get("forecastday", [])
    hours = []
    for fd in forecastdays:
        hours.extend(fd.get("hour", []))

    h = nearest_hour_record(hours, int(when_dt.timestamp()))
    if not h:
        return {"temp_f": None, "wind_mph": None, "wind_dir": None, "precip_in": None, "chance_of_rain": None}
    return {
        "temp_f": h.get("temp_f"),
        "wind_mph": h.get("wind_mph"),
        "wind_dir": h.get("wind_dir"),
        "precip_in": h.get("precip_in"),
        "chance_of_rain": h.get("chance_of_rain"),
    }

def _to_float(x):
    """Parse numbers robustly from strings like '+7.5', ' -3 ', None -> None."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(str(x).replace("+", "").strip())
    except Exception:
        return None

def _trim_num(x):
    """Render numbers nicely as strings (drop trailing .0)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(x)

def main() -> int:
    # Validate inputs exist
    for p in [RAW_MATCHUPS, TEAM_STADIUMS, RAW_ODDS]:
        if not p.exists():
            logging.error(f"Missing required input: {p}")
            return 2

    # Load data
    matchups = pd.read_csv(RAW_MATCHUPS)
    stadiums = pd.read_csv(TEAM_STADIUMS)
    odds = pd.read_csv(RAW_ODDS)

    # Validate columns
    need_match = {"DAY", "DATE", "TIME", "TEAM_A", "TEAM_B", "weatherapi_datetime"}
    need_stad = {"TEAM", "STADIUM", "LATITUDE", "LONGITUDE"}
    need_odds = {"away_team", "home_team", "book", "spread_home", "spread_away", "total", "game_id"}

    missing = []
    if need_match - set(matchups.columns):
        missing.append(f"{RAW_MATCHUPS}: {sorted(need_match - set(matchups.columns))}")
    if need_stad - set(stadiums.columns):
        missing.append(f"{TEAM_STADIUMS}: {sorted(need_stad - set(stadiums.columns))}")
    if need_odds - set(odds.columns):
        missing.append(f"{RAW_ODDS}: {sorted(need_odds - set(odds.columns))}")
    if missing:
        for msg in missing:
            logging.error("Missing columns -> %s", msg)
        return 2

    # Stadium merge using home team (TEAM_B)
    stad_lookup = stadiums.set_index("TEAM")[["STADIUM", "LATITUDE", "LONGITUDE"]]
    df = matchups.merge(stad_lookup, left_on="TEAM_B", right_index=True, how="left")

    # Day/Date conversions
    def extract_year(v):
        try:
            return dateparser.parse(str(v)).year
        except Exception:
            return None

    year_hints = df["weatherapi_datetime"].map(extract_year)
    df["Day"] = df["DAY"].map(to_full_day)
    df["Date"] = [to_full_month_date(d, yh) for d, yh in zip(df["DATE"], year_hints)]
    df["away_team"] = df["TEAM_A"]
    df["home_team"] = df["TEAM_B"]
    df["Stadium"] = df["STADIUM"]
    df["Time"] = df["TIME"]

    # Weather fetch
    for c in ["temp_f", "wind_mph", "wind_dir", "precip_in", "chance_of_rain"]:
        df[c] = None

    session = requests.Session()
    cache: Dict[Tuple[str, str], dict] = {}
    for idx, row in df.iterrows():
        lat = row.get("LATITUDE")
        lon = row.get("LONGITUDE")
        when_iso = row.get("weatherapi_datetime")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(when_iso):
            continue
        try:
            w = fetch_hourly_weather(float(lat), float(lon), str(when_iso), session, cache)
            for k, v in w.items():
                df.at[idx, k] = v
        except Exception as e:
            logging.warning(f"Weather fetch failed for {row.get('TEAM_B')} @ {row.get('STADIUM')}: {e}")

    # Odds: use only CONSENSUS rows, match TEAM_A -> away_team
    odds["book_norm"] = odds["book"].astype(str).str.upper().str.strip()
    consensus = odds[odds["book_norm"] == "CONSENSUS"].copy()

    # Keep necessary columns
    consensus = consensus[["away_team", "home_team", "spread_home", "spread_away", "total", "game_id"]]

    # Merge on away team (TEAM_A)
    df = df.merge(consensus, left_on="TEAM_A", right_on="away_team", how="left", suffixes=("", "_odds"))

    # Numeric spreads
    df["spread_home_num"] = df["spread_home"].apply(_to_float)
    df["spread_away_num"] = df["spread_away"].apply(_to_float)

    # Choose odds & mark side
    def _choose_odds(row):
        sh, sa = row["spread_home_num"], row["spread_away_num"]
        if sh is None and sa is None:
            return None, None
        if sh is None:
            return sa, "away"
        if sa is None:
            return sh, "home"
        return (sh, "home") if sh <= sa else (sa, "away")

    chosen = df.apply(_choose_odds, axis=1, result_type="expand")
    df["Odds_num"] = chosen[0]
    df["favorite_side"] = chosen[1]

    # Favorite team name (use favorite_side; DO NOT self-reference favorite)
    df["favorite"] = df.apply(
        lambda r: r["home_team"] if r["favorite_side"] == "home"
        else (r["away_team"] if r["favorite_side"] == "away" else None),
        axis=1
    )

    # Total, Game ID from merge
    df["total"] = df["total"]
    df["game_id"] = df["game_id"]

    # Render Odds as string (drop .0)
    df["Odds"] = df["Odds_num"].apply(_trim_num)

    # Final column order
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
        "favorite",
        "total",
        "game_id",
    ]

    # Ensure output directory and write
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUTPUT, index=False)
    logging.info("Wrote %s with %d rows.", OUTPUT.resolve(), len(df))

    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Build the official games.csv at data/processed/games.csv

Updates:
- Append Odds (from data/raw/odds.csv where book == "CONSENSUS"), favorite, total, and game_id.
- Match odds by TEAM_A (week_matchups_odds.csv) -> away_team (odds.csv).
- Odds = lower (more negative) of spread_home vs spread_away.
- favorite = home_team if chosen value came from spread_home, else away_team.
- Keep Odds column (now sourced from odds.csv) and append game_id as the final column.

Inputs:
- data/raw/week_matchups_odds.csv  [DAY, DATE, TIME, TEAM_A, TEAM_B, weatherapi_datetime]
- data/team_stadiums.csv           [TEAM, STADIUM, LATITUDE, LONGITUDE]
- data/raw/odds.csv                [away_team, home_team, book, spread_home, spread_away, total, game_id]

Env:
- WEATHERAPI_KEY for WeatherAPI.com
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import requests
from dateutil import parser as dateparser

# ---------- Config ----------
RAW_MATCHUPS = Path("data/raw/week_matchups_odds.csv")
TEAM_STADIUMS = Path("data/team_stadiums.csv")
RAW_ODDS = Path("data/raw/odds.csv")
OUTPUT = Path("data/processed/games.csv")

WEATHER_API_KEY = os.getenv("WEATHERAPI_KEY")
WEATHER_BASE = "https://api.weatherapi.com/v1/forecast.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Helpers ----------
DAY_ABBR_TO_FULL = {'SUN':'Sunday','MON':'Monday','TUE':'Tuesday','WED':'Wednesday','THU':'Thursday','FRI':'Friday','SAT':'Saturday'}
MONTH_ABBR_TO_FULL = {'Jan':'January','Feb':'February','Mar':'March','Apr':'April','May':'May','Jun':'June','Jul':'July','Aug':'August','Sep':'September','Sept':'September','Oct':'October','Nov':'November','Dec':'December'}

def to_full_day(day_abbr: str) -> str:
    if pd.isna(day_abbr): return ""
    d = str(day_abbr).strip().upper()
    return DAY_ABBR_TO_FULL.get(d, d.title())

def to_full_month_date(date_str: str, year_hint: Optional[int]) -> str:
    if pd.isna(date_str) or not str(date_str).strip(): return ""
    s = str(date_str).strip().replace(",", "")
    parts = s.split()
    if len(parts) >= 2 and parts[0] in MONTH_ABBR_TO_FULL:
        try:
            return f"{MONTH_ABBR_TO_FULL[parts[0]]} {int(parts[1])}"
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
    best = None; best_diff = 10**18
    for h in hours or []:
        te = h.get("time_epoch")
        if te is None: continue
        diff = abs(te - target_epoch)
        if diff < best_diff:
            best_diff = diff; best = h
    return best

def fetch_hourly_weather(lat: float, lon: float, when_iso: str, session: requests.Session, cache: Dict[Tuple[str, str], dict]) -> dict:
    if not WEATHER_API_KEY:
        raise RuntimeError("WEATHERAPI_KEY is not set.")
    when_dt = dateparser.parse(str(when_iso))
    if when_dt is None:
        raise ValueError(f"Could not parse weatherapi_datetime: {when_iso}")
    date_key = when_dt.strftime("%Y-%m-%d")
    q = f"{lat:.4f},{lon:.4f}"
    ck = (q, date_key)

    if ck not in cache:
        params = {"key": WEATHER_API_KEY, "q": q, "dt": date_key, "aqi":"no", "alerts":"no"}
        resp = session.get(WEATHER_BASE, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"WeatherAPI error {resp.status_code}: {resp.text}")
        cache[ck] = resp.json()

    data = cache[ck]
    hours = []
    for fd in data.get("forecast", {}).get("forecastday", []):
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

def main() -> int:
    for p in [RAW_MATCHUPS, TEAM_STADIUMS, RAW_ODDS]:
        if not p.exists():
            logging.error(f"Missing required input: {p}")
            return 2

    matchups = pd.read_csv(RAW_MATCHUPS)
    stadiums = pd.read_csv(TEAM_STADIUMS)
    odds = pd.read_csv(RAW_ODDS)

    req_match = {"DAY","DATE","TIME","TEAM_A","TEAM_B","weatherapi_datetime"}
    req_stad  = {"TEAM","STADIUM","LATITUDE","LONGITUDE"}
    req_odds  = {"away_team","home_team","book","spread_home","spread_away","total","game_id"}
    missing = []
    if req_match - set(matchups.columns): missing.append(f"{RAW_MATCHUPS}: {sorted(req_match - set(matchups.columns))}")
    if req_stad  - set(stadiums.columns): missing.append(f"{TEAM_STADIUMS}: {sorted(req_stad  - set(stadiums.columns))}")
    if req_odds  - set(odds.columns):     missing.append(f"{RAW_ODDS}: {sorted(req_odds  - set(odds.columns))}")
    if missing:
        for m in missing: logging.error(f"Missing columns -> {m}")
        return 2

    # Merge stadium info for home team
    stad_lookup = stadiums.set_index("TEAM")[["STADIUM","LATITUDE","LONGITUDE"]]
    df = matchups.merge(stad_lookup, left_on="TEAM_B", right_index=True, how="left")

    # Day/Date transforms
    def year_from(v):
        try: return dateparser.parse(str(v)).year
        except Exception: return None
    year_hints = df["weatherapi_datetime"].map(year_from)
    df["Day"] = df["DAY"].map(to_full_day)
    df["Date"] = [to_full_month_date(d, y) for d, y in zip(df["DATE"], year_hints)]
    df["away_team"] = df["TEAM_A"]
    df["home_team"] = df["TEAM_B"]
    df["Stadium"]   = df["STADIUM"]
    df["Time"]      = df["TIME"]

    # Weather
    for c in ["temp_f","wind_mph","wind_dir","precip_in","chance_of_rain"]:
        df[c] = None
    session = requests.Session(); cache: Dict[Tuple[str,str],dict] = {}
    for idx, row in df.iterrows():
        lat, lon, when_iso = row.get("LATITUDE"), row.get("LONGITUDE"), row.get("weatherapi_datetime")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(when_iso):
            continue
        try:
            w = fetch_hourly_weather(float(lat), float(lon), str(when_iso), session, cache)
            for k,v in w.items(): df.at[idx, k] = v
        except Exception as e:
            logging.warning(f"Weather fetch failed for {row.get('TEAM_B')} @ {row.get('STADIUM')}: {e}")

    # Odds merge: use only CONSENSUS rows, match TEAM_A -> away_team
    odds["book_norm"] = odds["book"].astype(str).str.upper().str.strip()
    consensus = odds[odds["book_norm"] == "CONSENSUS"].copy()
    consensus = consensus[["away_team","home_team","spread_home","spread_away","total","game_id"]]

    df = df.merge(consensus, left_on="TEAM_A", right_on="away_team", how="left", suffixes=("", "_odds"))

    # Compute Odds and favorite
    def to_float(v):
        try: return float(v)
        except Exception: return None
    df["spread_home_num"] = df["spread_home"].apply(to_float)
    df["spread_away_num"] = df["spread_away"].apply(to_float)

    def choose(row):
        sh, sa = row["spread_home_num"], row["spread_away_num"]
        if sh is None and sa is None: return None, None
        if sh is None: return sa, "away"
        if sa is None: return sh, "home"
        return (sh, "home") if sh <= sa else (sa, "away")

    chosen = df.apply(choose, axis=1, result_type="expand")
    df["Odds"] = chosen[0]
    df["favorite"] = chosen[1].map({"home": df["home_team"], "away": df["away_team"]})
    # map above doesn't broadcast; do row-wise selection:
    df["favorite"] = df.apply(lambda r: r["home_team"] if r["favorite"] == "home" else (r["away_team"] if r["favorite"] == "away" else None), axis=1)

    # total and game_id already present from merge
    # Final column order
    out_cols = ["Day","Date","Time","away_team","home_team","Stadium","temp_f","wind_mph","wind_dir","precip_in","chance_of_rain","Odds","favorite","total","game_id"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUTPUT, index=False)
    logging.info(f"Wrote {OUTPUT.resolve()} with {len(df)} rows.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

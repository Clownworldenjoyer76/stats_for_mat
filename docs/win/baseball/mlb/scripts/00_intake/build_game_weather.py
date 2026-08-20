#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/build_game_weather.py
#
# Reads:
#   docs/win/baseball/mlb/data/weather/metno_raw/{date}_metno_raw.csv
#   docs/win/baseball/mlb/maps/mlb_venue_ids.csv
#
# Writes:
#   docs/win/baseball/mlb/data/weather/{date}_weather.csv
#
# Output columns:
#   gamePk,venue_id,weather_applicable,weather_time,temp_f,wind_mph,wind_dir,
#   precip_in,humidity,will_it_rain,wind_blowing_out,
#   air_pressure_at_sea_level,dew_point_f,symbol_code

import re
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path("docs/win/baseball/mlb")
MAPS_DIR = Path("docs/win/baseball/mlb/maps")
WEATHER_DIR = BASE_DIR / "data/weather"
RAW_IN_DIR = WEATHER_DIR / "metno_raw"
ERROR_DIR = BASE_DIR / "errors/00_intake"

WEATHER_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "build_game_weather.txt"

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "gamePk",
    "venue_id",
    "weather_applicable",
    "weather_time",
    "temp_f",
    "wind_mph",
    "wind_dir",
    "precip_in",
    "humidity",
    "will_it_rain",
    "wind_blowing_out",
    "air_pressure_at_sea_level",
    "dew_point_f",
    "symbol_code",
]

COMPASS_16 = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]

HPA_TO_INHG = 0.0295299830714

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
        f.write(
            f"{_now()} | "
            f"{level:<5} | "
            f"{safe_msg}\n"
        )


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


def _round(value, digits: int):
    if value is None:
        return ""

    return round(value, digits)


def c_to_f(celsius):
    value = _to_float(celsius)

    if value is None:
        return ""

    return _round(
        (value * 9 / 5) + 32,
        1,
    )


def hpa_to_inhg(hpa):
    value = _to_float(hpa)

    if value is None:
        return ""

    return _round(
        value * HPA_TO_INHG,
        2,
    )


def ms_to_mph(ms):
    value = _to_float(ms)

    if value is None:
        return ""

    return _round(
        value * 2.236936,
        1,
    )


def mm_to_inches(mm):
    value = _to_float(mm)

    if value is None:
        return ""

    return _round(
        value / 25.4,
        2,
    )


def degrees_to_compass(degrees):
    value = _to_float(degrees)

    if value is None:
        return ""

    value = value % 360
    index = int(
        (value + 11.25) / 22.5
    ) % 16

    return COMPASS_16[index]


def parse_metno_time_utc(value: str):
    s = _clean(value)

    if not s:
        return None

    try:
        return datetime.fromisoformat(
            s.replace("Z", "+00:00")
        ).astimezone(UTC)
    except Exception:
        return None


def metno_time_to_local_hour(
    value: str,
    time_zone_id: str,
) -> str:
    dt_utc = parse_metno_time_utc(value)
    tz = _clean(time_zone_id)

    if dt_utc is None or not tz:
        return ""

    try:
        local_dt = dt_utc.astimezone(
            ZoneInfo(tz)
        )
        return local_dt.strftime(
            "%Y-%m-%d %H:00"
        )
    except Exception:
        return ""


def load_venue_map() -> dict:
    path = MAPS_DIR / "mlb_venue_ids.csv"

    df = pd.read_csv(
        path,
        dtype={"venue_id": str},
        encoding="utf-8-sig",
    )

    df["venue_id"] = (
        df["venue_id"]
        .astype(str)
        .str.strip()
    )

    return (
        df
        .set_index("venue_id")
        .to_dict("index")
    )


def get_wind_out_direction(
    venue_id: str,
    venue_map: dict,
) -> str:
    vinfo = venue_map.get(
        _clean(venue_id),
        {},
    )

    return _clean(
        vinfo.get("wind_out_direction")
    ).upper()


def get_weather_applicable(row: dict) -> int:
    value = _clean(
        row.get("weather_applicable")
    )

    try:
        return int(float(value))
    except Exception:
        return 0


def get_will_it_rain(
    precipitation_amount,
):
    value = _to_float(
        precipitation_amount
    )

    if value is None:
        return ""

    return 1.0 if value > 0 else 0.0


def get_wind_blowing_out(
    wind_dir: str,
    wind_out_direction: str,
    weather_applicable: int,
):
    if not weather_applicable:
        return ""

    wind = _clean(
        wind_dir
    ).upper()
    wind_out = _clean(
        wind_out_direction
    ).upper()

    if (
        not wind
        or not wind_out
        or wind_out == "NULL"
    ):
        return ""

    return (
        1.0
        if wind == wind_out
        else 0.0
    )


def build_output_row(
    raw_row: dict,
    venue_map: dict,
) -> dict:
    venue_id = _clean(
        raw_row.get("venue_id")
    )
    weather_applicable = (
        get_weather_applicable(raw_row)
    )

    if not weather_applicable:
        return {
            "gamePk": _clean(
                raw_row.get("gamePk")
            ),
            "venue_id": venue_id,
            "weather_applicable": weather_applicable,
            "weather_time": "",
            "temp_f": "",
            "wind_mph": "",
            "wind_dir": "",
            "precip_in": "",
            "humidity": "",
            "will_it_rain": "",
            "wind_blowing_out": "",
            "air_pressure_at_sea_level": "",
            "dew_point_f": "",
            "symbol_code": "",
        }

    wind_dir = degrees_to_compass(
        raw_row.get("wind_from_direction")
    )
    wind_out_direction = (
        get_wind_out_direction(
            venue_id,
            venue_map,
        )
    )

    return {
        "gamePk": _clean(
            raw_row.get("gamePk")
        ),
        "venue_id": venue_id,
        "weather_applicable": weather_applicable,
        "weather_time": metno_time_to_local_hour(
            raw_row.get("time"),
            raw_row.get("time_zone_id"),
        ),
        "temp_f": c_to_f(
            raw_row.get("air_temperature")
        ),
        "wind_mph": ms_to_mph(
            raw_row.get("wind_speed")
        ),
        "wind_dir": wind_dir,
        "precip_in": mm_to_inches(
            raw_row.get("precipitation_amount")
        ),
        "humidity": _round(
            _to_float(
                raw_row.get("relative_humidity")
            ),
            1,
        ),
        "will_it_rain": get_will_it_rain(
            raw_row.get("precipitation_amount")
        ),
        "wind_blowing_out": get_wind_blowing_out(
            wind_dir=wind_dir,
            wind_out_direction=wind_out_direction,
            weather_applicable=weather_applicable,
        ),
        "air_pressure_at_sea_level": hpa_to_inhg(
            raw_row.get(
                "air_pressure_at_sea_level"
            )
        ),
        "dew_point_f": c_to_f(
            raw_row.get(
                "dew_point_temperature"
            )
        ),
        "symbol_code": _clean(
            raw_row.get("symbol_code")
        ),
    }


# ─────────────────────────────────────────────
# PROCESS ONE RAW FILE
# ─────────────────────────────────────────────

def process_raw_file(
    raw_path: Path,
    venue_map: dict,
    summary: dict,
) -> None:
    date_str = raw_path.name.replace(
        "_metno_raw.csv",
        "",
    )
    out_path = (
        WEATHER_DIR
        / f"{date_str}_weather.csv"
    )

    try:
        raw_df = pd.read_csv(
            raw_path,
            dtype=str,
        )
    except Exception:
        _log(
            f"{raw_path.name} | "
            f"failed to read raw MET Norway CSV",
            "ERROR",
        )
        summary["errors"] += 1
        return

    rows = []

    for _, row in raw_df.iterrows():
        rows.append(
            build_output_row(
                raw_row=row.to_dict(),
                venue_map=venue_map,
            )
        )

    pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    ).to_csv(
        out_path,
        index=False,
    )

    _log(
        f"WROTE: {out_path} "
        f"({len(rows)} rows)"
    )
    summary["files_written"] += 1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== build_game_weather RUN "
            f"{_now()} ===\n"
        )

    summary = {
        "files_written": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        venue_map = load_venue_map()

        _log(
            f"Venue map loaded: "
            f"{len(venue_map)} entries"
        )

        raw_files = sorted(
            RAW_IN_DIR.glob(
                "*_metno_raw.csv"
            )
        )

        _log(
            f"Raw MET Norway files found: "
            f"{len(raw_files)}"
        )

        if not raw_files:
            _log(
                "No raw MET Norway files found "
                "— nothing to build",
                "WARN",
            )
            summary["skipped"] += 1

        for raw_path in raw_files:
            process_raw_file(
                raw_path,
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

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines) + "\n"
        )

    print(
        f"build_game_weather complete. "
        f"{summary['files_written']} "
        f"files written. "
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/enrich_game_context.py
#
# Runs after scrape_mlb_raw.py.
# Joins Statcast, park factors (lineup-weighted by bat side), and weather
# from cache to produce {date}_game_context.csv.
#
# Weather is NOT fetched here. Run fetch_park_weather.py / build_game_weather.py first.

import argparse
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR    = Path("docs/win/baseball/mlb")
MLBraw_DIR  = BASE_DIR / "00_intake/mlb_raw"
MAPS_DIR    = BASE_DIR / "maps"
DATA_DIR    = BASE_DIR / "data"
WEATHER_DIR = DATA_DIR / "weather"
ERROR_DIR   = BASE_DIR / "errors/00_intake"

ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "enrich_game_context.txt"


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def normalize_date(value: str) -> str:
    return str(value or "").strip().replace("-", "_")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"MISSING {label}: {path}")


# ─────────────────────────────────────────────
# LOAD MAPS
# ─────────────────────────────────────────────

def load_maps():
    venue_path = MAPS_DIR / "mlb_venue_ids.csv"
    pitcher_path = MAPS_DIR / "mlb_pitcher_ids.csv"
    batter_path = MAPS_DIR / "mlb_batter_ids.csv"

    require_file(venue_path, "venue map")
    require_file(pitcher_path, "pitcher map")
    require_file(batter_path, "batter map")

    venue_df = pd.read_csv(
        venue_path,
        dtype={"venue_id": str},
        encoding="utf-8-sig"
    )
    venue_df["venue_id"] = venue_df["venue_id"].str.strip()
    venue_map = venue_df.set_index("venue_id").to_dict("index")

    pitcher_df = pd.read_csv(
        pitcher_path,
        dtype={"pitcher_id": str},
        encoding="utf-8-sig"
    )
    pitcher_df["pitcher_id"] = pitcher_df["pitcher_id"].str.strip()
    pitcher_map = pitcher_df.set_index("pitcher_id")["pitch_hand_code"].to_dict()

    batter_df = pd.read_csv(
        batter_path,
        dtype={"batter_id": str},
        encoding="utf-8-sig"
    )
    batter_df["batter_id"] = batter_df["batter_id"].str.strip()
    batter_map = batter_df.set_index("batter_id").to_dict("index")

    return venue_map, pitcher_map, batter_map


# ─────────────────────────────────────────────
# LOAD STATCAST DATA
# ─────────────────────────────────────────────

BATTING_COLS = [
    "player_id",
    "pa",
    "xwoba",
    "barrel_pct",
    "hard_hit_pct",
    "k_pct",
    "bb_pct",
    "exit_velo",
    "sample_flag",
]

PITCHING_COLS = [
    "player_id",
    "pa",
    "xwoba",
    "k_pct",
    "bb_pct",
    "barrel_pct",
    "whiff_pct",
    "exit_velo",
    "sample_flag",
]

STATCAST_PRIORITY = ["2022", "2023", "2024", "2025", "2026"]


def _load_statcast(directory: Path, cols: list, id_col: str = "player_id") -> dict:
    merged = {}

    if not directory.exists():
        _log(f"Statcast directory missing: {directory}", "WARN")
        return merged

    for yr in STATCAST_PRIORITY:
        files = sorted(directory.glob(f"*{yr}*_clean.csv"))
        if not files:
            continue

        df = pd.read_csv(files[0], dtype={id_col: str})
        if id_col not in df.columns:
            _log(f"Statcast file missing id column {id_col}: {files[0]}", "WARN")
            continue

        df[id_col] = df[id_col].str.strip()
        available = [c for c in cols if c in df.columns]

        if id_col not in available:
            available = [id_col] + available

        merged.update(df[available].set_index(id_col).to_dict("index"))

    return merged


def load_statcast():
    batting = _load_statcast(DATA_DIR / "batting", BATTING_COLS)
    pitching = _load_statcast(DATA_DIR / "pitching", PITCHING_COLS)

    fielding = {}
    fielding_dir = DATA_DIR / "fielding"
    if not fielding_dir.exists():
        _log(f"Statcast fielding directory missing: {fielding_dir}", "WARN")
    else:
        for yr in STATCAST_PRIORITY:
            for f in sorted(fielding_dir.glob(f"*{yr}*_clean.csv")):
                df = pd.read_csv(f, dtype={"id": str})
                if "id" not in df.columns:
                    _log(f"Fielding file missing id column: {f}", "WARN")
                    continue

                df["id"] = df["id"].str.strip()
                for _, row in df.iterrows():
                    fielding[row["id"]] = row.to_dict()

    baserunning = {}
    baserunning_dir = DATA_DIR / "baserunning"
    if not baserunning_dir.exists():
        _log(f"Statcast baserunning directory missing: {baserunning_dir}", "WARN")
    else:
        for yr in STATCAST_PRIORITY:
            for f in sorted(baserunning_dir.glob(f"*{yr}*_clean.csv")):
                df = pd.read_csv(f, dtype={"player_id": str})
                if "player_id" not in df.columns:
                    _log(f"Baserunning file missing player_id column: {f}", "WARN")
                    continue

                df["player_id"] = df["player_id"].str.strip()
                for _, row in df.iterrows():
                    baserunning[row["player_id"]] = row.to_dict()

    return batting, pitching, fielding, baserunning


# ─────────────────────────────────────────────
# LOAD PARK FACTORS
# ─────────────────────────────────────────────

PARK_COLS = ["Park Factor", "wOBAcon", "xwOBAcon", "HR", "R"]


def load_park_factors() -> dict:
    park_index = {}
    park_dir = DATA_DIR / "park_factors"

    if not park_dir.exists():
        _log(f"Park factors directory missing: {park_dir}", "WARN")
        return park_index

    for f in sorted(park_dir.glob("park_*_clean.csv")):
        parts = f.stem.replace("_clean", "").split("_", 2)
        if len(parts) != 3:
            _log(f"Skipping unexpected park factor filename: {f.name}", "WARN")
            continue

        _, bat_side, condition = parts

        df = pd.read_csv(f, dtype={"venue_id": str})
        if "venue_id" not in df.columns:
            _log(f"Park factor file missing venue_id column: {f}", "WARN")
            continue

        df["venue_id"] = df["venue_id"].str.strip()

        for _, row in df.iterrows():
            vid = str(row["venue_id"]).strip()
            park_index.setdefault(vid, {}).setdefault(condition, {})[bat_side] = row.to_dict()

    return park_index


def get_park_condition(roof_type: str, day_night: str) -> str:
    rt = (roof_type or "").strip().lower()
    dn = (day_night or "").strip().lower()

    if rt in {"dome", "indoor"}:
        return "roof_closed"

    if rt == "retractable":
        return "day" if dn == "day" else "night"

    return "open_air"


def weighted_park_factor(
    park_index: dict,
    venue_id: str,
    condition: str,
    n_left: int,
    n_right: int,
    n_switch: int,
) -> dict:
    venue_cond = park_index.get(venue_id, {}).get(condition, {})

    row_b = venue_cond.get("B", {})
    row_l = venue_cond.get("L", {})
    row_r = venue_cond.get("R", {})

    if not row_b and not row_l and not row_r:
        return {}

    eff_l = n_left + (n_switch * 0.5)
    eff_r = n_right + (n_switch * 0.5)
    total = eff_l + eff_r

    result = {}

    for col in PARK_COLS:
        val_b = row_b.get(col)
        val_l = row_l.get(col)
        val_r = row_r.get(col)

        result[f"park_{col}_B"] = val_b

        if total > 0 and val_l is not None and val_r is not None:
            try:
                result[f"park_{col}"] = round(
                    (float(val_l) * eff_l + float(val_r) * eff_r) / total,
                    4,
                )
            except (ValueError, TypeError):
                result[f"park_{col}"] = val_b
        else:
            result[f"park_{col}"] = val_b

    return result


# ─────────────────────────────────────────────
# LOAD WEATHER CACHE
# ─────────────────────────────────────────────

def load_weather(date_str: str) -> dict:
    weather_path = WEATHER_DIR / f"{date_str}_weather.csv"

    if not weather_path.exists():
        _log(f"No weather file for {date_str}: {weather_path}. Weather columns will be null.", "WARN")
        return {}

    df = pd.read_csv(weather_path, dtype={"gamePk": str})

    if "gamePk" not in df.columns:
        _log(f"Weather file missing gamePk column: {weather_path}", "WARN")
        return {}

    df["gamePk"] = df["gamePk"].str.strip()
    return df.set_index("gamePk").to_dict("index")


# ─────────────────────────────────────────────
# STATCAST LOOKUP HELPERS
# ─────────────────────────────────────────────

def get_pitcher_stats(pitcher_id: str, pitching: dict) -> tuple:
    pid = str(pitcher_id or "").strip()

    if not pid:
        _log("Blank pitcher_id", "WARN")
        return {}, False

    row = pitching.get(pid)

    if not row:
        _log(f"Pitcher {pid} not found in any Statcast file", "WARN")
        return {}, False

    return row, True


BATTER_AVG_COLS = [
    "xwoba",
    "barrel_pct",
    "hard_hit_pct",
    "k_pct",
    "bb_pct",
    "exit_velo",
]


def aggregate_lineup(
    batter_ids: list,
    batting: dict,
    fielding: dict,
    baserunning: dict,
    batter_map: dict,
    side: str,
) -> tuple:
    avg_accum = {c: [] for c in BATTER_AVG_COLS}

    frv_sum = 0.0
    brv_sum = 0.0
    low_sample = 0
    catcher_framing = None

    n_left = 0
    n_right = 0
    n_switch = 0
    batters_found = 0

    for i, bid in enumerate(batter_ids):
        bid = str(bid or "").strip()
        label = f"{side}_bat_{i + 1}"

        if not bid:
            _log(f"Blank batter id ({label})", "WARN")
            continue

        bmap_row = batter_map.get(bid, {})

        bat_side = str(bmap_row.get("bat_side_code", "")).strip().upper()
        if bat_side == "L":
            n_left += 1
        elif bat_side == "R":
            n_right += 1
        elif bat_side == "S":
            n_switch += 1

        bstats = batting.get(bid)
        if not bstats:
            _log(f"Batter {bid} ({label}) not found in any Statcast file", "WARN")
        else:
            batters_found += 1

            for col in BATTER_AVG_COLS:
                val = bstats.get(col)
                if val is not None:
                    try:
                        avg_accum[col].append(float(val))
                    except (ValueError, TypeError):
                        pass

            if bstats.get("sample_flag") == "low":
                low_sample += 1

        fstats = fielding.get(bid, {})
        try:
            frv_sum += float(fstats.get("total_runs", 0) or 0)
        except (ValueError, TypeError):
            pass

        if str(bmap_row.get("primary_position_code", "")).strip() == "2":
            try:
                catcher_framing = float(fstats.get("framing_runs", 0) or 0)
            except (ValueError, TypeError):
                catcher_framing = None

        brstats = baserunning.get(bid, {})
        try:
            brv_sum += float(brstats.get("runner_runs_tot", 0) or 0)
        except (ValueError, TypeError):
            pass

    result = {}

    for col in BATTER_AVG_COLS:
        vals = avg_accum[col]
        result[f"{side}_lineup_{col}"] = (sum(vals) / len(vals)) if vals else None

    result[f"{side}_lineup_frv"] = frv_sum
    result[f"{side}_lineup_brv"] = brv_sum
    result[f"{side}_catcher_framing"] = catcher_framing
    result[f"{side}_low_sample_count"] = low_sample
    result[f"{side}_n_left"] = n_left
    result[f"{side}_n_right"] = n_right
    result[f"{side}_n_switch"] = n_switch

    return result, n_left, n_right, n_switch, batters_found


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(
    date_str: str,
    venue_map: dict,
    pitcher_map: dict,
    batter_map: dict,
    batting: dict,
    pitching: dict,
    fielding: dict,
    baserunning: dict,
    park_index: dict,
    summary: dict,
) -> None:
    date_str = normalize_date(date_str)

    raw_path = MLBraw_DIR / f"{date_str}_mlb_raw.csv"

    if not raw_path.exists():
        _log(f"mlb_raw file not found: {raw_path}", "ERROR")
        summary["missing_raw"] += 1
        summary["errors"] += 1
        return

    df = pd.read_csv(raw_path, dtype=str)

    if df.empty:
        _log(f"{date_str} | mlb_raw file has zero rows: {raw_path}", "ERROR")
        summary["errors"] += 1
        return

    weather_map = load_weather(date_str)
    _log(f"--- {date_str} | raw games={len(df)} | weather rows={len(weather_map)}")

    output_rows = []

    for _, row in df.iterrows():
        game_pk = str(row.get("gamePk", "") or "").strip()
        game_date = row.get("game_date", "")
        venue_id = str(row.get("venue_id", "") or "").strip()
        day_night = str(row.get("day_night", "") or "").strip().lower()
        home_tid = str(row.get("home_team_id", "") or "").strip()
        away_tid = str(row.get("away_team_id", "") or "").strip()
        home_pid = str(row.get("home_pitcher_id", "") or "").strip()
        away_pid = str(row.get("away_pitcher_id", "") or "").strip()

        if not game_pk:
            _log(f"{date_str} | row missing gamePk: {row.to_dict()}", "ERROR")
            summary["errors"] += 1
            continue

        vinfo = venue_map.get(venue_id, {})
        roof_type = vinfo.get("roof_type", "")
        turf_type = vinfo.get("turf_type", "")

        home_hand = pitcher_map.get(home_pid, None)
        away_hand = pitcher_map.get(away_pid, None)

        hpstats, home_sp_found = get_pitcher_stats(home_pid, pitching)
        apstats, away_sp_found = get_pitcher_stats(away_pid, pitching)

        if not home_sp_found:
            summary["missing_pitcher"] += 1
        if not away_sp_found:
            summary["missing_pitcher"] += 1

        home_bats = [row.get(f"home_bat_{i}_id", "") for i in range(1, 10)]
        away_bats = [row.get(f"away_bat_{i}_id", "") for i in range(1, 10)]

        home_agg, home_l, home_r, home_s, home_batters_found = aggregate_lineup(
            home_bats,
            batting,
            fielding,
            baserunning,
            batter_map,
            "home",
        )

        away_agg, away_l, away_r, away_s, away_batters_found = aggregate_lineup(
            away_bats,
            batting,
            fielding,
            baserunning,
            batter_map,
            "away",
        )

        summary["missing_batter"] += (9 - home_batters_found) + (9 - away_batters_found)

        condition = get_park_condition(roof_type, day_night)

        total_l = home_l + away_l
        total_r = home_r + away_r
        total_s = home_s + away_s

        park = weighted_park_factor(
            park_index,
            venue_id,
            condition,
            total_l,
            total_r,
            total_s,
        )

        if not park:
            _log(f"Park factor not found: gamePk={game_pk} venue={venue_id} condition={condition}", "WARN")

        w = weather_map.get(game_pk, {})
        if w:
            summary["weather_cache_hits"] += 1

        output_rows.append({
            "game_date": game_date,
            "gamePk": game_pk,
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "venue_id": venue_id,
            "roof_type": roof_type,
            "turf_type": turf_type,
            "home_pitcher_id": home_pid,
            "away_pitcher_id": away_pid,
            "home_pitcher_hand": home_hand,
            "away_pitcher_hand": away_hand,
            "home_sp_xwoba": hpstats.get("xwoba"),
            "away_sp_xwoba": apstats.get("xwoba"),
            "home_sp_k_pct": hpstats.get("k_pct"),
            "away_sp_k_pct": apstats.get("k_pct"),
            "home_sp_bb_pct": hpstats.get("bb_pct"),
            "away_sp_bb_pct": apstats.get("bb_pct"),
            "home_sp_barrel_pct": hpstats.get("barrel_pct"),
            "away_sp_barrel_pct": apstats.get("barrel_pct"),
            "home_sp_whiff_pct": hpstats.get("whiff_pct"),
            "away_sp_whiff_pct": apstats.get("whiff_pct"),
            "home_sp_sample_flag": hpstats.get("sample_flag"),
            "away_sp_sample_flag": apstats.get("sample_flag"),
            **home_agg,
            **away_agg,
            "park_factor": park.get("park_Park Factor"),
            "park_wOBAcon": park.get("park_wOBAcon"),
            "park_xwOBAcon": park.get("park_xwOBAcon"),
            "park_HR": park.get("park_HR"),
            "park_R": park.get("park_R"),
            "park_factor_B": park.get("park_Park Factor_B"),
            "park_wOBAcon_B": park.get("park_wOBAcon_B"),
            "park_xwOBAcon_B": park.get("park_xwOBAcon_B"),
            "park_HR_B": park.get("park_HR_B"),
            "park_R_B": park.get("park_R_B"),
            "home_batters_found": home_batters_found,
            "away_batters_found": away_batters_found,
            "home_sp_found": 1 if home_sp_found else 0,
            "away_sp_found": 1 if away_sp_found else 0,
            "sp_data_available": 1 if (home_sp_found and away_sp_found) else 0,
            "lineup_data_available": 1 if (home_batters_found == 9 and away_batters_found == 9) else 0,
            "weather_applicable": w.get("weather_applicable"),
            "weather_time": w.get("weather_time"),
            "temp_f": w.get("temp_f"),
            "wind_mph": w.get("wind_mph"),
            "wind_dir": w.get("wind_dir"),
            "precip_in": w.get("precip_in"),
            "humidity": w.get("humidity"),
            "will_it_rain": w.get("will_it_rain"),
            "wind_blowing_out": w.get("wind_blowing_out"),
            "air_pressure_at_sea_level": w.get("air_pressure_at_sea_level"),
            "dew_point_f": w.get("dew_point_f"),
            "symbol_code": w.get("symbol_code"),
        })

    if not output_rows:
        _log(f"{date_str} | no output rows built; context file not written", "ERROR")
        summary["errors"] += 1
        return

    out_path = MLBraw_DIR / f"{date_str}_game_context.csv"
    pd.DataFrame(output_rows).to_csv(out_path, index=False)

    _log(f"{date_str} | WROTE: {out_path} ({len(output_rows)} rows)")

    summary["files_written"] += 1
    summary["rows_written"] += len(output_rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dates",
        nargs="*",
        help="Optional date(s) to process. Accepts YYYY_MM_DD or YYYY-MM-DD. If omitted, processes all *_mlb_raw.csv files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== enrich_game_context RUN {_now()} ===\n")

    summary = {
        "files_written": 0,
        "rows_written": 0,
        "missing_raw": 0,
        "missing_pitcher": 0,
        "missing_batter": 0,
        "weather_cache_hits": 0,
        "errors": 0,
    }

    try:
        _log(f"BASE_DIR={BASE_DIR}")
        _log(f"MLBraw_DIR={MLBraw_DIR}")
        _log(f"MAPS_DIR={MAPS_DIR}")
        _log(f"DATA_DIR={DATA_DIR}")
        _log(f"WEATHER_DIR={WEATHER_DIR}")
        _log(f"ERROR_DIR={ERROR_DIR}")

        _log("Loading maps...")
        venue_map, pitcher_map, batter_map = load_maps()
        _log(f"venue_map={len(venue_map)} | pitcher_map={len(pitcher_map)} | batter_map={len(batter_map)}")

        _log("Loading Statcast data...")
        batting, pitching, fielding, baserunning = load_statcast()
        _log(f"batting={len(batting)} | pitching={len(pitching)} | fielding={len(fielding)} | baserunning={len(baserunning)}")

        _log("Loading park factors...")
        park_index = load_park_factors()
        total_park = sum(
            len(sides)
            for venue_conditions in park_index.values()
            for sides in venue_conditions.values()
        )
        _log(f"park_index venues={len(park_index)} | total entries={total_park}")

        if args.dates:
            dates = [normalize_date(d) for d in args.dates]
            _log(f"dates requested: {dates}")
        else:
            raw_files = sorted(MLBraw_DIR.glob("*_mlb_raw.csv"))
            dates = [rf.stem.replace("_mlb_raw", "") for rf in raw_files]
            _log(f"mlb_raw files found: {len(raw_files)}")

        if not dates:
            _log("No dates to process.", "ERROR")
            summary["errors"] += 1

        for date_str in dates:
            try:
                process_date(
                    date_str,
                    venue_map,
                    pitcher_map,
                    batter_map,
                    batting,
                    pitching,
                    fielding,
                    baserunning,
                    park_index,
                    summary,
                )
            except Exception as e:
                _log(f"{date_str} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"

    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_written     : {summary['files_written']}",
        f"  rows_written      : {summary['rows_written']}",
        f"  missing_raw       : {summary['missing_raw']}",
        f"  missing_pitcher   : {summary['missing_pitcher']}",
        f"  missing_batter    : {summary['missing_batter']}",
        f"  weather_cache_hits: {summary['weather_cache_hits']}",
        f"  errors            : {summary['errors']}",
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"enrich_game_context complete. {summary['files_written']} files written. Status: {status}")

    if summary["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

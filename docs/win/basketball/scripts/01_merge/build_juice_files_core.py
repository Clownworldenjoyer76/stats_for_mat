#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/build_juice_files_core.py

import csv
import math
import traceback
import sys
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import norm
import pandas as pd
import yaml

# ============================================================
# PATHS / CONFIG
# ============================================================

LEAGUES = ["nba", "ncaam", "wnba"]

INPUT_DIR  = Path("docs/win/basketball/01_merge")
OUTPUT_DIR = Path("docs/win/basketball/01_merge/01_merguiced")
ERROR_DIR  = Path("docs/win/basketball/errors/01_merge")
LOG_FILE   = ERROR_DIR / "build_juice_files.txt"
CONFIG_PATH = Path("docs/win/basketball/config/model_config.yaml")

ERROR_DIR.mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== build_juice_files RUN {datetime.now(timezone.utc).isoformat()} ===\n\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def audit(stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            f.write(f"  MSG: {msg}\n")
        if df is not None:
            f.write(f"  ROWS: {len(df)}\n")
        f.write("-" * 40 + "\n")


# ============================================================
# MODEL CONFIG
# ============================================================

def _require_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def load_model_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing model config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg.get("leagues"), dict):
        raise ValueError("model_config.yaml must contain a top-level 'leagues' mapping")
    return cfg


def _calibration_cfg(league_cfg: dict, market: str, side: str) -> dict:
    market_cfg = ((league_cfg.get("calibration") or {}).get(market) or {})
    cfg = market_cfg.get(side) or {"method": "none"}
    if isinstance(cfg, str):
        cfg = {"method": cfg}
    if not isinstance(cfg, dict):
        raise ValueError(f"calibration.{market}.{side} must be a mapping")
    return cfg


def _complementary_calibration_cfg(
    league_cfg: dict,
    market: str,
    first_side: str,
    second_side: str,
) -> dict:
    """
    Load a binary-market calibration where exactly one side is calibrated and
    the opposite side is derived as 1 - calibrated_probability.
    """
    market_cfg = ((league_cfg.get("calibration") or {}).get(market) or {})
    if not isinstance(market_cfg, dict):
        raise ValueError(f"calibration.{market} must be a mapping")

    canonical_side = str(
        market_cfg.get("canonical_side", first_side)
    ).strip().lower()

    if canonical_side not in {first_side, second_side}:
        raise ValueError(
            f"calibration.{market}.canonical_side must be "
            f"{first_side!r} or {second_side!r}"
        )

    cfg = market_cfg.get(canonical_side) or {"method": "none"}
    if isinstance(cfg, str):
        cfg = {"method": cfg}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"calibration.{market}.{canonical_side} must be a mapping"
        )

    opposite_side = second_side if canonical_side == first_side else first_side
    opposite_cfg = market_cfg.get(opposite_side)
    if opposite_cfg not in (None, {}, "none", "raw"):
        if isinstance(opposite_cfg, dict):
            opposite_method = str(opposite_cfg.get("method", "none")).strip().lower()
            if opposite_method not in {"none", "raw", ""}:
                raise ValueError(
                    f"calibration.{market}.{opposite_side} must not define an "
                    "independent calibration when complementary calibration is enabled"
                )

    return {
        "canonical_side": canonical_side,
        "config": cfg,
    }

def build_league_settings(model_cfg: dict) -> dict:
    settings = {}
    for league in LEAGUES:
        league_cfg = (model_cfg.get("leagues") or {}).get(league)
        if not isinstance(league_cfg, dict):
            raise ValueError(f"Missing model config for league={league}")

        status = str(league_cfg.get("status", "")).strip().lower()
        if status != "active":
            raise ValueError(f"League {league.upper()} is not active in model_config.yaml")

        edge_cfg = league_cfg.get("edge") or {}
        std_cfg = league_cfg.get("std") or {}

        spread_std_cfg = std_cfg.get("spread") or {}
        total_std_cfg = std_cfg.get("total") or {}

        if str(spread_std_cfg.get("mode", "")).strip().lower() != "fixed":
            raise ValueError(f"{league.upper()} spread STD mode must be fixed")
        if str(total_std_cfg.get("mode", "")).strip().lower() != "fixed":
            raise ValueError(f"{league.upper()} total STD mode must be fixed")

        settings[league.upper()] = {
            "ML_EDGE": _require_number(edge_cfg.get("moneyline"), f"{league}.edge.moneyline"),
            "SPREAD_EDGE": _require_number(edge_cfg.get("spread"), f"{league}.edge.spread"),
            "TOTAL_EDGE": _require_number(edge_cfg.get("total"), f"{league}.edge.total"),
            "SPREAD_STD": _require_number(spread_std_cfg.get("value"), f"{league}.std.spread.value"),
            "TOTAL_STD": _require_number(total_std_cfg.get("value"), f"{league}.std.total.value"),
            "CALIBRATION": {
                "moneyline": {
                    "home": _calibration_cfg(league_cfg, "moneyline", "home"),
                    "away": _calibration_cfg(league_cfg, "moneyline", "away"),
                },
                "spread": _complementary_calibration_cfg(
                    league_cfg,
                    "spread",
                    "home",
                    "away",
                ),
                "total": _complementary_calibration_cfg(
                    league_cfg,
                    "total",
                    "over",
                    "under",
                ),
            },
        }
    return settings


MODEL_CONFIG = load_model_config()
LEAGUE_SETTINGS = build_league_settings(MODEL_CONFIG)


# ============================================================
# HELPERS
# ============================================================

def american_to_decimal(odds):
    if pd.isna(odds) or str(odds).strip() == "":
        return ""
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return ""
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def to_american(dec):
    if dec == "" or pd.isna(dec) or float(dec) <= 1:
        return ""
    dec = float(dec)
    if dec >= 2:
        return f"+{int((dec - 1) * 100)}"
    return f"-{int(100 / (dec - 1))}"


def clamp_probability(p):
    return min(max(float(p), 0.01), 0.99)


def apply_calibration(p, cfg: dict):
    if p is None or pd.isna(p):
        return ""

    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""

    if not math.isfinite(p):
        return ""

    method = str((cfg or {}).get("method", "none")).strip().lower()

    if method in {"none", "raw", ""}:
        return p

    if method == "beta":
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        intercept = _require_number(cfg.get("intercept"), "beta.intercept")
        coef_log_p = _require_number(cfg.get("coef_log_p"), "beta.coef_log_p")
        coef_log_1mp = _require_number(cfg.get("coef_log_1mp"), "beta.coef_log_1mp")
        z = (
            intercept
            + coef_log_p * math.log(p)
            + coef_log_1mp * math.log(1.0 - p)
        )
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    raise ValueError(f"Unsupported calibration method: {method!r}")


def apply_complementary_calibration(
    raw_first,
    raw_second,
    calibration: dict,
    first_side: str,
    second_side: str,
) -> tuple[float | str, float | str]:
    """
    Calibrate exactly one side of a binary market and derive the other side
    as its mathematical complement.
    """
    canonical_side = str(calibration["canonical_side"]).strip().lower()
    cfg = calibration["config"]

    raw_canonical = raw_first if canonical_side == first_side else raw_second
    calibrated = apply_calibration(raw_canonical, cfg)

    if calibrated == "" or pd.isna(calibrated):
        return "", ""

    try:
        p_canonical = clamp_probability(float(calibrated))
    except (TypeError, ValueError):
        return "", ""

    p_opposite = 1.0 - p_canonical

    if canonical_side == first_side:
        return p_canonical, p_opposite

    if canonical_side == second_side:
        return p_opposite, p_canonical

    raise ValueError(
        f"Unsupported canonical side {canonical_side!r}; "
        f"expected {first_side!r} or {second_side!r}"
    )

def safe_implied_prob(decimal_value):
    """Convert a decimal odds value to implied probability. Returns '' if invalid."""
    if decimal_value == "" or pd.isna(decimal_value):
        return ""
    try:
        d = float(decimal_value)
    except (ValueError, TypeError):
        return ""
    if not math.isfinite(d) or d <= 0:
        return ""
    return 1.0 / d


def devig_pair(p_a, p_b):
    """
    Normalize a pair of implied probabilities so they sum to 1.
    Removes the bookmaker's overround (vig).
    Returns ('', '') if either input is missing/invalid.
    """
    if p_a == "" or p_b == "" or pd.isna(p_a) or pd.isna(p_b):
        return "", ""

    try:
        a = float(p_a)
        b = float(p_b)
    except (ValueError, TypeError):
        return "", ""

    if not math.isfinite(a) or not math.isfinite(b):
        return "", ""

    s = a + b

    if not math.isfinite(s) or s <= 0:
        return "", ""

    return a / s, b / s

    s = a + b
    if not math.isfinite(s) or s <= 0:
        return ""

    return a / s, b / s


def wipe_outputs():
    for league in LEAGUES:
        for subdir in ["moneyline", "spread", "total"]:
            folder = OUTPUT_DIR / league / subdir
            folder.mkdir(parents=True, exist_ok=True)
            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)
    log("Wiped all output folders.")


# ============================================================
# PROCESS MONEYLINE
# ============================================================

def process_moneyline(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    ML_EDGE = settings["ML_EDGE"]
    cal = settings["CALIBRATION"]["moneyline"]

    ml_df = df.copy()

    ml_df["away_decimal"] = ml_df["away_dk_moneyline_american"].apply(american_to_decimal)
    ml_df["home_decimal"] = ml_df["home_dk_moneyline_american"].apply(american_to_decimal)

    ml_df["away_implied_prob"] = ml_df["away_decimal"].apply(safe_implied_prob)
    ml_df["home_implied_prob"] = ml_df["home_decimal"].apply(safe_implied_prob)

    market_pairs = ml_df.apply(
        lambda r: devig_pair(r["away_implied_prob"], r["home_implied_prob"]),
        axis=1,
    )
    ml_df["away_market_prob"] = market_pairs.apply(lambda t: t[0])
    ml_df["home_market_prob"] = market_pairs.apply(lambda t: t[1])

    raw_home = pd.to_numeric(ml_df["home_prob"], errors="coerce")
    raw_away = pd.to_numeric(ml_df["away_prob"], errors="coerce")
    ml_df["home_model_prob"] = raw_home.apply(lambda p: apply_calibration(p, cal["home"]))
    ml_df["away_model_prob"] = raw_away.apply(lambda p: apply_calibration(p, cal["away"]))

    ml_df["away_fair"] = ml_df["away_model_prob"].apply(
        lambda x: 1 / float(x) if x != "" and pd.notna(x) and float(x) > 0 else ""
    )
    ml_df["home_fair"] = ml_df["home_model_prob"].apply(
        lambda x: 1 / float(x) if x != "" and pd.notna(x) and float(x) > 0 else ""
    )

    ml_df["away_acceptable_decimal_moneyline"] = ml_df["away_fair"].apply(
        lambda x: float(x) * (1 + ML_EDGE) if x != "" else ""
    )
    ml_df["home_acceptable_decimal_moneyline"] = ml_df["home_fair"].apply(
        lambda x: float(x) * (1 + ML_EDGE) if x != "" else ""
    )
    ml_df["away_acceptable_american_moneyline"] = ml_df[
        "away_acceptable_decimal_moneyline"
    ].apply(to_american)
    ml_df["home_acceptable_american_moneyline"] = ml_df[
        "home_acceptable_decimal_moneyline"
    ].apply(to_american)

    out_path = OUTPUT_DIR / league / "moneyline" / f"{date}_{league_upper}_moneyline.csv"
    ml_df.to_csv(out_path, index=False)
    return out_path, len(ml_df)


# ============================================================
# PROCESS TOTALS
# ============================================================

def process_totals(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    TOTAL_EDGE = settings["TOTAL_EDGE"]
    TOTAL_STD = settings["TOTAL_STD"]
    cal = settings["CALIBRATION"]["total"]

    total_df = df.copy()

    over_model_prob = []
    under_model_prob = []
    fair_over = []
    fair_under = []
    acc_over = []
    acc_under = []

    for _, row in total_df.iterrows():
        try:
            T = float(row["total"])
            mean = float(row["total_projected_points"])
        except (ValueError, TypeError):
            over_model_prob.append("")
            under_model_prob.append("")
            fair_over.append("")
            fair_under.append("")
            acc_over.append("")
            acc_under.append("")
            continue

        if not math.isfinite(T) or not math.isfinite(mean):
            over_model_prob.append("")
            under_model_prob.append("")
            fair_over.append("")
            fair_under.append("")
            acc_over.append("")
            acc_under.append("")
            continue

        z = (T - mean) / TOTAL_STD

        if not math.isfinite(z):
            over_model_prob.append("")
            under_model_prob.append("")
            fair_over.append("")
            fair_under.append("")
            acc_over.append("")
            acc_under.append("")
            continue

        raw_under = clamp_probability(norm.cdf(z))
        raw_over = 1.0 - raw_under

        p_over, p_under = apply_complementary_calibration(
            raw_over,
            raw_under,
            cal,
            "over",
            "under",
        )

        if p_over == "" or p_under == "":
            over_model_prob.append("")
            under_model_prob.append("")
            fair_over.append("")
            fair_under.append("")
            acc_over.append("")
            acc_under.append("")
            continue

        p_over = float(p_over)
        p_under = float(p_under)

        if (
            not math.isfinite(p_over)
            or not math.isfinite(p_under)
            or p_over <= 0
            or p_under <= 0
            or not math.isclose(p_over + p_under, 1.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"{league_upper} total probabilities are not complementary: "
                f"over={p_over}, under={p_under}"
            )

        over_model_prob.append(p_over)
        under_model_prob.append(p_under)

        fair_over_dec = 1 / p_over
        fair_under_dec = 1 / p_under
        fair_over.append(fair_over_dec)
        fair_under.append(fair_under_dec)
        acc_over.append(fair_over_dec * (1 + TOTAL_EDGE))
        acc_under.append(fair_under_dec * (1 + TOTAL_EDGE))

    total_df["over_model_prob"] = over_model_prob
    total_df["under_model_prob"] = under_model_prob
    total_df["fair_over"] = fair_over
    total_df["fair_under"] = fair_under
    total_df["acceptable_over"] = acc_over
    total_df["acceptable_under"] = acc_under

    total_df["over_implied_prob"] = total_df[
        "dk_total_over_decimal"
    ].apply(safe_implied_prob)
    total_df["under_implied_prob"] = total_df[
        "dk_total_under_decimal"
    ].apply(safe_implied_prob)

    market_pairs = total_df.apply(
        lambda r: devig_pair(r["over_implied_prob"], r["under_implied_prob"]),
        axis=1,
    )
    total_df["over_market_prob"] = market_pairs.apply(lambda t: t[0])
    total_df["under_market_prob"] = market_pairs.apply(lambda t: t[1])

    out_path = OUTPUT_DIR / league / "total" / f"{date}_{league_upper}_total.csv"
    total_df.to_csv(out_path, index=False)
    return out_path, len(total_df)

# ============================================================
# PROCESS SPREAD
# ============================================================

def process_spread(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    SPREAD_EDGE = settings["SPREAD_EDGE"]
    SPREAD_STD = settings["SPREAD_STD"]
    cal = settings["CALIBRATION"]["spread"]

    spread_df = df.copy()

    home_model_prob = []
    away_model_prob = []
    fair_home = []
    fair_away = []
    acc_home = []
    acc_away = []

    for _, row in spread_df.iterrows():
        try:
            home_points = float(row["home_projected_points"])
            away_points = float(row["away_projected_points"])
            home_line = float(row["home_spread"])
        except (ValueError, TypeError):
            home_model_prob.append("")
            away_model_prob.append("")
            fair_home.append("")
            fair_away.append("")
            acc_home.append("")
            acc_away.append("")
            continue

        if (
            not math.isfinite(home_points)
            or not math.isfinite(away_points)
            or not math.isfinite(home_line)
        ):
            home_model_prob.append("")
            away_model_prob.append("")
            fair_home.append("")
            fair_away.append("")
            acc_home.append("")
            acc_away.append("")
            continue

        mean_margin = home_points - away_points
        cover_threshold = -home_line

        raw_home = 1 - norm.cdf(
            cover_threshold,
            loc=mean_margin,
            scale=SPREAD_STD,
        )

        if not math.isfinite(raw_home):
            home_model_prob.append("")
            away_model_prob.append("")
            fair_home.append("")
            fair_away.append("")
            acc_home.append("")
            acc_away.append("")
            continue

        raw_home = clamp_probability(raw_home)
        raw_away = 1.0 - raw_home

        p_home, p_away = apply_complementary_calibration(
            raw_home,
            raw_away,
            cal,
            "home",
            "away",
        )

        if p_home == "" or p_away == "":
            home_model_prob.append("")
            away_model_prob.append("")
            fair_home.append("")
            fair_away.append("")
            acc_home.append("")
            acc_away.append("")
            continue

        p_home = float(p_home)
        p_away = float(p_away)

        if (
            not math.isfinite(p_home)
            or not math.isfinite(p_away)
            or p_home <= 0
            or p_away <= 0
            or not math.isclose(p_home + p_away, 1.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"{league_upper} spread probabilities are not complementary: "
                f"home={p_home}, away={p_away}"
            )

        home_model_prob.append(p_home)
        away_model_prob.append(p_away)

        fair_home_dec = 1 / p_home
        fair_away_dec = 1 / p_away
        fair_home.append(fair_home_dec)
        fair_away.append(fair_away_dec)
        acc_home.append(fair_home_dec * (1 + SPREAD_EDGE))
        acc_away.append(fair_away_dec * (1 + SPREAD_EDGE))

    spread_df["home_spread_model_prob"] = home_model_prob
    spread_df["away_spread_model_prob"] = away_model_prob
    spread_df["fair_home_spread_decimal"] = fair_home
    spread_df["fair_away_spread_decimal"] = fair_away
    spread_df["home_acceptable_spread_decimal"] = acc_home
    spread_df["away_acceptable_spread_decimal"] = acc_away
    spread_df["home_acceptable_spread_american"] = spread_df[
        "home_acceptable_spread_decimal"
    ].apply(to_american)
    spread_df["away_acceptable_spread_american"] = spread_df[
        "away_acceptable_spread_decimal"
    ].apply(to_american)

    spread_df["home_spread_implied_prob"] = spread_df[
        "home_dk_spread_decimal"
    ].apply(safe_implied_prob)
    spread_df["away_spread_implied_prob"] = spread_df[
        "away_dk_spread_decimal"
    ].apply(safe_implied_prob)

    market_pairs = spread_df.apply(
        lambda r: devig_pair(
            r["home_spread_implied_prob"],
            r["away_spread_implied_prob"],
        ),
        axis=1,
    )
    spread_df["home_spread_market_prob"] = market_pairs.apply(lambda t: t[0])
    spread_df["away_spread_market_prob"] = market_pairs.apply(lambda t: t[1])

    out_path = OUTPUT_DIR / league / "spread" / f"{date}_{league_upper}_spread.csv"
    spread_df.to_csv(out_path, index=False)
    return out_path, len(spread_df)

# ============================================================
# MAIN
# ============================================================

def main():
    files_written = []
    files_skipped = 0

    try:
        wipe_outputs()
        log(f"MODEL_CONFIG: {CONFIG_PATH}")

        for league in LEAGUES:
            league_upper = league.upper()
            settings = LEAGUE_SETTINGS[league_upper]

            log(
                f"{league_upper} SETTINGS | ML_EDGE={settings['ML_EDGE']} "
                f"SPREAD_EDGE={settings['SPREAD_EDGE']} TOTAL_EDGE={settings['TOTAL_EDGE']} "
                f"SPREAD_STD={settings['SPREAD_STD']} TOTAL_STD={settings['TOTAL_STD']}"
            )

            for market_type in ["moneyline", "spread", "total"]:
                input_folder = INPUT_DIR / league / market_type

                if not input_folder.exists():
                    log(f"INPUT FOLDER NOT FOUND: {input_folder}")
                    continue

                input_files = sorted(
                    input_folder.glob(f"*_{league_upper}_{market_type}.csv")
                )

                if not input_files:
                    log(f"NO INPUT FILES: {input_folder}")
                    continue

                for file_path in input_files:
                    try:
                        df = pd.read_csv(file_path)

                        if df.empty:
                            log(f"EMPTY: {file_path.name} — skipping")
                            files_skipped += 1
                            continue

                        date = file_path.stem.replace(
                            f"_{league_upper}_{market_type}",
                            "",
                        )

                        if market_type == "moneyline":
                            out_path, count = process_moneyline(
                                df,
                                date,
                                league_upper,
                                settings,
                                league,
                            )

                        elif market_type == "total":
                            out_path, count = process_totals(
                                df,
                                date,
                                league_upper,
                                settings,
                                league,
                            )

                        elif market_type == "spread":
                            out_path, count = process_spread(
                                df,
                                date,
                                league_upper,
                                settings,
                                league,
                            )

                        files_written.append((str(out_path), count))
                        log(f"WROTE {out_path.name} ({count} rows)")
                        audit(
                            market_type.upper(),
                            "SUCCESS",
                            file_path.name,
                            df,
                        )

                    except Exception as e:
                        log(
                            f"ERROR processing {file_path.name}: "
                            f"{e}\n{traceback.format_exc()}"
                        )
                        files_skipped += 1

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")
        log(f"Files skipped: {files_skipped}")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")

        log("STATUS: SUCCESS")
        print("build_juice_files complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# docs/win/basketball/scripts/testing/basketball_backtest.py
#
# Historical replay/backtest runner for the basketball pipeline.
#
# Inputs:
#   docs/win/basketball/backtest/input/*_{NBA|NCAAM|WNBA}.csv
#   docs/win/basketball/backtest/configs/markets_test.yaml
#   docs/win/basketball/config/model_config.yaml
#
# Outputs remain isolated under docs/win/basketball/backtest/.
# Final-score columns are removed before selection and rejoined only for grading.

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import norm

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]
BASKETBALL_ROOT = Path("docs/win/basketball")
DEFAULT_BACKTEST_DIR = BASKETBALL_ROOT / "backtest"
DEFAULT_MODEL_CONFIG = BASKETBALL_ROOT / "config" / "model_config.yaml"
RESULT_COLUMNS = ["home_score", "away_score", "actual_total", "actual_home_spread", "actual_away_spread"]
REQUIRED_INPUT_COLUMNS = {
    "game_date", "game_id", "home_team", "away_team",
    "home_spread", "away_spread", "total",
    "home_dk_moneyline_american", "away_dk_moneyline_american",
    "home_dk_spread_american", "away_dk_spread_american",
    "dk_total_over_american", "dk_total_under_american",
    "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    "home_dk_spread_decimal", "away_dk_spread_decimal",
    "dk_total_over_decimal", "dk_total_under_decimal",
    "home_prob", "away_prob", "home_projected_points", "away_projected_points",
    "total_projected_points", "home_score", "away_score",
}
DEBUG_COUNTS: Counter = Counter()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def sanitize_run_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value.strip()).strip("._")
    if not cleaned:
        raise ValueError("run name becomes empty after sanitization")
    return cleaned


def ensure_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def fv(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"=== basketball_backtest RUN {now_utc()} ===\n", encoding="utf-8")

    def log(self, msg: str, level: str = "INFO") -> None:
        line = f"{now_utc()} | {level:<5} | {msg.rstrip()}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------- MODEL CONFIG ----------------

def calibration_cfg(league_cfg: dict, market: str, side: str) -> dict:
    cfg = (((league_cfg.get("calibration") or {}).get(market) or {}).get(side) or {"method": "none"})
    if isinstance(cfg, str):
        cfg = {"method": cfg}
    if not isinstance(cfg, dict):
        raise ValueError(f"calibration.{market}.{side} must be a mapping")
    return cfg


def complementary_calibration_cfg(
    league_cfg: dict,
    market: str,
    first_side: str,
    second_side: str,
) -> dict:
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
    leagues_cfg = ensure_mapping(model_cfg.get("leagues"), "model_config.leagues")
    settings = {}
    for league in LEAGUES:
        league_cfg = ensure_mapping(leagues_cfg.get(league), f"model_config.leagues.{league}")
        if str(league_cfg.get("status", "")).strip().lower() != "active":
            raise ValueError(f"League {league.upper()} is not active in model_config.yaml")
        edge_cfg = ensure_mapping(league_cfg.get("edge") or {}, f"{league}.edge")
        std_cfg = ensure_mapping(league_cfg.get("std") or {}, f"{league}.std")
        spread_std_cfg = ensure_mapping(std_cfg.get("spread") or {}, f"{league}.std.spread")
        total_std_cfg = ensure_mapping(std_cfg.get("total") or {}, f"{league}.std.total")
        if str(spread_std_cfg.get("mode", "")).strip().lower() != "fixed":
            raise ValueError(f"{league.upper()} spread STD mode must be fixed")
        if str(total_std_cfg.get("mode", "")).strip().lower() != "fixed":
            raise ValueError(f"{league.upper()} total STD mode must be fixed")
        settings[league] = {
            "ML_EDGE": require_number(edge_cfg.get("moneyline"), f"{league}.edge.moneyline"),
            "SPREAD_EDGE": require_number(edge_cfg.get("spread"), f"{league}.edge.spread"),
            "TOTAL_EDGE": require_number(edge_cfg.get("total"), f"{league}.edge.total"),
            "SPREAD_STD": require_number(spread_std_cfg.get("value"), f"{league}.std.spread.value"),
            "TOTAL_STD": require_number(total_std_cfg.get("value"), f"{league}.std.total.value"),
            "CALIBRATION": {
                "moneyline": {
                    "home": calibration_cfg(league_cfg, "moneyline", "home"),
                    "away": calibration_cfg(league_cfg, "moneyline", "away"),
                },
                "spread": complementary_calibration_cfg(
                    league_cfg,
                    "spread",
                    "home",
                    "away",
                ),
                "total": complementary_calibration_cfg(
                    league_cfg,
                    "total",
                    "over",
                    "under",
                ),
            },
        }
    return settings

def apply_calibration(p: Any, cfg: dict) -> float | str:
    if p is None or pd.isna(p):
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    method = str((cfg or {}).get("method", "none")).strip().lower()
    if method in {"none", "raw", ""}:
        return p
    if method == "beta":
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        z = (
            require_number(cfg.get("intercept"), "beta.intercept")
            + require_number(cfg.get("coef_log_p"), "beta.coef_log_p") * math.log(p)
            + require_number(cfg.get("coef_log_1mp"), "beta.coef_log_1mp") * math.log(1.0 - p)
        )
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)
    raise ValueError(f"Unsupported calibration method: {method!r}")


def apply_complementary_calibration(
    raw_first: Any,
    raw_second: Any,
    calibration: dict,
    first_side: str,
    second_side: str,
) -> tuple[float | str, float | str]:
    canonical_side = str(calibration["canonical_side"]).strip().lower()
    raw_canonical = raw_first if canonical_side == first_side else raw_second
    calibrated = apply_calibration(raw_canonical, calibration["config"])

    if calibrated == "" or pd.isna(calibrated):
        return "", ""

    p_canonical = clamp_probability(float(calibrated))
    p_opposite = 1.0 - p_canonical

    if canonical_side == first_side:
        return p_canonical, p_opposite
    if canonical_side == second_side:
        return p_opposite, p_canonical

    raise ValueError(
        f"Unsupported canonical side {canonical_side!r}; "
        f"expected {first_side!r} or {second_side!r}"
    )

def american_to_decimal(odds: Any) -> float | str:
    if odds is None or pd.isna(odds) or str(odds).strip() == "":
        return ""
    try:
        a = float(odds)
    except (TypeError, ValueError):
        return ""
    if a == 0:
        return ""
    return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))


def american_to_decimal_or_none(odds: Any) -> float | None:
    v = american_to_decimal(odds)
    return None if v == "" else float(v)


def to_american(decimal_value: Any) -> str:
    if decimal_value is None or decimal_value == "" or pd.isna(decimal_value):
        return ""
    try:
        dec = float(decimal_value)
    except (TypeError, ValueError):
        return ""
    if dec <= 1:
        return ""
    return f"+{int((dec - 1) * 100)}" if dec >= 2 else f"-{int(100 / (dec - 1))}"


def clamp_probability(p: Any) -> float:
    return min(max(float(p), 0.01), 0.99)


def safe_implied_prob(decimal_value: Any) -> float | str:
    if decimal_value is None or decimal_value == "" or pd.isna(decimal_value):
        return ""
    try:
        d = float(decimal_value)
    except (TypeError, ValueError):
        return ""
    return "" if d <= 0 else 1.0 / d


def devig_pair(p_a: Any, p_b: Any) -> tuple[float | str, float | str]:
    if p_a == "" or p_b == "" or pd.isna(p_a) or pd.isna(p_b):
        return "", ""
    try:
        a, b = float(p_a), float(p_b)
    except (TypeError, ValueError):
        return "", ""
    s = a + b
    return ("", "") if s <= 0 else (a / s, b / s)


# ---------------- INPUT ----------------

def validate_historical_input(df: pd.DataFrame, path: Path, expected_league: str) -> None:
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{path.name} is empty")
    blank_ids = df["game_id"].isna() | (df["game_id"].astype(str).str.strip() == "")
    if blank_ids.any():
        raise ValueError(f"{path.name} contains blank game_id values")
    if "league" in df.columns:
        seen = {str(x).strip().lower() for x in df["league"].dropna().unique() if str(x).strip()}
        if seen and seen != {expected_league}:
            raise ValueError(f"{path.name} league values {sorted(seen)} do not match expected {expected_league}")


def split_features_and_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [c for c in RESULT_COLUMNS if c in df.columns]
    scores = df[["game_id", *score_cols]].copy().drop_duplicates(subset=["game_id"], keep="last")
    features = df.drop(columns=score_cols, errors="ignore").copy()
    return features, scores


# ---------------- JUICE ----------------

def process_moneyline_juice(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = df.copy()
    edge = settings["ML_EDGE"]
    cal = settings["CALIBRATION"]["moneyline"]
    out["away_decimal"] = out["away_dk_moneyline_american"].apply(american_to_decimal)
    out["home_decimal"] = out["home_dk_moneyline_american"].apply(american_to_decimal)
    out["away_implied_prob"] = out["away_decimal"].apply(safe_implied_prob)
    out["home_implied_prob"] = out["home_decimal"].apply(safe_implied_prob)
    pairs = out.apply(lambda r: devig_pair(r["away_implied_prob"], r["home_implied_prob"]), axis=1)
    out["away_market_prob"] = pairs.apply(lambda x: x[0])
    out["home_market_prob"] = pairs.apply(lambda x: x[1])
    out["home_model_prob"] = pd.to_numeric(out["home_prob"], errors="coerce").apply(lambda p: apply_calibration(p, cal["home"]))
    out["away_model_prob"] = pd.to_numeric(out["away_prob"], errors="coerce").apply(lambda p: apply_calibration(p, cal["away"]))
    out["away_fair"] = out["away_model_prob"].apply(lambda x: 1.0 / float(x) if x != "" and pd.notna(x) and float(x) > 0 else "")
    out["home_fair"] = out["home_model_prob"].apply(lambda x: 1.0 / float(x) if x != "" and pd.notna(x) and float(x) > 0 else "")
    out["away_acceptable_decimal_moneyline"] = out["away_fair"].apply(lambda x: float(x) * (1.0 + edge) if x != "" else "")
    out["home_acceptable_decimal_moneyline"] = out["home_fair"].apply(lambda x: float(x) * (1.0 + edge) if x != "" else "")
    out["away_acceptable_american_moneyline"] = out["away_acceptable_decimal_moneyline"].apply(to_american)
    out["home_acceptable_american_moneyline"] = out["home_acceptable_decimal_moneyline"].apply(to_american)
    return out


def process_total_juice(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = df.copy()
    edge, std = settings["TOTAL_EDGE"], settings["TOTAL_STD"]
    cal = settings["CALIBRATION"]["total"]
    vals = {k: [] for k in ["over_model_prob", "under_model_prob", "fair_over", "fair_under", "acceptable_over", "acceptable_under"]}

    for _, row in out.iterrows():
        line, mean = fv(row.get("total")), fv(row.get("total_projected_points"))
        if line is None or mean is None:
            for k in vals:
                vals[k].append("")
            continue

        raw_under = clamp_probability(norm.cdf((line - mean) / std))
        raw_over = 1.0 - raw_under

        p_over, p_under = apply_complementary_calibration(
            raw_over,
            raw_under,
            cal,
            "over",
            "under",
        )

        if p_over == "" or p_under == "":
            for k in vals:
                vals[k].append("")
            continue

        p_over = float(p_over)
        p_under = float(p_under)

        if not math.isclose(p_over + p_under, 1.0, abs_tol=1e-12):
            raise ValueError(
                f"Total probabilities are not complementary: "
                f"over={p_over}, under={p_under}"
            )

        fo, fu = 1.0 / p_over, 1.0 / p_under
        vals["over_model_prob"].append(p_over)
        vals["under_model_prob"].append(p_under)
        vals["fair_over"].append(fo)
        vals["fair_under"].append(fu)
        vals["acceptable_over"].append(fo * (1.0 + edge))
        vals["acceptable_under"].append(fu * (1.0 + edge))

    for k, v in vals.items():
        out[k] = v

    out["over_implied_prob"] = out["dk_total_over_decimal"].apply(safe_implied_prob)
    out["under_implied_prob"] = out["dk_total_under_decimal"].apply(safe_implied_prob)
    pairs = out.apply(
        lambda r: devig_pair(r["over_implied_prob"], r["under_implied_prob"]),
        axis=1,
    )
    out["over_market_prob"] = pairs.apply(lambda x: x[0])
    out["under_market_prob"] = pairs.apply(lambda x: x[1])
    return out

def process_spread_juice(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = df.copy()
    edge, std = settings["SPREAD_EDGE"], settings["SPREAD_STD"]
    cal = settings["CALIBRATION"]["spread"]
    vals = {k: [] for k in ["home_spread_model_prob", "away_spread_model_prob", "fair_home_spread_decimal", "fair_away_spread_decimal", "home_acceptable_spread_decimal", "away_acceptable_spread_decimal"]}

    for _, row in out.iterrows():
        hp, ap, line = (
            fv(row.get("home_projected_points")),
            fv(row.get("away_projected_points")),
            fv(row.get("home_spread")),
        )
        if hp is None or ap is None or line is None:
            for k in vals:
                vals[k].append("")
            continue

        raw_home = clamp_probability(
            1.0 - norm.cdf(-line, loc=hp - ap, scale=std)
        )
        raw_away = 1.0 - raw_home

        p_home, p_away = apply_complementary_calibration(
            raw_home,
            raw_away,
            cal,
            "home",
            "away",
        )

        if p_home == "" or p_away == "":
            for k in vals:
                vals[k].append("")
            continue

        p_home = float(p_home)
        p_away = float(p_away)

        if not math.isclose(p_home + p_away, 1.0, abs_tol=1e-12):
            raise ValueError(
                f"Spread probabilities are not complementary: "
                f"home={p_home}, away={p_away}"
            )

        fh, fa = 1.0 / p_home, 1.0 / p_away
        vals["home_spread_model_prob"].append(p_home)
        vals["away_spread_model_prob"].append(p_away)
        vals["fair_home_spread_decimal"].append(fh)
        vals["fair_away_spread_decimal"].append(fa)
        vals["home_acceptable_spread_decimal"].append(fh * (1.0 + edge))
        vals["away_acceptable_spread_decimal"].append(fa * (1.0 + edge))

    for k, v in vals.items():
        out[k] = v

    out["home_acceptable_spread_american"] = out["home_acceptable_spread_decimal"].apply(to_american)
    out["away_acceptable_spread_american"] = out["away_acceptable_spread_decimal"].apply(to_american)
    out["home_spread_implied_prob"] = out["home_dk_spread_decimal"].apply(safe_implied_prob)
    out["away_spread_implied_prob"] = out["away_dk_spread_decimal"].apply(safe_implied_prob)
    pairs = out.apply(
        lambda r: devig_pair(r["home_spread_implied_prob"], r["away_spread_implied_prob"]),
        axis=1,
    )
    out["home_spread_market_prob"] = pairs.apply(lambda x: x[0])
    out["away_spread_market_prob"] = pairs.apply(lambda x: x[1])
    return out

def compute_ev(model_prob: Any, book_decimal: Any) -> float | None:
    p, d = fv(model_prob), fv(book_decimal)
    return None if p is None or d is None else p * d - 1.0


def compute_kelly(model_prob: Any, book_decimal: Any) -> float | None:
    p, d = fv(model_prob), fv(book_decimal)
    if p is None or d is None or d <= 1.0:
        return None
    b = d - 1.0
    k = ((b * p) - (1.0 - p)) / b
    return None if not math.isfinite(k) else max(k, 0.0)


def process_moneyline_ev(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ("home", "away"):
        out[f"{side}_ml_ev"] = out.apply(lambda r, s=side: compute_ev(r.get(f"{s}_model_prob"), r.get(f"{s}_dk_moneyline_decimal")), axis=1)
        out[f"{side}_ml_edge_vs_market"] = pd.to_numeric(out[f"{side}_model_prob"], errors="coerce") - pd.to_numeric(out[f"{side}_market_prob"], errors="coerce")
        out[f"{side}_ml_kelly"] = out.apply(lambda r, s=side: compute_kelly(r.get(f"{s}_model_prob"), r.get(f"{s}_dk_moneyline_decimal")), axis=1)
        out[f"{side}_ml_ev_pct"] = out[f"{side}_ml_ev"] * 100.0
        out[f"{side}_ml_edge_vs_market_pct"] = out[f"{side}_ml_edge_vs_market"] * 100.0
    return out


def process_spread_ev(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ("home", "away"):
        out[f"{side}_spread_ev"] = out.apply(lambda r, s=side: compute_ev(r.get(f"{s}_spread_model_prob"), r.get(f"{s}_dk_spread_decimal")), axis=1)
        out[f"{side}_spread_edge_vs_market"] = pd.to_numeric(out[f"{side}_spread_model_prob"], errors="coerce") - pd.to_numeric(out[f"{side}_spread_market_prob"], errors="coerce")
        out[f"{side}_spread_kelly"] = out.apply(lambda r, s=side: compute_kelly(r.get(f"{s}_spread_model_prob"), r.get(f"{s}_dk_spread_decimal")), axis=1)
        out[f"{side}_spread_ev_pct"] = out[f"{side}_spread_ev"] * 100.0
        out[f"{side}_spread_edge_vs_market_pct"] = out[f"{side}_spread_edge_vs_market"] * 100.0
    return out


def process_total_ev(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ("over", "under"):
        out[f"{side}_ev"] = out.apply(lambda r, s=side: compute_ev(r.get(f"{s}_model_prob"), r.get(f"dk_total_{s}_decimal")), axis=1)
        out[f"{side}_edge_vs_market"] = pd.to_numeric(out[f"{side}_model_prob"], errors="coerce") - pd.to_numeric(out[f"{side}_market_prob"], errors="coerce")
        out[f"{side}_kelly"] = out.apply(lambda r, s=side: compute_kelly(r.get(f"{s}_model_prob"), r.get(f"dk_total_{s}_decimal")), axis=1)
        out[f"{side}_ev_pct"] = out[f"{side}_ev"] * 100.0
        out[f"{side}_edge_vs_market_pct"] = out[f"{side}_edge_vs_market"] * 100.0
    return out


# ---------------- SELECTION ----------------

def in_any_band(value: float | None, bands: Any) -> bool:
    if value is None or bands is None:
        return False
    try:
        return any(float(lo) <= value <= float(hi) for lo, hi in bands)
    except Exception:
        return False


def parse_game_date(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    for fmt in ("%Y_%m_%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def date_ok(game_date: Any, months: list, exclude_dow: list) -> bool:
    if not months and not exclude_dow:
        return True
    dt = parse_game_date(game_date)
    if dt is None:
        return True
    if months and dt.month not in months:
        DEBUG_COUNTS["fail_month"] += 1
        return False
    if exclude_dow and dt.weekday() in exclude_dow:
        DEBUG_COUNTS["fail_dow"] += 1
        return False
    return True


def passes_filters(values: dict, side_cfg: dict, game_date: Any) -> bool:
    if "odds_bands" in side_cfg and not in_any_band(values.get("odds"), side_cfg["odds_bands"]):
        DEBUG_COUNTS["fail_odds"] += 1
        return False
    if "line_bands" in side_cfg and values.get("line") is not None and not in_any_band(values.get("line"), side_cfg["line_bands"]):
        DEBUG_COUNTS["fail_line"] += 1
        return False
    if "ev_bands" in side_cfg and not in_any_band(values.get("ev"), side_cfg["ev_bands"]):
        DEBUG_COUNTS["fail_ev"] += 1
        return False
    if "kelly_bands" in side_cfg and not in_any_band(values.get("kelly"), side_cfg["kelly_bands"]):
        DEBUG_COUNTS["fail_kelly"] += 1
        return False
    if "model_prob_bands" in side_cfg and not in_any_band(values.get("model_prob"), side_cfg["model_prob_bands"]):
        DEBUG_COUNTS["fail_model_prob"] += 1
        return False
    if "edge_vs_market_bands" in side_cfg and not in_any_band(values.get("edge_vs_market_pct"), side_cfg["edge_vs_market_bands"]):
        DEBUG_COUNTS["fail_edge_vs_market"] += 1
        return False
    return date_ok(game_date, side_cfg.get("months", []) or [], side_cfg.get("exclude_days_of_week", []) or [])


def model_edge_threshold(settings: dict, league: str, market: str) -> float:
    return float(settings[league][{"moneyline": "ML_EDGE", "spread": "SPREAD_EDGE", "total": "TOTAL_EDGE"}[market]])


def passes_model_edge(ev: float | None, settings: dict, league: str, market: str) -> bool:
    if ev is None or ev < model_edge_threshold(settings, league, market):
        DEBUG_COUNTS[f"fail_model_edge_{market}"] += 1
        return False
    return True


def pick_one(qualifying: list[dict], preference: dict) -> dict | None:
    if not qualifying:
        return None
    metric = preference.get("metric", "ev")
    direction = preference.get("direction", "max")
    def key(c):
        v = c.get(metric)
        if v is None:
            return float("-inf") if direction == "max" else float("inf")
        return float(v)
    return max(qualifying, key=key) if direction == "max" else min(qualifying, key=key)


def stake_pct(kelly: float | None, fraction: float, cap: float) -> float | None:
    if kelly is None or kelly <= 0:
        return None
    return min(kelly * fraction, cap)


def market_config(filter_cfg: dict, league: str, market: str) -> dict:
    try:
        cfg = filter_cfg["markets"][league][market]
    except KeyError as exc:
        raise KeyError(f"No test config for league={league} market={market}") from exc
    return ensure_mapping(cfg, f"markets.{league}.{market}")


def build_moneyline_sides(row, league, game_date, cfg, settings):
    sides = []
    for side in ("home", "away"):
        scfg = ensure_mapping(cfg.get(side), f"markets.{league}.moneyline.{side}")
        if not scfg.get("enabled", True):
            continue
        odds = fv(row.get(f"{side}_dk_moneyline_american"))
        ev = fv(row.get(f"{side}_ml_ev"))
        kelly = fv(row.get(f"{side}_ml_kelly"))
        mp = fv(row.get(f"{side}_model_prob"))
        mp = mp if mp is not None else fv(row.get(f"{side}_prob"))
        evm = fv(row.get(f"{side}_ml_edge_vs_market_pct"))
        if not passes_model_edge(ev, settings, league, "moneyline"):
            DEBUG_COUNTS["rejected_ml"] += 1
            continue
        vals = {"odds": odds, "ev": ev, "kelly": kelly, "model_prob": mp, "edge_vs_market_pct": evm}
        if passes_filters(vals, scfg, game_date):
            sides.append({"side": side, "line": odds, "odds": odds, "ev": ev, "kelly": kelly, "model_prob": mp, "edge_vs_market": evm})
        else:
            DEBUG_COUNTS["rejected_ml"] += 1
    return sides


def build_spread_sides(row, league, game_date, cfg, settings):
    sides = []
    for side in ("home", "away"):
        scfg = ensure_mapping(cfg.get(side), f"markets.{league}.spread.{side}")
        if not scfg.get("enabled", True):
            continue
        line = fv(row.get(f"{side}_spread"))
        odds = fv(row.get(f"{side}_dk_spread_american"))
        ev = fv(row.get(f"{side}_spread_ev"))
        kelly = fv(row.get(f"{side}_spread_kelly"))
        mp = fv(row.get(f"{side}_spread_model_prob"))
        evm = fv(row.get(f"{side}_spread_edge_vs_market_pct"))
        if not passes_model_edge(ev, settings, league, "spread"):
            DEBUG_COUNTS["rejected_spread"] += 1
            continue
        vals = {"odds": odds, "line": line, "ev": ev, "kelly": kelly, "model_prob": mp, "edge_vs_market_pct": evm}
        if passes_filters(vals, scfg, game_date):
            sides.append({"side": side, "line": line, "odds": odds, "ev": ev, "kelly": kelly, "model_prob": mp, "edge_vs_market": evm})
        else:
            DEBUG_COUNTS["rejected_spread"] += 1
    return sides


def build_total_sides(row, league, game_date, cfg, settings):
    sides, line = [], fv(row.get("total"))
    for side in ("over", "under"):
        scfg = ensure_mapping(cfg.get(side), f"markets.{league}.total.{side}")
        if not scfg.get("enabled", True):
            continue
        odds = fv(row.get(f"dk_total_{side}_american"))
        ev = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp = fv(row.get(f"{side}_model_prob"))
        evm = fv(row.get(f"{side}_edge_vs_market_pct"))
        if not passes_model_edge(ev, settings, league, "total"):
            DEBUG_COUNTS["rejected_total"] += 1
            continue
        vals = {"odds": odds, "line": line, "ev": ev, "kelly": kelly, "model_prob": mp, "edge_vs_market_pct": evm}
        if passes_filters(vals, scfg, game_date):
            sides.append({"side": side, "line": line, "odds": odds, "ev": ev, "kelly": kelly, "model_prob": mp, "edge_vs_market": evm})
        else:
            DEBUG_COUNTS["rejected_total"] += 1
    return sides


SIDE_BUILDERS = {"moneyline": build_moneyline_sides, "spread": build_spread_sides, "total": build_total_sides}


def select_bets_for_market(df, league, market, filter_cfg, settings, kelly_fraction, kelly_cap):
    cfg = market_config(filter_cfg, league, market)
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    mode = str(cfg.get("selection_mode", "pick_one")).strip().lower()
    preference = cfg.get("pick_preference") or {"metric": "ev", "direction": "max"}
    out_rows = []
    for _, row in df.iterrows():
        game_date = row.get("game_date")
        sides = SIDE_BUILDERS[market](row, league, game_date, cfg, settings)
        if not sides:
            continue
        if mode == "all_qualifying":
            picks = sides
        else:
            chosen = pick_one(sides, preference)
            picks = [chosen] if chosen else []
        for sel in picks:
            DEBUG_COUNTS["selected"] += 1
            r = row.to_dict()
            r.update({
                "bet_side": sel["side"], "bet_line": sel["line"], "bet_odds_american": sel["odds"],
                "bet_ev": sel["ev"], "bet_kelly": sel["kelly"], "bet_model_prob": sel["model_prob"],
                "bet_edge_vs_market": sel["edge_vs_market"], "bet_stake_pct": stake_pct(sel["kelly"], kelly_fraction, kelly_cap),
                "market_type": market, "league_lower": league, "league": league.upper(), "game_date": game_date,
            })
            out_rows.append(r)
    return pd.DataFrame(out_rows)


# ---------------- GRADING ----------------

def determine_outcome(row) -> str:
    market, side = str(row.get("market_type", "")).lower(), str(row.get("bet_side", "")).lower()
    home, away = fv(row.get("home_score")), fv(row.get("away_score"))
    if home is None or away is None:
        return "Unknown"
    if market == "moneyline":
        if home == away:
            return "Push"
        home_won = home > away
        return "Win" if (side == "home" and home_won) or (side == "away" and not home_won) else "Loss"
    if market == "spread":
        line = fv(row.get("bet_line"))
        if line is None:
            return "Unknown"
        if side == "home":
            diff = home + line - away
        elif side == "away":
            diff = away + line - home
        else:
            return "Unknown"
        return "Push" if abs(diff) < 1e-9 else ("Win" if diff > 0 else "Loss")
    if market == "total":
        line = fv(row.get("bet_line"))
        if line is None:
            return "Unknown"
        total = home + away
        if abs(total - line) < 1e-9:
            return "Push"
        return "Win" if (total > line and side == "over") or (total < line and side == "under") else "Loss"
    return "Unknown"


def compute_profits(row):
    result = str(row.get("bet_result", "")).strip()
    decimal = american_to_decimal_or_none(row.get("bet_odds_american"))
    if result == "Push":
        return 0.0, 0.0
    if result not in ("Win", "Loss") or decimal is None or decimal <= 1:
        return None, None
    stake = fv(row.get("bet_stake_pct"))
    if result == "Win":
        return decimal - 1.0, (stake * (decimal - 1.0) if stake is not None else None)
    return -1.0, (-stake if stake is not None else None)


def grade_selections(selections: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    merged = selections.merge(scores, on="game_id", how="left", suffixes=("", "_score"))
    for col in RESULT_COLUMNS:
        sc = f"{col}_score"
        if sc in merged.columns:
            merged[col] = merged[sc].combine_first(merged[col]) if col in merged.columns else merged[sc]
            merged = merged.drop(columns=[sc])
    merged["bet_result"] = merged.apply(determine_outcome, axis=1)
    profits = merged.apply(compute_profits, axis=1, result_type="expand")
    profits.columns = ["profit_unit", "profit_kelly"]
    merged = pd.concat([merged, profits], axis=1)
    keys = [c for c in ["source_file", "game_id", "market_type", "bet_side"] if c in merged.columns]
    return merged.drop_duplicates(subset=keys, keep="last") if keys else merged


# ---------------- SIMPLE REPORTS ----------------

def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [*group_cols, "bets", "wins", "losses", "pushes", "unknown", "win_rate", "profit_units", "roi_units", "profit_kelly", "kelly_staked", "roi_kelly", "avg_ev", "avg_kelly", "avg_model_prob", "avg_edge_vs_market", "avg_odds_american"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    records = []
    grouped = df.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), df)]
    for keys, group in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        elif not group_cols:
            keys = ()
        wins = int((group["bet_result"] == "Win").sum())
        losses = int((group["bet_result"] == "Loss").sum())
        pushes = int((group["bet_result"] == "Push").sum())
        unknown = int((group["bet_result"] == "Unknown").sum())
        bets, decisions, graded_stakes = wins + losses + pushes + unknown, wins + losses, wins + losses + pushes
        pu = float(pd.to_numeric(group["profit_unit"], errors="coerce").sum(skipna=True))
        pk = float(pd.to_numeric(group["profit_kelly"], errors="coerce").sum(skipna=True))
        ks = float(pd.to_numeric(group["bet_stake_pct"], errors="coerce").fillna(0.0).sum())
        rec = {c: v for c, v in zip(group_cols, keys)}
        rec.update({
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes, "unknown": unknown,
            "win_rate": wins / decisions if decisions else None, "profit_units": pu, "roi_units": pu / graded_stakes if graded_stakes else None,
            "profit_kelly": pk, "kelly_staked": ks, "roi_kelly": pk / ks if ks > 0 else None,
            "avg_ev": pd.to_numeric(group["bet_ev"], errors="coerce").mean(), "avg_kelly": pd.to_numeric(group["bet_kelly"], errors="coerce").mean(),
            "avg_model_prob": pd.to_numeric(group["bet_model_prob"], errors="coerce").mean(), "avg_edge_vs_market": pd.to_numeric(group["bet_edge_vs_market"], errors="coerce").mean(),
            "avg_odds_american": pd.to_numeric(group["bet_odds_american"], errors="coerce").mean(),
        })
        records.append(rec)
    return pd.DataFrame(records, columns=cols)


def build_reports(graded: pd.DataFrame, reports_dir: Path) -> dict[str, pd.DataFrame]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "overall": summarize_group(graded, []),
        "by_source": summarize_group(graded, ["source_file", "league"]),
        "by_league": summarize_group(graded, ["league"]),
        "by_market": summarize_group(graded, ["league", "market_type"]),
        "by_market_side": summarize_group(graded, ["league", "market_type", "bet_side"]),
    }
    names = {"overall": "overall.csv", "by_source": "performance_by_source.csv", "by_league": "performance_by_league.csv", "by_market": "performance_by_market.csv", "by_market_side": "performance_by_market_side.csv"}
    for name, report in reports.items():
        atomic_write_csv(report, reports_dir / names[name])
    atomic_write_csv(pd.DataFrame([{"reason": k, "count": v} for k, v in sorted(DEBUG_COUNTS.items())]), reports_dir / "filter_counts.csv")
    return reports


# ---------------- PRODUCTION-STYLE BUCKET REPORTS ----------------

DOW_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
CANON_COLS_NO_SIDE = ["league", "market_type", "bucket_dimension", "bucket", "bets", "wins", "losses", "pushes", "total", "win_pct", "units_flat", "roi_flat", "units_kelly", "roi_kelly", "avg_ev", "avg_edge_vs_market_pp", "avg_kelly_pct", "avg_model_prob", "avg_odds_american"]
CANON_COLS_WITH_SIDE = ["league", "market_type", "side_group", "bucket_dimension", "bucket", "bets", "wins", "losses", "pushes", "total", "win_pct", "units_flat", "roi_flat", "units_kelly", "roi_kelly", "avg_ev", "avg_edge_vs_market_pp", "avg_kelly_pct", "avg_model_prob", "avg_odds_american"]
COMMON_BUCKETS = [("ev", "ev_bucket"), ("kelly", "kelly_bucket"), ("odds", "odds_bucket"), ("win_prob", "model_prob_bucket"), ("edge_vs_market", "edge_vs_market_bucket"), ("dow", "dow_bucket"), ("month", "month_bucket")]
CROSS_DIMS = [("ev", "ev_bucket"), ("kelly", "kelly_bucket"), ("odds", "odds_bucket"), ("win_prob", "model_prob_bucket"), ("edge_vs_market", "edge_vs_market_bucket"), ("dow", "dow_bucket"), ("month", "month_bucket"), ("side", "side_group")]
CROSS_COLS = ["league", "market_type", "dimension_1", "bucket_1", "dimension_2", "bucket_2", "bets", "wins", "losses", "pushes", "total", "win_pct", "units_flat", "roi_flat", "units_kelly", "roi_kelly", "avg_ev", "avg_edge_vs_market_pp", "avg_kelly_pct", "avg_model_prob", "avg_odds_american"]


def side_group(row):
    mt, side = str(row.get("market_type", "")).strip().lower(), str(row.get("bet_side", "")).strip().lower()
    if mt in {"moneyline", "spread"}:
        return "HOME" if side == "home" else "AWAY" if side == "away" else ""
    if mt == "total":
        return "OVER" if side == "over" else "UNDER" if side == "under" else ""
    return ""


def ev_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    if v < 0: return "<0"
    if v < .01: return "0.00_to_0.0099"
    if v < .02: return "0.01_to_0.0199"
    if v < .03: return "0.02_to_0.0299"
    if v < .04: return "0.03_to_0.0399"
    if v < .05: return "0.04_to_0.0499"
    if v < .075: return "0.05_to_0.0749"
    if v < .10: return "0.075_to_0.0999"
    return "0.10_plus"


def edge_vs_market_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    if v < -10: return "below_neg10pp"
    if v < -5: return "neg10_to_neg5pp"
    if v < -2: return "neg5_to_neg2pp"
    if v < 0: return "neg2_to_0pp"
    if v < 1: return "0_to_1pp"
    if v < 2: return "1_to_2pp"
    if v < 3: return "2_to_3pp"
    if v < 5: return "3_to_5pp"
    if v < 7: return "5_to_7pp"
    if v < 10: return "7_to_10pp"
    return "10pp_plus"


def kelly_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    if v < 0: return "<0"
    if v < .025: return "0.0_to_2.5pct"
    if v < .05: return "2.5_to_5pct"
    if v < .10: return "5_to_10pct"
    if v < .20: return "10_to_20pct"
    return "20pct_plus"


def model_prob_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    if v < .20: return "below_0.20"
    if v < .30: return "0.20_to_0.30"
    if v < .40: return "0.30_to_0.40"
    if v < .50: return "0.40_to_0.50"
    if v < .60: return "0.50_to_0.60"
    if v < .70: return "0.60_to_0.70"
    if v < .80: return "0.70_to_0.80"
    if v < .90: return "0.80_to_0.90"
    return "0.90_plus"


def odds_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    if v <= -200: return "minus_200_or_lower"
    if v <= -150: return "minus_199_to_minus_150"
    if v <= -125: return "minus_149_to_minus_125"
    if v <= -110: return "minus_124_to_minus_110"
    if v <= -101: return "minus_109_to_minus_101"
    if v <= 100: return "minus_100_to_plus_100"
    if v <= 125: return "plus_101_to_plus_125"
    if v <= 150: return "plus_126_to_plus_150"
    if v <= 200: return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def spread_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    a = abs(v)
    if a < 1: return "0.0_to_0.9"
    if a < 2: return "1.0_to_1.9"
    if a < 3: return "2.0_to_2.9"
    if a < 4: return "3.0_to_3.9"
    if a < 5: return "4.0_to_4.9"
    if a < 6: return "5.0_to_5.9"
    if a < 7: return "6.0_to_6.9"
    if a < 8: return "7.0_to_7.9"
    if a < 9: return "8.0_to_8.9"
    if a < 10: return "9.0_to_9.9"
    if a < 12: return "10.0_to_11.9"
    if a < 15: return "12.0_to_14.9"
    return "15.0_plus"


def total_bucket(value):
    v = fv(value)
    if v is None: return "UNBUCKETED"
    start = int(v // 5) * 5
    return f"{start}_to_{start + 4.9:.1f}"


def dow_bucket(value):
    dt = parse_game_date(value)
    return "UNBUCKETED" if dt is None else DOW_NAMES[dt.weekday()]


def month_bucket(value):
    dt = parse_game_date(value)
    return "UNBUCKETED" if dt is None else MONTH_NAMES[dt.month - 1]


def prepare_bucket_work(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work.empty:
        return work
    work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    work["bet_side"] = work["bet_side"].astype(str).str.strip().str.lower()
    work["side_group"] = work.apply(side_group, axis=1)
    work["ev_bucket"] = work["bet_ev"].apply(ev_bucket)
    work["edge_vs_market_bucket"] = work["bet_edge_vs_market"].apply(edge_vs_market_bucket)
    work["kelly_bucket"] = work["bet_kelly"].apply(kelly_bucket)
    work["model_prob_bucket"] = work["bet_model_prob"].apply(model_prob_bucket)
    work["odds_bucket"] = work["bet_odds_american"].apply(odds_bucket)
    work["spread_bucket"] = work.apply(lambda r: spread_bucket(r.get("bet_line")) if r["market_type"] == "spread" else "UNBUCKETED", axis=1)
    work["total_bucket"] = work.apply(lambda r: total_bucket(r.get("bet_line")) if r["market_type"] == "total" else "UNBUCKETED", axis=1)
    work["dow_bucket"] = work["game_date"].apply(dow_bucket) if "game_date" in work.columns else "UNBUCKETED"
    work["month_bucket"] = work["game_date"].apply(month_bucket) if "game_date" in work.columns else "UNBUCKETED"
    return work


def aggregate_block(df, league, market_type, bucket_dimension, bucket_col, side_group_col=None):
    cols = CANON_COLS_WITH_SIDE if side_group_col else CANON_COLS_NO_SIDE
    if df.empty:
        return pd.DataFrame(columns=cols)
    work = df.copy()
    for c in ("profit_unit", "profit_kelly", "bet_stake_pct", "bet_ev", "bet_edge_vs_market", "bet_kelly", "bet_model_prob", "bet_odds_american"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    res = work["bet_result"].astype(str).str.strip().str.lower()
    work["_is_win"] = (res == "win").astype(int)
    work["_is_loss"] = (res == "loss").astype(int)
    work["_is_push"] = (res == "push").astype(int)
    group_cols = [bucket_col] if not side_group_col else [side_group_col, bucket_col]
    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        wins, losses, pushes = int(sub["_is_win"].sum()), int(sub["_is_loss"].sum()), int(sub["_is_push"].sum())
        bets = wins + losses + pushes
        uf = float(sub["profit_unit"].sum(skipna=True))
        uk = float(sub["profit_kelly"].sum(skipna=True))
        stake = float(sub["bet_stake_pct"].sum(skipna=True))
        mt = market_type if market_type is not None else (str(sub["market_type"].iloc[0]).lower() if sub["market_type"].nunique() == 1 else "mixed")
        row = {
            "league": league, "market_type": mt, "bucket_dimension": bucket_dimension, "bucket": keys[-1],
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes, "total": bets,
            "win_pct": round(wins / (wins + losses), 4) if wins + losses else math.nan,
            "units_flat": round(uf, 4), "roi_flat": round(uf / bets, 4) if bets else math.nan,
            "units_kelly": round(uk, 6), "roi_kelly": round(uk / stake, 4) if stake else math.nan,
            "avg_ev": round(float(sub["bet_ev"].mean(skipna=True)), 4),
            "avg_edge_vs_market_pp": round(float(sub["bet_edge_vs_market"].mean(skipna=True)), 4),
            "avg_kelly_pct": round(float(sub["bet_kelly"].mean(skipna=True)), 4),
            "avg_model_prob": round(float(sub["bet_model_prob"].mean(skipna=True)), 4),
            "avg_odds_american": round(float(sub["bet_odds_american"].mean(skipna=True)), 1),
        }
        if side_group_col:
            row["side_group"] = keys[0]
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    sort_cols = [c for c in ("side_group", "bucket") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def aggregate_cross(df, league, market_type, d1_label, d1_col, d2_label, d2_col):
    if df.empty or d1_col not in df.columns or d2_col not in df.columns:
        return pd.DataFrame(columns=CROSS_COLS)
    work = df.copy()
    for c in ("profit_unit", "profit_kelly", "bet_stake_pct", "bet_ev", "bet_edge_vs_market", "bet_kelly", "bet_model_prob", "bet_odds_american"):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    res = work["bet_result"].astype(str).str.strip().str.lower()
    work["_is_win"] = (res == "win").astype(int)
    work["_is_loss"] = (res == "loss").astype(int)
    work["_is_push"] = (res == "push").astype(int)
    rows = []
    for (b1, b2), sub in work.groupby([d1_col, d2_col], dropna=False, observed=True):
        wins, losses, pushes = int(sub["_is_win"].sum()), int(sub["_is_loss"].sum()), int(sub["_is_push"].sum())
        bets = wins + losses + pushes
        uf = float(sub["profit_unit"].sum(skipna=True))
        uk = float(sub["profit_kelly"].sum(skipna=True))
        stake = float(sub["bet_stake_pct"].sum(skipna=True))
        rows.append({
            "league": league, "market_type": market_type, "dimension_1": d1_label, "bucket_1": b1, "dimension_2": d2_label, "bucket_2": b2,
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes, "total": bets,
            "win_pct": round(wins / (wins + losses), 4) if wins + losses else math.nan,
            "units_flat": round(uf, 4), "roi_flat": round(uf / bets, 4) if bets else math.nan,
            "units_kelly": round(uk, 6), "roi_kelly": round(uk / stake, 4) if stake else math.nan,
            "avg_ev": round(float(sub["bet_ev"].mean(skipna=True)), 4), "avg_edge_vs_market_pp": round(float(sub["bet_edge_vs_market"].mean(skipna=True)), 4),
            "avg_kelly_pct": round(float(sub["bet_kelly"].mean(skipna=True)), 4), "avg_model_prob": round(float(sub["bet_model_prob"].mean(skipna=True)), 4),
            "avg_odds_american": round(float(sub["bet_odds_american"].mean(skipna=True)), 1),
        })
    return pd.DataFrame(rows, columns=CROSS_COLS)


def side_suffix(market_type: str) -> str:
    return "over_under" if market_type == "total" else "home_away"


def write_bucket_market_reports(work_df, league, market_type, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = work_df[work_df["market_type"] == market_type].copy()
    if sub.empty:
        return
    sfx = side_suffix(market_type)
    for label, bucket_col in COMMON_BUCKETS:
        atomic_write_csv(aggregate_block(sub, league, market_type, label, bucket_col), out_dir / f"{league}_{market_type}_by_{label}.csv")
        atomic_write_csv(aggregate_block(sub, league, market_type, label, bucket_col, "side_group"), out_dir / f"{league}_{market_type}_by_{label}_{sfx}_summary.csv")
    if market_type in {"spread", "total"}:
        agg = aggregate_block(sub, league, market_type, "side", "side_group")
        atomic_write_csv(agg, out_dir / f"{league}_{market_type}_by_side.csv")
        agg_s = agg.copy()
        if not agg_s.empty:
            agg_s.insert(2, "side_group", agg_s["bucket"])
            agg_s = agg_s[CANON_COLS_WITH_SIDE]
        else:
            agg_s = pd.DataFrame(columns=CANON_COLS_WITH_SIDE)
        atomic_write_csv(agg_s, out_dir / f"{league}_{market_type}_by_side_{sfx}_summary.csv")
    if market_type == "total":
        atomic_write_csv(aggregate_block(sub, league, market_type, "total_range", "total_bucket"), out_dir / f"{league}_{market_type}_by_total_range.csv")
        atomic_write_csv(aggregate_block(sub, league, market_type, "total_range", "total_bucket", "side_group"), out_dir / f"{league}_{market_type}_by_total_range_{sfx}_summary.csv")
    if market_type == "spread":
        atomic_write_csv(aggregate_block(sub, league, market_type, "spread_range", "spread_bucket"), out_dir / f"{league}_{market_type}_by_spread_range.csv")
        atomic_write_csv(aggregate_block(sub, league, market_type, "spread_range", "spread_bucket", "side_group"), out_dir / f"{league}_{market_type}_by_spread_range_{sfx}_summary.csv")


def write_bucket_crosses(work_df, league, market_type, out_dir):
    sub = work_df[work_df["market_type"] == market_type].copy()
    pieces = []
    for i in range(len(CROSS_DIMS)):
        for j in range(i + 1, len(CROSS_DIMS)):
            d1, c1 = CROSS_DIMS[i]
            d2, c2 = CROSS_DIMS[j]
            pieces.append(aggregate_cross(sub, league, market_type, d1, c1, d2, c2))
    pieces = [p for p in pieces if not p.empty]
    if pieces:
        atomic_write_csv(pd.concat(pieces, ignore_index=True), out_dir / f"{league}_{market_type}_crosses.csv")


def bucket_summary_overall(work_df, league):
    rows = []
    for mt in MARKETS:
        sub = work_df[work_df["market_type"] == mt]
        res = sub["bet_result"].astype(str).str.strip().str.lower()
        wins, losses, pushes = int((res == "win").sum()), int((res == "loss").sum()), int((res == "push").sum())
        rows.append({"league": league.upper(), "market_type": mt, "Win": wins, "Loss": losses, "Push": pushes, "Total": wins + losses + pushes, "Win_Pct": round(wins / (wins + losses), 4) if wins + losses else math.nan})
    return pd.DataFrame(rows)


def bucket_summary_grand_total(work_df, league):
    if work_df.empty:
        return pd.DataFrame([{
            "league": league.upper(), "bets": 0, "wins": 0, "losses": 0, "pushes": 0, "total": 0,
            "win_pct": math.nan, "units_flat": 0.0, "roi_flat": math.nan,
            "units_kelly": 0.0, "roi_kelly": math.nan, "avg_ev": math.nan,
            "avg_edge_vs_market_pp": math.nan, "avg_kelly_pct": math.nan,
            "avg_model_prob": math.nan, "avg_odds_american": math.nan,
        }])
    res = work_df["bet_result"].astype(str).str.strip().str.lower()
    wins = int((res == "win").sum())
    losses = int((res == "loss").sum())
    pushes = int((res == "push").sum())
    bets = wins + losses + pushes
    units_flat = float(pd.to_numeric(work_df["profit_unit"], errors="coerce").sum(skipna=True))
    units_kelly = float(pd.to_numeric(work_df["profit_kelly"], errors="coerce").sum(skipna=True))
    stake_total = float(pd.to_numeric(work_df["bet_stake_pct"], errors="coerce").sum(skipna=True))
    return pd.DataFrame([{
        "league": league.upper(), "bets": bets, "wins": wins, "losses": losses, "pushes": pushes, "total": bets,
        "win_pct": round(wins / (wins + losses), 4) if wins + losses else math.nan,
        "units_flat": round(units_flat, 4), "roi_flat": round(units_flat / bets, 4) if bets else math.nan,
        "units_kelly": round(units_kelly, 6), "roi_kelly": round(units_kelly / stake_total, 4) if stake_total else math.nan,
        "avg_ev": round(float(pd.to_numeric(work_df["bet_ev"], errors="coerce").mean()), 4),
        "avg_edge_vs_market_pp": round(float(pd.to_numeric(work_df["bet_edge_vs_market"], errors="coerce").mean()), 4),
        "avg_kelly_pct": round(float(pd.to_numeric(work_df["bet_kelly"], errors="coerce").mean()), 4),
        "avg_model_prob": round(float(pd.to_numeric(work_df["bet_model_prob"], errors="coerce").mean()), 4),
        "avg_odds_american": round(float(pd.to_numeric(work_df["bet_odds_american"], errors="coerce").mean()), 1),
    }])


def write_bucket_overview(work_df, league, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if work_df.empty:
        return
    by_market = []
    for mt, sub in work_df.groupby("market_type", dropna=False, observed=True):
        by_market.append(aggregate_block(sub, league, str(mt).lower(), "market_type", "market_type"))
    atomic_write_csv(pd.concat(by_market, ignore_index=True), out_dir / f"{league}_summary_by_market.csv")
    atomic_write_csv(aggregate_block(work_df, league, None, "side_group", "side_group"), out_dir / f"{league}_summary_by_side_group.csv")
    if "game_date" in work_df.columns:
        atomic_write_csv(aggregate_block(work_df, league, None, "game_date", "game_date"), out_dir / f"{league}_summary_by_date.csv")
    log_cols = ["game_date", "league", "market_type", "side_group", "home_team", "away_team", "bet_side", "bet_line", "bet_odds_american", "bet_ev", "bet_edge_vs_market", "bet_kelly", "bet_model_prob", "bet_stake_pct", "ev_bucket", "edge_vs_market_bucket", "kelly_bucket", "odds_bucket", "model_prob_bucket", "spread_bucket", "total_bucket", "dow_bucket", "month_bucket", "bet_result", "profit_unit", "profit_kelly"]
    atomic_write_csv(work_df[[c for c in log_cols if c in work_df.columns]], out_dir / f"{league}_bet_log.csv")
    atomic_write_csv(bucket_summary_overall(work_df, league), out_dir / f"{league}_summary_overall.csv")
    atomic_write_csv(bucket_summary_grand_total(work_df, league), out_dir / f"{league}_summary_grand_total.csv")


def build_bucket_reports(graded: pd.DataFrame, bucket_root: Path) -> None:
    bucket_root.mkdir(parents=True, exist_ok=True)
    work = prepare_bucket_work(graded)
    for league in LEAGUES:
        league_df = work[work["league"].astype(str).str.lower() == league].copy() if not work.empty else pd.DataFrame()
        for market in MARKETS:
            out_dir = bucket_root / league / market
            write_bucket_market_reports(league_df, league, market, out_dir)
            write_bucket_crosses(league_df, league, market, out_dir)
        write_bucket_overview(league_df, league, bucket_root / league / "overview")


# ---------------- CONFIG WARNINGS ----------------

def collect_config_warnings(filter_cfg: dict) -> list[str]:
    warnings = []
    supported = {"ev", "kelly", "model_prob", "edge_vs_market"}
    markets = filter_cfg.get("markets") or {}
    for league in LEAGUES:
        for market in MARKETS:
            mcfg = ((markets.get(league) or {}).get(market) or {})
            metric = str((mcfg.get("pick_preference") or {}).get("metric", "ev")).strip()
            if metric not in supported:
                warnings.append(f"markets.{league}.{market}.pick_preference.metric={metric!r} is not a production candidate key; production falls back to first qualifying side on ties")
            for side_name, scfg in mcfg.items():
                if side_name not in {"home", "away", "over", "under"} or not isinstance(scfg, dict):
                    continue
                for band_name in ("odds_bands", "line_bands", "ev_bands", "kelly_bands", "model_prob_bands", "edge_vs_market_bands"):
                    bands = scfg.get(band_name)
                    if not isinstance(bands, list):
                        continue
                    for idx, band in enumerate(bands):
                        if not isinstance(band, (list, tuple)) or len(band) != 2:
                            warnings.append(f"markets.{league}.{market}.{side_name}.{band_name}[{idx}] is not [min, max]")
                            continue
                        lo, hi = fv(band[0]), fv(band[1])
                        if lo is not None and hi is not None and lo > hi:
                            warnings.append(f"markets.{league}.{market}.{side_name}.{band_name}[{idx}] has min > max ({lo} > {hi}); it will match nothing")
    return warnings


# ---------------- MANIFEST / RUN INDEX ----------------

def write_manifest(path, run_id, model_config_path, filter_config_path, input_files, settings, config_warnings, total_rows, total_selected, total_graded):
    manifest = {
        "schema_version": 1, "run_id": run_id, "generated_at_utc": now_utc(),
        "backtest_method": "frozen_historical_predictions_current_downstream_logic",
        "historical_bias_handling": "preserve_stored_predictions_no_rebias",
        "outcome_leakage_prevention": "final_score_columns_removed_before_selection_and_rejoined_by_game_id_after_selection",
        "ml_vs_spread_reconciliation": "not_applied_to_match_current_production_selector",
        "model_config": {"path": str(model_config_path), "sha256": sha256_file(model_config_path)},
        "filter_config": {"path": str(filter_config_path), "sha256": sha256_file(filter_config_path)},
        "input_files": [{"path": str(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in input_files],
        "model_settings": settings, "config_warnings": config_warnings,
        "counts": {"historical_rows": total_rows, "selected_bets": total_selected, "graded_bets": total_graded},
        "filter_counts": dict(sorted(DEBUG_COUNTS.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)


def append_run_index(index_path, run_id, overall):
    row = {"run_id": run_id, "generated_at_utc": now_utc(), "bets": 0, "wins": 0, "losses": 0, "pushes": 0, "unknown": 0, "win_rate": None, "profit_units": 0.0, "roi_units": None, "profit_kelly": 0.0, "roi_kelly": None}
    if not overall.empty:
        first = overall.iloc[0]
        for key in list(row):
            if key not in {"run_id", "generated_at_utc"} and key in first.index:
                row[key] = first[key]
    new = pd.DataFrame([row])
    combined = pd.concat([pd.read_csv(index_path), new], ignore_index=True) if index_path.exists() else new
    atomic_write_csv(combined.drop_duplicates(subset=["run_id"], keep="last"), index_path)


# ---------------- ONE INPUT FILE ----------------

def process_historical_file(path, league, settings, filter_cfg, working_dir, selections_dir, graded_dir, kelly_fraction, kelly_cap, logger):
    source_file = path.stem
    logger.log(f"[{league.upper()}] reading {path}")
    raw = pd.read_csv(path, dtype={"game_id": str, "game_date": str})
    validate_historical_input(raw, path, league)
    raw["source_file"] = source_file
    feature_df, scores = split_features_and_scores(raw)
    feature_df["source_file"] = source_file
    scores["source_file"] = source_file
    frames = {
        "moneyline": process_moneyline_ev(process_moneyline_juice(feature_df, settings[league])),
        "spread": process_spread_ev(process_spread_juice(feature_df, settings[league])),
        "total": process_total_ev(process_total_juice(feature_df, settings[league])),
    }
    selected_parts = []
    for market, market_df in frames.items():
        atomic_write_csv(market_df, working_dir / league / market / f"{source_file}_{market}.csv")
        selected = select_bets_for_market(market_df, league, market, filter_cfg, settings, kelly_fraction, kelly_cap)
        if not selected.empty:
            selected_parts.append(selected)
        logger.log(f"[{league.upper()}] {source_file} {market}: rows={len(market_df)} selected={len(selected)}")
    selections = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    atomic_write_csv(selections, selections_dir / league / f"{source_file}_selected.csv")
    if selections.empty:
        graded = pd.DataFrame()
    else:
        score_cols = [c for c in scores.columns if c != "source_file"]
        graded = grade_selections(selections, scores[score_cols])
    atomic_write_csv(graded, graded_dir / league / f"{source_file}_graded.csv")
    logger.log(f"[{league.upper()}] {source_file}: historical_rows={len(raw)} selected={len(selections)} graded={len(graded)}")
    return selections, graded, len(raw)


# ---------------- MAIN ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Replay combined historical basketball files through current downstream model and selection logic.")
    p.add_argument("--backtest-dir", default=str(DEFAULT_BACKTEST_DIR))
    p.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    input_dir = backtest_dir / "input"
    configs_dir = backtest_dir / "configs"
    working_dir = backtest_dir / "working"
    selections_dir = backtest_dir / "selections"
    graded_dir = backtest_dir / "graded"
    reports_dir = backtest_dir / "reports"
    runs_dir = backtest_dir / "runs"
    model_config_path = Path(args.model_config)
    filter_config_path = configs_dir / "markets_test.yaml"
    for folder in (input_dir, configs_dir, working_dir, selections_dir, graded_dir, reports_dir, runs_dir):
        folder.mkdir(parents=True, exist_ok=True)
    run_id = sanitize_run_name(args.run_name) if args.run_name else timestamp_id()
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run snapshot already exists: {run_dir}")
    clear_directory_contents(working_dir)
    clear_directory_contents(selections_dir)
    clear_directory_contents(graded_dir)
    clear_directory_contents(reports_dir)
    logger = RunLogger(reports_dir / "basketball_backtest.txt")
    logger.log(f"run_id={run_id}")
    logger.log(f"backtest_dir={backtest_dir}")
    logger.log(f"model_config={model_config_path}")
    logger.log(f"filter_config={filter_config_path}")
    model_cfg, filter_cfg = read_yaml(model_config_path), read_yaml(filter_config_path)
    settings = build_league_settings(model_cfg)
    stake_cfg = filter_cfg.get("stake_sizing") or {}
    kelly_fraction = require_number(stake_cfg.get("kelly_fraction", 1.0), "stake_sizing.kelly_fraction")
    kelly_cap = require_number(stake_cfg.get("kelly_cap", 1.0), "stake_sizing.kelly_cap")
    config_warnings = collect_config_warnings(filter_cfg)
    for w in config_warnings:
        logger.log(w, "WARN")
    input_files = []
    for league in LEAGUES:
        fs = sorted(input_dir.glob(f"*_{league.upper()}.csv"))
        if not fs:
            raise FileNotFoundError(f"No historical input files found for {league.upper()} in {input_dir}")
        input_files.extend(fs)
    all_selections, all_graded, total_rows = [], [], 0
    for league in LEAGUES:
        for path in sorted(input_dir.glob(f"*_{league.upper()}.csv")):
            selections, graded, n = process_historical_file(path, league, settings, filter_cfg, working_dir, selections_dir, graded_dir, kelly_fraction, kelly_cap, logger)
            total_rows += n
            if not selections.empty:
                all_selections.append(selections)
            if not graded.empty:
                all_graded.append(graded)
    combined_selected = pd.concat(all_selections, ignore_index=True) if all_selections else pd.DataFrame()
    combined_graded = pd.concat(all_graded, ignore_index=True) if all_graded else pd.DataFrame()
    atomic_write_csv(combined_selected, selections_dir / "all_selected.csv")
    atomic_write_csv(combined_graded, graded_dir / "all_graded.csv")
    reports = build_reports(combined_graded, reports_dir)
    build_bucket_reports(combined_graded, reports_dir / "bucket_reports")
    write_manifest(reports_dir / "run_manifest.yaml", run_id, model_config_path, filter_config_path, input_files, settings, config_warnings, total_rows, len(combined_selected), len(combined_graded))
    logger.log("--- FINAL SUMMARY ---")
    logger.log(f"historical_rows={total_rows}")
    logger.log(f"selected_bets={len(combined_selected)}")
    logger.log(f"graded_bets={len(combined_graded)}")
    if not reports["overall"].empty:
        row = reports["overall"].iloc[0]
        roi = row["roi_units"]
        logger.log(f"W/L/P/U={int(row['wins'])}/{int(row['losses'])}/{int(row['pushes'])}/{int(row['unknown'])} profit_units={float(row['profit_units']):+.4f} roi_units={float(roi):+.4%}" if pd.notna(roi) else f"W/L/P/U={int(row['wins'])}/{int(row['losses'])}/{int(row['pushes'])}/{int(row['unknown'])} profit_units={float(row['profit_units']):+.4f} roi_units=N/A")
    logger.log(f"run_snapshot={run_dir}")
    logger.log("STATUS: SUCCESS")
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(filter_config_path, run_dir / "markets_test.yaml")
    shutil.copy2(model_config_path, run_dir / "model_config.yaml")
    copy_tree_contents(reports_dir, run_dir / "reports")
    copy_tree_contents(selections_dir, run_dir / "selections")
    copy_tree_contents(graded_dir, run_dir / "graded")
    append_run_index(runs_dir / "index.csv", run_id, reports["overall"])
    print("basketball_backtest complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STATUS: FAILED | {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

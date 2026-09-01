#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_model.yaml"
HISTORY_ROOT = BASE / "00_intake/sdv/history"
COMBINED_ROOT = BASE / "00_intake/final_combined_files/combined"
OUT = BASE / "backtest/model_validation/sdv_ensemble/advanced_features"
TESTING_DIR = BASE / "scripts/testing"
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))
import sdv_model_train as smt

LEAGUES = ["nba", "ncaam", "wnba"]
LABELS = {"nba": "NBA", "ncaam": "NCAAM", "wnba": "WNBA"}
SEASONS = [2023, 2024, 2025]
FAMILY_PREFIXES = {
    "lineup": ("home_lineup_", "away_lineup_", "diff_lineup_"),
    "possessions": ("home_possession_", "away_possession_", "diff_possession_"),
    "shots": ("home_shot_", "away_shot_", "diff_shot_"),
}
EPS = 1e-12


def clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def clean_id(v: Any) -> str:
    s = clean(v)
    if not s:
        return ""
    try:
        x = float(s)
        if math.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return s


def fnum(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(str(v).strip().replace("+", ""))
    except Exception:
        return None
    return x if math.isfinite(x) else None


def norm_name(v: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", clean(v).lower())


def norm_date(v: Any) -> str:
    return clean(v).replace("_", "-")[:10]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_cfg() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    adv = cfg.get("advanced_features")
    if not isinstance(adv, dict):
        raise RuntimeError("sdv_model.yaml advanced_features mapping missing")
    if clean(adv.get("promotion_status")) != "candidate_not_promoted":
        raise RuntimeError("advanced_features.promotion_status must be candidate_not_promoted")
    return cfg


def candidate_root(cfg: dict[str, Any]) -> Path:
    adv = cfg["advanced_features"]
    value = clean(adv.get("candidate_history_output_root"))
    if not value:
        raise RuntimeError("candidate_history_output_root missing")
    return Path(value)


def load_season_rows(cfg: dict[str, Any], league: str, season: int) -> list[dict[str, Any]]:
    label = LABELS[league]
    feat_path = candidate_root(cfg) / league / f"{season}_{label}_features.parquet"
    games_path = HISTORY_ROOT / league / str(season) / "games.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    if not games_path.exists():
        raise FileNotFoundError(games_path)

    features = pl.read_parquet(feat_path).to_dicts()
    games = pl.read_parquet(games_path).to_dicts()
    game_map: dict[str, dict[str, Any]] = {}
    for g in games:
        gid = clean_id(g.get("game_id"))
        if gid:
            game_map[gid] = g

    rows: list[dict[str, Any]] = []
    for f in features:
        gid = clean_id(f.get("game_id"))
        g = game_map.get(gid)
        if not g:
            continue
        hs = fnum(g.get("home_score"))
        aws = fnum(g.get("away_score"))
        if hs is None or aws is None:
            continue
        row = dict(f)
        row["game_id"] = gid
        row["_target_margin"] = hs - aws
        row["_target_total"] = hs + aws
        row["actual_margin"] = hs - aws
        row["actual_total"] = hs + aws
        row["home_score"] = hs
        row["away_score"] = aws
        row["home_team"] = g.get("home_display_name") or g.get("home_name") or ""
        row["away_team"] = g.get("away_display_name") or g.get("away_name") or ""
        row["game_date"] = norm_date(g.get("game_date") or f.get("game_date"))
        rows.append(row)
    rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
    return rows


def feature_columns(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    cols = sorted({k for r in rows for k in r.keys() if k.startswith(prefixes)})
    return [c for c in cols if not c.endswith("_games_used")]


def non_null_fraction(rows: list[dict[str, Any]], cols: list[str]) -> float:
    if not rows or not cols:
        return 0.0
    total = len(rows) * len(cols)
    have = 0
    for r in rows:
        for c in cols:
            if fnum(r.get(c)) is not None:
                have += 1
    return have / total if total else 0.0


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, np.ndarray, Any]:
    enc = smt.SparseFeatureEncoder.fit(train, numeric, categorical)
    xtr = enc.transform(train)
    xte = enc.transform(test)
    mcoef = smt.fit_ridge(xtr, smt.target_array(train, "_target_margin"), float(CFG["training"]["ridge_alpha"]))
    tcoef = smt.fit_ridge(xtr, smt.target_array(train, "_target_total"), float(CFG["training"]["ridge_alpha"]))
    return smt.predict(xte, mcoef), smt.predict(xte, tcoef), enc


def chronological_oos_residuals(rows: list[dict[str, Any]], numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, np.ndarray]:
    dates = sorted({r["game_date"] for r in rows})
    folds = int(CFG["training"]["oos"]["folds"])
    frac = float(CFG["training"]["oos"]["minimum_training_date_fraction"])
    start = max(1, int(len(dates) * frac))
    val_dates = dates[start:]
    chunks = np.array_split(np.asarray(val_dates, dtype=object), folds)
    margin_res: list[float] = []
    total_res: list[float] = []
    for chunk in chunks:
        ds = [str(x) for x in chunk.tolist()]
        if not ds:
            continue
        lo, hi = min(ds), max(ds)
        train = [r for r in rows if r["game_date"] < lo]
        val = [r for r in rows if lo <= r["game_date"] <= hi]
        if not train or not val:
            continue
        pm, pt, _ = fit_predict(train, val, numeric, categorical)
        for i, r in enumerate(val):
            margin_res.append(float(r["_target_margin"]) - float(pm[i]))
            total_res.append(float(r["_target_total"]) - float(pt[i]))
    if len(margin_res) < 10 or len(total_res) < 10:
        raise RuntimeError("Too few chronological OOS residuals")
    return np.asarray(margin_res), np.asarray(total_res)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def residual_params(train: list[dict[str, Any]], numeric: list[str], categorical: list[str]) -> tuple[float, float, float, float]:
    mr, tr = chronological_oos_residuals(train, numeric, categorical)
    ms = float(np.std(mr, ddof=1))
    ts = float(np.std(tr, ddof=1))
    if ms <= 0 or ts <= 0:
        raise RuntimeError("Invalid OOS residual standard deviation")
    return float(np.mean(mr)), ms, float(np.mean(tr)), ts


def regression_metrics(rows: list[dict[str, Any]], pm: np.ndarray, pt: np.ndarray) -> dict[str, float]:
    am = np.asarray([float(r["_target_margin"]) for r in rows])
    at = np.asarray([float(r["_target_total"]) for r in rows])
    return {
        "games": float(len(rows)),
        "margin_mae": float(np.mean(np.abs(pm - am))),
        "margin_rmse": float(np.sqrt(np.mean((pm - am) ** 2))),
        "total_mae": float(np.mean(np.abs(pt - at))),
        "total_rmse": float(np.sqrt(np.mean((pt - at) ** 2))),
        "winner_accuracy": float(np.mean((pm > 0) == (am > 0))),
    }


def load_combined(league: str, season: int) -> list[dict[str, Any]]:
    path = COMBINED_ROOT / f"{season}_{LABELS[league]}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def match_market_rows(pred_rows: list[dict[str, Any]], combined: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pred_rows:
        by_key[(r["game_date"], int(round(r["home_score"])), int(round(r["away_score"])))].append(r)
    matched: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for c in combined:
        d = norm_date(c.get("game_date"))
        hs, aws = fnum(c.get("home_score")), fnum(c.get("away_score"))
        if hs is None or aws is None:
            continue
        candidates = [r for r in by_key.get((d, int(round(hs)), int(round(aws))), []) if r["game_id"] not in used]
        if not candidates:
            continue
        ch, ca = norm_name(c.get("home_team")), norm_name(c.get("away_team"))
        candidates.sort(key=lambda r: (norm_name(r.get("home_team")) == ch) + (norm_name(r.get("away_team")) == ca), reverse=True)
        best = candidates[0]
        used.add(best["game_id"])
        matched[best["game_id"]] = c
    return matched


def settle_profit(won: bool | None, dec: float | None) -> float | None:
    if won is None or dec is None or dec <= 1:
        return None
    return dec - 1.0 if won else -1.0


def market_metrics(rows: list[dict[str, Any]], pm: np.ndarray, pt: np.ndarray, residual: tuple[float, float, float, float], market_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mm, ms, tm, ts = residual
    b_ml: list[float] = []
    b_sp: list[float] = []
    b_tot: list[float] = []
    profits: list[float] = []
    bet_count = 0
    for i, r in enumerate(rows):
        c = market_map.get(r["game_id"])
        if not c:
            continue
        actual_margin = float(r["actual_margin"])
        actual_total = float(r["actual_total"])
        pred_margin = float(pm[i]) + mm
        pred_total = float(pt[i]) + tm

        p_home = min(max(normal_cdf(pred_margin / ms), EPS), 1 - EPS)
        y_home = 1.0 if actual_margin > 0 else 0.0
        b_ml.append((p_home - y_home) ** 2)
        hdec, adec = fnum(c.get("home_dk_moneyline_decimal")), fnum(c.get("away_dk_moneyline_decimal"))
        evh = p_home * hdec - 1 if hdec and hdec > 1 else -999
        eva = (1 - p_home) * adec - 1 if adec and adec > 1 else -999
        if max(evh, eva) > 0:
            if evh >= eva:
                p = settle_profit(actual_margin > 0, hdec)
            else:
                p = settle_profit(actual_margin < 0, adec)
            if p is not None:
                profits.append(p); bet_count += 1

        hs, aws = fnum(c.get("home_spread")), fnum(c.get("away_spread"))
        hsd, asd = fnum(c.get("home_dk_spread_decimal")), fnum(c.get("away_dk_spread_decimal"))
        if hs is not None:
            delta = actual_margin + hs
            if delta != 0:
                p_cover = min(max(normal_cdf((pred_margin + hs) / ms), EPS), 1 - EPS)
                b_sp.append((p_cover - (1.0 if delta > 0 else 0.0)) ** 2)
                evh = p_cover * hsd - 1 if hsd and hsd > 1 else -999
                eva = (1 - p_cover) * asd - 1 if asd and asd > 1 else -999
                if max(evh, eva) > 0:
                    if evh >= eva:
                        p = settle_profit(delta > 0, hsd)
                    else:
                        away_delta = -actual_margin + (aws if aws is not None else -hs)
                        p = settle_profit(away_delta > 0, asd)
                    if p is not None:
                        profits.append(p); bet_count += 1

        line = fnum(c.get("total"))
        odec, udec = fnum(c.get("dk_total_over_decimal")), fnum(c.get("dk_total_under_decimal"))
        if line is not None:
            delta = actual_total - line
            if delta != 0:
                p_over = min(max(normal_cdf((pred_total - line) / ts), EPS), 1 - EPS)
                b_tot.append((p_over - (1.0 if delta > 0 else 0.0)) ** 2)
                evo = p_over * odec - 1 if odec and odec > 1 else -999
                evu = (1 - p_over) * udec - 1 if udec and udec > 1 else -999
                if max(evo, evu) > 0:
                    p = settle_profit(delta > 0, odec) if evo >= evu else settle_profit(delta < 0, udec)
                    if p is not None:
                        profits.append(p); bet_count += 1

    all_brier = b_ml + b_sp + b_tot
    return {
        "moneyline_brier": float(np.mean(b_ml)) if b_ml else None,
        "spread_brier": float(np.mean(b_sp)) if b_sp else None,
        "total_brier": float(np.mean(b_tot)) if b_tot else None,
        "calibration_brier_mean": float(np.mean(all_brier)) if all_brier else None,
        "profit_bets": bet_count,
        "profit_units": float(sum(profits)),
        "profit_roi": float(sum(profits) / bet_count) if bet_count else None,
        "market_games": len(market_map),
    }


def train_fit_metrics(train: list[dict[str, Any]], numeric: list[str], categorical: list[str]) -> dict[str, float]:
    pm, pt, _ = fit_predict(train, train, numeric, categorical)
    m = regression_metrics(train, pm, pt)
    return {"train_margin_mae": m["margin_mae"], "train_total_mae": m["total_mae"]}


def evaluate_candidate(league: str, name: str, numeric: list[str], categorical: list[str], train: list[dict[str, Any]], eval_rows: list[dict[str, Any]], season: int) -> dict[str, Any]:
    residual = residual_params(train, numeric, categorical)
    pm, pt, _ = fit_predict(train, eval_rows, numeric, categorical)
    result = {
        "league": LABELS[league],
        "candidate": name,
        "training_seasons": "+".join(sorted({str(r.get("internal_season")) for r in train})),
        "evaluation_season": season,
        "lockbox": season == 2025,
        "numeric_features": len(numeric),
        **train_fit_metrics(train, numeric, categorical),
        **regression_metrics(eval_rows, pm, pt),
    }
    combined = load_combined(league, season)
    market_map = match_market_rows(eval_rows, combined) if combined else {}
    result.update(market_metrics(eval_rows, pm, pt, residual, market_map) if market_map else {
        "moneyline_brier": None, "spread_brier": None, "total_brier": None,
        "calibration_brier_mean": None, "profit_bets": 0, "profit_units": 0.0,
        "profit_roi": None, "market_games": 0,
    })
    return result


def score_error(row: dict[str, Any], base: dict[str, Any]) -> float:
    return 0.5 * (float(row["margin_mae"]) / float(base["margin_mae"]) + float(row["total_mae"]) / float(base["total_mae"]))


def promotion_status(row: dict[str, Any], base: dict[str, Any]) -> tuple[str, str]:
    err_ratio = score_error(row, base)
    cb, bb = row.get("calibration_brier_mean"), base.get("calibration_brier_mean")
    cr, br = row.get("profit_roi"), base.get("profit_roi")
    reasons: list[str] = []
    if err_ratio > 1.0 + 1e-9:
        reasons.append(f"validation_error_worse={err_ratio:.6f}")
    if cb is not None and bb is not None and float(cb) > float(bb) + 1e-9:
        reasons.append(f"calibration_worse={float(cb):.6f}>{float(bb):.6f}")
    if cr is not None and br is not None and float(cr) < float(br) - 1e-9:
        reasons.append(f"profit_roi_worse={float(cr):.6f}<{float(br):.6f}")
    if reasons:
        return "REJECT", ";".join(reasons)
    improved = err_ratio < 1.0 - 1e-9 or (cb is not None and bb is not None and float(cb) < float(bb) - 1e-9) or (cr is not None and br is not None and float(cr) > float(br) + 1e-9)
    return ("PROMOTION_READY", "no_lockbox_metric_degraded") if improved else ("NO_MATERIAL_GAIN", "no_lockbox_metric_degraded_but_no_gain")


CFG = load_cfg()
BASE_NUMERIC = [clean(x) for x in CFG["model_inputs"]["numeric"]]
CATEGORICAL = [clean(x) for x in CFG["model_inputs"].get("categorical", [])]
OUT.mkdir(parents=True, exist_ok=True)

summary: list[dict[str, Any]] = []
coverage_rows: list[dict[str, Any]] = []
promotion_rows: list[dict[str, Any]] = []

for league in LEAGUES:
    season_rows = {s: load_season_rows(CFG, league, s) for s in SEASONS}
    all_rows = season_rows[2023] + season_rows[2024] + season_rows[2025]
    family_cols: dict[str, list[str]] = {}
    for family, prefixes in FAMILY_PREFIXES.items():
        cols = feature_columns(all_rows, prefixes)
        family_cols[family] = cols
        for season in SEASONS:
            coverage_rows.append({
                "league": LABELS[league], "season": season, "family": family,
                "columns": len(cols), "non_null_fraction": non_null_fraction(season_rows[season], cols),
                "source_status": "available" if cols and non_null_fraction(season_rows[season], cols) > 0 else "explicitly_missing",
            })

    candidates: dict[str, list[str]] = {"v1": BASE_NUMERIC}
    for family, cols in family_cols.items():
        if cols:
            candidates[family] = BASE_NUMERIC + cols
    all_advanced = sorted({c for cols in family_cols.values() for c in cols})
    if all_advanced:
        candidates["all_advanced"] = BASE_NUMERIC + all_advanced

    stages = [([2023], 2024), ([2023, 2024], 2025)]
    for train_seasons, eval_season in stages:
        train = [r for s in train_seasons for r in season_rows[s]]
        eval_rows = season_rows[eval_season]
        for name, numeric in candidates.items():
            summary.append(evaluate_candidate(league, name, numeric, CATEGORICAL, train, eval_rows, eval_season))

    lockbox = [r for r in summary if r["league"] == LABELS[league] and r["evaluation_season"] == 2025]
    base = next(r for r in lockbox if r["candidate"] == "v1")
    for row in lockbox:
        if row["candidate"] == "v1":
            continue
        status, reason = promotion_status(row, base)
        promotion_rows.append({
            "league": LABELS[league], "family": row["candidate"], "status": status, "reason": reason,
            "error_ratio_vs_v1": score_error(row, base),
            "candidate_calibration_brier": row.get("calibration_brier_mean"),
            "v1_calibration_brier": base.get("calibration_brier_mean"),
            "candidate_profit_roi": row.get("profit_roi"), "v1_profit_roi": base.get("profit_roi"),
        })

write_csv(OUT / "item20_validation_summary.csv", summary)
write_csv(OUT / "item20_source_coverage.csv", coverage_rows)
write_csv(OUT / "item20_promotion_decisions.csv", promotion_rows)

lines = [
    "ITEM 20 ADVANCED SDV FEATURE VALIDATION",
    "=" * 90,
    f"Candidate feature version: {CFG.get('feature_version')}",
    f"Production V1 feature version preserved: {CFG['advanced_features'].get('production_feature_version')}",
    "2025 lockbox is evaluated only after candidate families are fixed.",
    "No production model or market band is modified by this script.",
    "",
    "PROMOTION DECISIONS",
    "=" * 90,
]
for r in promotion_rows:
    lines.append(f"{r['league']:5s} {r['family']:13s} {r['status']:18s} {r['reason']}")
lines += ["", "Rule: reject any candidate that worsens untouched lockbox error, calibration, or profit ROI versus V1."]
(OUT / "ITEM20_ADVANCED_FEATURE_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("ITEM 20 ADVANCED FEATURE VALIDATION COMPLETE")
for r in promotion_rows:
    print(f"{r['league']} {r['family']}: {r['status']} | {r['reason']}")
print(f"REPORT: {OUT / 'ITEM20_ADVANCED_FEATURE_REPORT.txt'}")

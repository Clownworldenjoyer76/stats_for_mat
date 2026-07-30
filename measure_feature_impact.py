#!/usr/bin/env python3
"""Measure feature impact for MLB score and win-probability predictions.

For every eligible raw feature this script records:
  * train-only screening score
  * neutral-ablation change in score RMSE, total RMSE, and Brier score
  * permutation change in those metrics
  * SHAP importance and direction when supported
  * stability across rolling season holdouts
  * keep/review/drop-candidate status

The ablation method removes a feature from the fitted model at inference by
replacing its transformed columns with their training means. This gives every
selected feature a comparable removal test without thousands of full retrains.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_all_models import (  # noqa: E402
    ID_COLUMNS, TARGETS, evaluate_predictions, fit_bundle,
    predict_estimator, score_to_win_probability,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="model/full_features.csv")
    p.add_argument("--lineage", default="model/feature_lineage.csv")
    p.add_argument("--leaderboard", default="model/model_leaderboard.csv")
    p.add_argument("--model-manifest", default="model/model_manifest.json")
    p.add_argument("--out-dir", default="model")
    p.add_argument("--models", nargs="+", default=["winner"],
                   help="Use 'winner', 'all', or explicit model names")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--shap-sample", type=int, default=400)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-rmse-threshold", type=float, default=0.001)
    return p.parse_args()


def predict_matrix(bundle: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = bundle["model_name"]
    h = np.clip(predict_estimator(bundle["home_model"], X, name), 0.05, 20.0)
    a = np.clip(predict_estimator(bundle["away_model"], X, name), 0.05, 20.0)
    return h, a, score_to_win_probability(h, a)


def model_names(args: argparse.Namespace, manifest: dict[str, Any], leaderboard: pd.DataFrame) -> list[str]:
    requested = args.models
    if requested == ["winner"]:
        return [str(manifest["winner"])]
    if requested == ["all"]:
        return leaderboard["model"].astype(str).tolist()
    return requested


def feature_direction(x: np.ndarray, contribution: np.ndarray) -> tuple[str, float]:
    x = np.asarray(x, dtype=float)
    contribution = np.asarray(contribution, dtype=float)
    mask = np.isfinite(x) & np.isfinite(contribution)
    if mask.sum() < 20 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(contribution[mask]) < 1e-12:
        return "mixed_or_unknown", np.nan
    corr = float(np.corrcoef(x[mask], contribution[mask])[0, 1])
    if corr > 0.08:
        return "increases_projected_total", corr
    if corr < -0.08:
        return "decreases_projected_total", corr
    return "mixed_or_nonlinear", corr


def linear_or_tree_shap(model: Any, X: np.ndarray, model_name: str) -> tuple[np.ndarray | None, str]:
    """Return SHAP-like matrix for a fitted estimator and the method label."""
    try:
        import shap
    except Exception as exc:
        return None, f"unavailable:{exc}"

    try:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
            Xt = model[:-1].transform(X)
            if hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, Xt)
                values = explainer(Xt).values
                return np.asarray(values, dtype=float), "shap.LinearExplainer"
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[0]
        return np.asarray(values, dtype=float), "shap.TreeExplainer"
    except Exception as tree_exc:
        if model_name == "negative_binomial":
            try:
                params = np.asarray(model["result"].params, dtype=float)[1:]
                centered = X - np.nanmean(X, axis=0)
                return centered * params, "coefficient_contribution_proxy"
            except Exception as nb_exc:
                return None, f"failed:{tree_exc};{nb_exc}"
        return None, f"failed:{tree_exc}"


def shap_by_raw_feature(bundle: dict[str, Any], X: np.ndarray,
                        sample_size: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        idx = np.sort(rng.choice(len(X), size=sample_size, replace=False))
        Xs = X[idx]
    else:
        Xs = X
    sh, method_h = linear_or_tree_shap(bundle["home_model"], Xs, bundle["model_name"])
    sa, method_a = linear_or_tree_shap(bundle["away_model"], Xs, bundle["model_name"])
    rows = []
    mapping = bundle["raw_to_transformed_indices"]
    if sh is None or sa is None:
        for feature in mapping:
            rows.append({"feature": feature, "shap_mean_abs": np.nan,
                         "shap_mean_signed": np.nan, "direction": "unavailable",
                         "direction_corr": np.nan,
                         "shap_method": f"home={method_h};away={method_a}"})
        return pd.DataFrame(rows)

    for feature, indices in mapping.items():
        idx = np.asarray(indices, dtype=int)
        home_contrib = np.nansum(sh[:, idx], axis=1)
        away_contrib = np.nansum(sa[:, idx], axis=1)
        total_contrib = home_contrib + away_contrib
        raw_value = np.nanmean(Xs[:, idx], axis=1)
        direction, corr = feature_direction(raw_value, total_contrib)
        rows.append({
            "feature": feature,
            "shap_mean_abs": float(np.nanmean(np.abs(home_contrib)) + np.nanmean(np.abs(away_contrib))),
            "shap_mean_signed": float(np.nanmean(total_contrib)),
            "direction": direction,
            "direction_corr": corr,
            "shap_method": f"home={method_h};away={method_a}",
        })
    return pd.DataFrame(rows)


def impact_one_fold(train: pd.DataFrame, test: pd.DataFrame, model_name: str,
                    settings: dict[str, Any], repeats: int, random_state: int,
                    shap_sample: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle, screening, eligibility = fit_bundle(
        train, model_name, settings["max_features"], settings.get("max_missing", 0.995),
        settings.get("max_categories", 200), settings.get("profile", "quick"),
        random_state, settings.get("n_jobs", -1),
    )
    selected = bundle["selected_features"]
    X = np.asarray(bundle["preprocessor"].transform(test[selected]), dtype=float)
    h0, a0, p0 = predict_matrix(bundle, X)
    base = evaluate_predictions(test, h0, a0, p0)
    train_means = np.asarray(bundle["transformed_train_means"], dtype=float)
    mapping = bundle["raw_to_transformed_indices"]
    rng = np.random.default_rng(random_state)
    rows = []

    for feature, indices in mapping.items():
        idx = np.asarray(indices, dtype=int)
        original = X[:, idx].copy()

        X[:, idx] = train_means[idx]
        ha, aa, pa = predict_matrix(bundle, X)
        abl = evaluate_predictions(test, ha, aa, pa)
        X[:, idx] = original

        perm_metrics = []
        for _ in range(max(1, repeats)):
            order = rng.permutation(len(X))
            X[:, idx] = original[order]
            hp, ap, pp = predict_matrix(bundle, X)
            perm_metrics.append(evaluate_predictions(test, hp, ap, pp))
            X[:, idx] = original

        perm = {k: float(np.mean([m[k] for m in perm_metrics])) for k in perm_metrics[0]}
        rows.append({
            "feature": feature,
            "model_name": model_name,
            "removed_score_rmse_delta": abl["score_rmse"] - base["score_rmse"],
            "removed_total_rmse_delta": abl["total_rmse"] - base["total_rmse"],
            "removed_brier_delta": abl["brier"] - base["brier"],
            "permutation_score_rmse_delta": perm["score_rmse"] - base["score_rmse"],
            "permutation_total_rmse_delta": perm["total_rmse"] - base["total_rmse"],
            "permutation_brier_delta": perm["brier"] - base["brier"],
            "base_score_rmse": base["score_rmse"],
            "base_total_rmse": base["total_rmse"],
            "base_brier": base["brier"],
        })

    shap = shap_by_raw_feature(bundle, X, shap_sample, random_state)
    return pd.DataFrame(rows), screening, shap


def aggregate_impact(fold_impacts: pd.DataFrame, screenings: pd.DataFrame,
                     shap_frames: pd.DataFrame, all_features: list[str],
                     model_name: str, threshold: float) -> pd.DataFrame:
    if fold_impacts.empty:
        agg = pd.DataFrame({"feature": all_features})
    else:
        grouped = fold_impacts.groupby("feature")
        agg = grouped.agg(
            selected_folds=("test_season", "nunique"),
            removed_score_rmse_delta=("removed_score_rmse_delta", "mean"),
            removed_total_rmse_delta=("removed_total_rmse_delta", "mean"),
            removed_brier_delta=("removed_brier_delta", "mean"),
            permutation_score_rmse_delta=("permutation_score_rmse_delta", "mean"),
            permutation_total_rmse_delta=("permutation_total_rmse_delta", "mean"),
            permutation_brier_delta=("permutation_brier_delta", "mean"),
            impact_season_sd=("permutation_score_rmse_delta", "std"),
            impact_positive_seasons=("permutation_score_rmse_delta", lambda s: float((s > 0).mean())),
        ).reset_index()
        agg = pd.DataFrame({"feature": all_features}).merge(agg, on="feature", how="left")

    if not screenings.empty:
        scr = screenings.groupby("feature", as_index=False).agg(
            screening_score=("screening_score", "mean"),
            selected_by_screen_folds=("selected_by_screen", "sum"),
        )
        agg = agg.merge(scr, on="feature", how="left")
    if not shap_frames.empty:
        shp = shap_frames.groupby("feature", as_index=False).agg(
            shap_mean_abs=("shap_mean_abs", "mean"),
            shap_mean_signed=("shap_mean_signed", "mean"),
            direction_corr=("direction_corr", "mean"),
            direction=("direction", lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else "unavailable"),
            shap_method=("shap_method", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        agg = agg.merge(shp, on="feature", how="left")

    agg["model_name"] = model_name
    agg["selected_folds"] = agg.get("selected_folds", 0).fillna(0).astype(int)
    agg["impact_consistency"] = agg.get("impact_positive_seasons", np.nan)

    def status(row: pd.Series) -> tuple[str, str]:
        if int(row.get("selected_folds", 0)) == 0:
            return "screened_out", "not used by this model in any rolling fold"
        rmse_delta = float(row.get("permutation_score_rmse_delta", np.nan))
        brier_delta = float(row.get("permutation_brier_delta", np.nan))
        consistency = float(row.get("impact_positive_seasons", np.nan))
        if np.isfinite(rmse_delta) and rmse_delta >= threshold and consistency >= 0.67:
            return "keep", "permutation worsened score RMSE consistently"
        if np.isfinite(rmse_delta) and rmse_delta < 0 and np.isfinite(brier_delta) and brier_delta < 0:
            return "drop_candidate", "removal/permutation improved both score and win metrics"
        return "review", "signal is small, unstable, or metric-dependent"

    decisions = agg.apply(status, axis=1, result_type="expand")
    agg["keep_drop_status"] = decisions[0]
    agg["decision_reason"] = decisions[1]
    return agg


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(args.features, nrows=0)
    parse_dates = ["date"] if "date" in header.columns else None
    df = pd.read_csv(args.features, low_memory=False, parse_dates=parse_dates)
    df = df.dropna(subset=TARGETS).copy()
    seasons = sorted(int(x) for x in pd.unique(df["season"]))
    if len(seasons) < 2:
        raise SystemExit("At least two seasons are required")

    manifest = json.loads(Path(args.model_manifest).read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(args.leaderboard)
    models = model_names(args, manifest, leaderboard)
    settings = {
        "max_features": int(manifest.get("max_features", 350)),
        "profile": manifest.get("profile", "quick"),
        "max_missing": 0.995,
        "max_categories": 200,
        "n_jobs": args.n_jobs,
    }
    all_features = [c for c in df.columns if c not in set(TARGETS) | ID_COLUMNS]

    checkpoint_path = out_dir / "feature_impact_fold_checkpoint.csv"
    existing = pd.read_csv(checkpoint_path) if args.resume and checkpoint_path.exists() else pd.DataFrame()
    fold_frames = [existing] if not existing.empty else []
    screening_frames = []
    shap_frames = []
    completed = set()
    if not existing.empty:
        completed = set(zip(existing["model_name"].astype(str), existing["test_season"].astype(int)))

    final_frames = []
    for model_name in models:
        print(f"Feature impact: {model_name}")
        for test_season in seasons[1:]:
            if (model_name, test_season) in completed:
                print(f"  fold {test_season} already checkpointed")
                continue
            train = df[df["season"] < test_season].copy()
            test = df[df["season"] == test_season].copy()
            print(f"  fold <{test_season} -> {test_season}")
            impacts, screening, shap = impact_one_fold(
                train, test, model_name, settings, args.repeats,
                args.random_state + test_season, args.shap_sample,
            )
            impacts["test_season"] = test_season
            screening["test_season"] = test_season
            screening["model_name"] = model_name
            shap["test_season"] = test_season
            shap["model_name"] = model_name
            fold_frames.append(impacts)
            screening_frames.append(screening)
            shap_frames.append(shap)
            pd.concat(fold_frames, ignore_index=True).to_csv(checkpoint_path, index=False)

        all_fold = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        model_fold = all_fold[all_fold["model_name"] == model_name].copy() if not all_fold.empty else pd.DataFrame()
        model_screen = pd.concat(screening_frames, ignore_index=True) if screening_frames else pd.DataFrame()
        if not model_screen.empty:
            model_screen = model_screen[model_screen["model_name"] == model_name]
        model_shap = pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame()
        if not model_shap.empty:
            model_shap = model_shap[model_shap["model_name"] == model_name]
        final_frames.append(aggregate_impact(
            model_fold, model_screen, model_shap, all_features,
            model_name, args.keep_rmse_threshold,
        ))

    result = pd.concat(final_frames, ignore_index=True)
    lineage_path = Path(args.lineage)
    if lineage_path.exists():
        lineage = pd.read_csv(lineage_path)
        result = result.merge(lineage, on="feature", how="left")

    order = [
        "feature", "source_file", "source_columns", "transformation", "feature_group",
        "model_name", "direction", "direction_corr", "screening_score",
        "removed_score_rmse_delta", "removed_total_rmse_delta", "removed_brier_delta",
        "permutation_score_rmse_delta", "permutation_total_rmse_delta",
        "permutation_brier_delta", "shap_mean_abs", "shap_mean_signed",
        "selected_folds", "impact_consistency", "impact_season_sd",
        "keep_drop_status", "decision_reason", "shap_method",
    ]
    order = [c for c in order if c in result.columns] + [c for c in result.columns if c not in order]
    result = result[order].sort_values(
        ["model_name", "keep_drop_status", "permutation_score_rmse_delta"],
        ascending=[True, True, False], na_position="last",
    )
    result.to_csv(out_dir / "feature_impact.csv", index=False)
    print(f"Wrote {len(result):,} feature-model impact rows -> {(out_dir / 'feature_impact.csv').resolve()}")


if __name__ == "__main__":
    main()

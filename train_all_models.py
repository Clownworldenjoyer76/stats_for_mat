#!/usr/bin/env python3
"""Train and compare MLB score models on identical rolling-season folds.

Training:
  python train_all_models.py --features model/full_features.csv --out-dir model

Prediction using the saved winner:
  python train_all_models.py --predict-only --features model/live_features.csv \
      --model model/winning_model.joblib \
      --prediction-output docs/win/baseball/00_intake/predictions/2026-07-30_MLB.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["home_runs", "away_runs", "home_win"]
ID_COLUMNS = {
    "gid", "date", "season", "visteam", "hometeam", "game_id",
    "home_date", "away_date", "home_season", "away_season",
    "starttime", "gametype", "number",
}
DEFAULT_MODELS = [
    "ridge", "poisson", "negative_binomial", "hist_gbm", "random_forest",
    "xgboost", "lightgbm", "catboost",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="model/full_features.csv")
    p.add_argument("--out-dir", default="model")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--max-features", type=int, default=250,
                   help="Train-only univariate screen; 0 keeps every eligible feature")
    p.add_argument("--max-missing", type=float, default=0.995)
    p.add_argument("--max-categories", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--profile", choices=["quick", "full"], default="quick")
    p.add_argument("--save-all-models", action="store_true",
                   help="Persist every final model; default persists only the winner")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--model", default="model/winning_model.joblib")
    p.add_argument("--prediction-output", default="model/live_predictions.csv")
    return p.parse_args()


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def score_to_win_probability(home_mean: np.ndarray, away_mean: np.ndarray) -> np.ndarray:
    h = np.clip(np.asarray(home_mean, dtype=float), 0.05, 20.0)
    a = np.clip(np.asarray(away_mean, dtype=float), 0.05, 20.0)
    tie = skellam.pmf(0, h, a)
    p = 1.0 - skellam.cdf(0, h, a) + 0.5 * tie
    return np.clip(np.nan_to_num(p, nan=0.5), 0.001, 0.999)


def eligible_features(df: pd.DataFrame, max_missing: float,
                      max_categories: int) -> tuple[list[str], list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    numeric: list[str] = []
    categorical: list[str] = []
    for c in df.columns:
        if c in TARGETS or c in ID_COLUMNS or c.startswith("Unnamed:"):
            rows.append({"feature": c, "eligible": False, "reason": "identifier_or_target"})
            continue
        s = df[c]
        missing = float(s.isna().mean())
        unique = int(s.nunique(dropna=True))
        if missing > max_missing:
            rows.append({"feature": c, "eligible": False, "reason": "too_missing",
                         "missing_pct": missing * 100, "unique": unique})
            continue
        if unique <= 1:
            rows.append({"feature": c, "eligible": False, "reason": "constant",
                         "missing_pct": missing * 100, "unique": unique})
            continue
        n = pd.to_numeric(s, errors="coerce")
        non_null = max(int(s.notna().sum()), 1)
        numeric_fraction = float(n.notna().sum() / non_null)
        if numeric_fraction >= 0.90:
            numeric.append(c)
            kind = "numeric"
            reason = "eligible"
            eligible = True
        elif unique <= max_categories:
            categorical.append(c)
            kind = "categorical"
            reason = "eligible"
            eligible = True
        else:
            kind = "high_cardinality"
            reason = "high_cardinality"
            eligible = False
        rows.append({"feature": c, "eligible": eligible, "reason": reason,
                     "kind": kind, "missing_pct": missing * 100, "unique": unique,
                     "numeric_fraction": numeric_fraction})
    return numeric, categorical, pd.DataFrame(rows)


def screen_numeric_features(train: pd.DataFrame, numeric: list[str], max_features: int,
                            categorical_count: int) -> tuple[list[str], pd.DataFrame]:
    if not numeric:
        return [], pd.DataFrame(columns=["feature", "screening_score"])
    y_home = pd.to_numeric(train["home_runs"], errors="coerce").to_numpy(float)
    y_away = pd.to_numeric(train["away_runs"], errors="coerce").to_numpy(float)
    records = []
    for c in numeric:
        x = pd.to_numeric(train[c], errors="coerce")
        if x.notna().sum() < 25 or x.nunique(dropna=True) <= 1:
            score = 0.0
        else:
            xv = x.fillna(x.median()).to_numpy(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ch = np.corrcoef(xv, y_home)[0, 1]
                ca = np.corrcoef(xv, y_away)[0, 1]
            score = float(np.nanmax(np.abs([ch, ca]))) if np.isfinite([ch, ca]).any() else 0.0
        records.append({"feature": c, "screening_score": score})
    scores = pd.DataFrame(records).sort_values(["screening_score", "feature"], ascending=[False, True])
    if max_features <= 0:
        keep = scores["feature"].tolist()
    else:
        numeric_limit = max(1, max_features - categorical_count)
        keep = scores.head(numeric_limit)["feature"].tolist()
    scores["selected_by_screen"] = scores["feature"].isin(keep)
    return keep, scores


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 dtype=np.float32)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True)


def transformed_mapping(preprocessor: ColumnTransformer, numeric: list[str],
                        categorical: list[str]) -> tuple[list[str], dict[str, list[int]], np.ndarray]:
    names = preprocessor.get_feature_names_out().tolist()
    mapping: dict[str, list[int]] = {}
    idx = 0
    for c in numeric:
        mapping[c] = [idx]
        idx += 1
    if categorical:
        enc: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for c, cats in zip(categorical, enc.categories_):
            mapping[c] = list(range(idx, idx + len(cats)))
            idx += len(cats)
    return names, mapping, np.arange(len(names), dtype=int)


def make_estimator(name: str, profile: str, random_state: int, n_jobs: int):
    quick = profile == "quick"
    if name == "ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 5, 12))),
        ])
    if name == "poisson":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=2.0, max_iter=250, tol=1e-5)),
        ])
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(
            learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
            max_iter=250 if quick else 600, l2_regularization=1.0,
            early_stopping=True, random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=250 if quick else 600, max_features="sqrt",
            min_samples_leaf=3, n_jobs=n_jobs, random_state=random_state,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="reg:squarederror", n_estimators=300 if quick else 800,
            learning_rate=0.035, max_depth=5, min_child_weight=8,
            subsample=0.85, colsample_bytree=0.75, reg_lambda=3.0,
            tree_method="hist", n_jobs=n_jobs, random_state=random_state,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="regression", n_estimators=350 if quick else 900,
            learning_rate=0.03, num_leaves=31, min_child_samples=35,
            subsample=0.85, colsample_bytree=0.75, reg_lambda=3.0,
            n_jobs=n_jobs, random_state=random_state, verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            loss_function="RMSE", iterations=350 if quick else 900,
            learning_rate=0.035, depth=6, l2_leaf_reg=5.0,
            random_seed=random_state, thread_count=n_jobs, verbose=False,
            allow_writing_files=False,
        )
    if name == "negative_binomial":
        return "statsmodels_negative_binomial"
    raise ValueError(f"Unknown model: {name}")


def fit_negative_binomial(X: np.ndarray, y: np.ndarray):
    import statsmodels.api as sm
    y = np.clip(np.asarray(y, dtype=float), 0, None)
    mean = max(float(np.mean(y)), 1e-6)
    var = float(np.var(y))
    alpha = max((var - mean) / (mean * mean), 1e-5)
    Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    model = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha))
    result = model.fit(maxiter=60, disp=0)
    return {"result": result, "alpha": alpha}


def predict_estimator(model: Any, X: np.ndarray, name: str) -> np.ndarray:
    if name == "negative_binomial":
        import statsmodels.api as sm
        Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
        return np.asarray(model["result"].predict(Xc), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def fit_bundle(train: pd.DataFrame, model_name: str, max_features: int,
               max_missing: float, max_categories: int, profile: str,
               random_state: int, n_jobs: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    numeric_all, categorical_all, eligibility = eligible_features(train, max_missing, max_categories)
    selected_numeric, screening = screen_numeric_features(
        train, numeric_all, max_features, len(categorical_all)
    )
    selected_cat = categorical_all
    selected = selected_numeric + selected_cat
    if not selected:
        raise RuntimeError("No eligible features after screening")

    pre = build_preprocessor(selected_numeric, selected_cat)
    X = pre.fit_transform(train[selected])
    y_home = pd.to_numeric(train["home_runs"], errors="coerce").to_numpy(float)
    y_away = pd.to_numeric(train["away_runs"], errors="coerce").to_numpy(float)

    estimator_home = make_estimator(model_name, profile, random_state, n_jobs)
    estimator_away = make_estimator(model_name, profile, random_state + 1, n_jobs)
    if model_name == "negative_binomial":
        estimator_home = fit_negative_binomial(X, y_home)
        estimator_away = fit_negative_binomial(X, y_away)
    else:
        estimator_home.fit(X, y_home)
        estimator_away.fit(X, y_away)

    names, mapping, _ = transformed_mapping(pre, selected_numeric, selected_cat)
    train_means = np.asarray(X, dtype=float).mean(axis=0)
    bundle = {
        "model_name": model_name,
        "preprocessor": pre,
        "home_model": estimator_home,
        "away_model": estimator_away,
        "selected_features": selected,
        "selected_numeric": selected_numeric,
        "selected_categorical": selected_cat,
        "transformed_feature_names": names,
        "raw_to_transformed_indices": mapping,
        "transformed_train_means": train_means,
        "profile": profile,
        "max_features": max_features,
    }
    return bundle, screening, eligibility


def predict_bundle(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = bundle["selected_features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
    X = bundle["preprocessor"].transform(df[features])
    h = np.clip(predict_estimator(bundle["home_model"], X, bundle["model_name"]), 0.05, 20.0)
    a = np.clip(predict_estimator(bundle["away_model"], X, bundle["model_name"]), 0.05, 20.0)
    p = score_to_win_probability(h, a)
    return h, a, p


def evaluate_predictions(test: pd.DataFrame, h: np.ndarray, a: np.ndarray,
                         p: np.ndarray) -> dict[str, float]:
    yh = pd.to_numeric(test["home_runs"], errors="coerce").to_numpy(float)
    ya = pd.to_numeric(test["away_runs"], errors="coerce").to_numpy(float)
    yw = pd.to_numeric(test["home_win"], errors="coerce").to_numpy(float)
    return {
        "home_rmse": rmse(yh, h),
        "away_rmse": rmse(ya, a),
        "score_rmse": float(np.sqrt(np.mean(np.r_[np.square(yh - h), np.square(ya - a)]))),
        "total_rmse": rmse(yh + ya, h + a),
        "home_mae": float(mean_absolute_error(yh, h)),
        "away_mae": float(mean_absolute_error(ya, a)),
        "brier": float(brier_score_loss(yw, p)),
        "home_bias": float(np.mean(h - yh)),
        "away_bias": float(np.mean(a - ya)),
    }


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.full(len(test), float(train["home_runs"].mean()))
    a = np.full(len(test), float(train["away_runs"].mean()))
    return h, a, score_to_win_probability(h, a)


def available_model(name: str) -> tuple[bool, str]:
    module = {
        "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
        "negative_binomial": "statsmodels",
    }.get(name)
    if not module:
        return True, ""
    try:
        __import__(module)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def train_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features, low_memory=False, parse_dates=["date"] if "date" in pd.read_csv(args.features, nrows=0).columns else None)
    needed = set(TARGETS + ["season"])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"Feature file missing required columns: {missing}")
    df = df.dropna(subset=TARGETS).copy()
    seasons = sorted(int(x) for x in pd.unique(df["season"]))
    folds = [(s, [x for x in seasons if x < s]) for s in seasons[1:]]
    if not folds:
        raise SystemExit("At least two seasons are required for rolling validation")

    models = []
    skipped = []
    for name in args.models:
        ok, reason = available_model(name)
        if ok:
            models.append(name)
        else:
            skipped.append({"model": name, "reason": reason})
    if not models:
        raise SystemExit("No requested models are available")

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for test_season, train_seasons in folds:
        train = df[df["season"].isin(train_seasons)].copy()
        test = df[df["season"] == test_season].copy()
        print(f"Fold {train_seasons} -> {test_season}: {len(train):,} train, {len(test):,} test")

        bh, ba, bp = baseline_predictions(train, test)
        metrics = evaluate_predictions(test, bh, ba, bp)
        metric_rows.append({"model": "baseline", "test_season": test_season,
                            "train_seasons": json.dumps(train_seasons), "runtime_sec": 0.0,
                            "selected_features": 0, **metrics})

        for name in models:
            print(f"  fitting {name}...")
            started = time.perf_counter()
            try:
                bundle, _, _ = fit_bundle(
                    train, name, args.max_features, args.max_missing,
                    args.max_categories, args.profile, args.random_state, args.n_jobs,
                )
                h, a, p = predict_bundle(bundle, test.copy())
                elapsed = time.perf_counter() - started
                metrics = evaluate_predictions(test, h, a, p)
                metric_rows.append({
                    "model": name, "test_season": test_season,
                    "train_seasons": json.dumps(train_seasons),
                    "runtime_sec": elapsed,
                    "selected_features": len(bundle["selected_features"]), **metrics,
                })
                pred = test[[c for c in ["gid", "date", "season", "visteam", "hometeam",
                                         "home_runs", "away_runs", "home_win"] if c in test.columns]].copy()
                pred["model"] = name
                pred["pred_home_runs"] = h
                pred["pred_away_runs"] = a
                pred["home_prob"] = p
                pred_frames.append(pred)
            except Exception as exc:
                skipped.append({"model": name, "test_season": test_season,
                                "reason": f"{type(exc).__name__}: {exc}"})
                print(f"    skipped: {exc}")

    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(out_dir / "model_fold_metrics.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "model_cv_predictions.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "model_skipped.csv", index=False)

    candidates = fold_metrics[fold_metrics["model"] != "baseline"].copy()
    summary = candidates.groupby("model", as_index=False).agg(
        folds=("test_season", "nunique"),
        score_rmse=("score_rmse", "mean"), total_rmse=("total_rmse", "mean"),
        brier=("brier", "mean"), home_rmse=("home_rmse", "mean"),
        away_rmse=("away_rmse", "mean"), runtime_sec=("runtime_sec", "sum"),
    )
    for metric in ["score_rmse", "total_rmse", "brier"]:
        summary[f"rank_{metric}"] = summary[metric].rank(method="min")
    summary["overall_rank"] = summary[["rank_score_rmse", "rank_total_rmse", "rank_brier"]].mean(axis=1)
    summary = summary.sort_values(["overall_rank", "score_rmse", "brier"]).reset_index(drop=True)
    if summary.empty:
        raise SystemExit("Every candidate model failed; inspect model_skipped.csv")
    summary.to_csv(out_dir / "model_leaderboard.csv", index=False)
    winner = str(summary.iloc[0]["model"])
    print(f"Winner: {winner}")

    # Final train-only feature screening is saved for the impact audit.
    final_bundles = {}
    final_screen = None
    final_eligibility = None
    final_names = summary["model"].tolist() if args.save_all_models else [winner]
    for name in final_names:
        print(f"Final fit: {name}")
        try:
            bundle, screen, eligibility = fit_bundle(
                df, name, args.max_features, args.max_missing, args.max_categories,
                args.profile, args.random_state, args.n_jobs,
            )
            bundle["fit_seasons"] = seasons
            path = out_dir / f"model_{name}.joblib"
            joblib.dump(bundle, path, compress=3)
            final_bundles[name] = str(path)
            if name == winner:
                final_screen = screen
                final_eligibility = eligibility
                joblib.dump(bundle, out_dir / "winning_model.joblib", compress=3)
        except Exception as exc:
            skipped.append({"model": name, "stage": "final_fit", "reason": str(exc)})

    if final_screen is not None:
        final_screen.to_csv(out_dir / "feature_screening_scores.csv", index=False)
    if final_eligibility is not None:
        final_eligibility.to_csv(out_dir / "feature_eligibility.csv", index=False)
    manifest = {
        "winner": winner,
        "fit_seasons": seasons,
        "rolling_test_seasons": [x[0] for x in folds],
        "models_requested": args.models,
        "models_ranked": summary["model"].tolist(),
        "model_files": final_bundles,
        "max_features": args.max_features,
        "profile": args.profile,
        "win_probability_method": "independent Poisson/Skellam from predicted score means",
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(skipped).to_csv(out_dir / "model_skipped.csv", index=False)


def predict_only(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    df = pd.read_csv(args.features, low_memory=False)
    h, a, p = predict_bundle(bundle, df.copy())
    out = pd.DataFrame({
        "game_id": df.get("gid", df.get("game_id", pd.Series(range(len(df))))),
        "home_team": df.get("hometeam", df.get("home_team", "")),
        "away_team": df.get("visteam", df.get("away_team", "")),
        "home_prob": p,
        "away_prob": 1.0 - p,
        "home_projected_runs": h,
        "away_projected_runs": a,
        "total_projected_runs": h + a,
    })
    path = Path(args.prediction_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} live predictions -> {path.resolve()}")


def main() -> None:
    args = parse_args()
    if args.predict_only:
        predict_only(args)
    else:
        train_all(args)


if __name__ == "__main__":
    main()

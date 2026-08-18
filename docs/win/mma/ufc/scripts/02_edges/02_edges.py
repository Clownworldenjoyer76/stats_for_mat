"""
02_edges.py

Loads the trained model, runs predictions on upcoming matchups,
merges with dratings predictions, computes edge, EV, and Kelly.

Input:
    docs/win/mma/ufc/01_feature_engineering/{date}_ufc_features.csv
    docs/win/mma/ufc/00_intake/predictions/{date}_ufc_predictions.csv
    data/model/ufc_model.pkl

Output:
    docs/win/mma/ufc/02_edges/{date}_ufc_edges.csv
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths ---
FEATURES_DIR = Path("docs/win/mma/ufc/01_feature_engineering")
PREDICTIONS_DIR = Path("docs/win/mma/ufc/00_intake/predictions")
MODEL_PATH = Path("data/model/ufc_model.pkl")
OUT_DIR = Path("docs/win/mma/ufc/02_edges")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load model ---
with MODEL_PATH.open("rb") as f:
    saved = pickle.load(f)

model = saved["xgb"]
median_fill = saved["median_fill"]
FEATURES = saved["features"]

# --- Helpers ---
def implied_prob_from_ml(moneyline: str) -> float | None:
    try:
        ml = float(str(moneyline).replace("+", ""))
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)
    except:
        return None

def compute_ev(model_prob: float, implied_prob: float) -> float:
    if implied_prob <= 0 or implied_prob >= 1:
        return 0.0
    odds = (1 - implied_prob) / implied_prob
    return round((model_prob * odds) - (1 - model_prob), 4)

def compute_kelly(model_prob: float, implied_prob: float, fraction: float = 0.25) -> float:
    if implied_prob <= 0 or implied_prob >= 1:
        return 0.0
    odds = (1 - implied_prob) / implied_prob
    kelly = (model_prob * (odds + 1) - 1) / odds
    return round(max(0, kelly * fraction), 4)

def load_dratings(date_str: str) -> dict:
    path = PREDICTIONS_DIR / f"{date_str}_ufc_predictions.csv"
    lookup = {}
    if not path.exists():
        return lookup
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["fighter_1"].strip(), row["fighter_2"].strip())
            lookup[key] = {
                "dratings_prob_f1": float(row["fighter_1_win_prob"]),
                "dratings_prob_f2": float(row["fighter_2_win_prob"]),
            }
    return lookup

# --- Process each features file ---
feature_files = sorted(FEATURES_DIR.glob("*_ufc_features.csv"))
if not feature_files:
    print("No feature files found.")
    raise SystemExit(1)

for feat_file in feature_files:
    date_str = feat_file.stem.replace("_ufc_features", "")
    dratings = load_dratings(date_str)

    with feat_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows in {feat_file.name}, skipping")
        continue

    df = pd.DataFrame(rows)
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    X = df[FEATURES].fillna(median_fill)
    model_probs = model.predict_proba(X)[:, 1]

    out_rows = []
    for i, row in enumerate(rows):
        f1 = row["fighter_1"].strip()
        f2 = row["fighter_2"].strip()
        ml1 = row["moneyline_f1"].strip()
        ml2 = row["moneyline_f2"].strip()

        ip1_raw = implied_prob_from_ml(ml1)
        ip2_raw = implied_prob_from_ml(ml2)
        total = (ip1_raw or 0) + (ip2_raw or 0)
        ip1 = ip1_raw / total if total > 0 else None
        ip2 = ip2_raw / total if total > 0 else None

        mp1 = round(float(model_probs[i]), 4)
        mp2 = round(1 - mp1, 4)

        edge1 = round(mp1 - ip1, 4) if ip1 else None
        edge2 = round(mp2 - ip2, 4) if ip2 else None

        ev1 = compute_ev(mp1, ip1) if ip1 else None
        ev2 = compute_ev(mp2, ip2) if ip2 else None

        kelly1 = compute_kelly(mp1, ip1) if ip1 else None
        kelly2 = compute_kelly(mp2, ip2) if ip2 else None

        dr = dratings.get((f1, f2)) or dratings.get((f2, f1), {})
        if (f2, f1) in dratings:
            dr_f1 = dr.get("dratings_prob_f2")
            dr_f2 = dr.get("dratings_prob_f1")
        else:
            dr_f1 = dr.get("dratings_prob_f1")
            dr_f2 = dr.get("dratings_prob_f2")

        out_rows.append({
            "match_date": date_str,
            "fighter_1": f1,
            "fighter_2": f2,
            "moneyline_f1": ml1,
            "moneyline_f2": ml2,
            "implied_prob_f1": round(ip1, 4) if ip1 else "",
            "implied_prob_f2": round(ip2, 4) if ip2 else "",
            "model_prob_f1": mp1,
            "model_prob_f2": mp2,
            "dratings_prob_f1": round(dr_f1, 4) if dr_f1 is not None else "",
            "dratings_prob_f2": round(dr_f2, 4) if dr_f2 is not None else "",
            "edge_f1": edge1 if edge1 is not None else "",
            "edge_f2": edge2 if edge2 is not None else "",
            "ev_f1": ev1 if ev1 is not None else "",
            "ev_f2": ev2 if ev2 is not None else "",
            "kelly_f1": kelly1 if kelly1 is not None else "",
            "kelly_f2": kelly2 if kelly2 is not None else "",
        })

    out_file = OUT_DIR / f"{date_str}_ufc_edges.csv"
    if out_rows:
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"WROTE {out_file} ({len(out_rows)} fights)")

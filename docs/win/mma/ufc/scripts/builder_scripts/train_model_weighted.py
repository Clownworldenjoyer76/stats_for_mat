import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("ufc_features.parquet")
df = df.sort_values("match_date").reset_index(drop=True)

df["fight_key"] = df.apply(lambda r: "_".join(sorted([r["fighter_1"], r["fighter_2"]])) + "_" + str(r["match_date"].date()), axis=1)
df["is_mirror"] = df.duplicated(subset="fight_key", keep="first")
original = df[~df["is_mirror"]].copy()

FEATURES = [
    "implied_prob_f1",
    # Rolling stats (2020+)
    "f1_win_rate_all", "f1_win_rate_last5", "f1_streak", "f1_experience",
    "f1_days_since_last", "f1_sos",
    "f2_win_rate_all", "f2_win_rate_last5", "f2_streak", "f2_experience",
    "f2_days_since_last", "f2_sos",
    # Physical
    "f1_age", "f2_age", "f1_reach", "f2_reach", "f1_height", "f2_height",
    # Historical stats (time-gated from ufcstats fight-by-fight)
    "f1_h_career_wr", "f2_h_career_wr",
    "f1_h_career_fights", "f2_h_career_fights",
    "f1_h_slpm", "f2_h_slpm",
    "f1_h_str_acc", "f2_h_str_acc",
    "f1_h_td_acc", "f2_h_td_acc",
    # Differentials
    "diff_win_rate_all", "diff_win_rate_last5", "diff_streak", "diff_experience",
    "diff_days_since_last", "diff_sos", "diff_age", "diff_reach", "diff_height",
    "diff_h_career_wr", "diff_h_career_fights",
    "diff_h_slpm", "diff_h_str_acc", "diff_h_td_acc",
]

TARGET = "result"

train = original[original["match_date"] < "2025-01-01"].copy()
test  = original[original["match_date"] >= "2025-01-01"].copy()
train = train.dropna(subset=["implied_prob_f1"])
test  = test.dropna(subset=["implied_prob_f1"])

median_fill = train[FEATURES].median()
X_train = train[FEATURES].fillna(median_fill)
y_train = train[TARGET]
X_test  = test[FEATURES].fillna(median_fill)
y_test  = test[TARGET]

cutoff = pd.Timestamp("2025-01-01")
days_before_cutoff = (cutoff - train["match_date"]).dt.days
sample_weights = np.exp(-days_before_cutoff * np.log(2) / 365)
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

print(f"Train: {len(X_train)} fights | Test: {len(X_test)} fights | Features: {len(FEATURES)}")

# --- Baseline ---
print("\n--- BASELINE: Implied Probability Only ---")
lr_base = LogisticRegression()
lr_base.fit(train[["implied_prob_f1"]].fillna(0.5), y_train)
base_preds = lr_base.predict_proba(test[["implied_prob_f1"]].fillna(0.5))[:, 1]
base_brier = brier_score_loss(y_test, base_preds)
base_logloss = log_loss(y_test, base_preds)
print(f"Brier Score: {base_brier:.4f}")
print(f"Log Loss:    {base_logloss:.4f}")

# --- Logistic Regression ---
print("\n--- LOGISTIC REGRESSION (recency weighted) ---")
lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.1))])
lr_pipe.fit(X_train, y_train, lr__sample_weight=sample_weights.values)
lr_preds = lr_pipe.predict_proba(X_test)[:, 1]
lr_brier = brier_score_loss(y_test, lr_preds)
lr_logloss = log_loss(y_test, lr_preds)
print(f"Brier Score: {lr_brier:.4f}  (baseline: {base_brier:.4f})")
print(f"Log Loss:    {lr_logloss:.4f}  (baseline: {base_logloss:.4f})")

# --- XGBoost CV ---
print("\n--- XGBOOST (recency weighted) with TimeSeriesSplit CV ---")
tscv = TimeSeriesSplit(n_splits=5)
cv_brier, cv_logloss = [], []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="logloss", random_state=42, verbosity=0)
    model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], sample_weight=sample_weights.values[tr_idx])
    preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    cv_brier.append(brier_score_loss(y_train.iloc[val_idx], preds))
    cv_logloss.append(log_loss(y_train.iloc[val_idx], preds))
    print(f"  Fold {fold+1}: Brier={cv_brier[-1]:.4f}  LogLoss={cv_logloss[-1]:.4f}")
print(f"\nCV Mean Brier:   {np.mean(cv_brier):.4f} (+/- {np.std(cv_brier):.4f})")
print(f"CV Mean LogLoss: {np.mean(cv_logloss):.4f} (+/- {np.std(cv_logloss):.4f})")

# --- Final XGBoost ---
print("\n--- FINAL XGBOOST: Full train -> test ---")
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric="logloss", random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights.values)
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
xgb_brier = brier_score_loss(y_test, xgb_preds)
xgb_logloss = log_loss(y_test, xgb_preds)
print(f"Brier Score: {xgb_brier:.4f}  (baseline: {base_brier:.4f})")
print(f"Log Loss:    {xgb_logloss:.4f}  (baseline: {base_logloss:.4f})")

best_preds = xgb_preds if xgb_brier < lr_brier else lr_preds
best_name = "XGBoost" if xgb_brier < lr_brier else "Logistic Regression"
print(f"\nBest model: {best_name}")

# --- ROI Simulation (capped Kelly) ---
def simulate_roi(test_df, preds, threshold, kelly_fraction=0.25, max_stake_pct=0.10):
    df_sim = test_df.copy().reset_index(drop=True)
    df_sim["model_prob"] = preds
    df_sim["edge"] = df_sim["model_prob"] - df_sim["implied_prob_f1"]
    bets = df_sim[df_sim["edge"] > threshold].copy()
    if len(bets) == 0:
        return None, 0, 0
    starting = 1000.0
    bankroll = starting
    max_stake = starting * max_stake_pct
    profits = []
    for _, row in bets.iterrows():
        p = row["model_prob"]
        try:
            ml = float(str(row["moneyline_f1"]).replace("+", ""))
            odds = ml / 100 if ml > 0 else 100 / abs(ml)
        except:
            continue
        kelly = (p * (odds + 1) - 1) / odds
        stake = max(0, min(kelly * kelly_fraction * starting, max_stake))
        profit = stake * odds if row["result"] == 1 else -stake
        bankroll += profit
        profits.append(profit)
    roi = (bankroll - starting) / starting * 100
    win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100 if profits else 0
    return roi, len(bets), win_rate

print("\n--- ROI SIMULATION (ALL FIGHTERS) ---")
print(f"\n{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for t in [0.03, 0.05, 0.07, 0.10]:
    roi, n, wr = simulate_roi(test, best_preds, t)
    if roi is not None: print(f"{t:>10.0%} {n:>6} {wr:>6.1f}% {roi:>7.1f}%")
    else: print(f"{t:>10.0%} {'0':>6} {'N/A':>7} {'N/A':>8}")

print("\n--- ROI SIMULATION (UNDERDOGS ONLY) ---")
mask = test["implied_prob_f1"].values < 0.5
test_dogs = test[mask].copy().reset_index(drop=True)
dog_preds = xgb_model.predict_proba(X_test[mask])[:, 1] if best_name == "XGBoost" else lr_pipe.predict_proba(X_test[mask])[:, 1]
print(f"Underdog fights: {len(test_dogs)}")
print(f"\n{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for t in [0.03, 0.05, 0.07, 0.10]:
    roi, n, wr = simulate_roi(test_dogs, dog_preds, t)
    if roi is not None: print(f"{t:>10.0%} {n:>6} {wr:>6.1f}% {roi:>7.1f}%")
    else: print(f"{t:>10.0%} {'0':>6} {'N/A':>7} {'N/A':>8}")

# --- Feature Importance ---
print("\n--- FEATURE IMPORTANCE (top 15) ---")
importance = pd.Series(xgb_model.feature_importances_, index=FEATURES)
print(importance.sort_values(ascending=False).head(15).to_string())

# --- Save ---
with open("ufc_model.pkl", "wb") as f:
    pickle.dump({"xgb": xgb_model, "lr": lr_pipe, "median_fill": median_fill,
                 "features": FEATURES, "best": best_name}, f)

test = test.reset_index(drop=True)
test["model_prob"] = best_preds
test["edge"] = test["model_prob"] - test["implied_prob_f1"]
test[["match_date","fighter_1","fighter_2","result","implied_prob_f1","model_prob","edge"]].to_csv("test_predictions.csv", index=False)
print("\nSaved: ufc_model.pkl, test_predictions.csv")
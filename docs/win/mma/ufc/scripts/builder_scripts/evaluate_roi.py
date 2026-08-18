import pandas as pd
import numpy as np
import pickle

# Load model and data
with open("ufc_model.pkl", "rb") as f:
    saved = pickle.load(f)

xgb_model = saved["xgb"]
median_fill = saved["median_fill"]
FEATURES = saved["features"]

df = pd.read_parquet("ufc_features.parquet")
df = df.sort_values("match_date").reset_index(drop=True)
df["fight_key"] = df.apply(lambda r: "_".join(sorted([r["fighter_1"], r["fighter_2"]])) + "_" + str(r["match_date"].date()), axis=1)
df["is_mirror"] = df.duplicated(subset="fight_key", keep="first")
original = df[~df["is_mirror"]].copy()

test = original[original["match_date"] >= "2025-01-01"].copy()
test = test.dropna(subset=["implied_prob_f1"])
X_test = test[FEATURES].fillna(median_fill)
test = test.reset_index(drop=True)
test["model_prob"] = xgb_model.predict_proba(X_test)[:, 1]
test["edge"] = test["model_prob"] - test["implied_prob_f1"]

def simulate_roi(df_sim, threshold, kelly_fraction=0.25, max_stake_pct=0.10):
    """
    Simulate ROI with flat starting bankroll.
    max_stake_pct: cap any single bet at this % of STARTING bankroll (not running bankroll)
    to prevent compounding explosion.
    """
    bets = df_sim[df_sim["edge"] > threshold].copy()
    if len(bets) == 0:
        return None, 0, 0

    starting_bankroll = 1000.0
    bankroll = starting_bankroll
    max_single_stake = starting_bankroll * max_stake_pct  # cap at 10% of starting bankroll
    profits = []

    for _, row in bets.iterrows():
        p = row["model_prob"]
        try:
            ml = float(str(row["moneyline_f1"]).replace("+", ""))
            odds = ml / 100 if ml > 0 else 100 / abs(ml)
        except:
            continue

        kelly = (p * (odds + 1) - 1) / odds
        stake = kelly * kelly_fraction * starting_bankroll  # use STARTING bankroll for sizing
        stake = max(0, min(stake, max_single_stake))  # cap stake

        profit = stake * odds if row["result"] == 1 else -stake
        bankroll += profit
        profits.append(profit)

    roi = (bankroll - starting_bankroll) / starting_bankroll * 100
    win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100 if profits else 0
    total_wagered = sum(abs(p) / (odds if p > 0 else 1) for p, odds in
                       zip(profits, [1]*len(profits)))  # approx
    return roi, len(bets), win_rate

print(f"Test fights: {len(test)}")
print(f"Date range: {test['match_date'].min().date()} to {test['match_date'].max().date()}")

print("\n--- ROI SIMULATION (ALL FIGHTERS) ---")
print(f"Kelly fraction: 25% | Max stake: 10% of bankroll per bet")
print(f"\n{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for threshold in [0.03, 0.05, 0.07, 0.10]:
    roi, n_bets, win_rate = simulate_roi(test, threshold)
    if roi is not None:
        print(f"{threshold:>10.0%} {n_bets:>6} {win_rate:>6.1f}% {roi:>7.1f}%")
    else:
        print(f"{threshold:>10.0%} {'0':>6} {'N/A':>7} {'N/A':>8}")

print("\n--- ROI SIMULATION (UNDERDOGS ONLY, implied_prob < 0.5) ---")
mask = test["implied_prob_f1"] < 0.5
test_dogs = test[mask].copy().reset_index(drop=True)
print(f"Underdog fights: {len(test_dogs)}")
print(f"\n{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for threshold in [0.03, 0.05, 0.07, 0.10]:
    roi, n_bets, win_rate = simulate_roi(test_dogs, threshold)
    if roi is not None:
        print(f"{threshold:>10.0%} {n_bets:>6} {win_rate:>6.1f}% {roi:>7.1f}%")
    else:
        print(f"{threshold:>10.0%} {'0':>6} {'N/A':>7} {'N/A':>8}")

print("\n--- ROI SIMULATION (FAVORITES ONLY, implied_prob >= 0.5) ---")
mask_fav = test["implied_prob_f1"] >= 0.5
test_favs = test[mask_fav].copy().reset_index(drop=True)
print(f"Favorite fights: {len(test_favs)}")
print(f"\n{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for threshold in [0.03, 0.05, 0.07, 0.10]:
    roi, n_bets, win_rate = simulate_roi(test_favs, threshold)
    if roi is not None:
        print(f"{threshold:>10.0%} {n_bets:>6} {win_rate:>6.1f}% {roi:>7.1f}%")
    else:
        print(f"{threshold:>10.0%} {'0':>6} {'N/A':>7} {'N/A':>8}")

# Save updated predictions
test[["match_date","fighter_1","fighter_2","result","implied_prob_f1","model_prob","edge"]].to_csv("test_predictions.csv", index=False)
print("\nSaved: test_predictions.csv")

import pandas as pd
import json
import numpy as np

df = pd.read_parquet("ufc_master_clean.parquet")
with open("fighter_attributes.json") as f:
    attrs = json.load(f)
hist_df = pd.read_parquet("fighter_historical_stats.parquet")

df = df.sort_values("match_date").reset_index(drop=True)

# --- Prep historical stats: fix fighter_1/fighter_2 swap ---
# hist_df fighter_1 is always the fighter whose page was scraped (alphabetical order from scraper)
# ufc_master_clean fighter_1 may differ — use fight_key to align

hist_df["fight_key"] = hist_df.apply(
    lambda r: "_".join(sorted([r["fighter_1"], r["fighter_2"]])) + "_" + str(r["match_date"].date()), axis=1
)
df["fight_key"] = df.apply(
    lambda r: "_".join(sorted([r["fighter_1"], r["fighter_2"]])) + "_" + str(r["match_date"].date()), axis=1
)

# Build lookup: fight_key -> {f1_stats, f2_stats} aligned to df's fighter_1
hist_lookup = {}
for _, hrow in hist_df.iterrows():
    key = hrow["fight_key"]
    hist_lookup[key] = {
        "hist_f1": hrow["fighter_1"],  # which fighter is f1 in hist
        "data": hrow
    }

# --- Attribute helpers ---
def parse_dob(name):
    try: return pd.to_datetime(attrs[name]["dob"])
    except: return None

def parse_height_inches(name):
    try:
        h = attrs[name]["height"]
        parts = h.replace('"', '').split("'")
        return int(parts[0]) * 12 + int(parts[1].strip())
    except: return None

def parse_reach(name):
    try: return float(attrs[name]["reach"].replace('"', '').strip())
    except: return None

# --- Build fight history for rolling stats ---
def build_fighter_history(df):
    history = {}
    for _, row in df.iterrows():
        date = row["match_date"]
        for fighter, result in [(row["fighter_1"], row["result_fighter_1"]),
                                (row["fighter_2"], row["result_fighter_2"])]:
            if fighter not in history:
                history[fighter] = []
            history[fighter].append((date, 1 if result == "Win" else 0))
    for f in history:
        history[f] = sorted(history[f], key=lambda x: x[0])
    return history

history = build_fighter_history(df)

def get_stats(fighter, fight_date, history):
    fights = [(d, w) for d, w in history.get(fighter, []) if d < fight_date]
    if not fights:
        return {"win_rate_all": None, "win_rate_last5": None, "streak": 0, "experience": 0, "days_since_last": None}
    wins = [w for _, w in fights]
    dates = [d for d, _ in fights]
    streak = 0
    last = wins[-1]
    for w in reversed(wins):
        if w == last: streak += 1
        else: break
    streak = streak if last == 1 else -streak
    return {
        "win_rate_all": np.mean(wins),
        "win_rate_last5": np.mean(wins[-5:]),
        "streak": streak,
        "experience": len(fights),
        "days_since_last": (fight_date - dates[-1]).days,
    }

def get_sos(fighter, fight_date, history, df):
    past_opps = []
    for _, row in df.iterrows():
        if row["match_date"] >= fight_date: continue
        if row["fighter_1"] == fighter: past_opps.append(row["fighter_2"])
        elif row["fighter_2"] == fighter: past_opps.append(row["fighter_1"])
    if not past_opps: return None
    rates = []
    for opp in past_opps:
        opp_fights = [(d, w) for d, w in history.get(opp, []) if d < fight_date]
        if opp_fights: rates.append(np.mean([w for _, w in opp_fights]))
    return np.mean(rates) if rates else None

def implied_prob(moneyline):
    try:
        ml = float(str(moneyline).replace("+", ""))
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)
    except: return None

def diff(a, b):
    if a is None or b is None: return 0
    return a - b

print("Building features...")
rows = []
for idx, row in df.iterrows():
    if idx % 200 == 0:
        print(f"  Row {idx}/{len(df)}")

    f1, f2 = row["fighter_1"], row["fighter_2"]
    date = row["match_date"]
    key = row["fight_key"]

    s1 = get_stats(f1, date, history)
    s2 = get_stats(f2, date, history)
    sos1 = get_sos(f1, date, history, df)
    sos2 = get_sos(f2, date, history, df)

    ip1_raw = implied_prob(row["moneyline_fighter_1"])
    ip2_raw = implied_prob(row["moneyline_fighter_2"])
    if ip1_raw and ip2_raw:
        total = ip1_raw + ip2_raw
        ip1, ip2 = ip1_raw / total, ip2_raw / total
    else:
        ip1 = ip2 = None

    age1 = (date - parse_dob(f1)).days / 365.25 if parse_dob(f1) else None
    age2 = (date - parse_dob(f2)).days / 365.25 if parse_dob(f2) else None

    # --- Historical stats (time-gated, from ufcstats fight-by-fight) ---
    h1 = {}
    h2 = {}
    if key in hist_lookup:
        hdata = hist_lookup[key]["data"]
        hist_f1 = hist_lookup[key]["hist_f1"]
        if hist_f1 == f1:
            # hist fighter_1 matches our fighter_1
            h1 = {k.replace("f1_", ""): v for k, v in hdata.items() if k.startswith("f1_h_")}
            h2 = {k.replace("f2_", ""): v for k, v in hdata.items() if k.startswith("f2_h_")}
        else:
            # swapped — hist fighter_1 is our fighter_2
            h1 = {k.replace("f2_", ""): v for k, v in hdata.items() if k.startswith("f2_h_")}
            h2 = {k.replace("f1_", ""): v for k, v in hdata.items() if k.startswith("f1_h_")}

    def hget(d, key, default=None):
        val = d.get(key, default)
        if pd.isna(val) if val is not None else False: return default
        return val

    h_career_wr1 = hget(h1, "h_career_wr")
    h_career_wr2 = hget(h2, "h_career_wr")
    h_career_fights1 = hget(h1, "h_career_fights")
    h_career_fights2 = hget(h2, "h_career_fights")
    h_slpm1 = hget(h1, "h_slpm")
    h_slpm2 = hget(h2, "h_slpm")
    h_str_acc1 = min(hget(h1, "h_str_acc") or 0, 1.0)  # cap at 1.0
    h_str_acc2 = min(hget(h2, "h_str_acc") or 0, 1.0)
    h_td_acc1 = min(hget(h1, "h_td_acc") or 0, 1.0)
    h_td_acc2 = min(hget(h2, "h_td_acc") or 0, 1.0)

    result = 1 if row["result_fighter_1"] == "Win" else 0

    feature_row = {
        "match_date": date,
        "fighter_1": f1,
        "fighter_2": f2,
        "result": result,
        "moneyline_f1": row["moneyline_fighter_1"],
        "moneyline_f2": row["moneyline_fighter_2"],
        "implied_prob_f1": ip1,
        "implied_prob_f2": ip2,

        # Rolling stats (from your data, 2020+)
        "f1_win_rate_all": s1["win_rate_all"], "f1_win_rate_last5": s1["win_rate_last5"],
        "f1_streak": s1["streak"], "f1_experience": s1["experience"],
        "f1_days_since_last": s1["days_since_last"], "f1_sos": sos1,
        "f2_win_rate_all": s2["win_rate_all"], "f2_win_rate_last5": s2["win_rate_last5"],
        "f2_streak": s2["streak"], "f2_experience": s2["experience"],
        "f2_days_since_last": s2["days_since_last"], "f2_sos": sos2,

        # Physical
        "f1_age": age1, "f2_age": age2,
        "f1_reach": parse_reach(f1), "f2_reach": parse_reach(f2),
        "f1_height": parse_height_inches(f1), "f2_height": parse_height_inches(f2),

        # Historical stats (time-gated from ufcstats)
        "f1_h_career_wr": h_career_wr1, "f2_h_career_wr": h_career_wr2,
        "f1_h_career_fights": h_career_fights1, "f2_h_career_fights": h_career_fights2,
        "f1_h_slpm": h_slpm1, "f2_h_slpm": h_slpm2,
        "f1_h_str_acc": h_str_acc1, "f2_h_str_acc": h_str_acc2,
        "f1_h_td_acc": h_td_acc1, "f2_h_td_acc": h_td_acc2,

        # Differentials
        "diff_win_rate_all": diff(s1["win_rate_all"], s2["win_rate_all"]),
        "diff_win_rate_last5": diff(s1["win_rate_last5"], s2["win_rate_last5"]),
        "diff_streak": s1["streak"] - s2["streak"],
        "diff_experience": s1["experience"] - s2["experience"],
        "diff_days_since_last": diff(s1["days_since_last"], s2["days_since_last"]),
        "diff_sos": diff(sos1, sos2),
        "diff_age": diff(age1, age2),
        "diff_reach": diff(parse_reach(f1), parse_reach(f2)),
        "diff_height": diff(parse_height_inches(f1), parse_height_inches(f2)),
        "diff_h_career_wr": diff(h_career_wr1, h_career_wr2),
        "diff_h_career_fights": diff(h_career_fights1, h_career_fights2),
        "diff_h_slpm": diff(h_slpm1, h_slpm2),
        "diff_h_str_acc": diff(h_str_acc1, h_str_acc2),
        "diff_h_td_acc": diff(h_td_acc1, h_td_acc2),
    }
    rows.append(feature_row)

features = pd.DataFrame(rows)

# --- Mirror rows ---
mirror = features.copy()
mirror["fighter_1"] = features["fighter_2"]
mirror["fighter_2"] = features["fighter_1"]
mirror["result"] = 1 - features["result"]
mirror["implied_prob_f1"] = features["implied_prob_f2"]
mirror["implied_prob_f2"] = features["implied_prob_f1"]
mirror["moneyline_f1"] = features["moneyline_f2"]
mirror["moneyline_f2"] = features["moneyline_f1"]

swap_cols = ["win_rate_all","win_rate_last5","streak","experience","days_since_last","sos",
             "age","reach","height","h_career_wr","h_career_fights","h_slpm","h_str_acc","h_td_acc"]
for col in swap_cols:
    mirror[f"f1_{col}"] = features[f"f2_{col}"]
    mirror[f"f2_{col}"] = features[f"f1_{col}"]

diff_cols = ["win_rate_all","win_rate_last5","streak","experience","days_since_last","sos",
             "age","reach","height","h_career_wr","h_career_fights","h_slpm","h_str_acc","h_td_acc"]
for col in diff_cols:
    if f"diff_{col}" in features.columns:
        mirror[f"diff_{col}"] = -features[f"diff_{col}"]

full = pd.concat([features, mirror], ignore_index=True)
full = full.sort_values("match_date").reset_index(drop=True)

print(f"\nFeature matrix shape: {full.shape}")
missing = full.isnull().sum()
print(f"Missing values:\n{missing[missing > 0]}")

full.to_parquet("ufc_features.parquet", index=False)
print("\nSaved to ufc_features.parquet")
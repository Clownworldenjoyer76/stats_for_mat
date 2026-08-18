"""
01_build_features.py

Builds features for upcoming UFC matchups by combining:
- Odds from docs/win/mma/ufc/00_intake/sportsbook/*_ufc_odds.csv
- Fighter attributes from data/model/fighter_attributes.json
- Fighter historical stats from data/model/fighter_historical_stats.parquet
- Rolling fight history from data/model/ufc_master_clean.parquet

Output: docs/win/mma/ufc/01_feature_engineering/{date}_ufc_features.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths ---
ODDS_DIR = Path("docs/win/mma/ufc/00_intake/sportsbook")
ATTRS_PATH = Path("data/model/fighter_attributes.json")
HISTORY_PATH = Path("data/model/fighter_history.json")
MASTER_PATH = Path("data/model/ufc_master_clean.parquet")
OUT_DIR = Path("docs/win/mma/ufc/01_feature_engineering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
with ATTRS_PATH.open(encoding="utf-8") as f:
    attrs = json.load(f)

with HISTORY_PATH.open(encoding="utf-8") as f:
    raw_history = json.load(f)

fighter_history = {
    name: [
        {**fight, "date": datetime.strptime(fight["date"], "%Y-%m-%d")}
        for fight in fights
    ]
    for name, fights in raw_history.items()
}

master = pd.read_parquet(MASTER_PATH)
master = master.sort_values("match_date").reset_index(drop=True)

# --- Attribute helpers ---
def parse_dob(name):
    try:
        return pd.to_datetime(attrs[name]["dob"])
    except:
        return None

def parse_height_inches(name):
    try:
        h = attrs[name]["height"]
        parts = h.replace('"', '').split("'")
        return int(parts[0]) * 12 + int(parts[1].strip())
    except:
        return None

def parse_reach(name):
    try:
        return float(attrs[name]["reach"].replace('"', '').strip())
    except:
        return None

# --- Rolling stats from master fight history ---
def build_rolling_history(master):
    history = {}
    for _, row in master.iterrows():
        date = row["match_date"]
        for fighter, result in [
            (row["fighter_1"], row["result_fighter_1"]),
            (row["fighter_2"], row["result_fighter_2"]),
        ]:
            if fighter not in history:
                history[fighter] = []
            history[fighter].append((date, 1 if result == "Win" else 0))
    for f in history:
        history[f] = sorted(history[f], key=lambda x: x[0])
    return history

rolling_history = build_rolling_history(master)

def get_rolling_stats(fighter, fight_date):
    fights = [(d, w) for d, w in rolling_history.get(fighter, []) if d < fight_date]
    if not fights:
        return {"win_rate_all": None, "win_rate_last5": None, "streak": 0,
                "experience": 0, "days_since_last": None}
    wins = [w for _, w in fights]
    dates = [d for d, _ in fights]
    streak, last = 0, wins[-1]
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

def get_sos(fighter, fight_date):
    past_opps = []
    for _, row in master.iterrows():
        if row["match_date"] >= fight_date: continue
        if row["fighter_1"] == fighter: past_opps.append(row["fighter_2"])
        elif row["fighter_2"] == fighter: past_opps.append(row["fighter_1"])
    if not past_opps: return None
    rates = []
    for opp in past_opps:
        opp_fights = [(d, w) for d, w in rolling_history.get(opp, []) if d < fight_date]
        if opp_fights: rates.append(np.mean([w for _, w in opp_fights]))
    return np.mean(rates) if rates else None

# --- Historical stats (time-gated) ---
def get_historical_stats(fighter, fight_date):
    fights = [f for f in fighter_history.get(fighter, []) if f["date"] < fight_date]
    if not fights:
        return {}
    wins = sum(1 for f in fights if f["result"] == "win")
    losses = sum(1 for f in fights if f["result"] == "loss")
    total_min = sum(f["minutes"] for f in fights)
    sig_landed = sum(f["sig_landed"] for f in fights)
    sig_attempted = sum(f["sig_attempted"] for f in fights)
    td_landed = sum(f["td_landed"] for f in fights)
    td_attempted = sum(f["td_attempted"] for f in fights)
    return {
        "h_career_wins": wins,
        "h_career_losses": losses,
        "h_career_fights": wins + losses,
        "h_career_wr": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "h_slpm": round(sig_landed / total_min, 4) if total_min > 0 else 0,
        "h_str_acc": min(round(sig_landed / sig_attempted, 4), 1.0) if sig_attempted > 0 else 0,
        "h_td_acc": min(round(td_landed / td_attempted, 4), 1.0) if td_attempted > 0 else 0,
    }

# --- Implied probability ---
def implied_prob(moneyline):
    try:
        ml = float(str(moneyline).replace("+", ""))
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)
    except:
        return None

def vig_removed(ip1, ip2):
    if ip1 and ip2:
        total = ip1 + ip2
        return ip1 / total, ip2 / total
    return ip1, ip2

def diff(a, b):
    if a is None or b is None: return 0
    return a - b

# --- Process each odds file ---
odds_files = sorted(ODDS_DIR.glob("*_ufc_odds.csv"))
if not odds_files:
    print("No odds files found.")
    raise SystemExit(1)

for odds_file in odds_files:
    date_str = odds_file.stem.replace("_ufc_odds", "")
    try:
        fight_date = pd.Timestamp(datetime.strptime(date_str, "%Y_%m_%d"))
    except:
        print(f"Could not parse date from {odds_file.name}, skipping")
        continue

    with odds_file.open(encoding="utf-8") as f:
        odds_rows = list(csv.DictReader(f))

    if not odds_rows:
        print(f"No rows in {odds_file.name}, skipping")
        continue

    out_rows = []
    for row in odds_rows:
        f1 = row["fighter_1"].strip()
        f2 = row["fighter_2"].strip()
        ml1 = row["moneyline_fighter_1"].strip()
        ml2 = row["moneyline_fighter_2"].strip()

        ip1_raw = implied_prob(ml1)
        ip2_raw = implied_prob(ml2)
        ip1, ip2 = vig_removed(ip1_raw, ip2_raw)

        s1 = get_rolling_stats(f1, fight_date)
        s2 = get_rolling_stats(f2, fight_date)
        sos1 = get_sos(f1, fight_date)
        sos2 = get_sos(f2, fight_date)
        h1 = get_historical_stats(f1, fight_date)
        h2 = get_historical_stats(f2, fight_date)

        age1 = (fight_date - parse_dob(f1)).days / 365.25 if parse_dob(f1) else None
        age2 = (fight_date - parse_dob(f2)).days / 365.25 if parse_dob(f2) else None

        out_rows.append({
            "match_date": date_str,
            "fighter_1": f1,
            "fighter_2": f2,
            "moneyline_f1": ml1,
            "moneyline_f2": ml2,
            "implied_prob_f1": round(ip1, 4) if ip1 else "",
            "implied_prob_f2": round(ip2, 4) if ip2 else "",
            "f1_win_rate_all": round(s1["win_rate_all"], 4) if s1["win_rate_all"] is not None else "",
            "f1_win_rate_last5": round(s1["win_rate_last5"], 4) if s1["win_rate_last5"] is not None else "",
            "f1_streak": s1["streak"],
            "f1_experience": s1["experience"],
            "f1_days_since_last": s1["days_since_last"] if s1["days_since_last"] is not None else "",
            "f1_sos": round(sos1, 4) if sos1 is not None else "",
            "f1_age": round(age1, 2) if age1 else "",
            "f1_reach": parse_reach(f1) or "",
            "f1_height": parse_height_inches(f1) or "",
            "f2_win_rate_all": round(s2["win_rate_all"], 4) if s2["win_rate_all"] is not None else "",
            "f2_win_rate_last5": round(s2["win_rate_last5"], 4) if s2["win_rate_last5"] is not None else "",
            "f2_streak": s2["streak"],
            "f2_experience": s2["experience"],
            "f2_days_since_last": s2["days_since_last"] if s2["days_since_last"] is not None else "",
            "f2_sos": round(sos2, 4) if sos2 is not None else "",
            "f2_age": round(age2, 2) if age2 else "",
            "f2_reach": parse_reach(f2) or "",
            "f2_height": parse_height_inches(f2) or "",
            "f1_h_career_wr": round(h1.get("h_career_wr", 0), 4) if h1 else "",
            "f1_h_career_fights": h1.get("h_career_fights", "") if h1 else "",
            "f1_h_slpm": h1.get("h_slpm", "") if h1 else "",
            "f1_h_str_acc": h1.get("h_str_acc", "") if h1 else "",
            "f1_h_td_acc": h1.get("h_td_acc", "") if h1 else "",
            "f2_h_career_wr": round(h2.get("h_career_wr", 0), 4) if h2 else "",
            "f2_h_career_fights": h2.get("h_career_fights", "") if h2 else "",
            "f2_h_slpm": h2.get("h_slpm", "") if h2 else "",
            "f2_h_str_acc": h2.get("h_str_acc", "") if h2 else "",
            "f2_h_td_acc": h2.get("h_td_acc", "") if h2 else "",
            "diff_win_rate_all": round(diff(s1["win_rate_all"], s2["win_rate_all"]), 4),
            "diff_win_rate_last5": round(diff(s1["win_rate_last5"], s2["win_rate_last5"]), 4),
            "diff_streak": s1["streak"] - s2["streak"],
            "diff_experience": s1["experience"] - s2["experience"],
            "diff_days_since_last": diff(s1["days_since_last"], s2["days_since_last"]),
            "diff_sos": round(diff(sos1, sos2), 4),
            "diff_age": round(diff(age1, age2), 2),
            "diff_reach": diff(parse_reach(f1), parse_reach(f2)),
            "diff_height": diff(parse_height_inches(f1), parse_height_inches(f2)),
            "diff_h_career_wr": round(diff(h1.get("h_career_wr"), h2.get("h_career_wr")), 4) if h1 and h2 else "",
            "diff_h_career_fights": diff(h1.get("h_career_fights"), h2.get("h_career_fights")) if h1 and h2 else "",
            "diff_h_slpm": round(diff(h1.get("h_slpm"), h2.get("h_slpm")), 4) if h1 and h2 else "",
            "diff_h_str_acc": round(diff(h1.get("h_str_acc"), h2.get("h_str_acc")), 4) if h1 and h2 else "",
            "diff_h_td_acc": round(diff(h1.get("h_td_acc"), h2.get("h_td_acc")), 4) if h1 and h2 else "",
        })

    out_file = OUT_DIR / f"{date_str}_ufc_features.csv"
    if out_rows:
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"WROTE {out_file} ({len(out_rows)} fights)")
    else:
        print(f"No output rows for {date_str}")

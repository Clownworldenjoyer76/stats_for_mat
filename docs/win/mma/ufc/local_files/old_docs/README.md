# UFC Fight Prediction Model

A machine learning pipeline to predict UFC fight outcomes and identify betting edges against sportsbook implied probabilities.

---

## Repo Structure

```
bet_tracker/
├── data/
│   └── model/    
│       ├── fighter_attributes.json
│       ├── fighter_history.json
│       ├── fighter_historical_stats.parquet
│       ├── ufc_master_clean.parquet
│       └── ufc_model.pkl
├── docs/win/mma/ufc/
│   ├── 00_intake/
│   │   ├── sportsbook/                 # *_ufc_odds.csv (output of 00_scrape_odds.py)
│   │   └── predictions/                # *_ufc_predictions.csv (output of 00_scrape_predictions.py)
│   ├── 01_feature_engineering/         # *_ufc_features.csv (output of 01_build_features.py)
│   ├── 02_edges/                       # *_ufc_edges.csv (output of 02_edges.py)
│   ├── 03_select/                      # *_ufc_select.csv (output of 03_select.py)
│   ├── config/
│   │   └── markets.yaml                # adjustable filter settings for 03_select.py
│   └── scripts/
│       ├── 00_intake/
│       │   ├── 00_scrape_odds.py
│       │   ├── 00_scrape_predictions.py
│       │   └── 00_name_normalization.py
│       ├── 01_feature_engineering/
│       │   └── 01_build_features.py
│       ├── 02_edges/
│       │   └── 02_edges.py
│       ├── 03_select/
│       │   └── 03_select.py
│       └── builder_scripts/            # local machine only — model training and data prep
│           ├── parse_ufc_files.py
│           ├── fix_fighter_names.py
│           ├── apply_corrections.py
│           ├── scrape_fighter_stats.py
│           ├── fix_unmatched_fighters.py
│           ├── scrape_historical_stats.py
│           ├── build_features.py
│           ├── train_model_weighted.py
│           ├── evaluate_roi.py
│           └── phase6_backtest.py
├── mappings/mma/ufc/
│   ├── fighter_name_map.csv            # alias -> canonical name mapping
│   └── no_map_fighter_name.csv         # fighters not found in name map (auto-generated)
└── .github/workflows/
    └── ufc_daily_pipeline.yml
```
## Local Machine Setup

All builder scripts run from `C:\Users\ntmal\Downloads\bet_tracker_files\UFC_Master\`.

```powershell
pip install pandas pyarrow requests beautifulsoup4 scikit-learn xgboost pyyaml
```

## When to Update `data/model/` Files

These 5 files live on the `model-data` branch at `bet_tracker/data/model/`. The daily pipeline pulls them automatically. Update them as follows:

### `ufc_model.pkl`
- **When:** After retraining the model on your local machine
- **How often:** Monthly — after enough new UFC event results have been added to retrain on

### `ufc_master_clean.parquet`
- **When:** After adding new completed UFC event CSVs and re-running `parse_ufc_files.py` and `apply_corrections.py` locally
- **How often:** After every UFC event — roughly 2-3 times per month

### `fighter_attributes.json`
- **When:** After running `scrape_fighter_stats.py` or `add_missing_fighters.py` locally to add new fighters
- **How often:** Only when new fighters appear in `no_map_fighter_name.csv` after a pipeline run

### `fighter_history.json` and `fighter_historical_stats.parquet`
- **When:** After running `scrape_historical_stats.py` locally
- **How often:** Same as `ufc_master_clean.parquet` — after every UFC event



---

## New Fighter Process

How to run:

Download mappings/mma/ufc/no_map_fighter_name.csv from your GitHub repo
Save it to Downloads\bet_tracker_files\UFC_Master\mappings\mma\ufc\no_map_fighter_name.csv
From the UFC_Master folder, run:

powershell python scripts\builder_scripts\add_missing_fighters.py

After it finishes, upload the updated fighter_attributes.json to the model-data folder
For fighters reported as "Not found on ufcstats"
    1. Add them manually to mappings/mma/ufc/fighter_name_map.csv 
    2. Add them manually to the local folder at UFC_Master\mappings\mma\ufc\ALERT-----missing_attributes_retry.txt
    
---

## Model Retraining Process (Local Machine)

Run these scripts in order from `C:\Users\ntmal\Downloads\bet_tracker_files\UFC_Master\scripts\builder_scripts\`:
1. Add completed matches to UFC_Master\match_results

2. Run from:

```cmd
C:\Users\ntmal\Downloads\bet_tracker_files\UFC_Master
```
3. Steps:

```cmd
python scripts\builder_scripts\parse_ufc_files.py
python scripts\builder_scripts\apply_corrections.py
python scripts\builder_scripts\audit_missing_attributes.py
## pip install requests beautifulsoup4 pandas pyarrow
python scripts\builder_scripts\scrape_historical_stats.py
## scrape_historical_stats.py takes ~35-40 min to run
python scripts\builder_scripts\build_features.py
python scripts\builder_scripts\train_model_weighted.py
python scripts\builder_scripts\evaluate_roi.py
```





--
wHAT THEY DO:

1. `parse_ufc_files.py` — parses Match results CSV files into master dataset
2. `apply_corrections.py` — fixes mangled fighter names
3. `audit_missing_attributes.py` — finds errors
4. `scrape_historical_stats.py` — scrapes fight-by-fight history from ufcstats (~35-40 min)
5. `build_features.py` — builds full feature matrix
6. `train_model_weighted.py` — retrains XGBoost model with recency weighting
7. `evaluate_roi.py` — evaluates ROI on test set
8. Upload updated `ufc_model.pkl`, `ufc_master_clean.parquet`, `fighter_history.json`, `fighter_historical_stats.parquet` to `model-data` branch

---

## markets.yaml — Adjusting Filters

Located at `docs/win/mma/ufc/config/markets.yaml`. Edit directly on GitHub to adjust pick filters without rerunning anything — changes take effect on the next pipeline run.

```yaml
ufc:
  moneyline:
    enabled: true
    pick_preference: best_ev        # options: best_ev, best_edge, best_kelly, best_model_prob, best_dratings_prob
    odds_bands:
      - [-360, -150]
      - [-120, 150]
    edge_bands:
      - [0.01, 0.25]
    ev_bands:
      - [0.01, 0.25]
    kelly_bands:
      - [0.01, 0.25]
    model_probability_minimum: 0.40
    dratings_probability_minimum: 0.35
```

---

## Key Concepts

**Edge:** `model_probability - sportsbook_implied_probability`. Positive edge means the model thinks the fighter is underpriced by the book.

**Implied probability from moneyline:**
- Positive line (underdog): `100 / (line + 100)`
- Negative line (favorite): `|line| / (|line| + 100)`
- Both normalized to remove the vig so they sum to 1.

**EV (Expected Value):** `(model_prob × odds_payout) - (1 - model_prob)`. Positive EV means the bet is theoretically profitable.

**Kelly criterion:** `((model_prob × (odds + 1)) - 1) / odds × 0.25`. Fractional Kelly (25%) used to reduce variance. Capped at 5% of bankroll per bet.

**Recency weighting:** Exponential decay with 365-day half-life. Fights from 2020–2021 count less than recent fights during model training.

**Brier score:** Lower is better. Measures how well-calibrated the predicted probabilities are.

**Data leakage prevention:** All historical stats (career record, SLpM, strike accuracy, TD accuracy) are computed from fight-by-fight history strictly before each fight date.

---

## Data Sources

- **Fight results & odds:** 434 CSV files (one per event, 2020–2026), 2,876 fights total
- **Fighter attributes & fight history:** [ufcstats.com](http://ufcstats.com), 1,168 fighters
- **Win predictions:** [dratings.com](https://www.dratings.com/predictor/ufc-mma-predictions/)
- **Live odds:** [oddstrader.com](https://www.oddstrader.com/ufc/)

---

## Model Performance

| Model | Brier Score | Log Loss |
|---|---|---|
| Baseline (implied prob only) | 0.1979 | 0.5796 |
| Logistic Regression (recency weighted) | 0.1828 | 0.5488 |
| XGBoost (recency weighted) | 0.1796 | 0.5411 |

**Test set:** 648 fights, January 2025 – April 2026
**Beats sportsbook baseline by 9.2% on Brier score**

**Calibration (XGBoost on test set):**

| Model Probability | Actual Win Rate | Count |
|---|---|---|
| 0–40% | 23% | 270 |
| 40–50% | 51% | 51 |
| 50–60% | 59% | 46 |
| 60–70% | 64% | 56 |
| 70–80% | 76% | 58 |
| 80–90% | 81% | 89 |
| 90–100% | 88% | 78 |

**Top features:** `diff_h_career_wr`, `implied_prob_f1`, `diff_win_rate_all`, `diff_h_str_acc`

---

## Backtesting Results (Phase 6)

**Flat betting at 3% edge threshold (648 test fights, 2025–2026):**

| Segment | Bets | Win% | ROI/bet | Total P/L |
|---|---|---|---|---|
| All fighters | 286 | 74.8% | $31.02 | $8,872 |
| Favorites (≥50%) | 213 | 81.2% | $17.76 | $3,782 |
| Underdogs (<50%) | 73 | 56.2% | $69.72 | $5,090 |
| Big underdogs (<35%) | 40 | 47.5% | $76.24 | $3,050 |

**Note:** ROI results show selection bias toward favorites. Model is well-calibrated but edge detection requires longer validation period (500+ bets minimum). Underdogs show highest ROI per bet and are the most promising segment.

**Kelly sizing (25% Kelly, max 5%/bet, $1,000 start):**
- 3% threshold: $5,270 final bankroll, 427% ROI over 15 months

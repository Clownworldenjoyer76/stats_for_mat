import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/stats")
OUT_DIR = Path("data/processed/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Files to process
files = {
    "defensive": "defensive.csv",
    "kicking": "kicking.csv",
    "kickoffs_punts": "kickoffs_punts.csv",
    "passing": "passing.csv",
    "receiving": "receiving.csv",
    "returning": "returning.csv",
    "rushing": "rushing.csv",
    "scoring": "scoring.csv"
}

# Columns that are percentages or rates and should NOT be divided by GP
PERCENT_COLUMNS = [
    "FG_PCT", "XP_PCT", "KO_TB_PCT", "IN20_PCT", "IN20 %", "PUNT_TB_PCT", 
    "TB_PCT", "TB %", "XP %", "FG %", "COMP_PCT", "PCT", "P%", "KO %", "NET AVG"
]

for name, filename in files.items():
    path = RAW_DIR / filename
    if not path.exists():
        print(f"⚠️ Missing: {path}")
        continue

    # --- Load file
    df = pd.read_csv(path)

    # --- Normalize column casing
    df.columns = [col.strip().upper().replace(" ", "_") for col in df.columns]

    # --- Identify per-game eligible numeric columns
    if "GP" not in df.columns:
        print(f"⚠️ No GP column in {filename}, skipping.")
        continue

    # Convert all numeric columns safely
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # --- Create per-game copy
    per_game = df.copy()

    # Divide totals by GP, skipping percent columns
    for col in numeric_cols:
        if col != "GP" and not any(pct in col for pct in PERCENT_COLUMNS):
            per_game[col] = per_game[col] / per_game["GP"]

    # --- Handle missing values (replace NaNs with 0)
    per_game = per_game.fillna(0)

    # --- Round all floats to 2 decimals for readability
    per_game = per_game.round(2)

    # --- Save with consistent naming
    out_file = OUT_DIR / f"{name}_per_game.csv"
    per_game.to_csv(out_file, index=False)
    print(f"✅ Saved {out_file}")

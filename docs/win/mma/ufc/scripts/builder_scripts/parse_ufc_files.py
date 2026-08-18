import pandas as pd
import glob
import os

# Set this to the folder where all your UFC CSV files are
DATA_FOLDER = "UFC_Master"  # change if your folder is named differently

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
print(f"Found {len(csv_files)} files")

frames = []
for filepath in csv_files:
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")  # utf-8-sig handles the BOM character in your files
        frames.append(df)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

master = pd.concat(frames, ignore_index=True)

# Standardize fighter names
for col in ["fighter_1", "fighter_2"]:
    master[col] = master[col].str.strip().str.lower().str.title()

# Standardize match_date to actual date format
master["match_date"] = pd.to_datetime(master["match_date"], format="%Y_%m_%d")

# Standardize result columns
for col in ["result_fighter_1", "result_fighter_2"]:
    master[col] = master[col].str.strip().str.capitalize()

# Sort by date
master = master.sort_values("match_date").reset_index(drop=True)

# Basic validation
print(f"Total fights: {len(master)}")
print(f"Date range: {master['match_date'].min()} to {master['match_date'].max()}")
print(f"Missing results: {master['result_fighter_1'].isna().sum()}")
print(f"Duplicate rows: {master.duplicated().sum()}")

# Save
master.to_parquet("ufc_master.parquet", index=False)
print("Saved to ufc_master.parquet")

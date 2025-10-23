import pandas as pd

# Load alias map
aliases = pd.read_csv("data/team_aliases.csv")
alias_map = dict(zip(aliases["ABBR"], aliases["TEAM"]))

# Files to normalize
files = [
    "data/raw/stats/defensive.csv",
    "data/raw/stats/kicking.csv",
    "data/raw/stats/kickoffs_punts.csv",
    "data/raw/stats/passing.csv",
    "data/raw/stats/receiving.csv",
    "data/raw/stats/returning.csv",
    "data/raw/stats/rushing.csv",
    "data/raw/stats/scoring.csv",
    "data/raw/week_matchups_odds.csv",
]

def normalize_team_column(df: pd.DataFrame):
    """Normalize any TEAM column."""
    for col in df.columns:
        if col.strip().lower() == "team":
            df[col] = df[col].map(alias_map).fillna(df[col])
    return df

def normalize_matchup_columns(df: pd.DataFrame):
    """Normalize TEAM_A and TEAM_B for the matchups file."""
    if "TEAM_A" in df.columns:
        df["TEAM_A"] = df["TEAM_A"].map(alias_map).fillna(df["TEAM_A"])
    if "TEAM_B" in df.columns:
        df["TEAM_B"] = df["TEAM_B"].map(alias_map).fillna(df["TEAM_B"])
    return df

for file_path in files:
    try:
        df = pd.read_csv(file_path)

        if "week_matchups_odds" in file_path:
            df = normalize_matchup_columns(df)
        else:
            df = normalize_team_column(df)

        df.to_csv(file_path, index=False)
        print(f"✅ Normalized: {file_path}")
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")

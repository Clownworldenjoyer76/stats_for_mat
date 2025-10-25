import pandas as pd
from pathlib import Path

# Input/output folders
RAW_DIR = Path("data/raw/stats")
OUT_DIR = Path("data/processed/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def average_per_game(file_path: Path):
    """Divide all numeric columns by GP and return per-game DataFrame."""
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]   # clean header whitespace

    # Identify GP column (case-insensitive)
    gp_col = next((c for c in df.columns if c.lower() == "gp"), None)
    if gp_col is None:
        print(f"⚠️  Skipping {file_path.name}: no GP column found")
        return None

    # Work on a copy to avoid modifying original
    df_per_game = df.copy()

    # Convert all possible numeric columns
    for col in df.columns:
        if col in [gp_col, "PLAYER", "TEAM", "GS"]:
            continue
        df_per_game[col] = pd.to_numeric(df[col], errors="coerce") / df[gp_col]

    return df_per_game

def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in data/raw/stats/")
        return

    for file_path in csv_files:
        df_pg = average_per_game(file_path)
        if df_pg is None:
            continue

        out_file = OUT_DIR / f"{file_path.stem}_per_game.csv"
        df_pg.to_csv(out_file, index=False)
        print(f"✅ Wrote {out_file}")

if __name__ == "__main__":
    main()

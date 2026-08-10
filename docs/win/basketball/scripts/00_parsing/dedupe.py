#!/usr/bin/env python3
# docs/win/basketball/scripts/00_parsing/dedupe.py

import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# =========================
# LOGGER UTILITY
# =========================

def audit(log_path, stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. EXHAUSTIVE LOG (TXT)
    with open(log_path, "a") as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg: f.write(f"  MSG: {msg}\n")
        if df is not None and isinstance(df, pd.DataFrame):
            f.write(f"  STATS: {len(df)} rows | {len(df.columns)} cols\n")
            f.write(f"  NULLS: {df.isnull().sum().sum()} total\n")
            f.write(f"  SAMPLE:\n{df.head(3).to_string(index=False)}\n")
        f.write("-" * 40 + "\n")

    # 2. CONDENSED SUMMARY (TXT)
    if df is not None and isinstance(df, pd.DataFrame):
        summary_path = log_path.parent / "condensed_summary.txt"
        
        play_cols = [c for c in ['home_play', 'away_play', 'over_play', 'under_play'] if c in df.columns]
        
        if play_cols:
            signals = df[df[play_cols].any(axis=1)].copy()
            
            if not signals.empty:
                with open(summary_path, "a") as f:
                    f.write(f"\n--- BETTING SIGNALS: {ts} ---\n")
                    base_cols = ['game_date', 'home_team', 'away_team']
                    edge_cols = [c for c in df.columns if 'edge_pct' in c]
                    
                    final_cols = [c for c in base_cols + edge_cols if c in signals.columns]
                    f.write(signals[final_cols].to_string(index=False))
                    f.write("\n" + "="*30 + "\n")

# =========================
# PATHS
# =========================

BASE_DIR = Path("docs/win/basketball/00_intake")
PRED_DIR = BASE_DIR / "predictions"
SPORTSBOOK_DIR = BASE_DIR / "sportsbook"

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "dedupe.txt"

# overwrite log each run
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")

# =========================
# DEDUPE FUNCTION
# =========================

def dedupe_file(csv_file: Path):
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            log(f"SKIP: {csv_file} has no header")
            return

        rows = list(reader)

    # required fields for basketball
    required_fields = ["game_date", "market", "home_team", "away_team"]
    if not all(k in fieldnames for k in required_fields):
        log(f"SKIP: {csv_file} missing required columns")
        return

    seen = set()
    deduped = []
    duplicates = 0

    for r in rows:
        key = (
            r.get("game_date", ""),
            r.get("market", ""),
            r.get("home_team", ""),
            r.get("away_team", ""),
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        deduped.append(r)

    # atomic rewrite
    temp_file = csv_file.with_suffix(".tmp")

    with open(temp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in deduped:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    temp_file.replace(csv_file)

    log(f"{csv_file.name}: removed {duplicates} duplicates, final_rows={len(deduped)}")
    
    # Audit Call
    df_deduped = pd.DataFrame(deduped)
    audit(LOG_FILE, "DEDUPE_STAGE", "SUCCESS", msg=f"Deduped {csv_file.name}", df=df_deduped)

# =========================
# PROCESS ALL FILES
# =========================

files_processed = 0

for directory in [PRED_DIR, SPORTSBOOK_DIR]:
    if not directory.exists():
        continue

    for csv_file in directory.glob("*.csv"):
        files_processed += 1
        dedupe_file(csv_file)

log(f"SUMMARY: files_processed={files_processed}")
print("Basketball dedupe complete.")

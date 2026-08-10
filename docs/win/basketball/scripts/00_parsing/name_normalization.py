#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/name_normalization.py

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
    
    with open(log_path, "a") as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg: f.write(f"  MSG: {msg}\n")
        if df is not None and isinstance(df, pd.DataFrame):
            f.write(f"  STATS: {len(df)} rows | {len(df.columns)} cols\n")
            f.write(f"  NULLS: {df.isnull().sum().sum()} total\n")
            f.write(f"  SAMPLE:\n{df.head(3).to_string(index=False)}\n")
        f.write("-" * 40 + "\n")

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

SPORTSBOOK_DIR = Path("docs/win/basketball/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/basketball/00_intake/predictions")

NBA_MAP_FILE = Path("docs/win/basketball/maps/team_map_nba.csv")
NCAAB_MAP_FILE = Path("docs/win/basketball/maps/team_map_ncaab.csv")

NO_MAP_DIR = Path("docs/win/basketball/maps/no_map")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_basketball.csv"

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "name_normalization_log.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")

# =========================
# LOAD TEAM MAPS
# =========================

team_map = {}

def load_map(map_file: Path):
    if not map_file.exists():
        log(f"WARNING: {map_file.name} not found")
        return

    with open(map_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market = row.get("league", "").strip().lower()
            alias = row.get("alias", "").strip().lower()
            canonical = row.get("canonical_team", "").strip()

            if market and alias and canonical:
                team_map[(market, alias)] = canonical

load_map(NBA_MAP_FILE)
load_map(NCAAB_MAP_FILE)

# =========================
# TARGET FILES ONLY
# =========================

target_files = []

for f in SPORTSBOOK_DIR.glob("basketball_NBA_*.csv"):
    target_files.append(f)

for f in SPORTSBOOK_DIR.glob("basketball_NCAAB_*.csv"):
    target_files.append(f)

for f in PREDICTIONS_DIR.glob("basketball_NBA_*.csv"):
    target_files.append(f)

for f in PREDICTIONS_DIR.glob("basketball_NCAAB_*.csv"):
    target_files.append(f)

# =========================
# PROCESS FILES
# =========================

unmapped = set()
files_processed = 0
rows_processed = 0
rows_updated = 0

for csv_file in target_files:
    files_processed += 1
    updated_rows = []
    modified = False

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            rows_processed += 1
            market = row.get("market", "").strip().lower()

            for side in ["home_team", "away_team"]:
                team = row.get(side, "").strip()
                if not team:
                    continue

                key = (market, team.lower())

                if key in team_map:
                    canonical = team_map[key]
                    if row.get(side) != canonical:
                        row[side] = canonical
                        modified = True
                        rows_updated += 1
                else:
                    unmapped.add((market, team))

            updated_rows.append(row)

    if modified and fieldnames:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    if updated_rows:
        df_audit = pd.DataFrame(updated_rows)
        audit(LOG_FILE, "NORMALIZATION", "SUCCESS", msg=f"Processed {csv_file.name}", df=df_audit)

# =========================
# WRITE UNMAPPED
# =========================

existing = set()

if NO_MAP_FILE.exists():
    with open(NO_MAP_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "market" in reader.fieldnames and "team" in reader.fieldnames:
            for row in reader:
                m = (row.get("market") or "").strip().lower()
                t = (row.get("team") or "").strip()
                if m and t:
                    existing.add((m, t))

new_only = unmapped - existing
combined = existing | unmapped

with open(NO_MAP_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["market", "team"])
    for market, team in sorted(combined):
        writer.writerow([market, team])

# =========================
# LOG SUMMARY
# =========================

log(
    f"SUMMARY: files_processed={files_processed}, "
    f"rows_processed={rows_processed}, "
    f"rows_updated={rows_updated}, "
    f"unmapped_found={len(unmapped)}, "
    f"unmapped_new_added={len(new_only)}"
)

print("Basketball name normalization complete.")

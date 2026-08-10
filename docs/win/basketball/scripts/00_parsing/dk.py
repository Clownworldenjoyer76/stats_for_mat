#!/usr/bin/env python3
# docs/win/basketball/scripts/00_parsing/dk.py

import sys
import re
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ----------------------------
# LOGGER UTILITY
# ----------------------------

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

# ----------------------------
# Logging
# ----------------------------
ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "dk_log.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


# ----------------------------
# Args
# ----------------------------
if len(sys.argv) < 4:
    log("ERROR: Expected args: league market raw_text_file")
    raise SystemExit(1)

league_input = sys.argv[1].strip()
market_input = sys.argv[2].strip()
raw_path = sys.argv[3]

try:
    raw_text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
except Exception as e:
    log(f"ERROR: Failed reading raw_text file '{raw_path}': {e}")
    raise

league_out = "Basketball"

market_map = {
    "NBA": "NBA",
    "NCAA Men": "NCAAB",
    "NCAAB": "NCAAB",
}
market_out = market_map.get(market_input)
if not market_out:
    log(f"ERROR: Invalid basketball market input: '{market_input}'")
    raise ValueError("Invalid basketball market")

# ----------------------------
# Helpers
# ----------------------------
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

RE_HEADER_DATE = re.compile(
    r"\b(?:MON|TUE|WED|THU|FRI|SAT|SUN)?\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE
)
RE_TODAY = re.compile(r"\bTODAY\b", re.IGNORECASE)
RE_TOMORROW = re.compile(r"\bTOMORROW\b", re.IGNORECASE)
RE_TIME = re.compile(r"\b(\d{1,2}:\d{2})\s*(AM|PM)\b", re.IGNORECASE)
RE_RANK = re.compile(r"^\d{1,3}$")
RE_AMERICAN = re.compile(r"^[+\-]\d+$")
RE_SPREAD = re.compile(r"^[+\-]\d+(?:\.\d+)?$")
RE_TOTAL = re.compile(r"^\d+(?:\.\d+)?$")

def norm_minus(s: str) -> str:
    return s.replace("−", "-").strip()

def clean_team(s: str) -> str:
    return s.replace("-logo", "").strip()

def is_team_candidate(s: str) -> bool:
    if not s:
        return False
    if "-logo" in s:
        return False
    if s.lower() in {"today", "tomorrow", "spread", "total", "moneyline", "more bets", "at", "o", "u"}:
        return False
    if RE_RANK.match(s.strip()):
        return False
    if RE_AMERICAN.match(norm_minus(s)):
        return False
    if RE_SPREAD.match(norm_minus(s)):
        return False
    if RE_TOTAL.match(norm_minus(s)):
        return False
    return True

def extract_game_date(text: str) -> str:
    today = datetime.today()

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        if RE_TODAY.search(line_clean):
            return today.strftime("%Y_%m_%d")
        if RE_TOMORROW.search(line_clean):
            return (today + timedelta(days=1)).strftime("%Y_%m_%d")

        m = RE_HEADER_DATE.search(line_clean.upper())
        if m:
            mon = MONTH_MAP[m.group(1)[:3].upper()]
            day = int(m.group(2))
            dt = datetime(today.year, mon, day)
            return dt.strftime("%Y_%m_%d")

    return today.strftime("%Y_%m_%d")

def extract_game_time(lines):
    for l in lines:
        m = RE_TIME.search(l)
        if m:
            return f"{m.group(1)} {m.group(2).upper()}"
    return ""

# ----------------------------
# Parse
# ----------------------------
FIELDNAMES = [
    "league","market","game_date","game_time",
    "home_team","away_team",
    "away_spread","home_spread","total",
    "away_dk_spread_american","home_dk_spread_american",
    "dk_total_over_american","dk_total_under_american",
    "away_dk_moneyline_american","home_dk_moneyline_american",
]

game_date = extract_game_date(raw_text)

blocks = raw_text.split("More Bets")
rows = []
errors = 0

for block in blocks:
    raw_lines = [l.strip() for l in block.splitlines() if l.strip()]
    if not raw_lines:
        continue

    if "at" not in raw_lines:
        continue

    try:
        at_idx = raw_lines.index("at")

        away_team = ""
        for i in range(at_idx - 1, -1, -1):
            if is_team_candidate(raw_lines[i]):
                away_team = clean_team(raw_lines[i])
                break

        home_team = ""
        for i in range(at_idx + 1, len(raw_lines)):
            if is_team_candidate(raw_lines[i]):
                home_team = clean_team(raw_lines[i])
                break

        if not away_team or not home_team:
            errors += 1
            continue

        o_idx = None
        u_idx = None
        for i, s in enumerate(raw_lines):
            if s == "O" and o_idx is None:
                o_idx = i
            if s == "U" and u_idx is None:
                u_idx = i

        if o_idx is None or u_idx is None:
            errors += 1
            continue

        L = [norm_minus(x) for x in raw_lines]

        away_spread = L[o_idx - 2] if o_idx >= 2 else ""
        away_dk_spread_american = L[o_idx - 1] if o_idx >= 1 else ""

        total = L[o_idx + 1] if (o_idx + 1) < len(L) else ""
        dk_total_over_american = L[o_idx + 2] if (o_idx + 2) < len(L) else ""
        away_dk_moneyline_american = L[o_idx + 3] if (o_idx + 3) < len(L) else ""

        home_spread = L[u_idx - 2] if u_idx >= 2 else ""
        home_dk_spread_american = L[u_idx - 1] if u_idx >= 1 else ""
        dk_total_under_american = L[u_idx + 2] if (u_idx + 2) < len(L) else ""
        home_dk_moneyline_american = L[u_idx + 3] if (u_idx + 3) < len(L) else ""

        if not (RE_SPREAD.match(away_spread) and RE_SPREAD.match(home_spread) and RE_TOTAL.match(total)):
            errors += 1
            continue

        if not (RE_AMERICAN.match(away_dk_spread_american) and RE_AMERICAN.match(home_dk_spread_american)
                and RE_AMERICAN.match(dk_total_over_american) and RE_AMERICAN.match(dk_total_under_american)
                and RE_AMERICAN.match(away_dk_moneyline_american) and RE_AMERICAN.match(home_dk_moneyline_american)):
            errors += 1
            continue

        game_time = extract_game_time(raw_lines)

        rows.append({
            "league": league_out,
            "market": market_out,
            "game_date": game_date,
            "game_time": game_time,
            "home_team": home_team,
            "away_team": away_team,
            "away_spread": away_spread,
            "home_spread": home_spread,
            "total": total,
            "away_dk_spread_american": away_dk_spread_american,
            "home_dk_spread_american": home_dk_spread_american,
            "dk_total_over_american": dk_total_over_american,
            "dk_total_under_american": dk_total_under_american,
            "away_dk_moneyline_american": away_dk_moneyline_american,
            "home_dk_moneyline_american": home_dk_moneyline_american,
        })

    except Exception:
        errors += 1

if not rows:
    log("SUMMARY: wrote 0 rows")
    raise SystemExit(1)

# ----------------------------
# Write (FULL REBUILD FOR DAY)
# ----------------------------
out_dir = Path("docs/win/basketball/00_intake/sportsbook")
out_dir.mkdir(parents=True, exist_ok=True)

outfile = out_dir / f"{league_input}_{market_out}_{game_date}.csv"

with open(outfile, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in FIELDNAMES})

log(f"SUMMARY: wrote {len(rows)} rows (full rebuild), {errors} errors")
print(f"Wrote {outfile} ({len(rows)} rows)")

# Final Audit Call
df_rows = pd.DataFrame(rows)
audit(LOG_FILE, "PARSING_STAGE", "SUCCESS", msg=f"Parsed DraftKings raw text for {market_out}", df=df_rows)

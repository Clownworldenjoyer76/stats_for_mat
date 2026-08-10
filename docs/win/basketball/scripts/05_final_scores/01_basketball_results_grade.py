#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/01_basketball_results_grade.py
#
# Grades selected bets against final scores.
#
# Inputs:
#   docs/win/basketball/04_select/{league}/daily_picks/*.csv
#       league in {nba, ncaam, wnba}
#   docs/win/basketball/05_final_scores/results/{league}/*.csv
#       (final scores, joined by game_id)
#
# Outputs:
#   docs/win/basketball/05_final_scores/results/{league}/graded/{LEAGUE}_final.csv
#   docs/win/basketball/errors/05_final_scores/{league}_game_id_no_match.csv
#       (one per league; only created if there are unmatched picks)
#   docs/win/basketball/errors/05_final_scores/01_basketball_results_grade.txt
#
# Grading:
#   moneyline  -> winning side based on home/away score, bet_side ∈ {home, away}
#   spread     -> uses bet_line (the side's signed spread); cover iff
#                   home: (home_score + bet_line) > away_score
#                   away: (away_score + bet_line) > home_score
#   total      -> uses bet_line (the book total); over wins iff total > line, etc.
#
# Per-row P&L columns added:
#   profit_unit   = (decimal - 1) if Win else -1 if Loss else 0     # flat $1 stake
#   profit_kelly  = bet_stake_pct * (decimal - 1) if Win
#                 = -bet_stake_pct                if Loss
#                 = 0                              if Push
#                 (None if bet_stake_pct is missing)

import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

BASE             = Path("docs/win/basketball")
SELECT_BASE      = BASE / "04_select"
FINAL_SCORES_DIR = BASE / "05_final_scores/results"
ERROR_DIR        = BASE / "errors/05_final_scores"
LOG_FILE         = ERROR_DIR / "01_basketball_results_grade.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _init_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== 01_basketball_results_grade RUN {_now()} ===\n")


def _log(msg: str, level: str = "INFO"):
    line = f"{_now()} | {level:<5} | {msg.rstrip()}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================
# HELPERS
# =========================

def american_to_decimal(odds):
    try:
        a = float(odds)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return 1 + (a / 100.0) if a > 0 else 1 + (100.0 / abs(a))


def f(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def determine_outcome(row) -> str:
    market = str(row.get("market_type", "")).lower()
    side   = str(row.get("bet_side", "")).lower()

    home = f(row.get("home_score"))
    away = f(row.get("away_score"))
    if home is None or away is None:
        return "Unknown"

    if market == "moneyline":
        if home == away:
            return "Push"
        home_won = home > away
        if (side == "home" and home_won) or (side == "away" and not home_won):
            return "Win"
        return "Loss"

    if market == "spread":
        line = f(row.get("bet_line"))
        if line is None:
            return "Unknown"
        if side == "home":
            diff = (home + line) - away
        elif side == "away":
            diff = (away + line) - home
        else:
            return "Unknown"
        if abs(diff) < 1e-9:
            return "Push"
        return "Win" if diff > 0 else "Loss"

    if market == "total":
        line = f(row.get("bet_line"))
        if line is None:
            return "Unknown"
        total = home + away
        if abs(total - line) < 1e-9:
            return "Push"
        if (total > line and side == "over") or (total < line and side == "under"):
            return "Win"
        return "Loss"

    return "Unknown"


def compute_profits(row) -> tuple:
    """Returns (profit_unit, profit_kelly). profit_kelly is None if bet_stake_pct missing."""
    result = str(row.get("bet_result", "")).strip()
    odds   = row.get("bet_odds_american")
    decimal = american_to_decimal(odds)

    if result == "Push":
        return 0.0, 0.0
    if result not in ("Win", "Loss"):
        return None, None
    if decimal is None or decimal <= 1:
        return None, None

    stake = f(row.get("bet_stake_pct"))

    if result == "Win":
        unit  = decimal - 1.0
        kelly = (stake * (decimal - 1.0)) if stake is not None else None
    else:  # Loss
        unit  = -1.0
        kelly = -stake if stake is not None else None

    return unit, kelly


def load_picks_for_league(league: str) -> pd.DataFrame:
    folder = SELECT_BASE / league / "daily_picks"
    if not folder.exists():
        _log(f"[{league}] picks folder missing: {folder}", "WARN")
        return pd.DataFrame()
    files = sorted(folder.glob("*.csv"))
    if not files:
        _log(f"[{league}] no pick files in {folder}", "WARN")
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            _log(f"[{league}] FAILED reading {fp.name}: {e}", "ERROR")
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    _log(f"[{league}] loaded {len(out)} pick rows from {len(files)} files")
    return out


def load_scores_for_league(league: str) -> pd.DataFrame:
    folder = FINAL_SCORES_DIR / league
    if not folder.exists():
        _log(f"[{league}] scores folder missing: {folder}", "WARN")
        return pd.DataFrame()
    files = sorted(folder.glob("*.csv"))
    if not files:
        _log(f"[{league}] no score files in {folder}", "WARN")
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            _log(f"[{league}] FAILED reading {fp.name}: {e}", "ERROR")
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # Keep only the columns we actually need from scores; preserve game_id as join key.
    keep = [c for c in ["game_id", "home_score", "away_score"] if c in out.columns]
    if "game_id" not in keep:
        _log(f"[{league}] scores have no game_id column", "ERROR")
        return pd.DataFrame()
    out = out[keep].copy()
    # If the same game_id appears multiple times in the score files, keep the last
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    _log(f"[{league}] loaded {len(out)} unique scored games from {len(files)} files")
    return out


def grade_league(league: str, summary: dict) -> None:
    upper = league.upper()

    out_dir = FINAL_SCORES_DIR / league / "graded"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.csv"):
        stale.unlink(missing_ok=True)

    picks = load_picks_for_league(league)
    if picks.empty:
        _log(f"[{league}] no picks; skipping")
        return

    if "game_id" not in picks.columns:
        _log(f"[{league}] picks missing game_id; cannot grade", "ERROR")
        summary["errors"] += 1
        return

    scores = load_scores_for_league(league)
    if scores.empty:
        _log(f"[{league}] no scores; skipping")
        return

    # Inner merge on game_id
    merged = picks.merge(scores, on="game_id", how="left", suffixes=("", "_score"))

    # Coalesce duplicate score columns if picks already had them (use scores version)
    for col in ("home_score", "away_score"):
        side = f"{col}_score"
        if side in merged.columns:
            merged[col] = merged[side].combine_first(merged.get(col))
            merged = merged.drop(columns=[side])

    # Identify unmatched picks (no final score yet)
    matched_mask = merged["home_score"].notna() & merged["away_score"].notna()
    unmatched = merged.loc[~matched_mask].copy()
    matched   = merged.loc[ matched_mask].copy()

    if not unmatched.empty:
        unmatched_path = ERROR_DIR / f"{league}_game_id_no_match.csv"
        no_match_cols = [c for c in [
            "game_id", "game_date", "home_team", "away_team",
            "league", "market_type", "bet_side"
        ] if c in unmatched.columns]
        unmatched_out = unmatched[no_match_cols].copy()
        unmatched_out["source"] = "pick_no_score"
        unmatched_out.to_csv(unmatched_path, index=False)
        _log(f"[{league}] {len(unmatched_out)} picks had no matching final score -> {unmatched_path}")
        summary[f"{league}_unmatched"] = len(unmatched_out)

    if matched.empty:
        _log(f"[{league}] no picks matched a final score; nothing to grade")
        return

    # Grade
    matched["bet_result"] = matched.apply(determine_outcome, axis=1)

    # P&L per row (flat unit + kelly-scaled)
    profits = matched.apply(compute_profits, axis=1, result_type="expand")
    profits.columns = ["profit_unit", "profit_kelly"]
    matched = pd.concat([matched, profits], axis=1)

    # Deduplicate (defensive) — same bet shouldn't appear twice
    key_cols = [c for c in ["game_id", "market_type", "bet_side"] if c in matched.columns]
    if key_cols:
        before = len(matched)
        matched = matched.drop_duplicates(subset=key_cols, keep="last")
        dropped = before - len(matched)
        if dropped:
            _log(f"[{league}] deduped {dropped} duplicate graded rows")

    out_path = out_dir / f"{upper}_final.csv"
    matched.to_csv(out_path, index=False)

    # Summary stats per league
    n = len(matched)
    wins   = int((matched["bet_result"] == "Win").sum())
    losses = int((matched["bet_result"] == "Loss").sum())
    pushes = int((matched["bet_result"] == "Push").sum())
    unknown = int((matched["bet_result"] == "Unknown").sum())
    pnl_unit  = float(matched["profit_unit"].sum(skipna=True)) if "profit_unit" in matched.columns else 0.0
    pnl_kelly = float(matched["profit_kelly"].sum(skipna=True)) if "profit_kelly" in matched.columns else 0.0

    summary[f"{league}_graded"]   = n
    summary[f"{league}_wins"]     = wins
    summary[f"{league}_losses"]   = losses
    summary[f"{league}_pushes"]   = pushes
    summary[f"{league}_unknown"]  = unknown
    summary[f"{league}_pnl_unit"] = round(pnl_unit, 4)
    summary[f"{league}_pnl_kelly"]= round(pnl_kelly, 4)

    _log(f"[{league}] graded {n} bets -> {out_path}")
    _log(f"[{league}] W/L/P/Unk = {wins}/{losses}/{pushes}/{unknown}  "
         f"PnL_unit={pnl_unit:+.2f}  PnL_kelly={pnl_kelly:+.4f}")


# =========================
# MAIN
# =========================

def main():
    _init_log()
    _log(f"SELECT_BASE      : {SELECT_BASE}")
    _log(f"FINAL_SCORES_DIR : {FINAL_SCORES_DIR}")
    _log(f"ERROR_DIR        : {ERROR_DIR}")

    summary = {"errors": 0}
    try:
        for league in LEAGUES:
            try:
                grade_league(league, summary)
            except Exception as e:
                _log(f"[{league}] FATAL during grading: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

        _log("--- SUMMARY ---")
        for k in sorted(summary.keys()):
            _log(f"  {k:<20} : {summary[k]}")
        status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
        _log(f"STATUS: {status}")

        print("Grading complete.")

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()

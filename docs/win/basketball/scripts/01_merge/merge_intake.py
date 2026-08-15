#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/merge_intake.py
"""Merge cleaned predictions and sportsbook data with game_id-first identity.

Full historical rebuild behavior is retained. Matching now prefers canonical game_id
and uses date/home/away only as a controlled unique fallback. Current in-season
coverage gaps are fatal so a partially merged live slate cannot pass green.
"""
from __future__ import annotations

import csv
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LEAGUES = ["nba", "ncaam", "wnba"]
INTAKE_DIR = Path("docs/win/basketball/00_intake")
PREDICTIONS_DIR = INTAKE_DIR / "predictions" / "predictions_cleaned"
SPORTSBOOK_DIR = INTAKE_DIR / "sportsbook" / "sportsbook_cleaned"
MERGE_DIR = Path("docs/win/basketball/01_merge")
ERROR_DIR = Path("docs/win/basketball/errors/01_merge")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "merge_intake.txt"
COVERAGE_FILE = ERROR_DIR / "merge_coverage.csv"
NY = ZoneInfo("America/New_York")

PROVENANCE_FIELDS = ["bias_applied", "margin_bias", "total_bias"]
MONEYLINE_FIELDS = [
    "sport", "league", "game_id", "game_date", "game_time", "home_team", "away_team",
    "home_prob", "away_prob", "away_projected_points", "home_projected_points", "total_projected_points",
    *PROVENANCE_FIELDS, "total", "home_dk_moneyline_american", "away_dk_moneyline_american",
    "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
]
SPREAD_FIELDS = [
    "sport", "league", "game_id", "game_date", "game_time", "home_team", "away_team",
    "home_prob", "away_prob", "away_projected_points", "home_projected_points", "total_projected_points",
    *PROVENANCE_FIELDS, "total", "home_spread", "away_spread", "home_dk_spread_american",
    "away_dk_spread_american", "home_dk_spread_decimal", "away_dk_spread_decimal",
]
TOTAL_FIELDS = [
    "sport", "league", "game_id", "game_date", "game_time", "home_team", "away_team",
    "home_prob", "away_prob", "away_projected_points", "home_projected_points", "total_projected_points",
    *PROVENANCE_FIELDS, "total", "dk_total_over_american", "dk_total_under_american",
    "dk_total_over_decimal", "dk_total_under_decimal",
]
COVERAGE_FIELDS = [
    "league", "game_date", "prediction_rows", "sportsbook_rows", "matched_rows", "missing_matches",
    "match_by_game_id", "match_by_composite", "identity_mismatches", "coverage_pct", "current_in_season", "status",
]


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def comp_key(r: dict) -> tuple[str, str, str]:
    return (clean(r.get("game_date")), clean(r.get("home_team")).casefold(), clean(r.get("away_team")).casefold())


def id_rank(game_id: str) -> int:
    gid = clean(game_id)
    if not gid:
        return 0
    return 2 if gid.isdigit() else 1


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def in_season(league: str, now: datetime) -> bool:
    if league in {"nba", "ncaam"}:
        return now.month >= 9 or now.month <= 6 or (now.month == 7 and now.day == 1)
    if league == "wnba":
        return 5 <= now.month <= 10
    return True


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wipe_outputs() -> None:
    for league in LEAGUES:
        for subdir in ["moneyline", "spread", "total"]:
            folder = MERGE_DIR / league / subdir
            folder.mkdir(parents=True, exist_ok=True)
            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)
    log("Wiped all output folders for full replay rebuild.")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonicalize_book_rows(rows: list[dict], league: str, date: str) -> tuple[dict[str, dict], dict[tuple, dict], int]:
    by_comp: dict[tuple, dict] = {}
    identity_mismatches = 0
    for row in rows:
        key = comp_key(row)
        if key not in by_comp:
            by_comp[key] = row
            continue
        old = by_comp[key]
        old_id, new_id = clean(old.get("game_id")), clean(row.get("game_id"))
        if old_id.isdigit() and new_id.isdigit() and old_id != new_id:
            raise ValueError(f"{league.upper()} {date}: conflicting numeric sportsbook IDs {old_id} vs {new_id} for {key}")
        if old_id != new_id and old_id and new_id:
            identity_mismatches += 1
            log(f"ID ALIAS | {league.upper()} {date} | {old_id} <-> {new_id} | {key[1]} vs {key[2]}")
        if id_rank(new_id) > id_rank(old_id):
            by_comp[key] = row

    by_id: dict[str, dict] = {}
    for row in by_comp.values():
        gid = clean(row.get("game_id"))
        if not gid:
            continue
        if gid in by_id and comp_key(by_id[gid]) != comp_key(row):
            raise ValueError(f"{league.upper()} {date}: game_id {gid} maps to multiple game identities")
        by_id[gid] = row
    return by_id, by_comp, identity_mismatches


def canonical_game_id(pred: dict, book: dict) -> str:
    p = clean(pred.get("game_id")); b = clean(book.get("game_id"))
    if p.isdigit() and b.isdigit() and p != b:
        raise ValueError(f"Conflicting numeric game_id prediction={p} sportsbook={b} for {comp_key(pred)}")
    return b if id_rank(b) > id_rank(p) else p


def build_base(p: dict, b: dict) -> dict:
    return {
        "sport": p.get("sport", ""), "league": p.get("league", ""),
        "game_id": canonical_game_id(p, b), "game_date": p.get("game_date", ""),
        "game_time": p.get("game_time", "") or b.get("game_time", ""),
        "home_team": p.get("home_team", ""), "away_team": p.get("away_team", ""),
        "home_prob": p.get("home_prob", ""), "away_prob": p.get("away_prob", ""),
        "away_projected_points": p.get("away_projected_points", ""),
        "home_projected_points": p.get("home_projected_points", ""),
        "total_projected_points": p.get("total_projected_points", ""),
        "bias_applied": p.get("bias_applied", ""), "margin_bias": p.get("margin_bias", ""),
        "total_bias": p.get("total_bias", ""), "total": b.get("total", ""),
    }


def main() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== merge_intake RUN {datetime.now().isoformat()} ===\n")

    files_written = total_merged = total_missing = slates_skipped = 0
    errors = 0
    coverage_rows: list[dict] = []
    now = datetime.now(NY)
    current_date = now.strftime("%Y_%m_%d")
    full_rebuild = truthy(os.getenv("BASKETBALL_FULL_REBUILD"))

    try:
        if full_rebuild:
            wipe_outputs()
        else:
            for league in LEAGUES:
                upper = league.upper()
                for market in ["moneyline", "spread", "total"]:
                    path = MERGE_DIR / league / market / f"{current_date}_{upper}_{market}.csv"
                    path.unlink(missing_ok=True)
            log(f"Incremental mode: rebuilding only {current_date}; historical merge outputs preserved.")
        for league in LEAGUES:
            league_upper = league.upper()
            pred_dir = PREDICTIONS_DIR / league
            book_dir = SPORTSBOOK_DIR / league
            if not pred_dir.exists():
                log(f"PREDICTIONS DIR NOT FOUND: {pred_dir}")
                continue
            if full_rebuild:
                pred_files = sorted(pred_dir.glob(f"*_{league_upper}_predictions.csv"))
            else:
                current_pred = pred_dir / f"{current_date}_{league_upper}_predictions.csv"
                pred_files = [current_pred] if current_pred.exists() else []
            if not pred_files:
                log(f"NO PREDICTION FILES: {pred_dir}")
                continue

            for pred_file in pred_files:
                date = pred_file.stem.replace(f"_{league_upper}_predictions", "")
                book_file = book_dir / f"{date}_{league_upper}_odds.csv"
                current_live = date == current_date and in_season(league, now)
                pred_rows = load_rows(pred_file)
                if not pred_rows:
                    log(f"EMPTY PREDICTIONS: {pred_file} — skipping")
                    slates_skipped += 1
                    continue
                if not book_file.exists():
                    log(f"NO SPORTSBOOK FILE: {book_file} — skipping")
                    slates_skipped += 1
                    coverage_rows.append({
                        "league": league_upper, "game_date": date, "prediction_rows": len(pred_rows),
                        "sportsbook_rows": 0, "matched_rows": 0, "missing_matches": len(pred_rows),
                        "match_by_game_id": 0, "match_by_composite": 0, "identity_mismatches": 0,
                        "coverage_pct": 0.0, "current_in_season": int(current_live),
                        "status": "ERROR" if current_live else "SKIPPED",
                    })
                    if current_live:
                        errors += 1
                    continue

                book_rows = load_rows(book_file)
                if not book_rows:
                    log(f"EMPTY SPORTSBOOK: {book_file} — skipping")
                    slates_skipped += 1
                    if current_live:
                        errors += 1
                    continue

                book_by_id, book_by_comp, identity_mismatches = canonicalize_book_rows(book_rows, league, date)
                ml_rows: list[dict] = []; spread_rows: list[dict] = []; total_rows: list[dict] = []
                missing = by_id_matches = by_comp_matches = 0

                for p in pred_rows:
                    p_gid = clean(p.get("game_id"))
                    b = book_by_id.get(p_gid) if p_gid else None
                    if b is not None:
                        by_id_matches += 1
                    else:
                        b = book_by_comp.get(comp_key(p))
                        if b is not None:
                            by_comp_matches += 1
                    if b is None:
                        missing += 1; total_missing += 1
                        log(f"MISSING MATCH | {league_upper} {date} | {p.get('home_team')} vs {p.get('away_team')} | game_id={p_gid}")
                        continue

                    if p_gid and clean(b.get("game_id")) and p_gid != clean(b.get("game_id")):
                        identity_mismatches += 1
                        log(f"IDENTITY FALLBACK | {league_upper} {date} | prediction_id={p_gid} sportsbook_id={clean(b.get('game_id'))}")

                    base = build_base(p, b)
                    ml_rows.append({**base,
                        "home_dk_moneyline_american": b.get("home_dk_moneyline_american", ""),
                        "away_dk_moneyline_american": b.get("away_dk_moneyline_american", ""),
                        "home_dk_moneyline_decimal": b.get("home_dk_moneyline_decimal", ""),
                        "away_dk_moneyline_decimal": b.get("away_dk_moneyline_decimal", ""),
                    })
                    spread_rows.append({**base,
                        "home_spread": b.get("home_spread", ""), "away_spread": b.get("away_spread", ""),
                        "home_dk_spread_american": b.get("home_dk_spread_american", ""),
                        "away_dk_spread_american": b.get("away_dk_spread_american", ""),
                        "home_dk_spread_decimal": b.get("home_dk_spread_decimal", ""),
                        "away_dk_spread_decimal": b.get("away_dk_spread_decimal", ""),
                    })
                    total_rows.append({**base,
                        "dk_total_over_american": b.get("dk_total_over_american", ""),
                        "dk_total_under_american": b.get("dk_total_under_american", ""),
                        "dk_total_over_decimal": b.get("dk_total_over_decimal", ""),
                        "dk_total_under_decimal": b.get("dk_total_under_decimal", ""),
                    })

                matched = len(ml_rows)
                pct = round((matched / len(pred_rows) * 100.0) if pred_rows else 100.0, 2)
                status = "OK" if missing == 0 else ("ERROR" if current_live else "PARTIAL")
                coverage_rows.append({
                    "league": league_upper, "game_date": date, "prediction_rows": len(pred_rows),
                    "sportsbook_rows": len(book_rows), "matched_rows": matched, "missing_matches": missing,
                    "match_by_game_id": by_id_matches, "match_by_composite": by_comp_matches,
                    "identity_mismatches": identity_mismatches, "coverage_pct": pct,
                    "current_in_season": int(current_live), "status": status,
                })
                log(f"COVERAGE | {league_upper} {date} | matched={matched}/{len(pred_rows)} ({pct:.2f}%) | missing={missing} | id={by_id_matches} fallback={by_comp_matches}")
                if current_live and missing:
                    errors += 1

                if not ml_rows:
                    log(f"NO MERGED ROWS: {league_upper} {date} — skipping")
                    slates_skipped += 1
                    continue

                ml_path = MERGE_DIR / league / "moneyline" / f"{date}_{league_upper}_moneyline.csv"
                spread_path = MERGE_DIR / league / "spread" / f"{date}_{league_upper}_spread.csv"
                total_path = MERGE_DIR / league / "total" / f"{date}_{league_upper}_total.csv"
                write_csv(ml_path, MONEYLINE_FIELDS, ml_rows)
                write_csv(spread_path, SPREAD_FIELDS, spread_rows)
                write_csv(total_path, TOTAL_FIELDS, total_rows)
                total_merged += matched; files_written += 3
                log(f"WROTE {ml_path.name} | {spread_path.name} | {total_path.name} ({matched} rows each)")

    except Exception as exc:
        errors += 1
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}")

    write_csv(COVERAGE_FILE, COVERAGE_FIELDS, coverage_rows)
    log("--- SUMMARY ---")
    log(f"Mode: {'full_rebuild' if full_rebuild else 'incremental_current_date'}")
    log(f"Files written: {files_written}")
    log(f"Total rows merged: {total_merged}")
    log(f"Total missing matches: {total_missing}")
    log(f"Slates skipped: {slates_skipped}")
    log(f"Errors: {errors}")
    log(f"STATUS: {'SUCCESS' if errors == 0 else 'FAILED'}")
    if errors:
        sys.exit(1)
    print("merge_intake complete.")


if __name__ == "__main__":
    main()

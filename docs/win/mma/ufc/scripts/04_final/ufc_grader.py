#!/usr/bin/env python3
"""
# docs/win/mma/ufc/scripts/04_final/ufc_grader.py

Grades bets from the detailed select files against the manual results files.

Inputs:
    docs/win/mma/ufc/03_select/detailed/{date}_ufc_select_detailed.csv
    docs/win/mma/ufc/manual_files/{date}_ufc.csv

Output:
    docs/win/mma/ufc/04_final/graded/{date}_ufc_graded.csv

Rules:
    - Iterate over detailed files. For each date, look for the matching manual
      results file. If the results file is missing, skip the date.
    - Exclude rows where bet == 'no_bet'.
    - Match each bet's fight pair to the results file by frozenset of
      normalized fighter names. If no match, skip the row with a printed
      warning.
    - Derive `match_result` from the bet side's outcome column. Anything that
      isn't a clean Win or Loss (draw, NC, blank, etc.) becomes Push.
    - Result columns in the output are aligned to the DETAILED file's
      fighter_1 / fighter_2 ordering (so result_fighter_1 always refers to
      detailed's fighter_1, regardless of how the results file ordered them).
    - Every run starts with a clean slate: existing *_ufc_graded.csv files in
      the output directory are deleted before processing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# --- Paths ---
DETAILED_DIR = Path("docs/win/mma/ufc/03_select/detailed")
RESULTS_DIR = Path("docs/win/mma/ufc/manual_files")
OUT_DIR = Path("docs/win/mma/ufc/04_final/graded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Clean slate ---
for stale in OUT_DIR.glob("*_ufc_graded.csv"):
    stale.unlink()
    print(f"DELETED stale {stale}")


# --- Helpers ---

def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def grade_result(s: str) -> str:
    """Map raw result cell to Win / Loss / Push. Anything not a clean win or
    loss becomes Push."""
    v = (s or "").strip().lower()
    if v == "win":
        return "Win"
    if v == "loss":
        return "Loss"
    return "Push"


# --- Output columns ---
OUTPUT_FIELDS = [
    "match_date",
    "fighter_1",
    "fighter_2",
    "moneyline_f1",
    "moneyline_f2",
    "implied_prob_f1",
    "implied_prob_f2",
    "model_prob_f1",
    "model_prob_f2",
    "dratings_prob_f1",
    "dratings_prob_f2",
    "edge_f1",
    "edge_f2",
    "ev_f1",
    "ev_f2",
    "kelly_f1",
    "kelly_f2",
    "bet",
    "result_fighter_1",
    "result_fighter_2",
    "match_result",
]


# --- Process each detailed file ---
detailed_files = sorted(DETAILED_DIR.glob("*_ufc_select_detailed.csv"))

if not detailed_files:
    print("No detailed files found.")
    raise SystemExit(1)


for det_file in detailed_files:
    date_str = det_file.stem.replace("_ufc_select_detailed", "")
    results_file = RESULTS_DIR / f"{date_str}_ufc.csv"

    if not results_file.exists():
        print(f"SKIP {date_str}: no results file at {results_file}")
        continue

    # Read results (utf-8-sig handles the BOM in the manual files)
    with results_file.open(encoding="utf-8-sig", newline="") as f:
        results_rows = list(csv.DictReader(f))

    # Index results by frozenset of normalized names.
    results_idx = {}
    for r in results_rows:
        f1_norm = normalize_name(r.get("fighter_1", ""))
        f2_norm = normalize_name(r.get("fighter_2", ""))
        if f1_norm and f2_norm:
            results_idx[frozenset((f1_norm, f2_norm))] = {
                "fighter_1_norm": f1_norm,
                "result_fighter_1": r.get("result_fighter_1", ""),
                "result_fighter_2": r.get("result_fighter_2", ""),
            }

    # Read detailed
    with det_file.open(encoding="utf-8", newline="") as f:
        det_rows = list(csv.DictReader(f))

    graded_rows = []

    for row in det_rows:
        bet = (row.get("bet") or "").strip()

        # Rule: exclude no_bet rows.
        if bet == "no_bet" or bet == "":
            continue

        det_f1_norm = normalize_name(row.get("fighter_1", ""))
        det_f2_norm = normalize_name(row.get("fighter_2", ""))

        if not det_f1_norm or not det_f2_norm:
            print(f"WARN {date_str}: blank fighter name in detailed row, skipping")
            continue

        key = frozenset((det_f1_norm, det_f2_norm))
        result = results_idx.get(key)

        if not result:
            print(
                f"WARN {date_str}: no result match for "
                f"{row.get('fighter_1')} vs {row.get('fighter_2')}, skipping"
            )
            continue

        # Align result columns to detailed's fighter_1/fighter_2 ordering.
        if result["fighter_1_norm"] == det_f1_norm:
            res_f1 = result["result_fighter_1"]
            res_f2 = result["result_fighter_2"]
        else:
            res_f1 = result["result_fighter_2"]
            res_f2 = result["result_fighter_1"]

        # Derive match_result from the bet side.
        if bet == "fighter_1":
            target = res_f1
        elif bet == "fighter_2":
            target = res_f2
        else:
            print(f"WARN {date_str}: unknown bet value '{bet}', skipping")
            continue

        match_result = grade_result(target)

        graded_rows.append({
            "match_date": row.get("match_date", ""),
            "fighter_1": row.get("fighter_1", ""),
            "fighter_2": row.get("fighter_2", ""),
            "moneyline_f1": row.get("moneyline_f1", ""),
            "moneyline_f2": row.get("moneyline_f2", ""),
            "implied_prob_f1": row.get("implied_prob_f1", ""),
            "implied_prob_f2": row.get("implied_prob_f2", ""),
            "model_prob_f1": row.get("model_prob_f1", ""),
            "model_prob_f2": row.get("model_prob_f2", ""),
            "dratings_prob_f1": row.get("dratings_prob_f1", ""),
            "dratings_prob_f2": row.get("dratings_prob_f2", ""),
            "edge_f1": row.get("edge_f1", ""),
            "edge_f2": row.get("edge_f2", ""),
            "ev_f1": row.get("ev_f1", ""),
            "ev_f2": row.get("ev_f2", ""),
            "kelly_f1": row.get("kelly_f1", ""),
            "kelly_f2": row.get("kelly_f2", ""),
            "bet": bet,
            "result_fighter_1": res_f1,
            "result_fighter_2": res_f2,
            "match_result": match_result,
        })

    if not graded_rows:
        print(f"No bets to grade for {date_str}")
        continue

    out_file = OUT_DIR / f"{date_str}_ufc_graded.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(graded_rows)

    print(f"WROTE {out_file} ({len(graded_rows)} graded bets)")

#!/usr/bin/env python3
"""
UFC reports script.

Repo path: docs/win/mma/ufc/scripts/04_final/ufc_reports.py

Reads already-graded UFC bet files from:

  docs/win/mma/ufc/04_final/graded/{date}_ufc_graded.csv

and writes the same summary + bucketed breakdown CSVs to 04_final/.

Inputs:
  docs/win/mma/ufc/04_final/graded/{date}_ufc_graded.csv

Outputs:
  docs/win/mma/ufc/04_final/ufc_summary_overall.csv
  docs/win/mma/ufc/04_final/reports/ufc_moneyline_by_ev.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_odds.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_implied_prob.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_model_prob.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_dratings_prob.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_date.csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from glob import glob
from math import floor

# ---------- Paths ----------
# script lives at docs/win/mma/ufc/scripts/04_final/ufc_reports.py
# go up 6 levels to reach repo root
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
    )
)

UFC_BASE = os.path.join(REPO_ROOT, "docs", "win", "mma", "ufc")
OUT_DIR = os.path.join(UFC_BASE, "04_final")
GRADED_DIR = os.path.join(OUT_DIR, "graded")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")

LEAGUE = "ufc"
MARKET = "moneyline"

# ---------- Helpers ----------

def american_to_profit(ml: int) -> float:
    """Profit on a 1-unit bet at American odds."""
    if ml >= 0:
        return ml / 100.0
    return 100.0 / -ml


def parse_ml(s: str) -> int:
    s = (s or "").strip().replace(" ", "")
    if not s:
        raise ValueError("empty moneyline")
    return int(s)


def fmt_ml(ml: int) -> str:
    return f"+{ml}" if ml >= 0 else f"{ml}"


def safe_float(s):
    if s is None:
        return None
    v = str(s).strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def grade_result(s: str) -> str:
    """Anything not Win/Loss = Push."""
    v = (s or "").strip().lower()
    if v == "win":
        return "Win"
    if v == "loss":
        return "Loss"
    return "Push"


# ---------- Bucketing ----------

def ev_bucket(ev: float):
    """0.01-wide. e.g. 0.0499 -> (0.04, '0.04 to 0.05')."""
    lo = floor(ev * 100) / 100.0
    hi = lo + 0.01
    return lo, f"{lo:.2f} to {hi:.2f}"


def prob_bucket(p: float):
    """0.10-wide. e.g. 0.4533 -> (0.40, '0.40 to 0.50')."""
    lo = floor(p * 10) / 10.0
    hi = lo + 0.10
    return lo, f"{lo:.2f} to {hi:.2f}"


def odds_bucket(ml: int):
    """50-wide symmetric.
       +112 -> '+100 to +150'
       -125 -> '-150 to -100'
       |ml|<100 -> '-100 to +100'
    """
    if abs(ml) < 100:
        return 0, "-100 to +100"

    if ml >= 100:
        lo = ((ml - 100) // 50) * 50 + 100
        hi = lo + 50
        return lo, f"+{lo} to +{hi}"

    a = abs(ml)
    lo_abs = ((a - 100) // 50) * 50 + 100
    hi_abs = lo_abs + 50
    return -hi_abs, f"-{hi_abs} to -{lo_abs}"


# ---------- Discovery / IO ----------

DATE_RE = re.compile(r"(\d{4}_\d{2}_\d{2})_ufc_graded\.csv$")


def discover_graded_files() -> list[str]:
    if not os.path.isdir(GRADED_DIR):
        return []

    return sorted(glob(os.path.join(GRADED_DIR, "*_ufc_graded.csv")))


def read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------- Graded collection ----------

def collect_graded() -> list[dict]:
    """
    Read already-graded files and normalize each row into the report format.

    Expected graded input columns:
      match_date,fighter_1,fighter_2,moneyline_f1,moneyline_f2,
      implied_prob_f1,implied_prob_f2,model_prob_f1,model_prob_f2,
      dratings_prob_f1,dratings_prob_f2,edge_f1,edge_f2,ev_f1,ev_f2,
      kelly_f1,kelly_f2,bet,result_fighter_1,result_fighter_2,match_result
    """
    graded = []

    for graded_path in discover_graded_files():
        rows = read_csv(graded_path)

        if not rows:
            continue

        for row in rows:
            bet = (row.get("bet") or "").strip()

            if bet == "fighter_1":
                fighter = row.get("fighter_1", "")
                opponent = row.get("fighter_2", "")
                ml_raw = row.get("moneyline_f1", "")
                implied_prob = safe_float(row.get("implied_prob_f1"))
                model_prob = safe_float(row.get("model_prob_f1"))
                dratings_prob = safe_float(row.get("dratings_prob_f1"))
                ev = safe_float(row.get("ev_f1"))
            elif bet == "fighter_2":
                fighter = row.get("fighter_2", "")
                opponent = row.get("fighter_1", "")
                ml_raw = row.get("moneyline_f2", "")
                implied_prob = safe_float(row.get("implied_prob_f2"))
                model_prob = safe_float(row.get("model_prob_f2"))
                dratings_prob = safe_float(row.get("dratings_prob_f2"))
                ev = safe_float(row.get("ev_f2"))
            else:
                continue

            try:
                ml = parse_ml(ml_raw)
            except Exception:
                continue

            outcome = grade_result(row.get("match_result", ""))

            if outcome == "Win":
                units = american_to_profit(ml)
            elif outcome == "Loss":
                units = -1.0
            else:
                units = 0.0

            graded.append({
                "date": row.get("match_date", ""),
                "fighter": fighter,
                "opponent": opponent,
                "moneyline": ml,
                "implied_prob": implied_prob,
                "model_prob": model_prob,
                "dratings_prob": dratings_prob,
                "ev": ev,
                "outcome": outcome,
                "units": units,
            })

    return graded


# ---------- Aggregation ----------

def empty_agg() -> dict:
    return {
        "bets": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "units": 0.0,
        "sum_implied": 0.0,
        "n_implied": 0,
        "sum_model": 0.0,
        "n_model": 0,
        "sum_dr": 0.0,
        "n_dr": 0,
        "sum_ml": 0,
        "n_ml": 0,
    }


def add_to_agg(a: dict, g: dict) -> None:
    a["bets"] += 1

    if g["outcome"] == "Win":
        a["wins"] += 1
    elif g["outcome"] == "Loss":
        a["losses"] += 1
    else:
        a["pushes"] += 1

    a["units"] += g["units"]

    if g["implied_prob"] is not None:
        a["sum_implied"] += g["implied_prob"]
        a["n_implied"] += 1

    if g["model_prob"] is not None:
        a["sum_model"] += g["model_prob"]
        a["n_model"] += 1

    if g["dratings_prob"] is not None:
        a["sum_dr"] += g["dratings_prob"]
        a["n_dr"] += 1

    a["sum_ml"] += g["moneyline"]
    a["n_ml"] += 1


def render_row(bucket_dim: str, bucket_label: str, a: dict) -> dict:
    decided = a["wins"] + a["losses"]
    win_pct = (a["wins"] / decided) if decided else ""
    roi = (a["units"] / a["bets"]) if a["bets"] else ""

    return {
        "league": LEAGUE,
        "market_type": MARKET,
        "bucket_dimension": bucket_dim,
        "bucket": bucket_label,
        "bets": a["bets"],
        "wins": a["wins"],
        "losses": a["losses"],
        "pushes": a["pushes"],
        "total": a["bets"],
        "win_pct": f"{win_pct:.4f}" if win_pct != "" else "",
        "units_flat": f"{a['units']:.4f}",
        "roi_flat": f"{roi:.4f}" if roi != "" else "",
        "avg_implied_prob": f"{(a['sum_implied'] / a['n_implied']):.4f}" if a["n_implied"] else "",
        "avg_model_prob": f"{(a['sum_model'] / a['n_model']):.4f}" if a["n_model"] else "",
        "avg_dratings_prob": f"{(a['sum_dr'] / a['n_dr']):.4f}" if a["n_dr"] else "",
        "avg_odds_american": fmt_ml(round(a["sum_ml"] / a["n_ml"])) if a["n_ml"] else "",
    }


REPORT_HEADERS = [
    "league",
    "market_type",
    "bucket_dimension",
    "bucket",
    "bets",
    "wins",
    "losses",
    "pushes",
    "total",
    "win_pct",
    "units_flat",
    "roi_flat",
    "avg_implied_prob",
    "avg_model_prob",
    "avg_dratings_prob",
    "avg_odds_american",
]


def write_report(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_HEADERS)
        w.writeheader()
        w.writerows(rows)


def write_summary(path: str, graded: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    a = empty_agg()
    for g in graded:
        add_to_agg(a, g)

    decided = a["wins"] + a["losses"]
    win_pct = (a["wins"] / decided) if decided else ""

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "league",
                "market_type",
                "Win",
                "Loss",
                "Push",
                "Total",
                "Win_Pct",
            ],
        )
        w.writeheader()
        w.writerow({
            "league": LEAGUE,
            "market_type": MARKET,
            "Win": a["wins"],
            "Loss": a["losses"],
            "Push": a["pushes"],
            "Total": a["bets"],
            "Win_Pct": f"{win_pct:.4f}" if win_pct != "" else "",
        })


def bucketed_rows(graded: list[dict], dim_name: str, key_fn) -> list[dict]:
    buckets = defaultdict(empty_agg)
    sort_keys = {}

    for g in graded:
        kv = key_fn(g)
        if kv is None:
            continue

        sort_val, label = kv
        add_to_agg(buckets[label], g)
        sort_keys[label] = sort_val

    return [
        render_row(dim_name, label, buckets[label])
        for label in sorted(buckets.keys(), key=lambda k: sort_keys[k])
    ]


# ---------- main ----------

def main() -> int:
    graded = collect_graded()

    print(
        f"Graded {len(graded)} bets across "
        f"{len({g['date'] for g in graded if g.get('date')})} dates."
    )

    write_summary(
        os.path.join(OUT_DIR, "ufc_summary_overall.csv"),
        graded,
    )

    write_report(
        os.path.join(REPORTS_DIR, "ufc_moneyline_by_ev.csv"),
        bucketed_rows(
            graded,
            "ev",
            lambda g: ev_bucket(g["ev"]) if g["ev"] is not None else None,
        ),
    )

    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_odds.csv"),
        bucketed_rows(
            graded,
            "odds",
            lambda g: odds_bucket(g["moneyline"]),
        ),
    )

    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_implied_prob.csv"),
        bucketed_rows(
            graded,
            "implied_prob",
            lambda g: prob_bucket(g["implied_prob"]) if g["implied_prob"] is not None else None,
        ),
    )

    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_model_prob.csv"),
        bucketed_rows(
            graded,
            "model_prob",
            lambda g: prob_bucket(g["model_prob"]) if g["model_prob"] is not None else None,
        ),
    )

    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_dratings_prob.csv"),
        bucketed_rows(
            graded,
            "dratings_prob",
            lambda g: prob_bucket(g["dratings_prob"]) if g["dratings_prob"] is not None else None,
        ),
    )

    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_date.csv"),
        bucketed_rows(
            graded,
            "by_date",
            lambda g: (g["date"], g["date"]) if g.get("date") else None,
        ),
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

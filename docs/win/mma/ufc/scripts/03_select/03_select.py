#!/usr/bin/env python3
"""
# docs/win/mma/ufc/scripts/03_select/03_select.py

Filters edge output using rules defined in docs/win/mma/ufc/config/markets.yaml.

Input:
    docs/win/mma/ufc/02_edges/{date}_ufc_edges.csv
    docs/win/mma/ufc/config/markets.yaml

Outputs:
    docs/win/mma/ufc/03_select/{date}_ufc_select.csv
        Legacy format. Picks only, sorted by pick_preference. Consumed by
        the_picks.html and kelly_calculator.html. Not written when no picks.

    docs/win/mma/ufc/03_select/detailed/{date}_ufc_select_detailed.csv
        Full audit format. Every fight from the edges file in edges-file
        order, with a `bet` column = fighter_1 | fighter_2 | no_bet.
        Always written, even when every row is no_bet.

Every run starts with a clean slate: existing files in both output directories
are deleted before processing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

# --- Paths ---
EDGES_DIR = Path("docs/win/mma/ufc/02_edges")
CONFIG_PATH = Path("docs/win/mma/ufc/config/markets.yaml")
OUT_DIR = Path("docs/win/mma/ufc/03_select")
DETAILED_DIR = OUT_DIR / "detailed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DETAILED_DIR.mkdir(parents=True, exist_ok=True)

# --- Clean slate: remove every stale select file (both dirs) before this run ---
for stale in OUT_DIR.glob("*_ufc_select.csv"):
    stale.unlink()
    print(f"DELETED stale legacy {stale}")

for stale in DETAILED_DIR.glob("*_ufc_select_detailed.csv"):
    stale.unlink()
    print(f"DELETED stale detailed {stale}")

# --- Load config ---
with CONFIG_PATH.open(encoding="utf-8") as f:
    config = yaml.safe_load(f)

ml_config = config["ufc"]["moneyline"]

enabled = ml_config.get("enabled", True)
pick_pref = ml_config.get("pick_preference", "best_ev")

odds_bands = ml_config.get("odds_bands", [])
edge_bands = ml_config.get("edge_bands", [])
ev_bands = ml_config.get("ev_bands", [])
kelly_bands = ml_config.get("kelly_bands", [])

# Default to None. When the key is absent from markets.yaml, no minimum is
# enforced. When the key is present, the corresponding probability is strictly
# required: missing / blank / non-numeric values fail the candidate.
model_prob_min = ml_config.get("model_probability_minimum", None)
dratings_prob_min = ml_config.get("dratings_probability_minimum", None)


def safe_float(val):
    try:
        if val is None:
            return None

        s = str(val).strip()
        if s == "":
            return None

        return float(s.replace("+", ""))
    except Exception:
        return None


def ml_to_float(ml_str):
    return safe_float(ml_str)


def in_any_band(value, bands):
    """Returns True if value falls within at least one [min, max] band."""
    v = safe_float(value)
    if v is None:
        return False

    try:
        return any(float(lo) <= v <= float(hi) for lo, hi in bands)
    except Exception:
        return False


# Maps pick_preference -> candidate dict key for sorting/selection.
PICK_METRIC_MAP = {
    "best_ev": "ev",
    "best_edge": "edge",
    "best_kelly": "kelly",
    "best_model_prob": "model_prob",
    "best_dratings_prob": "dratings_prob",
}


def pick_metric_from_values(candidate: dict, pref: str) -> float:
    """
    Return the metric value used for within-fight selection and final sorting.

    Uses candidate output keys. Missing / non-numeric values sort last.
    """
    col = PICK_METRIC_MAP.get(pref, "ev")
    v = safe_float(candidate.get(col))
    return v if v is not None else float("-inf")


def passes_filters(ml, edge, ev, kelly, model_prob, dratings_prob):
    if not enabled:
        return False

    if odds_bands and not in_any_band(ml, odds_bands):
        return False

    if edge_bands and not in_any_band(edge, edge_bands):
        return False

    if ev_bands and not in_any_band(ev, ev_bands):
        return False

    if kelly_bands and not in_any_band(kelly, kelly_bands):
        return False

    # When model_probability_minimum is configured, model_prob is strictly required.
    # Blank / missing / non-numeric model probability fails.
    if model_prob_min is not None:
        model_prob_val = safe_float(model_prob)
        if model_prob_val is None:
            return False
        if model_prob_val < float(model_prob_min):
            return False

    # When dratings_probability_minimum is configured, dratings_prob is strictly required.
    # Blank / missing / non-numeric DRatings probability fails.
    if dratings_prob_min is not None:
        dratings_prob_val = safe_float(dratings_prob)
        if dratings_prob_val is None:
            return False
        if dratings_prob_val < float(dratings_prob_min):
            return False

    return True


def make_candidate(row: dict, fighter_key: str) -> dict:
    """
    Build one fighter candidate (legacy format) from a fight row.

    fighter_key:
        f1 = fighter_1 side
        f2 = fighter_2 side
    """
    if fighter_key == "f1":
        return {
            "match_date": row["match_date"],
            "fighter": row["fighter_1"],
            "opponent": row["fighter_2"],
            "moneyline": row["moneyline_f1"],
            "implied_prob": row["implied_prob_f1"],
            "model_prob": row["model_prob_f1"],
            "dratings_prob": row["dratings_prob_f1"],
            "edge": row["edge_f1"],
            "ev": row["ev_f1"],
            "kelly": row["kelly_f1"],
        }

    return {
        "match_date": row["match_date"],
        "fighter": row["fighter_2"],
        "opponent": row["fighter_1"],
        "moneyline": row["moneyline_f2"],
        "implied_prob": row["implied_prob_f2"],
        "model_prob": row["model_prob_f2"],
        "dratings_prob": row["dratings_prob_f2"],
        "edge": row["edge_f2"],
        "ev": row["ev_f2"],
        "kelly": row["kelly_f2"],
    }


def candidate_passes(row: dict, fighter_key: str) -> bool:
    """
    Apply filters to one fighter side from the raw edge row.
    """
    suffix = "_f1" if fighter_key == "f1" else "_f2"

    ml = ml_to_float(row.get(f"moneyline{suffix}"))
    edge = safe_float(row.get(f"edge{suffix}"))
    ev = safe_float(row.get(f"ev{suffix}"))
    kelly = safe_float(row.get(f"kelly{suffix}"))
    model_prob = safe_float(row.get(f"model_prob{suffix}"))
    dratings_prob = safe_float(row.get(f"dratings_prob{suffix}"))

    return passes_filters(
        ml=ml,
        edge=edge,
        ev=ev,
        kelly=kelly,
        model_prob=model_prob,
        dratings_prob=dratings_prob,
    )


def make_detailed_row(row: dict, bet_value: str) -> dict:
    """
    Build one detailed (audit) row from a fight row.

    bet_value:
        'fighter_1' | 'fighter_2' | 'no_bet'
    """
    return {
        "match_date": row["match_date"],
        "fighter_1": row["fighter_1"],
        "fighter_2": row["fighter_2"],
        "moneyline_f1": row["moneyline_f1"],
        "moneyline_f2": row["moneyline_f2"],
        "implied_prob_f1": row["implied_prob_f1"],
        "implied_prob_f2": row["implied_prob_f2"],
        "model_prob_f1": row["model_prob_f1"],
        "model_prob_f2": row["model_prob_f2"],
        "dratings_prob_f1": row["dratings_prob_f1"],
        "dratings_prob_f2": row["dratings_prob_f2"],
        "edge_f1": row["edge_f1"],
        "edge_f2": row["edge_f2"],
        "ev_f1": row["ev_f1"],
        "ev_f2": row["ev_f2"],
        "kelly_f1": row["kelly_f1"],
        "kelly_f2": row["kelly_f2"],
        "bet": bet_value,
    }


# Output column orders.
LEGACY_FIELDS = [
    "match_date",
    "fighter",
    "opponent",
    "moneyline",
    "implied_prob",
    "model_prob",
    "dratings_prob",
    "edge",
    "ev",
    "kelly",
]

DETAILED_FIELDS = [
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
]


# --- Process each edges file ---
edges_files = sorted(EDGES_DIR.glob("*_ufc_edges.csv"))

if not edges_files:
    print("No edges files found.")
    raise SystemExit(1)


for edges_file in edges_files:
    date_str = edges_file.stem.replace("_ufc_edges", "")

    with edges_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows in {edges_file.name}, skipping")
        continue

    selected = []          # legacy output: picks only
    detailed_rows = []     # detailed output: every fight in edges-file order

    for row in rows:
        candidates = []  # list of (side_label, candidate_dict)

        if candidate_passes(row, "f1"):
            candidates.append(("fighter_1", make_candidate(row, "f1")))

        if candidate_passes(row, "f2"):
            candidates.append(("fighter_2", make_candidate(row, "f2")))

        if candidates:
            best_side, best_candidate = max(
                candidates,
                key=lambda sc: pick_metric_from_values(sc[1], pick_pref),
            )
            selected.append(best_candidate)
            bet_value = best_side
        else:
            bet_value = "no_bet"

        detailed_rows.append(make_detailed_row(row, bet_value))

    # Legacy: sort all picks by pick_preference.
    selected.sort(key=lambda c: pick_metric_from_values(c, pick_pref), reverse=True)

    # --- Write detailed (always, edges-file order) ---
    detailed_file = DETAILED_DIR / f"{date_str}_ufc_select_detailed.csv"
    with detailed_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAILED_FIELDS)
        writer.writeheader()
        writer.writerows(detailed_rows)

    bet_count = sum(1 for r in detailed_rows if r["bet"] != "no_bet")
    print(f"WROTE {detailed_file} ({len(detailed_rows)} fights, {bet_count} bets)")

    # --- Write legacy (only when there are picks) ---
    out_file = OUT_DIR / f"{date_str}_ufc_select.csv"
    if selected:
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LEGACY_FIELDS)
            writer.writeheader()
            writer.writerows(selected)
        print(f"WROTE {out_file} ({len(selected)} picks)")
    else:
        print(f"No picks passed filters for {date_str}")

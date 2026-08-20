#!/usr/bin/env python3
"""Validated launcher for basketball grading.

The locked-picks/replay grading implementation remains in
01_basketball_results_grade_core.py. This launcher clears stale mismatch diagnostics
before every run and rejects conflicting duplicate final scores instead of allowing
an arbitrary keep-last result.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

BASE = Path("docs/win/basketball")
RESULTS = BASE / "05_final_scores/results"
ERROR_DIR = BASE / "errors/05_final_scores"
CORE_PATH = Path(__file__).with_name("01_basketball_results_grade_core.py")
LEAGUES = ["nba", "ncaam", "wnba"]


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def comp(row: dict) -> tuple[str, str, str]:
    return (clean(row.get("game_date")), clean(row.get("home_team")).casefold(), clean(row.get("away_team")).casefold())


def score_sig(row: dict) -> tuple[str, str]:
    return clean(row.get("home_score")), clean(row.get("away_score"))


def clear_stale_diagnostics() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    for league in LEAGUES:
        for name in [f"{league}_game_id_no_match.csv", f"{league}_locked_game_id_no_match.csv"]:
            (ERROR_DIR / name).unlink(missing_ok=True)


def validate_final_scores() -> None:
    errors: list[str] = []
    for league in LEAGUES:
        by_id: dict[str, tuple[tuple[str, str, str], tuple[str, str], str]] = {}
        by_comp: dict[tuple[str, str, str], tuple[tuple[str, str], str, str]] = {}
        folder = RESULTS / league
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            with open(path, newline="", encoding="utf-8") as f:
                for n, row in enumerate(csv.DictReader(f), start=2):
                    key = comp(row); scores = score_sig(row); gid = clean(row.get("game_id"))
                    if not all(key) or not all(scores):
                        continue
                    if gid:
                        prior = by_id.get(gid)
                        current = (key, scores, f"{path}:{n}")
                        if prior and (prior[0] != key or prior[1] != scores):
                            errors.append(
                                f"{league.upper()} game_id {gid} has conflicting finals: {prior[2]} vs {current[2]}"
                            )
                        else:
                            by_id[gid] = current
                    prior_comp = by_comp.get(key)
                    if prior_comp and prior_comp[0] != scores:
                        errors.append(
                            f"{league.upper()} {key} has conflicting scores: {prior_comp[1]} vs {path}:{n}"
                        )
                    else:
                        by_comp[key] = (scores, f"{path}:{n}", gid)
    if errors:
        raise RuntimeError("Final-score integrity validation failed:\n" + "\n".join(errors[:50]))


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_results_grade_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load grading core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    clear_stale_diagnostics()
    validate_final_scores()
    core = load_core()
    core.main()
    text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    if "STATUS: COMPLETED WITH ERRORS" in text or "STATUS: FAILED" in text:
        sys.exit(1)


if __name__ == "__main__":
    main()

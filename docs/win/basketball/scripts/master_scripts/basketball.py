#!/usr/bin/env python3
# docs/win/basketball/scripts/master_scripts/basketball.py

import subprocess
import sys
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(".")

MASTER_LOG = Path("docs/win/basketball/errors/master_scripts/01_03_04_05_basketball.txt")

JOBS = [
    Path("docs/win/basketball/scripts/01_merge/merge_intake.py"),
    Path("docs/win/basketball/scripts/01_merge/build_juice_files.py"),
    Path("docs/win/basketball/scripts/03_edges/compute_ev_kelly.py"),
    Path("docs/win/basketball/scripts/04_select/basketball_select_bets.py"),
    Path("docs/win/basketball/scripts/04_select/daily_slate.py"),
    Path("docs/win/basketball/scripts/05_final_scores/01_basketball_results_grade.py"),
    Path("docs/win/basketball/scripts/05_final_scores/02_basketball_results_analyze.py"),
    Path("docs/win/basketball/scripts/05_final_scores/03_basketball_results_reports.py"),
    Path("docs/win/basketball/scripts/05_final_scores/04_basketball_results_dashboard.py"),
]

OUTPUT_DIRS = [
    Path("docs/win/basketball/errors/master_scripts"),

    Path("docs/win/basketball/errors/01_merge"),
    Path("docs/win/basketball/01_merge/nba/moneyline"),
    Path("docs/win/basketball/01_merge/nba/spread"),
    Path("docs/win/basketball/01_merge/nba/total"),
    Path("docs/win/basketball/01_merge/ncaam/moneyline"),
    Path("docs/win/basketball/01_merge/ncaam/spread"),
    Path("docs/win/basketball/01_merge/ncaam/total"),
    Path("docs/win/basketball/01_merge/wnba/moneyline"),
    Path("docs/win/basketball/01_merge/wnba/spread"),
    Path("docs/win/basketball/01_merge/wnba/total"),

    Path("docs/win/basketball/01_merge/01_merguiced/nba/moneyline"),
    Path("docs/win/basketball/01_merge/01_merguiced/nba/spread"),
    Path("docs/win/basketball/01_merge/01_merguiced/nba/total"),
    Path("docs/win/basketball/01_merge/01_merguiced/ncaam/moneyline"),
    Path("docs/win/basketball/01_merge/01_merguiced/ncaam/spread"),
    Path("docs/win/basketball/01_merge/01_merguiced/ncaam/total"),
    Path("docs/win/basketball/01_merge/01_merguiced/wnba/moneyline"),
    Path("docs/win/basketball/01_merge/01_merguiced/wnba/spread"),
    Path("docs/win/basketball/01_merge/01_merguiced/wnba/total"),

    Path("docs/win/basketball/errors/03_edges"),
    Path("docs/win/basketball/03_edges/ev_kelly/nba/moneyline"),
    Path("docs/win/basketball/03_edges/ev_kelly/nba/spread"),
    Path("docs/win/basketball/03_edges/ev_kelly/nba/total"),
    Path("docs/win/basketball/03_edges/ev_kelly/ncaam/moneyline"),
    Path("docs/win/basketball/03_edges/ev_kelly/ncaam/spread"),
    Path("docs/win/basketball/03_edges/ev_kelly/ncaam/total"),
    Path("docs/win/basketball/03_edges/ev_kelly/wnba/moneyline"),
    Path("docs/win/basketball/03_edges/ev_kelly/wnba/spread"),
    Path("docs/win/basketball/03_edges/ev_kelly/wnba/total"),

    Path("docs/win/basketball/errors/04_select"),
    Path("docs/win/basketball/04_select/daily_slate"),
    Path("docs/win/basketball/04_select/nba/daily_picks"),
    Path("docs/win/basketball/04_select/ncaam/daily_picks"),
    Path("docs/win/basketball/04_select/wnba/daily_picks"),

    Path("docs/win/basketball/errors/05_final_scores"),
    Path("docs/win/basketball/05_final_scores/results/nba/graded"),
    Path("docs/win/basketball/05_final_scores/results/ncaam/graded"),
    Path("docs/win/basketball/05_final_scores/results/wnba/graded"),

    Path("docs/win/basketball/05_final_scores/reports/nba/moneyline"),
    Path("docs/win/basketball/05_final_scores/reports/nba/spread"),
    Path("docs/win/basketball/05_final_scores/reports/nba/total"),
    Path("docs/win/basketball/05_final_scores/reports/nba/overview"),

    Path("docs/win/basketball/05_final_scores/reports/ncaam/moneyline"),
    Path("docs/win/basketball/05_final_scores/reports/ncaam/spread"),
    Path("docs/win/basketball/05_final_scores/reports/ncaam/total"),
    Path("docs/win/basketball/05_final_scores/reports/ncaam/overview"),

    Path("docs/win/basketball/05_final_scores/reports/wnba/moneyline"),
    Path("docs/win/basketball/05_final_scores/reports/wnba/spread"),
    Path("docs/win/basketball/05_final_scores/reports/wnba/total"),
    Path("docs/win/basketball/05_final_scores/reports/wnba/overview"),

    Path("docs"),
]


def init_log() -> None:
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_LOG, "w", encoding="utf-8") as f:
        f.write(f"=== 01_03_04_05_basketball RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat()} | {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
        log(f"ENSURED DIR: {path}")


def validate_scripts() -> None:
    missing = [str(path) for path in JOBS if not path.exists()]

    if missing:
        for path in missing:
            log(f"MISSING SCRIPT: {path}")
        raise FileNotFoundError("One or more required scripts do not exist.")


def run_script(script_path: Path) -> None:
    log(f"START SCRIPT: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            log(f"[STDOUT] {line}")

    if result.stderr:
        for line in result.stderr.splitlines():
            log(f"[STDERR] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed with exit code {result.returncode}: {script_path}"
        )

    log(f"SUCCESS SCRIPT: {script_path}")


def main():
    jobs_completed = 0

    try:
        init_log()
        log("MASTER SCRIPT START")

        ensure_output_dirs()
        validate_scripts()

        for script_path in JOBS:
            run_script(script_path)
            jobs_completed += 1

        log("--- SUMMARY ---")
        log(f"Jobs requested: {len(JOBS)}")
        log(f"Jobs completed: {jobs_completed}")
        log("STATUS: SUCCESS")

        print("Basketball 01/03/04/05 master complete.", flush=True)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("--- SUMMARY ---")
        log(f"Jobs requested: {len(JOBS)}")
        log(f"Jobs completed: {jobs_completed}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()

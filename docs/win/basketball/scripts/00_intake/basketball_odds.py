#!/usr/bin/env python3
"""Season-aware launcher for the ESPN basketball odds collector.

The implementation remains in basketball_odds_core.py. Normal production runs only
query leagues that are in season for the current New York date. Set
BASKETBALL_FORCE_ALL_LEAGUES=1 (or true/yes/on) to run every league explicitly.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
CORE_PATH = Path(__file__).with_name("basketball_odds_core.py")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def in_season(league: str, now: datetime) -> bool:
    league = league.lower()
    if league in {"nba", "ncaam"}:
        # Mirrors calculate_rolling_bias.py season boundaries:
        # Sep 1 through Jul 1, with Jul 2-Aug 31 treated as offseason.
        return now.month >= 9 or now.month <= 6 or (now.month == 7 and now.day == 1)
    if league == "wnba":
        # Operational WNBA window; full/offseason runs remain available via override.
        return 5 <= now.month <= 10
    return True


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_odds_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load odds core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    core = load_core()
    now = datetime.now(NY)
    force_all = truthy(os.getenv("BASKETBALL_FORCE_ALL_LEAGUES"))

    original = dict(core.LEAGUES)
    if force_all:
        active = original
    else:
        active = {k: v for k, v in original.items() if in_season(k, now)}

    skipped = sorted(set(original) - set(active))
    core.LEAGUES = active
    core.log(
        "SEASON GATE | "
        f"date={now.strftime('%Y_%m_%d')} | active={','.join(active) or 'none'} | "
        f"skipped={','.join(skipped) or 'none'} | force_all={int(force_all)}"
    )

    if not active:
        core.log("STATUS: SUCCESS (no leagues in season)")
        return

    core.main()

    # The core historically reported STATUS: SUCCESS even when per-event HTTP/build
    # errors were counted. Convert a non-zero current-run error count into a failed
    # process so GitHub Actions cannot be falsely green.
    text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    errors = 0
    for line in reversed(text.splitlines()):
        if " | Errors: " in line:
            try:
                errors = int(line.rsplit("Errors:", 1)[1].strip())
            except ValueError:
                errors = 1
            break
    if "STATUS: FAILED" in text or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Season-aware launcher for the D-Ratings basketball scraper."""
from __future__ import annotations

import importlib.util
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
CORE_PATH = Path(__file__).with_name("basketball_drat_scraper_core.py")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def in_season(key: str, now: datetime) -> bool:
    league = "ncaam" if key == "ncaa" else key
    if league in {"nba", "ncaam"}:
        return now.month >= 9 or now.month <= 6 or (now.month == 7 and now.day == 1)
    if league == "wnba":
        return 5 <= now.month <= 10
    return True


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_drat_scraper_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scraper core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    core = load_core()
    now = datetime.now(NY)
    force_all = truthy(os.getenv("BASKETBALL_FORCE_ALL_LEAGUES"))
    original = dict(core.URLS)
    active = original if force_all else {k: v for k, v in original.items() if in_season(k, now)}
    skipped = sorted(set(original) - set(active))
    core.URLS = active
    core.log(
        "SEASON GATE | "
        f"date={now.strftime('%Y_%m_%d')} | active={','.join(active) or 'none'} | "
        f"skipped={','.join(skipped) or 'none'} | force_all={int(force_all)}"
    )
    if not active:
        core.log("STATUS: SUCCESS (no leagues in season)")
        return

    core.main()
    text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    if "STATUS: FAILED" in text or "ERROR scraping" in text:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

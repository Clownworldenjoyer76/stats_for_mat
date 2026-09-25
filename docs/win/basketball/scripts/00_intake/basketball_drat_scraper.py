#!/usr/bin/env python3
"""Season-aware launcher for the D-Ratings basketball scraper.

Operational season dates are loaded from:
    docs/win/basketball/config/season_dates.yaml

Set BASKETBALL_FORCE_ALL_LEAGUES=1 (or true/yes/on) to run every
league explicitly.

Expected season_dates.yaml format:

nba:
  start_month: 10
  start_day: 15
  end_month: 7
  end_day: 1

ncaam:
  start_month: 10
  start_day: 31
  end_month: 7
  end_day: 1

wnba:
  start_month: 5
  start_day: 1
  end_month: 10
  end_day: 31

Both normal calendar-year windows and cross-year windows are supported.
"""
from __future__ import annotations

import importlib.util
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


NY = ZoneInfo("America/New_York")

CORE_PATH = Path(__file__).with_name("basketball_drat_scraper_core.py")

BASE = Path("docs/win/basketball")
SEASON_CONFIG = BASE / "config/season_dates.yaml"

SUPPORTED_LEAGUES = ("nba", "ncaam", "wnba")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_league(key: str) -> str:
    key = str(key).strip().lower()
    return "ncaam" if key == "ncaa" else key


SEASON_FIELDS = (
    "start_month",
    "start_day",
    "end_month",
    "end_day",
)


def validated_season_row(
    league: str,
    row,
) -> dict[str, int]:
    if not isinstance(
        row,
        dict,
    ):
        raise ValueError(
            "Missing season configuration "
            f"for league={league}"
        )

    values: dict[
        str,
        int,
    ] = {}

    for field in SEASON_FIELDS:
        if field not in row:
            raise ValueError(
                f"Missing {league}.{field} "
                f"in {SEASON_CONFIG}"
            )

        raw_value = row[
            field
        ]

        try:
            values[
                field
            ] = int(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid {league}.{field}: "
                f"{raw_value!r}"
            ) from exc

    for label in (
        "start",
        "end",
    ):
        month = values[
            f"{label}_month"
        ]
        day = values[
            f"{label}_day"
        ]

        try:
            datetime(
                2000,
                month,
                day,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid {league}.{label}: "
                f"month={month}, day={day}"
            ) from exc

    return values


def load_season_config() -> dict[str, dict[str, int]]:
    if not SEASON_CONFIG.exists():
        raise FileNotFoundError(
            f"Season config not found: "
            f"{SEASON_CONFIG}"
        )

    raw = (
        yaml.safe_load(
            SEASON_CONFIG.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            f"{SEASON_CONFIG} must contain "
            "a top-level mapping"
        )

    return {
        league: validated_season_row(
            league,
            raw.get(
                league
            ),
        )
        for league in SUPPORTED_LEAGUES
    }

def in_season(
    key: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> bool:
    """Return True when the current date is inside the league's season."""
    league = normalize_league(key)

    if league not in season_config:
        raise KeyError(
            f"No season configuration found for league={league}"
        )

    cfg = season_config[league]

    current_mmdd = (now.month, now.day)
    start_mmdd = (
        cfg["start_month"],
        cfg["start_day"],
    )
    end_mmdd = (
        cfg["end_month"],
        cfg["end_day"],
    )

    # Normal season window contained within one calendar year.
    # Example: May 1 through October 31.
    if start_mmdd <= end_mmdd:
        return start_mmdd <= current_mmdd <= end_mmdd

    # Cross-year season window.
    # Example: October 15 through July 1.
    return current_mmdd >= start_mmdd or current_mmdd <= end_mmdd


def load_core():
    spec = importlib.util.spec_from_file_location(
        "basketball_drat_scraper_core",
        CORE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load scraper core: {CORE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def select_active_urls(
    urls: dict,
    now: datetime,
) -> tuple[
    bool,
    dict,
    list[str],
]:
    force_all = truthy(
        os.getenv(
            "BASKETBALL_FORCE_ALL_LEAGUES"
        )
    )

    season_config = (
        load_season_config()
    )

    active = dict(
        urls
    )

    if not force_all:
        active = {
            key: value
            for key, value
            in urls.items()
            if in_season(
                key,
                now,
                season_config,
            )
        }

    skipped = sorted(
        set(urls)
        - set(active)
    )

    return (
        force_all,
        active,
        skipped,
    )

def main() -> None:
    core = load_core()
    now = datetime.now(NY)

    try:
        (
            force_all,
            active,
            skipped,
        ) = select_active_urls(
            core.URLS,
            now,
        )
    except Exception as exc:
        core.log(
            f"SEASON CONFIG FAILED: {exc}"
        )
        core.log("STATUS: FAILED")
        raise SystemExit(1) from exc

    core.URLS = active

    core.log(
        "SEASON GATE | "
        f"date={now.strftime('%Y_%m_%d')} | "
        f"config={SEASON_CONFIG} | "
        f"active={','.join(active) or 'none'} | "
        f"skipped={','.join(skipped) or 'none'} | "
        f"force_all={int(force_all)}"
    )

    if not active:
        core.log(
            "STATUS: SUCCESS (no leagues in season)"
        )
        return

    core.main()

    text = core.LOG_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if (
        "STATUS: FAILED" in text
        or "ERROR scraping" in text
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

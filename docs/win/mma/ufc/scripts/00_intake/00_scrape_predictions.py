# docs/win/mma/ufc/scripts/00_intake/00_scrape_predictions.py

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


URL = "https://www.dratings.com/predictor/ufc-mma-predictions/"
OUTDIR = Path("docs/win/mma/ufc/00_intake/predictions")

PLAYWRIGHT_TIMEZONE = "America/New_York"

NOISE_LINES = {
    "Sports Ratings, Prediction, & Analysis",
    "Ratings",
    "Predictions",
    "Tools",
    "Sportsbooks",
    "About",
    "Blog",
    "★ PREMIUM",
    "Odds Feed",
    "Offshore Odds",
    "Vegas Odds",
    "UFC MMA Predictions",
    "Upcoming",
    "Completed",
    "Season",
    "Methodology",
    "Related",
    "Time Fighters Win Best",
    "ML Bet",
    "Value",
    "More Details",
}

SECTION_HEADER_PATTERNS = [
    r"^Upcoming Fights for ",
    r"^Fights for ",
]

STOP_PATTERNS = [
    r"^Completed Fights$",
    r"^Load More Fights$",
    r"^Season Prediction Results$",
]


def ensure_pkg(pkg: str, import_name: str | None = None) -> None:
    module_name = import_name or pkg
    try:
        __import__(module_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


ensure_pkg("playwright")

try:
    from playwright.sync_api import sync_playwright
except Exception:
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.sync_api import sync_playwright

try:
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
except Exception:
    pass


def clean_line(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_date_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", value))


def is_time_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}\s+(AM|PM)", value, re.I))


def is_percent_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}\.\d+%", value))


def is_moneyline(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]\d{2,5}", value))


def is_section_header(value: str) -> bool:
    return any(re.search(pattern, value, re.I) for pattern in SECTION_HEADER_PATTERNS)


def is_stop_line(value: str) -> bool:
    return any(re.search(pattern, value, re.I) for pattern in STOP_PATTERNS)


def normalize_date(value: str) -> str:
    dt = datetime.strptime(value, "%m/%d/%Y")
    return dt.strftime("%Y_%m_%d")


def pct_to_decimal_str(value: str) -> str:
    num = float(value.replace("%", "")) / 100.0
    return f"{num:.3f}"


def split_fighter_and_prob(value: str) -> tuple[str | None, str | None]:
    """
    Example:
      'Mike Malott 25.4%' -> ('Mike Malott', '0.254')
    """
    match = re.fullmatch(r"(.+?)\s+(\d{1,3}\.\d+%)", value)
    if not match:
        return None, None

    fighter = match.group(1).strip()
    prob = pct_to_decimal_str(match.group(2))
    return fighter, prob


def expand_embedded_percent_line(value: str) -> list[str]:
    """
    Handles lines like:
      Song Yadong 33.1%

    Converts them into:
      Song Yadong
      33.1%

    Leaves normal lines unchanged.
    """
    fighter, prob = split_fighter_and_prob(value)
    if fighter and prob:
        raw_pct = f"{float(prob) * 100:.1f}%"
        return [fighter, raw_pct]
    return [value]


def scrape_body_text() -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        context = browser.new_context(
            viewport={"width": 1600, "height": 5000},
            timezone_id=PLAYWRIGHT_TIMEZONE,
            locale="en-US",
        )

        page = context.new_page()

        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(6000)

        for _ in range(16):
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(700)

        body_text = page.locator("body").inner_text(timeout=30000)

        browser.close()
        return body_text


def normalize_lines(body_text: str) -> list[str]:
    raw_lines = [clean_line(line) for line in body_text.splitlines()]
    raw_lines = [line for line in raw_lines if line and line not in NOISE_LINES]

    lines: list[str] = []
    for line in raw_lines:
        lines.extend(expand_embedded_percent_line(line))

    return [line for line in lines if line and line not in NOISE_LINES]


def parse_fight_block(raw_date: str, block: list[str]) -> dict[str, str] | None:
    """
    Handles both formats:

      05/30/2026
      04:00 AM
      Deiveson Figueiredo
      Song Yadong
      33.1%
      66.9%
      +330
      -415

    And older / compressed text like:

      04/18/2026
      05:00 PM
      Gilbert Burns
      Mike Malott 25.4%
      74.6%
      +290
      -330

    Moneylines are optional because some Dratings rows do not show them.
    This predictions file only needs fighter names and win probabilities.
    """
    expanded: list[str] = []
    for item in block:
        expanded.extend(expand_embedded_percent_line(item))

    percents = [x for x in expanded if is_percent_line(x)]
    non_market_text = [
        x for x in expanded
        if not is_percent_line(x)
        and not is_moneyline(x)
        and not is_time_line(x)
        and not is_date_line(x)
        and not is_section_header(x)
        and not is_stop_line(x)
        and x not in NOISE_LINES
    ]

    if len(non_market_text) < 2 or len(percents) < 2:
        return None

    fighter_1 = non_market_text[0].strip()
    fighter_2 = non_market_text[1].strip()

    if not fighter_1 or not fighter_2:
        return None

    return {
        "match_date": normalize_date(raw_date),
        "fighter_1": fighter_1,
        "fighter_2": fighter_2,
        "fighter_1_win_prob": pct_to_decimal_str(percents[0]),
        "fighter_2_win_prob": pct_to_decimal_str(percents[1]),
    }


def parse_rows(body_text: str) -> list[dict[str, str]]:
    lines = normalize_lines(body_text)

    rows: list[dict[str, str]] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]

        if is_stop_line(line):
            break

        if not is_date_line(line):
            idx += 1
            continue

        if idx + 1 >= len(lines) or not is_time_line(lines[idx + 1]):
            idx += 1
            continue

        raw_date = lines[idx]
        idx += 2

        block: list[str] = []

        while idx < len(lines):
            current = lines[idx]

            if is_stop_line(current):
                break

            if is_section_header(current):
                idx += 1
                continue

            if is_date_line(current):
                break

            block.append(current)
            idx += 1

        parsed = parse_fight_block(raw_date, block)
        if parsed:
            rows.append(parsed)
        else:
            print(f"SKIP unparsable block | date={raw_date} | block={block}")

    return rows


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, str]] = []

    for row in rows:
        key = (
            row["match_date"],
            row["fighter_1"].strip().lower(),
            row["fighter_2"].strip().lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(row)

    return out


def write_output_files(rows: list[dict[str, str]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = dedupe_rows(rows)

    rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_date[row["match_date"]].append(row)

    fieldnames = [
        "match_date",
        "fighter_1",
        "fighter_2",
        "fighter_1_win_prob",
        "fighter_2_win_prob",
    ]

    for match_date, date_rows in sorted(rows_by_date.items()):
        outfile = OUTDIR / f"{match_date}_ufc_predictions.csv"
        with outfile.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(date_rows)

        print(f"WROTE {outfile} ({len(date_rows)} rows)")


def main() -> int:
    try:
        body_text = scrape_body_text()
        rows = parse_rows(body_text)

        if not rows:
            print("No rows parsed from page.")
            return 1

        write_output_files(rows)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

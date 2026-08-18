from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


URL = "https://www.oddstrader.com/ufc/?eid&g=game&m=money"
OUTDIR = Path("docs/win/mma/ufc/00_intake/sportsbook")

NOISE_PATTERNS = [
    r"^Personalize your notifications",
    r"^GOT IT$",
    r"^GET THE APP$",
    r"^ODDS$",
    r"^OFFSHORE-SPORTSBOOKS$",
    r"^NEWS$",
    r"^PICKS$",
    r"^BETTING$",
    r"^STATES$",
    r"^Today$",
    r"^Yesterday$",
    r"^Money$",
    r"^Game$",
    r"^Total$",
    r"^Merged$",
    r"^NCAAB$",
    r"^NBA$",
    r"^MLB$",
    r"^NHL$",
    r"^NFL$",
    r"^NCAAF$",
    r"^SOCCER$",
    r"^TENNIS$",
    r"^OTHER$",
    r"^UFC$",
]

BOOK_NAMES = {
    "BetOnline",
    "BetAnything",
    "Bovada",
    "Heritage",
    "Bookmaker",
    "JustBet",
    "Bet105",
    "MyBookie",
    "Everygame",
}


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
    value = value.replace("\u00bd", ".5").replace("\u00bc", ".25").replace("\u00be", ".75")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_noise(value: str) -> bool:
    return any(re.search(pattern, value, re.I) for pattern in NOISE_PATTERNS)


def is_date_line(value: str) -> bool:
    return bool(re.fullmatch(r"(MON|TUE|WED|THU|FRI|SAT|SUN)\s+\d{2}/\d{2}", value, re.I))


def is_time_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}\s+(AM|PM)", value, re.I))


def is_initials_line(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z.\-]*", value))


def is_percent_or_dash(value: str) -> bool:
    return value == "-" or bool(re.fullmatch(r"\d{1,3}%", value))


def is_moneyline(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]\d{2,5}", value))


def normalize_match_date(date_line: str) -> str:
    match = re.search(r"(\d{2})/(\d{2})", date_line)
    if not match:
        raise ValueError(f"Could not parse date from line: {date_line}")

    month, day = match.group(1), match.group(2)
    year = datetime.now().year
    return f"{year}_{month}_{day}"


def filter_lines(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    for line in lines:
        if not line:
            continue
        if is_noise(line):
            continue
        filtered.append(line)
    return filtered


def parse_fighter_block(lines: list[str], start_idx: int) -> dict | None:
    """
    Expected pattern:
      initials
      fighter name
      "-" or "100%"
      opener moneyline
      sportsbook anchor
      first post-anchor moneyline  <-- desired key value
    """
    if start_idx + 5 >= len(lines):
        return None

    if not is_initials_line(lines[start_idx]):
        return None

    fighter_code = lines[start_idx]
    fighter_name = lines[start_idx + 1]
    pct_or_dash = lines[start_idx + 2]
    opener = lines[start_idx + 3]
    anchor_book = lines[start_idx + 4]
    key_moneyline = lines[start_idx + 5]

    if is_date_line(fighter_name) or is_time_line(fighter_name):
        return None
    if not is_percent_or_dash(pct_or_dash):
        return None
    if not is_moneyline(opener):
        return None
    if anchor_book not in BOOK_NAMES:
        return None
    if not is_moneyline(key_moneyline):
        return None

    idx = start_idx + 6
    while idx < len(lines):
        current = lines[idx]
        if is_date_line(current) or is_time_line(current) or is_initials_line(current):
            break
        idx += 1

    return {
        "fighter_code": fighter_code,
        "fighter_name": fighter_name,
        "moneyline": key_moneyline,
        "next_idx": idx,
    }


def parse_rows(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    idx = 0

    while idx < len(lines):
        if is_date_line(lines[idx]):
            match_date = normalize_match_date(lines[idx])

            idx += 1
            if idx < len(lines) and is_time_line(lines[idx]):
                idx += 1

            fighter_1 = parse_fighter_block(lines, idx)
            if not fighter_1:
                continue
            idx = fighter_1["next_idx"]

            fighter_2 = parse_fighter_block(lines, idx)
            if not fighter_2:
                continue
            idx = fighter_2["next_idx"]

            rows.append(
                {
                    "sport": "mma",
                    "league": "ufc",
                    "match_date": match_date,
                    "fighter_1": fighter_1["fighter_name"],
                    "fighter_2": fighter_2["fighter_name"],
                    "moneyline_fighter_1": fighter_1["moneyline"],
                    "moneyline_fighter_2": fighter_2["moneyline"],
                }
            )
            continue

        idx += 1

    return rows


def scrape_body_text() -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 3200})

        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(8000)

        try:
            page.locator("text=GOT IT").click(timeout=2500)
            page.wait_for_timeout(1200)
        except Exception:
            pass

        for _ in range(12):
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(700)

        body_text = page.locator("body").inner_text(timeout=30000)
        browser.close()
        return body_text


def write_output_files(rows: list[dict[str, str]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_date[row["match_date"]].append(row)

    fieldnames = [
        "sport",
        "league",
        "match_date",
        "fighter_1",
        "fighter_2",
        "moneyline_fighter_1",
        "moneyline_fighter_2",
    ]

    for match_date, date_rows in sorted(rows_by_date.items()):
        outfile = OUTDIR / f"{match_date}_ufc_odds.csv"
        with outfile.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(date_rows)

        print(f"WROTE {outfile} ({len(date_rows)} rows)")


def main() -> int:
    try:
        body_text = scrape_body_text()
        lines = [clean_line(line) for line in body_text.splitlines()]
        lines = filter_lines(lines)

        rows = parse_rows(lines)
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

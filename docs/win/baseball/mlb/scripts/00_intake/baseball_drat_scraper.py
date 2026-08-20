#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/00_intake/baseball_drat_scraper.py

import json
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

import pytz
from playwright.sync_api import sync_playwright


URLS = {
    "mlb": "https://www.dratings.com/predictor/mlb-baseball-predictions/",
}


UTC = pytz.utc
ET = pytz.timezone("America/New_York")

ERROR_DIR = Path("docs/win/baseball/mlb/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "baseball_drat_scraper.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== baseball_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        dt_utc = UTC.localize(dt)
        dt_et = dt_utc.astimezone(ET)
        return dt_et.strftime("%m/%d/%Y %I:%M %p")
    except Exception:
        return date_time_str


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")

    all_rows = []
    tables = page.query_selector_all("table")

    for table in tables:
        rows = table.query_selector_all("tbody tr")
        for r in rows:
            cells = [c.inner_text().strip() for c in r.query_selector_all("td")]
            if cells:
                try:
                    cells[0] = convert_utc_to_et(cells[0].replace("\n", " "))
                except Exception:
                    pass
                all_rows.append(cells)

    return all_rows


def main():
    files_written = []

    try:
        date = datetime.now(ET).strftime("%Y_%m_%d")
        raw_dir = Path("docs/win/baseball/mlb/00_intake/drat_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    )
                }
            )

            for sport, url in URLS.items():
                log(f"Scraping {sport.upper()}")

                try:
                    raw = scrape_page(page, url)
                    raw_path = raw_dir / f"{date}_{sport}_raw.json"

                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(raw, f, indent=2)

                    files_written.append((str(raw_path), len(raw)))
                    log(f"WROTE {raw_path} ({len(raw)} rows)")

                except Exception as e:
                    log(f"ERROR scraping {sport}: {e}\n{traceback.format_exc()}")

                time.sleep(random.uniform(2, 4))

            browser.close()

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("Baseball drat scraper complete.")


if __name__ == "__main__":
    main()

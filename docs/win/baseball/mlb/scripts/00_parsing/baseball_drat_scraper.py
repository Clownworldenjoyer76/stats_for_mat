#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/baseball_drat_scraper.py

import json
from pathlib import Path
from datetime import datetime
import pytz
from playwright.sync_api import sync_playwright

URLS = {
    "mlb": "https://www.dratings.com/predictor/mlb-baseball-predictions/",
}

UTC = pytz.utc
ET  = pytz.timezone("America/New_York")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt     = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        dt_utc = UTC.localize(dt)
        dt_et  = dt_utc.astimezone(ET)
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
                # apply UTC → ET conversion on first column (date column)
                try:
                    cells[0] = convert_utc_to_et(cells[0].replace("\n", " "))
                except:
                    pass

                all_rows.append(cells)

    return all_rows


def main():
    date = datetime.now(ET).strftime("%Y_%m_%d")

    raw_dir = Path("docs/win/baseball/00_intake/drat_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        })

        raw = scrape_page(page, URLS["mlb"])

        raw_path = raw_dir / f"{date}_mlb_raw.json"
        with open(raw_path, "w") as f:
            json.dump(raw, f, indent=2)

        browser.close()


if __name__ == "__main__":
    main()

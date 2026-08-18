#!/usr/bin/env python3
"""
ufc_input.py

Reads raw UFC manual data from:
    docs/manual_data.html

Outputs one predictions file and one sportsbook file per converted NYC match date:

    docs/win/mma/ufc/00_intake/predictions/YYYY_MM_DD_ufc_predictions.csv
    docs/win/mma/ufc/00_intake/sportsbook/YYYY_MM_DD_ufc_odds.csv

Run from repo root:
    python docs/win/mma/ufc/scripts/00_parsing/ufc_input.py
"""

from __future__ import annotations

import csv
import html
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo


ROOT = Path.cwd()

INPUT_HTML = ROOT / "docs" / "manual_data.html"

PREDICTIONS_DIR = ROOT / "docs" / "win" / "mma" / "ufc" / "00_intake" / "predictions"
SPORTSBOOK_DIR = ROOT / "docs" / "win" / "mma" / "ufc" / "00_intake" / "sportsbook"

UTC = ZoneInfo("UTC")
NYC = ZoneInfo("America/New_York")


DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*[AP]M$", re.IGNORECASE)
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
MONEYLINE_RE = re.compile(r"^[+-]\d+$")

IGNORE_LINES = {
    "time",
    "fighters",
    "win",
    "best",
    "ml",
    "bet",
    "value",
    "more details",
    "value more details",
    "time fighters win best",
    "ml bet",
}


@dataclass
class ParsedFight:
    match_date: str
    fighter_1: str
    fighter_2: str
    fighter_1_win_prob: str
    fighter_2_win_prob: str
    moneyline_fighter_1: str
    moneyline_fighter_2: str


def read_manual_html_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    raw = path.read_text(encoding="utf-8-sig", errors="replace")

    # Prefer textarea contents if the frontend stores pasted data there.
    textarea_matches = re.findall(
        r"<textarea\b[^>]*>(.*?)</textarea>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if textarea_matches:
        text = "\n".join(textarea_matches)
    else:
        # Convert common HTML boundaries to newlines before stripping tags.
        text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", raw)
        text = re.sub(r"(?i)</\s*(div|p|tr|td|th|li|span|section|article|pre)\s*>", "\n", text)
        text = re.sub(r"<[^>]+>", "\n", text)

    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    return text


def normalize_lines(text: str) -> list[str]:
    lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"[ ]{2,}", " ", line)

        if not line:
            continue

        # Keep tab-separated fighter/probability rows together.
        line = re.sub(r"\t+", "\t", line)

        if should_ignore_line(line):
            continue

        lines.append(line)

    return lines


def should_ignore_line(line: str) -> bool:
    cleaned = line.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("\t", " ")

    if cleaned in IGNORE_LINES:
        return True

    if "volatility bet value active" in cleaned:
        return True

    return False


def convert_utc_to_nyc_match_date(date_text: str, time_text: str | None) -> str:
    if time_text:
        raw_dt = datetime.strptime(
            f"{date_text} {time_text.upper().replace(' ', '')}",
            "%m/%d/%Y %I:%M%p",
        )
    else:
        raw_dt = datetime.strptime(date_text, "%m/%d/%Y")

    utc_dt = raw_dt.replace(tzinfo=UTC)
    nyc_dt = utc_dt.astimezone(NYC)

    return nyc_dt.strftime("%Y_%m_%d")


def parse_percent_to_prob(value: str) -> str:
    match = PERCENT_RE.search(value)
    if not match:
        raise ValueError(f"Could not parse percentage from: {value}")

    prob = float(match.group(1)) / 100.0
    return f"{prob:.3f}"


def clean_fighter_name(value: str) -> str:
    value = PERCENT_RE.sub("", value)
    value = value.replace("\t", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_percent(value: str) -> str | None:
    match = PERCENT_RE.search(value)
    if not match:
        return None

    return f"{float(match.group(1)) / 100.0:.3f}"


def is_date_line(line: str) -> bool:
    return bool(DATE_RE.match(line.strip()))


def is_time_line(line: str) -> bool:
    return bool(TIME_RE.match(line.strip()))


def is_moneyline_line(line: str) -> bool:
    return bool(MONEYLINE_RE.match(line.strip()))


def next_useful_line(lines: list[str], start: int) -> tuple[int, str]:
    i = start

    while i < len(lines):
        line = lines[i].strip()

        if line and not should_ignore_line(line):
            return i, line

        i += 1

    raise ValueError("Unexpected end of input while parsing fight block.")


def parse_fights(lines: list[str]) -> list[ParsedFight]:
    fights: list[ParsedFight] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not is_date_line(line):
            i += 1
            continue

        date_text = line
        i += 1

        time_text: str | None = None

        if i < len(lines) and is_time_line(lines[i]):
            time_text = lines[i]
            i += 1

        match_date = convert_utc_to_nyc_match_date(date_text, time_text)

        i, fighter_1_line = next_useful_line(lines, i)
        fighter_1 = clean_fighter_name(fighter_1_line)
        i += 1

        i, fighter_2_line = next_useful_line(lines, i)
        fighter_2 = clean_fighter_name(fighter_2_line)
        fighter_1_prob = extract_percent(fighter_2_line)
        i += 1

        if not fighter_1_prob:
            i, pct_1_line = next_useful_line(lines, i)
            fighter_1_prob = parse_percent_to_prob(pct_1_line)
            i += 1

        i, pct_2_line = next_useful_line(lines, i)
        fighter_2_prob = parse_percent_to_prob(pct_2_line)
        i += 1

        i, ml_1_line = next_useful_line(lines, i)
        if not is_moneyline_line(ml_1_line):
            raise ValueError(f"Expected fighter 1 moneyline, got: {ml_1_line}")
        moneyline_fighter_1 = ml_1_line.strip()
        i += 1

        i, ml_2_line = next_useful_line(lines, i)
        if not is_moneyline_line(ml_2_line):
            raise ValueError(f"Expected fighter 2 moneyline, got: {ml_2_line}")
        moneyline_fighter_2 = ml_2_line.strip()
        i += 1

        fights.append(
            ParsedFight(
                match_date=match_date,
                fighter_1=fighter_1,
                fighter_2=fighter_2,
                fighter_1_win_prob=fighter_1_prob,
                fighter_2_win_prob=fighter_2_prob,
                moneyline_fighter_1=moneyline_fighter_1,
                moneyline_fighter_2=moneyline_fighter_2,
            )
        )

    return fights


def group_by_match_date(fights: Iterable[ParsedFight]) -> dict[str, list[ParsedFight]]:
    grouped: dict[str, list[ParsedFight]] = defaultdict(list)

    for fight in fights:
        grouped[fight.match_date].append(fight)

    return dict(grouped)


def write_predictions_file(match_date: str, fights: list[ParsedFight]) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    out_file = PREDICTIONS_DIR / f"{match_date}_ufc_predictions.csv"

    with out_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "match_date",
                "fighter_1",
                "fighter_2",
                "fighter_1_win_prob",
                "fighter_2_win_prob",
            ]
        )

        for fight in fights:
            writer.writerow(
                [
                    fight.match_date,
                    fight.fighter_1,
                    fight.fighter_2,
                    fight.fighter_1_win_prob,
                    fight.fighter_2_win_prob,
                ]
            )

    return out_file


def write_sportsbook_file(match_date: str, fights: list[ParsedFight]) -> Path:
    SPORTSBOOK_DIR.mkdir(parents=True, exist_ok=True)

    out_file = SPORTSBOOK_DIR / f"{match_date}_ufc_odds.csv"

    with out_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sport",
                "league",
                "match_date",
                "fighter_1",
                "fighter_2",
                "moneyline_fighter_1",
                "moneyline_fighter_2",
            ]
        )

        for fight in fights:
            writer.writerow(
                [
                    "mma",
                    "ufc",
                    fight.match_date,
                    fight.fighter_1,
                    fight.fighter_2,
                    fight.moneyline_fighter_1,
                    fight.moneyline_fighter_2,
                ]
            )

    return out_file


def main() -> int:
    print("=== UFC manual input parser ===")
    print(f"Reading: {INPUT_HTML}")

    text = read_manual_html_text(INPUT_HTML)
    lines = normalize_lines(text)
    fights = parse_fights(lines)

    if not fights:
        print("No UFC fights parsed. No files written.")
        return 0

    grouped = group_by_match_date(fights)

    print(f"Parsed fights: {len(fights)}")
    print(f"Match dates: {', '.join(sorted(grouped))}")

    for match_date, date_fights in sorted(grouped.items()):
        pred_file = write_predictions_file(match_date, date_fights)
        odds_file = write_sportsbook_file(match_date, date_fights)

        print(f"Wrote predictions: {pred_file}")
        print(f"Wrote sportsbook:   {odds_file}")

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

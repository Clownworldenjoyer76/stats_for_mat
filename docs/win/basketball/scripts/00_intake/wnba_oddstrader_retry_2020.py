#!/usr/bin/env python3

import csv
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE = Path(r"C:\Users\Mat\Downloads\chatgpt_is_retarded")

FAIL_SOURCES = [
    BASE / "2020_WNBA_after_alias_retry.csv",
    BASE / "2020_WNBA_still_failed.csv",
    BASE / "2020_WNBA_failed_dates.csv",
]

FINAL_FAILFILE = BASE / "2020_WNBA_final_failed.csv"

FIELDS = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "odds_last_update",
    "game_time",
    "home_team",
    "away_team",
    "home_spread",
    "away_spread",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]


def clean(value):
    return (
        str(value or "")
        .replace("½", ".5")
        .replace("−", "-")
        .strip()
    )


def decimal_american(value):
    value = clean(value)

    if not re.fullmatch(r"[+-]\d+", value):
        return ""

    number = int(value)

    if number > 0:
        decimal = 1 + number / 100
    else:
        decimal = 1 + 100 / abs(number)

    return f"{decimal:.6f}".rstrip("0").rstrip(".")


def parse_cell(value, market):
    value = clean(value)

    if not value or value == "-":
        return None

    parts = value.split()

    if market == "money":
        for part in parts:
            if re.fullmatch(r"[+-]\d+", part):
                return {
                    "price": part,
                }

        return None

    if len(parts) < 2:
        return None

    line = clean(parts[0])
    price = clean(parts[-1])

    if not re.fullmatch(r"[+-]\d+", price):
        return None

    if market == "spread":
        if not (
            re.fullmatch(
                r"[+-]?\d+(?:\.5)?",
                line,
            )
            or line.upper() == "PK"
        ):
            return None

    if market == "total":
        if not re.fullmatch(
            r"[ou]\d+(?:\.5)?",
            line,
            re.I,
        ):
            return None

    return {
        "line": line,
        "price": price,
    }


def select_market(page, name):
    clicked = page.evaluate(
        """
        name => {
            const wanted = name.toLowerCase();

            const candidates = [
                ...document.querySelectorAll("li")
            ];

            for (const element of candidates) {
                const text =
                    (element.innerText || "")
                    .trim()
                    .toLowerCase();

                const rect =
                    element.getBoundingClientRect();

                if (
                    text === wanted &&
                    rect.width > 0 &&
                    rect.height > 0
                ) {
                    element.click();
                    return true;
                }
            }

            return false;
        }
        """,
        name,
    )

    if not clicked:
        raise RuntimeError(
            f"Could not select {name}"
        )

    page.wait_for_timeout(3500)


def scrape_rows(page, market):
    raw_rows = page.evaluate(
        """
        () => [
            ...document.querySelectorAll("tbody tr")
        ]
        .map(row => {
            const team =
                row.querySelector(".teamName");

            if (!team) {
                return null;
            }

            const values =
                (row.innerText || "")
                .split("\\n")
                .map(x => x.trim())
                .filter(Boolean);

            return {
                team: team.innerText.trim(),
                values: values
            };
        })
        .filter(Boolean)
        """
    )

    output = []

    for raw in raw_rows:
        values = [
            clean(value)
            for value in raw["values"]
            if clean(value)
        ]

        record_index = None

        for index, value in enumerate(values):
            if re.fullmatch(
                r"\d+-\d+",
                value,
            ):
                record_index = index
                break

        if record_index is None:
            continue

        #
        # Proven rendered-row structure:
        #
        # abbreviation
        # team
        # record
        # score
        # best line
        # best-line sportsbook
        # opener
        # BetOnline
        # BetAnything
        # Bovada
        # Heritage
        # Bookmaker
        # JustBet
        #

        score_index = record_index + 1
        best_line_index = record_index + 2
        best_book_index = record_index + 3
        opener_index = record_index + 4
        betonline_index = record_index + 5

        if opener_index >= len(values):
            opener_text = ""
        else:
            opener_text = values[
                opener_index
            ]

        if betonline_index >= len(values):
            betonline_text = ""
        else:
            betonline_text = values[
                betonline_index
            ]

        output.append(
            {
                "team": clean(
                    raw["team"]
                ),
                "score": (
                    values[score_index]
                    if score_index < len(values)
                    else ""
                ),
                "best_line": (
                    values[best_line_index]
                    if best_line_index < len(values)
                    else ""
                ),
                "best_book": (
                    values[best_book_index]
                    if best_book_index < len(values)
                    else ""
                ),
                "opener_raw":
                    opener_text,
                "betonline_raw":
                    betonline_text,
                "opener":
                    parse_cell(
                        opener_text,
                        market,
                    ),
                "betonline":
                    parse_cell(
                        betonline_text,
                        market,
                    ),
            }
        )

    if not output:
        raise RuntimeError(
            f"{market}: zero rendered team rows"
        )

    if len(output) % 2 != 0:
        raise RuntimeError(
            f"{market}: odd number of team rows: "
            f"{len(output)}"
        )

    return output


def choose(row):
    opener = row.get(
        "opener"
    )

    if opener:
        return (
            opener,
            "Opener",
        )

    betonline = row.get(
        "betonline"
    )

    if betonline:
        return (
            betonline,
            "BetOnline",
        )

    return (
        None,
        "",
    )


def build_pairs(rows):
    games = []

    for index in range(
        0,
        len(rows),
        2,
    ):
        away = rows[index]
        home = rows[index + 1]

        games.append(
            {
                "away_team":
                    away["team"],
                "home_team":
                    home["team"],
                "away":
                    away,
                "home":
                    home,
            }
        )

    return games


def assert_same_order(
    spread_rows,
    money_rows,
    total_rows,
):
    spread_teams = [
        row["team"]
        for row in spread_rows
    ]

    money_teams = [
        row["team"]
        for row in money_rows
    ]

    total_teams = [
        row["team"]
        for row in total_rows
    ]

    if spread_teams != money_teams:
        raise RuntimeError(
            "Team order changed between "
            "Spread and Money"
        )

    if spread_teams != total_teams:
        raise RuntimeError(
            "Team order changed between "
            "Spread and Total"
        )


def spread_number(value):
    value = clean(value)

    if value.upper() == "PK":
        return 0.0

    try:
        return float(value)
    except Exception:
        return None


def make_game_row(
    game_date,
    away_team,
    home_team,
    away_spread_row,
    home_spread_row,
    away_money_row,
    home_money_row,
    away_total_row,
    home_total_row,
):
    away_spread, away_spread_source = (
        choose(
            away_spread_row
        )
    )

    home_spread, home_spread_source = (
        choose(
            home_spread_row
        )
    )

    away_money, away_money_source = (
        choose(
            away_money_row
        )
    )

    home_money, home_money_source = (
        choose(
            home_money_row
        )
    )

    away_total, away_total_source = (
        choose(
            away_total_row
        )
    )

    home_total, home_total_source = (
        choose(
            home_total_row
        )
    )

    missing = []

    for name, value in [
        (
            "away_spread",
            away_spread,
        ),
        (
            "home_spread",
            home_spread,
        ),
        (
            "away_money",
            away_money,
        ),
        (
            "home_money",
            home_money,
        ),
        (
            "away_total",
            away_total,
        ),
        (
            "home_total",
            home_total,
        ),
    ]:
        if not value:
            missing.append(
                name
            )

    if missing:
        raise RuntimeError(
            f"{away_team} @ {home_team}: "
            f"missing "
            + ",".join(missing)
        )

    away_spread_line = clean(
        away_spread["line"]
    )

    home_spread_line = clean(
        home_spread["line"]
    )

    away_spread_number = (
        spread_number(
            away_spread_line
        )
    )

    home_spread_number = (
        spread_number(
            home_spread_line
        )
    )

    if (
        away_spread_number
        is not None
        and home_spread_number
        is not None
        and abs(
            away_spread_number
            + home_spread_number
        ) > 0.001
    ):
        raise RuntimeError(
            f"{away_team} @ {home_team}: "
            f"spread mismatch "
            f"{away_spread_line} / "
            f"{home_spread_line}"
        )

    total_sides = {}

    for item in [
        away_total,
        home_total,
    ]:
        line = clean(
            item["line"]
        )

        match = re.fullmatch(
            r"([ou])(\d+(?:\.5)?)",
            line,
            re.I,
        )

        if not match:
            raise RuntimeError(
                f"{away_team} @ {home_team}: "
                f"invalid total "
                f"{line}"
            )

        side = (
            match.group(1)
            .lower()
        )

        number = (
            match.group(2)
        )

        total_sides[
            side
        ] = {
            "number":
                number,
            "price":
                clean(
                    item["price"]
                ),
        }

    if "o" not in total_sides:
        raise RuntimeError(
            f"{away_team} @ {home_team}: "
            "missing over"
        )

    if "u" not in total_sides:
        raise RuntimeError(
            f"{away_team} @ {home_team}: "
            "missing under"
        )

    if (
        total_sides["o"]["number"]
        !=
        total_sides["u"]["number"]
    ):
        raise RuntimeError(
            f"{away_team} @ {home_team}: "
            f"total mismatch "
            f"{total_sides['o']['number']} / "
            f"{total_sides['u']['number']}"
        )

    total_value = (
        total_sides["o"]["number"]
    )

    over_price = (
        total_sides["o"]["price"]
    )

    under_price = (
        total_sides["u"]["price"]
    )

    row = {
        "sport":
            "Basketball",
        "league":
            "WNBA",
        "game_date":
            game_date,
        "game_id":
            "",
        "odds_last_update":
            "",
        "game_time":
            "",
        "home_team":
            home_team,
        "away_team":
            away_team,
        "home_spread":
            home_spread_line,
        "away_spread":
            away_spread_line,
        "total":
            total_value,
        "home_dk_moneyline_american":
            clean(
                home_money["price"]
            ),
        "away_dk_moneyline_american":
            clean(
                away_money["price"]
            ),
        "home_dk_spread_american":
            clean(
                home_spread["price"]
            ),
        "away_dk_spread_american":
            clean(
                away_spread["price"]
            ),
        "dk_total_over_american":
            over_price,
        "dk_total_under_american":
            under_price,
        "home_dk_moneyline_decimal":
            decimal_american(
                home_money["price"]
            ),
        "away_dk_moneyline_decimal":
            decimal_american(
                away_money["price"]
            ),
        "home_dk_spread_decimal":
            decimal_american(
                home_spread["price"]
            ),
        "away_dk_spread_decimal":
            decimal_american(
                away_spread["price"]
            ),
        "dk_total_over_decimal":
            decimal_american(
                over_price
            ),
        "dk_total_under_decimal":
            decimal_american(
                under_price
            ),
    }

    source_text = (
        f"S={away_spread_source}/"
        f"{home_spread_source} "
        f"M={away_money_source}/"
        f"{home_money_source} "
        f"T={away_total_source}/"
        f"{home_total_source}"
    )

    return (
        row,
        source_text,
    )


def find_failure_file():
    for path in FAIL_SOURCES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No 2020 failed-date CSV found"
    )


def load_failed_dates():
    path = find_failure_file()

    dates = set()

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            game_date = clean(
                row.get(
                    "game_date"
                )
            )

            if game_date:
                dates.add(
                    game_date
                )

    print(
        f"FAIL SOURCE: {path}"
    )

    return sorted(
        dates
    )


def main():
    failed_dates = (
        load_failed_dates()
    )

    print(
        f"DATES TO RETRY: "
        f"{len(failed_dates)}"
    )

    recovered_dates = 0
    recovered_games = 0
    still_failed = []

    with sync_playwright() as p:
        browser = (
            p.chromium.launch(
                headless=True,
                args=[
                    "--window-size="
                    "2400,1400"
                ],
            )
        )

        page = browser.new_page(
            viewport={
                "width": 2400,
                "height": 1400,
            }
        )

        for game_date in failed_dates:
            url_date = (
                game_date.replace(
                    "_",
                    "",
                )
            )

            outfile = (
                BASE
                / (
                    f"{game_date}"
                    "_WNBA_odds.csv"
                )
            )

            print()
            print(
                f"=== {game_date} ==="
            )

            try:
                page.goto(
                    (
                        "https://www."
                        "oddstrader.com/"
                        "wnba/"
                        f"?date={url_date}"
                        "&g=game&m=merged"
                    ),
                    wait_until=(
                        "domcontentloaded"
                    ),
                    timeout=60000,
                )

                page.wait_for_timeout(
                    5000
                )

                select_market(
                    page,
                    "Spread",
                )

                spread_rows = (
                    scrape_rows(
                        page,
                        "spread",
                    )
                )

                select_market(
                    page,
                    "Money",
                )

                money_rows = (
                    scrape_rows(
                        page,
                        "money",
                    )
                )

                select_market(
                    page,
                    "Total",
                )

                total_rows = (
                    scrape_rows(
                        page,
                        "total",
                    )
                )

                assert_same_order(
                    spread_rows,
                    money_rows,
                    total_rows,
                )

                games = build_pairs(
                    spread_rows
                )

                print(
                    f"RENDERED GAMES: "
                    f"{len(games)}"
                )

                rows = []

                for game_index, game in enumerate(
                    games
                ):
                    away_team = (
                        game[
                            "away_team"
                        ]
                    )

                    home_team = (
                        game[
                            "home_team"
                        ]
                    )

                    index = (
                        game_index * 2
                    )

                    row, sources = (
                        make_game_row(
                            game_date,
                            away_team,
                            home_team,
                            spread_rows[
                                index
                            ],
                            spread_rows[
                                index + 1
                            ],
                            money_rows[
                                index
                            ],
                            money_rows[
                                index + 1
                            ],
                            total_rows[
                                index
                            ],
                            total_rows[
                                index + 1
                            ],
                        )
                    )

                    rows.append(
                        row
                    )

                    print(
                        f"{away_team} @ "
                        f"{home_team} | "
                        f"{sources}"
                    )

                if not rows:
                    raise RuntimeError(
                        "Zero complete games"
                    )

                with outfile.open(
                    "w",
                    newline="",
                    encoding="utf-8-sig",
                ) as handle:
                    writer = (
                        csv.DictWriter(
                            handle,
                            fieldnames=FIELDS,
                        )
                    )

                    writer.writeheader()
                    writer.writerows(
                        rows
                    )

                recovered_dates += 1
                recovered_games += (
                    len(rows)
                )

                print(
                    f"WROTE: {outfile}"
                )

            except Exception as exc:
                reason = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                still_failed.append(
                    {
                        "game_date":
                            game_date,
                        "reason":
                            reason,
                    }
                )

                print(
                    f"FAILED: {reason}"
                )

        browser.close()

    with FINAL_FAILFILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game_date",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(
            still_failed
        )

    print()
    print("==============================")
    print("RENDERED-TABLE RETRY FINISHED")
    print("==============================")
    print(
        f"RECOVERED DATES:    "
        f"{recovered_dates}"
    )
    print(
        f"RECOVERED GAMES:    "
        f"{recovered_games}"
    )
    print(
        f"STILL FAILED DATES: "
        f"{len(still_failed)}"
    )
    print(
        f"FAIL LOG: "
        f"{FINAL_FAILFILE}"
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
WNBA 2023/2024 historical odds backfill from SportsDataverse's archived ESPN game JSON.

IMPORTANT:
- Does NOT call ESPN.
- Uses local SDV games.parquet files for the canonical game list.
- Uses each row's existing game_json_url, which points to the public
  sportsdataverse/wehoop-wnba-raw GitHub archive.
- Does NOT modify Step 18 files or markets.yaml.

Default inputs:
  docs/win/basketball/00_intake/sdv/history/wnba/2023/games.parquet
  docs/win/basketball/00_intake/sdv/history/wnba/2024/games.parquet

Default outputs:
  docs/win/basketball/00_intake/sportsbook_history/espn/wnba/
    2023_WNBA_espn_archived_draftkings.csv
    2023_WNBA_espn_archived_all_providers.csv
    2024_WNBA_espn_archived_draftkings.csv
    2024_WNBA_espn_archived_all_providers.csv
    ESPN_WNBA_2023_2024_COVERAGE.json
    ESPN_WNBA_2023_2024_COVERAGE.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit("This script requires pandas + parquet support.") from exc

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception as exc:
    raise SystemExit("This script requires requests.") from exc


BASE = Path("docs/win/basketball")
SDV_HISTORY_ROOT = BASE / "00_intake/sdv/history/wnba"
DEFAULT_OUTPUT_ROOT = BASE / "00_intake/sportsbook_history/espn/wnba"

USER_AGENT = (
    "stats_for_mat historical-validation/1.0 "
    "(public SportsDataverse GitHub archive reader)"
)

ALL_PROVIDER_FIELDS = [
    "sport","league","season","season_type","game_date","game_id",
    "game_time","game_datetime","home_team","away_team",
    "home_abbreviation","away_abbreviation","sportsbook_provider",
    "provider_id","provider_priority","details","home_spread",
    "away_spread","total","home_moneyline_american",
    "away_moneyline_american","home_spread_american",
    "away_spread_american","total_over_american",
    "total_under_american","home_moneyline_decimal",
    "away_moneyline_decimal","home_spread_decimal",
    "away_spread_decimal","total_over_decimal",
    "total_under_decimal","has_moneyline","has_spread","has_total",
    "has_all_three_markets","archive_url","archive_source",
    "snapshot_phase","fetched_at_utc",
]

DK_FIELDS = [
    "sport","league","game_date","game_id","odds_last_update",
    "sportsbook_provider","scraped_at_utc","provider_updated_at_utc",
    "game_time","home_team","away_team","home_spread","away_spread",
    "total","home_dk_moneyline_american","away_dk_moneyline_american",
    "home_dk_spread_american","away_dk_spread_american",
    "dk_total_over_american","dk_total_under_american",
    "home_dk_moneyline_decimal","away_dk_moneyline_decimal",
    "home_dk_spread_decimal","away_dk_spread_decimal",
    "dk_total_over_decimal","dk_total_under_decimal",
    "archive_url","snapshot_phase",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def clean_id(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    rounded = round(number)
    if abs(number - rounded) > 1e-9:
        return None
    return int(rounded)


def american_to_decimal(value: Any) -> float | None:
    odds = to_float(value)
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return round(1.0 + odds / 100.0, 4)
    return round(1.0 + 100.0 / abs(odds), 4)


def normalize_archive_url(url: str) -> str:
    url = clean(url)
    if not url:
        return ""

    if url.startswith("https://raw.githubusercontent.com/"):
        return url

    marker = "https://github.com/"
    if url.startswith(marker) and "/blob/" in url:
        tail = url[len(marker):]
        repo_path, branch_path = tail.split("/blob/", 1)
        return f"https://raw.githubusercontent.com/{repo_path}/{branch_path}"

    if url.startswith(marker) and "/raw/" in url:
        tail = url[len(marker):]
        repo_path, branch_path = tail.split("/raw/", 1)
        return f"https://raw.githubusercontent.com/{repo_path}/{branch_path}"

    return url


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def fetch_json(
    session: requests.Session,
    url: str,
    timeout: float,
) -> tuple[dict[str, Any] | None, str]:
    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"

    try:
        payload = response.json()
    except ValueError as exc:
        return None, f"invalid_json: {exc}"

    if not isinstance(payload, dict):
        return None, "json_root_not_object"

    return payload, ""


def get_nested_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def provider_name(item: dict[str, Any]) -> str:
    return clean(get_nested_dict(item, "provider").get("name"))


def provider_id(item: dict[str, Any]) -> str:
    return clean_id(get_nested_dict(item, "provider").get("id"))


def provider_priority(item: dict[str, Any]) -> int | None:
    return to_int(get_nested_dict(item, "provider").get("priority"))


def is_draftkings(name: str) -> bool:
    return clean(name).lower() == "draftkings"


def get_pickcenter(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("pickcenter")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def local_game_time(row: dict[str, Any]) -> tuple[str, str]:
    dt = row.get("game_date_time")
    if dt is None:
        return clean(row.get("game_date")).replace("-", "_"), ""

    try:
        ts = pd.Timestamp(dt)
        return ts.strftime("%Y_%m_%d"), ts.strftime("%I:%M %p")
    except Exception:
        return clean(row.get("game_date")).replace("-", "_"), ""


def infer_spreads(
    item: dict[str, Any],
    home_abbrev: str,
    away_abbrev: str,
) -> tuple[float | None, float | None]:
    raw_spread = to_float(item.get("spread"))
    details = clean(item.get("details"))

    home_odds = get_nested_dict(item, "homeTeamOdds")
    away_odds = get_nested_dict(item, "awayTeamOdds")

    home_favorite = home_odds.get("favorite")
    away_favorite = away_odds.get("favorite")

    if raw_spread is not None:
        magnitude = abs(raw_spread)

        if home_favorite is True and away_favorite is not True:
            return -magnitude, magnitude

        if away_favorite is True and home_favorite is not True:
            return magnitude, -magnitude

        if magnitude == 0:
            return 0.0, 0.0

    parts = details.split()
    if len(parts) >= 2:
        token_team = parts[0].strip().upper()
        token_line = to_float(parts[-1])

        if token_line is not None:
            magnitude = abs(token_line)

            if token_team == clean(home_abbrev).upper():
                return -magnitude, magnitude

            if token_team == clean(away_abbrev).upper():
                return magnitude, -magnitude

    if raw_spread is not None:
        return raw_spread, -raw_spread

    return None, None


def normalize_pickcenter_item(
    *,
    season: int,
    game: dict[str, Any],
    item: dict[str, Any],
    archive_url: str,
    fetched_at_utc: str,
) -> dict[str, Any]:
    home = get_nested_dict(item, "homeTeamOdds")
    away = get_nested_dict(item, "awayTeamOdds")

    home_ml = to_int(home.get("moneyLine"))
    away_ml = to_int(away.get("moneyLine"))
    home_spread_price = to_int(home.get("spreadOdds"))
    away_spread_price = to_int(away.get("spreadOdds"))
    over_price = to_int(item.get("overOdds"))
    under_price = to_int(item.get("underOdds"))
    total = to_float(item.get("overUnder"))

    home_spread, away_spread = infer_spreads(
        item,
        clean(game.get("home_abbreviation")),
        clean(game.get("away_abbreviation")),
    )

    has_moneyline = home_ml is not None and away_ml is not None
    has_spread = (
        home_spread is not None
        and away_spread is not None
        and home_spread_price is not None
        and away_spread_price is not None
    )
    has_total = (
        total is not None
        and over_price is not None
        and under_price is not None
    )

    game_date, game_time = local_game_time(game)

    return {
        "sport": "Basketball",
        "league": "WNBA",
        "season": season,
        "season_type": to_int(game.get("season_type")) or "",
        "game_date": game_date,
        "game_id": clean_id(game.get("game_id")),
        "game_time": game_time,
        "game_datetime": clean(game.get("game_date_time")),
        "home_team": clean(game.get("home_display_name")),
        "away_team": clean(game.get("away_display_name")),
        "home_abbreviation": clean(game.get("home_abbreviation")),
        "away_abbreviation": clean(game.get("away_abbreviation")),
        "sportsbook_provider": provider_name(item),
        "provider_id": provider_id(item),
        "provider_priority": provider_priority(item) or "",
        "details": clean(item.get("details")),
        "home_spread": home_spread if home_spread is not None else "",
        "away_spread": away_spread if away_spread is not None else "",
        "total": total if total is not None else "",
        "home_moneyline_american": home_ml if home_ml is not None else "",
        "away_moneyline_american": away_ml if away_ml is not None else "",
        "home_spread_american": (
            home_spread_price if home_spread_price is not None else ""
        ),
        "away_spread_american": (
            away_spread_price if away_spread_price is not None else ""
        ),
        "total_over_american": over_price if over_price is not None else "",
        "total_under_american": under_price if under_price is not None else "",
        "home_moneyline_decimal": american_to_decimal(home_ml) or "",
        "away_moneyline_decimal": american_to_decimal(away_ml) or "",
        "home_spread_decimal": american_to_decimal(home_spread_price) or "",
        "away_spread_decimal": american_to_decimal(away_spread_price) or "",
        "total_over_decimal": american_to_decimal(over_price) or "",
        "total_under_decimal": american_to_decimal(under_price) or "",
        "has_moneyline": has_moneyline,
        "has_spread": has_spread,
        "has_total": has_total,
        "has_all_three_markets": (
            has_moneyline and has_spread and has_total
        ),
        "archive_url": archive_url,
        "archive_source": "sportsdataverse/wehoop-wnba-raw",
        "snapshot_phase": "archived_game_summary",
        "fetched_at_utc": fetched_at_utc,
    }


def dk_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sport": row["sport"],
        "league": row["league"],
        "game_date": row["game_date"],
        "game_id": row["game_id"],
        "odds_last_update": row["fetched_at_utc"],
        "sportsbook_provider": "DraftKings",
        "scraped_at_utc": row["fetched_at_utc"],
        "provider_updated_at_utc": "",
        "game_time": row["game_time"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_spread": row["home_spread"],
        "away_spread": row["away_spread"],
        "total": row["total"],
        "home_dk_moneyline_american": row["home_moneyline_american"],
        "away_dk_moneyline_american": row["away_moneyline_american"],
        "home_dk_spread_american": row["home_spread_american"],
        "away_dk_spread_american": row["away_spread_american"],
        "dk_total_over_american": row["total_over_american"],
        "dk_total_under_american": row["total_under_american"],
        "home_dk_moneyline_decimal": row["home_moneyline_decimal"],
        "away_dk_moneyline_decimal": row["away_moneyline_decimal"],
        "home_dk_spread_decimal": row["home_spread_decimal"],
        "away_dk_spread_decimal": row["away_spread_decimal"],
        "dk_total_over_decimal": row["total_over_decimal"],
        "dk_total_under_decimal": row["total_under_decimal"],
        "archive_url": row["archive_url"],
        "snapshot_phase": row["snapshot_phase"],
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")

    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def metric_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    games: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        gid = clean_id(row.get("game_id"))
        if gid:
            games.setdefault(gid, []).append(row)

    out = {
        "games_with_any_row": len(games),
        "games_with_moneyline": 0,
        "games_with_spread": 0,
        "games_with_total": 0,
        "games_with_all_three": 0,
    }

    for group in games.values():
        out["games_with_moneyline"] += int(
            any(row.get("has_moneyline") is True for row in group)
        )
        out["games_with_spread"] += int(
            any(row.get("has_spread") is True for row in group)
        )
        out["games_with_total"] += int(
            any(row.get("has_total") is True for row in group)
        )
        out["games_with_all_three"] += int(
            any(row.get("has_all_three_markets") is True for row in group)
        )

    return out


def season_report(
    season: int,
    completed_games: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    dk_rows: list[dict[str, Any]],
    fetch_success: int,
    pickcenter_games: int,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    denominator = len(completed_games)

    any_counts = metric_counts(all_rows)
    dk_counts = metric_counts(dk_rows)

    provider_games: dict[str, set[str]] = defaultdict(set)
    provider_all_three: dict[str, set[str]] = defaultdict(set)

    for row in all_rows:
        provider = clean(row.get("sportsbook_provider")) or "UNKNOWN"
        gid = clean_id(row.get("game_id"))
        if not gid:
            continue
        provider_games[provider].add(gid)
        if row.get("has_all_three_markets") is True:
            provider_all_three[provider].add(gid)

    provider_summary = {
        provider: {
            "games": len(provider_games[provider]),
            "all_three_games": len(provider_all_three[provider]),
        }
        for provider in sorted(provider_games)
    }

    return {
        "season": season,
        "completed_games_in_local_sdv": denominator,
        "archive_fetch_success": fetch_success,
        "archive_fetch_success_pct": pct(fetch_success, denominator),
        "games_with_pickcenter": pickcenter_games,
        "pickcenter_coverage_pct": pct(pickcenter_games, denominator),
        "draftkings": {
            **dk_counts,
            "moneyline_coverage_pct": pct(
                dk_counts["games_with_moneyline"], denominator
            ),
            "spread_coverage_pct": pct(
                dk_counts["games_with_spread"], denominator
            ),
            "total_coverage_pct": pct(
                dk_counts["games_with_total"], denominator
            ),
            "all_three_coverage_pct": pct(
                dk_counts["games_with_all_three"], denominator
            ),
        },
        "any_provider": {
            **any_counts,
            "moneyline_coverage_pct": pct(
                any_counts["games_with_moneyline"], denominator
            ),
            "spread_coverage_pct": pct(
                any_counts["games_with_spread"], denominator
            ),
            "total_coverage_pct": pct(
                any_counts["games_with_total"], denominator
            ),
            "all_three_coverage_pct": pct(
                any_counts["games_with_all_three"], denominator
            ),
        },
        "provider_summary": provider_summary,
        "fetch_failures": failures,
    }


def text_report(payload: dict[str, Any]) -> str:
    lines = [
        "WNBA 2023-2024 ARCHIVED ESPN ODDS COVERAGE",
        "=========================================",
        "",
        f"generated_at_utc: {payload['generated_at_utc']}",
        "source: sportsdataverse/wehoop-wnba-raw public GitHub archive",
        "direct_espn_requests: NO",
        "step18_touched: NO",
        "markets_yaml_touched: NO",
        "",
    ]

    for season in payload["seasons"]:
        d = season["completed_games_in_local_sdv"]
        dk = season["draftkings"]
        anyp = season["any_provider"]

        lines.extend(
            [
                f"SEASON {season['season']}",
                "-" * 40,
                f"completed SDV games: {d}",
                (
                    "archive fetch success: "
                    f"{season['archive_fetch_success']}/{d} "
                    f"({season['archive_fetch_success_pct']}%)"
                ),
                (
                    "pickcenter present: "
                    f"{season['games_with_pickcenter']}/{d} "
                    f"({season['pickcenter_coverage_pct']}%)"
                ),
                "",
                "DraftKings:",
                (
                    f"  moneyline: {dk['games_with_moneyline']}/{d} "
                    f"({dk['moneyline_coverage_pct']}%)"
                ),
                (
                    f"  spread:    {dk['games_with_spread']}/{d} "
                    f"({dk['spread_coverage_pct']}%)"
                ),
                (
                    f"  total:     {dk['games_with_total']}/{d} "
                    f"({dk['total_coverage_pct']}%)"
                ),
                (
                    f"  all three: {dk['games_with_all_three']}/{d} "
                    f"({dk['all_three_coverage_pct']}%)"
                ),
                "",
                "Any provider:",
                (
                    f"  moneyline: {anyp['games_with_moneyline']}/{d} "
                    f"({anyp['moneyline_coverage_pct']}%)"
                ),
                (
                    f"  spread:    {anyp['games_with_spread']}/{d} "
                    f"({anyp['spread_coverage_pct']}%)"
                ),
                (
                    f"  total:     {anyp['games_with_total']}/{d} "
                    f"({anyp['total_coverage_pct']}%)"
                ),
                (
                    f"  all three: {anyp['games_with_all_three']}/{d} "
                    f"({anyp['all_three_coverage_pct']}%)"
                ),
                "",
                f"providers: {season['provider_summary']}",
                f"fetch failures: {len(season['fetch_failures'])}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_seasons(value: str) -> tuple[int, ...]:
    seasons = sorted(
        {
            int(token.strip())
            for token in value.split(",")
            if token.strip()
        }
    )
    if not seasons:
        raise ValueError("No seasons supplied")
    return tuple(seasons)


def load_games(season: int) -> list[dict[str, Any]]:
    path = SDV_HISTORY_ROOT / str(season) / "games.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    wanted = [
        "game_id",
        "game_date",
        "game_date_time",
        "season",
        "season_type",
        "status_type_completed",
        "home_display_name",
        "away_display_name",
        "home_abbreviation",
        "away_abbreviation",
        "game_json",
        "game_json_url",
    ]

    frame = pd.read_parquet(path)
    missing = [column for column in wanted if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path}: missing required columns {missing}")

    frame = frame[wanted].copy()
    frame = frame[
        frame["status_type_completed"].fillna(False).astype(bool)
    ].copy()
    frame["game_id"] = frame["game_id"].map(clean_id)
    frame = frame[frame["game_id"] != ""].copy()
    frame = frame.drop_duplicates(subset=["game_id"], keep="last")
    frame = frame.sort_values(["game_date", "game_id"])

    return frame.to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2023,2024")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--cache-raw", action="store_true")
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    session = build_session()

    overall = {
        "schema_version": 2,
        "generated_at_utc": utc_now(),
        "source": "sportsdataverse/wehoop-wnba-raw public GitHub archive",
        "direct_espn_requests": False,
        "step18_touched": False,
        "markets_yaml_touched": False,
        "seasons": [],
    }

    print("=== WNBA ARCHIVED ESPN ODDS BACKFILL ===", flush=True)
    print(f"seasons={seasons}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print("Direct ESPN requests: NO", flush=True)
    print("Step 18 files modified: NO", flush=True)

    for season in seasons:
        games = load_games(season)
        print(
            f"\n[{season}] completed local SDV games={len(games)}",
            flush=True,
        )

        all_rows: list[dict[str, Any]] = []
        dk_rows: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        fetch_success = 0
        pickcenter_games = 0

        for index, game in enumerate(games, start=1):
            gid = clean_id(game.get("game_id"))
            url = normalize_archive_url(clean(game.get("game_json_url")))

            if not url:
                failures.append(
                    {"game_id": gid, "reason": "missing_game_json_url"}
                )
                continue

            payload = None
            error = ""

            cache_path = (
                output_root / "raw" / str(season) / f"{gid}.json"
            )

            if args.cache_raw and cache_path.exists():
                try:
                    payload = json.loads(
                        cache_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(payload, dict):
                        payload = None
                        error = "cached_json_root_not_object"
                except Exception as exc:
                    payload = None
                    error = f"cached_json_error: {exc}"

            if payload is None:
                payload, error = fetch_json(
                    session,
                    url,
                    args.timeout,
                )

                if payload is not None and args.cache_raw:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(payload, separators=(",", ":")),
                        encoding="utf-8",
                    )

            if payload is None:
                failures.append(
                    {
                        "game_id": gid,
                        "reason": error or "unknown_fetch_failure",
                        "url": url,
                    }
                )
                continue

            fetch_success += 1
            pickcenter = get_pickcenter(payload)

            if pickcenter:
                pickcenter_games += 1

            fetched_at = utc_now()

            game_provider_rows = [
                normalize_pickcenter_item(
                    season=season,
                    game=game,
                    item=item,
                    archive_url=url,
                    fetched_at_utc=fetched_at,
                )
                for item in pickcenter
            ]

            all_rows.extend(game_provider_rows)

            dk_candidates = [
                row
                for row in game_provider_rows
                if is_draftkings(
                    clean(row.get("sportsbook_provider"))
                )
            ]

            if dk_candidates:
                dk_candidates.sort(
                    key=lambda row: (
                        int(row.get("has_all_three_markets") is True),
                        int(row.get("has_spread") is True),
                        int(row.get("has_total") is True),
                        int(row.get("has_moneyline") is True),
                    ),
                    reverse=True,
                )
                dk_rows.append(dk_candidates[0])

            if index == 1 or index % 25 == 0 or index == len(games):
                print(
                    f"[{season}] progress {index}/{len(games)} | "
                    f"archive_ok={fetch_success} "
                    f"pickcenter={pickcenter_games} "
                    f"dk_games={len(dk_rows)}",
                    flush=True,
                )

            if args.request_delay > 0:
                time.sleep(args.request_delay)

        all_rows.sort(
            key=lambda row: (
                clean(row.get("game_date")),
                clean_id(row.get("game_id")),
                clean(row.get("sportsbook_provider")),
            )
        )
        dk_rows.sort(
            key=lambda row: (
                clean(row.get("game_date")),
                clean_id(row.get("game_id")),
            )
        )

        write_csv(
            output_root
            / f"{season}_WNBA_espn_archived_all_providers.csv",
            all_rows,
            ALL_PROVIDER_FIELDS,
        )
        write_csv(
            output_root
            / f"{season}_WNBA_espn_archived_draftkings.csv",
            [dk_output_row(row) for row in dk_rows],
            DK_FIELDS,
        )

        report = season_report(
            season,
            games,
            all_rows,
            dk_rows,
            fetch_success,
            pickcenter_games,
            failures,
        )
        overall["seasons"].append(report)

        print(
            f"[{season}] DraftKings all-three coverage: "
            f"{report['draftkings']['games_with_all_three']}/"
            f"{report['completed_games_in_local_sdv']} "
            f"({report['draftkings']['all_three_coverage_pct']}%)",
            flush=True,
        )
        print(
            f"[{season}] Any-provider all-three coverage: "
            f"{report['any_provider']['games_with_all_three']}/"
            f"{report['completed_games_in_local_sdv']} "
            f"({report['any_provider']['all_three_coverage_pct']}%)",
            flush=True,
        )

    json_path = output_root / "ESPN_WNBA_2023_2024_COVERAGE.json"
    txt_path = output_root / "ESPN_WNBA_2023_2024_COVERAGE.txt"

    write_json(json_path, overall)
    txt_path.write_text(
        text_report(overall),
        encoding="utf-8",
    )

    print("\n=== COVERAGE REPORT ===", flush=True)
    print(txt_path.read_text(encoding="utf-8"), end="", flush=True)
    print(f"coverage_json={json_path}", flush=True)
    print(f"coverage_txt={txt_path}", flush=True)
    print("Direct ESPN requests: NO", flush=True)
    print("Step 18 files modified: NO", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

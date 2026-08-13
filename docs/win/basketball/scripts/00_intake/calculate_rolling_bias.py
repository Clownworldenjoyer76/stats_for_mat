#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/calculate_rolling_bias.py
#
# Calculates current league bias values from completed games.
#
# Permanent bias rules:
#   docs/win/basketball/config/model_config.yaml
#
# Historical completed games:
#   docs/win/basketball/00_intake/final_combined_files/combined/{season}_{LEAGUE}.csv
#
# Current-season RAW predictions:
#   docs/win/basketball/00_intake/predictions/{league}/*_{LEAGUE}_predictions.csv
#
# Current-season final scores:
#   docs/win/basketball/05_final_scores/results/{league}/*_final_scores_{LEAGUE}.csv
#
# Generated state:
#   docs/win/basketball/config/rolling_bias_state.yaml
#
# IMPORTANT:
# - Current-season prediction files are raw/pre-bias and are NEVER adjusted here.
# - Historical combined rows with bias_applied=1 are converted back to raw using
#   the legacy bias values that created those historical adjusted projections.
# - Current finals already represented in historical combined files are reported
#   as historical coverage, not as unmatched current-season finals.
# - Current predictions/finals match by game_id first, then by the complete
#   game_date + home_team + away_team composite key.
# - Rolling windows cross season boundaries automatically.
# - A rolling value is not produced unless the full configured window exists.

from __future__ import annotations

import csv
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
_EXPECTED_REPO_ROOT = SCRIPT_PATH.parents[5] if len(SCRIPT_PATH.parents) > 5 else Path.cwd()
_CWD_REPO_ROOT = Path.cwd().resolve()

if (_EXPECTED_REPO_ROOT / "docs/win/basketball/config/model_config.yaml").exists():
    REPO_ROOT = _EXPECTED_REPO_ROOT
elif (_CWD_REPO_ROOT / "docs/win/basketball/config/model_config.yaml").exists():
    REPO_ROOT = _CWD_REPO_ROOT
else:
    REPO_ROOT = _EXPECTED_REPO_ROOT

CONFIG_PATH = REPO_ROOT / "docs/win/basketball/config/model_config.yaml"
STATE_PATH = REPO_ROOT / "docs/win/basketball/config/rolling_bias_state.yaml"
HISTORICAL_DIR = REPO_ROOT / "docs/win/basketball/00_intake/final_combined_files/combined"
RAW_PREDICTIONS_ROOT = REPO_ROOT / "docs/win/basketball/00_intake/predictions"
FINAL_SCORES_ROOT = REPO_ROOT / "docs/win/basketball/05_final_scores/results"
ERROR_DIR = REPO_ROOT / "docs/win/basketball/errors/00_intake"
LOG_PATH = ERROR_DIR / "calculate_rolling_bias.txt"

SUPPORTED_LEAGUES = ("nba", "ncaam", "wnba")


# ============================================================================
# LEGACY HISTORICAL BIASES
# ============================================================================

# These are the biases that were applied when the historical combined-file
# projections were produced by clean_basketball_inputs.py.
#
# Historical reversal is ONLY performed when the row explicitly has
# bias_applied=1. Current raw prediction files never use this mapping.
HISTORICAL_APPLIED_BIAS = {
    ("nba", 2025): {"margin": 0.4, "total": 0.4},
    ("ncaam", 2025): {"margin": 0.6, "total": 1.2},
    ("wnba", 2025): {"margin": 0.5, "total": 0.0},
}


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

PREDICTION_REQUIRED = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_projected_points",
    "away_projected_points",
}

FINAL_REQUIRED = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}

HISTORICAL_REQUIRED = {
    "game_date",
    "home_team",
    "away_team",
    "home_projected_points",
    "away_projected_points",
    "home_score",
    "away_score",
}


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(frozen=True)
class CompletedGame:
    league: str
    game_id: str
    game_date: str
    game_time: str
    home_team: str
    away_team: str
    home_projected_points: float
    away_projected_points: float
    total_projected_points: float
    home_score: float
    away_score: float
    source: str
    source_priority: int

    @property
    def projected_margin(self) -> float:
        return self.home_projected_points - self.away_projected_points

    @property
    def actual_margin(self) -> float:
        return self.home_score - self.away_score

    @property
    def margin_error(self) -> float:
        return self.projected_margin - self.actual_margin

    @property
    def actual_total(self) -> float:
        return self.home_score + self.away_score

    @property
    def total_error(self) -> float:
        return self.total_projected_points - self.actual_total

    @property
    def match_key(self) -> str:
        return composite_key(self.game_date, self.home_team, self.away_team)

    @property
    def sort_key(self) -> tuple:
        return (
            parse_game_datetime(self.game_date, self.game_time),
            normalize_text(self.home_team),
            normalize_text(self.away_team),
            canonical_game_id(self.game_id),
        )


# ============================================================================
# LOGGING
# ============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_log() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"=== calculate_rolling_bias RUN {utc_now_iso()} ===\n")
        f.write(f"REPO_ROOT={REPO_ROOT}\n")
        f.write(f"CONFIG_PATH={CONFIG_PATH}\n")
        f.write(f"STATE_PATH={STATE_PATH}\n")


def log(message: str, level: str = "INFO") -> None:
    line = f"{utc_now_iso()} | {level:<5} | {message}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def canonical_game_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_date(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""

    text = text.replace("/", "-").replace("_", "-")

    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m-%d-%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return text


def composite_key(game_date: Any, home_team: Any, away_team: Any) -> str:
    date = normalize_date(game_date)
    home = normalize_text(home_team)
    away = normalize_text(away_team)

    if not date or not home or not away:
        return ""

    return f"{date}|{home}|{away}"


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def flag_is_applied(value: Any) -> bool:
    text = normalize_text(value)
    return text in {"1", "1.0", "true", "yes", "y"}


def parse_game_datetime(game_date: Any, game_time: Any = "") -> datetime:
    date_text = normalize_date(game_date)
    try:
        base = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return datetime.min

    time_text = "" if game_time is None else str(game_time).strip()
    if not time_text:
        return base

    cleaned = re.sub(r"\s+", " ", time_text.upper())
    for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(cleaned, fmt).time()
            return datetime.combine(base.date(), parsed_time)
        except ValueError:
            pass

    return base


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def require_columns(path: Path, fieldnames: Iterable[str], required: set[str]) -> None:
    present = set(fieldnames)
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def league_upper(league: str) -> str:
    return league.strip().upper()


def prediction_signature(row: dict[str, str]) -> tuple:
    return (
        normalize_date(row.get("game_date")),
        normalize_text(row.get("game_time")),
        normalize_text(row.get("home_team")),
        normalize_text(row.get("away_team")),
        to_float(row.get("home_projected_points")),
        to_float(row.get("away_projected_points")),
        to_float(row.get("total_projected_points")),
    )


# ============================================================================
# CONFIG
# ============================================================================

def load_model_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing model config: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    leagues = cfg.get("leagues")
    if not isinstance(leagues, dict):
        raise ValueError(f"{CONFIG_PATH} must contain a top-level 'leagues' mapping")

    return cfg


def resolve_bias_rule(league_cfg: dict[str, Any], component: str) -> dict[str, Any]:
    bias_cfg = league_cfg.get("bias") or {}
    rule = bias_cfg.get(component) or {}

    if not isinstance(rule, dict):
        return {"method": None, "window_games": None, "value": None}

    method_raw = rule.get("method")
    method = None if method_raw is None else str(method_raw).strip().lower()
    window = rule.get("window_games")
    value = rule.get("value")

    if method and method.startswith("rolling_") and window in (None, ""):
        suffix = method.split("_", 1)[1]
        if suffix.isdigit():
            window = int(suffix)
            method = "rolling"

    if window not in (None, ""):
        try:
            window = int(window)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid window_games={window!r}") from exc

    if value not in (None, ""):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fixed bias value={value!r}") from exc

    return {"method": method, "window_games": window, "value": value}


# ============================================================================
# HISTORICAL COMBINED FILES
# ============================================================================

def historical_files_for_league(league: str) -> list[Path]:
    if not HISTORICAL_DIR.exists():
        return []

    suffix = f"_{league_upper(league)}.csv"
    files = [
        path
        for path in HISTORICAL_DIR.glob("*.csv")
        if path.name.upper().endswith(suffix.upper())
    ]

    def season_key(path: Path) -> tuple[int, str]:
        match = re.match(r"^(\d{4})_", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return sorted(files, key=season_key)


def historical_season_from_path(path: Path) -> int | None:
    match = re.match(r"^(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def reverse_historical_projection_bias(
    league: str,
    historical_season: int | None,
    home_projected_points: float,
    away_projected_points: float,
    total_projected_points: float,
    bias_applied: Any,
) -> tuple[float, float, float, bool]:
    """
    Convert a historical adjusted projection back to the raw projection.

    Historical cleaner formula:
        adjusted_home  = raw_home - margin_bias/2 - total_bias/2
        adjusted_away  = raw_away + margin_bias/2 - total_bias/2
        adjusted_total = raw_total - total_bias

    Therefore:
        raw_home  = adjusted_home + margin_bias/2 + total_bias/2
        raw_away  = adjusted_away - margin_bias/2 + total_bias/2
        raw_total = adjusted_total + total_bias
    """
    if not flag_is_applied(bias_applied):
        return (
            home_projected_points,
            away_projected_points,
            total_projected_points,
            False,
        )

    legacy = HISTORICAL_APPLIED_BIAS.get((league.lower(), historical_season))
    if legacy is None:
        raise ValueError(
            f"Cannot safely reverse bias_applied=1 for {league_upper(league)} "
            f"historical season {historical_season!r}. Add that season's exact applied "
            "bias values or preserve raw/per-game applied-bias fields before using it."
        )

    margin_bias = float(legacy["margin"])
    total_bias = float(legacy["total"])

    raw_home = home_projected_points + (margin_bias / 2.0) + (total_bias / 2.0)
    raw_away = away_projected_points - (margin_bias / 2.0) + (total_bias / 2.0)
    raw_total = total_projected_points + total_bias

    return raw_home, raw_away, raw_total, True


def load_historical_completed_games(
    league: str,
) -> tuple[list[CompletedGame], dict[str, int]]:
    games: list[CompletedGame] = []
    stats = {
        "historical_files_scanned": 0,
        "historical_rows": 0,
        "historical_usable_games": 0,
        "historical_incomplete_rows": 0,
        "historical_rows_bias_reversed": 0,
        "historical_rows_raw_unadjusted": 0,
    }

    for path in historical_files_for_league(league):
        fieldnames, rows = read_csv_rows(path)
        require_columns(path, fieldnames, HISTORICAL_REQUIRED)
        if "bias_applied" not in set(fieldnames):
            raise ValueError(
                f"{path} is missing required historical bias flag column: bias_applied"
            )
        historical_season = historical_season_from_path(path)
        stats["historical_files_scanned"] += 1
        stats["historical_rows"] += len(rows)

        file_used = 0
        file_skipped = 0
        file_reversed = 0
        file_raw = 0

        for row in rows:
            home_proj = to_float(row.get("home_projected_points"))
            away_proj = to_float(row.get("away_projected_points"))
            total_proj = to_float(row.get("total_projected_points"))
            home_score = to_float(row.get("home_score"))
            away_score = to_float(row.get("away_score"))

            if home_proj is None or away_proj is None or home_score is None or away_score is None:
                file_skipped += 1
                stats["historical_incomplete_rows"] += 1
                continue

            if total_proj is None:
                total_proj = home_proj + away_proj

            game_date = normalize_date(row.get("game_date"))
            home_team = str(row.get("home_team", "")).strip()
            away_team = str(row.get("away_team", "")).strip()

            if not game_date or not home_team or not away_team:
                file_skipped += 1
                stats["historical_incomplete_rows"] += 1
                continue

            home_proj, away_proj, total_proj, was_reversed = reverse_historical_projection_bias(
                league=league,
                historical_season=historical_season,
                home_projected_points=home_proj,
                away_projected_points=away_proj,
                total_projected_points=total_proj,
                bias_applied=row.get("bias_applied"),
            )

            if was_reversed:
                file_reversed += 1
                stats["historical_rows_bias_reversed"] += 1
            else:
                file_raw += 1
                stats["historical_rows_raw_unadjusted"] += 1

            games.append(
                CompletedGame(
                    league=league,
                    game_id=canonical_game_id(row.get("game_id")),
                    game_date=game_date,
                    game_time=str(row.get("game_time", "")).strip(),
                    home_team=home_team,
                    away_team=away_team,
                    home_projected_points=home_proj,
                    away_projected_points=away_proj,
                    total_projected_points=total_proj,
                    home_score=home_score,
                    away_score=away_score,
                    source=str(path.relative_to(REPO_ROOT)),
                    source_priority=1,
                )
            )
            file_used += 1
            stats["historical_usable_games"] += 1

        log(
            f"{league_upper(league)} | HISTORICAL | {path.name} | "
            f"rows={len(rows)} usable={file_used} skipped={file_skipped} "
            f"bias_reversed={file_reversed} raw_unadjusted={file_raw}"
        )

    return games, stats


# ============================================================================
# CURRENT-SEASON RAW PREDICTIONS + FINAL SCORES
# ============================================================================

def current_prediction_files(league: str) -> list[Path]:
    folder = RAW_PREDICTIONS_ROOT / league.lower()
    if not folder.exists():
        return []
    pattern = f"*_{league_upper(league)}_predictions.csv"
    return sorted(folder.glob(pattern))


def current_final_files(league: str) -> list[Path]:
    folder = FINAL_SCORES_ROOT / league.lower()
    if not folder.exists():
        return []
    pattern = f"*_final_scores_{league_upper(league)}.csv"
    return sorted(folder.glob(pattern))


def load_raw_prediction_rows(league: str) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    files = current_prediction_files(league)

    for path in files:
        fieldnames, rows = read_csv_rows(path)
        require_columns(path, fieldnames, PREDICTION_REQUIRED)

        for row in rows:
            copy = dict(row)
            copy["_source_file"] = str(path.relative_to(REPO_ROOT))
            rows_out.append(copy)

    log(
        f"{league_upper(league)} | CURRENT RAW PREDICTIONS | "
        f"files={len(files)} rows={len(rows_out)}"
    )
    return rows_out


def load_final_score_rows(league: str) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    files = current_final_files(league)

    for path in files:
        fieldnames, rows = read_csv_rows(path)
        require_columns(path, fieldnames, FINAL_REQUIRED)

        for row in rows:
            copy = dict(row)
            copy["_source_file"] = str(path.relative_to(REPO_ROOT))
            rows_out.append(copy)

    log(
        f"{league_upper(league)} | CURRENT FINAL SCORES | "
        f"files={len(files)} rows={len(rows_out)}"
    )
    return rows_out


def add_prediction_index_entry(
    index: dict[str, dict[str, str]],
    ambiguous: set[str],
    key: str,
    row: dict[str, str],
) -> tuple[bool, bool]:
    """
    Returns (duplicate_seen, conflict_seen).

    Identical duplicate rows keep the first row.
    Conflicting duplicate keys are removed from the usable index and marked
    ambiguous so they can never be matched silently.
    """
    if not key:
        return False, False

    if key in ambiguous:
        return True, True

    existing = index.get(key)
    if existing is None:
        index[key] = row
        return False, False

    if prediction_signature(existing) == prediction_signature(row):
        return True, False

    del index[key]
    ambiguous.add(key)
    return True, True


def build_prediction_indexes(
    prediction_rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    set[str],
    set[str],
    dict[str, int],
]:
    by_id: dict[str, dict[str, str]] = {}
    by_composite: dict[str, dict[str, str]] = {}
    ambiguous_ids: set[str] = set()
    ambiguous_composites: set[str] = set()

    stats = {
        "prediction_rows": len(prediction_rows),
        "prediction_rows_missing_game_id": 0,
        "prediction_rows_missing_composite": 0,
        "duplicate_prediction_game_ids": 0,
        "duplicate_prediction_composites": 0,
        "conflicting_prediction_game_ids": 0,
        "conflicting_prediction_composites": 0,
    }

    for row in prediction_rows:
        gid = canonical_game_id(row.get("game_id"))
        comp = composite_key(row.get("game_date"), row.get("home_team"), row.get("away_team"))

        if gid:
            duplicate, conflict = add_prediction_index_entry(
                by_id, ambiguous_ids, gid, row
            )
            if duplicate:
                stats["duplicate_prediction_game_ids"] += 1
            if conflict:
                stats["conflicting_prediction_game_ids"] += 1
        else:
            stats["prediction_rows_missing_game_id"] += 1

        if comp:
            duplicate, conflict = add_prediction_index_entry(
                by_composite, ambiguous_composites, comp, row
            )
            if duplicate:
                stats["duplicate_prediction_composites"] += 1
            if conflict:
                stats["conflicting_prediction_composites"] += 1
        else:
            stats["prediction_rows_missing_composite"] += 1

    return by_id, by_composite, ambiguous_ids, ambiguous_composites, stats


def historical_coverage_sets(
    historical_games: list[CompletedGame],
) -> tuple[set[str], set[str]]:
    game_ids = {
        canonical_game_id(game.game_id)
        for game in historical_games
        if canonical_game_id(game.game_id)
    }
    composites = {
        game.match_key
        for game in historical_games
        if game.match_key
    }
    return game_ids, composites


def final_is_covered_by_history(
    final: dict[str, str],
    historical_ids: set[str],
    historical_composites: set[str],
) -> bool:
    gid = canonical_game_id(final.get("game_id"))
    comp = composite_key(final.get("game_date"), final.get("home_team"), final.get("away_team"))

    if gid and gid in historical_ids:
        return True
    if comp and comp in historical_composites:
        return True
    return False


def load_current_completed_games(
    league: str,
    historical_games: list[CompletedGame],
) -> tuple[list[CompletedGame], dict[str, int]]:
    predictions = load_raw_prediction_rows(league)
    finals = load_final_score_rows(league)

    (
        pred_by_id,
        pred_by_composite,
        ambiguous_ids,
        ambiguous_composites,
        prediction_stats,
    ) = build_prediction_indexes(predictions)

    historical_ids, historical_composites = historical_coverage_sets(historical_games)

    games: list[CompletedGame] = []
    stats = {
        **prediction_stats,
        "final_rows_scanned": len(finals),
        "finals_already_covered_by_historical": 0,
        "current_finals_to_match": 0,
        "matched_by_game_id": 0,
        "matched_by_composite": 0,
        "ambiguous_prediction_matches": 0,
        "true_unmatched_current_finals": 0,
        "invalid_current_matches": 0,
    }

    for final in finals:
        if final_is_covered_by_history(final, historical_ids, historical_composites):
            stats["finals_already_covered_by_historical"] += 1
            continue

        stats["current_finals_to_match"] += 1

        gid = canonical_game_id(final.get("game_id"))
        comp = composite_key(final.get("game_date"), final.get("home_team"), final.get("away_team"))

        pred = pred_by_id.get(gid) if gid else None
        match_method = "game_id"

        if pred is None and comp:
            pred = pred_by_composite.get(comp)
            match_method = "composite"

        if pred is None:
            is_ambiguous = (
                (bool(gid) and gid in ambiguous_ids)
                or (bool(comp) and comp in ambiguous_composites)
            )

            if is_ambiguous:
                stats["ambiguous_prediction_matches"] += 1
                log(
                    f"{league_upper(league)} | AMBIGUOUS CURRENT FINAL | "
                    f"game_id={gid!r} key={comp!r} source={final.get('_source_file', '')}",
                    "WARN",
                )
            else:
                stats["true_unmatched_current_finals"] += 1
                log(
                    f"{league_upper(league)} | TRUE UNMATCHED CURRENT FINAL | "
                    f"game_id={gid!r} key={comp!r} source={final.get('_source_file', '')}",
                    "WARN",
                )
            continue

        home_proj = to_float(pred.get("home_projected_points"))
        away_proj = to_float(pred.get("away_projected_points"))
        total_proj = to_float(pred.get("total_projected_points"))
        home_score = to_float(final.get("home_score"))
        away_score = to_float(final.get("away_score"))

        if home_proj is None or away_proj is None or home_score is None or away_score is None:
            stats["invalid_current_matches"] += 1
            log(
                f"{league_upper(league)} | INVALID CURRENT MATCH | "
                f"game_id={gid!r} key={comp!r}",
                "WARN",
            )
            continue

        if total_proj is None:
            total_proj = home_proj + away_proj

        game_date = normalize_date(final.get("game_date") or pred.get("game_date"))
        home_team = str(final.get("home_team") or pred.get("home_team") or "").strip()
        away_team = str(final.get("away_team") or pred.get("away_team") or "").strip()

        if not game_date or not home_team or not away_team:
            stats["invalid_current_matches"] += 1
            log(
                f"{league_upper(league)} | INVALID CURRENT IDENTIFIERS | "
                f"game_id={gid!r} key={comp!r}",
                "WARN",
            )
            continue

        source = f"{pred.get('_source_file', '')} + {final.get('_source_file', '')}"
        games.append(
            CompletedGame(
                league=league,
                game_id=gid or canonical_game_id(pred.get("game_id")),
                game_date=game_date,
                game_time=str(pred.get("game_time", "")).strip(),
                home_team=home_team,
                away_team=away_team,
                home_projected_points=home_proj,
                away_projected_points=away_proj,
                total_projected_points=total_proj,
                home_score=home_score,
                away_score=away_score,
                source=source,
                source_priority=2,
            )
        )

        if match_method == "game_id":
            stats["matched_by_game_id"] += 1
        else:
            stats["matched_by_composite"] += 1

    stats["current_matched_games"] = len(games)

    log(
        f"{league_upper(league)} | CURRENT MATCH SUMMARY | "
        f"final_rows_scanned={stats['final_rows_scanned']} "
        f"covered_by_historical={stats['finals_already_covered_by_historical']} "
        f"current_to_match={stats['current_finals_to_match']} "
        f"matched_by_game_id={stats['matched_by_game_id']} "
        f"matched_by_composite={stats['matched_by_composite']} "
        f"ambiguous={stats['ambiguous_prediction_matches']} "
        f"true_unmatched={stats['true_unmatched_current_finals']} "
        f"invalid={stats['invalid_current_matches']}"
    )

    return games, stats


# ============================================================================
# UNIFIED COMPLETED-GAME HISTORY
# ============================================================================

def deduplicate_completed_games(
    games: list[CompletedGame],
) -> tuple[list[CompletedGame], int]:
    chosen: dict[str, CompletedGame] = {}
    duplicates = 0

    for game in games:
        key = game.match_key
        if not key:
            raise ValueError(
                f"Cannot deduplicate completed game with incomplete composite key: {game}"
            )

        existing = chosen.get(key)
        if existing is None:
            chosen[key] = game
            continue

        duplicates += 1

        if game.source_priority > existing.source_priority:
            chosen[key] = game
        elif game.source_priority == existing.source_priority and game.sort_key >= existing.sort_key:
            chosen[key] = game

    unique = sorted(chosen.values(), key=lambda game: game.sort_key)
    return unique, duplicates


def build_completed_history(league: str) -> tuple[list[CompletedGame], dict[str, Any]]:
    historical, historical_stats = load_historical_completed_games(league)
    current, current_stats = load_current_completed_games(league, historical)
    combined, duplicates = deduplicate_completed_games(historical + current)

    meta = {
        **historical_stats,
        **current_stats,
        "duplicates_removed_from_completed_history": duplicates,
        "unique_completed_games": len(combined),
    }

    if combined:
        meta["first_game_date"] = combined[0].game_date
        meta["last_game_date"] = combined[-1].game_date
    else:
        meta["first_game_date"] = None
        meta["last_game_date"] = None

    log(
        f"{league_upper(league)} | UNIFIED HISTORY | "
        f"historical={len(historical)} current={len(current)} "
        f"duplicates_removed={duplicates} unique={len(combined)} "
        f"range={meta['first_game_date']}..{meta['last_game_date']}"
    )

    return combined, meta


# ============================================================================
# BIAS CALCULATION
# ============================================================================

def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return sum(values) / len(values)


def calculate_component_bias(
    league: str,
    component: str,
    rule: dict[str, Any],
    history: list[CompletedGame],
) -> dict[str, Any]:
    method = rule.get("method")

    if method in (None, "", "null"):
        return {
            "status": "skipped_no_rule",
            "method": None,
            "value": None,
            "window_games": None,
            "games_used": 0,
        }

    if method == "none":
        return {
            "status": "ready",
            "method": "none",
            "value": 0.0,
            "window_games": 0,
            "games_used": 0,
            "first_game_date": None,
            "last_game_date": None,
        }

    if method == "fixed":
        value = rule.get("value")
        if value is None:
            raise ValueError(
                f"{league_upper(league)} {component} fixed bias requires "
                f"bias.{component}.value"
            )
        return {
            "status": "ready",
            "method": "fixed",
            "value": round(float(value), 6),
            "window_games": 0,
            "games_used": 0,
            "first_game_date": None,
            "last_game_date": None,
        }

    if method != "rolling":
        raise ValueError(
            f"{league_upper(league)} {component} bias method {method!r} is unsupported. "
            "Supported methods: rolling, fixed, none."
        )

    window = rule.get("window_games")
    if window is None or window <= 0:
        raise ValueError(
            f"{league_upper(league)} {component} rolling bias requires window_games > 0"
        )

    if len(history) < window:
        raise ValueError(
            f"{league_upper(league)} {component} rolling bias requires {window} completed games, "
            f"but only {len(history)} unique completed games are available across all sources"
        )

    selected = history[-window:]

    if component == "margin":
        errors = [game.margin_error for game in selected]
    elif component == "total":
        errors = [game.total_error for game in selected]
    else:
        raise ValueError(f"Unsupported bias component: {component}")

    value = average(errors)

    return {
        "status": "ready",
        "method": "rolling",
        "value": round(value, 6),
        "window_games": int(window),
        "games_used": len(selected),
        "first_game_date": selected[0].game_date,
        "last_game_date": selected[-1].game_date,
        "mean_error_definition": "projected_minus_actual",
    }


# ============================================================================
# STATE OUTPUT
# ============================================================================

def write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(state, f, sort_keys=False, default_flow_style=False)

    tmp_path.replace(STATE_PATH)
    log(f"STATE WRITTEN | {STATE_PATH}")


# ============================================================================
# LEAGUE PROCESSOR
# ============================================================================

def process_league(
    league: str,
    league_cfg: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    upper = league_upper(league)
    config_status = str(league_cfg.get("status", "")).strip().lower()

    margin_rule = resolve_bias_rule(league_cfg, "margin")
    total_rule = resolve_bias_rule(league_cfg, "total")

    has_margin_rule = margin_rule.get("method") not in (None, "", "null")
    has_total_rule = total_rule.get("method") not in (None, "", "null")

    if not has_margin_rule and not has_total_rule:
        log(f"{upper} | SKIPPED | no configured bias rules")
        return (
            {
                "status": "skipped_no_bias_rules",
                "config_status": config_status or None,
                "margin_bias": {
                    "status": "skipped_no_rule",
                    "method": None,
                    "value": None,
                    "window_games": None,
                    "games_used": 0,
                },
                "total_bias": {
                    "status": "skipped_no_rule",
                    "method": None,
                    "value": None,
                    "window_games": None,
                    "games_used": 0,
                },
            },
            True,
        )

    try:
        history, history_meta = build_completed_history(league)
        margin = calculate_component_bias(league, "margin", margin_rule, history)
        total = calculate_component_bias(league, "total", total_rule, history)

        league_state = {
            "status": "ready",
            "config_status": config_status or None,
            "margin_bias": margin,
            "total_bias": total,
            "history": history_meta,
        }

        log(
            f"{upper} | READY | "
            f"margin_bias={margin.get('value')} margin_method={margin.get('method')} "
            f"margin_games={margin.get('games_used')} | "
            f"total_bias={total.get('value')} total_method={total.get('method')} "
            f"total_games={total.get('games_used')}"
        )
        return league_state, True

    except Exception as exc:
        log(f"{upper} | FAILED | {exc}", "ERROR")
        return (
            {
                "status": "error",
                "config_status": config_status or None,
                "error": str(exc),
                "margin_bias": {
                    "method": margin_rule.get("method"),
                    "window_games": margin_rule.get("window_games"),
                    "value": None,
                },
                "total_bias": {
                    "method": total_rule.get("method"),
                    "window_games": total_rule.get("window_games"),
                    "value": None,
                },
            },
            False,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    init_log()

    try:
        cfg = load_model_config()
        config_leagues = cfg.get("leagues", {})

        state: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "source_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "leagues": {},
        }

        configured_failures = 0

        for league in SUPPORTED_LEAGUES:
            league_cfg = config_leagues.get(league) or {}
            league_state, ok = process_league(league, league_cfg)
            state["leagues"][league] = league_state
            if not ok:
                configured_failures += 1

        for league in config_leagues:
            league_key = str(league).strip().lower()
            if league_key not in SUPPORTED_LEAGUES:
                state["leagues"][league_key] = {
                    "status": "skipped_unsupported_league",
                    "error": f"Unsupported league key: {league}",
                }
                log(
                    f"{str(league).upper()} | SKIPPED | unsupported league key",
                    "WARN",
                )

        state["run_status"] = (
            "success" if configured_failures == 0 else "completed_with_errors"
        )
        write_state(state)

        print("basketball rolling-bias calculation complete.")
        print(f"state: {STATE_PATH}")

        if configured_failures:
            print(
                f"configured league failures: {configured_failures}",
                file=sys.stderr,
            )
            return 1

        return 0

    except Exception as exc:
        log(f"FATAL | {exc}\n{traceback.format_exc()}", "ERROR")
        print(f"calculate_rolling_bias failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

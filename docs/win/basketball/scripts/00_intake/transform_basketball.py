#!/usr/bin/env python3
"""Season-aware transform launcher with final-score upsert semantics.

The parser/prediction implementation remains in transform_basketball_core.py. This
launcher replaces the old "skip if date file exists" final-score behavior with an
idempotent upsert so late finals and corrected scores can be incorporated.
"""
from __future__ import annotations

import importlib.util
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
CORE_PATH = Path(__file__).with_name("transform_basketball_core.py")

FINAL_COLUMNS = [
    "sport", "league", "game_id", "game_date", "home_team", "away_team",
    "home_score", "away_score", "total", "home_spread", "away_spread",
]


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def in_season(league: str, now: datetime) -> bool:
    if league in {"nba", "ncaam"}:
        return now.month >= 9 or now.month <= 6 or (now.month == 7 and now.day == 1)
    if league == "wnba":
        return 5 <= now.month <= 10
    return True


def load_core():
    spec = importlib.util.spec_from_file_location("transform_basketball_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load transform core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def composite_key(row: dict) -> tuple[str, str, str]:
    return (
        clean(row.get("game_date")),
        clean(row.get("home_team")).casefold(),
        clean(row.get("away_team")).casefold(),
    )


def id_rank(game_id: str) -> int:
    gid = clean(game_id)
    if not gid:
        return 0
    if gid.isdigit():
        return 2
    return 1


def collapse_rows(rows: list[dict]) -> list[dict]:
    """One row per date/home/away, preserving the strongest known game_id."""
    by_comp: dict[tuple[str, str, str], dict] = {}
    for raw in rows:
        row = {c: raw.get(c, "") for c in FINAL_COLUMNS}
        key = composite_key(row)
        if key not in by_comp:
            by_comp[key] = row
            continue
        old = by_comp[key]
        old_gid, new_gid = clean(old.get("game_id")), clean(row.get("game_id"))
        if old_gid.isdigit() and new_gid.isdigit() and old_gid != new_gid:
            raise ValueError(
                f"Conflicting numeric game_ids for final score {key}: {old_gid} vs {new_gid}"
            )
        # New score values are authoritative; preserve/prefer canonical ID.
        chosen_gid = new_gid if id_rank(new_gid) > id_rank(old_gid) else old_gid
        merged = dict(old)
        for col in FINAL_COLUMNS:
            value = row.get(col, "")
            if clean(value) != "":
                merged[col] = value
        merged["game_id"] = chosen_gid
        by_comp[key] = merged
    return list(by_comp.values())


def make_process_final_scores(core):
    def process_final_scores(df, files_written, league_key, stats):
        cfg = core.LEAGUE_CONFIG[league_key]
        league_label = cfg["league_label"]
        mask = df["score1"].notna() & (df["score1"].astype(str).str.strip() != "")
        completed = df[mask].copy()
        stats["completed_games"] += len(completed)
        if completed.empty:
            core.log(league_key, f"No completed {league_label} games found in this file.")
            return

        for date_val, group in completed.groupby("game_date"):
            path = Path(cfg["final_scores_dir"]) / f"{date_val}_final_scores_{league_label}.csv"
            incoming = []
            for _, row in group.iterrows():
                try:
                    away_score = int(float(row["score1"]))
                    home_score = int(float(row["score2"]))
                    total = away_score + home_score
                    away_spread = away_score - home_score
                    home_spread = home_score - away_score
                except (ValueError, TypeError):
                    away_score = home_score = total = away_spread = home_spread = ""
                incoming.append({
                    "sport": "Basketball",
                    "league": league_label,
                    "game_id": "",
                    "game_date": date_val,
                    "home_team": row["team2"],
                    "away_team": row["team1"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "total": total,
                    "home_spread": home_spread,
                    "away_spread": away_spread,
                })

            existing: list[dict] = []
            if path.exists():
                old = pd.read_csv(path, dtype=str, keep_default_na=False)
                existing = old.to_dict("records")

            before = len(existing)
            merged = collapse_rows(existing + incoming)
            out = pd.DataFrame(merged, columns=FINAL_COLUMNS)
            core.save(out, str(path), files_written, league_key)
            stats["final_score_files_written"] += 1
            stats["final_score_rows_written"] += len(out)
            core.log(
                league_key,
                f"UPSERT final scores: {path} existing_rows={before} incoming_rows={len(incoming)} "
                f"result_rows={len(out)}",
            )
    return process_final_scores


def main() -> None:
    core = load_core()
    core.process_final_scores = make_process_final_scores(core)

    now = datetime.now(NY)
    force_all = truthy(os.getenv("BASKETBALL_FORCE_ALL_LEAGUES"))
    leagues = list(core.LEAGUE_CONFIG)
    active = leagues if force_all else [lg for lg in leagues if in_season(lg, now)]

    had_errors = False
    for league_key in active:
        core.process_league(league_key)
        text = Path(core.LEAGUE_CONFIG[league_key]["log_file"]).read_text(
            encoding="utf-8", errors="replace"
        )
        if "STATUS: FAILED" in text or "STATUS: PARTIAL" in text:
            had_errors = True

    skipped = sorted(set(leagues) - set(active))
    for league_key in skipped:
        core.init_log(league_key)
        core.log(
            league_key,
            f"SEASON GATE: skipped on {now.strftime('%Y_%m_%d')} America/New_York; "
            "set BASKETBALL_FORCE_ALL_LEAGUES=1 to override",
        )
        core.write_summary(league_key, [], {
            "input_files_found": 0, "input_files_processed": 0, "games_loaded": 0,
            "upcoming_games": 0, "completed_games": 0, "prediction_files_written": 0,
            "prediction_rows_written": 0, "final_score_files_written": 0,
            "final_score_rows_written": 0, "file_errors": 0,
        }, "SUCCESS (offseason skipped)")

    if had_errors:
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()

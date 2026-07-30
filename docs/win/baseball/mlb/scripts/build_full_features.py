#!/usr/bin/env python3
"""Build a leak-safe, lineage-tracked MLB feature matrix.

Historical mode:
  python build_full_features.py --data-dir baseball_projector --out-dir model

Live mode (same feature functions):
  python build_full_features.py --data-dir baseball_projector --out-dir model \
      --mode live --upcoming-csv model/upcoming_games.csv

Historical output is one row per game with home/away targets. Every historical
statistic is shifted so the current game never contributes to its own features.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

WINDOWS = (7, 14, 30)
SLOT_WEIGHTS = np.array([0.1235, 0.1211, 0.1188, 0.1164, 0.1140,
                         0.1093, 0.1046, 0.0999, 0.0924], dtype=float)
SLOT_WEIGHTS /= SLOT_WEIGHTS.sum()

GAME_CONTEXT = [
    "gid", "date", "season", "visteam", "hometeam", "site", "number",
    "starttime", "daynight", "tiebreaker", "usedh", "fieldcond", "precip",
    "sky", "temp", "winddir", "windspeed", "umphome", "ump1b", "ump2b",
    "ump3b", "umplf", "umprf", "gametype",
]

CURRENT_TARGETS = {"vruns", "hruns", "wteam", "lteam", "wp", "lp", "save"}
NON_FEATURE_ADMIN = {
    "box", "pbp", "stattype", "gametype", "number", "date", "gid", "site",
    "team", "opp", "vishome", "id", "event", "vis_home", "batteam", "pitteam",
    "batter", "pitcher", "starttime",
}
PLAY_EXCLUDE = {
    "score_v", "score_h", "br1_pre", "br2_pre", "br3_pre", "br1_post",
    "br2_post", "br3_post", "lob_id1", "lob_id2", "lob_id3", "pr1_pre",
    "pr2_pre", "pr3_pre", "pr1_post", "pr2_post", "pr3_post", "run_b",
    "run1", "run2", "run3", "prun_b", "prun1", "prun2", "prun3",
    "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9",
    "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9",
    "umphome", "ump1b", "ump2b", "ump3b", "umplf", "umprf", "pitches",
}


@dataclass(frozen=True)
class LineageRow:
    feature: str
    source_file: str
    source_columns: str
    transformation: str
    timing_rule: str
    feature_group: str
    notes: str = ""


class Lineage:
    def __init__(self) -> None:
        self._rows: dict[str, LineageRow] = {}

    def add(self, feature: str, source_file: str, source_columns: Sequence[str] | str,
            transformation: str, feature_group: str, notes: str = "") -> None:
        cols = source_columns if isinstance(source_columns, str) else "|".join(source_columns)
        self._rows[feature] = LineageRow(
            feature=feature,
            source_file=source_file,
            source_columns=cols,
            transformation=transformation,
            timing_rule="strictly before current game date unless marked direct_current",
            feature_group=feature_group,
            notes=notes,
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(x) for x in self._rows.values()]).sort_values("feature")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="baseball_projector")
    p.add_argument("--out-dir", default="model")
    p.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--mode", choices=["historical", "live"], default="historical")
    p.add_argument("--upcoming-csv")
    p.add_argument("--windows", nargs="+", type=int, default=list(WINDOWS))
    p.add_argument("--regular-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-category-games", type=int, default=100)
    return p.parse_args()


def find_file(data_dir: Path, season: int, kind: str) -> Path | None:
    matches = sorted(data_dir.rglob(f"{season}{kind}.csv"))
    return matches[0] if matches else None


def load_kind(data_dir: Path, seasons: Iterable[int], kind: str,
              regular_only: bool = True, usecols=None) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = find_file(data_dir, season, kind)
        if path is None:
            continue
        df = pd.read_csv(path, low_memory=False, usecols=usecols)
        df["season"] = season
        if regular_only and "gametype" in df.columns:
            df = df[df["gametype"].astype(str).str.lower().eq("regular")]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {kind} files found for requested seasons")
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return out


def numeric_stat_columns(df: pd.DataFrame, prefixes: Sequence[str] | None = None,
                         exclude: set[str] | None = None) -> list[str]:
    exclude = (exclude or set()) | NON_FEATURE_ADMIN
    cols: list[str] = []
    for c in df.columns:
        if c in exclude or c == "season":
            continue
        if prefixes and not any(c.startswith(p) for p in prefixes):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.50:
            cols.append(c)
    return cols


def dedupe_daily(df: pd.DataFrame, entity_cols: list[str], value_cols: list[str],
                 agg: str = "sum") -> pd.DataFrame:
    keys = entity_cols + ["date", "season"]
    work = df[keys + value_cols].copy()
    for c in value_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    grouped = work.groupby(keys, as_index=False, dropna=False)[value_cols]
    return grouped.sum(min_count=1) if agg == "sum" else grouped.mean()


def add_history_features(base: pd.DataFrame, entity_cols: list[str], value_cols: list[str],
                         prefix: str, windows: Sequence[int], lineage: Lineage,
                         source_file: str, feature_group: str,
                         include_std: bool = True, include_prior: bool = True,
                         include_home_away: bool = False) -> pd.DataFrame:
    """Return keys plus shifted rolling/season/prior features.

    base must contain one row per entity/date (or entity/date/home-away split).
    All rolling windows use closed='left', excluding every game on current date.
    """
    if not value_cols:
        return base[entity_cols + ["date", "season"]].copy()
    keys = entity_cols + ["date", "season"]
    raw = base[keys + value_cols].copy()
    for c in value_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    # One state per entity/date prevents doubleheaders from creating many-to-many merges.
    work = raw.groupby(keys, as_index=False, dropna=False)[value_cols].sum(min_count=1)
    work = work.sort_values(entity_cols + ["date"]).reset_index(drop=True)
    result = work[keys].copy()

    index_work = work.set_index("date")
    for w in windows:
        rolled = (
            index_work.groupby(entity_cols, dropna=False)[value_cols]
            .rolling(f"{int(w)}D", closed="left", min_periods=1)
            .mean()
            .reset_index()
        )
        new_cols = {c: f"{prefix}_{c}_{w}d_mean" for c in value_cols}
        rolled = rolled.rename(columns=new_cols)
        result = result.merge(rolled, on=entity_cols + ["date"], how="left")
        for raw, feat in new_cols.items():
            lineage.add(feat, source_file, raw, f"mean over prior {w} calendar days",
                        feature_group)

    # Season-to-date mean/std, excluding current row and all same-day rows.
    day = work.groupby(entity_cols + ["season", "date"], as_index=False)[value_cols].sum(min_count=1)
    day = day.sort_values(entity_cols + ["season", "date"]).reset_index(drop=True)
    group_keys = entity_cols + ["season"]
    grp = day.groupby(group_keys, dropna=False)
    values = day[value_cols].fillna(0.0)
    prior_sum = grp[value_cols].cumsum() - values
    prior_n = grp.cumcount().astype(float)
    denom = prior_n.replace(0, np.nan)

    mean_df = prior_sum.div(denom, axis=0)
    mean_df.columns = [f"{prefix}_{c}_std_mean" for c in value_cols]
    pieces = [day[entity_cols + ["season", "date"]].reset_index(drop=True), mean_df.reset_index(drop=True)]
    for c, feat in zip(value_cols, mean_df.columns):
        lineage.add(feat, source_file, c, "season-to-date mean excluding current date", feature_group)

    if include_std:
        squared = values.pow(2)
        temp = pd.concat([day[group_keys], squared.add_prefix("__sq__")], axis=1)
        sq_cols = [f"__sq__{c}" for c in value_cols]
        prior_sumsq = temp.groupby(group_keys, dropna=False)[sq_cols].cumsum() - squared.to_numpy()
        prior_sumsq.columns = value_cols
        variance = (prior_sumsq - prior_sum.pow(2).div(denom, axis=0)).div((prior_n - 1).replace(0, np.nan), axis=0)
        variance = variance.clip(lower=0)
        std_df = np.sqrt(variance)
        std_df.columns = [f"{prefix}_{c}_std_sd" for c in value_cols]
        pieces.append(std_df.reset_index(drop=True))
        for c, feat in zip(value_cols, std_df.columns):
            lineage.add(feat, source_file, c, "season-to-date standard deviation excluding current date", feature_group)

    std_block = pd.concat(pieces, axis=1)
    result = result.merge(std_block, on=entity_cols + ["season", "date"], how="left")

    if include_prior:
        season_agg = work.groupby(entity_cols + ["season"], as_index=False)[value_cols].mean()
        season_agg["season"] = season_agg["season"] + 1
        ren = {c: f"{prefix}_{c}_prior_season" for c in value_cols}
        season_agg = season_agg.rename(columns=ren)
        result = result.merge(season_agg, on=entity_cols + ["season"], how="left")
        for raw, feat in ren.items():
            lineage.add(feat, source_file, raw, "previous season mean", feature_group)

    return result


def add_split_30d(base: pd.DataFrame, entity: str, split_col: str,
                  value_cols: list[str], prefix: str, lineage: Lineage,
                  source_file: str, feature_group: str) -> pd.DataFrame:
    if not value_cols or split_col not in base.columns:
        return base[[entity, "date", "season"]].copy()
    work = base[[entity, split_col, "date", "season"] + value_cols].copy()
    for c in value_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.groupby([entity, split_col, "date", "season"], as_index=False, dropna=False)[value_cols].sum(min_count=1)
    idx = work.sort_values([entity, split_col, "date"]).set_index("date")
    rolled = (
        idx.groupby([entity, split_col], dropna=False)[value_cols]
        .rolling("30D", closed="left", min_periods=1).mean().reset_index()
    )
    ren = {c: f"{prefix}_{c}_same_{split_col}_30d" for c in value_cols}
    rolled = rolled.rename(columns=ren)
    for raw, feat in ren.items():
        lineage.add(feat, source_file, [raw, split_col],
                    "prior 30-day mean within same home/away split", feature_group)
    return rolled


def context_features(gameinfo: pd.DataFrame, lineage: Lineage) -> pd.DataFrame:
    cols = [c for c in GAME_CONTEXT if c in gameinfo.columns]
    g = gameinfo[cols + [c for c in ("vruns", "hruns") if c in gameinfo.columns]].copy()
    g["month"] = g["date"].dt.month
    g["dow"] = g["date"].dt.dayofweek
    g["day_of_year"] = g["date"].dt.dayofyear
    g["start_hour"] = pd.to_datetime(g.get("starttime", pd.Series(index=g.index, dtype=str)),
                                      format="%I:%M%p", errors="coerce").dt.hour
    for c in ["month", "dow", "day_of_year", "start_hour", "site", "daynight", "tiebreaker",
              "usedh", "fieldcond", "precip", "sky", "temp", "winddir", "windspeed",
              "umphome", "ump1b", "ump2b", "ump3b"]:
        if c in g.columns:
            lineage.add(c, "{season}gameinfo.csv", c, "direct_current", "context",
                        "Expected to be supplied by the live intake before first pitch")
    return g


def build_team_history(teamstats: pd.DataFrame, windows: Sequence[int], lineage: Lineage) -> pd.DataFrame:
    value_cols = numeric_stat_columns(
        teamstats,
        prefixes=("b_", "p_", "d_", "inn"),
        exclude={"win", "loss", "tie"},
    )
    for extra in ["lob", "win", "loss", "tie"]:
        if extra in teamstats.columns:
            value_cols.append(extra)
    keys = ["gid", "team", "date", "season", "vishome", "opp"]
    daily = teamstats[keys + value_cols].copy()
    for c in value_cols:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    hist = add_history_features(daily, ["team"], value_cols, "team", windows, lineage,
                                "{season}teamstats.csv", "team_history")
    split = add_split_30d(daily, "team", "vishome", value_cols, "team", lineage,
                          "{season}teamstats.csv", "team_home_away")
    hist = hist.merge(split, on=["team", "date"], how="left", suffixes=("", "_split"))
    out = daily[keys].merge(hist, on=["team", "date", "season"], how="left")
    return out


def build_bullpen_history(pitching: pd.DataFrame, windows: Sequence[int], lineage: Lineage) -> pd.DataFrame:
    p = pitching.copy()
    p["p_gs_num"] = pd.to_numeric(p.get("p_gs"), errors="coerce").fillna(0)
    rel = p[p["p_gs_num"] != 1].copy()
    value_cols = numeric_stat_columns(rel, prefixes=("p_",), exclude={"p_gs"})
    keys = ["gid", "team", "date", "season"]
    for c in value_cols:
        rel[c] = pd.to_numeric(rel[c], errors="coerce")
    daily = rel.groupby(keys, as_index=False)[value_cols].sum(min_count=1)
    count = rel.groupby(keys, as_index=False).size().rename(columns={"size": "relievers_used"})
    daily = daily.merge(count, on=keys, how="left")
    value_cols.append("relievers_used")
    hist = add_history_features(daily, ["team"], value_cols, "bullpen", windows, lineage,
                                "{season}pitching.csv", "bullpen")
    return daily[keys].merge(hist, on=["team", "date", "season"], how="left")


def build_starter_history(pitching: pd.DataFrame, windows: Sequence[int], lineage: Lineage) -> pd.DataFrame:
    p = pitching.copy()
    p["p_gs_num"] = pd.to_numeric(p.get("p_gs"), errors="coerce").fillna(0)
    sp = p[p["p_gs_num"] == 1].copy()
    value_cols = numeric_stat_columns(sp, prefixes=("p_",), exclude={"p_gs"})
    keys = ["gid", "id", "team", "date", "season"]
    sp = sp[keys + value_cols].drop_duplicates(["gid", "id", "team"])
    hist = add_history_features(sp, ["id"], value_cols, "starter", windows, lineage,
                                "{season}pitching.csv", "starter")
    out = sp[keys].merge(hist, on=["id", "date", "season"], how="left")
    sp_dates = sp.sort_values(["id", "date"])
    sp_dates["starter_days_rest"] = sp_dates.groupby("id")["date"].diff().dt.days
    out = out.merge(sp_dates[["gid", "id", "starter_days_rest"]], on=["gid", "id"], how="left")
    lineage.add("starter_days_rest", "{season}pitching.csv", "date",
                "days since previous start", "starter")
    return out


def build_batter_history(batting: pd.DataFrame, windows: Sequence[int], lineage: Lineage) -> pd.DataFrame:
    value_cols = numeric_stat_columns(batting, prefixes=("b_",), exclude=set())
    for c in ["dh", "ph", "pr"]:
        if c in batting.columns:
            value_cols.append(c)
    keys = ["gid", "id", "team", "date", "season"]
    bg = dedupe_daily(batting, ["gid", "id", "team"], value_cols, agg="sum")
    hist = add_history_features(bg, ["id"], value_cols, "batter", windows, lineage,
                                "{season}batting.csv", "batter")
    return bg[keys].merge(hist, on=["id", "date", "season"], how="left")


def aggregate_lineup(teamstats: pd.DataFrame, batter_hist: pd.DataFrame,
                     allplayers: pd.DataFrame, lineage: Lineage) -> pd.DataFrame:
    lineup_cols = [f"start_l{i}" for i in range(1, 10) if f"start_l{i}" in teamstats.columns]
    keys = ["gid", "team", "date", "season"]
    long = teamstats[keys + lineup_cols].melt(keys, var_name="slot", value_name="id")
    long["slot_num"] = pd.to_numeric(long["slot"].str.extract(r"(\d+)")[0], errors="coerce")
    long["slot_weight"] = long["slot_num"].map({i + 1: SLOT_WEIGHTS[i] for i in range(9)}).fillna(0.0)
    form_cols = [c for c in batter_hist.columns if c.startswith("batter_")]
    long = long.merge(batter_hist[["gid", "id"] + form_cols], on=["gid", "id"], how="left")

    hand = allplayers[["season", "id", "bat", "throw"]].drop_duplicates(["season", "id"])
    long = long.merge(hand, on=["season", "id"], how="left")
    long["is_left_bat"] = long["bat"].eq("L").astype(float)
    long["is_switch_bat"] = long["bat"].eq("B").astype(float)
    long["is_right_bat"] = long["bat"].eq("R").astype(float)

    values = long[form_cols].apply(pd.to_numeric, errors="coerce")
    weights = long["slot_weight"]
    weighted_values = values.mul(weights, axis=0)
    weighted_values = weighted_values.where(values.notna())
    weighted_den = values.notna().mul(weights, axis=0)

    group_index = [long[k] for k in keys]
    numerator = weighted_values.groupby(group_index, dropna=False).sum(min_count=1)
    denominator = weighted_den.groupby(group_index, dropna=False).sum(min_count=1).replace(0, np.nan)
    wavg = numerator / denominator
    wavg.columns = [f"lineup_{c}_wavg" for c in form_cols]

    # Cross-lineup spread is useful but expensive only in width, not runtime.
    spread = values.groupby(group_index, dropna=False).std(ddof=0)
    spread.columns = [f"lineup_{c}_sd" for c in form_cols]

    hand_counts = long.groupby(keys, dropna=False).agg(
        lineup_left_count=("is_left_bat", "sum"),
        lineup_switch_count=("is_switch_bat", "sum"),
        lineup_right_count=("is_right_bat", "sum"),
        lineup_known_count=("id", "count"),
    )
    out = pd.concat([wavg, spread, hand_counts], axis=1).reset_index()

    for c in form_cols:
        lineage.add(f"lineup_{c}_wavg", "{season}batting.csv|{season}teamstats.csv",
                    [c.replace("batter_", ""), "start_l1..start_l9"],
                    "batting-slot-weighted average of pregame batter feature", "lineup")
        lineage.add(f"lineup_{c}_sd", "{season}batting.csv|{season}teamstats.csv",
                    [c.replace("batter_", ""), "start_l1..start_l9"],
                    "standard deviation across announced starting lineup", "lineup")
    for c in ["lineup_left_count", "lineup_switch_count", "lineup_right_count", "lineup_known_count"]:
        lineage.add(c, "{season}allplayers.csv|{season}teamstats.csv", ["bat", "start_l1..start_l9"],
                    "count across current starting lineup", "lineup")
    return out

def build_fielding_history(fielding: pd.DataFrame, windows: Sequence[int], lineage: Lineage) -> pd.DataFrame:
    value_cols = numeric_stat_columns(fielding, prefixes=("d_",), exclude={"d_pos", "d_seq", "d_gs"})
    keys = ["gid", "team", "date", "season"]
    daily = fielding.groupby(keys, as_index=False)[value_cols].sum(min_count=1)

    # Positional detail: retain every numeric field by position.
    pos = fielding[[*keys, "d_pos", *value_cols]].copy()
    pos["d_pos"] = pd.to_numeric(pos["d_pos"], errors="coerce").astype("Int64")
    pos = pos[pos["d_pos"].between(1, 9, inclusive="both")]
    pos = pos.groupby(keys + ["d_pos"], as_index=False)[value_cols].sum(min_count=1)
    piv = pos.pivot_table(index=keys, columns="d_pos", values=value_cols, aggfunc="sum")
    piv.columns = [f"{raw}_pos{int(position)}" for raw, position in piv.columns]
    piv = piv.reset_index()
    daily = daily.merge(piv, on=keys, how="left")
    all_values = [c for c in daily.columns if c not in keys]
    hist = add_history_features(daily, ["team"], all_values, "fielding", windows, lineage,
                                "{season}fielding.csv", "fielding")
    return daily[keys].merge(hist, on=["team", "date", "season"], how="left")


def build_play_history(plays: pd.DataFrame, windows: Sequence[int], lineage: Lineage,
                       min_category_games: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclude = NON_FEATURE_ADMIN | PLAY_EXCLUDE | {"inning", "top_bot", "pn", "lp", "bat_f"}
    value_cols = numeric_stat_columns(plays, prefixes=None, exclude=exclude)
    # Keep event/batted-ball categorical frequencies when sufficiently common.
    cat_cols = [c for c in ["event", "hittype", "loc", "bathand", "pithand"] if c in plays.columns]
    p = plays[["gid", "date", "season", "batteam", "pitteam"] + value_cols + cat_cols].copy()
    for c in value_cols:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    cat_features = {}
    for c in cat_cols:
        values_as_text = p[c].astype(str)
        counts = values_as_text.value_counts()
        keep_levels = counts[counts >= min_category_games].index[:30]
        for lv in keep_levels:
            safe = re.sub(r"[^A-Za-z0-9]+", "_", str(lv)).strip("_")[:32] or "blank"
            base_col = f"cat_{c}_{safe}"
            col = base_col
            suffix = 2
            while col in cat_features or col in value_cols:
                col = f"{base_col}_{suffix}"
                suffix += 1
            cat_features[col] = values_as_text.eq(str(lv)).astype(float)
            value_cols.append(col)
            lineage.add(f"plays_off_{col}_rolling", "{season}plays.csv", c,
                        f"historical frequency for category {lv}", "plays")
    if cat_features:
        p = pd.concat([p, pd.DataFrame(cat_features, index=p.index)], axis=1).copy()

    keys_off = ["gid", "batteam", "date", "season"]
    keys_def = ["gid", "pitteam", "date", "season"]
    off = p.groupby(keys_off, as_index=False)[value_cols].mean()
    deff = p.groupby(keys_def, as_index=False)[value_cols].mean()
    off_hist = add_history_features(off, ["batteam"], value_cols, "plays_off", windows, lineage,
                                    "{season}plays.csv", "play_offense")
    def_hist = add_history_features(deff, ["pitteam"], value_cols, "plays_def", windows, lineage,
                                    "{season}plays.csv", "play_defense")
    off_out = off[keys_off].merge(off_hist, on=["batteam", "date", "season"], how="left")
    def_out = deff[keys_def].merge(def_hist, on=["pitteam", "date", "season"], how="left")
    return off_out, def_out


def build_park_umpire_history(gameinfo: pd.DataFrame, lineage: Lineage) -> pd.DataFrame:
    g = gameinfo[[c for c in ["gid", "date", "season", "site", "umphome", "vruns", "hruns"] if c in gameinfo.columns]].copy()
    if not {"vruns", "hruns"}.issubset(g.columns):
        return g[["gid"]]
    g["total_runs"] = pd.to_numeric(g["vruns"], errors="coerce") + pd.to_numeric(g["hruns"], errors="coerce")
    g["home_run_share"] = np.where(g["total_runs"] > 0, pd.to_numeric(g["hruns"], errors="coerce") / g["total_runs"], 0.5)
    result = g[["gid", "date", "season", "site", "umphome"]].copy()
    for entity, pref in [("site", "park"), ("umphome", "umpire")]:
        h = add_history_features(g[[entity, "date", "season", "total_runs", "home_run_share"]],
                                 [entity], ["total_runs", "home_run_share"], pref, (30,), lineage,
                                 "{season}gameinfo.csv", pref, include_std=True, include_prior=True)
        result = result.merge(h, on=[entity, "date", "season"], how="left")
    return result


def prefix_team_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    keep_keys = ["gid"]
    rename = {c: f"{side}_{c}" for c in df.columns if c not in keep_keys}
    return df.rename(columns=rename)


def combine_game_level(game: pd.DataFrame, teamstats: pd.DataFrame,
                       components: list[pd.DataFrame]) -> pd.DataFrame:
    # Start from one row per team-game and merge every team-keyed component.
    team_base = teamstats[["gid", "team", "date", "season", "vishome", "opp"]].drop_duplicates(["gid", "team"])
    for comp in components:
        merge_keys = [c for c in ["gid", "team"] if c in comp.columns]
        if "batteam" in comp.columns:
            comp = comp.rename(columns={"batteam": "team"})
            merge_keys = ["gid", "team"]
        if "pitteam" in comp.columns:
            comp = comp.rename(columns={"pitteam": "team"})
            merge_keys = ["gid", "team"]
        if not merge_keys:
            continue
        drop = [c for c in ["date", "season", "vishome", "opp"] if c in comp.columns]
        team_base = team_base.merge(comp.drop(columns=drop), on=merge_keys, how="left")

    away = prefix_team_side(team_base[team_base["vishome"].astype(str).str.lower().isin(["v", "0", "away"])].drop_duplicates("gid"), "away")
    home = prefix_team_side(team_base[team_base["vishome"].astype(str).str.lower().isin(["h", "1", "home"])].drop_duplicates("gid"), "home")
    # prefix_team_side renamed team context too; retain only gid plus all prefixed columns.
    out = game.merge(home, on="gid", how="left").merge(away, on="gid", how="left")
    return out


def construct_live_inputs(upcoming: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"gid", "date", "visteam", "hometeam", "site", "away_starter", "home_starter"}
    missing = sorted(required - set(upcoming.columns))
    if missing:
        raise ValueError(f"Upcoming CSV missing required columns: {missing}")
    up = upcoming.copy()
    up["date"] = pd.to_datetime(up["date"], errors="coerce")
    up["season"] = up["date"].dt.year
    up["gametype"] = up.get("gametype", "regular")
    up["vruns"] = np.nan
    up["hruns"] = np.nan

    rows = []
    for _, r in up.iterrows():
        for side, vishome, team, opp in [
            ("away", "v", r["visteam"], r["hometeam"]),
            ("home", "h", r["hometeam"], r["visteam"]),
        ]:
            rec = {
                "gid": r["gid"], "team": team, "opp": opp, "vishome": vishome,
                "date": r["date"], "season": int(r["season"]), "gametype": "regular",
                "start_f1": r[f"{side}_starter"],
            }
            for i in range(1, 10):
                rec[f"start_l{i}"] = r.get(f"{side}_l{i}", np.nan)
            rows.append(rec)
    return up, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage = Lineage()

    gameinfo = load_kind(data_dir, args.seasons, "gameinfo", args.regular_only)
    teamstats = load_kind(data_dir, args.seasons, "teamstats", args.regular_only)
    batting = load_kind(data_dir, args.seasons, "batting", args.regular_only)
    pitching = load_kind(data_dir, args.seasons, "pitching", args.regular_only)
    fielding = load_kind(data_dir, args.seasons, "fielding", args.regular_only)
    plays = load_kind(data_dir, args.seasons, "plays", args.regular_only)
    allplayers = load_kind(data_dir, args.seasons, "allplayers", False)

    if args.mode == "live":
        if not args.upcoming_csv:
            raise SystemExit("--upcoming-csv is required in live mode")
        upcoming = pd.read_csv(args.upcoming_csv, low_memory=False)
        live_game, live_team = construct_live_inputs(upcoming)
        # Append blank live rows so the same closed-left history functions create as-of features.
        gameinfo = pd.concat([gameinfo, live_game.reindex(columns=gameinfo.columns)], ignore_index=True)
        teamstats = pd.concat([teamstats, live_team.reindex(columns=teamstats.columns)], ignore_index=True)

        blank_bat = []
        for _, r in live_team.iterrows():
            for i in range(1, 10):
                if pd.notna(r.get(f"start_l{i}")):
                    blank_bat.append({"gid": r["gid"], "id": r[f"start_l{i}"], "team": r["team"],
                                      "date": r["date"], "season": r["season"], "gametype": "regular"})
        if blank_bat:
            batting = pd.concat([batting, pd.DataFrame(blank_bat).reindex(columns=batting.columns)], ignore_index=True)
        blank_sp = live_team[["gid", "start_f1", "team", "date", "season"]].rename(columns={"start_f1": "id"})
        blank_sp["p_gs"] = 1
        blank_sp["gametype"] = "regular"
        pitching = pd.concat([pitching, blank_sp.reindex(columns=pitching.columns)], ignore_index=True)
        # Blank team rows for source-specific team histories.
        blank_team = live_team[["gid", "team", "date", "season"]].copy()
        fielding = pd.concat([fielding, blank_team.reindex(columns=fielding.columns)], ignore_index=True)
        p_off = blank_team.rename(columns={"team": "batteam"})
        p_off["pitteam"] = live_team["opp"].values
        plays = pd.concat([plays, p_off.reindex(columns=plays.columns)], ignore_index=True)

    print("Building context...")
    game = context_features(gameinfo, lineage)
    print("Building team history...")
    team_hist = build_team_history(teamstats, args.windows, lineage)
    print("Building batter and lineup history...")
    batter_hist = build_batter_history(batting, args.windows, lineage)
    lineup_hist = aggregate_lineup(teamstats, batter_hist, allplayers, lineage)
    print("Building starter and bullpen history...")
    starter_hist = build_starter_history(pitching, args.windows, lineage)
    bullpen_hist = build_bullpen_history(pitching, args.windows, lineage)
    print("Building fielding history...")
    fielding_hist = build_fielding_history(fielding, args.windows, lineage)
    print("Building play-level history...")
    plays_off, plays_def = build_play_history(plays, args.windows, lineage, args.min_category_games)
    print("Building park and umpire history...")
    park_ump = build_park_umpire_history(gameinfo, lineage)

    # Attach current starter form to its own team-game.
    current_starters = teamstats[["gid", "team", "start_f1"]].drop_duplicates(["gid", "team"]).rename(columns={"start_f1": "id"})
    starter_current = current_starters.merge(starter_hist.drop(columns=["team"], errors="ignore"), on=["gid", "id"], how="left")
    starter_current = starter_current.drop(columns=["id"], errors="ignore")

    components = [team_hist, lineup_hist, starter_current, bullpen_hist, fielding_hist, plays_off, plays_def]
    full = combine_game_level(game, teamstats, components)
    full = full.merge(park_ump.drop(columns=["date", "season", "site", "umphome"], errors="ignore"), on="gid", how="left")

    # Targets and stable identifiers.
    if "hruns" in full.columns:
        full["home_runs"] = pd.to_numeric(full["hruns"], errors="coerce")
        full["away_runs"] = pd.to_numeric(full["vruns"], errors="coerce")
        full["home_win"] = np.where(full["home_runs"].notna() & full["away_runs"].notna(),
                                     (full["home_runs"] > full["away_runs"]).astype(float), np.nan)
    for c in ["hruns", "vruns"]:
        full = full.drop(columns=c, errors="ignore")

    full = full.sort_values(["date", "gid"]).drop_duplicates("gid").reset_index(drop=True)
    if args.mode == "live":
        live_ids = set(pd.read_csv(args.upcoming_csv)["gid"].astype(str))
        full = full[full["gid"].astype(str).isin(live_ids)].copy()
        out_path = out_dir / "live_features.csv"
    else:
        full = full[full["home_runs"].notna() & full["away_runs"].notna()].copy()
        out_path = out_dir / "full_features.csv"

    full.to_csv(out_path, index=False)
    lin = lineage.frame()
    # Expand lineage to the exact home_/away_ column names used by the model.
    base_map = {r["feature"]: r for _, r in lin.iterrows()}
    expanded = []
    for col in full.columns:
        if col in base_map:
            continue
        side = None
        base = col
        if col.startswith("home_"):
            side, base = "home", col[5:]
        elif col.startswith("away_"):
            side, base = "away", col[5:]
        if side and base in base_map:
            rec = dict(base_map[base])
            rec["feature"] = col
            rec["transformation"] = f"{rec['transformation']}; assigned to {side} team side"
            expanded.append(rec)
        elif col in {"home_team", "away_team", "home_opp", "away_opp"}:
            expanded.append({
                "feature": col, "source_file": "{season}teamstats.csv",
                "source_columns": "team|opp|vishome", "transformation": "direct_current team identity",
                "timing_rule": "direct_current", "feature_group": "identity", "notes": "categorical team identity"
            })
    if expanded:
        lin = pd.concat([lin, pd.DataFrame(expanded)], ignore_index=True).drop_duplicates("feature")
    lin.to_csv(out_dir / "feature_lineage.csv", index=False)

    # A compact manifest makes downstream scripts deterministic.
    target_cols = [c for c in ["home_runs", "away_runs", "home_win"] if c in full.columns]
    id_cols = [c for c in ["gid", "date", "season", "visteam", "hometeam"] if c in full.columns]
    features = [c for c in full.columns if c not in set(target_cols + id_cols)]
    manifest = {
        "mode": args.mode,
        "seasons": args.seasons,
        "windows": args.windows,
        "rows": int(len(full)),
        "columns": int(full.shape[1]),
        "id_columns": id_cols,
        "target_columns": target_cols,
        "feature_columns": features,
    }
    (out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(full):,} rows x {full.shape[1]:,} columns -> {out_path.resolve()}")
    print(f"Wrote {len(lin):,} lineage rows -> {(out_dir / 'feature_lineage.csv').resolve()}")


if __name__ == "__main__":
    main()

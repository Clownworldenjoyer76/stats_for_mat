#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

BASKETBALL_ROOT = Path(__file__).resolve().parents[1]
STAKING_CONFIG_PATH = BASKETBALL_ROOT / 'config' / 'staking.yaml'
MODEL_CONFIG_PATH = BASKETBALL_ROOT / 'config' / 'model_config.yaml'
LEAGUES = ('nba', 'ncaam', 'wnba')
MARKETS = ('moneyline', 'spread', 'total')


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'YAML root must be a mapping: {path}')
    return payload


STAKING_CONFIG = _read_yaml(STAKING_CONFIG_PATH)
MODEL_CONFIG = _read_yaml(MODEL_CONFIG_PATH)

KELLY_CFG = STAKING_CONFIG.get('kelly') or {}
EXPOSURE_CFG = STAKING_CONFIG.get('exposure_limits') or {}
UNCERTAINTY_CFG = STAKING_CONFIG.get('uncertainty_adjustment') or {}
RANKING_CFG = STAKING_CONFIG.get('ranking') or {}

KELLY_FRACTION = float(KELLY_CFG['fractional_multiplier'])
KELLY_CAP = float(EXPOSURE_CFG['maximum_individual_bet_kelly_fraction'])
MAX_EXPOSURE_PER_GAME = float(EXPOSURE_CFG['maximum_exposure_per_game'])
MAX_EXPOSURE_PER_LEAGUE_DAY = float(EXPOSURE_CFG['maximum_exposure_per_league_per_day'])
MAX_TOTAL_DAILY_EXPOSURE = float(EXPOSURE_CFG['maximum_total_daily_exposure'])
UNCERTAINTY_METHOD = str(UNCERTAINTY_CFG.get('method', '')).strip()
UNCERTAINTY_VERSION = str(UNCERTAINTY_CFG.get('version', '')).strip()
UNCERTAINTY_SOURCE = str(UNCERTAINTY_CFG.get('uncertainty_source', '')).strip()
STAKE_SCALING = str(UNCERTAINTY_CFG.get('stake_scaling', '')).strip()
RANKING_PRIMARY = str(RANKING_CFG.get('primary', '')).strip()
RANKING_TIE_BREAKERS = [str(v).strip() for v in (RANKING_CFG.get('tie_breakers') or [])]

for name, value in {
    'kelly.fractional_multiplier': KELLY_FRACTION,
    'maximum_individual_bet_kelly_fraction': KELLY_CAP,
    'maximum_exposure_per_game': MAX_EXPOSURE_PER_GAME,
    'maximum_exposure_per_league_per_day': MAX_EXPOSURE_PER_LEAGUE_DAY,
    'maximum_total_daily_exposure': MAX_TOTAL_DAILY_EXPOSURE,
}.items():
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f'staking.yaml {name} must be between 0 and 1')

if not (MAX_EXPOSURE_PER_GAME <= MAX_EXPOSURE_PER_LEAGUE_DAY <= MAX_TOTAL_DAILY_EXPOSURE):
    raise ValueError('staking exposure limits must satisfy per_game <= per_league_day <= total_day')
if UNCERTAINTY_METHOD != 'signal_to_noise_market_shrink':
    raise ValueError(f'Unsupported uncertainty method={UNCERTAINTY_METHOD!r}')
if STAKE_SCALING not in {'uncertainty_multiplier', 'none', 'raw'}:
    raise ValueError(f'Unsupported uncertainty stake_scaling={STAKE_SCALING!r}')
if RANKING_PRIMARY != 'uncertainty_adjusted_ev':
    raise ValueError('staking ranking.primary must be uncertainty_adjusted_ev')


def fv(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or str(value).strip() == '':
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _league_key(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        league = str(explicit).strip().lower()
    else:
        values = []
        for column in ('league_lower', 'league'):
            if column in df.columns:
                values.extend(
                    str(v).strip().lower()
                    for v in df[column].dropna().unique().tolist()
                    if str(v).strip()
                )
        unique = sorted(set(values))
        if len(unique) != 1:
            raise ValueError(f'Cannot resolve one league for staking uncertainty; values={unique}')
        league = unique[0]
    if league not in LEAGUES:
        raise ValueError(f'Unsupported league={league!r}')
    return league


def _oos_uncertainty_points(league: str, market: str) -> float:
    sources = UNCERTAINTY_CFG.get('market_std_source') or {}
    std_market = str(sources.get(market, '')).strip().lower()
    if std_market not in {'spread', 'total'}:
        raise ValueError(f'Invalid market_std_source.{market}={std_market!r}')
    try:
        value = float(MODEL_CONFIG['leagues'][league]['std'][std_market]['value'])
    except Exception as exc:
        raise ValueError(
            f'Missing OOS uncertainty: model_config.leagues.{league}.std.{std_market}.value'
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid OOS uncertainty {league}.{std_market}={value!r}')
    return value


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce')


def _signal_points(df: pd.DataFrame, market: str) -> pd.Series:
    margin = _numeric(df, 'home_projected_points') - _numeric(df, 'away_projected_points')
    if market == 'moneyline':
        return margin.abs()
    if market == 'spread':
        return (margin + _numeric(df, 'home_spread')).abs()
    if market == 'total':
        return (_numeric(df, 'total_projected_points') - _numeric(df, 'total')).abs()
    raise ValueError(f'Unsupported market={market!r}')


def add_uncertainty_adjusted_ev(
    df: pd.DataFrame,
    market: str,
    side_specs: list[tuple[str, str, str, str, str]],
    league: str | None = None,
) -> pd.DataFrame:
    out = df
    league_key = _league_key(out, league)
    uncertainty_points = _oos_uncertainty_points(league_key, market)
    signal = _signal_points(out, market)
    denom = np.sqrt(np.square(signal.to_numpy(float)) + uncertainty_points ** 2)
    multiplier = np.divide(
        signal.to_numpy(float),
        denom,
        out=np.zeros(len(out), dtype=float),
        where=np.isfinite(denom) & (denom > 0),
    )
    if not bool(UNCERTAINTY_CFG.get('enabled', True)):
        multiplier = np.ones(len(out), dtype=float)

    for prefix, raw_prob_col, market_prob_col, decimal_col, raw_kelly_col in side_specs:
        raw_prob = _numeric(out, raw_prob_col)
        market_prob = _numeric(out, market_prob_col)
        decimal = _numeric(out, decimal_col)
        if league_key == 'nba' and market == 'moneyline':
            adjusted_prob = raw_prob.clip(0.001, 0.999)
        else:
            adjusted_prob = (market_prob + multiplier * (raw_prob - market_prob)).clip(0.001, 0.999)
        adjusted_ev = adjusted_prob * decimal - 1.0
        raw_ev_col = f'{prefix}_ev'
        out[f'{prefix}_raw_ev'] = out[raw_ev_col]
        out[f'{prefix}_raw_kelly'] = out[raw_kelly_col]
        out[f'{prefix}_uncertainty_adjusted_ev'] = adjusted_ev
        out[f'{prefix}_adjusted_model_prob'] = adjusted_prob
        out[f'{prefix}_uncertainty_multiplier'] = multiplier
        out[f'{prefix}_uncertainty_points'] = uncertainty_points
        out[f'{prefix}_signal_points'] = signal

    out['uncertainty_adjustment_method'] = UNCERTAINTY_METHOD
    out['uncertainty_adjustment_version'] = UNCERTAINTY_VERSION
    out['uncertainty_source'] = UNCERTAINTY_SOURCE
    return out


def attach_candidate_uncertainty(row: Any, market: str, candidate: dict) -> dict:
    side = candidate['side']
    prefix = f'{side}_ml' if market == 'moneyline' else f'{side}_spread' if market == 'spread' else side
    adjusted_ev = fv(row.get(f'{prefix}_uncertainty_adjusted_ev'))
    adjusted_prob = fv(row.get(f'{prefix}_adjusted_model_prob'))
    multiplier = fv(row.get(f'{prefix}_uncertainty_multiplier'))
    uncertainty_points = fv(row.get(f'{prefix}_uncertainty_points'))
    signal_points = fv(row.get(f'{prefix}_signal_points'))
    out = dict(candidate)
    out.update({
        'raw_ev': candidate.get('ev'),
        'raw_kelly': candidate.get('kelly'),
        'uncertainty_adjusted_ev': candidate.get('ev') if adjusted_ev is None else adjusted_ev,
        'adjusted_model_prob': candidate.get('model_prob') if adjusted_prob is None else adjusted_prob,
        'uncertainty_multiplier': 1.0 if multiplier is None else multiplier,
        'uncertainty_points': uncertainty_points,
        'signal_points': signal_points,
    })
    return out


def requested_stake(raw_kelly: Any, uncertainty_multiplier: Any = 1.0) -> tuple[float, float, float]:
    kelly = fv(raw_kelly)
    multiplier = fv(uncertainty_multiplier)
    if kelly is None or kelly <= 0:
        return 0.0, 0.0, 0.0
    if multiplier is None:
        multiplier = 1.0
    multiplier = min(max(multiplier, 0.0), 1.0)
    fractional = max(kelly * KELLY_FRACTION, 0.0)
    individual_capped = min(fractional, KELLY_CAP)
    requested = individual_capped * multiplier if STAKE_SCALING == 'uncertainty_multiplier' else individual_capped
    return fractional, individual_capped, requested


RANK_COLUMN_MAP = {
    'uncertainty_adjusted_ev': 'bet_uncertainty_adjusted_ev',
    'raw_ev': 'bet_raw_ev',
    'raw_kelly': 'bet_raw_kelly',
    'model_prob': 'bet_model_prob',
}


def apply_exposure_limits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    rank_names = [RANKING_PRIMARY, *RANKING_TIE_BREAKERS]
    rank_cols = []
    for name in rank_names:
        col = RANK_COLUMN_MAP.get(name)
        if col is None:
            raise ValueError(f'Unsupported staking ranking field={name!r}')
        if col not in rank_cols:
            rank_cols.append(col)
    for col in rank_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['_rank_date'] = out['game_date'].astype(str)
    out = out.sort_values(
        ['_rank_date', *rank_cols, 'game_id', 'market_type', 'bet_side'],
        ascending=[True, *([False] * len(rank_cols)), True, True, True],
        kind='stable',
        na_position='last',
    ).reset_index(drop=True)

    game_used = defaultdict(float)
    league_day_used = defaultdict(float)
    day_used = defaultdict(float)
    day_rank = defaultdict(int)
    final_rows = []
    eps = 1e-12

    for _, row in out.iterrows():
        raw_kelly = fv(row.get('bet_raw_kelly'))
        if raw_kelly is None:
            raw_kelly = fv(row.get('bet_kelly'))
        multiplier = fv(row.get('bet_uncertainty_multiplier'))
        if multiplier is None:
            multiplier = 1.0
        fractional, individual_capped, requested = requested_stake(raw_kelly, multiplier)

        league = str(row.get('league_lower') or row.get('league') or '').strip().lower()
        game_date = str(row.get('game_date') or '').strip()
        game_id = str(row.get('game_id') or '').strip()
        game_key = (league, game_id)
        league_day_key = (league, game_date)
        day_key = game_date

        remaining_game = max(MAX_EXPOSURE_PER_GAME - game_used[game_key], 0.0)
        remaining_league = max(MAX_EXPOSURE_PER_LEAGUE_DAY - league_day_used[league_day_key], 0.0)
        remaining_day = max(MAX_TOTAL_DAILY_EXPOSURE - day_used[day_key], 0.0)
        final_stake = max(min(requested, remaining_game, remaining_league, remaining_day), 0.0)

        reasons = []
        if fractional > KELLY_CAP + eps:
            reasons.append('individual_kelly_cap')
        if requested > remaining_game + eps:
            reasons.append('game_exposure_cap')
        if requested > remaining_league + eps:
            reasons.append('league_day_exposure_cap')
        if requested > remaining_day + eps:
            reasons.append('total_daily_exposure_cap')
        if final_stake <= eps:
            continue

        game_used[game_key] += final_stake
        league_day_used[league_day_key] += final_stake
        day_used[day_key] += final_stake
        day_rank[day_key] += 1

        r = row.to_dict()
        r.update({
            'bet_fractional_kelly_pct': fractional,
            'bet_individual_capped_stake_pct': individual_capped,
            'bet_requested_stake_pct': requested,
            'bet_final_stake_pct': final_stake,
            'bet_stake_pct': final_stake,
            'exposure_rank': day_rank[day_key],
            'exposure_limited': bool(reasons or final_stake + eps < requested),
            'exposure_limit_reason': ';'.join(reasons),
            'game_exposure_after_pct': game_used[game_key],
            'league_day_exposure_after_pct': league_day_used[league_day_key],
            'total_day_exposure_after_pct': day_used[day_key],
            'maximum_exposure_per_game': MAX_EXPOSURE_PER_GAME,
            'maximum_exposure_per_league_per_day': MAX_EXPOSURE_PER_LEAGUE_DAY,
            'maximum_total_daily_exposure': MAX_TOTAL_DAILY_EXPOSURE,
            'maximum_individual_bet_kelly_fraction': KELLY_CAP,
            'uncertainty_adjustment_method': UNCERTAINTY_METHOD,
            'uncertainty_adjustment_version': UNCERTAINTY_VERSION,
        })
        final_rows.append(r)

    if not final_rows:
        return out.iloc[0:0].drop(columns=['_rank_date'], errors='ignore')
    final = pd.DataFrame(final_rows)
    return final.drop(columns=['_rank_date'], errors='ignore')

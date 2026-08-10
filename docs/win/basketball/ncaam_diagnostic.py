#!/usr/bin/env python3
# ncaam_diagnostic.py
#
# Phase 1: Find NCAAM bet slices with positive ROI.
# Reads ev_kelly combined files + final scores. Outputs ROI per slice
# across multiple feature dimensions for each market.

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# =========================
# LOAD
# =========================

scores = pd.read_csv('/mnt/user-data/uploads/combined_final_scores_all.csv')
scores['home_score'] = pd.to_numeric(scores['home_score'], errors='coerce')
scores['away_score'] = pd.to_numeric(scores['away_score'], errors='coerce')
scores = scores.dropna(subset=['home_score','away_score']).drop_duplicates(subset=['game_id'], keep='first')
sm = scores.set_index('game_id')[['home_score','away_score']].to_dict('index')

ml = pd.read_csv('/mnt/user-data/uploads/moneyline_ev_kelly_combined.csv')
sp = pd.read_csv('/mnt/user-data/uploads/spread_ev_kelly_combined.csv')
tt = pd.read_csv('/mnt/user-data/uploads/total_ev_kelly_combined.csv')

def lg_norm(s): return s.lower() if isinstance(s,str) else s
for d in (ml, sp, tt):
    d['league_n'] = d['league'].apply(lg_norm)

ml = ml[ml['league_n']=='ncaam'].copy()
sp = sp[sp['league_n']=='ncaam'].copy()
tt = tt[tt['league_n']=='ncaam'].copy()

# =========================
# PARSE GAME DATE
# =========================

def parse_date(s):
    try: return datetime.strptime(s, '%Y_%m_%d')
    except: return pd.NaT

for d in (ml, sp, tt):
    d['gd'] = d['game_date'].apply(parse_date)
    d['month'] = d['gd'].dt.month
    d['dow'] = d['gd'].dt.dayofweek  # Mon=0
    # Season segment
    def seg(m):
        if pd.isna(m): return 'unknown'
        if m in (11, 12): return 'early (Nov-Dec)'
        if m in (1, 2):   return 'mid (Jan-Feb)'
        if m in (3, 4):   return 'late (Mar-Apr)'
        return 'offseason'
    d['season_seg'] = d['month'].apply(seg)

# =========================
# COLLECT BETS
# =========================

def collect(df, market):
    bets = []
    for _, r in df.iterrows():
        gid = r['game_id']
        if gid not in sm: continue
        sc = sm[gid]
        actual_total = sc['home_score'] + sc['away_score']
        actual_margin = sc['home_score'] - sc['away_score']

        if market == 'ML':
            home_won = 1 if sc['home_score']>sc['away_score'] else 0
            for side in ('home','away'):
                ev = pd.to_numeric(r.get(f'{side}_ml_ev'), errors='coerce')
                k  = pd.to_numeric(r.get(f'{side}_ml_kelly'), errors='coerce')
                d  = pd.to_numeric(r.get(f'{side}_dk_moneyline_decimal'), errors='coerce')
                if pd.notna(k) and pd.notna(ev) and ev > 0 and pd.notna(d):
                    won = home_won if side=='home' else 1-home_won
                    bets.append({
                        'side': side, 'won': won, 'kelly': k, 'decimal': d, 'ev': ev,
                        'season_seg': r['season_seg'], 'month': r['month'], 'dow': r['dow'],
                        'book_total': pd.to_numeric(r.get('total'), errors='coerce'),
                        'home_prob': pd.to_numeric(r.get('home_prob'), errors='coerce'),
                        'edge_vs_market': pd.to_numeric(r.get(f'{side}_ml_edge_vs_market'), errors='coerce'),
                    })
        elif market == 'SPREAD':
            spread = pd.to_numeric(r.get('home_spread'), errors='coerce')
            if pd.isna(spread): continue
            cover_th = -spread
            if actual_margin == cover_th: continue
            home_won = 1 if actual_margin > cover_th else 0
            for side in ('home','away'):
                ev = pd.to_numeric(r.get(f'{side}_spread_ev'), errors='coerce')
                k  = pd.to_numeric(r.get(f'{side}_spread_kelly'), errors='coerce')
                d  = pd.to_numeric(r.get(f'{side}_dk_spread_decimal'), errors='coerce')
                if pd.notna(k) and pd.notna(ev) and ev > 0 and pd.notna(d):
                    won = home_won if side=='home' else 1-home_won
                    bets.append({
                        'side': side, 'won': won, 'kelly': k, 'decimal': d, 'ev': ev,
                        'season_seg': r['season_seg'], 'month': r['month'], 'dow': r['dow'],
                        'book_spread_abs': abs(spread),
                        'book_total': pd.to_numeric(r.get('total'), errors='coerce'),
                        'edge_vs_market': pd.to_numeric(r.get(f'{side}_spread_edge_vs_market'), errors='coerce'),
                    })
        else:  # TOTAL
            book_total = pd.to_numeric(r.get('total'), errors='coerce')
            if pd.isna(book_total) or actual_total == book_total: continue
            over_won = 1 if actual_total > book_total else 0
            for side in ('over','under'):
                ev = pd.to_numeric(r.get(f'{side}_ev'), errors='coerce')
                k  = pd.to_numeric(r.get(f'{side}_kelly'), errors='coerce')
                d  = pd.to_numeric(r.get(f'dk_total_{side}_decimal'), errors='coerce')
                if pd.notna(k) and pd.notna(ev) and ev > 0 and pd.notna(d):
                    won = over_won if side=='over' else 1-over_won
                    bets.append({
                        'side': side, 'won': won, 'kelly': k, 'decimal': d, 'ev': ev,
                        'season_seg': r['season_seg'], 'month': r['month'], 'dow': r['dow'],
                        'book_total': book_total,
                        'edge_vs_market': pd.to_numeric(r.get(f'{side}_edge_vs_market'), errors='coerce'),
                    })
    df_bets = pd.DataFrame(bets)
    if len(df_bets):
        df_bets['profit'] = np.where(df_bets['won']==1, df_bets['decimal']-1, -1.0)
    return df_bets

ml_b = collect(ml, 'ML')
sp_b = collect(sp, 'SPREAD')
tt_b = collect(tt, 'TOTAL')

# =========================
# REPORT HELPER
# =========================

def report(label, b, by, min_n=80):
    if len(b)==0:
        print(f"\n{label}: no bets"); return
    g = b.groupby(by, observed=True).agg(
        n=('won','size'), wr=('won','mean'),
        roi=('profit','mean'), pnl=('profit','sum'),
    )
    g = g[g['n'] >= min_n].round(4).sort_values('roi', ascending=False)
    if len(g)==0:
        print(f"\n{label}: no slices with n >= {min_n}"); return
    print(f"\n{label} (min n={min_n})")
    print(g.to_string())

def report_two(label, b, by1, by2, min_n=80):
    if len(b)==0: return
    g = b.groupby([by1, by2], observed=True).agg(
        n=('won','size'), wr=('won','mean'),
        roi=('profit','mean'), pnl=('profit','sum'),
    )
    g = g[g['n'] >= min_n].round(4).sort_values('roi', ascending=False)
    if len(g)==0:
        print(f"\n{label}: no slices with n >= {min_n}"); return
    print(f"\n{label} (min n={min_n})")
    print(g.to_string())

# =========================
# RUN — MONEYLINE
# =========================
print("="*78); print("NCAAM MONEYLINE SLICES"); print("="*78)
print(f"\nTotal +EV ML bets: {len(ml_b)}")
if len(ml_b):
    print(f"Overall ROI: {ml_b['profit'].mean():+.4f}  WR: {ml_b['won'].mean():.4f}")
    report("By side", ml_b, 'side', min_n=200)
    report("By season segment", ml_b, 'season_seg', min_n=200)
    report("By month", ml_b, 'month', min_n=100)
    report("By day of week (Mon=0)", ml_b, 'dow', min_n=150)
    ml_b['model_prob_bin'] = pd.cut(ml_b['home_prob'], bins=[0,.3,.4,.5,.6,.7,1.0],
                                     labels=['<30','30-40','40-50','50-60','60-70','70+'])
    report("By model home_prob bin", ml_b, 'model_prob_bin', min_n=150)
    ml_b['edge_bin'] = pd.cut(ml_b['edge_vs_market'], bins=[0,.01,.02,.03,.05,.10,1.0],
                              labels=['0-1%','1-2%','2-3%','3-5%','5-10%','10%+'])
    report("By edge_vs_market", ml_b, 'edge_bin', min_n=150)
    report_two("By season_seg × side", ml_b, 'season_seg', 'side', min_n=100)

# =========================
# RUN — SPREAD
# =========================
print("\n" + "="*78); print("NCAAM SPREAD SLICES"); print("="*78)
print(f"\nTotal +EV spread bets: {len(sp_b)}")
if len(sp_b):
    print(f"Overall ROI: {sp_b['profit'].mean():+.4f}  WR: {sp_b['won'].mean():.4f}")
    report("By side", sp_b, 'side', min_n=200)
    report("By season segment", sp_b, 'season_seg', min_n=200)
    report("By month", sp_b, 'month', min_n=100)
    report("By day of week (Mon=0)", sp_b, 'dow', min_n=150)
    sp_b['spread_mag'] = pd.cut(sp_b['book_spread_abs'], bins=[0,3,6,10,15,100],
                                labels=['0-3','3-6','6-10','10-15','15+'])
    report("By spread magnitude (|home_spread|)", sp_b, 'spread_mag', min_n=150)
    sp_b['edge_bin'] = pd.cut(sp_b['edge_vs_market'], bins=[0,.005,.01,.02,.05,.10,1.0],
                              labels=['0-0.5%','0.5-1%','1-2%','2-5%','5-10%','10%+'])
    report("By edge_vs_market", sp_b, 'edge_bin', min_n=150)
    report_two("By season_seg × side", sp_b, 'season_seg', 'side', min_n=100)
    report_two("By spread_mag × side", sp_b, 'spread_mag', 'side', min_n=100)

# =========================
# RUN — TOTAL
# =========================
print("\n" + "="*78); print("NCAAM TOTAL SLICES"); print("="*78)
print(f"\nTotal +EV total bets: {len(tt_b)}")
if len(tt_b):
    print(f"Overall ROI: {tt_b['profit'].mean():+.4f}  WR: {tt_b['won'].mean():.4f}")
    report("By side", tt_b, 'side', min_n=200)
    report("By season segment", tt_b, 'season_seg', min_n=200)
    report("By month", tt_b, 'month', min_n=100)
    tt_b['total_bin'] = pd.cut(tt_b['book_total'], bins=[0,125,135,145,155,300],
                               labels=['<125','125-135','135-145','145-155','155+'])
    report("By book total bin", tt_b, 'total_bin', min_n=150)
    tt_b['edge_bin'] = pd.cut(tt_b['edge_vs_market'], bins=[0,.005,.01,.02,.05,.10,1.0],
                              labels=['0-0.5%','0.5-1%','1-2%','2-5%','5-10%','10%+'])
    report("By edge_vs_market", tt_b, 'edge_bin', min_n=150)
    report_two("By season_seg × side", tt_b, 'season_seg', 'side', min_n=100)
    report_two("By total_bin × side", tt_b, 'total_bin', 'side', min_n=100)

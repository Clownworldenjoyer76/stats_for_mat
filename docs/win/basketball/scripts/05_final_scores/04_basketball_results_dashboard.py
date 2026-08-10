#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/04_basketball_results_dashboard.py
#
# Builds a single self-contained HTML dashboard combining all three leagues
# (NBA, NCAAM, WNBA) into one file with a league switcher at the top.
#
# Inputs (per league nba, ncaam, wnba):
#   docs/win/basketball/05_final_scores/{league}_summary_grand_total.csv
#   docs/win/basketball/05_final_scores/{league}_summary_overall.csv
#   docs/win/basketball/05_final_scores/reports/{league}/{moneyline,spread,total,overview}/*.csv
#
# Output (single file):
#   docs/basketball_dashboard.html
#
# No server required — open the file in a browser.

from datetime import datetime, UTC
from pathlib import Path
import html
import json

import pandas as pd

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

BASE        = Path("docs/win/basketball/05_final_scores")
REPORT_DIR  = BASE / "reports"
OUTPUT_FILE = Path("docs/basketball_dashboard.html")


# =========================
# CSS
# =========================

CSS = """
:root {
  --bg: #0e1117;
  --panel: #161b22;
  --panel-2: #1c232c;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --good: #3fb950;
  --bad:  #f85149;
  --border: #30363d;
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; }
header { padding: 18px 24px; border-bottom: 1px solid var(--border); display:flex; justify-content:space-between; align-items:baseline; flex-wrap: wrap; gap: 12px; }
header h1 { margin: 0; font-size: 20px; }
header .ts { color: var(--muted); font-size: 12px; }
.league-bar { padding: 10px 24px; border-bottom: 1px solid var(--border); background: var(--panel); display: flex; gap: 6px; flex-wrap: wrap; align-items: center; position: sticky; top: 0; z-index: 10; }
.league-bar .lbl { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; margin-right: 6px; }
.league-btn { background: var(--panel-2); border: 1px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.league-btn.active { background: var(--accent); color: #0e1117; border-color: var(--accent); font-weight: 600; }
main { padding: 18px 24px; max-width: 1500px; margin: 0 auto; }
.league-section { display: none; }
.league-section.active { display: block; }
h2 { font-size: 16px; margin: 24px 0 8px 0; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
h3 { font-size: 13px; margin: 14px 0 6px 0; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0 4px 0; }
.kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 22px; margin-top: 4px; font-weight: 600; }
.kpi .value.good { color: var(--good); }
.kpi .value.bad  { color: var(--bad); }
.tabs { display:flex; gap: 4px; margin: 16px 0 0 0; flex-wrap: wrap; }
.tab { background: var(--panel); border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px 6px 0 0; cursor: pointer; color: var(--muted); font-size: 13px; }
.tab.active { background: var(--panel-2); color: var(--text); border-bottom-color: var(--panel-2); }
.tab-body { background: var(--panel-2); border: 1px solid var(--border); border-top: none; padding: 14px; border-radius: 0 6px 6px 6px; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
.controls label { color: var(--muted); font-size: 12px; }
.controls select { background: var(--panel); color: var(--text); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; font-size: 13px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--panel-2); }
th .arrow { opacity: 0.4; margin-left: 4px; }
th.sorted .arrow { opacity: 1; color: var(--accent); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: var(--good); }
.neg { color: var(--bad); }
.muted { color: var(--muted); }
.scroll { max-height: 60vh; overflow: auto; border: 1px solid var(--border); border-radius: 6px; }
footer { color: var(--muted); font-size: 11px; padding: 16px 24px; }
"""

# =========================
# JS
# =========================

JS = """
function fmtPct(v) { if (v == null || isNaN(v)) return ''; return (v*100).toFixed(2) + '%'; }
function fmtNum(v, d) { if (v == null || isNaN(v)) return ''; return Number(v).toFixed(d); }
function fmtInt(v) { if (v == null || isNaN(v)) return ''; return Number(v).toLocaleString(); }
function classForSigned(v) { if (v == null || isNaN(v)) return ''; return v > 0 ? 'pos' : (v < 0 ? 'neg' : ''); }

function showTab(host, key) {
  host.querySelectorAll(':scope > .tabs .tab').forEach(el => el.classList.toggle('active', el.dataset.key === key));
  host.querySelectorAll(':scope > .tab-body > .tab-panel').forEach(el => el.style.display = (el.dataset.key === key ? '' : 'none'));
}

function renderTable(data, columns, container) {
  if (!data || data.length === 0) { container.innerHTML = '<div class="muted">No rows.</div>'; return; }
  const wrap = document.createElement('div'); wrap.className = 'scroll';
  const tbl = document.createElement('table');
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  let sortCol = null, sortDir = 'desc';
  columns.forEach((c) => {
    const th = document.createElement('th');
    th.innerHTML = c.label + ' <span class="arrow">&#9662;</span>';
    th.onclick = () => {
      if (sortCol === c.key) sortDir = (sortDir === 'asc' ? 'desc' : 'asc');
      else { sortCol = c.key; sortDir = 'desc'; }
      sortAndRender();
      thead.querySelectorAll('th').forEach(h => h.classList.remove('sorted'));
      th.classList.add('sorted');
    };
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const tbody = document.createElement('tbody');
  function sortAndRender() {
    const sorted = data.slice();
    if (sortCol) {
      sorted.sort((a,b) => {
        const av = a[sortCol], bv = b[sortCol];
        if (av == null) return 1; if (bv == null) return -1;
        if (typeof av === 'number' && typeof bv === 'number') return sortDir === 'asc' ? av-bv : bv-av;
        return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
    }
    tbody.innerHTML = '';
    sorted.forEach(row => {
      const tr = document.createElement('tr');
      columns.forEach(c => {
        const td = document.createElement('td');
        let v = row[c.key];
        let cls = '';
        if (c.fmt === 'int')      { td.classList.add('num'); td.textContent = fmtInt(v); }
        else if (c.fmt === 'pct') { td.classList.add('num'); td.textContent = fmtPct(v); cls = c.color ? classForSigned(v) : ''; }
        else if (c.fmt === 'num') { td.classList.add('num'); td.textContent = fmtNum(v, c.decimals != null ? c.decimals : 2); cls = c.color ? classForSigned(v) : ''; }
        else if (c.fmt === 'roi') { td.classList.add('num'); td.textContent = fmtPct(v); cls = classForSigned(v); }
        else                      { td.textContent = (v == null ? '' : String(v)); }
        if (cls) td.classList.add(cls);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
  sortAndRender();
  tbl.appendChild(thead); tbl.appendChild(tbody);
  wrap.appendChild(tbl);
  container.innerHTML = ''; container.appendChild(wrap);
}

const STD_COLUMNS = [
  { key: 'bucket', label: 'Bucket' },
  { key: 'bets', label: 'Bets', fmt: 'int' },
  { key: 'wins', label: 'W', fmt: 'int' },
  { key: 'losses', label: 'L', fmt: 'int' },
  { key: 'pushes', label: 'P', fmt: 'int' },
  { key: 'win_pct', label: 'Win %', fmt: 'pct' },
  { key: 'units_flat', label: 'Units (flat)', fmt: 'num', decimals: 2, color: true },
  { key: 'roi_flat', label: 'ROI flat', fmt: 'roi' },
  { key: 'units_kelly', label: 'Units (Kelly)', fmt: 'num', decimals: 4, color: true },
  { key: 'roi_kelly', label: 'ROI Kelly', fmt: 'roi' },
  { key: 'avg_ev', label: 'Avg EV', fmt: 'pct' },
  { key: 'avg_edge_vs_market_pp', label: 'Avg edge (pp)', fmt: 'num', decimals: 2 },
  { key: 'avg_kelly_pct', label: 'Avg Kelly', fmt: 'pct' },
  { key: 'avg_model_prob', label: 'Avg model p', fmt: 'pct' },
  { key: 'avg_odds_american', label: 'Avg odds', fmt: 'num', decimals: 0 },
];
const STD_COLUMNS_WITH_SIDE = [
  { key: 'side_group', label: 'Side' },
  ...STD_COLUMNS,
];

function selectLeague(lg) {
  document.querySelectorAll('.league-btn').forEach(b => b.classList.toggle('active', b.dataset.league === lg));
  document.querySelectorAll('.league-section').forEach(s => s.classList.toggle('active', s.dataset.league === lg));
  try { localStorage.setItem('basketball_dash_league', lg); } catch (e) {}
}

function buildLeagueSection(lg, data) {
  const section = document.querySelector('.league-section[data-league="' + lg + '"]');
  if (!section) return;

  // KPIs
  const gt = data.grand_total || {};
  const kpiHtml = (label, value, fmt) => {
    let disp = '\\u2014', cls = '';
    if (value !== null && value !== undefined && !(typeof value === 'number' && isNaN(value))) {
      if (fmt === 'pct')         { disp = (value*100).toFixed(2) + '%'; cls = value > 0 ? 'good' : (value < 0 ? 'bad' : ''); }
      else if (fmt === 'int')    { disp = Number(value).toLocaleString(); }
      else if (fmt === 'signed') { disp = (value >= 0 ? '+' : '') + Number(value).toFixed(2); cls = value > 0 ? 'good' : (value < 0 ? 'bad' : ''); }
      else                       { disp = String(value); }
    }
    return '<div class="kpi"><div class="label">' + label + '</div><div class="value ' + cls + '">' + disp + '</div></div>';
  };

  const kpis = [
    kpiHtml('Bets',          gt.bets,         'int'),
    kpiHtml('Wins',          gt.wins,         'int'),
    kpiHtml('Losses',        gt.losses,       'int'),
    kpiHtml('Pushes',        gt.pushes,       'int'),
    kpiHtml('Win %',         gt.win_pct,      'pct'),
    kpiHtml('Units (flat)',  gt.units_flat,   'signed'),
    kpiHtml('ROI flat',      gt.roi_flat,     'pct'),
    kpiHtml('Units (Kelly)', gt.units_kelly,  'signed'),
    kpiHtml('ROI Kelly',     gt.roi_kelly,    'pct'),
  ].join('');
  section.querySelector('.kpis').innerHTML = kpis;

  // By-market summary table
  const byMarketCols = [
    { key: 'market_type', label: 'Market' },
    { key: 'Win', label: 'W', fmt: 'int' },
    { key: 'Loss', label: 'L', fmt: 'int' },
    { key: 'Push', label: 'P', fmt: 'int' },
    { key: 'Total', label: 'Total', fmt: 'int' },
    { key: 'Win_Pct', label: 'Win %', fmt: 'pct' },
  ];
  renderTable(data.by_market_summary, byMarketCols, section.querySelector('.by-market-summary'));

  // Per-market drilldown
  ['moneyline', 'spread', 'total'].forEach(mt => {
    const wrap = section.querySelector('.panel-' + mt);
    const md = (data.markets && data.markets[mt]) || {by:{}, by_side:{}};
    const dimensions = Object.keys(md.by);

    wrap.innerHTML =
      '<div class="controls">' +
      '<label>Dimension: <select class="dim-select">' +
      dimensions.map(d => '<option value="' + d + '">' + d + '</option>').join('') +
      '</select></label>' +
      '<label>View: <select class="view-select">' +
      '<option value="overall">Overall</option>' +
      '<option value="side">Split by side</option>' +
      '</select></label>' +
      '</div>' +
      '<div class="market-table"></div>';

    const dimSel = wrap.querySelector('.dim-select');
    const viewSel = wrap.querySelector('.view-select');
    const tbl = wrap.querySelector('.market-table');
    function refresh() {
      const dim = dimSel.value;
      const view = viewSel.value;
      const rows = view === 'side' ? (md.by_side[dim] || []) : (md.by[dim] || []);
      const cols = view === 'side' ? STD_COLUMNS_WITH_SIDE : STD_COLUMNS;
      renderTable(rows, cols, tbl);
    }
    dimSel.onchange = refresh;
    viewSel.onchange = refresh;
    refresh();
  });

  // Market tabs
  const marketArea = section.querySelector('.market-area');
  marketArea.querySelectorAll(':scope > .tabs .tab').forEach(t => {
    t.onclick = () => showTab(marketArea, t.dataset.key);
  });

  // Overview
  renderTable(data.overview.by_market, [
    { key: 'market_type', label: 'Market' },
    ...STD_COLUMNS.filter(c => c.key !== 'bucket'),
  ], section.querySelector('.overview-market'));
  renderTable(data.overview.by_side_group, [
    { key: 'side_group', label: 'Side' },
    ...STD_COLUMNS.filter(c => c.key !== 'bucket'),
  ], section.querySelector('.overview-side'));
  renderTable(data.overview.by_date, [
    { key: 'bucket', label: 'Date' },
    ...STD_COLUMNS.filter(c => c.key !== 'bucket'),
  ], section.querySelector('.overview-date'));

  const overviewArea = section.querySelector('.overview-area');
  overviewArea.querySelectorAll(':scope > .tabs .tab').forEach(t => {
    t.onclick = () => showTab(overviewArea, t.dataset.key);
  });
}
"""


# =========================
# DATA LOADING
# =========================

def df_to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    df = df.copy()
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_row_dict(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    rec = df.iloc[0].to_dict()
    return {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}


def collect_league_data(league: str) -> dict:
    data = {
        "league": league.upper(),
        "grand_total": first_row_dict(safe_read(BASE / f"{league}_summary_grand_total.csv")),
        "by_market_summary": df_to_records(safe_read(BASE / f"{league}_summary_overall.csv")),
        "overview": {
            "by_market":     df_to_records(safe_read(REPORT_DIR / league / "overview" / f"{league}_summary_by_market.csv")),
            "by_side_group": df_to_records(safe_read(REPORT_DIR / league / "overview" / f"{league}_summary_by_side_group.csv")),
            "by_date":       df_to_records(safe_read(REPORT_DIR / league / "overview" / f"{league}_summary_by_date.csv")),
        },
        "markets": {},
    }

    for mt in MARKETS:
        mt_dir = REPORT_DIR / league / mt
        market_data = {"by": {}, "by_side": {}}
        for label in ("ev", "kelly", "odds", "win_prob", "edge_vs_market", "dow", "month"):
            no_side = mt_dir / f"{league}_{mt}_by_{label}.csv"
            with_side_sfx = "over_under" if mt == "total" else "home_away"
            with_side = mt_dir / f"{league}_{mt}_by_{label}_{with_side_sfx}_summary.csv"
            market_data["by"][label]      = df_to_records(safe_read(no_side))
            market_data["by_side"][label] = df_to_records(safe_read(with_side))
        if mt in ("spread", "total"):
            with_side_sfx = "over_under" if mt == "total" else "home_away"
            market_data["by"]["side"]      = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_side.csv"))
            market_data["by_side"]["side"] = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_side_{with_side_sfx}_summary.csv"))
        if mt == "total":
            market_data["by"]["total_range"]      = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_total_range.csv"))
            market_data["by_side"]["total_range"] = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_total_range_over_under_summary.csv"))

        data["markets"][mt] = market_data

    return data


# =========================
# HTML BUILD
# =========================

def league_section_html(league: str) -> str:
    upper = league.upper()
    return f"""
<section class="league-section" data-league="{league}">
  <h2>{upper} &mdash; Headline</h2>
  <div class="kpis"></div>

  <h2>By Market</h2>
  <div class="by-market-summary"></div>

  <h2>Per Market Drilldown</h2>
  <div class="market-area">
    <div class="tabs">
      <div class="tab active" data-key="moneyline">Moneyline</div>
      <div class="tab" data-key="spread">Spread</div>
      <div class="tab" data-key="total">Total</div>
    </div>
    <div class="tab-body">
      <div class="tab-panel panel-moneyline" data-key="moneyline"></div>
      <div class="tab-panel panel-spread"    data-key="spread"    style="display:none"></div>
      <div class="tab-panel panel-total"     data-key="total"     style="display:none"></div>
    </div>
  </div>

  <h2>Overview</h2>
  <div class="overview-area">
    <div class="tabs">
      <div class="tab active" data-key="market">By market</div>
      <div class="tab" data-key="side">By side group</div>
      <div class="tab" data-key="date">By date</div>
    </div>
    <div class="tab-body">
      <div class="tab-panel overview-market" data-key="market"></div>
      <div class="tab-panel overview-side"   data-key="side"   style="display:none"></div>
      <div class="tab-panel overview-date"   data-key="date"   style="display:none"></div>
    </div>
  </div>
</section>
"""


def build_dashboard() -> str:
    ts = datetime.now(UTC).isoformat(timespec="seconds")

    payloads = {lg: collect_league_data(lg) for lg in LEAGUES}
    payload_json = json.dumps(payloads, default=str)

    league_buttons = "\n".join(
        f'<button class="league-btn{" active" if i == 0 else ""}" data-league="{lg}" '
        f'onclick="selectLeague(\'{lg}\')">{lg.upper()}</button>'
        for i, lg in enumerate(LEAGUES)
    )
    sections = "\n".join(league_section_html(lg) for lg in LEAGUES)

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Basketball Dashboard</title>
<style>{CSS}
.league-section[data-league="{LEAGUES[0]}"] {{ display: block; }}
</style></head>
<body>
<header>
  <h1>Basketball Dashboard</h1>
  <span class="ts">Built {html.escape(ts)} UTC</span>
</header>
<div class="league-bar">
  <span class="lbl">League:</span>
  {league_buttons}
</div>
<main>
{sections}
</main>
<footer>Click column headers to sort. Built from CSVs in <code>{html.escape(str(BASE))}</code>.</footer>

<script>
{JS}

const ALL_DATA = {payload_json};

document.addEventListener('DOMContentLoaded', () => {{
  Object.keys(ALL_DATA).forEach(lg => buildLeagueSection(lg, ALL_DATA[lg]));

  let initial = '{LEAGUES[0]}';
  try {{
    const stored = localStorage.getItem('basketball_dash_league');
    if (stored && ALL_DATA[stored]) initial = stored;
  }} catch (e) {{}}
  selectLeague(initial);
}});
</script>
</body></html>"""


def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    page = build_dashboard()
    OUTPUT_FILE.write_text(page, encoding="utf-8")
    print(f"dashboard -> {OUTPUT_FILE}")


if __name__ == "__main__":
    run()

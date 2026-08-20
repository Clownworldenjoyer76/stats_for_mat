#!/usr/bin/env python3
"""Incremental/full-rebuild launcher for basketball_select_bets_core.py.

Incremental mode rebuilds only today's candidate selection and preserves historical
`daily_picks`. Full rebuild mode retains the original behavior and is intended for
selection/model experiments across stored history.
"""
from __future__ import annotations
import importlib.util, os, shutil, sys, tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CORE_PATH=Path(__file__).with_name("basketball_select_bets_core.py")
REAL_INPUT=Path("docs/win/basketball/03_edges/ev_kelly")
REAL_SELECT=Path("docs/win/basketball/04_select")
NY=ZoneInfo("America/New_York")
LEAGUES=["nba","ncaam","wnba"]; MARKETS=["moneyline","spread","total"]
def truthy(v): return str(v or "").strip().lower() in {"1","true","yes","on"}
def load_core():
    spec=importlib.util.spec_from_file_location("basketball_select_bets_core",CORE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(f"Unable to load {CORE_PATH}")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    core=load_core()
    if truthy(os.getenv("BASKETBALL_FULL_REBUILD")):
        core.main()
    else:
        date=datetime.now(NY).strftime("%Y_%m_%d")
        with tempfile.TemporaryDirectory(prefix="basketball_select_") as td:
            root=Path(td); inp=root/"input"; out=root/"select"
            for lg in LEAGUES:
                up=lg.upper()
                for market in MARKETS:
                    src=REAL_INPUT/lg/market/f"{date}_{up}_{market}.csv"
                    if src.exists():
                        dst=inp/lg/market/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
            core.INPUT_DIR=inp; core.SELECT_DIR=out; core.main()
            for lg in LEAGUES:
                dest=REAL_SELECT/lg/"daily_picks"/f"{date}_{lg}_selected.csv"
                src=out/lg/"daily_picks"/dest.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                if src.exists(): shutil.copy2(src,dest)
                else: dest.unlink(missing_ok=True)
    text=core.LOG_FILE.read_text(encoding="utf-8",errors="replace")
    if "STATUS: FAILED" in text or "STATUS: COMPLETED WITH ERRORS" in text: sys.exit(1)
if __name__=="__main__": main()

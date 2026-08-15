#!/usr/bin/env python3
"""Incremental/full-rebuild launcher for build_juice_files_core.py."""
from __future__ import annotations
import importlib.util, os, shutil, sys, tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CORE_PATH = Path(__file__).with_name("build_juice_files_core.py")
REAL_INPUT = Path("docs/win/basketball/01_merge")
REAL_OUTPUT = Path("docs/win/basketball/01_merge/01_merguiced")
NY = ZoneInfo("America/New_York")
LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

def truthy(v): return str(v or "").strip().lower() in {"1","true","yes","on"}

def load_core():
    spec=importlib.util.spec_from_file_location("build_juice_files_core", CORE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(f"Unable to load {CORE_PATH}")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    core=load_core()
    if truthy(os.getenv("BASKETBALL_FULL_REBUILD")):
        core.main()
        text=core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
        if "STATUS: FAILED" in text or "ERROR processing" in text:
            sys.exit(1)
        return
    date=datetime.now(NY).strftime("%Y_%m_%d")
    with tempfile.TemporaryDirectory(prefix="basketball_juice_") as td:
        root=Path(td); inp=root/"input"; out=root/"output"
        for lg in LEAGUES:
            up=lg.upper()
            for market in MARKETS:
                src=REAL_INPUT/lg/market/f"{date}_{up}_{market}.csv"
                if src.exists():
                    dst=inp/lg/market/src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst)
        core.INPUT_DIR=inp; core.OUTPUT_DIR=out
        core.main()
        for lg in LEAGUES:
            up=lg.upper()
            for market in MARKETS:
                dest=REAL_OUTPUT/lg/market/f"{date}_{up}_{market}.csv"
                src=out/lg/market/dest.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.exists(): shutil.copy2(src,dest)
                else: dest.unlink(missing_ok=True)
    text=core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    if "STATUS: FAILED" in text or "STATUS: COMPLETED WITH ERRORS" in text or "ERROR processing" in text:
        sys.exit(1)

if __name__=="__main__": main()

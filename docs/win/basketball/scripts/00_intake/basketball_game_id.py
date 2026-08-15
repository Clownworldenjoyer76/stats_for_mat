#!/usr/bin/env python3
"""Failure-signaling launcher for basketball_game_id_core.py."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

CORE_PATH=Path(__file__).with_name("basketball_game_id_core.py")

def load_core():
    spec=importlib.util.spec_from_file_location("basketball_game_id_core",CORE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(f"Unable to load {CORE_PATH}")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    core=load_core(); core.main()
    text=core.LOG_FILE.read_text(encoding="utf-8",errors="replace")
    fatal_markers=("ERROR loading", "ERROR updating", "FATAL ERROR", "STATUS: FAILED")
    if any(marker in text for marker in fatal_markers): sys.exit(1)
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Production-settings launcher for the historical master validation test.

The full validation engine remains in basketball_pipeline_final_master_test_core.py.
This launcher replaces its stale hardcoded CURRENT_SETTINGS with values loaded from
model_config.yaml plus rolling_bias_state.yaml at execution time.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path("docs/win/basketball")
MODEL_CONFIG = ROOT / "config/model_config.yaml"
ROLLING_STATE = ROOT / "config/rolling_bias_state.yaml"
CORE_PATH = Path(__file__).with_name("basketball_pipeline_final_master_test_core.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def bias_value(league: str, kind: str, cfg: dict, state: dict) -> float:
    bc = cfg["leagues"][league]["bias"][kind]
    method = str(bc.get("method", "")).lower()
    if method == "fixed":
        return float(bc["value"])
    state_key = "margin_bias" if kind == "margin" else "total_bias"
    value = (((state.get("leagues") or {}).get(league) or {}).get(state_key) or {}).get("value")
    if value is None:
        raise ValueError(f"No current rolling {kind} bias in {ROLLING_STATE} for {league}")
    return float(value)


def production_settings() -> dict[str, dict[str, float]]:
    cfg = load_yaml(MODEL_CONFIG)
    state = load_yaml(ROLLING_STATE)
    out = {}
    for league in ["nba", "ncaam", "wnba"]:
        lc = cfg["leagues"][league]
        spread_std = lc["std"]["spread"]
        total_std = lc["std"]["total"]
        if str(spread_std.get("mode", "")).lower() != "fixed" or str(total_std.get("mode", "")).lower() != "fixed":
            raise ValueError(
                f"Master test compatibility currently requires fixed STD modes; {league} has "
                f"spread={spread_std.get('mode')} total={total_std.get('mode')}"
            )
        out[league.upper()] = {
            "MARGIN_BIAS": bias_value(league, "margin", cfg, state),
            "TOTAL_BIAS": bias_value(league, "total", cfg, state),
            "SPREAD_STD": float(spread_std["value"]),
            "TOTAL_STD": float(total_std["value"]),
            "ML_EDGE": float(lc["edge"]["moneyline"]),
            "SPREAD_EDGE": float(lc["edge"]["spread"]),
            "TOTAL_EDGE": float(lc["edge"]["total"]),
        }
    return out


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_pipeline_final_master_test_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load master-test core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def maybe_write_provenance(settings: dict) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input")
    parser.add_argument("--league")
    parser.add_argument("--season")
    args, _ = parser.parse_known_args()
    if not args.input:
        return
    path = Path(args.input)
    if not path.exists():
        return
    league = (args.league or "NBA").upper()
    output = path.parent / f"{league}_{args.season or 'current'}_MASTER_SETTINGS_PROVENANCE.json"
    payload = {
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "rolling_bias_state": str(ROLLING_STATE),
        "rolling_bias_state_sha256": sha256(ROLLING_STATE),
        "settings": settings,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    settings = production_settings()
    core = load_core()
    core.CURRENT_SETTINGS = settings
    print(f"Loaded production settings from {MODEL_CONFIG} sha256={sha256(MODEL_CONFIG)}", flush=True)
    print(f"Loaded rolling state from {ROLLING_STATE} sha256={sha256(ROLLING_STATE)}", flush=True)
    maybe_write_provenance(settings)
    core.main()


if __name__ == "__main__":
    main()

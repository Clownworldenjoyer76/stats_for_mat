#!/usr/bin/env python3
"""Remove the incorrect Step 18 patch from the existing master-test core.

This restores basketball_pipeline_final_master_test_core.py to its pre-Step-18
state without touching markets.yaml. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path("docs/win/basketball")
CORE = ROOT / "scripts/testing/basketball_pipeline_final_master_test_core.py"
LAUNCHER = ROOT / "scripts/testing/basketball_pipeline_final_master_test.py"

MARKER = "# Step 18 additions for basketball_pipeline_final_master_test_core.py"

SOURCE_OLD = """    if 'model_source' in df.columns:
        seen_sources = {
            str(v).strip().lower()
            for v
            in df[
                'model_source'
            ].dropna().tolist()
            if str(v).strip()
        }

        if (
            seen_sources
            and seen_sources
            != {
                model_source
            }
        ):
            raise ValueError(
                'Input model_source values '
                f'{sorted(seen_sources)} '
                'do not match requested '
                f'model_source={model_source}'
            )

    df[
        'model_source'
    ] = model_source
"""

SOURCE_NEW = """    df = apply_historical_model_source(
        df,
        league,
        settings,
        model_source,
        internal_season,
    )
"""

PARSER_OLD = """    parser.add_argument(
        '--quick',
        action='store_true',
        help=(
            'Same logic, fewer bootstrap '
            'scenarios for code validation'
        ),
    )
"""

PARSER_NEW = """    parser.add_argument(
        '--quick',
        action='store_true',
        help=(
            'Same logic, fewer bootstrap '
            'scenarios for code validation'
        ),
    )

    parser.add_argument(
        '--wnba-market-bands',
        action='store_true',
        help=(
            'Run Step 18 WNBA market-band validation: '
            'ensemble development/OOS discovery, then one '
            'untouched-lockbox comparison of dratings/sdv/ensemble.'
        ),
    )

    parser.add_argument(
        '--market-bands-output',
        default=str(
            WNBA_MARKET_BANDS_REPORT
        ),
        help=(
            'Output path for the frozen WNBA market-band report.'
        ),
    )
"""

SEASON_OLD = """    season = str(
        internal_season
    )
"""

SEASON_NEW = """    season = str(
        internal_season
    )

    if args.wnba_market_bands:
        if league != 'WNBA':
            raise ValueError(
                '--wnba-market-bands requires --league WNBA'
            )

        report_path = run_wnba_market_band_validation(
            input_file=input_file,
            internal_season=internal_season,
            settings=settings,
            selection_policies=selection_policies,
            markets_file=markets_file,
            output_path=Path(
                args.market_bands_output
            ),
        )

        progress(
            'WNBA market-band validation complete: '
            f'{report_path}'
        )

        return
"""


def main() -> int:
    if not CORE.exists():
        raise FileNotFoundError(CORE)
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = CORE.read_text(encoding="utf-8")

    if MARKER not in text:
        print(f"No incorrect Step 18 patch found: {CORE}")
        return 0

    # Remove the injected Step 18 additions block.
    marker_pos = text.index(MARKER)
    block_start = text.rfind("\nimport hashlib", 0, marker_pos)
    if block_start < 0:
        block_start = text.rfind("\n\nimport hashlib", 0, marker_pos)

    main_pos = text.find("\ndef main() -> None:", marker_pos)
    if block_start < 0 or main_pos < 0:
        raise RuntimeError(
            "Could not identify injected Step 18 block boundaries safely."
        )

    restored = text[:block_start] + "\n" + text[main_pos + 1:]

    if SOURCE_NEW in restored:
        restored = restored.replace(SOURCE_NEW, SOURCE_OLD, 1)

    if PARSER_NEW in restored:
        restored = restored.replace(PARSER_NEW, PARSER_OLD, 1)

    if SEASON_NEW in restored:
        restored = restored.replace(SEASON_NEW, SEASON_OLD, 1)

    if MARKER in restored:
        raise RuntimeError("Step 18 marker still present after repair.")

    if "--wnba-market-bands" in restored:
        raise RuntimeError("Step 18 parser arguments still present after repair.")

    # Validate syntax before replacing the real file.
    compile(restored, str(CORE), "exec")

    backup = CORE.with_suffix(CORE.suffix + ".bad_step18_backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    tmp = CORE.with_suffix(CORE.suffix + ".repair.tmp")
    tmp.write_text(restored, encoding="utf-8")
    tmp.replace(CORE)

    print(f"Repaired: {CORE}")
    print(f"Backup of bad patch: {backup}")
    print("Incorrect Step 18 patch removed: YES")
    print("markets.yaml modified: NO")
    print("2025 lockbox leakage removed: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import pytest

"""Guard tests for EraVocalProfile propagation at critical VQI callsites.

These checks prevent regressions where historical-vocal paths call compute_vqi
without era_profile, which would reintroduce false-negative VQI behavior.
"""


import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _assert_vqi_calls_have_era_profile(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    call_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*compute_vqi\s*\(")
    missing: list[int] = []

    for idx, line in enumerate(lines):
        if not call_pattern.search(line):
            continue
        window = "\n".join(lines[idx : min(len(lines), idx + 16)])
        if "era_profile=" not in window:
            missing.append(idx + 1)

    assert not missing, (
        f"EraVocalProfile guard failed in {path}: compute_vqi calls without era_profile at lines {missing}."
    )


@pytest.mark.unit
def test_vqi_callsites_include_era_profile_in_critical_paths() -> None:
    critical_files = [
        ROOT / "backend/core/unified_restorer_v3.py",
        ROOT / "backend/core/vocal_no_harm_gate.py",
        ROOT / "backend/core/feedback_chain.py",
        ROOT / "backend/core/dsp/stem_level_restorer.py",
        ROOT / "backend/core/real_audio_execution_golden_gate.py",
    ]

    for file_path in critical_files:
        assert file_path.exists(), f"Missing guard target file: {file_path}"
        _assert_vqi_calls_have_era_profile(file_path)

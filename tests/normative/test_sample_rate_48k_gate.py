from __future__ import annotations

"""Normative guard: critical DSP tests must use 48 kHz fixture defaults."""


import re
from pathlib import Path

import pytest

CRITICAL_TEST_FILES = [
    "tests/test_tape_specialist.py",
    "tests/test_tonal_balance_restorer.py",
    "tests/test_transparent_dynamics.py",
]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_critical_dsp_tests_default_to_48k() -> None:
    pattern = re.compile(r"def\s+sample_rate\s*\([^)]*\):[\s\S]*?return\s+(\d+)", re.MULTILINE)

    for rel in CRITICAL_TEST_FILES:
        path = Path(rel)
        assert path.exists(), f"Testdatei fehlt: {rel}"
        text = path.read_text(encoding="utf-8")

        m = pattern.search(text)
        assert m is not None, f"Keine sample_rate-Fixture in {rel} gefunden"
        assert int(m.group(1)) == 48_000, f"{rel} nutzt nicht 48k Fixture-Default. Gefunden: {m.group(1)}"

from __future__ import annotations

"""Normative guard: tests must target current functionality/spec baseline.

Blocks known stale test markers that indicate outdated references instead of
current copilot-instructions/spec contracts.
"""


from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"

# Keep this list strict and explicit; each marker corresponds to a known stale
# reference style that should not reappear in active Python tests.
STALE_MARKERS: tuple[str, ...] = (
    "highend_studio",
    "v10.0.0.",
    "legacy_tests_to_remove.txt",
)


@pytest.mark.timeout(20)
def test_no_stale_markers_in_python_tests() -> None:
    offenders: list[str] = []
    this_file = Path(__file__).resolve()

    for path in TESTS_DIR.rglob("*.py"):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in STALE_MARKERS:
            if marker in text:
                rel = path.relative_to(ROOT)
                offenders.append(f"{rel}: contains '{marker}'")

    assert not offenders, "Stale test markers found:\n" + "\n".join(offenders)

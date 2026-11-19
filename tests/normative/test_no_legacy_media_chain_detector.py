from __future__ import annotations

"""Normative guard: legacy MediaChainDetector must not reappear in active code paths.

Spec intent:
- Authoritative carrier-chain detection is MediumDetector-based.
- No second, parallel legacy detector path should exist.
"""


from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY_FILE = ROOT / "backend" / "media_chain_detector.py"
SEARCH_DIRS = [ROOT / "backend", ROOT / "Aurik10", ROOT / "denker", ROOT / "plugins", ROOT / "dsp"]
LEGACY_PATTERNS = ("MediaChainDetector", "media_chain_detector", "detect_chain(")


@pytest.mark.timeout(20)
def test_legacy_media_chain_detector_file_removed() -> None:
    """Legacy detector module must stay deleted."""
    assert not LEGACY_FILE.exists(), "Legacy file backend/media_chain_detector.py must remain removed"


@pytest.mark.timeout(20)
def test_no_legacy_media_chain_detector_references_in_active_code() -> None:
    """No active code path may import/call legacy detector symbols."""
    offenders: list[str] = []

    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in LEGACY_PATTERNS:
                if pattern in text:
                    rel = path.relative_to(ROOT)
                    offenders.append(f"{rel}: contains '{pattern}'")
                    break

    assert not offenders, "Legacy MediaChainDetector references found:\n" + "\n".join(offenders)

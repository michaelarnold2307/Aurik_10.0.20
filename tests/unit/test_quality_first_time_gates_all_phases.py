from __future__ import annotations

import pytest

"""Quality-first policy guard for time-limited phase paths.

Ensures known runtime caps/skip guards are quality-gated in high-end modes
(quality/maximum) so time factor does not silently reduce restoration quality.
"""


from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_known_time_limited_phases_are_quality_gated() -> None:
    root = Path(__file__).resolve().parents[2]

    checks: list[tuple[Path, str, str]] = [
        (
            root / "backend/core/phases/phase_06_frequency_restoration.py",
            "short_clip_guard",
            'quality_mode not in ("quality", "maximum")',
        ),
        (
            root / "backend/core/phases/phase_06_frequency_restoration.py",
            "join(timeout=timeout_s)",
            'if quality_mode in ("quality", "maximum"):',
        ),
        (
            root / "backend/core/phases/phase_12_wow_flutter_fix.py",
            "_PYIN_CAP_S =",
            "_PYIN_CAP_S = 0 if _quality_first_unleashed else 30",
        ),
        (
            root / "backend/core/phases/phase_19_de_esser.py",
            "_STAGE_CAP_S =",
            "_STAGE_CAP_S = 0 if quality_first_unleashed else 30",
        ),
        (
            root / "backend/core/phases/phase_42_vocal_enhancement.py",
            "long_audio_",
            'quality_mode not in ("quality", "maximum")',
        ),
    ]

    for path, marker, required in checks:
        text = _read(path)
        assert marker in text, f"Expected marker '{marker}' not found in {path}"
        assert required in text, f"Expected quality-gate '{required}' not found in {path}"

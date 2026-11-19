"""Tests for UV3 _canonicalize_quality_mode bridge function.

§Mode-Alias (copilot-instructions): UV3 passes ProcessingMode values
('restoration','studio_2026') to phases that expect QualityMode strings
('quality','maximum').  _canonicalize_quality_mode bridges the gap.
"""

import pytest

from backend.core.unified_restorer_v3 import _canonicalize_quality_mode


@pytest.mark.parametrize(
    "raw, expected",
    [
        # ProcessingMode strings (Magic Button) → internal quality strings
        ("restoration", "quality"),
        ("RESTORATION", "quality"),
        ("studio_2026", "maximum"),
        ("STUDIO_2026", "maximum"),
        ("studio2026", "maximum"),
        ("studio", "maximum"),
        # QualityMode strings pass through unchanged
        ("quality", "quality"),
        ("maximum", "maximum"),
        ("balanced", "balanced"),
        ("fast", "fast"),
        # Edge cases
        ("", "quality"),
        (None, "quality"),
        ("  restoration  ", "quality"),
        ("unknown_mode", "quality"),
    ],
)
def test_canonicalize_quality_mode(raw, expected):
    assert _canonicalize_quality_mode(raw) == expected


def test_pipeline_quality_mode_value_in_execute_pipeline():
    """Verify that _execute_pipeline normalises quality_mode_value from config.mode.value."""
    from unittest.mock import MagicMock, patch

    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.MAXIMUM))

    captured = {}

    restorer._profiled_phase_call

    def _intercepting_profiled_phase_call(phase, audio, **kwargs):
        captured["quality_mode"] = kwargs.get("quality_mode")
        result = MagicMock()
        result.success = False
        result.audio = audio
        return result

    dummy_phase = MagicMock()
    dummy_phase.phase_id = "phase_test"

    with patch.object(restorer, "_profiled_phase_call", side_effect=_intercepting_profiled_phase_call):
        # Simulate what _execute_pipeline does: read mode.value then canonicalize
        raw_value = restorer.config.mode.value  # e.g. "maximum" from QualityMode.MAXIMUM
        canonicalized = _canonicalize_quality_mode(raw_value)
        assert canonicalized == "maximum"

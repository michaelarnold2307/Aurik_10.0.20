from __future__ import annotations

"""[RELEASE_MUST] §8.6b Worldclass Composite Score Gate.

Prueft die WCS-Berechnung und Gate-Entscheidung inkl. Artifact-Veto.
"""


from pathlib import Path

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_07 = _ROOT / ".github" / "specs" / "07_quality_and_tests.md"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestWorldclassCompositeGate:
    def test_spec_declares_wcs_thresholds(self) -> None:
        content = _SPEC_07.read_text(encoding="utf-8")

        assert "§8.6b Psychoakustischer Weltspitzen-Composite-Score" in content
        assert "WCS >= 0.88" in content
        assert "WCS >= 0.91" in content
        assert "WCS >= 0.85" in content

    def test_uv3_exports_worldclass_composite_gate_metadata(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"worldclass_composite_gate": dict(_worldclass_composite_gate)' in content
        assert '"error_code": "WCS_FAIL"' in content

    def test_wcs_passes_for_high_quality_vocal_restoration(self) -> None:
        vector = UnifiedRestorerV3._build_hybrid_engineer_vector(
            {
                "artifact_freedom": 0.98,
                "vocal_identity_preservation": 0.95,
                "formant_integrity": 0.93,
                "micro_dynamic_correlation": 0.92,
                "emotional_arc_preservation": 0.94,
                "spectral_color_preservation": 0.93,
                "stereo_scene_stability": 0.90,
            }
        )
        gate = UnifiedRestorerV3._evaluate_worldclass_composite_gate(
            vector=vector,
            panns_singing=0.60,
            is_studio_mode=False,
            artifact_freedom=0.98,
        )

        assert gate["wcs"] >= gate["threshold"]
        assert gate["threshold"] == pytest.approx(0.88)
        assert gate["passed"] is True

    def test_artifact_veto_blocks_even_with_high_wcs(self) -> None:
        vector = UnifiedRestorerV3._build_hybrid_engineer_vector(
            {
                "artifact_freedom": 0.94,
                "vocal_identity_preservation": 0.99,
                "formant_integrity": 0.99,
                "micro_dynamic_correlation": 0.99,
                "emotional_arc_preservation": 0.99,
                "spectral_color_preservation": 0.99,
                "stereo_scene_stability": 0.99,
            }
        )
        gate = UnifiedRestorerV3._evaluate_worldclass_composite_gate(
            vector=vector,
            panns_singing=0.70,
            is_studio_mode=True,
            artifact_freedom=0.94,
        )

        assert gate["artifact_veto"] is True
        assert gate["wcs_pass"] is True
        assert gate["passed"] is False

    def test_instrumental_profile_uses_085_threshold(self) -> None:
        vector = UnifiedRestorerV3._build_hybrid_engineer_vector(
            {
                "artifact_freedom": 0.97,
                "vocal_identity_preservation": 1.0,
                "formant_integrity": 0.88,
                "micro_dynamic_correlation": 0.89,
                "emotional_arc_preservation": 0.90,
                "spectral_color_preservation": 0.90,
                "stereo_scene_stability": 0.90,
            }
        )
        gate = UnifiedRestorerV3._evaluate_worldclass_composite_gate(
            vector=vector,
            panns_singing=0.10,
            is_studio_mode=False,
            artifact_freedom=0.97,
        )

        assert gate["profile"] == "instrumental"
        assert gate["threshold"] == pytest.approx(0.85)

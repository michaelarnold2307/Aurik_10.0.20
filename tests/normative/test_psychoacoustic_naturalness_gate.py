from __future__ import annotations

"""[RELEASE_MUST] §8.6g Psychoakustischer Natuerlichkeits-Guard.

Sichert die nicht-klinische Klangprioritaet im Endgate: natuerlich, echt,
psychoakustisch glaubwuerdig.
"""


from pathlib import Path

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_07 = _ROOT / ".github" / "specs" / "07_quality_and_tests.md"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestPsychoacousticNaturalnessGate:
    def test_spec_declares_psychoacoustic_naturalness_guard(self) -> None:
        content = _SPEC_07.read_text(encoding="utf-8")

        assert "§8.6g [RELEASE_MUST] Psychoakustischer Natuerlichkeits-Guard" in content
        assert "PSYCHO = 0.28 * noise_texture_authenticity" in content
        assert "PSYCHO >= 0.84" in content
        assert "PSYCHO >= 0.87" in content
        assert "PSYCHO >= 0.82" in content

    def test_uv3_exports_psychoacoustic_gate_metadata(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"psychoacoustic_naturalness_gate": dict(_psychoacoustic_naturalness_gate)' in content
        assert '"error_code": "PSYCHO_NATURALNESS_FAIL"' in content
        assert '"psychoacoustic_feedback_recovery": (self._phase_metadata_accumulator or {}).get(' in content
        assert 'self._phase_metadata_accumulator["psychoacoustic_feedback_recovery"]' in content

    def test_uv3_has_phasewise_psycho_strength_scalar_feedback(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "def _compute_psychoacoustic_phase_strength_scalar(" in content
        assert "§8.6g-II Psycho-Scalar" in content
        assert '"psycho_strength_scalar"' in content
        assert '"psycho_strength_risk_score"' in content
        assert '"psycho_strength_signals"' in content
        assert '"_psycho_runtime_state"' in content
        assert '"psycho_delta_penalty"' in content
        assert '"psycho_runtime_rolling_risk"' in content

    def test_gate_passes_for_natural_vocal_profile(self) -> None:
        vector = UnifiedRestorerV3._build_hybrid_engineer_vector(
            {
                "noise_texture_authenticity": 0.90,
                "micro_dynamic_correlation": 0.89,
                "emotional_arc_preservation": 0.90,
                "spectral_color_preservation": 0.88,
            }
        )
        gate = UnifiedRestorerV3._evaluate_psychoacoustic_naturalness_gate(
            vector=vector,
            panns_singing=0.60,
            is_studio_mode=False,
        )

        assert gate["profile"] == "vocal"
        assert gate["passed"] is True

    def test_gate_fails_for_clinical_texture_drop(self) -> None:
        vector = UnifiedRestorerV3._build_hybrid_engineer_vector(
            {
                "noise_texture_authenticity": 0.71,
                "micro_dynamic_correlation": 0.93,
                "emotional_arc_preservation": 0.92,
                "spectral_color_preservation": 0.90,
            }
        )
        gate = UnifiedRestorerV3._evaluate_psychoacoustic_naturalness_gate(
            vector=vector,
            panns_singing=0.55,
            is_studio_mode=False,
        )

        assert gate["score_pass"] is True
        assert gate["floor_pass"] is False
        assert gate["passed"] is False

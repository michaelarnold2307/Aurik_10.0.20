from __future__ import annotations

"""[RELEASE_MUST] §8.6a Hybrid-Engineer-Vector Contract.

Sichert den normativen Vertrag fuer den Human-Talent-Emulation-Vektor (HTEV):
- Abschnitt in Spec 07 vorhanden
- UV3 erzeugt den Vektor ueber kanonische Helper
- Vektor ist vollstaendig, normiert und deterministisch
"""


from pathlib import Path

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_07 = _ROOT / ".github" / "specs" / "07_quality_and_tests.md"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestWorldclassHybridEngineerVector:
    def test_spec_declares_worldclass_hybrid_engineer_protocol(self) -> None:
        content = _SPEC_07.read_text(encoding="utf-8")

        assert "§8.6 [RELEASE_MUST] Worldclass Hybrid-Engineer Protocol" in content
        assert "§8.6a Human-Talent-Emulation-Vektor (HTEV)" in content

    def test_uv3_exports_hybrid_engineer_vector_metadata(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"hybrid_engineer_vector": dict(_hybrid_engineer_vector)' in content
        assert "_build_hybrid_engineer_vector" in content

    def test_vector_contains_all_required_dimensions_and_is_clamped(self) -> None:
        vector = UnifiedRestorerV3._build_hybrid_engineer_vector(
            {
                "vocal_identity_preservation": 1.2,
                "formant_integrity": -0.2,
                "vibrato_depth_preservation": 0.7,
                "breath_naturalness": 0.8,
                "micro_dynamic_correlation": 0.9,
                "transient_articulation": 0.95,
                "stereo_scene_stability": 0.85,
                "noise_texture_authenticity": 0.92,
                "spectral_color_preservation": 0.91,
                "emotional_arc_preservation": 0.93,
                "artifact_freedom": 0.98,
                "goal_team_balance": 0.88,
            }
        )

        required = {
            "vocal_identity_preservation",
            "formant_integrity",
            "vibrato_depth_preservation",
            "breath_naturalness",
            "micro_dynamic_correlation",
            "transient_articulation",
            "stereo_scene_stability",
            "noise_texture_authenticity",
            "spectral_color_preservation",
            "emotional_arc_preservation",
            "artifact_freedom",
            "goal_team_balance",
        }
        assert set(vector) == required
        assert vector["vocal_identity_preservation"] == pytest.approx(1.0)
        assert vector["formant_integrity"] == pytest.approx(0.0)
        assert all(0.0 <= float(v) <= 1.0 for v in vector.values())

    def test_vector_builder_is_deterministic(self) -> None:
        payload = {
            "vocal_identity_preservation": 0.93,
            "formant_integrity": 0.89,
            "artifact_freedom": 0.97,
            "goal_team_balance": 0.86,
        }
        a = UnifiedRestorerV3._build_hybrid_engineer_vector(payload)
        b = UnifiedRestorerV3._build_hybrid_engineer_vector(payload)
        assert a == b

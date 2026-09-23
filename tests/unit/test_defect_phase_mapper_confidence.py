from __future__ import annotations

import pytest

"""tests/unit/test_defect_phase_mapper_confidence.py

Regression-Tests fuer confidence-aware Priorisierung im DefectPhaseMapper.
"""


from dataclasses import dataclass

from backend.core.defect_phase_mapper import DefectPhaseMapper
from backend.core.defect_scanner import DefectType


@dataclass
class _DummyDefect:
    defect_type: DefectType
    severity: float
    confidence: float


@pytest.mark.unit
def test_low_confidence_downweights_secondary_phases_in_restoration() -> None:
    mapper = DefectPhaseMapper()
    defect = _DummyDefect(defect_type=DefectType.HIGH_FREQ_NOISE, severity=1.0, confidence=0.10)

    phases = mapper.phases_for_defect_profile([defect], mode="restoration", max_phases=8)

    # Primary muss weiterhin vorne bleiben.
    assert "phase_03_denoise" in phases
    assert "phase_29_tape_hiss_reduction" in phases
    # Bei niedriger Confidence koennen sekundäre Phasen konservativ ausfallen.
    if "phase_18_noise_gate" in phases:
        idx_primary = phases.index("phase_03_denoise")
        idx_secondary = phases.index("phase_18_noise_gate")
        assert idx_primary < idx_secondary


# ── §2.69f (2026-09-23): Live-Defektzähler — konservative Behebungs-Meldungen ──


class TestLiveResolvedClaims:
    def test_repair_phase_claims_target_defects_conservatively(self) -> None:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        sev_map = {"crackle": 0.8, "clicks": 0.6, "noise_level": 0.5}
        claims = UnifiedRestorerV3._compute_live_resolved_claims("phase_09_crackle_removal", sev_map)
        assert claims, "Reparatur-Phase muss Ziel-Defekte melden"
        for key, residual in claims.items():
            assert 0.0 <= residual <= 0.3  # max_residual aus resolved_defects_helper
            assert residual <= 0.5 * sev_map.get(key, 1.0) + 1e-9  # ≤ reduction_ratio × Severity

    def test_absent_defects_not_claimed(self) -> None:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        claims = UnifiedRestorerV3._compute_live_resolved_claims("phase_02_hum_removal", {"crackle": 0.7})
        assert "crackle" not in claims

    def test_enhancement_phase_claims_nothing(self) -> None:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        assert (
            UnifiedRestorerV3._compute_live_resolved_claims("phase_38_presence_boost", {"noise_level": 0.9})
            == {}
        )

    def test_empty_severity_map_no_claims(self) -> None:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        assert UnifiedRestorerV3._compute_live_resolved_claims("phase_09_crackle_removal", {}) == {}

    def test_deterministic(self) -> None:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        m = {"crackle": 0.8, "clicks": 0.6}
        assert UnifiedRestorerV3._compute_live_resolved_claims(
            "phase_09_crackle_removal", m
        ) == UnifiedRestorerV3._compute_live_resolved_claims("phase_09_crackle_removal", dict(m))


def test_high_confidence_keeps_secondary_recovery_relevant() -> None:
    mapper = DefectPhaseMapper()
    defect = _DummyDefect(defect_type=DefectType.HIGH_FREQ_NOISE, severity=1.0, confidence=0.95)

    phases = mapper.phases_for_defect_profile([defect], mode="restoration", max_phases=6)

    assert "phase_03_denoise" in phases
    assert "phase_29_tape_hiss_reduction" in phases
    assert "phase_18_noise_gate" in phases


def test_missing_confidence_falls_back_to_neutral_weighting() -> None:
    mapper = DefectPhaseMapper()

    class _NoConfidence:
        def __init__(self) -> None:
            self.defect_type = DefectType.CLICKS
            self.severity = 0.9

    phases = mapper.phases_for_defect_profile([_NoConfidence()], mode="restoration", max_phases=5)

    assert "phase_01_click_removal" in phases


def test_very_low_confidence_suppresses_secondary_in_restoration() -> None:
    mapper = DefectPhaseMapper()
    defect = _DummyDefect(defect_type=DefectType.HIGH_FREQ_NOISE, severity=1.0, confidence=0.05)

    phases = mapper.phases_for_defect_profile([defect], mode="restoration", max_phases=8)

    assert "phase_03_denoise" in phases
    assert "phase_29_tape_hiss_reduction" in phases
    assert "phase_18_noise_gate" not in phases

"""Unit-Tests: TransparentDynamics._compute_transparent_dynamics_profile() (§2.56)

Verifiziert dass der adaptive mix_delta-Parameter korrekt aus
Quality-Mode und Restorability berechnet wird.
"""

from typing import cast

import numpy as np
import pytest

from backend.core.phases.phase_54_transparent_dynamics import (
    TransparentDynamicsV1 as TransparentDynamicsPhase,
)
from backend.core.phases.phase_54_transparent_dynamics import _extract_compression_pressure

# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _profile(material: str = "vinyl", qm: str = "balanced", rest: float = 50.0) -> dict:
    return TransparentDynamicsPhase._compute_transparent_dynamics_profile(material, qm, rest)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransparentDynamicsProfileQualityMode:
    """Quality-Mode-Anpassungen auf mix_delta."""

    def test_quality_positive_delta(self):
        delta = _profile(qm="quality")["mix_delta"]
        assert delta > 0.0, f"quality should increase mix, got {delta}"

    def test_maximum_positive_delta(self):
        delta = _profile(qm="maximum")["mix_delta"]
        assert delta > 0.0

    def test_fast_negative_delta(self):
        delta = _profile(qm="fast")["mix_delta"]
        assert delta < 0.0, f"fast should decrease mix, got {delta}"

    def test_balanced_zero_delta(self):
        delta = _profile(qm="balanced")["mix_delta"]
        assert delta == pytest.approx(0.0), f"balanced should have 0 delta, got {delta}"

    def test_none_quality_mode_zero_delta(self):
        delta = _profile(qm=cast(str, None))["mix_delta"]
        assert delta == pytest.approx(0.0)


class TestTransparentDynamicsProfileRestorability:
    """Restorability-Einfluss auf mix_delta."""

    def test_low_restorability_negative_delta(self):
        """Low-Rest (<40) → mehr dry (weniger Kompression auf fragiles Material)."""
        delta = _profile(rest=25.0)["mix_delta"]
        assert delta < 0.0, f"low restorability should decrease mix, got {delta}"

    def test_high_restorability_no_change(self):
        delta = _profile(rest=80.0)["mix_delta"]
        assert delta == pytest.approx(0.0)

    def test_combined_quality_maximum_low_rest(self):
        """quality + low_rest: Net-delta = +0.05 - 0.10 = -0.05."""
        delta = _profile(qm="quality", rest=20.0)["mix_delta"]
        assert delta == pytest.approx(-0.05)

    def test_combined_fast_low_rest(self):
        """fast + low_rest: -0.05 - 0.10 = -0.15"""
        delta = _profile(qm="fast", rest=20.0)["mix_delta"]
        assert delta == pytest.approx(-0.15)


class TestTransparentDynamicsProfileBounds:
    """Grenzwerte."""

    def test_delta_within_limits(self):
        """mix_delta muss in [-0.20, +0.20] bleiben."""
        for material in ["shellac", "vinyl", "cd_digital", "unknown"]:
            for qm in ["quality", "maximum", "fast", "balanced", None]:
                for rest in [5.0, 50.0, 90.0]:
                    delta = _profile(material, cast(str, qm), rest)["mix_delta"]
                    assert -0.20 <= delta <= 0.20, f"mix_delta out of bounds: {delta} ({material}, {qm}, {rest})"

    def test_unknown_material_no_error(self):
        delta = _profile("unknown_xyz")["mix_delta"]
        assert isinstance(delta, float)


class TestTransparentDynamicsProfileIntegration:
    """End-to-End: Profil fließt in PhaseResult metadata."""

    def test_profile_in_metadata(self):
        from backend.core.phases.phase_54_transparent_dynamics import TransparentDynamicsV1 as TransparentDynamicsPhase

        phase = TransparentDynamicsPhase(sample_rate=48000)
        audio = np.random.uniform(-0.3, 0.3, (48000, 2)).astype(np.float32)

        # Use quality mode kwarg
        result = phase.process(
            audio,
            quality_mode="quality",
            restorability_score=60.0,
            strength=0.5,
        )
        assert result.success
        assert "transparent_dynamics_profile" in result.metadata
        assert "mix_delta" in result.metadata
        # quality mode → positive mix_delta
        assert result.metadata["mix_delta"] >= 0.0


class TestTransparentDynamicsCompressionPressure:
    def test_extract_compression_pressure_from_numeric_scores(self):
        pressure = _extract_compression_pressure(
            {
                "compression_artifacts": 0.80,
                "dynamic_compression_excess": 0.30,
            }
        )
        assert pressure == pytest.approx(0.80)

    def test_extract_compression_pressure_from_mixed_objects(self):
        class _Score:
            def __init__(self, severity: float):
                self.severity = severity

        pressure = _extract_compression_pressure(
            {
                "DefectType.COMPRESSION_ARTIFACTS": _Score(0.62),
                "DefectType.DIGITAL_ARTIFACTS": _Score(0.90),
            }
        )
        assert pressure == pytest.approx(0.62)

    def test_high_compression_pressure_enables_hard_intervention(self):
        phase = TransparentDynamicsPhase(sample_rate=48000)
        audio = np.random.uniform(-0.2, 0.2, (48000, 2)).astype(np.float32)

        result = phase.process(
            audio,
            strength=0.05,
            defect_scores={"compression_artifacts": 0.90},
            quality_mode="balanced",
            restorability_score=45.0,
        )

        assert result.success
        assert result.metadata["hard_intervention_active"] is True
        assert result.metadata["control_strength"] >= 0.75
        assert result.metadata["mix"] >= 0.70


class TestTransparentDynamicsLocalityProfile:
    def test_locality_profile_is_event_strength_adaptive(self):
        sr = 48000
        profile, coverage = TransparentDynamicsPhase._build_locality_profile(
            n_samples=sr * 3,
            sample_rate=sr,
            defect_locations={"compression_artifacts": [(0.30, 0.90)], "transient_smearing": [(1.70, 2.30)]},
            event_metadata={
                "compression_artifacts": {"severity": 0.95, "confidence": 0.95},
                "transient_smearing": {"severity": 0.35, "confidence": 0.65},
            },
        )

        assert profile.shape == (sr * 3,)
        assert 0.0 < coverage < 0.50
        strong_region = float(np.mean(profile[int(0.45 * sr) : int(0.75 * sr)]))
        mild_region = float(np.mean(profile[int(1.85 * sr) : int(2.15 * sr)]))
        clean_region = float(np.mean(profile[int(1.10 * sr) : int(1.35 * sr)]))
        assert strong_region > mild_region * 1.5
        assert clean_region < 0.04

    def test_vibrato_zone_caps_locality_profile(self):
        sr = 48000
        free, _ = TransparentDynamicsPhase._build_locality_profile(
            n_samples=sr * 3,
            sample_rate=sr,
            defect_locations={"nr_breathing_artifact": [(1.20, 1.80)]},
            event_metadata={"nr_breathing_artifact": {"severity": 0.95, "confidence": 0.95}},
        )
        capped, _ = TransparentDynamicsPhase._build_locality_profile(
            n_samples=sr * 3,
            sample_rate=sr,
            defect_locations={"nr_breathing_artifact": [(1.20, 1.80)]},
            event_metadata={"nr_breathing_artifact": {"severity": 0.95, "confidence": 0.95}},
            protected_zones=[(1.15, 1.85, 0.20)],
        )

        free_strength = float(np.mean(free[int(1.35 * sr) : int(1.65 * sr)]))
        capped_strength = float(np.mean(capped[int(1.35 * sr) : int(1.65 * sr)]))
        assert capped_strength <= 0.21
        assert capped_strength < free_strength * 0.55

    def test_low_compression_pressure_keeps_hard_intervention_off(self):
        phase = TransparentDynamicsPhase(sample_rate=48000)
        audio = np.random.uniform(-0.2, 0.2, (48000, 2)).astype(np.float32)

        result = phase.process(
            audio,
            strength=0.20,
            defect_scores={"compression_artifacts": 0.20},
            quality_mode="balanced",
            restorability_score=45.0,
        )

        assert result.success
        assert result.metadata["hard_intervention_active"] is False
        assert result.metadata["control_floor"] == pytest.approx(0.0)


def test_masking_curve_robust_against_spike_outlier():
    phase = TransparentDynamicsPhase(sample_rate=48000)
    audio = np.zeros(48000, dtype=np.float32)
    audio[12000] = 1.0

    masking = phase._detect_psychoacoustic_masking(audio)
    assert masking.shape == audio.shape
    assert np.all(np.isfinite(masking))
    assert float(np.min(masking)) >= 0.0
    assert float(np.max(masking)) <= 1.0


def test_transient_detector_finds_impulses_without_false_allmask():
    phase = TransparentDynamicsPhase(sample_rate=48000)
    audio = np.zeros(48000, dtype=np.float32)
    audio[8000] = 0.9
    audio[24000] = 0.8
    audio[36000] = 0.7

    transient_mask = phase._detect_transients(audio)
    assert transient_mask.shape == audio.shape
    assert np.all(np.isfinite(transient_mask))
    assert float(np.max(transient_mask)) > 0.5
    assert float(np.mean(transient_mask)) < 0.2


def test_apply_compression_short_buffer_no_index_error():
    phase = TransparentDynamicsPhase(sample_rate=48000)
    audio = np.array([0.1], dtype=np.float32)
    masking_curve = np.array([0.0], dtype=np.float32)
    transient_mask = np.array([0.0], dtype=np.float32)

    out = phase._apply_compression(
        audio,
        ratio=2.0,
        threshold_db=-20.0,
        knee_db=6.0,
        attack_ms=10.0,
        release_ms=100.0,
        masking_curve=masking_curve,
        transient_mask=transient_mask,
        compression_pressure=0.0,
    )

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))

    masking_short = phase._detect_psychoacoustic_masking(audio)
    assert masking_short.shape == audio.shape
    assert np.all(np.isfinite(masking_short))

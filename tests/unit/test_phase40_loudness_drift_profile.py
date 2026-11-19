import pytest

"""Unit-Tests fuer Phase 40 Amplituden-Drift-Lokalitaet."""

import numpy as np

from backend.core.phases.phase_40_loudness_normalization import LoudnessNormalizationPhase


@pytest.mark.unit
def test_drift_locality_profile_is_event_strength_adaptive():
    sr = 48000
    profile, coverage = LoudnessNormalizationPhase._build_drift_locality_profile(
        n_samples=sr * 8,
        sample_rate=sr,
        defect_locations={"amplitude_drift": [(0.80, 2.30)], "gain_sag": [(5.40, 6.60)]},
        event_metadata={
            "amplitude_drift": {"severity": 0.95, "confidence": 0.95},
            "gain_sag": {"severity": 0.35, "confidence": 0.65},
        },
    )

    assert profile.shape == (sr * 8,)
    assert 0.0 < coverage < 0.80
    strong_region = float(np.mean(profile[int(1.10 * sr) : int(1.90 * sr)]))
    mild_region = float(np.mean(profile[int(5.70 * sr) : int(6.30 * sr)]))
    clean_region = float(np.mean(profile[int(3.40 * sr) : int(4.40 * sr)]))
    assert strong_region > mild_region * 1.5
    assert clean_region < 0.10


def test_vibrato_zone_caps_drift_locality_profile():
    sr = 48000
    free, _ = LoudnessNormalizationPhase._build_drift_locality_profile(
        n_samples=sr * 5,
        sample_rate=sr,
        defect_locations={"amplitude_drift": [(1.20, 3.20)]},
        event_metadata={"amplitude_drift": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = LoudnessNormalizationPhase._build_drift_locality_profile(
        n_samples=sr * 5,
        sample_rate=sr,
        defect_locations={"amplitude_drift": [(1.20, 3.20)]},
        event_metadata={"amplitude_drift": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.60, 2.80, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.80 * sr) : int(2.60 * sr)]))
    capped_strength = float(np.mean(capped[int(1.80 * sr) : int(2.60 * sr)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.35

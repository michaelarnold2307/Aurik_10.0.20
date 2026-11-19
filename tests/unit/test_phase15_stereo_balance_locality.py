from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_15_stereo_balance import StereoBalancePhaseV2


@pytest.mark.unit
def test_locality_profile_is_event_strength_adaptive() -> None:
    sr = 48000
    profile, coverage = StereoBalancePhaseV2._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"stereo_imbalance": [(0.30, 0.90)], "crosstalk": [(1.70, 2.30)]},
        event_metadata={
            "stereo_imbalance": {"severity": 0.95, "confidence": 0.95},
            "crosstalk": {"severity": 0.35, "confidence": 0.65},
        },
    )

    assert profile.shape == (sr * 3,)
    assert 0.0 < coverage < 0.50
    strong_region = float(np.mean(profile[int(0.45 * sr) : int(0.75 * sr)]))
    mild_region = float(np.mean(profile[int(1.85 * sr) : int(2.15 * sr)]))
    clean_region = float(np.mean(profile[int(1.10 * sr) : int(1.35 * sr)]))
    assert strong_region > mild_region * 1.5
    assert clean_region < 0.04


def test_vibrato_zone_caps_locality_profile() -> None:
    sr = 48000
    free, _ = StereoBalancePhaseV2._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"stereo_imbalance": [(1.20, 1.80)]},
        event_metadata={"stereo_imbalance": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = StereoBalancePhaseV2._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"stereo_imbalance": [(1.20, 1.80)]},
        event_metadata={"stereo_imbalance": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.15, 1.85, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.35 * sr) : int(1.65 * sr)]))
    capped_strength = float(np.mean(capped[int(1.35 * sr) : int(1.65 * sr)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.55

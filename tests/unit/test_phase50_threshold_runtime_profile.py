from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase


@pytest.mark.unit
def test_runtime_profile_keys_and_bounds() -> None:
    p = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="vinyl",
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(p.keys()) == {"strength_floor", "side_multiplier"}
    assert 0.06 <= p["strength_floor"] <= 0.18
    assert 1.60 <= p["side_multiplier"] <= 2.40


def test_fast_mode_more_conservative_than_quality() -> None:
    fast = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="cd_digital",
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="cd_digital",
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["strength_floor"] > quality["strength_floor"]
    assert fast["side_multiplier"] > quality["side_multiplier"]


def test_low_restorability_relaxes_floor_and_side_multiplier() -> None:
    low = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="shellac",
        quality_mode="balanced",
        restorability_score=10.0,
    )
    high = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="shellac",
        quality_mode="balanced",
        restorability_score=90.0,
    )

    assert low["strength_floor"] < high["strength_floor"]
    assert low["side_multiplier"] < high["side_multiplier"]


def test_locality_profile_is_event_strength_adaptive() -> None:
    sr = 48000
    profile, coverage = SpectralRepairPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"spectral_spike": [(0.30, 0.90)], "codec_artifact": [(1.70, 2.30)]},
        event_metadata={
            "spectral_spike": {"severity": 0.95, "confidence": 0.95},
            "codec_artifact": {"severity": 0.35, "confidence": 0.65},
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
    free, _ = SpectralRepairPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"spectral_spike": [(1.20, 1.80)]},
        event_metadata={"spectral_spike": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = SpectralRepairPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"spectral_spike": [(1.20, 1.80)]},
        event_metadata={"spectral_spike": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.15, 1.85, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.35 * sr) : int(1.65 * sr)]))
    capped_strength = float(np.mean(capped[int(1.35 * sr) : int(1.65 * sr)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.55


def test_pre_echo_events_respect_top_level_vibrato_zones(monkeypatch) -> None:
    sr = 48000
    audio = np.ones(sr, dtype=np.float32) * 0.2

    class _DummyPreEchoDetector:
        def repair_region(self, x: np.ndarray, event: dict, sample_rate: int) -> np.ndarray:
            assert sample_rate == sr
            out = np.asarray(x, dtype=np.float32).copy()
            out[int(event["pre_echo_start"]) : int(event["pre_echo_end"])] = 0.0
            return out

    monkeypatch.setattr(
        "backend.core.dsp.pre_echo_detector.get_pre_echo_detector",
        lambda: _DummyPreEchoDetector(),
    )

    result = SpectralRepairPhase().process(
        audio,
        sample_rate=sr,
        material_type="mp3_low",
        pre_echo_events=[{"pre_echo_start": int(0.20 * sr), "pre_echo_end": int(0.30 * sr)}],
        vibrato_zones=[(0.10, 0.40)],
    )

    repaired_region = result.audio[int(0.22 * sr) : int(0.28 * sr)]
    assert float(np.mean(repaired_region)) >= 0.15

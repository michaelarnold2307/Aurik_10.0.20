from typing import cast

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase


def _test_audio(sr: int = 48000) -> np.ndarray:
    duration_s = 1.0
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    return cast(np.ndarray, 0.2 * np.sin(2.0 * np.pi * 440.0 * t))


@pytest.mark.unit
def test_studio_2026_alias_routes_to_maximum_hybrid(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    called = {"mode": None}

    def _fake_detect_pitch_ml_hybrid(audio, sample_rate, quality_mode):
        called["mode"] = quality_mode
        return 440.0, 0.95, {"strategy": "polyphonic_speed_curve"}

    monkeypatch.setattr(phase, "_detect_pitch_ml_hybrid", _fake_detect_pitch_ml_hybrid)

    result = phase.process(
        _test_audio(),
        material_type="tape",
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="studio_2026",
    )

    assert called["mode"] == "maximum"
    assert result.metadata.get("quality_mode") == "maximum"


def test_restoration_alias_routes_to_balanced_hybrid(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    called = {"mode": None}

    def _fake_detect_pitch_ml_hybrid(audio, sample_rate, quality_mode):
        called["mode"] = quality_mode
        return 440.0, 0.95, {"strategy": "adaptive"}

    monkeypatch.setattr(phase, "_detect_pitch_ml_hybrid", _fake_detect_pitch_ml_hybrid)

    result = phase.process(
        _test_audio(),
        material_type="tape",
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="restoration",
    )

    assert called["mode"] == "balanced"
    assert result.metadata.get("quality_mode") == "balanced"


def test_phase31_accepts_material_enum_inputs(monkeypatch):
    phase = SpeedPitchCorrectionPhase()

    def _fake_detect_pitch_pyin(_audio, _params):
        return 440.0, 0.95

    def _fake_tuning_offset(_audio, _sample_rate, _reference_pitch, _detected_pitch):
        return 24.0, 1.02

    monkeypatch.setattr(phase, "_detect_pitch_pyin", _fake_detect_pitch_pyin)
    monkeypatch.setattr(phase, "_compute_tuning_offset", _fake_tuning_offset)
    monkeypatch.setattr(phase, "_correct_wsola", lambda audio, ratio, params: np.asarray(audio, dtype=np.float64))

    result = phase.process(
        _test_audio(),
        material_type=MaterialType.TAPE,
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="fast",
    )

    assert result.metadata.get("material_type") == "tape"
    assert result.success is True


def test_phase31_maps_cassette_alias_to_tape_profile(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    captured: dict[str, dict[str, object] | None] = {"params": None}

    def _fake_detect_pitch_pyin(_audio, _params):
        captured["params"] = dict(_params)
        return 440.0, 0.95

    def _fake_tuning_offset(_audio, _sample_rate, _reference_pitch, _detected_pitch):
        return 24.0, 1.02

    monkeypatch.setattr(phase, "_detect_pitch_pyin", _fake_detect_pitch_pyin)
    monkeypatch.setattr(phase, "_compute_tuning_offset", _fake_tuning_offset)
    monkeypatch.setattr(phase, "_correct_wsola", lambda audio, ratio, params: np.asarray(audio, dtype=np.float64))

    result = phase.process(
        _test_audio(),
        material_type="cassette",
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="fast",
    )

    assert captured["params"] is not None
    assert captured["params"]["max_speed_error"] == 0.10
    assert result.metadata.get("material_type") == "tape"


def test_phase31_locality_profile_is_event_strength_adaptive():
    sr = 48000
    profile, coverage = SpeedPitchCorrectionPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"transport_bump": [(0.30, 0.90)], "scrape_flutter": [(1.70, 2.30)]},
        event_metadata={
            "transport_bump": {"severity": 0.95, "confidence": 0.95},
            "scrape_flutter": {"severity": 0.35, "confidence": 0.65},
        },
    )

    assert profile.shape == (sr * 3,)
    assert 0.0 < coverage < 0.50
    strong_region = float(np.mean(profile[int(0.45 * sr) : int(0.75 * sr)]))
    mild_region = float(np.mean(profile[int(1.85 * sr) : int(2.15 * sr)]))
    clean_region = float(np.mean(profile[int(1.10 * sr) : int(1.35 * sr)]))
    assert strong_region > mild_region * 1.5
    assert clean_region < 0.04


def test_phase31_vibrato_zone_caps_locality_profile():
    sr = 48000
    free, _ = SpeedPitchCorrectionPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"transport_bump": [(1.20, 1.80)]},
        event_metadata={"transport_bump": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = SpeedPitchCorrectionPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"transport_bump": [(1.20, 1.80)]},
        event_metadata={"transport_bump": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.15, 1.85, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.35 * sr) : int(1.65 * sr)]))
    capped_strength = float(np.mean(capped[int(1.35 * sr) : int(1.65 * sr)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.55

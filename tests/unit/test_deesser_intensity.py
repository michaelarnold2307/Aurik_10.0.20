from __future__ import annotations

import pytest

"""Regressionstests fuer gemeinsamen De-Esser-Intensitaetsrechner und Phasen-Integration."""


from dataclasses import dataclass
from typing import cast

import numpy as np

from backend.core.defect_scanner import MaterialType

SR = 48_000


def _make_sibilant(freq: float = 7000.0, duration: float = 0.4, amp: float = 0.45) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    env = np.sin(np.linspace(0.0, np.pi, t.size, dtype=np.float32)) ** 2
    audio = cast(np.ndarray, np.asarray(amp * env * np.sin(2.0 * np.pi * freq * t), dtype=np.float32))
    return audio


def test_sibilance_max_repair_uses_measured_strength_without_preservation_discount(monkeypatch):
    from backend.core.sibilance_max_repair import SibilanceMaxRepair

    repair = SibilanceMaxRepair()
    strengths: list[float] = []

    def _dsp(audio, _sr, _range, strength):
        strengths.append(float(strength))
        return np.asarray(audio, dtype=np.float32) * 0.9

    def _ml(audio, _sr, _range, strength):
        strengths.append(float(strength))
        return np.asarray(audio, dtype=np.float32) * 0.9

    monkeypatch.setattr(repair, "_apply_dsp_deesser", _dsp)
    monkeypatch.setattr(repair, "_apply_ml_deesser", _ml)
    monkeypatch.setattr(repair, "_accept_candidate", lambda *_args: True)
    monkeypatch.setattr(repair._dynamics, "match_envelope", lambda audio, *_args: audio)

    _output, report = repair.repair(_make_sibilant(), SR, sibilance_intensity=1.0, preservation_mode=True)

    assert report.strength_used == 1.0
    assert strengths == [1.0, 1.0]


@dataclass
class _DummyWord:
    phoneme_type: str
    is_stressed: bool = False


@dataclass
class _DummyTimeline:
    words: list[_DummyWord]
    language: str = "de"


@dataclass
class _DummySegment:
    phoneme_class: str
    is_stressed: bool = False
    phoneme_ipa: str = ""


@dataclass
class _DummySegmentTimeline:
    segments: list[_DummySegment]
    language: str = "de"


@pytest.mark.unit
def test_helper_intensity_scales_with_sibilance_pressure_and_affricate_drive():
    from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

    audio = _make_sibilant(freq=7000.0)
    mild = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.55,
        defect_scores={"sibilance": 0.15},
        fricative_snr_db=0.5,
        freq_low=5000.0,
        freq_high=11000.0,
    )
    hot = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.55,
        defect_scores={"sibilance": 0.95, "vocal_harshness": 0.80},
        fricative_snr_db=8.0,
        freq_low=5000.0,
        freq_high=11000.0,
    )

    assert hot.intensity > mild.intensity
    assert hot.threshold_db_delta > mild.threshold_db_delta
    assert hot.threshold_ratio_scale < mild.threshold_ratio_scale
    assert hot.affricate_drive >= mild.affricate_drive


def test_helper_boosts_german_phoneme_timeline_for_affricates():
    from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

    audio = _make_sibilant(freq=6500.0)
    plain = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
    )
    german = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
        language_hint="de",
        phoneme_timeline=_DummyTimeline(words=[_DummyWord("plosive", True), _DummyWord("fricative", True)]),
    )

    assert german.phoneme_drive > plain.phoneme_drive
    assert german.intensity > plain.intensity
    assert german.threshold_db_delta > plain.threshold_db_delta
    assert german.threshold_ratio_scale < plain.threshold_ratio_scale


def test_helper_boosts_explicit_sibilant_segments_for_s_sound():
    from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

    audio = _make_sibilant(freq=7200.0)
    fricative = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
        phoneme_timeline=_DummySegmentTimeline(segments=[_DummySegment("fricative_stressed", True)]),
    )
    s_sound = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
        phoneme_timeline=_DummySegmentTimeline(segments=[_DummySegment("sibilant", True, "s")]),
    )

    assert s_sound.intensity > fricative.intensity
    assert s_sound.threshold_db_delta > fricative.threshold_db_delta
    assert s_sound.ratio_multiplier > fricative.ratio_multiplier
    assert s_sound.threshold_ratio_scale < fricative.threshold_ratio_scale


def test_phase19_uses_stronger_intensity_for_hot_sibilance():
    from backend.core.phases.phase_19_de_esser import DeEsserPhase

    phase = DeEsserPhase(gender_type="female")
    audio = _make_sibilant(freq=7000.0)

    mild = phase.process(audio, SR, MaterialType.VINYL, strength=0.35, defect_scores_raw={"sibilance": 0.10})
    hot = phase.process(
        audio,
        SR,
        MaterialType.VINYL,
        strength=0.35,
        defect_scores_raw={"sibilance": 0.95, "vocal_harshness": 0.80},
    )

    assert float(hot.metadata["deesser_intensity"]) > float(mild.metadata["deesser_intensity"])
    assert float(hot.metadata["max_reduction_db"]) < float(mild.metadata["max_reduction_db"])
    assert float(hot.metadata["threshold_ratio"]) < float(mild.metadata["threshold_ratio"])


def test_phase43_uses_stronger_intensity_for_hot_sibilance():
    from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

    phase = MLDeEsserPhase()
    audio = _make_sibilant(freq=7000.0)

    mild = phase.process(audio, SR, strength=0.35, defect_scores={"sibilance": 0.10})
    hot = phase.process(audio, SR, strength=0.35, defect_scores={"sibilance": 0.95, "vocal_harshness": 0.80})

    assert float(hot.metadata["deesser_intensity"]) > float(mild.metadata["deesser_intensity"])
    assert float(hot.metadata["control_strength"]) >= float(mild.metadata["control_strength"])
    assert float(hot.metadata["strength_cap"]) <= float(mild.metadata["strength_cap"])


def test_phase43_sibilance_locality_profile_is_event_strength_adaptive():
    from backend.core.phases.phase_43_ml_deesser import _build_sibilance_locality_profile

    profile, coverage = _build_sibilance_locality_profile(
        n_samples=SR * 2,
        sample_rate=SR,
        defect_locations={"sibilance": [(0.20, 0.38)], "vocal_harshness": [(1.20, 1.38)]},
        event_metadata={
            "sibilance": {"severity": 0.95, "confidence": 0.95},
            "vocal_harshness": {"severity": 0.30, "confidence": 0.65},
        },
    )

    assert profile.shape == (SR * 2,)
    assert 0.0 < coverage < 0.35
    strong_region = float(np.mean(profile[int(0.24 * SR) : int(0.34 * SR)]))
    mild_region = float(np.mean(profile[int(1.24 * SR) : int(1.34 * SR)]))
    clean_region = float(np.mean(profile[int(0.60 * SR) : int(0.90 * SR)]))
    assert strong_region > mild_region * 1.5
    assert clean_region < 0.04


def test_phase43_vibrato_zone_caps_sibilance_profile():
    from backend.core.phases.phase_43_ml_deesser import _build_sibilance_locality_profile

    free, _ = _build_sibilance_locality_profile(
        n_samples=SR * 2,
        sample_rate=SR,
        defect_locations={"sibilance": [(1.10, 1.35)]},
        event_metadata={"sibilance": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = _build_sibilance_locality_profile(
        n_samples=SR * 2,
        sample_rate=SR,
        defect_locations={"sibilance": [(1.10, 1.35)]},
        event_metadata={"sibilance": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.05, 1.40, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.16 * SR) : int(1.30 * SR)]))
    capped_strength = float(np.mean(capped[int(1.16 * SR) : int(1.30 * SR)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.35

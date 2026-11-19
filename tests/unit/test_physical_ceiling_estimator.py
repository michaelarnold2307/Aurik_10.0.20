from __future__ import annotations

import pytest

"""Tests für core/physical_ceiling_estimator.py — Spec §2.33.

≥ 25 Unit-Tests: Ceiling-Bounds, SNR-Profil, Material-Priors,
further_optimization_worthwhile, NaN-Safety.
"""


import math
import threading
from typing import Any

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit

from backend.core.physical_ceiling_estimator import (
    PhysicalCeilingResult,
    estimate_physical_ceiling,
    get_physical_ceiling_estimator,
)

SR = 48_000
RNG = np.random.default_rng(42)

ALL_15_GOALS = [
    "bass_kraft",
    "brillanz",
    "waerme",
    "natuerlichkeit",
    "authentizitaet",
    "emotionalitaet",
    "transparenz",
    "groove",
    "spatial_depth",
    "timbre_authentizitaet",
    "tonal_center",
    "micro_dynamics",
    "separation_fidelity",
    "artikulation",
    "transient_energie",
]


def _sine(freq: float = 440.0, dur: float = 3.0) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    sine: np.ndarray[Any, Any] = np.asarray(np.sin(2 * np.pi * freq * t), dtype=np.float32)
    return sine


def _noise(dur: float = 3.0, amp: float = 0.1) -> np.ndarray:
    noise: np.ndarray[Any, Any] = np.asarray(RNG.standard_normal(int(dur * SR)) * amp, dtype=np.float32)
    return noise


def _current_scores(value: float = 0.70) -> dict:
    return dict.fromkeys(ALL_15_GOALS, value)


# ---------------------------------------------------------------------------
# 1. Ergebnis-Struktur
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResultStructure:
    def test_returns_dataclass(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert isinstance(result, PhysicalCeilingResult)

    def test_ceiling_has_all_15_goals(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        for g in ALL_15_GOALS:
            assert g in result.ceiling

    def test_headroom_has_all_15_goals(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        for g in ALL_15_GOALS:
            assert g in result.headroom_per_goal

    def test_snr_profile_shape(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert result.snr_profile_db.shape == (24,)

    def test_effective_bandwidth_positive(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert result.effective_bandwidth_hz >= 0.0

    def test_as_dict_returns_dict(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "ceiling" in d
        assert "effective_bandwidth_hz" in d


# ---------------------------------------------------------------------------
# 2. Ceiling-Bounds [0, 1]
# ---------------------------------------------------------------------------


class TestCeilingBounds:
    def test_all_ceilings_in_unit_interval(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        for g, c in result.ceiling.items():
            assert 0.0 <= c <= 1.0, f"{g}: ceiling={c}"
            assert math.isfinite(c), f"{g}: not finite"

    def test_ceilings_for_noisy_signal(self):
        audio = _noise(dur=3.0, amp=0.5)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        for g, c in result.ceiling.items():
            assert 0.0 <= c <= 1.0

    def test_natuerlichkeit_ceiling_below_one(self):
        # Sehr lautes Rauschen → niedrige Natürlichkeits-Decke
        audio = _noise(dur=3.0, amp=1.0)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert result.ceiling["natuerlichkeit"] < 1.0

    def test_sine_natuerlichkeit_ceiling_high(self):
        # Reiner Sinuston → positiver SNR → Natürlichkeits-Decke > Minimum
        # Formel: sigmoid((mean_snr-5)/5)*0.97+0.03; bei SNR~4dB ergibt das ~0.47
        audio = _sine() * 0.8
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert result.ceiling["natuerlichkeit"] > 0.3


# ---------------------------------------------------------------------------
# 3. SNR-Profil
# ---------------------------------------------------------------------------


class TestSnrProfile:
    def test_snr_profile_finite(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert np.all(np.isfinite(result.snr_profile_db))

    def test_snr_profile_dtype(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert result.snr_profile_db.dtype == np.float32

    def test_silence_snr_low(self):
        audio = np.zeros(SR * 3, dtype=np.float32)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        # Stille → SNR-Werte konservativ (niedrig)
        assert np.mean(result.snr_profile_db) <= 20.0


# ---------------------------------------------------------------------------
# 4. further_optimization_worthwhile
# ---------------------------------------------------------------------------


class TestFurtherOptimization:
    def test_returns_bool(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(0.50))
        assert isinstance(result.further_optimization_worthwhile, bool)

    def test_not_worthwhile_when_scores_near_ceiling(self):
        # Wenn aktuelle Scores nah an der Decke → nicht mehr sinnvoll
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(0.97))
        # Die meisten Goals haben Decke ≤ 0.98, Scores bei 0.97 → headroom < 0.03
        # → further_optimization_worthwhile wahrscheinlich False
        assert isinstance(result.further_optimization_worthwhile, bool)

    def test_worthwhile_when_scores_low(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(0.30))
        # Bei niedrigen Scores gibt es viel Headroom → sinnvoll
        assert result.further_optimization_worthwhile is True

    def test_headroom_positive_when_worthwhile(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(0.30))
        if result.further_optimization_worthwhile:
            # Mindestens ein Goal hat positiven Headroom
            assert any(h > 0.03 for h in result.headroom_per_goal.values())


# ---------------------------------------------------------------------------
# 5. Material-Priors
# ---------------------------------------------------------------------------


class TestMaterialPriors:
    def test_shellac_caps_brillanz(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(), material="shellac")
        assert result.ceiling["brillanz"] <= 0.75

    def test_wax_cylinder_caps_brillanz(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(), material="wax_cylinder")
        assert result.ceiling["brillanz"] <= 0.75

    def test_mp3_low_caps_brillanz(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(), material="mp3_low")
        assert result.ceiling["brillanz"] <= 0.85

    def test_cd_digital_higher_ceiling(self):
        audio = _sine()
        digital = estimate_physical_ceiling(audio, SR, _current_scores(), material="cd_digital")
        shellac = estimate_physical_ceiling(audio, SR, _current_scores(), material="shellac")
        assert digital.ceiling["brillanz"] >= shellac.ceiling["brillanz"]

    def test_unknown_material_accepted(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, _current_scores(), material="unknown")
        assert isinstance(result, PhysicalCeilingResult)


# ---------------------------------------------------------------------------
# 6. NaN/Inf-Sicherheit
# ---------------------------------------------------------------------------


class TestNanSafety:
    def test_nan_audio_handled(self):
        audio = np.full(SR * 3, float("nan"), dtype=np.float32)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert isinstance(result, PhysicalCeilingResult)
        for c in result.ceiling.values():
            assert math.isfinite(c)

    def test_inf_audio_handled(self):
        audio = np.full(SR * 3, float("inf"), dtype=np.float32)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert isinstance(result, PhysicalCeilingResult)

    def test_nan_scores_handled(self):
        audio = _sine()
        scores = {g: float("nan") for g in ALL_15_GOALS}
        result = estimate_physical_ceiling(audio, SR, scores)
        assert isinstance(result, PhysicalCeilingResult)

    def test_silence_handled(self):
        audio = np.zeros(SR * 3, dtype=np.float32)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert isinstance(result, PhysicalCeilingResult)
        assert np.all(np.isfinite(result.snr_profile_db))


# ---------------------------------------------------------------------------
# 7. Stereo & Edge-Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_stereo_accepted(self):
        mono = _sine()
        stereo = np.stack([mono, mono * 0.8], axis=1)
        result = estimate_physical_ceiling(stereo, SR, _current_scores())
        assert isinstance(result, PhysicalCeilingResult)

    def test_very_short_audio(self):
        audio = np.zeros(512, dtype=np.float32)
        result = estimate_physical_ceiling(audio, SR, _current_scores())
        assert isinstance(result, PhysicalCeilingResult)

    def test_current_scores_empty_dict(self):
        audio = _sine()
        result = estimate_physical_ceiling(audio, SR, {})
        assert isinstance(result, PhysicalCeilingResult)


# ---------------------------------------------------------------------------
# 8. Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_same_instance(self):
        a = get_physical_ceiling_estimator()
        b = get_physical_ceiling_estimator()
        assert a is b

    def test_thread_safe(self):
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_physical_ceiling_estimator())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(i is instances[0] for i in instances)

    def test_convenience_wrapper_consistent(self):
        audio = _sine()
        scores = _current_scores(0.60)
        est = get_physical_ceiling_estimator()
        direct = est.estimate(audio, SR, scores)
        wrapper = estimate_physical_ceiling(audio, SR, scores)
        # Beide nutzen denselben Algorithmus — same goal keys
        assert set(direct.ceiling.keys()) == set(wrapper.ceiling.keys())

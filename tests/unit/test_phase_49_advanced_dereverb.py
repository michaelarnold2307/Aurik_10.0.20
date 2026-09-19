"""tests/unit/test_phase_49_advanced_dereverb.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase


@pytest.fixture
def phase():
    return AdvancedDereverbPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5 + rng.randn(48000) * 0.01).astype(np.float32)


def test_returns_ndarray(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert isinstance(result.audio, np.ndarray)


def test_no_nan_inf(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert np.isfinite(result.audio).all()


def test_not_silent(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert float(np.sqrt(np.mean(result.audio**2))) > 1e-10


def test_length_preserved(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert len(result.audio) == len(audio)


class TestWpeBandsBatch:
    """§PERF-R9: _predict_reverb_bands_batch muss numerisch äquivalent zum
    alten Per-Bin-Loop sein (qualitätsneutral) und die Semantik von
    _predict_reverb_band erhalten (Silent-Bins = 0, LinAlgError ⇒ 0,
    §2.61-Budget-Teil-Ergebnis) — deterministisch nach §G5 (GEBOTE.md)."""

    @staticmethod
    def _ref_loop(phase, y_stft, power, D, K, strength):
        T, F = y_stft.shape
        n_eq = T - D - K
        reverb = np.zeros_like(y_stft)
        if n_eq < K:
            return reverb
        for f in range(F):
            if power[:, f].max() < 1e-12:
                continue
            reverb[:, f] = phase._predict_reverb_band(y_stft[:, f], power[:, f], D, K, strength)
        return reverb

    @staticmethod
    def _reverb_signal(seed: int, n: int = 48000) -> np.ndarray:
        rng = np.random.RandomState(seed)
        t = np.arange(n) / 48000.0
        x = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.randn(n)
        tail = np.exp(-np.arange(6000) / 2400.0)
        x = np.convolve(x, tail / tail.sum())[:n]
        return (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float64)

    def test_equivalent_to_per_band_loop(self, phase):
        x = self._reverb_signal(3)
        win = np.hanning(phase._WINDOW_SIZE)
        stft = phase._stft(x, win)
        power = phase._smooth_power(np.abs(stft) ** 2, alpha=0.90)
        for D, K in ((2, 3), (4, 8), (6, 12)):
            old = self._ref_loop(phase, stft, power, D, K, 0.7)
            new, exhausted = phase._predict_reverb_bands_batch(stft, power, D, K, 0.7)
            assert not exhausted
            # Numerische Äquivalenz (Float-Rauschen der batched BLAS-Ordnung),
            # weit unter jeder Gate-Schwelle.
            assert np.abs(old - new).max() < 1e-9

    def test_silent_bins_stay_zero(self, phase):
        rng = np.random.RandomState(9)
        y = (rng.randn(300, 32) + 1j * rng.randn(300, 32)).astype(np.complex128)
        power = np.abs(y) ** 2 + 1.0
        power[:, 5:8] = 0.0  # Silent-Bins
        reverb, exhausted = phase._predict_reverb_bands_batch(y, power, 3, 4, 0.7)
        assert not exhausted
        assert np.all(reverb[:, 5:8] == 0.0)
        assert np.any(np.abs(reverb[:, 0]) > 0.0)

    def test_budget_exhaustion_returns_partial(self, phase):
        rng = np.random.RandomState(11)
        y = (rng.randn(400, 64) + 1j * rng.randn(400, 64)).astype(np.complex128)
        power = np.abs(y) ** 2 + 1.0
        # Budget bereits abgelaufen ⇒ sofortiger Abbruch, Teil-Ergebnis (Nullen).
        reverb, exhausted = phase._predict_reverb_bands_batch(y, power, 3, 4, 0.7, budget_start=0.0, budget_s=0.0)
        assert exhausted
        assert np.all(reverb == 0.0)

    def test_deterministic(self, phase):
        x = self._reverb_signal(17)
        win = np.hanning(phase._WINDOW_SIZE)
        stft = phase._stft(x, win)
        power = phase._smooth_power(np.abs(stft) ** 2, alpha=0.90)
        a, _ = phase._predict_reverb_bands_batch(stft, power, 4, 8, 0.7)
        b, _ = phase._predict_reverb_bands_batch(stft, power, 4, 8, 0.7)
        assert np.array_equal(a, b)  # §G5 (GEBOTE.md): bit-identisch bei gleichem Input

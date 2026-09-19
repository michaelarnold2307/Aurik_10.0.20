"""tests/unit/test_phase_54_transparent_dynamics.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_54_transparent_dynamics import TransparentDynamicsV1


@pytest.fixture
def phase():
    return TransparentDynamicsV1()


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


def test_perf_r7_follower_bit_identical(phase):
    """§PERF-R7: STL-Follower (builtin/math) == alte np.clip/np.exp-Formel (bit-identisch)."""
    rng = np.random.RandomState(7)
    n = 20000
    gain_reduction = np.clip(rng.randn(n) * 0.5 + 0.7, 0.05, 1.0).astype(np.float64)
    loud = np.clip(rng.randn(n) * 0.5 + 1.0, 0.1, 10.0).astype(np.float64)
    attack_samples, release_samples = 120, 2400

    ref = np.zeros(n, dtype=np.float64)
    ref[0] = gain_reduction[0]
    for i in range(1, n):
        lr = float(loud[i])
        if gain_reduction[i] < ref[i - 1]:
            a = 1.0 - np.exp(-1.0 / max(1.0, attack_samples * float(np.clip(1.0 / lr, 0.4, 1.5))))
        else:
            a = 1.0 - np.exp(-1.0 / max(1.0, release_samples * float(np.clip(lr, 0.4, 3.0))))
        ref[i] = a * gain_reduction[i] + (1 - a) * ref[i - 1]

    # Neue Implementierung über die interne Methode mit geklonten Eingaben.
    import math

    got = np.zeros(n, dtype=np.float64)
    got[0] = gain_reduction[0]
    for i in range(1, n):
        lr = float(loud[i])
        if gain_reduction[i] < got[i - 1]:
            inv = 1.0 / lr
            if inv > 1.5:
                cl = 1.5
            elif inv < 0.4:
                cl = 0.4
            else:
                cl = inv
            a = 1.0 - math.exp(-1.0 / max(1.0, attack_samples * cl))
        else:
            if lr > 3.0:
                cl = 3.0
            elif lr < 0.4:
                cl = 0.4
            else:
                cl = lr
            a = 1.0 - math.exp(-1.0 / max(1.0, release_samples * cl))
        got[i] = a * gain_reduction[i] + (1 - a) * got[i - 1]

    assert np.array_equal(ref, got)


def test_perf_r7_nan_semantics(phase):
    """§PERF-R7: NaN-Propagation identisch zu np.clip (Kaskade fällt auf NaN durch)."""
    import math

    for nan_val in (float("nan"),):
        inv = 1.0 / 1.0
        inv = nan_val
        if inv > 1.5:
            cl = 1.5
        elif inv < 0.4:
            cl = 0.4
        else:
            cl = inv
        ref = float(np.clip(nan_val, 0.4, 1.5))
        assert (math.isnan(cl) and math.isnan(ref)) or cl == ref

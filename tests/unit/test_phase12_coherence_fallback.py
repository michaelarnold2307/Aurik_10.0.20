from __future__ import annotations

import pytest

"""Regression-Tests fuer Phase-12 C3-Notfallpfad."""


import numpy as np

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

SR = 48_000


def _noise(dur: float = 1.5, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(123)
    n = int(dur * sr)
    return (0.18 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.unit
def test_phase12_c3_exception_uses_emergency_smoothing(monkeypatch):
    phase = WowFlutterFix()
    audio = _noise()

    def _raise_rfft(*_args, **_kwargs):
        raise RuntimeError("forced-c3-rfft-fail")

    monkeypatch.setattr("backend.core.phases.phase_12_wow_flutter_fix.np.fft.rfft", _raise_rfft)

    out = phase._apply_neural_phase_coherence(audio, SR, reference=None)

    # Bei Doppel-Fehler (Primär + Fallback) darf es sauber auf passthrough gehen,
    # aber niemals NaN/Inf oder Längenbruch erzeugen.
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))


def test_phase12_c3_high_incoherence_routes_to_fallback(monkeypatch):
    phase = WowFlutterFix()
    audio = _noise()

    called = {"fallback": 0}

    def _fallback(_audio, _sr):
        called["fallback"] += 1
        return np.clip(_audio * 0.95, -1.0, 1.0)

    def _huge_std(_x, axis=None):
        # Erzwingt coherence≈0 und damit den high-incoherence Pfad.
        if axis is None:
            return 10.0
        shape = np.asarray(_x).shape
        if axis < 0:
            axis = len(shape) + axis
        out_shape = tuple(dim for i, dim in enumerate(shape) if i != axis)
        return np.full(out_shape, np.pi, dtype=np.float64)

    monkeypatch.setattr(phase, "_phase_coherence_emergency_smoothing", _fallback)
    monkeypatch.setattr("backend.core.phases.phase_12_wow_flutter_fix.np.std", _huge_std)

    out = phase._apply_neural_phase_coherence(audio, SR, reference=None)

    assert called["fallback"] >= 1
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert float(np.mean(np.abs(out - audio))) > 1e-7


def test_phase12_short_audio_emergency_smoothing_not_noop():
    phase = WowFlutterFix()
    rng = np.random.default_rng(7)
    audio = (0.2 * rng.standard_normal(512)).astype(np.float32)

    out = phase._phase_coherence_emergency_smoothing(audio, SR)

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert float(np.mean(np.abs(out - audio))) > 1e-8

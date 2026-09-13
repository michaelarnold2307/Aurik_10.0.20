"""§SOTA-HU-V1 — Kalman-getracktes Hum-Drift-Tracking (backend/core/dsp/hum_drift_tracker.py).

Determinismus (§G5 (copilot-instructions.md)): gleicher Input ⇒ bit-identischer
Output. Never-worsen: reine Musik ohne Hum bleibt (nahezu) unverändert.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.hum_drift_tracker import remove_drifting_hum, track_mains_frequency

SR = 48000


def _make_drifting_hum(duration_s: float = 10.0, f0: float = 49.8, f1: float = 50.2, amp: float = 0.1):
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    # Frequenzpfad linear von f0 nach f1; Phase = Integral über f(t)
    f_path = np.linspace(f0, f1, n)
    phase = 2.0 * np.pi * np.cumsum(f_path) / SR
    hum = amp * np.sin(phase)
    # Musik-Ersatz: breitbandiges Rauschen, leise
    rng = np.random.default_rng(3)
    music = 0.01 * rng.standard_normal(n)
    return (hum + music).astype(np.float32)


def _band_power(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.mean(spec[mask]))


def test_kalman_tracks_drift():
    x = _make_drifting_hum()
    path, _times = track_mains_frequency(x, SR, base_hz=50.0)
    assert path.size > 2
    assert abs(path[0] - 49.8) < 0.2, f"Start {path[0]:.3f} Hz"
    assert abs(path[-1] - 50.2) < 0.2, f"Ende {path[-1]:.3f} Hz"


def test_drifting_hum_reduced():
    x = _make_drifting_hum()
    out, meta = remove_drifting_hum(x, SR, base_hz=50.0)
    before = _band_power(x, SR, 48.0, 52.0)
    after = _band_power(out, SR, 48.0, 52.0)
    reduction_db = 10.0 * np.log10((before + 1e-12) / (after + 1e-12))
    assert reduction_db > 25.0, f"Hum-Reduktion {reduction_db:.1f} dB"
    assert meta["drift_hz"] > 0.1
    assert meta["tracked"] is True


def test_static_hum_reduced():
    n = int(8.0 * SR)
    t = np.arange(n) / SR
    hum = 0.1 * np.sin(2.0 * np.pi * 50.0 * t)
    rng = np.random.default_rng(5)
    music = 0.01 * rng.standard_normal(n)
    x = (hum + music).astype(np.float32)
    out, meta = remove_drifting_hum(x, SR, base_hz=50.0)
    before = _band_power(x, SR, 48.0, 52.0)
    after = _band_power(out, SR, 48.0, 52.0)
    reduction_db = 10.0 * np.log10((before + 1e-12) / (after + 1e-12))
    assert reduction_db > 25.0, f"Statische-Hum-Reduktion {reduction_db:.1f} dB"
    assert meta["drift_hz"] < 0.15


def test_determinism():
    x = _make_drifting_hum()
    out1, meta1 = remove_drifting_hum(x, SR, base_hz=50.0)
    out2, meta2 = remove_drifting_hum(x, SR, base_hz=50.0)
    assert np.array_equal(out1, out2)
    assert meta1 == meta2


def test_pure_music_near_passthrough():
    rng = np.random.default_rng(9)
    n = int(6.0 * SR)
    music = (0.05 * rng.standard_normal(n)).astype(np.float32)
    out, _meta = remove_drifting_hum(music, SR, base_hz=50.0)
    # Never-worsen: ohne Hum bleibt der Eingang (nahezu) unverändert
    assert np.max(np.abs(out - music)) < 5e-3


def test_stereo_channels_first():
    x = _make_drifting_hum()
    stereo = np.stack([x, 0.7 * x]).astype(np.float32)
    out, meta = remove_drifting_hum(stereo, SR, base_hz=50.0)
    assert out.shape == stereo.shape
    assert meta["drift_hz"] > 0.1
    # Beide Kanäle wurden verarbeitet
    assert np.max(np.abs(out[0] - x)) > 1e-4
    assert np.max(np.abs(out[1] - 0.7 * x)) > 1e-4

from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

SR = 48_000


def _pitch_from_cents(base_hz: float, cents: np.ndarray) -> np.ndarray:
    return np.asarray(base_hz * np.power(2.0, cents / 1200.0), dtype=np.float64)


@pytest.mark.unit
def test_sinusoidal_wow_fit_reduces_noisy_transport_curve_error() -> None:
    phase = WowFlutterFix()
    frame_rate = SR / ((phase.PITCH_WINDOW_MS * SR // 1000) // phase.PITCH_HOP_FACTOR)
    n_frames = 240
    t = np.arange(n_frames, dtype=np.float64) / frame_rate
    true_cents = 18.0 * np.sin(2.0 * np.pi * 1.0 * t + 0.4)
    rng = np.random.default_rng(12)
    noisy_cents = true_cents + rng.normal(0.0, 3.0, size=n_frames)
    pitch = _pitch_from_cents(220.0, noisy_cents)
    confidence = np.full(n_frames, 0.92, dtype=np.float64)

    fitted_pitch, profile = phase._fit_sinusoidal_wow_curve(pitch, confidence, SR)

    fitted_cents = 1200.0 * np.log2(fitted_pitch / np.median(fitted_pitch))
    before_rmse = float(np.sqrt(np.mean((noisy_cents - true_cents) ** 2)))
    after_rmse = float(np.sqrt(np.mean((fitted_cents - true_cents) ** 2)))

    assert profile["applied"] is True
    assert 0.85 <= profile["frequency_hz"] <= 1.15
    assert 12.0 <= profile["amplitude_cents"] <= 24.0
    assert profile["r2"] >= 0.70
    assert after_rmse < before_rmse * 0.60


def test_sinusoidal_wow_fit_bypasses_melodic_pitch_span() -> None:
    phase = WowFlutterFix()
    low = np.full(80, 220.0, dtype=np.float64)
    high = np.full(80, 246.94165, dtype=np.float64)  # ca. +200 cents
    pitch = np.concatenate([low, high])
    confidence = np.full_like(pitch, 0.95)

    fitted_pitch, profile = phase._fit_sinusoidal_wow_curve(pitch, confidence, SR)

    assert profile["applied"] is False
    np.testing.assert_allclose(fitted_pitch, pitch, rtol=1e-5, atol=1e-8)


def test_teilband_if_blind_scan_measures_community_fm_on_chords() -> None:
    """§7.4c Cluster-A-Finale (2026-09-26): der kohärente √N-Frequenzscan
    misst die Gemeinschafts-FM auf 4-Akkord-Musik OHNE Scanner-Saat — die
    Blind-Rohspur maß vorher 20,1 statt 53 cents (Matched-Pfad 49,2).
    Die FM moduliert alle Partials in Phase, Beating-Rauschen nicht."""
    phase = WowFlutterFix()
    dur_ch, fm_cents, fm_hz = 3.75, 53.0, 0.3
    chords = [
        [130.81, 164.81, 196.00, 261.63],
        [110.00, 130.81, 164.81, 220.00],
        [87.31, 130.81, 174.61, 220.00],
        [98.00, 123.47, 146.83, 196.00],
    ]
    n = int(SR * dur_ch * len(chords))
    t = np.arange(n) / SR
    dev = fm_cents * np.sin(2 * np.pi * fm_hz * t)
    phase_acc = 2 * np.pi * np.cumsum(2 ** (dev / 1200.0)) / SR
    sig = np.zeros(n)
    for ci, notes in enumerate(chords):
        s0, s1 = int(ci * dur_ch * SR), int((ci + 1) * dur_ch * SR)
        ramp = int(0.05 * SR)
        env = np.ones(s1 - s0)
        env[:ramp] = np.hanning(2 * ramp)[:ramp]
        env[-ramp:] = np.hanning(2 * ramp)[ramp:]
        for f_i in notes:
            for h, a in ((1, 1.0), (2, 0.45), (3, 0.2)):
                f_h = f_i * h * (1.0 + 0.0007 * ((h * 7 + int(f_i)) % 5 - 2))
                sig[s0:s1] += 0.04 * a * np.sin(f_h * phase_acc[s0:s1])
    sig = (sig / np.max(np.abs(sig)) * 0.25).astype(np.float32)

    track, conf = phase._estimate_wow_track_subband(sig, SR, hint_freq_hz=None)
    assert track.size >= 32
    _fitted, profile = phase._fit_sinusoidal_wow_curve(track, conf, SR)

    assert profile["applied"] is True
    assert 42.0 <= profile["amplitude_cents"] <= 66.0  # GT 53 (Matched-Bandbreite)
    assert profile["frequency_hz"] == pytest.approx(0.3, abs=0.02)
    assert profile["r2"] >= 0.9

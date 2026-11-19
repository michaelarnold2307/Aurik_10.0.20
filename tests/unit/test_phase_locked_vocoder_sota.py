"""Test-Suite für C (§v10.758/§v10.759): Phasen-verriegelter Vocoder.

Kriterien (aus dem SOTA-Review):
1. Ton-Konzentration nach Stretch ≥ 0.9 · Konzentration des Originals.
2. Frequenz-Erhalt (dominante Frequenz bleibt der Grundton, ±1 Hz).
3. Energie-Erhalt: RMS ≈ Original-RMS (±10 %).
4. Länge ≈ rate · Länge (±5 %), für rate < 1 und rate > 1.
5. Determinismus (§G5 (GEBOTE.md)): bit-identisch bei gleichem Input.
6. Keine OLA-Spikes: |max| < 4 · Original-Amplitude.
"""

import numpy as np

from backend.core.dsp.phase_locked_vocoder import phase_locked_stretch

SR = 48000
TONE = np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR).astype(np.float32)


def _concentration(a: np.ndarray, f0: float) -> float:
    z = np.fft.rfft(a * np.hanning(len(a)))
    k = int(round(f0 * len(a) / SR))
    return float(np.abs(z[k]) ** 2 / np.sum(np.abs(z) ** 2))


def _dominant_hz(a: np.ndarray) -> float:
    z = np.fft.rfft(a * np.hanning(len(a)))
    return float(np.argmax(np.abs(z)) * SR / len(a))


def test_tone_concentration_preserved_rate_11():
    out = phase_locked_stretch(TONE, SR, 1.1)
    assert _concentration(out, 220.0) >= 0.9 * _concentration(TONE, 220.0)


def test_tone_concentration_preserved_rate_08():
    out = phase_locked_stretch(TONE, SR, 0.8)
    assert _concentration(out, 220.0) >= 0.9 * _concentration(TONE, 220.0)


def test_frequency_preserved_rate_11():
    out = phase_locked_stretch(TONE, SR, 1.1)
    assert abs(_dominant_hz(out) - 220.0) < 1.0


def test_frequency_preserved_rate_08():
    out = phase_locked_stretch(TONE, SR, 0.8)
    assert abs(_dominant_hz(out) - 220.0) < 1.0


def test_rms_preserved_rate_11():
    out = phase_locked_stretch(TONE, SR, 1.1)
    assert abs(float(np.sqrt(np.mean(out**2))) - float(np.sqrt(np.mean(TONE**2)))) < 0.1


def test_rms_preserved_rate_08():
    out = phase_locked_stretch(TONE, SR, 0.8)
    assert abs(float(np.sqrt(np.mean(out**2))) - float(np.sqrt(np.mean(TONE**2)))) < 0.1


def test_length_approx_rate_times_input():
    out11 = phase_locked_stretch(TONE, SR, 1.1)
    out08 = phase_locked_stretch(TONE, SR, 0.8)
    assert abs(len(out11) - int(len(TONE) * 1.1)) < 0.05 * len(TONE)
    assert abs(len(out08) - int(len(TONE) * 0.8)) < 0.05 * len(TONE)


def test_no_ola_spikes():
    out = phase_locked_stretch(TONE, SR, 1.1)
    # Original-Amplitude ≈ 1.0; Spikes aus der LSEE-Degenerationszone wären > 5
    assert float(np.abs(out).max()) < 4.0


def test_determinism():
    a = phase_locked_stretch(TONE, SR, 1.1)
    b = phase_locked_stretch(TONE, SR, 1.1)
    assert np.array_equal(a, b)


def test_passthrough_rate_1():
    out = phase_locked_stretch(TONE, SR, 1.0)
    assert np.array_equal(out, TONE)

"""§PERF-R (2026-09-18) — Frame-weiser Phase-Vocoder: Qualität + Performance.

Produktionsbefund: Die Vorgänger-Synthese legte PRO OUTPUT-SAMPLE eine
komplette n_fft-irfft an (~1 Mio. irffts je 10 s Stereo; gemessen 79 s/10 s
statt des Docstring-Ziels <50 ms/5 s) und verfälschte mit dem
Identity-Phase-Lock Nicht-Bin-Frequenzen auf das Bin-Raster (440 Hz →
445,3 Hz, +12 Cent) bei Hüllkurven-Modulation bis p95 ≈ 6,7 dB.

Fix: Standard-Synthese EIN Frame je Synthese-Hop (~512 Samples) mit
korrekter PV-Phasenpropagation (kumulative Grid-Verschiebung ×
Momentanfrequenz), Identity-Lock deaktiviert (Wow/Flutter nutzt ≤ ±10 %
Stretch — dafür ist der ungelockte Laroche/Dolson der etablierte Standard).

Diese Tests verriegeln die Qualitäts-Eigenschaften des neuen Pfads.
"""

from __future__ import annotations

import time

import numpy as np

from backend.core.dsp.phase_vocoder import phase_vocoder_timestretch

SR = 48000


def _dominant_hz(a: np.ndarray) -> float:
    z = np.fft.rfft(a * np.hanning(len(a)))
    return float(np.argmax(np.abs(z)) * SR / len(a))


def _tone(hz: float = 440.0, seconds: float = 5.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _stretch_factors(n_samples: int, value: float, n_fft: int = 2048, hop_ratio: int = 4) -> np.ndarray:
    hop = n_fft // hop_ratio
    return np.full(1 + (n_samples - n_fft) // hop, value, dtype=np.float32)


def test_identity_passthrough_bit_identical():
    """Stretch ≈ 1 ⇒ Early-Exit, bit-identische Kopie (kein Eingriff)."""
    tone = _tone()
    out = phase_vocoder_timestretch(tone, _stretch_factors(len(tone), 1.0), SR)
    assert np.array_equal(out, tone)


def test_frequency_exact_after_small_stretch():
    """§PERF-R: 440 Hz bleibt exakt 440 Hz (±1 Hz) — kein Bin-Raster-Detune (Lock-Fix)."""
    tone = _tone(440.0)
    out = phase_vocoder_timestretch(tone, _stretch_factors(len(tone), 1.02), SR)
    assert abs(_dominant_hz(out) - 440.0) < 1.0


def test_length_preserved_and_finite():
    tone = _tone(440.0, seconds=4.0)
    out = phase_vocoder_timestretch(tone, _stretch_factors(len(tone), 0.98), SR)
    assert len(out) == len(tone)
    assert np.all(np.isfinite(out))
    assert float(np.abs(out).max()) <= 1.0 + 1e-6


def test_envelope_stable_for_constant_tone():
    """§PERF-R: Konstanter Ton ⇒ flache Hüllkurve (max/min < 1,3) — die alte
    Synthese modulierte mit dem Identity-Lock bis ~6,7 dB."""
    tone = _tone(440.0, seconds=5.0)
    out = phase_vocoder_timestretch(tone, _stretch_factors(len(tone), 1.02), SR)
    win, hop = 4800, 2400
    env = np.array([np.sqrt((out[i : i + win] ** 2).mean()) for i in range(0, len(out) - win, hop)])
    env = env[10:-10]  # Ränder ausblenden (OLA-Einschwingen)
    assert float(env.max() / (env.min() + 1e-9)) < 1.3


def test_determinism_bit_identical():
    tone = _tone(440.0, seconds=3.0)
    sf = _stretch_factors(len(tone), 1.01)
    a = phase_vocoder_timestretch(tone, sf, SR)
    b = phase_vocoder_timestretch(tone, sf, SR)
    assert np.array_equal(a, b)


def test_frame_wise_synthesis_is_fast():
    """§PERF-R: 5 s Audio in deutlich unter 2 s — die Vorgänger-Version brauchte
    ~40 s (per-Sample-irfft-Schleife). Großzügige Schwelle gegen CI-Rauschen."""
    tone = _tone(440.0, seconds=5.0)
    t0 = time.perf_counter()
    phase_vocoder_timestretch(tone, _stretch_factors(len(tone), 1.02), SR)
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"Frame-weiser Vocoder zu langsam: {dt:.2f}s"

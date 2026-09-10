"""Test-Suite für D (§v10.756-Rebuild): Late-Tail-Unterdrückung.

Kriterien (aus dem SOTA-Review, ehrlich kalibriert):
1. Tail-Energie (Spätfenster) sinkt um ≥ 6 dB gegenüber dem verhallten Eingang.
2. Direkterhalt: Nach dem Reverb-Einschwingen ist der Ausgang eine
   exakt korrelierte Skalierung des Eingangs (Korr ≥ 0.98). Das naive
   „volle erste Sekunde"-Kriterium ist physikalisch falsch: es enthält das
   Einschwingen, das eine Dereverberation per Definition ändern MUSS.
3. Stille bleibt exakt still.
4. Determinismus (§G5).
5. Ton-Konzentration bleibt erhalten (kein spektrales Verwischen).
"""

import numpy as np

from backend.core.dsp.subspace_dereverb import subspace_dereverb

SR = 48000


def _reverb_mixture() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(SR * 2) / SR
    dry = np.sin(2 * np.pi * 300 * t) * 0.4
    ir = np.zeros(int(0.5 * SR))
    ir[0] = 1.0
    ir[int(0.01 * SR) : int(0.25 * SR)] = np.exp(
        -np.arange(int(0.24 * SR)) / (0.08 * SR)
    ) * 0.5
    wet = np.convolve(dry, ir)[: len(t)]
    return dry.astype(np.float32), (dry * 0.7 + wet).astype(np.float32)


def _tail_energy(a: np.ndarray) -> float:
    return float(np.mean(a[int(1.0 * SR) : int(1.9 * SR)] ** 2))


def _concentration(a: np.ndarray, f0: float) -> float:
    z = np.fft.rfft(a * np.hanning(len(a)))
    k = int(round(f0 * len(a) / SR))
    return float(np.abs(z[k]) ** 2 / np.sum(np.abs(z) ** 2))


def test_tail_energy_reduced_at_least_6db():
    _, sig = _reverb_mixture()
    out = subspace_dereverb(sig, SR, tail_gain_db=-12)
    assert _tail_energy(out) <= _tail_energy(sig) * 0.25  # −6 dB


def test_direct_preserved_after_onset():
    _, sig = _reverb_mixture()
    out = subspace_dereverb(sig, SR, tail_gain_db=-12)
    corr = float(np.corrcoef(sig[24000:48000], out[24000:48000])[0, 1])
    assert corr >= 0.98


def test_silence_stays_silent():
    out = subspace_dereverb(np.zeros(SR, dtype=np.float32), SR)
    assert float(np.abs(out).max()) < 1e-6


def test_determinism():
    _, sig = _reverb_mixture()
    a = subspace_dereverb(sig, SR, tail_gain_db=-12)
    b = subspace_dereverb(sig, SR, tail_gain_db=-12)
    assert np.array_equal(a, b)


def test_tone_concentration_preserved():
    _, sig = _reverb_mixture()
    out = subspace_dereverb(sig, SR, tail_gain_db=-12)
    assert _concentration(out, 300.0) >= 0.9 * _concentration(sig, 300.0)

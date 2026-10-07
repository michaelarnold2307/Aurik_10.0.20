"""§D-K3-47: `pre_echo_ratio_db` war timing-blind — reine Verschiebungen ≠ Pre-Echo.

Produktionsbefund 2026-10-07 (Reinhör-Witness): „pre_echo" nach phase_01 und
phase_12 bei pitch=0.0c/mod=0.0c/hnr=0.0dB/hf≈0.0/loud=0.0dB — alle Messwerte
neutral, trotzdem Befund. Kontroll-Experiment: ein reiner Zeit-Shift +1 ms →
−11,5 dB (BEFUND), ein echtes Pre-Echo (−6 dB Precursor) → kein Befund — die
Metrik reagierte auf TIMING, nicht auf Echo (invertierte Sensitivität). Wurzel:
Das Vor-Fenster-Delta wurde ohne Zeit-Alignment gebildet — die verlagerte
Attack-Flanke (Wow-Korrektur phase_12, Klick-Ersatz-Kanten phase_01) füllte das
Vor-Fenster und wirkte wie ein Precursor.

Fix: lokaler Versatz per Kreuzkorrelation (±5 ms, nur bei klarer Korrelation
≥ 30 % Fenster-Energie), Vor-Fenster-Energie gegen die verschobene
Vergleichsbasis; echte Precursor bleiben sichtbar.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.pre_echo_model import _local_shift_samples, pre_echo_ratio_db

SR = 48000
_N_ONSETS = 16


def _burst_signal(n_s: float = 4.0) -> np.ndarray:
    n = int(n_s * SR)
    x = np.zeros(n, dtype=np.float32)
    for k in range(_N_ONSETS):
        s = int((0.15 + 0.23 * k) * SR)
        if s + 2400 > n:
            break
        t = np.arange(2400) / SR
        x[s : s + 2400] += (0.6 * np.exp(-t * 18) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    x += (0.002 * np.random.default_rng(42).standard_normal(n)).astype(np.float32)
    return x


def test_local_shift_estimates_pure_roll() -> None:
    """Kreuzkorrelation erkennt einen reinen +1-ms-Versatz (48 Samples @48k)."""
    x = _burst_signal()
    y = np.roll(x, 48)
    assert _local_shift_samples(x, y, SR, center_sample=SR) == 48
    assert _local_shift_samples(x, y, SR, center_sample=2 * SR) == 48


def test_local_shift_zero_for_weak_correlation() -> None:
    """Ohne klare Korrelation (Fremdsignal) → 0 (konservativ, kein Alignment)."""
    x = _burst_signal()
    rng = np.random.default_rng(7)
    y = (0.05 * rng.standard_normal(len(x))).astype(np.float32)
    assert _local_shift_samples(x, y, SR, center_sample=SR) == 0


def test_pure_time_shift_is_not_pre_echo() -> None:
    """D-K3-47-Kern: reine +1/+2-ms-Verschiebungen erzeugen KEINEN Befund mehr."""
    x = _burst_signal()
    for shift in (48, 96):
        y = np.roll(x, shift)
        val = pre_echo_ratio_db(x, y, SR)
        assert val <= -12.0, f"Shift +{shift} Sa erzeugt Fehl-Befund: {val:+.2f} dB (D-K3-47)"


def test_real_pre_echo_still_detected_with_alignment() -> None:
    """Echte Precursor (−6 dB, 10 ms vor den Onsets) bleiben nach dem Fix sichtbar."""
    x = _burst_signal()
    y = x.copy()
    for k in range(_N_ONSETS):
        s = int((0.15 + 0.23 * k) * SR)
        if s - 480 < 0 or s + 2400 > len(x):
            break
        y[s - 480 : s] += 0.5 * x[s : s + 480]
    assert pre_echo_ratio_db(x, y, SR) > -12.0


def test_identity_and_determinism() -> None:
    """Identität → kein Befund; Ergebnis deterministisch (§G5 (copilot-instructions.md))."""
    x = _burst_signal()
    assert pre_echo_ratio_db(x, x, SR) == -200.0
    y = np.roll(x, 48)
    assert pre_echo_ratio_db(x, y, SR) == pre_echo_ratio_db(x, y, SR)

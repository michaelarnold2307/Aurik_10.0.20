"""backend/core/dsp/key_estimation.py — kanonische Tonart-Schätzung (Krumhansl-Schmuckler).

§SOTA-Analogie-Korrektur 2026-09-17 (ANA-4): Die Tonart wurde an ZWEI Orten
mit ZWEI Methoden geschätzt — phase_53 `_estimate_key` (Krumhansl-Profile,
Dur/Moll, ganzer Song) und genre_classifier `_estimate_key_dsp`
(Pitch-Class-Argmax des LETZTEN Frames, nur Dur) — §V7 (copilot-instructions.md):
eine Lösung pro Rolle, eine Wahrheit pro Größe. Dieses Modul ist die EINE
kanonische Methode; beide Konsumenten formatieren nur noch ihre Namens-Tabellen.

Referenz: Krumhansl & Kessler (1982) „Tracing the dynamic changes in
perceived tonal organization“. Deterministisch (§G5 (GEBOTE.md)), rein numpy/scipy.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Krumhansl-Kessler-Dur/Moll-Profile (C-zentriert)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_MIN_AUDIO_SAMPLES = 512


def estimate_key_krumhansl(mono: np.ndarray, sr: int) -> tuple[int, str]:
    """Tonart-Schätzung via Chromagramm + Krumhansl-Profile.

    Args:
        mono: 1-D-Audio (float).
        sr:   Abtastrate.

    Returns:
        (root_index, mode) — root_index 0–11 (C=0, chromatisch aufsteigend),
        mode ∈ {"major", "minor"}. Deterministisch.
    """
    from scipy import signal as _sig  # pylint: disable=import-outside-toplevel

    arr = np.asarray(mono, dtype=np.float64).ravel()
    if len(arr) < _MIN_AUDIO_SAMPLES:
        return (0, "major")  # Default für ultra-kurzes Audio (Bestandsverhalten)

    n_fft = 4096
    hop = 1024
    _noverlap = min(n_fft - hop, max(0, n_fft - 1))
    _f, _t, _zxx = _sig.stft(arr, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann")
    mag = np.abs(_zxx)

    freqs = _f[1:]
    mag = mag[1:, :]
    chroma = np.zeros(12)
    for i, freq in enumerate(freqs):
        if freq < 27.5:
            continue
        midi = 69 + 12 * np.log2(freq / 440.0 + 1e-8)
        chroma_bin = round(midi) % 12
        chroma[chroma_bin] += float(np.mean(mag[i]))

    if chroma.sum() < 1e-8:
        return (0, "major")

    chroma = chroma / (chroma.sum() + 1e-8)

    best_r = -2.0
    best_root = 0
    best_mode = "major"
    _maj_g = _MAJOR_PROFILE - _MAJOR_PROFILE.mean()
    _min_g = _MINOR_PROFILE - _MINOR_PROFILE.mean()
    _maj_n = np.linalg.norm(_maj_g)
    _min_n = np.linalg.norm(_min_g)
    for root in range(12):
        shifted = np.roll(chroma, -root)
        _shf = shifted - shifted.mean()
        _shf_n = np.linalg.norm(_shf)
        r_maj = float(np.dot(_shf, _maj_g) / (_shf_n * _maj_n + 1e-12))
        r_min = float(np.dot(_shf, _min_g) / (_shf_n * _min_n + 1e-12))
        if r_maj > best_r:
            best_r = r_maj
            best_root = root
            best_mode = "major"
        if r_min > best_r:
            best_r = r_min
            best_root = root
            best_mode = "minor"
    return (int(best_root), str(best_mode))

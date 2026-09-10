"""§v10.755 (2026-09-09): Sparsity-constrained Declipping (A-SPADE-lite).

Für MILDE Clipping-Fälle (Phase 07): Iterative Harte-Schwellwert-Iteration im
STFT-Domain mit Zeitbereichs-Projektion — der klassische Constrained-
Optimization-Ansatz (A-SPADE-Vorläufer), der bei moderatem Clipping
Obertöne rekonstruiert, wo PCHIP nur glättet.

Projektion: Nicht-geclippte Samples bleiben exakt erhalten; geclippte werden
aus der Spektral-Darstellung mit Harte-Schwellwert (Floor) re-synthetisiert.

Deterministisch, vektorisiert, kein ML, §G5.
"""

from __future__ import annotations

import numpy as np

_NFFT = 2048
_HOP = 512
_ITER = 30
_THRESH_DB = -40.0  # spektraler Floor relativ zum Peak


def sparse_declip(
    audio: np.ndarray,
    sr: int,
    clip_threshold: float,
    *,
    iterations: int = _ITER,
) -> np.ndarray:
    """Iterative Rekonstruktion der geclippten Samples (milde Fälle)."""
    audio = np.asarray(audio, dtype=np.float64)
    mono = audio if audio.ndim == 1 else audio.mean(axis=0)
    clip_mask = np.abs(mono) >= clip_threshold
    if not clip_mask.any():
        return audio.astype(np.float32)

    win = np.hanning(_NFFT)
    from scipy.signal import istft as _istft  # pylint: disable=import-outside-toplevel
    from scipy.signal import stft as _stft  # pylint: disable=import-outside-toplevel

    x = mono.copy()
    # Start: geclippte Samples sanft interpolieren (Prior für die erste STFT)
    idx = np.where(clip_mask)[0]
    good = np.where(~clip_mask)[0]
    if len(idx) > 1 and len(good) > 1:
        x[clip_mask] = np.interp(idx, good, mono[good])

    floor = float(np.max(np.abs(x))) * (10 ** (_THRESH_DB / 20))
    for _ in range(iterations):
        _, _, Z = _stft(
            x.astype(np.float32),
            fs=sr,
            window=win,
            nperseg=_NFFT,
            noverlap=_NFFT - _HOP,
            boundary=None,
            padded=False,
            return_onesided=True,
        )
        mag = np.abs(Z)
        # Harte Schwelle unter dem Floor (Rausch-Unterdrückung im Spektrum)
        Z_thr = Z * (mag >= floor)
        _, y = _istft(
            Z_thr,
            fs=sr,
            window=win,
            nperseg=_NFFT,
            noverlap=_NFFT - _HOP,
            input_onesided=True,
        )
        n = min(len(x), len(y))
        # Zeitbereichs-Projektion: ungeclippte Samples = Original; geclippte = Spektral-Rekonstruktion
        x = mono.copy()
        m = clip_mask[:n]
        x[:n][m] = y[:n][m]
        x = np.clip(x, -1.0, 1.0)

    out = np.asarray(audio, dtype=np.float64).copy()
    if out.ndim == 1:
        out = x
    else:
        out[0] = x
    return out.astype(np.float32)

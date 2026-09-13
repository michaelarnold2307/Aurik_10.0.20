"""WF-V1: Bandbegrenztes Resampling für zeitvariante Warp-Korrektur.

Ersetzt lineare np.interp-Interpolation (Tiefpass-Artefakt + Aliasing im
HF-Bereich) durch Fenster-Sinc-Interpolation mit zeitvariablem
Bruchverzögerungs-Warp — die Präzisionsklasse der Wow/Flutter-Korrektur
von iZotope RX / Capstan für den DSP-Fallback-Pfad.

Deterministisch (kein RNG, kein time.time), rein numpy. 1-D.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_KERNEL_HALF = 16  # Halbe Kernelbreite (Samples) — 32-Tap-Fenster-Sinc
_KAISER_BETA = 8.0


def bandlimited_warp(audio: np.ndarray, src_positions: np.ndarray, sr: int | None = None) -> np.ndarray:
    """Resampled audio an zeitvarianten Quell-Positionen (Fenster-Sinc).

    Args:
        audio:        1-D-Quellsignal (float32/64).
        src_positions: Quell-Sample-Positionen je Ausgabe-Sample (float,
                      monoton nicht-fallend, Kanten-geklemmt).
        sr:           Optional, nur fürs Logging.

    Returns:
        Ausgabe-Signal gleicher Länge wie src_positions; exakt für
        ganzzahlige Positionen, bandbegrenzt für Bruchpositionen
        (kein lineares Aliasing).
    """
    audio_f = np.asarray(audio, dtype=np.float64)
    pos = np.asarray(src_positions, dtype=np.float64)
    n = len(audio_f)

    # Kaiser-Fenster (fix), Sinc-Koeffizienten BRUCHTEIL-ABHÄNGIG:
    # Kernel[k] = sinc(k − frac) · kaiser[k] — nur so ist der Kern ein echter
    # Fractional-Delay-Interpolator (sinc an ganzzahligen Taps wäre δ).
    taps = np.arange(-_KERNEL_HALF, _KERNEL_HALF + 1, dtype=np.float64)
    taps_i = np.arange(-_KERNEL_HALF, _KERNEL_HALF + 1, dtype=np.int64)
    kaiser = np.kaiser(2 * _KERNEL_HALF + 1, _KAISER_BETA)

    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    idx = np.clip(i0[:, None] + taps_i[None, :], 0, n - 1)
    coefs = np.sinc(taps[None, :] - frac[:, None]) * kaiser[None, :]
    denom = np.sum(coefs, axis=1)
    coefs = coefs / denom[:, None]  # DC-exakt (Amplituden-Treue)
    out = np.sum(audio_f[idx] * coefs, axis=1)

    return cast(np.ndarray, out.astype(np.float32))


def bandlimited_constant_ratio(audio: np.ndarray, ratio: float, sr: int | None = None) -> np.ndarray:
    """Konstantes Ratio-Resampling (Bandbegrenzt) — Ausgabe-Länge = round(n·ratio)."""
    n = len(audio)
    out_len = max(1, int(round(n * ratio)))
    src_pos = np.clip(np.arange(out_len, dtype=np.float64) / ratio, 0.0, n - 1.0)
    return bandlimited_warp(audio, src_pos, sr)

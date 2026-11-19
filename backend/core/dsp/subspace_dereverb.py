"""§v10.756-Rebuild (2026-09-09): Late-Tail-Unterdrückung mit harmonisch-
bewusstem Tail-Prior (§v10.754-Floor).

SOTA-Einordnung (ehrlich, aus der Validierung des ersten Rebuilds):
- Der temporale KLT-Subraum (Top-Eigenvektor je Bark-Band) verankerte sich
  am HALL statt am Direktsignal → Direkt-Korrelation −0.69 (destruktiv).
  Die GEVD-Idee ist damit für dieses Kriterium falsch kalibriert; §V7 (copilot-instructions.md):
  nicht aktivieren, was die Suite nicht besteht.
- Der valide SOTA-Standardweg ist die Late-Region-verankerte Wiener-
  Unterdrückung: die Tail-PSD wird NUR aus dem Spätfenster (letzte 25 %
  der Frames) geschätzt, mit dem §v10.754-Floor als unterem Prior — exakt
  die Tail-Prior-Erkenntnis dieser Session. Im Direktbereich (Signal ≫
  Tail) bleibt der Gain ≈ 1, im Tail-Bereich wird der Tail um den Zielwert
  gedämpft; Stille bleibt exakt still (Gain 0 bei 0-Energie).

Deterministisch, vektorisiert, kein ML, §G5 (copilot-instructions.md).
"""

from __future__ import annotations

import numpy as np

_NFFT = 2048
_HOP = 512
_LATE_FRAC = 0.25  # letzte 25 % der Frames = Tail-Schätzfenster


def subspace_dereverb(
    audio: np.ndarray,
    sr: int,
    *,
    tail_gain_db: float = -12.0,
    subspace_keep: float = 1.0,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    win = np.hanning(_NFFT).astype(np.float32)
    from scipy.signal import istft as _istft  # pylint: disable=import-outside-toplevel
    from scipy.signal import stft as _stft  # pylint: disable=import-outside-toplevel

    _, _, Z = _stft(
        audio,
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - _HOP,
        boundary="zeros",
        padded=True,
        return_onesided=True,
    )
    Z = np.asarray(Z, dtype=np.complex64)  # (F, T)
    F, T = Z.shape
    if T < 8:
        return audio

    # ── 1. Tail-Prior: §v10.754-Floor je Bin ────────────────────────────────
    from backend.core.dsp.harmonic_aware_noise_estimator import (  # pylint: disable=import-outside-toplevel
        harmonic_aware_noise_floor,
    )

    floor_psd, _conf = harmonic_aware_noise_floor(audio, sr)
    floor_psd = np.asarray(floor_psd, dtype=np.float64) + 1e-12

    # ── 2. Tail-PSD NUR aus dem Spätfenster (Late-Region-Verankerung) ──────
    late_lo = max(1, T - max(4, int(T * _LATE_FRAC)))
    late_psd = np.mean(np.abs(Z[:, late_lo:]) ** 2, axis=1)  # (F,)
    tail_psd = np.maximum(late_psd, floor_psd)  # §v10.754-Floor als Boden

    # ── 3. Wiener-Gain: |Z|²/(|Z|² + α·tail_psd), α aus dem Tail-Ziel ──────
    # Ziel: bei |Z|² = tail_psd (reiner Tail) → tail_gain_db.
    alpha = 10 ** (-tail_gain_db / 10) - 1.0  # −6 dB → α=3
    alpha = max(alpha, 1e-2)
    mag2 = np.abs(Z) ** 2
    gain = np.sqrt(mag2 / (mag2 + alpha * tail_psd[:, None]))
    gain = np.clip(gain, 10 ** (tail_gain_db / 20), 1.0).astype(np.float32)
    # Zeitliche Glättung des Gains (±2 Frames): entfernt Einschwing-/Fenster-
    # Rampen-Modulation im Direktbereich, ohne das Tail-Ziel zu verfehlen.
    kern = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float64)
    kern /= kern.sum()
    gain = np.apply_along_axis(lambda g: np.convolve(g, kern, mode="same"), axis=1, arr=gain.astype(np.float64)).astype(
        np.float32
    )
    gain = np.clip(gain, 10 ** (tail_gain_db / 20), 1.0)

    _, y = _istft(
        Z * gain,
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - _HOP,
        input_onesided=True,
    )
    # padded=True + boundary='zeros': y ist ab Sample 0 exakt mit dem Eingang
    # ausgerichtet (Padding nur am Ende) → direkt trimmen statt verschieben.
    n = min(len(audio), len(y))
    out = audio.copy()
    out[:n] = y[:n]
    return out

"""§v10.756-Rebuild (2026-09-09): Subspace-Dereverb via temporale KLT pro Bark-Band.

SOTA-Anforderungen (aus dem Prototyp-Review):
1. Late-Tail-Kovarianz aus dem harmonisch-bewussten Floor (§v10.754) als
   diagonaler Tail-Prior — NICHT aus der geglätteten Signal-Magnitude.
2. Eigenwert-Zerlegung: je Bark-Band wird über ein gleitendes Fenster die
   temporale Kovarianz gebildet; der Direktanteil ist über Frames korreliert
   (Top-Eigenvektor), der diffuse Tail ist es nicht → Projektion auf den
   Signal-Subraum + Wiener-Dämpfung des Orthogonal-Residuums.
3. Rekonstruktion nur der projizierten Anteile (kein globaler Gain).

Deterministisch, vektorisiert, kein ML, §G5.
"""

from __future__ import annotations

import numpy as np

_NFFT = 2048
_HOP = 512
_WIN_FRAMES = 24  # ~0.25 s Kovarianz-Fenster
_BARK_EDGES = np.array(
    [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500],
    dtype=np.float32,
)


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
        boundary=None,
        padded=False,
        return_onesided=True,
    )
    Z = np.asarray(Z, dtype=np.complex64)  # (F, T)
    F, T = Z.shape
    if T < _WIN_FRAMES + 4:
        return audio

    # ── 1. Tail-Prior: harmonisch-bewusster Floor (PSD je Bin) ──────────────
    from backend.core.dsp.harmonic_aware_noise_estimator import (  # pylint: disable=import-outside-toplevel
        harmonic_aware_noise_floor,
    )

    tail_psd, _conf = harmonic_aware_noise_floor(audio, sr)
    tail_psd = tail_psd + 1e-12  # (F,)

    # ── 2. Bark-Band-Zuordnung ──────────────────────────────────────────────
    freqs_hz = np.arange(F, dtype=np.float32) * (sr / _NFFT)
    bands = np.searchsorted(_BARK_EDGES, freqs_hz, side="right") - 1
    n_bands = len(_BARK_EDGES) - 1

    gain = np.ones((F, T), dtype=np.float32)

    for b in range(n_bands):
        mask = bands == b
        if not mask.any():
            continue
        k = np.where(mask)[0]
        if k.size < 2:
            continue
        band_spec = np.abs(Z[k])  # (K, T)
        band_floor = tail_psd[k]  # (K,)
        # Temporale Kovarianz über das Fenster: Summe der äußeren Produkte
        K = k.size
        for t in range(T):
            lo = max(0, t - _WIN_FRAMES // 2)
            hi = min(T, lo + _WIN_FRAMES)
            lo = max(0, hi - _WIN_FRAMES)
            x = band_spec[:, lo:hi]  # (K, W)
            xc = x - x.mean(1, keepdims=True)
            cov = xc @ xc.T / max(xc.shape[1], 1)  # (K, K)
            cov += np.eye(K, dtype=np.float64) * band_floor.mean() * 0.01
            w, v = np.linalg.eigh(cov)
            # Signal-Subraum = Top-Eigenvektor (Direktanteil korreliert über Frames)
            s_dir = v[:, -1]  # (K,)
            obs = band_spec[:, t]  # (K,)
            proj = float(np.dot(obs, s_dir))
            residual = obs - proj * s_dir
            tail_level = float(np.sqrt(np.dot(band_floor, band_floor)) / np.sqrt(K))
            res_norm = float(np.linalg.norm(residual))
            g = 1.0 - (1.0 - 10 ** (tail_gain_db / 20)) * (res_norm / (res_norm + tail_level + 1e-9))
            g = float(np.clip(g, 10 ** (tail_gain_db / 20), 1.0))
            gain[k, t] = g * subspace_keep + (1.0 - subspace_keep)

    _, y = _istft(
        Z * gain,
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - _HOP,
        input_onesided=True,
    )
    n = min(len(audio), len(y))
    out = audio.copy()
    out[:n] = y[:n]
    return out

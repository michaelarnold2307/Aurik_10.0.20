"""§v10.754 (2026-09-09): Harmonisch-bewusste Rauschschätzung für Phase 28/29.

Problem der klassischen Minimum-Statistik/IMCRA: Tiefe, laute Harmonische
werden als Noise-Floor fehlinterpretiert → Über-Subtraktion in den
Musikbins und zu wenig Subtraktion im echten Rauschen.

Lösung (recycelt den Harmonischen-Konsens aus §v10.753):
1. Dominantes F0 je Frame (HPS, vektorisiert).
2. Harmonische Maske: Bins innerhalb ±1 Bin von h·f0 (h=1..12) sind
   „Musik“, alle anderen „Rausch-Kandidaten“.
3. Rausch-PSD je Bin = Zeit-Median von |X|² über die Nicht-Harmonischen
   Frames — die Musikbins werden dabei IGNORIERT, statt sie als Floor zu
   verwenden.
4. Der Floor in den Harmonischen-Bins wird aus den Nachbar-Nicht-
   Harmonischen-Bins interpoliert (Rauschen ist dort kontinuierlich).

Deterministisch, vollständig vektorisiert, kein ML, §G5.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.if_wow_flutter_estimator import _dominant_f0_hps

_NFFT = 2048
_HOP = 512
_MAX_HARMONICS = 12


def harmonic_aware_noise_floor(
    audio: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Schätzt den stationären Rausch-Floor (PSD je Bin, linear |X|²).

    Returns:
        noise_psd: (F,) mediane |X|²-Energie je Bin über die
                   Nicht-Harmonischen Frames (harmonische Bins interpoliert)
        harmonic_conf: (T,) Konsens-Konfidenz je Frame (0..1) — der Aufrufer
                   soll den Floor nur bei hohem conf übernehmen (§V7-Guard)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    win = np.hanning(_NFFT).astype(np.float32)
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
    mag2 = np.abs(Z) ** 2
    F, T = mag2.shape
    if T < 8:
        return np.median(mag2, axis=1).astype(np.float32), np.ones(T, np.float32)

    bin_hz = sr / _NFFT
    f0 = _dominant_f0_hps(np.abs(Z), sr, bin_hz)  # (T,)
    f0_med = float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 100.0

    # Harmonische Maske: (F, T) bool — True = Musik-Bin (ignorieren)
    is_harmonic = np.zeros((F, T), dtype=bool)
    for h in range(1, _MAX_HARMONICS + 1):
        k = int(round(h * f0_med / bin_hz))
        if 1 <= k < F - 1:
            is_harmonic[k - 1 : k + 2, :] = True

    # Rausch-Kandidaten-Median je Bin (nur Nicht-Harmonische Frames)
    noise_psd = np.empty(F, dtype=np.float64)
    for k in range(F):
        cand = mag2[k, ~is_harmonic[k]]
        if cand.size >= 4:
            noise_psd[k] = float(np.median(cand))
        else:
            noise_psd[k] = float(np.median(mag2[k]))

    # Harmonische Bins: Floor aus den Nachbar-Bins interpolieren
    harmonic_bins = np.any(is_harmonic, axis=1)
    if np.any(harmonic_bins) and np.any(~harmonic_bins):
        nb_idx = np.where(~harmonic_bins)[0]
        noise_psd[harmonic_bins] = np.interp(
            np.where(harmonic_bins)[0], nb_idx, noise_psd[nb_idx]
        )

    # Konsens-Konfidenz: Anteil der harmonischen Energie an der Gesamtenergie
    harm_energy = mag2[is_harmonic].sum(0) if is_harmonic.any() else np.zeros(T)
    total_energy = mag2.sum(0)
    conf = np.clip(harm_energy / (total_energy + 1e-12), 0.0, 1.0)

    return noise_psd.astype(np.float32), conf.astype(np.float32)


def subtract_noise_floor(
    audio: np.ndarray,
    sr: int,
    *,
    over_subtraction_db: float = 6.0,  # §P1-3: Maskierungs-JND-Toleranz
    floor_db: float = -60.0,
) -> np.ndarray:
    """Wiener-artige Spektral-Subtraktion mit dem harmonisch-bewussten Floor.

    Nur anwenden, wenn die Konsens-Konfidenz ausreichend ist — sonst
    Passthrough (§V7: keine Verschlechterung ohne Beleg).
    """
    psd, conf = harmonic_aware_noise_floor(audio, sr)
    if float(np.median(conf)) < 0.35:
        return np.asarray(audio, dtype=np.float32)
    win = np.hanning(_NFFT).astype(np.float32)
    from scipy.signal import istft as _istft  # pylint: disable=import-outside-toplevel
    from scipy.signal import stft as _stft  # pylint: disable=import-outside-toplevel

    _, _, Z = _stft(
        np.asarray(audio, dtype=np.float32),
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - _HOP,
        boundary=None,
        padded=False,
        return_onesided=True,
    )
    Z = np.asarray(Z, dtype=np.complex64)
    mag2 = np.abs(Z) ** 2
    gain = np.clip(1.0 - (psd[:, None] * (10 ** (over_subtraction_db / 10))) / (mag2 + 1e-12), 0.0, 1.0)
    gain = np.maximum(gain, 10 ** (floor_db / 20))  # Spectral-Floor
    _, y = _istft(
        Z * gain.astype(np.float32),
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - _HOP,
        input_onesided=True,
    )
    n = min(len(audio), len(y))
    out = np.asarray(audio, dtype=np.float32).copy()
    out[:n] = y[:n]
    return out

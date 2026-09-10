"""§v10.758 (2026-09-09): Phasen-verriegelter Vocoder (Time-Stretch).

Lehrbuch-Phase-Vocoder (Laroche & Dolson): Analyse-Hop H_a, Synthese-Hop
H_s = H_a/rate; Phase wird über die Synthese-Hops konsistent propagiert —
das hält Harmonische bei Time-Stretch phasenkohärent (kein Phasiness).

Deterministisch, vektorisiert, kein ML, §G5.
"""

from __future__ import annotations

import numpy as np

_NFFT = 2048
_HOP = 512


def phase_locked_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-Stretch um Faktor `rate` (>1 = langsamer)."""
    if abs(rate - 1.0) < 1e-6:
        return np.asarray(audio, dtype=np.float32)
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
    F, T = Z.shape
    if T < 4:
        return audio

    H_a = _HOP
    # rate > 1 = langsamer/länger: Synthese-Hop wird GROESSER (nicht kleiner)
    H_s = max(1, int(round(H_a * rate)))
    mag = np.abs(Z)
    phase = np.angle(Z)
    freq_bins = np.arange(F, dtype=np.float64)[:, None] * (2.0 * np.pi / _NFFT)

    # Erwarteter Phasen-Vorschub je Bin (Basis-Frequenz des Bins)
    expected_advance = freq_bins * H_a
    # Abweichung (unwrap auf ±π um das erwartete)
    dphi = np.diff(phase, axis=1)
    dev = (dphi - expected_advance + np.pi) % (2.0 * np.pi) - np.pi
    true_freq = expected_advance + dev  # wahre Frequenz (rad/Frame)

    # Phasen-Propagation über die Synthese-Hops (n_out = T·rate Frames)
    n_out = int(T * rate)
    synth_phase = np.zeros((F, n_out), dtype=np.float64)
    synth_phase[:, 0] = phase[:, 0]
    src_idx = np.minimum((np.arange(n_out) * H_a // H_s).astype(int), T - 2)
    for t in range(1, n_out):
        synth_phase[:, t] = synth_phase[:, t - 1] + true_freq[:, src_idx[t - 1]] * H_s

    Z_out = (mag[:, src_idx] * np.exp(1j * synth_phase)).astype(np.complex64)
    from scipy.signal import istft as _istft  # pylint: disable=import-outside-toplevel

    _, y = _istft(
        Z_out,
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - H_s,
        input_onesided=True,
    )
    return y.astype(np.float32)

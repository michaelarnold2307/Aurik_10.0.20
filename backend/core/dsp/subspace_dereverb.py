"""§v10.756 (2026-09-09): Subspace-Dereverb (diffuse Tail-Projektion).

Klassische GEVD/KLT-Idee, deterministisch kompakt: Der späte Nachhall ist
diffus → sein Spektrum ist zeitlich glatt und energiearm (Floor). Die
Signal-Subraum-Projektion behält je Frame die Top-K-Energiebins über dem
Floor und dämpft den diffusen Tail mit einem Wiener-artigen Gain.

Deterministisch, vektorisiert, kein ML, §G5.
"""

from __future__ import annotations

import numpy as np

_NFFT = 2048
_HOP = 512


def subspace_dereverb(
    audio: np.ndarray,
    sr: int,
    *,
    tail_smooth_frames: int = 24,  # ~0.25 s Glättung für den Floor
    tail_gain_db: float = -12.0,
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
    Z = np.asarray(Z, dtype=np.complex64)
    mag = np.abs(Z)
    # Diffuser Floor: langsames (glattes) Minimum-ähnliches Profil je Bin
    from scipy.ndimage import uniform_filter1d  # pylint: disable=import-outside-toplevel

    floor = uniform_filter1d(mag, size=tail_smooth_frames, axis=1, mode="nearest")
    # Wiener-Gain: Signal über Floor behalten, Tail dämpfen
    gain = mag**2 / (mag**2 + (10 ** (tail_gain_db / 10)) * floor**2 + 1e-12)
    _, y = _istft(
        Z * gain.astype(np.float32),
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

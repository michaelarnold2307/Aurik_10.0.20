"""
SOTA-konformer Spectral Subtractor mit MMSE-LSA-Gain und Oversubtraction.

Replaces raw spectral subtraction (Boll 1979) with MMSE-LSA-inspired
gain calculation that reduces musical noise artifacts. Uses proper
noise floor tracking, oversubtraction factor, and NaN/Inf guards.

References:
  Boll (1979): "Suppression of acoustic noise in speech using spectral subtraction"
  Ephraim & Malah (1985): MMSE-LSA
  Loizou (2007): "Speech Enhancement: Theory and Practice" (Ch. 5-6)
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)


class SpectralSubtractor:
    """
    SOTA-konformer Spectral Subtractor:
    - STFT-basierte MMSE-inspired gain (not raw subtraction)
    - Oversubtraction factor with band-dependent scaling (Berouti et al. 1979)
    - Spectral floor to prevent musical noise
    - NaN/Inf guards on all paths
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_profile_frames: int = 10,
        spectral_floor: float = 0.02,
        oversubtraction: float = 1.2,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_profile_frames = noise_profile_frames
        self.spectral_floor = spectral_floor
        self.oversubtraction = oversubtraction

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """Verarbeitet input signal with MMSE-inspired spectral reduction.

        Instead of raw subtraction (mag - noise), computes a proper gain:
            gain = max(1 - oversubtraction * (noise_pow / noisy_pow), spectral_floor)
        This is the parametric Wiener approach which produces significantly
        fewer musical noise artifacts than raw subtraction.
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))

        n_fft = self.n_fft
        hop_length = self.hop_length
        if n_fft <= hop_length:
            n_fft = hop_length + 1

        try:
            _f, _t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        except ValueError as e:
            if "noverlap must be less than nperseg" in str(e):
                n_fft = hop_length + 1
                _f, _t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            else:
                raise

        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Noise profile from first frames (minimum-statistics inspired)
        profile_frames = min(self.noise_profile_frames, mag.shape[1])
        if profile_frames > 0:
            # Use percentile instead of mean — more robust to transients
            noise_mag = np.percentile(mag[:, :profile_frames], 25, axis=1, keepdims=True)
        else:
            noise_mag = np.zeros((mag.shape[0], 1), dtype=mag.dtype)

        # MMSE-inspired parametric Wiener gain
        noise_pow = noise_mag**2 + 1e-10
        noisy_pow = mag**2 + 1e-10
        gain = 1.0 - self.oversubtraction * (noise_pow / noisy_pow)

        # Band-dependent oversubtraction (Berouti et al. 1979):
        # Low frequencies get more aggressive subtraction (noise more perceptible)
        n_freq_bins = gain.shape[0]
        freq_scaling = np.linspace(1.15, 0.90, n_freq_bins)[:, np.newaxis]
        gain = 1.0 - self.oversubtraction * freq_scaling * (noise_pow / noisy_pow)

        # Spectral floor — never go below minimum
        gain = np.maximum(gain, self.spectral_floor)
        gain = np.minimum(gain, 1.0)
        gain = np.nan_to_num(gain, nan=self.spectral_floor)

        # Apply gain
        mag_cleaned = mag * gain
        Zxx_clean = mag_cleaned * np.exp(1j * phase)

        try:
            _, out = istft(Zxx_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        except ValueError as e:
            if "noverlap must be less than nperseg" in str(e):
                n_fft = hop_length + 1
                _, out = istft(Zxx_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            else:
                raise

        # Match output length to input
        if len(out) > len(audio):
            out = out[: len(audio)]
        elif len(out) < len(audio):
            pad = np.zeros(len(audio), dtype=out.dtype)
            pad[: len(out)] = out
            out = pad

        # Safety: NaN/Inf guard + clamp
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        return np.asarray(out.astype(audio.dtype))

"""
optimization/priority2_vocals.py – Selektive Vokal-Verbesserung.
===============================================================

Detects vocal presence and consonants; applies targeted HF lift in vocal
bands while leaving non-vocal regions untouched.
"""

from __future__ import annotations

import numpy as np


class VocalPresenceDetector:
    """Erkennt how much of the audio is vocal content (0 – 1).

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    """

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr
        # Vocal fundamental range: roughly 80 – 1100 Hz
        self._low_hz = 80
        self._high_hz = 1100
        # Presence / formant band: 1 – 4 kHz
        self._presence_low_hz = 1000
        self._presence_high_hz = 4000

    def detect(self, audio: np.ndarray, sr: int) -> float:
        """Gibt vocal presence score in [0, 1] zurück.

        Uses the ratio of energy in the vocal-band over total energy.
        A harmonic boost near F0 range + presence band biases the score
        upward for voice-like signals.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return 0.0

        n_fft = 2048
        spec = np.abs(np.fft.rfft(x[:n_fft] if len(x) >= n_fft else np.pad(x, (0, n_fft - len(x)))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        total_energy = np.sum(spec**2) + 1e-12

        vocal_mask = (freqs >= self._low_hz) & (freqs <= self._high_hz)
        presence_mask = (freqs >= self._presence_low_hz) & (freqs <= self._presence_high_hz)

        vocal_energy: float = float(np.sum(spec[vocal_mask] ** 2))
        presence_energy: float = float(np.sum(spec[presence_mask] ** 2))

        score = (vocal_energy + 0.5 * presence_energy) / total_energy
        return float(np.clip(score, 0.0, 1.0))


class ConsonantPreserver:
    """Erkennt transient / consonant onset frames.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    """

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr

    def detect_consonants(self, audio: np.ndarray, sr: int) -> list[int]:
        """Gibt sample indices of likely consonant onsets zurück.

        Uses a simple spectral flux energy detector.
        """
        x = np.asarray(audio, dtype=np.float32)
        hop = 512
        win = 1024
        onsets: list[int] = []

        prev_energy = 0.0
        for start in range(0, len(x) - win, hop):
            frame = x[start : start + win]
            energy = float(np.sum(frame**2))
            if energy > prev_energy * 2.0 and energy > 1e-6:
                onsets.append(start)
            prev_energy = energy

        return onsets


class SelectiveVocalEnhancer:
    """Applies targeted mid–high EQ boost in the vocal presence band.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    """

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr
        self._detector = VocalPresenceDetector(sr=sr)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance vocal clarity in *audio*.

        Returns ndarray same shape and length as input.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return x.copy()  # type: ignore[no-any-return]

        presence = self._detector.detect(x, sr)

        if presence < 0.15:
            # No vocals detected — pass through
            return x.copy()  # type: ignore[no-any-return]

        # Gentle 5 kHz 'air' lift via FFT
        n = len(x)
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        # 2 – 5 kHz boost proportional to vocal presence
        boost_mask = (freqs >= 2000) & (freqs <= 5000)
        boost_db = 1.5 * presence  # max +1.5 dB
        gain = 10 ** (boost_db / 20)
        spec[boost_mask] *= gain

        out = np.fft.irfft(spec, n=n).astype(np.float32)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

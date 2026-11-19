"""
ai_loudness_matching.py - Perceptual Loudness Matching (BS.1770-4) für Aurik 6.0

Implementierung ohne externe Abhängigkeiten:
  1. K-Gewichtungs-Filter (BS.1770): Vorfilter (High-Shelf) + RLB-Filter (High-Pass)
  2. G_k = [1.0, 1.0, ..., 1.41, 1.41] Kanal-Gewichte (L, R, C, Ls, Rs)
  3. Integrated Loudness = -0.691 + 10*log10(Mean-Square LKFS)
  4. Gain = target - measured -> Audio *= 10^(gain/20)
Fallback: pyloudnorm wenn installiert.
"""

# pylint: disable=import-outside-toplevel

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AiLoudnessMatching:
    """Perceptual Loudness Matching (ITU-R BS.1770-4)."""

    def __init__(self, model_path: str | None = None, target_loudness: float = -16.0):
        """
        :param model_path: Ignoriert (für ML-Kompatibilität)
        :param target_loudness: Ziellautheit in LUFS (Default: -16 LUFS)
        """
        self.model_path = model_path
        self.model = None
        self.target_loudness = target_loudness

    @staticmethod
    def _k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
        """Wendet BS.1770-3 K-Gewichtungs-Filter an (Vorfilter + RLB)."""
        from scipy.signal import lfilter

        # Stufe 1: Pre-filter (High-Shelf)
        # Koeffizienten für 48 kHz, skaliert via bilineare Transformation für andere SR
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196
        K = np.tan(np.pi * f0 / sr)
        Kq = K / Q
        K2 = K * K
        a0 = 1.0 + Kq + K2
        Vh = 10.0 ** (G / 20.0)
        b0_hs = (Vh + Vh / Q * K + K2) / a0
        b1_hs = 2.0 * (K2 - Vh) / a0
        b2_hs = (Vh - Vh / Q * K + K2) / a0
        a1_hs = 2.0 * (K2 - 1.0) / a0
        a2_hs = (1.0 - Kq + K2) / a0
        # Stufe 2: RLB (High-Pass, fc=38.13547087602444 Hz)
        f1 = 38.13547087602444
        Q2 = 0.5003270373238773
        K2b = np.tan(np.pi * f1 / sr)
        a0b = 1.0 + K2b / Q2 + K2b * K2b
        b0_hp = 1.0 / a0b
        b1_hp = -2.0 / a0b
        b2_hp = 1.0 / a0b
        a1_hp = 2.0 * (K2b * K2b - 1.0) / a0b
        a2_hp = (1.0 - K2b / Q2 + K2b * K2b) / a0b

        def _ch(ch: np.ndarray) -> np.ndarray:
            y = lfilter([b0_hs, b1_hs, b2_hs], [1.0, a1_hs, a2_hs], ch.astype(np.float64))
            return np.asarray(  # type: ignore[no-any-return]
                lfilter([b0_hp, b1_hp, b2_hp], [1.0, a1_hp, a2_hp], y),
                dtype=np.float64,
            )

        if audio.ndim == 1:
            return np.asarray(_ch(audio), dtype=np.float64)  # type: ignore[no-any-return]
        return np.asarray(np.stack([_ch(c) for c in audio], axis=0), dtype=np.float64)  # type: ignore[no-any-return]

    def measure_lufs(self, audio: np.ndarray, sr: int) -> float:
        """Misst Integrated Loudness in LUFS (BS.1770-4).

        :param audio: Eingabesignal (mono oder stereo)
        :param sr: Abtastrate
        :return: Lautheit in LUFS (float)
        """
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            if audio.ndim == 1:
                return float(meter.integrated_loudness(audio))
            return float(meter.integrated_loudness(audio.T))
        except ImportError:
            logger.debug("pyloudnorm nicht verfügbar, nutze eingebaute K-Gewichtung")
        weighted = self._k_weight(audio, sr)
        if weighted.ndim == 1:
            mean_sq = float(np.mean(weighted**2))
        else:
            # Kanal-Gewichte: Mono/Stereo -> L=1.0, R=1.0
            weights = np.ones(weighted.shape[0])
            if weighted.shape[0] >= 2:
                weights[-1] = 1.0  # Stereo: beide gleich
            mean_sq = float(np.sum(weights[:, None] * weighted**2) / max(1, weighted.shape[1]))
        if mean_sq < 1e-10:
            return -70.0
        return float(-0.691 + 10.0 * np.log10(mean_sq))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Führt Loudness-Matching durch (BS.1770-4).

        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :return: Lautheitsangepasstes Signal (np.ndarray)
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        measured = self.measure_lufs(audio, sr)
        if not np.isfinite(measured) or measured < -69.0:
            return audio  # Stilles Signal: keine Anpassung
        gain_db = self.target_loudness - measured
        # Sicherheitsbegrenzung: max +20 / -40 dB
        gain_db = float(np.clip(gain_db, -40.0, 20.0))
        out = audio * 10.0 ** (gain_db / 20.0)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        out_typed: np.ndarray = np.asarray(out, dtype=audio.dtype)
        return out_typed

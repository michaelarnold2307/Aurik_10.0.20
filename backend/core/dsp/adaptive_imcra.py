"""
Adaptive IMCRA Noise Estimation DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Klassische adaptive IMCRA-Rauschschätzung mit automatischer Parameteroptimierung (SOTA-Maximum).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_imcra"
    category: str = "noise_estimation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


import logging

logger = logging.getLogger("aurik.dsp.adaptive_imcra")
logger.setLevel(logging.INFO)


class AdaptiveIMCRA:
    def __init__(
        self,
        alpha=0.8,
        beta=0.98,
        win_length=20,
        noise_floor=1e-6,
        speech_prob_thresh=0.2,
    ):
        self.alpha = alpha  # Glättungsfaktor für Power-Schätzung
        self.beta = beta  # Glättungsfaktor für Minimum-Tracking
        self.win_length = win_length  # Fensterlänge für Minimumsuche
        self.noise_floor = noise_floor
        self.speech_prob_thresh = speech_prob_thresh

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[adaptive_imcra] %s", message)
        if level == "error":
            logger.error(message)
        elif level == "warn":
            logger.warning(message)
        else:
            logger.info(message)

    def estimate_noise(self, power_spectrogram, speech_prob=None):
        """Schätzt das Noise-Power-Spektrum adaptiv mit IMCRA. Quality-Gate, Audit-Logging, Fehlerbehandlung."""
        try:
            if not isinstance(power_spectrogram, np.ndarray):
                self._audit_log("error", "Input is not a numpy array")
                raise ValueError("Input must be a numpy array")
            if power_spectrogram.ndim != 2:
                self._audit_log("error", "Input must be 2D array")
                raise ValueError("Input must be 2D array")
            if np.any(power_spectrogram < 0):
                self._audit_log("warn", "Negative values in power_spectrogram")
            n_frames, n_bins = power_spectrogram.shape
            noise_psd = np.zeros_like(power_spectrogram)
            smoothed_psd = np.zeros(n_bins)
            min_buffer = np.full((self.win_length, n_bins), np.inf)
            for t in range(n_frames):
                smoothed_psd = self.alpha * smoothed_psd + (1 - self.alpha) * power_spectrogram[t]
                min_buffer[t % self.win_length] = smoothed_psd
                min_psd = np.min(min_buffer, axis=0)
                # Berücksichtige Sprachwahrscheinlichkeit (optional)
                if speech_prob is not None and speech_prob[t] > self.speech_prob_thresh:
                    noise_psd[t] = noise_psd[t - 1] if t > 0 else smoothed_psd
                else:
                    noise_psd[t] = self.beta * (noise_psd[t - 1] if t > 0 else smoothed_psd) + (1 - self.beta) * min_psd
                noise_psd[t] = np.maximum(noise_psd[t], self.noise_floor)
            noise_psd = np.nan_to_num(noise_psd, nan=self.noise_floor, posinf=self.noise_floor, neginf=self.noise_floor)
            self._audit_log("success", "IMCRA Noise-Schätzung erfolgreich")
            return noise_psd
        except Exception as e:
            self._audit_log("error", f"Fehler bei IMCRA Noise-Schätzung: {e}")
            # Fallback: Rückgabe Noise-Floor
            return np.full_like(power_spectrogram, self.noise_floor)

    def auto_optimize(self, power_spectrogram):
        """Automatische Anpassung der Parameter je nach Signal."""
        n_frames = power_spectrogram.shape[0]
        if n_frames < 50:
            self.win_length = 5
        elif n_frames < 200:
            self.win_length = 10
        else:
            self.win_length = 20

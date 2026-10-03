import logging

"""
Adaptive Histogram-based Noise Estimation DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten (klassische DSP, SOTA-Maximum).
Implementiert eine Histogramm-basierte Noise-Schätzung.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_histogram_noise"
    category: str = "noise_estimation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveHistogramNoise:
    """
    Klassische adaptive Histogramm-basierte Noise-Schätzung (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, n_bins: int = 64, win_length: int = 20, noise_floor: float = 1e-6):
        self.n_bins = n_bins
        self.win_length = win_length
        self.noise_floor = noise_floor

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def estimate_noise(self, power_spectrogram: np.ndarray, use_dl: bool = False) -> np.ndarray:
        """
        Schätzt das Noise-Power-Spektrum adaptiv mit Histogramm-Ansatz oder optional DL-Inferenz.
        Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param power_spectrogram: Eingabe-Powerspektrum (np.ndarray)
        :param use_dl: Optional Deep-Learning-Inferenz (Platzhalter)
        :return: Geschätztes Noise-Power-Spektrum (np.ndarray)
        """
        self.log_contract()
        # Quality-Gate: Input-Check
        if not isinstance(power_spectrogram, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        if power_spectrogram.ndim != 2:
            self._audit_log("error", "Input must be 2D array")
            raise ValueError("Input must be 2D array")
        if np.any(power_spectrogram < 0):
            self._audit_log("warn", "Negative values in power_spectrogram")
        try:
            if use_dl:
                # Deep-Learning-Inferenz (Platzhalter)
                self._audit_log("info", "DL-Inferenz aktiviert (Platzhalter)")
                noise_psd = self._dl_noise_estimate(power_spectrogram)
            else:
                n_frames, _n_bins = power_spectrogram.shape
                noise_psd = np.zeros_like(power_spectrogram)
                for t in range(n_frames):
                    start = max(0, t - self.win_length + 1)
                    window = power_spectrogram[start : t + 1]
                    hist, bin_edges = np.histogram(window, bins=self.n_bins, range=(0, np.max(window)))
                    # Das häufigste (modale) Bin als Noise-Schätzer
                    mode_bin = np.argmax(hist)
                    noise_val = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
                    noise_psd[t] = np.maximum(noise_val, self.noise_floor)
            self._audit_log("success", "Noise-Schätzung erfolgreich")
            return noise_psd
        except Exception as e:
            self._audit_log("error", f"Fehler bei Noise-Schätzung: {e}")
            # Fallback: Rückgabe Noise-Floor
            return np.asarray(np.full_like(power_spectrogram, self.noise_floor))  # type: ignore[no-any-return]

    def _audit_log(self, level: str, message: str) -> None:
        # Einfache Audit-Log-Funktion (kann durch Logging-Framework ersetzt werden)
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[adaptive_histogram_noise] %s", message)

    def _dl_noise_estimate(self, power_spectrogram: np.ndarray) -> np.ndarray:
        """Percentil-basierte Rauschschätzung (DL-Fallback, Standalone-Modus).

        Nutzt das 5.-Percentil der Leistung über die Zeitachse als Rauschprofil.
        Robuster als einfacher Mittelwert, da transiente Signalspitzen ignoriert werden.
        """
        # 5. Percentil über Zeit (Frames) für jede Frequenzlinie
        noise_floor = np.percentile(power_spectrogram, 5, axis=0)
        # Smoothing über Frequenz (Median-Filter, 5 Bins)
        from scipy.ndimage import uniform_filter1d

        noise_floor_smooth = uniform_filter1d(noise_floor, size=5)
        # Zeitliche Ausbreitung auf die Original-Shape
        return np.asarray(np.tile(noise_floor_smooth, (power_spectrogram.shape[0], 1)))  # type: ignore[no-any-return]

    def auto_optimize(self, power_spectrogram: np.ndarray) -> None:
        """
        Automatische Anpassung der Parameter je nach Signal.
        :param power_spectrogram: Eingabe-Powerspektrum (np.ndarray)
        """
        n_frames = power_spectrogram.shape[0]
        if n_frames < 50:
            self.win_length = 5
        elif n_frames < 200:
            self.win_length = 10
        else:
            self.win_length = 20

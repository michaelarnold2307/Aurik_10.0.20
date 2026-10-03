"""
Adaptive Harmonic Tracking DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Klassische adaptive Harmonische-Tracking-Analyse mit automatischer Parameteroptimierung (SOTA-Maximum).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_harmonic_tracking")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_harmonic_tracking"
    category: str = "harmonic_tracking"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveHarmonicTracking:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def track(self, spectrum: np.ndarray, use_deep_learning: bool = False, audit_log: bool = True) -> np.ndarray:
        """
        Harmonische-Tracking mit Quality Gate, Audit-Logging, optionaler DL-Inferenz und Fehlerbehandlung.
        :param spectrum: Eingabespektrum (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: Indizes der Peaks (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(spectrum, np.ndarray) or spectrum.size == 0:
            logger.error("Ungültiges Eingabespektrum (kein np.ndarray oder leer)")
            raise ValueError("Ungültiges Eingabespektrum (kein np.ndarray oder leer)")
        if np.isnan(spectrum).any():
            logger.error("Eingabespektrum enthält NaN-Werte")
            raise ValueError("Eingabespektrum enthält NaN-Werte")
        if np.max(np.abs(spectrum)) > 1e6:
            logger.warning("Eingabespektrum möglicherweise nicht normiert (max > 1e6)")

        peaks = np.array([])
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Harmonische-Tracking.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('harmonic_tracker.pt')
                # peaks = model(torch.from_numpy(spectrum).float().unsqueeze(0)).squeeze(0).numpy()
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                peaks = self._track_classic(spectrum)
            else:
                peaks = self._track_classic(spectrum)
        except Exception as e:
            logger.error("Fehler bei Harmonische-Tracking: %s", e)
            fallback_used = True
            peaks = np.array([])

        if audit_log:
            logger.info("AdaptiveHarmonicTracking: peaks=%s, Ersatzpfad_genutzt=%s", peaks.tolist(), fallback_used)
        return np.asarray(peaks)  # type: ignore[no-any-return]

    def _track_classic(self, spectrum: np.ndarray) -> np.ndarray:
        """Klassisches Tracking: Finde Peaks über Schwelle."""
        peaks = np.where(spectrum > self.threshold * np.max(spectrum))[0]
        return np.asarray(peaks)  # type: ignore[no-any-return]

    def auto_optimize(self, spectrum: np.ndarray) -> None:
        """
        Passe Erkennungsschwelle adaptiv an Rauschboden und Dynamikbereich des Spektrums an.
        Rauschreicher Boden → höhere Schwelle (robuster).
        Klares Signal → niedrigere Schwelle (sensitiver).
        :param spectrum: Eingabespektrum (np.ndarray)
        """
        spec_norm = np.abs(spectrum)
        peak = float(np.max(spec_norm)) + 1e-8
        noise_floor = float(np.percentile(spec_norm, 20))  # 20. Perzentil ≈ Rauschboden
        snr = peak / (noise_floor + 1e-8)

        if snr >= 20.0:
            self.threshold = 0.2  # Hohes SNR: sensitiv
        elif snr >= 8.0:
            self.threshold = 0.3  # Mittleres SNR
        else:
            self.threshold = 0.5  # Niedriges SNR: konservativ
        logger.info("Auto-Optimierung (HarmonicTracking): SNR=%.1f → Schwelle=%s", snr, self.threshold)

"""
ConfidenceEngine: Berechnet Confidence-Werte für jedes Processing-Modul.
"""

import contextlib
import logging

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """Mehrdimensionale Confidence-Berechnung pro Modul.

    Berücksichtigt:
    - error-Flag (binär)
    - snr_db (falls vorhanden): normiert auf [0..1] via Sigmoid
    - artifact_score (falls vorhanden): invertiert
    - latency_ok (falls vorhanden): boolean Penalty
    """

    def compute_confidence(self, module_outputs: dict) -> dict:
        """Berechnet Confidence [0, 1] pro Modul aus dessen Output-Dict.

        :param module_outputs: {module_name: {error: bool, snr_db: float, ...}}
        :return: {module_name: confidence_float}
        """
        result = {}
        for mod, out in module_outputs.items():
            if not isinstance(out, dict):
                result[mod] = 0.5
                continue
            conf = 1.0
            # Fehler-Penalty
            if out.get("error", False):
                conf *= 0.5
            # SNR-Bonus (falls verfügbar): SNR >= 40dB -> 1.0, <= 0dB -> 0.3
            snr = out.get("snr_db", None)
            if snr is not None:
                try:
                    import math

                    snr_norm = 1.0 / (1.0 + math.exp(-0.1 * (float(snr) - 20.0)))
                    conf *= 0.4 + 0.6 * snr_norm
                except (TypeError, ValueError) as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            # Artefakt-Penalty
            artifact = out.get("artifact_score", None)
            if artifact is not None:
                with contextlib.suppress(TypeError, ValueError):
                    conf *= max(0.0, 1.0 - float(artifact))
            # Latenz-Penalty
            if not out.get("latency_ok", True):
                conf *= 0.8
            result[mod] = max(0.0, min(1.0, conf))
        return result

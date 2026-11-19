"""
Epistemic Gate — prüft vor jeder Verarbeitung, ob Eingabedaten verarbeitbar sind.
Keine Audioveränderung; läuft im Read-Only-Modus (§2.15 Uncertainty Quantification).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EpistemicGate:
    """Prüft Verarbeitungszuständigkeit anhand struktureller Audio-Validierung.

    Gibt False zurück wenn das Signal:
    - None oder leer ist
    - ausschließlich NaN/Inf enthält
    - vollständig Null (Stille) über die gesamte Länge ist
    - kürzer als 100 ms bei 48 000 Hz ist (< 4800 Samples)
    """

    MIN_SAMPLES: int = 4800  # 100 ms @ 48 000 Hz
    SILENCE_THRESHOLD: float = 1e-9  # RMS unter diesem Wert = Stille

    def check_responsibility(self, audio_data: Any) -> bool:
        """Prüft ob das System für audio_data zuständig und verarbeitungsfähig ist.

        Args:
            audio_data: np.ndarray (float32/64, mono oder stereo) oder None.

        Returns:
            True  → verarbeitbar, Pipeline darf starten.
            False → unverarbeitbar, Pipeline muss abbrechen.
        """
        if audio_data is None:
            logger.warning("[EpistemicGate] audio_data ist None — nicht zuständig.")
            return False

        try:
            arr = np.asarray(audio_data, dtype=np.float64)
        except Exception as exc:
            logger.warning("[EpistemicGate] Konvertierung fehlgeschlagen: %s", exc)
            return False

        if arr.size == 0:
            logger.warning("[EpistemicGate] Leeres Array — nicht zuständig.")
            return False

        if arr.size < self.MIN_SAMPLES:
            logger.warning(
                "[EpistemicGate] Signal zu kurz (%d Samples < %d) — nicht zuständig.",
                arr.size,
                self.MIN_SAMPLES,
            )
            return False

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            logger.warning("[EpistemicGate] Ausschließlich NaN/Inf — nicht zuständig.")
            return False

        rms = float(np.sqrt(np.mean(arr[finite_mask] ** 2)))
        if not math.isfinite(rms) or rms < self.SILENCE_THRESHOLD:
            logger.warning("[EpistemicGate] Signal vollständig stumm (RMS=%.2e) — nicht zuständig.", rms)
            return False

        logger.debug("[EpistemicGate] OK — RMS=%.4f samples=%d", rms, arr.size)
        return True

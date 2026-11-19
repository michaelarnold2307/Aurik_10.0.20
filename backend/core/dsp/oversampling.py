import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


@dataclass
class DSPContract:
    name: str = "Oversampler"
    version: str = "1.0"
    description: str = "SOTA-konformer Oversampler (Anti-Aliasing, Pre/Post-Processing)"
    parameters: dict[str, Any] | None = None


oversampler_contract = DSPContract(parameters={"factor": 2})


class Oversampler:
    """
    SOTA-konformer Oversampler (Anti-Aliasing, Pre/Post-Processing)
    """

    def __init__(self, factor: int = 2):
        self.factor = factor

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(oversampler_contract))

    def upsample(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        SOTA: Oversampling, Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Ungültige Eingabe für Oversampler.upsample")
            result = np.asarray(resample_poly(audio, self.factor, 1))
            self._audit_log({"mode": "upsample", "factor": self.factor, "shape": result.shape})
            return result
        except Exception as e:
            logger.error("[Oversampler][upsample][Fehler] %s", e)
            self._audit_log({"mode": "upsample", "error": str(e)})
            return audio

    def downsample(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        SOTA: Downsampling, Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Ungültige Eingabe für Oversampler.downsample")
            result = np.asarray(resample_poly(audio, 1, self.factor))
            self._audit_log({"mode": "downsample", "factor": self.factor, "shape": result.shape})
            return result
        except Exception as e:
            logger.error("[Oversampler][downsample][Fehler] %s", e)
            self._audit_log({"mode": "downsample", "error": str(e)})
            return audio

    def _audit_log(self, result: dict[str, Any]):
        logger.info("[AuditLog][Oversampler] Ergebnis: %s", result)

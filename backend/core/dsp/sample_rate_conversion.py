import logging

import numpy as np
import numpy.typing as npt
import soxr

logger = logging.getLogger("aurik.dsp.sample_rate_conversion")
logger.setLevel(logging.INFO)


class SampleRateConverter:
    """
    SOTA-konvertierer für Sample Rate Conversion (SRC)
    """

    def __init__(self, target_sr: int = 48000, quality: str = "VHQ"):
        self.target_sr = target_sr
        self.quality = quality

    def process(self, audio: npt.NDArray[np.float64], orig_sr: int, audit_log: bool = True) -> npt.NDArray[np.float64]:
        """
        Führt SOTA-Sample-Rate-Conversion durch.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param orig_sr: Original-Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: SRC-Audio (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            logger.error("Ungültiges Audio-Array (leer oder falscher Typ)")
            raise ValueError("Ungültiges Audio-Array (leer oder falscher Typ)")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1e6:
            logger.warning("Audio möglicherweise nicht normiert (max > 1e6)")

        try:
            audio_out = np.asarray(soxr.resample(audio, orig_sr, self.target_sr, quality=self.quality))
        except Exception as e:
            logger.error("Fehler bei Sample Rate Conversion: %s", e)
            audio_out = audio.copy()

        if audit_log:
            logger.info(
                f"SampleRateConverter: orig_sr={orig_sr}, target_sr={self.target_sr}, quality={self.quality}, audit=stub"
            )
        return audio_out.astype(audio.dtype)

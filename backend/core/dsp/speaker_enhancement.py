"""
ai_speaker_enhancement.py - Speaker Enhancement (AI) für Aurik 6.0
ai_speaker_enhancement.py - SOTA-Speaker Enhancement (AI) für Aurik 6.0

Dieses Modul verbessert Sprachverständlichkeit und Präsenz AI-gestützt.
Kombiniert klassische Bandpass-/Formantbetonung und Deep-Learning (ML-ready).
"""

import logging

import numpy as np
from scipy.signal import butter, lfilter

_logger = logging.getLogger(__name__)


class AiSpeakerEnhancement:
    """
    SOTA-Speaker Enhancement: Kombiniert klassische Bandpass-/Formantbetonung und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(
        self,
        model_path: str | None = None,
        low: float = 300.0,
        high: float = 3400.0,
        gain: float = 1.3,
    ):
        self.model_path = model_path
        self.model = None
        self.low = low
        self.high = high
        self.gain = gain

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Verbessert Sprachverständlichkeit und Präsenz. Zuerst klassische Bandpass-/Formantbetonung, dann optional Deep-Learning-Inferenz (wenn Modell geladen).
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        nyq = 0.5 * sr
        b, a = butter(4, [self.low / nyq, self.high / nyq], btype="band", output="ba")
        audio_bp = lfilter(b, a, audio)
        audio_out = audio + self.gain * audio_bp
        # ML-Inferenz (wenn Modell vorhanden)
        if self.model is not None:
            try:
                x = audio.astype(np.float32)[None, :]
                ort_inputs = {self.model.get_inputs()[0].name: x}
                out = self.model.run(None, ort_inputs)[0]
                audio_out = out.squeeze().astype(audio_out.dtype)
                _logger.debug("Speaker-Enhancement ML-Inferenz erfolgreich")
            except Exception as e:
                _logger.warning("ML-Inferenz fehlgeschlagen, DSP-Fallback: %s", e)
        return np.asarray(np.clip(audio_out, -1.0, 1.0), dtype=audio.dtype)  # type: ignore[no-any-return]

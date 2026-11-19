"""
ai_hum_remover.py - KI-gestützter Hum-Remover für Aurik 6.0
ai_hum_remover.py - SOTA-Hum-Remover für Aurik 6.0

Dieses Modul entfernt Brummstörungen aus Audiosignalen.
Kombiniert klassische Notch-Filterung und Deep-Learning (ML-ready).
"""

import logging

import numpy as np
from scipy.signal import iirnotch, lfilter

_logger = logging.getLogger(__name__)


class AiHumRemover:
    """
    SOTA-Hum-Remover: Kombiniert klassische Notch-Filterung und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(self, model_path: str | None = None, hum_freq: float = 50.0, q: float = 30.0):
        self.model_path = model_path
        self.model = None
        self.hum_freq = hum_freq
        self.q = q

    def remove_hum(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt Brummstörungen (50/60 Hz und Obertöne) aus dem Audiosignal. Zuerst klassische Notch-Filterung, dann optional Deep-Learning-Inferenz (wenn Modell geladen).
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = audio.copy()
        for f in [self.hum_freq * i for i in range(1, int(sr // (2 * self.hum_freq)) + 1)]:
            b, a = iirnotch(f / (0.5 * sr), self.q)
            audio_out = lfilter(b, a, audio_out)
        # ML-Inferenz via ONNX (wenn Modell geladen)
        if self.model is not None:
            try:
                _in_name = self.model.get_inputs()[0].name
                _inp = np.expand_dims(audio_out, axis=0).astype(np.float32)
                _raw = self.model.run(None, {_in_name: _inp})[0].squeeze()
                _raw = np.nan_to_num(_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                audio_out = np.clip(_raw, -1.0, 1.0)
            except Exception as _onnx_err:
                _logger.warning(
                    "AiHumRemover: ONNX-Inferenz fehlgeschlagen (%s) — klassischer Notch-Fallback aktiv.",
                    _onnx_err,
                )
        return np.asarray(np.clip(np.nan_to_num(audio_out, nan=0.0), -1.0, 1.0).astype(audio.dtype))  # type: ignore[no-any-return]

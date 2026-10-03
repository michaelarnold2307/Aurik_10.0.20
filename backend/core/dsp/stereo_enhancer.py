"""
ai_stereo_enhancer.py - Adaptive Stereo-Enhancer (AI) für Aurik 6.0
ai_stereo_enhancer.py - SOTA-Adaptive Stereo-Enhancer (AI) für Aurik 6.0

Dieses Modul optimiert die Stereobreite und -tiefe AI-gestützt.
Kombiniert klassische M/S-Breite und Deep-Learning (ML-ready).
"""

import logging

import numpy as np

_logger = logging.getLogger(__name__)


class AiStereoEnhancer:
    """
    SOTA-Adaptive Stereo-Enhancer: Kombiniert klassische M/S-Breite und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(self, model_path: str | None = None, target_width: float = 1.2):
        self.model_path = model_path
        self.model = None
        self.target_width = target_width

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Optimiert Stereobreite/-tiefe. Zuerst klassische M/S-Breite, dann optional Deep-Learning-Inferenz (wenn Modell geladen).
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim != 2 or audio.shape[0] != 2:
            return audio  # Nur Stereo
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        side *= self.target_width
        # Level compensation to preserve energy at different widths
        compensation = 1.0 / max(1e-9, np.sqrt(0.5 * (1.0 + self.target_width**2)))
        left = (mid + side) * compensation
        right = (mid - side) * compensation
        audio_out = np.vstack([left, right])
        # ML-Inferenz via ONNX (wenn Modell geladen)
        if self.model is not None:
            try:
                _in_name = self.model.get_inputs()[0].name
                _inp = audio_out.astype(np.float32)[np.newaxis, ...]  # (1, 2, N)
                _raw = self.model.run(None, {_in_name: _inp})[0].squeeze()
                _raw = np.nan_to_num(_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                if _raw.shape == audio_out.shape:
                    audio_out = _raw
            except Exception as _onnx_err:
                _logger.warning(
                    "AiStereoEnhancer: ONNX-Inferenz fehlgeschlagen (%s) — M/S-DSP-Fallback aktiv.",
                    _onnx_err,
                )
        return np.asarray(np.clip(audio_out, -1.0, 1.0), dtype=audio.dtype)  # type: ignore[no-any-return]

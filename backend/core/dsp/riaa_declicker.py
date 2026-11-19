"""
riaa_declicker.py — KI-gestützter RIAA-Declicker für Aurik.

Entfernt Klicks/Knackser speziell im RIAA-Signalweg: optionale
ML-Inferenz (ONNX-CPU/Torch) oder klassischer DSP-Pfad über den
kanonischen Kern :mod:`backend.core.dsp.declick_core` (robuste
Impuls-Erkennung + margen-dosierte AR-Interpolation). Jeder Lauf wird
am Ausgang doppelt abgenommen (``guard_declick``). Jeder Rückfall ist
§V6 (VERBOTEN.md)-konform geloggt (Hörordnung Ebene 2).
"""

import logging

import numpy as np

from backend.core.dsp.declick_core import (
    declick_signal,
    guard_declick,
    load_declick_model,
    run_declick_model,
    strictness_from_sensitivity,
)

logger = logging.getLogger(__name__)


class AiRiaaDeclicker:
    """
    SOTA RIAA-Declicker:
    - ML-Inferenz (ONNX-CPU/Torch) wenn ein Modell konfiguriert ist
    - Robuste Pulsdetektion + margen-dosierte AR-Interpolation als DSP-Pfad
    """

    def __init__(self, model_path: str | None = None, sensitivity: float = 1.0):
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.model = None  # Legacy alias
        self.onnx_session = None
        self.torch_model = None
        self.backend = None
        if model_path:
            self.model, self.backend = load_declick_model(model_path, "riaa_declicker_onnx")
            if self.backend == "onnx":
                self.onnx_session = self.model
            elif self.backend == "torch":
                self.torch_model = self.model

    def declick_riaa(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> np.ndarray:
        """
        Entfernt Klicks/Knackser nach RIAA-Kennlinie.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung, optionale ML-Inferenz, Rückfallstrategie
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Deklicktes Audio (np.ndarray)
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

        audio_out = None
        fallback_used = False
        try:
            if self.model is not None:
                audio_out = run_declick_model(self.model, self.backend, audio)
                fallback_used = audio_out is None
            if audio_out is None:
                # DSP-Pfad: Robuste Pulsdetektion + margen-dosierte AR-Interpolation
                audio_out = declick_signal(
                    audio,
                    strictness_k=strictness_from_sensitivity(self.sensitivity),
                    method="levinson",
                    sr=sr,
                )
                fallback_used = True
        except Exception as e:
            logger.error("Fehler beim RIAA-Declicking: %s", e)
            audio_out = audio.copy()
            fallback_used = True

        if audit_log:
            declicking_error = float(np.mean(np.abs(audio - audio_out)))
            logger.info("AiRiaaDeclicker: declicking_error=%.4f, Ersatzpfad_used=%s", declicking_error, fallback_used)
        # Hörordnung-Ebene-2-Zielabnahme (Materialerhalt + Rest-unter-Ziel).
        audio_out = guard_declick(audio, audio_out, sr, label="AiRiaaDeclicker")
        return np.asarray(audio_out.astype(audio.dtype))  # type: ignore[no-any-return]

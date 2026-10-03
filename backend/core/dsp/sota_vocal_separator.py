"""
sota_vocal_separator.py - SOTA-Source-Separation für Aurik 6.0
Produktive Integration von Hybrid Demucs/Banquet für Vocal-/Instrumenten-Separation.
"""

import logging
import os

import numpy as np
import onnxruntime as ort

from dsp._memory_budget_guard import check_budget

logger = logging.getLogger(__name__)

MODEL_PATH_BANQUET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/banquet/banquet_vinyl_final.onnx")
)
MODEL_PATH_UVR_MDX_NET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/uvr_mdx_net/uvr_mdx_net_inst_hq_1.onnx")
)
MODEL_PATH_DEMUCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/demucs/htdemucs_6s.onnx"))


class SotaVocalSeparator:
    def __init__(self, model_path=None, use_uvr=False, use_demucs=False):
        if use_demucs:
            self.model_path = model_path or MODEL_PATH_DEMUCS
        elif use_uvr:
            self.model_path = model_path or MODEL_PATH_UVR_MDX_NET
        else:
            self.model_path = model_path or MODEL_PATH_BANQUET
        self.session = None
        if os.path.exists(self.model_path):
            _model_size_gb = os.path.getsize(self.model_path) / (1024**3)
            if check_budget("sota_vocal_separator", max(0.1, _model_size_gb)):
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            else:
                logger.warning("Speicher-Grenze für Vocal-Separator überschritten — Rückgabe des Originalsignals")
        else:
            logger.warning("[WARN] ONNX-Modell nicht gefunden: %s", self.model_path)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        result, _ = self.process_with_confidence(audio, sr)
        return result

    def process_with_confidence(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """Separate vocals and return (audio, vocal_confidence).

        vocal_confidence is in [0, 1]: ratio of separated vocal energy
        to total energy. 0.0 means no model / fallback.
        """
        if self.session is None:
            logger.warning("[WARN] Kein ONNX-Modell geladen, Rückgabe des Originalsignals.")
            return audio, 0.0
        x = audio.astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        ort_inputs = {self.session.get.inputs()[0].name: x}
        try:
            ort_outs = self.session.run(None, ort_inputs)
            separated = np.asarray(ort_outs[0].squeeze())
            # Estimate vocal confidence as energy ratio
            sep_energy = float(np.sum(separated**2))
            orig_energy = float(np.sum(audio.astype(np.float32) ** 2)) + 1e-10
            vocal_confidence = float(np.clip(sep_energy / orig_energy, 0.0, 1.0))
            return separated, vocal_confidence
        except Exception as e:
            logger.error("[ERROR] Inferenz fehlgeschlagen: %s", e)
            return audio, 0.0

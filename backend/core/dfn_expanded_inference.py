"""§v10.123: DFN Expanded Inference — ONNX-first Denoiser (Musik-Finetune).

§v10-DFN-ONNX (2026-09-10): Der frühere eager-Pfad lud
models/deepfilternet_v3_ii/finetuned/dfn_expanded_best.pt — die Datei existiert
nicht mehr, der Pfad war tot (FileNotFoundError im Konstruktor, stiller
Rückfall der SOTA-Pipeline auf das ungefilterte Signal). Primär ist jetzt der
dreiteilige ONNX-Pfad (enc + dec + erb_dec) des dfn_musik_best-Finetunes; diese
Klasse ist ein schlanker Adapter auf plugins/deepfilternet_v3_ii_plugin.py,
dessen Kette (ERB-Gain + Deep-Filter + ISTFT) produktiv etabliert und getestet
ist. Fallbacks übernimmt das Plugin §V6-konform (copilot-instructions.md) mit
Warnung + Grund: OMLSA → Spectral-Gating → Dry.
GPU-Provider über backend/core/ml_device_manager (Plugin-Verhalten unverändert).

Nutzung:
    from backend.core.dfn_expanded_inference import DFNExpandedDenoiser
    denoiser = DFNExpandedDenoiser()
    clean = denoiser.denoise(noisy_audio, sample_rate)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SR = 48_000

_FINETUNED_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "deepfilternet_v3_ii" / "finetuned"


class DFNExpandedDenoiser:
    """DFN-Musik-Denoiser (ONNX) — ehemals „DFN Expanded" (eager)."""

    def __init__(self) -> None:
        from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

        self._plugin = DeepFilterNetV3Plugin(model_dir=str(_FINETUNED_DIR))
        self._loaded = getattr(self._plugin, "_enc", None) is not None
        if self._loaded:
            logger.info("DFN Expanded (ONNX, finetuned) geladen: %s", _FINETUNED_DIR)
        else:
            # §V6 (copilot-instructions.md): kein Silent-Failure — das Plugin hat
            # bereits Warnung + Grund geloggt und fällt auf DSP-Ersatzpfade zurück.
            logger.warning("DFN Expanded (ONNX, finetuned) nicht ladbar — DSP-Ersatzpfade aktiv.")

    def denoise(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Entrauscht Audio via DFN-Musik-ONNX.

        Args:
            audio: float32 mono [N] oder stereo [2, N]/[N, 2].
            sample_rate: Sample-Rate in Hz (wird intern auf 48 kHz resampelt
                         und zurück).

        Returns:
            Denoisiertes Audio, gleiche Form wie Input, float32 ∈ [-1, 1].
        """
        return self._plugin.enhance(audio, sample_rate)

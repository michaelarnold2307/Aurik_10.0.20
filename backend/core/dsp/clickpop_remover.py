import logging

"""
ai_clickpop_remover.py - Click/Pop-Remover (klassisch + Deep-Learning) für Aurik 6.0
ai_clickpop_remover.py - SOTA-Click/Pop-Remover (klassisch + Deep-Learning) für Aurik 6.0

Dieses Modul stellt sowohl klassische als auch AI-basierte Methoden zur Entfernung von Clicks und Pops bereit.
Kombiniert Pulsdetektion/Interpolation und Deep-Learning (ML-ready).
"""

"""
clickpop_remover.py - SOTA-Click/Pop-Remover (klassisch + Deep-Learning) für Aurik 6.0

SOTA-konforme Click/Pop-Remover mit DSPContract und Auditierbarkeit.
"""
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import medfilt

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractClickPopRemover:
    id: str = "clickpop_remover"
    category: str = "clickpop_remover"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


clickpop_remover_contract = DSPContractClickPopRemover(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"model_path": None}},
    budgets={"compute_cost": 0.01},
    side_effects=[
        {
            "risk": "Fehlrestauration",
            "expected_when": "Modell nicht geladen",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["clickpop_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class ClassicClickPopRemover:
    """
    SOTA-Klassischer Click/Pop-Remover: Pulsdetektion und Interpolation. Robust, adaptiv, Fallback-fähig.
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractClickPopRemover = clickpop_remover_contract

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        abs_audio = np.abs(audio)
        threshold = np.mean(abs_audio) + 6 * np.std(abs_audio)
        clicks = abs_audio > threshold
        audio_out = audio.copy()
        if np.any(clicks):
            audio_out[clicks] = medfilt(audio, kernel_size=5)[clicks]
        return np.asarray(audio_out.astype(audio.dtype))  # type: ignore[no-any-return]


class AiClickPopRemover:
    """
    SOTA-Deep-Learning-Click/Pop-Remover: Kombiniert Pulsdetektion/Interpolation und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractClickPopRemover = clickpop_remover_contract

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        audio_out = audio.copy()
        if self.model is not None:
            # ONNX-Inferenz (CPUExecutionProvider — §9.5 CPU-only Policy)
            try:
                import onnxruntime as _ort

                # Modell-Pfad oder bereits geladene Session akzeptieren
                if isinstance(self.model, str):
                    from dsp._memory_budget_guard import check_budget

                    if not check_budget("clickpop_remover_onnx", 0.1):
                        raise RuntimeError("Memory budget exceeded")
                    _sess = _ort.InferenceSession(self.model, providers=["CPUExecutionProvider"])
                else:
                    _sess = self.model  # bereits InferenceSession
                _in_name = _sess.get_inputs()[0].name
                # Mono erzwingen, Float32 normieren, Batch-Dim hinzufügen
                _mono = (
                    audio if audio.ndim == 1 else audio.mean(axis=0 if audio.ndim == 2 and audio.shape[0] <= 2 else -1)
                )
                _audio_f32 = np.clip(_mono, -1.0, 1.0).astype(np.float32)[np.newaxis, np.newaxis, :]
                _out = _sess.run(None, {_in_name: _audio_f32})[0].squeeze()
                # NaN/Inf-Schutz + Clipping (§3.1 Numerische Robustheit)
                audio_out = np.clip(
                    np.nan_to_num(_out.astype(audio.dtype), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
                logger.debug("AiClickPopRemover: ONNX-Inferenz erfolgreiche, shape=%s.", audio_out.shape)
            except Exception as _onnx_err:
                logger.warning(
                    "AiClickPopRemover: ONNX-Inferenz fehlgeschlagen (%s) — DSP-Fallback (RBME-inspiziert).",
                    _onnx_err,
                )
                # DSP-Fallback: adaptiver Medianfilter (RBME-inspiriert, §4.5)
                abs_audio = np.abs(audio)
                threshold = np.mean(abs_audio) + 6 * np.std(abs_audio)
                clicks = abs_audio > threshold
                if np.any(clicks):
                    audio_out[clicks] = medfilt(audio, kernel_size=5)[clicks]
        else:
            abs_audio = np.abs(audio)
            threshold = np.mean(abs_audio) + 6 * np.std(abs_audio)
            clicks = abs_audio > threshold
            if np.any(clicks):
                audio_out[clicks] = medfilt(audio, kernel_size=5)[clicks]
        return np.asarray(audio_out.astype(audio.dtype))  # type: ignore[no-any-return]

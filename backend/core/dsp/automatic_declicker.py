"""
automatic_declicker.py — SOTA-Automatic Declicker für Aurik.

Entfernt Klicks/Knackser aus Audiosignalen: robuste Impuls-Erkennung
(lokal adaptive Residuum-Textur) und musikalische AR-Interpolation
(Burg) über den kanonischen Kern :mod:`backend.core.dsp.declick_core`.
Die Reparaturstärke ist margen-dosiert (Stärke ∝ Hörbarkeits-Marge,
Hörordnung §4) und jeder Lauf wird am Ausgang doppelt abgenommen
(Materialerhalt + Rest-unter-Ziel, ``guard_declick``).
Optionale ML-Inferenz (ONNX-CPU/Torch) vor dem klassischen Pfad;
jeder Rückfall ist §V6 (VERBOTEN.md)-konform geloggt. Auditierbar,
rollback-fähig, bit-identisch außerhalb messbarer Defekte.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from backend.core.dsp.declick_core import (
    declick_signal,
    guard_declick,
    load_declick_model,
    run_declick_model,
    strictness_from_threshold,
)

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_declicker"
    category: str = "disruptor_removal"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
declicker_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.6},
        "safe_ranges": {"threshold": {"min": 0.1, "max": 1.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.05,
        "identity_budget": 0.99,
        "spectral_change_budget": 0.1,
        "temporal_change_budget": 0.05,
        "compute_cost": 0.05,
    },
    side_effects=[{"risk": "transient_smear", "expected_when": "threshold < 0.2", "severity": 0.2}],
    reports={"self_metrics": ["click_removal_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutomaticDeclicker:
    def process(self, audio: "np.ndarray", sr: int) -> "np.ndarray":
        """Alias für declick() (Kompatibilität mit älteren Pipelines/Tests)."""
        return self.declick(audio, sr)

    def __init__(self, threshold: float = 0.6, model_path: str | None = None):
        self.threshold = threshold
        self.model_path = model_path
        self.model, self.backend = (
            load_declick_model(model_path, "automatic_declicker_model") if model_path else (None, None)  # type: ignore[var-annotated]  # type: ignore[var-annotated]  # type: ignore[var-annotated]  # type: ignore[var-annotated]
        )

    def declick(
        self, audio: np.ndarray, sr: int, use_deep_learning: bool = False, audit_log: bool = True
    ) -> np.ndarray:
        """
        Entfernt Klicks/Knackser per robuster Pulsdetektion und margen-
        dosierter AR-Interpolation (SOTA, keine ML/AI im klassischen Pfad).
        Quality Gate, Audit-Logging, optionale ML-Inferenz, robuste Fehlerbehandlung
        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :param use_deep_learning: Optional ML-Inferenz (braucht model_path)
        :param audit_log: Audit-Logging aktivieren
        :return: De-clicktes Signal (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr < 8000:
            logger.error("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
            raise ValueError("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1.5:
            logger.warning("Audio möglicherweise nicht normiert (max > 1.5)")

        audio_out = None
        fallback_used = False
        try:
            if use_deep_learning and self.model is not None:
                logger.info("ML-Inferenz aktiviert für Declicking (ONNX-CPU/Torch).")
                audio_out = run_declick_model(self.model, self.backend, audio)
                fallback_used = audio_out is None
            elif use_deep_learning:
                logger.warning("use_deep_learning ohne geladenes Modell — DSP-Ersatzpfad (§V6 (VERBOTEN.md))")
            if audio_out is None:
                audio_out = self._declick_classic(audio, sr)
        except Exception as e:
            logger.error("Fehler bei Declicking: %s", e)
            fallback_used = True
            audio_out = audio.copy()

        if audit_log:
            click_removal_score = float(np.mean(np.abs(audio - audio_out)))
            logger.info(
                f"AutomaticDeclicker: click_removal_score={click_removal_score:.4f}, fallback_used={fallback_used}"
            )
        logger.debug("[DSPContract] %s", asdict(declicker_contract))
        # Hörordnung-Ebene-2-Zielabnahme (Materialerhalt + Rest-unter-Ziel).
        audio_out = guard_declick(audio, audio_out, sr, label="AutomaticDeclicker")
        return np.asarray(audio_out.astype(audio.dtype))  # type: ignore[no-any-return]

    def _declick_classic(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Robuste Klick-Erkennung + margen-dosierte AR-Interpolation (declick_core)."""
        return declick_signal(audio, strictness_k=strictness_from_threshold(self.threshold), sr=sr)

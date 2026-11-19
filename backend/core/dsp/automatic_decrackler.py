"""
automatic_decrackler.py — SOTA-Automatic Decrackler für Aurik.

Entfernt Crackle/Knister (dichte Mikro-Impulse) aus Audiosignalen:
robuste Impuls-Erkennung + margen-dosierte AR-Interpolation nach
Levinson-Durbin (:mod:`backend.core.dsp.adaptive_ar_prediction_levinson`),
weil die Autokorrelation über viele Impulse mittelt. Kanonischer Kern:
:mod:`backend.core.dsp.declick_core`. Jeder Lauf wird am Ausgang doppelt
abgenommen (``guard_declick``: Materialerhalt + Rest-unter-Ziel).
Auditierbar, rollback-fähig, bit-identisch außerhalb messbarer Defekte
(Hörordnung Ebene 2).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from backend.core.dsp.declick_core import declick_signal, guard_declick, strictness_from_threshold

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_decrackler"
    category: str = "decrackler"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
automatic_decrackler_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.4},
        "safe_ranges": {"threshold": {"min": 0.05, "max": 1.0}},
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
    reports={"self_metrics": ["crackle_removal_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutomaticDecrackler:
    """
    Klassischer automatischer Decrackler (SOTA-Maximum, keine ML/AI):
    - Entfernt Crackle/Knister adaptiv per Pulsdetektion und margen-dosierter
      Interpolation (Levinson-Durbin)
    - Auditierbar, rollback-fähig
    """

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(automatic_decrackler_contract))

    def decrackle(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Crackle/Knister: robuste Detektion + Levinson-AR-Reparatur, margen-dosiert."""
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        out = declick_signal(
            audio,
            strictness_k=strictness_from_threshold(self.threshold),
            med_kernel=3,
            merge_gap=1,
            method="levinson",
            sr=sr,
        )
        # Hörordnung-Ebene-2-Zielabnahme (Materialerhalt + Rest-unter-Ziel).
        out = guard_declick(audio, out, sr, label="AutomaticDecrackler")
        return np.asarray(out.astype(audio.dtype))  # type: ignore[no-any-return]

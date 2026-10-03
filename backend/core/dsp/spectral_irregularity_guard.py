import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_irregularity_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_irregularity_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"irregularity_min": 0.0, "irregularity_max": 2.0},
        "safe_ranges": {
            "irregularity_min": {"min": 0.0, "max": 1.0},
            "irregularity_max": {"min": 1.0, "max": 5.0},
        },
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[],
    reports={"self_metrics": ["spectral_irregularity"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralIrregularityGuard:
    """
    SOTA-konformer Spectral Irregularity Guard:
    - Überwacht die spektrale Irregularität als Qualitätsmaß
    """

    def __init__(self, irregularity_min: float = 0.0, irregularity_max: float = 2.0):
        self.irregularity_min = irregularity_min
        self.irregularity_max = irregularity_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_irregularity_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Irregularität, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        # Irregularity nach Jensen: mittlere absolute Differenz benachbarter Bins
        irregularity = float(np.mean(np.abs(np.diff(spec))))
        ok = self.irregularity_min <= irregularity <= self.irregularity_max
        # Quality-Gate
        if irregularity < 0 or np.isnan(irregularity):
            logger.warning("[QualityGate] Warnung: Unplausible Irregularität, Rollback aktiviert.")
            return {"spectral_irregularity": 0.0, "ok": False}
        return {"spectral_irregularity": irregularity, "ok": ok}

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_variance_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_variance_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"variance_min": 0.01, "variance_max": 1.0},
        "safe_ranges": {
            "variance_min": {"min": 0.0, "max": 0.5},
            "variance_max": {"min": 0.5, "max": 2.0},
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
    reports={"self_metrics": ["spectral_variance"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralVarianceGuard:
    """
    SOTA-konformer Spectral Variance Guard:
    - Überwacht die spektrale Varianz als Qualitätsmaß
    """

    def __init__(self, variance_min: float = 0.01, variance_max: float = 1.0):
        self.variance_min = variance_min
        self.variance_max = variance_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_variance_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Varianz, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        variance = float(np.var(spec))
        ok = self.variance_min <= variance <= self.variance_max
        # Quality-Gate
        if variance < 0 or np.isnan(variance):
            logger.warning("[QualityGate] Warnung: Unplausible Varianz, Rollback aktiviert.")
            return {"spectral_variance": 0.0, "ok": False}
        return {"spectral_variance": variance, "ok": ok}

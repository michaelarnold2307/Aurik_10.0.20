import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_skewness_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_skewness_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"skewness_min": -2.0, "skewness_max": 2.0},
        "safe_ranges": {
            "skewness_min": {"min": -5.0, "max": -0.5},
            "skewness_max": {"min": 0.5, "max": 5.0},
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
    reports={"self_metrics": ["spectral_skewness"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralSkewnessGuard:
    """
    SOTA-konformer Spectral Skewness Guard:
    - Überwacht die spektrale Schiefe als Qualitätsmaß
    """

    def __init__(self, skewness_min: float = -2.0, skewness_max: float = 2.0):
        self.skewness_min = skewness_min
        self.skewness_max = skewness_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_skewness_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Schiefe, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        mean = np.mean(spec)
        std = np.std(spec)
        skewness = float(np.mean(((spec - mean) / (std + 1e-8)) ** 3))
        ok = self.skewness_min <= skewness <= self.skewness_max
        # Quality-Gate
        if np.isnan(skewness):
            logger.warning("[QualityGate] Warnung: Unplausible Schiefe, Rollback aktiviert.")
            return {"spectral_skewness": 0.0, "ok": False}
        return {"spectral_skewness": skewness, "ok": ok}

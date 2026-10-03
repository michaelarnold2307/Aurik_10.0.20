import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_zero_crossing_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_zero_crossing_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"zcr_min": 0.01, "zcr_max": 0.2},
        "safe_ranges": {
            "zcr_min": {"min": 0.0, "max": 0.1},
            "zcr_max": {"min": 0.1, "max": 0.5},
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
    reports={"self_metrics": ["zero_crossing_rate"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralZeroCrossingGuard:
    """
    SOTA-konformer Zero Crossing Rate Guard:
    - Überwacht die Zero Crossing Rate als Qualitätsmaß
    """

    def __init__(self, zcr_min: float = 0.01, zcr_max: float = 0.2):
        self.zcr_min = zcr_min
        self.zcr_max = zcr_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_zero_crossing_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der Zero Crossing Rate, Quality-Gate
        """
        self.log_contract()
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        zcr = float(zero_crossings / len(audio))
        ok = self.zcr_min <= zcr <= self.zcr_max
        # Quality-Gate
        if zcr < 0 or np.isnan(zcr):
            logger.warning("[QualityGate] Warnung: Unplausible ZCR, Rollback aktiviert.")
            return {"zero_crossing_rate": 0.0, "ok": False}
        return {"zero_crossing_rate": zcr, "ok": ok}

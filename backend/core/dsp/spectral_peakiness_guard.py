import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_peakiness_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_peakiness_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"peakiness_min": 1.0, "peakiness_max": 10.0},
        "safe_ranges": {
            "peakiness_min": {"min": 0.5, "max": 5.0},
            "peakiness_max": {"min": 5.0, "max": 20.0},
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
    reports={"self_metrics": ["spectral_peakiness"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralPeakinessGuard:
    """
    SOTA-konformer Spectral Peakiness Guard:
    - Überwacht die spektrale Spitzigkeit als Qualitätsmaß
    """

    def __init__(self, peakiness_min: float = 1.0, peakiness_max: float = 10.0):
        self.peakiness_min = peakiness_min
        self.peakiness_max = peakiness_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_peakiness_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Spitzigkeit, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        peakiness = float(np.max(spec) / (np.mean(spec) + 1e-8))
        ok = self.peakiness_min <= peakiness <= self.peakiness_max
        # Quality-Gate
        if peakiness < 0 or np.isnan(peakiness):
            logger.warning("[QualityGate] Warnung: Unplausible Peakiness, Rollback aktiviert.")
            return {"spectral_peakiness": 0.0, "ok": False}
        return {"spectral_peakiness": peakiness, "ok": ok}

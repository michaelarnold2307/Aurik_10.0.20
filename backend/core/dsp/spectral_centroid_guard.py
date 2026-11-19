import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_centroid_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_centroid_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"centroid_min": 500.0, "centroid_max": 6000.0},
        "safe_ranges": {
            "centroid_min": {"min": 20.0, "max": 2000.0},
            "centroid_max": {"min": 2000.0, "max": 20000.0},
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
    reports={"self_metrics": ["spectral_centroid"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralCentroidGuard:
    """
    SOTA-konformer Spectral Centroid Guard:
    - Überwacht spektralen Schwerpunkt als Qualitätsmaß
    """

    def __init__(self, centroid_min: float = 500.0, centroid_max: float = 6000.0):
        self.centroid_min = centroid_min
        self.centroid_max = centroid_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_centroid_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung des spektralen Schwerpunkts, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-8))
        ok = self.centroid_min <= centroid <= self.centroid_max
        # Quality-Gate
        if centroid < 0 or np.isnan(centroid):
            logger.warning("[QualityGate] Warnung: Unplausibler Centroid, Rollback aktiviert.")
            return {"spectral_centroid": 0.0, "ok": False}
        return {"spectral_centroid": centroid, "ok": ok}

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_roughness_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_roughness_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"roughness_min": 0.0, "roughness_max": 1.0},
        "safe_ranges": {
            "roughness_min": {"min": 0.0, "max": 0.5},
            "roughness_max": {"min": 0.5, "max": 2.0},
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
    reports={"self_metrics": ["spectral_roughness"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralRoughnessGuard:
    """
    SOTA-konformer Spectral Roughness Guard:
    - Überwacht die spektrale Rauigkeit als Qualitätsmaß
    """

    def __init__(self, roughness_min: float = 0.0, roughness_max: float = 1.0):
        self.roughness_min = roughness_min
        self.roughness_max = roughness_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_roughness_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Rauigkeit, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        # Rauigkeit als mittlere Differenz benachbarter Bins, normiert
        roughness = float(np.mean(np.abs(np.diff(spec))) / (np.mean(spec) + 1e-8))
        ok = self.roughness_min <= roughness <= self.roughness_max
        # Quality-Gate
        if roughness < 0 or np.isnan(roughness):
            logger.warning("[QualityGate] Warnung: Unplausible Rauigkeit, Rollback aktiviert.")
            return {"spectral_roughness": 0.0, "ok": False}
        return {"spectral_roughness": roughness, "ok": ok}

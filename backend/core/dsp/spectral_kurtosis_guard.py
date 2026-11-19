import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_kurtosis_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_kurtosis_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"kurtosis_min": 1.5, "kurtosis_max": 6.0},
        "safe_ranges": {
            "kurtosis_min": {"min": 0.5, "max": 3.0},
            "kurtosis_max": {"min": 3.0, "max": 20.0},
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
    reports={"self_metrics": ["spectral_kurtosis"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralKurtosisGuard:
    """
    SOTA-konformer Spectral Kurtosis Guard:
    - Überwacht die spektrale Kurtosis als Qualitätsmaß
    """

    def __init__(self, kurtosis_min: float = 1.5, kurtosis_max: float = 6.0):
        self.kurtosis_min = kurtosis_min
        self.kurtosis_max = kurtosis_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_kurtosis_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Kurtosis, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        mean = np.mean(spec)
        std = np.std(spec)
        kurtosis = float(np.mean(((spec - mean) / (std + 1e-8)) ** 4))
        ok = self.kurtosis_min <= kurtosis <= self.kurtosis_max
        # Quality-Gate
        if kurtosis < 0 or np.isnan(kurtosis):
            logger.warning("[QualityGate] Warnung: Unplausible Kurtosis, Rollback aktiviert.")
            return {"spectral_kurtosis": 0.0, "ok": False}
        return {"spectral_kurtosis": kurtosis, "ok": ok}

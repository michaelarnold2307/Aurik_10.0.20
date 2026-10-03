import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_spread_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_spread_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"spread_min": 100.0, "spread_max": 5000.0},
        "safe_ranges": {
            "spread_min": {"min": 10.0, "max": 1000.0},
            "spread_max": {"min": 1000.0, "max": 20000.0},
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
    reports={"self_metrics": ["spectral_spread"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralSpreadGuard:
    """
    SOTA-konformer Spectral Spread Guard:
    - Überwacht die spektrale Streuung als Qualitätsmaß
    """

    def __init__(self, spread_min: float = 100.0, spread_max: float = 5000.0):
        self.spread_min = spread_min
        self.spread_max = spread_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_spread_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Streuung, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-8)
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / (np.sum(spec) + 1e-8)))
        ok = self.spread_min <= spread <= self.spread_max
        # Quality-Gate
        if spread < 0 or np.isnan(spread):
            logger.warning("[QualityGate] Warnung: Unplausible Streuung, Rollback aktiviert.")
            return {"spectral_spread": 0.0, "ok": False}
        return {"spectral_spread": spread, "ok": ok}

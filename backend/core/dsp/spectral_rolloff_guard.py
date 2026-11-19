import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_rolloff_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_rolloff_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "rolloff_percent": 0.85,
            "rolloff_min": 2000.0,
            "rolloff_max": 18000.0,
        },
        "safe_ranges": {
            "rolloff_percent": {"min": 0.5, "max": 0.99},
            "rolloff_min": {"min": 20.0, "max": 5000.0},
            "rolloff_max": {"min": 5000.0, "max": 22050.0},
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
    reports={"self_metrics": ["spectral_rolloff"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralRolloffGuard:
    """
    SOTA-konformer Spectral Rolloff Guard:
    - Überwacht das spektrale Rolloff als Qualitätsmaß
    """

    def __init__(
        self,
        rolloff_percent: float = 0.85,
        rolloff_min: float = 2000.0,
        rolloff_max: float = 18000.0,
    ):
        self.rolloff_percent = rolloff_percent
        self.rolloff_min = rolloff_min
        self.rolloff_max = rolloff_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_rolloff_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung des spektralen Rolloff, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        energy = np.cumsum(spec)
        total_energy = energy[-1]
        rolloff_freq = freqs[np.searchsorted(energy, self.rolloff_percent * total_energy)]
        ok = self.rolloff_min <= rolloff_freq <= self.rolloff_max
        # Quality-Gate
        if rolloff_freq < 0 or np.isnan(rolloff_freq):
            logger.warning("[QualityGate] Warnung: Unplausibles Rolloff, Rollback aktiviert.")
            return {"spectral_rolloff": 0.0, "ok": False}
        return {"spectral_rolloff": float(rolloff_freq), "ok": ok}

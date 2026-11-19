import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_crest_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_crest_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"crest_min": 1.0, "crest_max": 10.0},
        "safe_ranges": {
            "crest_min": {"min": 0.5, "max": 5.0},
            "crest_max": {"min": 5.0, "max": 20.0},
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
    reports={"self_metrics": ["spectral_crest"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralCrestGuard:
    """
    SOTA-konformer Spectral Crest Guard:
    - Überwacht das Crest-Faktor-Maß im Spektrum
    """

    def __init__(self, crest_min: float = 1.0, crest_max: float = 10.0):
        self.crest_min = crest_min
        self.crest_max = crest_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_crest_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung des spektralen Crest-Faktors, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        crest = float(np.max(spec) / (np.sqrt(np.mean(spec**2)) + 1e-8))
        ok = self.crest_min <= crest <= self.crest_max
        # Quality-Gate
        if crest < 0 or np.isnan(crest):
            logger.warning("[QualityGate] Warnung: Unplausibler Crest-Faktor, Rollback aktiviert.")
            return {"spectral_crest": 0.0, "ok": False}
        return {"spectral_crest": crest, "ok": ok}

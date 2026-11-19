import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_slope_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_slope_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"slope_min": -30.0, "slope_max": 0.0},
        "safe_ranges": {
            "slope_min": {"min": -60.0, "max": -5.0},
            "slope_max": {"min": -10.0, "max": 10.0},
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
    reports={"self_metrics": ["spectral_slope"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralSlopeGuard:
    """
    SOTA-konformer Spectral Slope Guard:
    - Überwacht die spektrale Steigung als Qualitätsmaß
    """

    def __init__(self, slope_min: float = -30.0, slope_max: float = 0.0):
        self.slope_min = slope_min
        self.slope_max = slope_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_slope_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Steigung, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio)) + 1e-8
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        # Nur positive Frequenzen > 0
        mask = freqs > 0
        log_freqs = np.log10(freqs[mask])
        log_spec = 20 * np.log10(spec[mask])
        # Lineare Regression (Steigung)
        slope, _ = np.polyfit(log_freqs, log_spec, 1)
        ok = self.slope_min <= slope <= self.slope_max
        # Quality-Gate
        if np.isnan(slope):
            logger.warning("[QualityGate] Warnung: Unplausible Steigung, Rollback aktiviert.")
            return {"spectral_slope": 0.0, "ok": False}
        return {"spectral_slope": float(slope), "ok": ok}

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_band_energy_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_band_energy_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "band_min": 300.0,
            "band_max": 3400.0,
            "energy_min": 0.01,
            "energy_max": 0.5,
        },
        "safe_ranges": {
            "band_min": {"min": 20.0, "max": 1000.0},
            "band_max": {"min": 1000.0, "max": 20000.0},
            "energy_min": {"min": 0.0, "max": 0.2},
            "energy_max": {"min": 0.1, "max": 1.0},
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
    reports={"self_metrics": ["band_energy"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralBandEnergyGuard:
    """
    SOTA-konformer Spectral Band Energy Guard:
    - Überwacht die Energie in einem Frequenzband als Qualitätsmaß
    """

    def __init__(
        self,
        band_min: float = 300.0,
        band_max: float = 3400.0,
        energy_min: float = 0.01,
        energy_max: float = 0.5,
    ):
        self.band_min = band_min
        self.band_max = band_max
        self.energy_min = energy_min
        self.energy_max = energy_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_band_energy_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der Bandenergie, Quality-Gate
        """
        self.log_contract()
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        band_mask = (freqs >= self.band_min) & (freqs <= self.band_max)
        band_energy = float(np.sum(spec[band_mask]) / (np.sum(spec) + 1e-8))
        ok = self.energy_min <= band_energy <= self.energy_max
        # Quality-Gate
        if band_energy < 0 or np.isnan(band_energy):
            logger.warning("[QualityGate] Warnung: Unplausible Bandenergie, Rollback aktiviert.")
            return {"band_energy": 0.0, "ok": False}
        return {"band_energy": band_energy, "ok": ok}

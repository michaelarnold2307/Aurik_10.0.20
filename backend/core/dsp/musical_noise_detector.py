import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "musical_noise_detector"
    category: str = "validator"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


musical_noise_detector_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.2},
        "safe_ranges": {"threshold": {"min": 0.05, "max": 0.5}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.02,
    },
    side_effects=[],
    reports={"self_metrics": ["musical_noise_score"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class MusicalNoiseDetector:
    """
    SOTA-konformer Musical Noise Detector:
    - Erkennt musikalisches Rauschen nach Denoising (z.B. Burble, Birdies)
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(musical_noise_detector_contract))

    def process(self, audio: np.ndarray, sr: int) -> bool:
        """Detektion von musikalischem Rauschen via spektraler Fluktuation.

        Hohe `std(diff(|FFT|)) / mean(|FFT|)` deutet auf instabile Spektrallinien
        (Artefakte durch überstarke Rauschunterdrückung) hin.
        """
        self.log_contract()
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Spektrale Fluktuation: hohe std(diff(spectrum)) relativ zum Mittel
        # deutet auf 'Musical Noise' (artefaktbedingte unstabile Spektrallinen) hin
        spec = np.abs(np.fft.rfft(audio))
        fluct = np.std(np.diff(spec)) / (np.mean(spec) + 1e-8)
        if fluct > self.threshold:
            _logger.warning(
                "[QualityGate] Warnung: Musical Noise erkannt (Score=%.2f), Nachbearbeitung empfohlen.", fluct
            )
            return True
        return False

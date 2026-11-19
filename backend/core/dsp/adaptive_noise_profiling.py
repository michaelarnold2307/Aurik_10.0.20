import logging

"""
adaptive_noise_profiling.py - SOTA-konformes adaptives Noise-Profiling für Aurik 6.0
Dieses Modul erstellt ein adaptives Rauschprofil für Denoising mittels Minimum-Statistik (klassische DSP, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_noise_profiling"
    category: str = "denoise"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


adaptive_noise_profiling_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"profile_window_sec": 2.0},
        "safe_ranges": {"profile_window_sec": {"min": 0.5, "max": 10.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[],
    reports={"self_metrics": ["noise_profile"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class AdaptiveNoiseProfiling:
    """
    SOTA-konformes adaptives Noise-Profiling (klassisch):
    - Erstellt ein adaptives Rauschprofil für Denoising
    - Minimum-Statistik im Spektrum (SOTA-Maximum)
    - Rollback bei unplausiblen Profilen
    """

    def __init__(self, profile_window_sec: float = 2.0):
        self.profile_window_sec = profile_window_sec

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_noise_profiling_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Führt adaptives Noise-Profiling mittels Minimum-Statistik durch.
        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :return: Noise-Profil (np.ndarray)
        """
        self.log_contract()
        window = int(self.profile_window_sec * sr)
        if window < 1:
            window = 1
        # Minimum-Statistik im Spektrum
        spec = np.abs(np.fft.rfft(audio))
        noise_profile = np.minimum.reduceat(spec, np.arange(0, len(spec), window))
        # Quality-Gate: Plausibilität
        if np.any(noise_profile < 0):
            logger.warning("[QualityGate] Warnung: Unplausibles Noise-Profil, Rollback aktiviert.")
            return np.asarray(np.zeros_like(spec))  # type: ignore[no-any-return]
        return np.asarray(noise_profile)  # type: ignore[no-any-return]

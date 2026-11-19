import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "dynamic_spectral_tilt"
    category: str = "eq"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


dynamic_spectral_tilt_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"tilt_db_per_oct": 0.0},
        "safe_ranges": {"tilt_db_per_oct": {"min": -6.0, "max": 6.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.05,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {
            "risk": "Klangfärbung",
            "expected_when": "tilt_db_per_oct > 3.0 or < -3.0",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["spectral_tilt"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class DynamicSpectralTilt:
    """
    SOTA-konformer Dynamic Spectral Tilt:
    - Passt die spektrale Balance adaptiv an (Tilt-Filter)
    """

    def __init__(self, tilt_db_per_oct: float = 0.0):
        self.tilt_db_per_oct = tilt_db_per_oct

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(dynamic_spectral_tilt_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        SOTA: Tilt-Filter with clamped gain curve and NaN guards.
        Phase is preserved (real-valued gain multiplication).
        """
        self.log_contract()
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))
        n = len(audio)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        spectrum = np.fft.rfft(audio)

        # Tilt curve: dB/oct relative to 1 kHz, clamped to ±12 dB
        ratio = np.where(freqs > 0, freqs / 1000.0, 1.0)
        tilt_db = self.tilt_db_per_oct * np.log2(np.maximum(ratio, 1e-6))
        tilt_db = np.clip(tilt_db, -12.0, 12.0)
        tilt_curve = 10.0 ** (tilt_db / 20.0)
        tilt_curve[0] = 1.0  # DC unverändert

        spectrum_tilted = spectrum * tilt_curve
        audio_tilted = np.fft.irfft(spectrum_tilted, n=n)
        audio_tilted = np.nan_to_num(audio_tilted, nan=0.0, posinf=0.0, neginf=0.0)
        audio_tilted = np.clip(audio_tilted, -1.0, 1.0)
        return np.asarray(audio_tilted)  # type: ignore[no-any-return]

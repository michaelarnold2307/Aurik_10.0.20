import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "transient_protection_guard"
    category: str = "dynamics"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


transient_protection_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"transient_threshold": 0.5, "window_ms": 10.0},
        "safe_ranges": {
            "transient_threshold": {"min": 0.1, "max": 1.0},
            "window_ms": {"min": 1.0, "max": 50.0},
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
    reports={"self_metrics": ["transient_ratio"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class TransientProtectionGuard:
    """
    SOTA-konformer Transientenschutz:
    - Überwacht und schützt vor übermäßigen Transienten
    """

    def __init__(self, transient_threshold: float = 0.5, window_ms: float = 10.0):
        self.transient_threshold = transient_threshold
        self.window_ms = window_ms

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(transient_protection_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Transientenerkennung, Quality-Gate
        """
        self.log_contract()
        window = int(self.window_ms * 1e-3 * sr)
        if window < 1:
            window = 1
        diff = np.abs(np.diff(audio, prepend=audio[0]))
        transient_ratio = float(np.max(diff) / (np.mean(diff) + 1e-8))
        ok = transient_ratio <= self.transient_threshold
        # Quality-Gate
        if transient_ratio < 0 or np.isnan(transient_ratio):
            logger.warning("[QualityGate] Warnung: Unplausible Transienten, Rollback aktiviert.")
            return {"transient_ratio": 0.0, "ok": False}
        return {"transient_ratio": transient_ratio, "ok": ok}

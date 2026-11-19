import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "stereo_coherence_guard"
    category: str = "stereo"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


stereo_coherence_guard_contract = DSPContract(
    io={
        "channels": "stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"min_corr": 0.1},
        "safe_ranges": {"min_corr": {"min": -1.0, "max": 0.5}},
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
    reports={"self_metrics": ["stereo_coherence"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class StereoCoherenceGuard:
    """
    SOTA-konformer Stereo Coherence Guard:
    - Überwacht die Stereokorrelation und warnt bei Phasenauslöschung/Monoproblemen
    """

    def __init__(self, min_corr: float = 0.1):
        self.min_corr = min_corr

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(stereo_coherence_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Misst stereo coherence (cross-correlation).
        Returns metadata dict with correlation value and warning status.
        """
        self.log_contract()
        if audio.ndim != 2 or audio.shape[0] != 2:
            logger.info("[StereoCoherenceGuard] Nicht Stereo — übersprungen.")
            return {
                "stereo": False,
                "correlation": float("nan"),
                "mono_problem": False,
            }
        left = np.nan_to_num(audio[0])
        right = np.nan_to_num(audio[1])
        # Guarded Pearson correlation — avoids NaN on silent/constant signals (§VERBOTEN: np.corrcoef)
        _lc = left - float(np.mean(left))
        _rc = right - float(np.mean(right))
        _nl = float(np.linalg.norm(_lc))
        _nr = float(np.linalg.norm(_rc))
        if _nl < 1e-12 or _nr < 1e-12:
            corr = 1.0
        else:
            corr = float(np.dot(_lc, _rc) / (_nl * _nr))
            corr = float(np.clip(corr, -1.0, 1.0))
        mono_problem = corr < self.min_corr
        if mono_problem:
            logger.warning(
                "[StereoCoherenceGuard] Low coherence (corr=%.3f < %.3f), mono issues possible.",
                corr,
                self.min_corr,
            )
        return {
            "stereo": True,
            "correlation": corr,
            "mono_problem": mono_problem,
            "min_corr_threshold": self.min_corr,
        }

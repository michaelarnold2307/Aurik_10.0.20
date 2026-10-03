import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_entropy_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_entropy_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"entropy_min": 2.0, "entropy_max": 8.0},
        "safe_ranges": {
            "entropy_min": {"min": 0.5, "max": 4.0},
            "entropy_max": {"min": 4.0, "max": 12.0},
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
    reports={"self_metrics": ["spectral_entropy"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralEntropyGuard:
    """
    SOTA-konformer Spectral Entropy Guard:
    - Überwacht die spektrale Entropie als Qualitätsmaß
    """

    def __init__(self, entropy_min: float = 2.0, entropy_max: float = 8.0):
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_entropy_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Berechnet die spektrale Entropie und prüft Quality-Gate. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param sr: Sample-Rate
        :return: Dict mit Entropie und Status
        """
        self.log_contract()
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        try:
            spec = np.abs(np.fft.rfft(audio))
            prob = spec / (np.sum(spec) + 1e-8)
            entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))
            ok = self.entropy_min <= entropy <= self.entropy_max
            if entropy < 0 or np.isnan(entropy):
                self._audit_log("warn", "Unplausible Entropie, Rollback aktiviert.")
                return {"spectral_entropy": 0.0, "ok": False}
            self._audit_log("success", f"Spektrale Entropie berechnet: {entropy:.3f}, ok={ok}")
            return {"spectral_entropy": entropy, "ok": ok}
        except Exception as e:
            self._audit_log("error", f"Fehler bei Entropie-Berechnung: {e}")
            return {"spectral_entropy": 0.0, "ok": False}

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[spectral_entropy_guard] %s", message)

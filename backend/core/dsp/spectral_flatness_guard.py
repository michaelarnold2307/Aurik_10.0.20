import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_flatness_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_flatness_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"flatness_min": 0.1, "flatness_max": 0.7},
        "safe_ranges": {
            "flatness_min": {"min": 0.01, "max": 0.5},
            "flatness_max": {"min": 0.3, "max": 0.99},
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
    reports={"self_metrics": ["spectral_flatness"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralFlatnessGuard:
    """
    SOTA-konformer Spectral Flatness Guard:
    - Überwacht spektrale Flachheit als Qualitätsmaß
    """

    def __init__(self, flatness_min: float = 0.1, flatness_max: float = 0.7):
        self.flatness_min = flatness_min
        self.flatness_max = flatness_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_flatness_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Berechnet die spektrale Flachheit und prüft Quality-Gate. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param sr: Sample-Rate
        :return: Dict mit Flatness und Status
        """
        self.log_contract()
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        try:
            spec = np.abs(np.fft.rfft(audio)) + 1e-8
            geo_mean = np.exp(np.mean(np.log(spec)))
            arith_mean = np.mean(spec)
            flatness = float(geo_mean / arith_mean)
            ok = self.flatness_min <= flatness <= self.flatness_max
            if flatness < 0 or np.isnan(flatness):
                self._audit_log("warn", "Unplausible Flatness, Rollback aktiviert.")
                return {"spectral_flatness": 0.0, "ok": False}
            self._audit_log("success", f"Spektrale Flatness berechnet: {flatness:.3f}, ok={ok}")
            return {"spectral_flatness": flatness, "ok": ok}
        except Exception as e:
            self._audit_log("error", f"Fehler bei Flatness-Berechnung: {e}")
            return {"spectral_flatness": 0.0, "ok": False}

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[spectral_flatness_guard] %s", message)

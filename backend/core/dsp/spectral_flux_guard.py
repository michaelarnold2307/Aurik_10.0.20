import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_flux_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_flux_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"flux_min": 0.01, "flux_max": 0.5},
        "safe_ranges": {
            "flux_min": {"min": 0.0, "max": 0.2},
            "flux_max": {"min": 0.2, "max": 1.0},
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
    reports={"self_metrics": ["spectral_flux"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralFluxGuard:
    """
    SOTA-konformer Spectral Flux Guard:
    - Überwacht den spektralen Fluss als Qualitätsmaß
    """

    def __init__(self, flux_min: float = 0.01, flux_max: float = 0.5):
        self.flux_min = flux_min
        self.flux_max = flux_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_flux_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Berechnet den spektralen Fluss und prüft Quality-Gate. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param sr: Sample-Rate
        :return: Dict mit Flux und Status
        """
        self.log_contract()
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        try:
            spec = np.abs(np.fft.rfft(audio))
            flux = float(np.mean(np.abs(np.diff(spec))))
            ok = self.flux_min <= flux <= self.flux_max
            if flux < 0 or np.isnan(flux):
                self._audit_log("warn", "Unplausibler Flux, Rollback aktiviert.")
                return {"spectral_flux": 0.0, "ok": False}
            self._audit_log("success", f"Spektraler Fluss berechnet: {flux:.3f}, ok={ok}")
            return {"spectral_flux": flux, "ok": ok}
        except Exception as e:
            self._audit_log("error", f"Fehler bei Flux-Berechnung: {e}")
            return {"spectral_flux": 0.0, "ok": False}

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[spectral_flux_guard] %s", message)

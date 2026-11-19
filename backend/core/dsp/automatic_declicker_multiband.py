"""
automatic_declicker_multiband.py — SOTA-Multiband-Declicker für Aurik.

Detektiert Klicks/Knackser bandweise (bessere Detektions-SNR bei
Bandpass-Bändern) und repariert anschließend das ORIGINAL-Vollband-
signal über :mod:`backend.core.dsp.declick_core` (AR-Interpolation).
Damit bleiben Pegel und Phasengang außerhalb der Defekte bit-identisch
(keine Summations-Artefakte der Filterbänke). Auditierbar,
rollback-fähig (Hörordnung Ebene 2).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from backend.core.dsp.declick_core import detect_click_mask, repair_clicks, strictness_from_threshold

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_declicker_multiband"
    category: str = "declicker_multiband"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
declicker_multiband_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"bands": 3, "threshold": 0.6}},
    budgets={"compute_cost": 0.05},
    side_effects=[
        {
            "risk": "Artefakte",
            "expected_when": "zu niedriger threshold",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["multiband_click_removal_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutomaticDeclickerMultiband:
    """Multiband-Declicker: Bandweise Detektion, Vollband-Reparatur."""

    def __init__(self, bands: int = 3, threshold: float = 0.6):
        self.bands = bands
        self.threshold = threshold

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(declicker_multiband_contract))

    def declick_multiband(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Bandweise Klick-Detektion, AR-Reparatur auf dem Originalsignal.

        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :return: De-clicktes Signal (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        import scipy.signal

        band_edges = [(20, 800), (800, 4000), (4000, sr // 2 - 1)]
        strictness_k = strictness_from_threshold(self.threshold)
        mask = detect_click_mask(audio, strictness_k=strictness_k)
        for low, high in band_edges[: self.bands]:
            sos = scipy.signal.butter(4, [low / (sr / 2), high / (sr / 2)], btype="band", output="sos")
            band = scipy.signal.sosfilt(sos, audio)
            mask |= detect_click_mask(band, strictness_k=strictness_k)
        return np.asarray(repair_clicks(audio, mask).astype(audio.dtype))  # type: ignore[no-any-return]

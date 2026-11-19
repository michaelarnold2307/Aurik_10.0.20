import logging

"""
automatic_harmonics.py - SOTA-konforme automatische Harmonischen-Erzeugung für Aurik 6.0

Dieses Modul erzeugt und optimiert Obertöne/Harmonische (klassische DSP, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_harmonics"
    category: str = "harmonics"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


automatic_harmonics_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"harmonic_gain": 0.5, "num_harmonics": 2}},
    budgets={"compute_cost": 0.01},
    side_effects=[{"risk": "Aliasing", "expected_when": "hohe Harmonics-Zahl", "severity": 0.1}],
    reports={"self_metrics": ["harmonic_energy"]},
    rollback={"strategy": "bypass", "supports_partial": True},
)


class AutomaticHarmonics:
    """
    Klassische automatische Harmonischen-Erzeugung (SOTA-Maximum):
    - Erzeugt und mischt Obertöne (2. und 3. Harmonische) zum Originalsignal
    """

    contract: DSPContract = automatic_harmonics_contract

    def __init__(self, harmonic_gain: float = 0.5, num_harmonics: int = 2):
        self.harmonic_gain = harmonic_gain
        self.num_harmonics = num_harmonics

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def generate_harmonics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Erzeugt und mischt Obertöne (2. und 3. Harmonische) zum Originalsignal.
        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :return: Signal mit Obertönen
        """
        self.log_contract()
        y = np.copy(audio)
        for n in range(2, 2 + self.num_harmonics):
            y += self.harmonic_gain * np.power(audio, n)
        return np.asarray(y)  # type: ignore[no-any-return]

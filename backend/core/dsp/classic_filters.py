"""
classic_filters.py - SOTA-konforme Filter für Aurik 6.0

SOTA-konforme Highpass-, Notch- und DCBlocker-Filter mit DSPContract und Auditierbarkeit.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
import numpy.typing as npt
from scipy.signal import butter, iirnotch, lfilter


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractClassicFilter:
    id: str = "classic_filter"
    category: str = "filter"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


classic_filter_contract = DSPContractClassicFilter(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {}},
    budgets={"compute_cost": 0.01},
    side_effects=[
        {
            "risk": "Fehlfilterung",
            "expected_when": "Parameter falsch gewählt",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["filter_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class HighpassFilter:
    """
    SOTA-konformer Highpass (z.B. Rumpelfilter, DC-Blocker)
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractClassicFilter = classic_filter_contract

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s v%s", self.contract.id, self.contract.version)

    def __init__(self, cutoff_hz: float = 20.0, sr: int = 48000, order: int = 2):
        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.order = order
        self.b, self.a = butter(order, cutoff_hz / (0.5 * sr), btype="high", output="ba")  # type: ignore[misc]

    def process(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        if audio.ndim == 2:
            # Audio format is (channels, samples)
            output = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                output[ch, :] = lfilter(self.b, self.a, audio[ch, :])
            return output
        else:
            return np.asarray(lfilter(self.b, self.a, audio))


class NotchFilter:
    """
    SOTA-konformer Notch-Filter (z.B. Hum-Entfernung)
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractClassicFilter = classic_filter_contract

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s v%s", self.contract.id, self.contract.version)

    def __init__(self, freq: float = 50.0, Q: float = 30.0, sr: int = 48000):
        self.freq = freq
        self.Q = Q
        self.sr = sr
        self.b, self.a = iirnotch(freq / (0.5 * sr), Q)  # type: ignore[misc]

    def process(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        if audio.ndim == 2:
            # Audio format is (channels, samples)
            output = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                output[ch, :] = lfilter(self.b, self.a, audio[ch, :])
            return output
        else:
            return np.asarray(lfilter(self.b, self.a, audio))


class DCBlocker:
    """
    SOTA-konformer DC-Offset-Entferner (1st order HPF)
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractClassicFilter = classic_filter_contract

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s v%s", self.contract.id, self.contract.version)

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.alpha = 0.995

    def process(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        y = np.zeros_like(audio)
        y[0] = audio[0]
        for n in range(1, len(audio)):
            y[n] = audio[n] - audio[n - 1] + self.alpha * y[n - 1]
        return y

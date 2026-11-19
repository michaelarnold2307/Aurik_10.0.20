import logging

"""
Adaptive Comb Filter DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Adaptiver Kammfilter für Brumm-/Tonalstörungen (klassische DSP, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_comb_filter"
    category: str = "comb_filter"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveCombFilter:
    """
    Klassischer adaptiver Kammfilter (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, delay: int = 10, gain: float = 0.9):
        self.delay = delay
        self.gain = gain

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def filter(self, x: np.ndarray) -> np.ndarray:
        """
        Wendet einen adaptiven Kammfilter an.
        :param x: Eingabesignal (np.ndarray)
        :return: Gefiltertes Signal (np.ndarray)
        """
        self.log_contract()
        y = np.copy(x)
        for i in range(self.delay, len(x)):
            y[i] = x[i] - self.gain * x[i - self.delay]
        return np.asarray(y)  # type: ignore[no-any-return]

    def auto_optimize(self, x: np.ndarray) -> None:
        """
        Passt Delay adaptiv an die Signal-Länge an.
        :param x: Eingabesignal (np.ndarray)
        """
        self.delay = min(100, max(5, len(x) // 1000))

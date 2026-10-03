import logging

"""
Adaptive Cross-Correlation DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten (klassische DSP, SOTA-Maximum).
Verwendet numpy für die Berechnung.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_crosscorrelation"
    category: str = "crosscorrelation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveCrossCorrelation:
    """
    Klassische adaptive Kreuzkorrelationsberechnung (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, normalize: bool = True, max_lag: int | None = None):
        self.normalize = normalize
        self.max_lag = max_lag

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def cross_correlation(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Berechnet die Kreuzkorrelation adaptiv mit aktuellen Parametern.
        :param x: Signal 1 (np.ndarray)
        :param y: Signal 2 (np.ndarray)
        :return: Kreuzkorrelationsfunktion (np.ndarray)
        """
        self.log_contract()
        max_lag = kwargs.get("max_lag", self.max_lag)
        result = np.correlate(x, y, mode="full")
        mid = len(result) // 2
        result = result[mid : mid + max_lag] if max_lag is not None else result[mid:]
        if self.normalize:
            result = result / np.max(np.abs(result))
        return np.asarray(result)  # type: ignore[no-any-return]

    def auto_optimize(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Automatische Anpassung der max_lag-Parameter je nach Signal.
        :param x: Signal 1 (np.ndarray)
        :param y: Signal 2 (np.ndarray)
        """
        self.max_lag = min(2048, min(len(x), len(y)) // 2)

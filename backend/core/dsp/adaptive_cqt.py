import logging

"""
Adaptive CQT DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten (klassische DSP, SOTA-Maximum).
Verwendet librosa für die Constant-Q-Transformation.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_cqt"
    category: str = "cqt"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveCQT:
    """
    Klassische adaptive Constant-Q-Transformation (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float | None = None,
        n_bins: int = 84,
        bins_per_octave: int = 12,
    ) -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin or librosa.note_to_hz("C1")
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def cqt(self, y: np.ndarray, sr: int | None = None, **kwargs: Any) -> np.ndarray:
        """
        Berechnet die CQT adaptiv mit aktuellen Parametern.
        :param y: Eingabesignal (np.ndarray)
        :param sr: Samplingrate (Optional[int])
        :return: CQT-Matrix (np.ndarray)
        """
        self.log_contract()
        return librosa.cqt(  # type: ignore[no-any-return]
            y=y,
            sr=sr or self.sr,
            hop_length=kwargs.get("hop_length", self.hop_length),
            fmin=kwargs.get("fmin", self.fmin),
            n_bins=kwargs.get("n_bins", self.n_bins),
            bins_per_octave=kwargs.get("bins_per_octave", self.bins_per_octave),
        )

    def auto_optimize(self, y: np.ndarray, sr: int) -> None:
        """
        Automatische Anpassung der CQT-Parameter je nach Signal (SOTA-Ansatz).
        :param y: Eingabesignal (np.ndarray)
        :param sr: Samplingrate (int)
        """
        # Beispiel: Passe n_bins und bins_per_octave an die Signal-Länge und Samplingrate an
        if len(y) < 4096 or sr < 16000:
            self.n_bins = 36
            self.bins_per_octave = 12
        elif len(y) < 16384 or sr < 32000:
            self.n_bins = 60
            self.bins_per_octave = 24
        else:
            self.n_bins = 84
            self.bins_per_octave = 36
        self.sr = sr

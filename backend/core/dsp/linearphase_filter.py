import numpy as np
import numpy.typing as npt
from scipy.signal import firwin, lfilter


class LinearPhaseHighpass:
    """
    SOTA-konformer Linearphase-Highpass (FIR, Mastering)
    """

    def __init__(self, cutoff_hz: float = 20.0, sr: int = 48000, numtaps: int = 513):
        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.numtaps = numtaps
        self.coeffs = firwin(numtaps, cutoff_hz / (0.5 * sr), pass_zero=False)

    def process(self, audio: npt.NDArray[np.float64], sr: int | None = None) -> npt.NDArray[np.float64]:
        # sr wird ignoriert, für Kompatibilität mit PolicyEngine
        return np.asarray(lfilter(self.coeffs, [1.0], audio))

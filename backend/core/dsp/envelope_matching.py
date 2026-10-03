import numpy as np
import numpy.typing as npt


class EnvelopeMatcher:
    """
    SOTA-konformer Envelope Matcher (klassisch)
    """

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def process(self, audio: npt.NDArray[np.float64], target_env: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # RMS-Hüllkurve
        env = np.sqrt(np.convolve(audio**2, np.ones(1024) / 1024, mode="same"))
        gain = (target_env + 1e-8) / (env + 1e-8)
        gain = np.clip(gain, 0.5, 2.0) ** self.strength
        return np.asarray(audio * gain)

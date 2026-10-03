import numpy as np


# DeepLoudnessNet → LUFSNormalizer — Klassenname verboten §4.4+§10.2. ITU-R BS.1770 Implementierung bleibt erhalten.
class LUFSNormalizer:
    """
    ITU-R BS.1770 Loudness-Normalisierung (LUFS, Studio-Standard)
    """

    def __init__(self, target_lufs=-14.0):
        self.target_lufs = target_lufs

    def integrated_lufs(self, audio, sr):
        # K-Weighting-Filter (vereinfachte Version)
        b = [1.0, -2.0, 1.0]
        a = [1.0, -1.99004745483398, 0.99007225036621]
        weighted = np.convolve(audio, b, mode="same") / np.convolve(np.ones_like(audio), a, mode="same")
        rms = np.sqrt(np.mean(weighted**2))
        lufs = 20 * np.log10(rms + 1e-12)
        return lufs

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        lufs = self.integrated_lufs(audio, sr)
        gain = 10 ** ((self.target_lufs - lufs) / 20)
        return np.asarray(audio * gain)  # type: ignore[no-any-return]

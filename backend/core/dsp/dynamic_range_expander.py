import numpy as np
import numpy.typing as npt


class DynamicRangeExpander:
    """
    SOTA-konformer Dynamic Range Expander:
    - RMS/Peak-Detection, Soft-Knee, Ratio, Attack/Release, ML-ready
    """

    def __init__(
        self,
        threshold_db: float = -40.0,
        ratio: float = 0.5,
        knee_db: float = 6.0,
        attack_ms: float = 10.0,
        release_ms: float = 80.0,
    ) -> None:
        """
        threshold_db: Expander-Schwelle (dB)
        ratio: Expansionsrate (<1)
        knee_db: Soft-Knee (dB)
        attack_ms: Attack-Zeit (ms)
        release_ms: Release-Zeit (ms)
        """
        self.threshold_db = threshold_db
        self.ratio = max(float(ratio), 1e-6)
        self.knee_db = knee_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    @staticmethod
    def _moving_rms(audio: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
        window = max(1, int(window))
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 1:
            sq = np.square(x)
            left = window // 2
            right = window - left - 1
            padded = np.pad(sq, (left, right), mode="edge")
            csum = np.cumsum(np.concatenate(([0.0], padded)))
            avg = (csum[window:] - csum[:-window]) / float(window)
            return np.asarray(np.sqrt(np.maximum(avg, 0.0)), dtype=np.float64)
        ch_results = [DynamicRangeExpander._moving_rms(ch, window) for ch in x]
        return np.asarray(ch_results, dtype=np.float64)

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit Dynamikexpansion.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: expandiertes Signal (gleicher Typ wie audio)
        """
        orig_dtype = audio.dtype
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo/multi-channel: process each channel independently
        if x.ndim == 2:
            return np.stack(
                [self.process(np.asarray(x[c], dtype=orig_dtype), sr) for c in range(x.shape[0])],
                axis=0,
            ).astype(orig_dtype)

        # RMS-Detection
        window = int(sr * 0.01)
        rms = self._moving_rms(x, window)
        rms = np.nan_to_num(rms, nan=1e-8, posinf=1e-8, neginf=1e-8)
        rms_db = 20 * np.log10(rms + 1e-8)
        under = self.threshold_db - rms_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (under > -self.knee_db / 2) & (under < self.knee_db / 2)
        gain_db[idx_soft] = (1 / self.ratio - 1) * ((under[idx_soft] + self.knee_db / 2) ** 2) / (2 * self.knee_db)
        idx_under = under >= self.knee_db / 2
        gain_db[idx_under] = (1 / self.ratio - 1) * (under[idx_under])
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        attack_coeff = np.exp(-1.0 / (sr * self.attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * self.release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = x * env
        # Pegel normalisieren
        maxval: float = float(np.max(np.abs(out)))
        if maxval > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return np.asarray(np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), dtype=orig_dtype)

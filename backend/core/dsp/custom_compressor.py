import numpy as np
import numpy.typing as npt


class CustomCompressor:
    """
    SOTA-konformer Custom Compressor:
    - Modular, frei konfigurierbar, Soft-Knee, Sidechain, ML-ready
    """

    def __init__(
        self,
        threshold_db: float = -24.0,
        ratio: float = 2.0,
        knee_db: float = 6.0,
        attack_ms: float = 10.0,
        release_ms: float = 80.0,
        sidechain: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        threshold_db: Kompressor-Schwelle (dB)
        ratio: Kompressionsrate
        knee_db: Soft-Knee (dB)
        attack_ms: Attack-Zeit (ms)
        release_ms: Release-Zeit (ms)
        sidechain: Optionales Sidechain-Signal
        """
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_db = knee_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.sidechain = sidechain

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit frei konfigurierbarer Kompression.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: komprimiertes Signal (gleicher Typ wie audio)
        """
        # RMS-Detection (Sidechain oder Audio)
        sc = self.sidechain if self.sidechain is not None else audio
        window = int(sr * 0.01)
        rms = np.sqrt(np.convolve(sc**2, np.ones(window) / window, mode="same"))
        rms_db = 20 * np.log10(rms + 1e-8)
        over = rms_db - self.threshold_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (over > -self.knee_db / 2) & (over < self.knee_db / 2)
        gain_db[idx_soft] = (1 / self.ratio - 1) * ((over[idx_soft] + self.knee_db / 2) ** 2) / (2 * self.knee_db)
        idx_over = over >= self.knee_db / 2
        gain_db[idx_over] = (1 / self.ratio - 1) * (over[idx_over])
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        attack_coeff = np.exp(-1.0 / (sr * self.attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * self.release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        # env auf Länge von audio trimmen/padden
        if len(env) > len(audio):
            env = env[: len(audio)]
            audio = audio[: len(env)]
        elif len(env) < len(audio):
            pad = np.ones(len(audio), dtype=env.dtype)
            pad[: len(env)] = env
            env = pad
        out = audio * env
        # Pegel normalisieren
        maxval: float = float(np.max(np.abs(out)))
        if maxval > 1.0:
            out = np.clip(out, -1.0, 1.0)
        # Output-Länge exakt auf Input trimmen/padden
        if len(out) > len(audio):
            out = out[: len(audio)]
        elif len(out) < len(audio):
            pad = np.zeros(len(audio), dtype=out.dtype)
            pad[: len(out)] = out
            out = pad
        return np.asarray(out.astype(audio.dtype))

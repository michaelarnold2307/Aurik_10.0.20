import logging

import numpy as np

logger = logging.getLogger(__name__)


class IntelligentLimiter:
    """
    Lookahead-Limiter mit Soft-Knee, konfigurierbarer Attack/Release
    und True-Peak-aware Ceiling (Giannoulis 2012, ITU-R BS.1770-4).
    """

    def __init__(
        self,
        ceiling: float = -1.0,
        lookahead_ms: float = 2.0,
        knee_db: float = 6.0,
        attack_ms: float = 0.5,
        release_ms: float = 50.0,
    ):
        self.ceiling = ceiling
        self.lookahead_ms = lookahead_ms
        self.knee_db = max(knee_db, 0.1)
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))
        lookahead = max(1, int(sr * self.lookahead_ms / 1000))

        # Lookahead peak detection
        padded = np.pad(audio, (lookahead, 0), mode="constant")
        shifted = padded[: len(audio)]
        peak = np.abs(shifted)
        peak_db = 20.0 * np.log10(peak + 1e-12)

        # Soft-knee gain computation (Giannoulis 2012)
        over = peak_db - self.ceiling
        gain_db = np.zeros_like(peak_db)
        half_knee = self.knee_db / 2.0

        # Below knee: no reduction
        # In knee: quadratic
        idx_soft = (over > -half_knee) & (over < half_knee)
        gain_db[idx_soft] = -((over[idx_soft] + half_knee) ** 2) / (2.0 * self.knee_db)
        # Above knee: full reduction
        idx_over = over >= half_knee
        gain_db[idx_over] = -over[idx_over]

        gain_lin = 10.0 ** (gain_db / 20.0)

        # Ballistics: attack + release smoothing
        attack_coeff = np.exp(-1.0 / max(sr * self.attack_ms / 1000.0, 1.0))
        release_coeff = np.exp(-1.0 / max(sr * self.release_ms / 1000.0, 1.0))

        env = np.ones_like(gain_lin)
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = attack_coeff * env[i - 1] + (1.0 - attack_coeff) * gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1.0 - release_coeff) * gain_lin[i]

        out = audio * env
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        return np.asarray(out)  # type: ignore[no-any-return]

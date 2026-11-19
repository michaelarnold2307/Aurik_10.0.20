"""§v10.757 (2026-09-09): Modulationsdomänen-De-Essing.

Sibilanz ist breitbandige Energie im 4–8-kHz-Band mit Amplitudenmodulation
im Frikativ-Rhythmus (12–28 Hz) — Becken/Hi-Hats sind dagegen breitbandige
Transienten ohne diese Modulation. Der Detektor misst den AM-Anteil der
Band-Hüllkurve und dämpft nur, wenn das Modulationsmuster sibilant ist.

Deterministisch, vektorisiert, kein ML, §G5 (copilot-instructions.md).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d  # pylint: disable=import-outside-toplevel
from scipy.signal import butter, sosfiltfilt  # pylint: disable=import-outside-toplevel


def modulation_deess(
    audio: np.ndarray,
    sr: int,
    *,
    band=(4000.0, 8000.0),
    am_rate=(12.0, 28.0),
    reduction_db: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gibt (de-essed Audio, Sibilanz-Score 0..1 je Sample) zurück."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    nyq = sr / 2.0

    # Bandpass 4–8 kHz
    sos_bp = butter(4, [band[0] / nyq, min(band[1] / nyq, 0.99)], btype="band", output="sos")
    band_sig = sosfiltfilt(sos_bp, audio)
    env = np.abs(band_sig)
    env_smooth = uniform_filter1d(env.astype(np.float64), size=int(0.002 * sr), mode="nearest")

    # AM-Band 12–28 Hz der normierten Hüllkurve
    env_norm = env_smooth / (env_smooth.mean() + 1e-9)
    sos_am = butter(2, [am_rate[0] / nyq, am_rate[1] / nyq], btype="band", output="sos")
    am = sosfiltfilt(sos_am, env_norm - 1.0)
    am_power = uniform_filter1d(am.astype(np.float64) ** 2, size=int(0.01 * sr), mode="nearest")

    # Sibilanz-Score: normalisierte AM-Power
    score = np.clip(am_power / (am_power.max() + 1e-9), 0.0, 1.0)
    # Nur dämpfen, wenn genug Bandenergie vorhanden ist (kein Rauschen de-essen)
    band_active = uniform_filter1d(env_smooth, size=int(0.02 * sr), mode="nearest") > 1e-3
    score = score * band_active

    gain = 1.0 - score * (1.0 - 10 ** (-reduction_db / 20))
    # Band-Signal dämpfen und zurückmischen
    corrected = audio - band_sig * (1.0 - gain)
    return corrected.astype(np.float32), score.astype(np.float32)

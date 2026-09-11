from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import safe_to_mono
from backend.core.consonant_enhancement import measure_fricative_snr


@dataclass(frozen=True)
class DeEsserIntelligibilityReport:
    """Leichtgewichtiges intelligibility summary for de-essing decisions."""

    presence_ratio: float
    articulation_ratio: float
    air_ratio: float
    fricative_snr_delta_db: float
    intelligibility_score: float
    intelligibility_loss: float
    should_protect: bool


def _band_rms(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    mono = np.nan_to_num(np.asarray(mono, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if mono.size < 16:
        return 0.0

    nyquist = sr / 2.0
    high_hz = float(min(high_hz, nyquist * 0.98))
    low_hz = float(min(max(low_hz, 20.0), high_hz * 0.9))
    sos = sig.butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    try:
        band = sig.sosfiltfilt(sos, mono)
    except ValueError:
        # Degradiertes Ultra-Kurzsignal (< padlen): kausaler Fallback für die
        # RMS-Bandmessung — kein Audio-Ausgang, Phase unerheblich.
        band = sig.sosfilt(sos, mono)  # H-SCAN-EXEMPT: sosfilt (Mess-Fallback < padlen)
    return float(np.sqrt(np.mean(band**2) + 1e-12))


def _safe_ratio(after: float, before: float) -> float:
    if before <= 1e-9:
        return 1.0
    return float(np.clip(after / before, 0.0, 1.25))


def assess_deesser_intelligibility_preservation(
    before: np.ndarray,
    after: np.ndarray,
    sr: int,
    voice_gender: str = "unknown",
) -> DeEsserIntelligibilityReport:
    """Assess whether de-essing preserved vocal intelligibility instead of only reducing HF.

    The score intentionally weights articulation-heavy presence bands higher than airy
    top-octave energy so the guard follows intelligibility, not blunt brightness loss.
    """

    before_arr = np.nan_to_num(np.asarray(before, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    after_arr = np.nan_to_num(np.asarray(after, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    presence_before = _band_rms(before_arr, sr, 2000.0, 5000.0)
    presence_after = _band_rms(after_arr, sr, 2000.0, 5000.0)
    articulation_before = _band_rms(before_arr, sr, 4000.0, 8000.0)
    articulation_after = _band_rms(after_arr, sr, 4000.0, 8000.0)
    air_before = _band_rms(before_arr, sr, 8000.0, 12000.0)
    air_after = _band_rms(after_arr, sr, 8000.0, 12000.0)

    presence_ratio = _safe_ratio(presence_after, presence_before)
    articulation_ratio = _safe_ratio(articulation_after, articulation_before)
    air_ratio = _safe_ratio(air_after, air_before)

    fricative_snr_before = float(measure_fricative_snr(before_arr, sr, voice_gender))
    fricative_snr_after = float(measure_fricative_snr(after_arr, sr, voice_gender))
    fricative_snr_delta_db = fricative_snr_after - fricative_snr_before

    intelligibility_score = float(
        np.clip(0.60 * presence_ratio + 0.30 * articulation_ratio + 0.10 * air_ratio, 0.0, 1.10)
    )
    intelligibility_loss = float(max(0.0, 1.0 - min(intelligibility_score, 1.0)))
    should_protect = bool(presence_ratio < 0.88 or articulation_ratio < 0.82 or intelligibility_score < 0.86)

    return DeEsserIntelligibilityReport(
        presence_ratio=presence_ratio,
        articulation_ratio=articulation_ratio,
        air_ratio=air_ratio,
        fricative_snr_delta_db=fricative_snr_delta_db,
        intelligibility_score=intelligibility_score,
        intelligibility_loss=intelligibility_loss,
        should_protect=should_protect,
    )

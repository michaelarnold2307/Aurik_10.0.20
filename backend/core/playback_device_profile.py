"""Playback-device profiles for consumer-aware export optimisation.

A human mastering engineer performs a "translation check" on consumer devices
(headphones, laptop speakers, smartphones) at the end of a session and adjusts
the final mix accordingly. This module models the typical frequency-response
deviations of common consumer devices and provides inverse correction curves
for the export stage.

Usage:
    profile = get_playback_device_profile("consumer_headphone_avg")
    audio_out = apply_translation_eq(audio, sr, profile)

Spec: §PDV-1 Playback-Device-Awareness (v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device curves (frequency Hz → deviation in dB from flat reference)
# All curves are measurement-based (public measurements, direction: device vs. ideal).
# Inverse curve = -1 × these values.
# ---------------------------------------------------------------------------

# Format: list of (freq_hz, delta_db) breakpoints
# Interpolation: log-linear between breakpoints; outside range → last value

_DEVICE_CURVES: dict[str, list[tuple[float, float]]] = {
    # --- Headphones ---
    # Consumer average: typically bass-boosted, presence dip 2–4 kHz, air peak ~10 kHz
    "consumer_headphone_avg": [
        (20.0, +5.0),
        (60.0, +4.0),
        (100.0, +2.5),
        (200.0, +1.0),
        (500.0, 0.0),
        (1000.0, 0.0),
        (2000.0, -1.5),
        (3500.0, -2.5),  # Presence-Dip
        (6000.0, -1.0),
        (10000.0, +1.5),  # Air-Peak
        (16000.0, -1.0),
        (20000.0, -3.0),
    ],
    # AirPods (Gen2/3): V-shaped curve, strong sub-bass, high-frequency peak ~8 kHz
    "airpods": [
        (20.0, +7.0),
        (80.0, +5.0),
        (150.0, +2.0),
        (500.0, 0.0),
        (1000.0, 0.0),
        (2000.0, -2.0),
        (4000.0, -1.5),
        (8000.0, +3.0),  # high-frequency peak
        (12000.0, +1.0),
        (16000.0, -2.0),
        (20000.0, -5.0),
    ],
    # Studio headphone (reference class): nearly flat, slight high-frequency roll-off
    "studio_headphone": [
        (20.0, +1.0),
        (100.0, +0.5),
        (500.0, 0.0),
        (1000.0, 0.0),
        (5000.0, 0.0),
        (10000.0, -0.5),
        (16000.0, -1.5),
        (20000.0, -2.5),
    ],
    # --- Speakers ---
    # Laptop speaker: almost no bass, peak at 1–3 kHz (membrane resonance)
    "laptop_speaker": [
        (20.0, -12.0),
        (60.0, -10.0),
        (100.0, -6.0),
        (200.0, -2.0),
        (500.0, 0.0),
        (1000.0, +2.0),
        (2000.0, +3.0),  # membrane resonance peak
        (4000.0, +1.0),
        (8000.0, -1.0),
        (16000.0, -3.0),
        (20000.0, -8.0),
    ],
    # Smartphone speaker (small mono): even less bass, midrange emphasis
    "smartphone_speaker": [
        (20.0, -15.0),
        (80.0, -10.0),
        (150.0, -5.0),
        (300.0, -1.0),
        (800.0, +1.5),
        (1500.0, +2.5),
        (3000.0, +2.0),
        (6000.0, +0.0),
        (12000.0, -2.0),
        (20000.0, -6.0),
    ],
    # Studio monitor (flat reference): no correction needed
    "studio_monitor": [
        (20.0, 0.0),
        (20000.0, 0.0),
    ],
    # Hi-fi home speaker: slight bass boost, slight high-frequency roll-off
    "hifi_speaker": [
        (20.0, +2.0),
        (60.0, +2.5),
        (150.0, +1.0),
        (500.0, 0.0),
        (2000.0, 0.0),
        (5000.0, -0.5),
        (10000.0, -1.5),
        (16000.0, -3.0),
        (20000.0, -5.0),
    ],
    # ── §v10 Car Audio & Bluetooth Profiles ──
    "car_sedan_avg": [
        (60.0, +6.0),
        (80.0, +5.0),
        (100.0, +3.0),
        (200.0, +1.0),
        (500.0, 0.0),
        (1000.0, 0.0),
        (2000.0, -1.0),
        (3000.0, -3.0),
        (5000.0, -4.0),
        (8000.0, -2.0),
        (10000.0, -3.0),
        (15000.0, -6.0),
        (20000.0, -8.0),
    ],
    "car_suv_avg": [
        (60.0, +8.0),
        (80.0, +6.0),
        (100.0, +4.0),
        (200.0, +1.5),
        (500.0, 0.0),
        (1000.0, 0.0),
        (2000.0, -0.5),
        (3000.0, -2.0),
        (5000.0, -3.0),
        (8000.0, -3.0),
        (10000.0, -4.0),
        (15000.0, -8.0),
        (20000.0, -10.0),
    ],
    "bluetooth_speaker_avg": [
        (80.0, -6.0),
        (100.0, -4.0),
        (150.0, -2.0),
        (200.0, -1.0),
        (500.0, 0.0),
        (1000.0, +1.0),
        (2000.0, +2.0),
        (3000.0, +2.0),
        (5000.0, +1.0),
        (8000.0, -2.0),
        (10000.0, -5.0),
        (15000.0, -12.0),
        (20000.0, -15.0),
    ],
    "club_pa_system": [
        (40.0, +3.0),
        (60.0, +4.0),
        (80.0, +3.0),
        (100.0, +2.0),
        (200.0, +1.0),
        (500.0, 0.0),
        (1000.0, 0.0),
        (2000.0, 0.0),
        (3000.0, +0.5),
        (5000.0, +1.0),
        (8000.0, +0.5),
        (10000.0, 0.0),
        (15000.0, -2.0),
        (20000.0, -4.0),
    ],
}

# Alias mapping: alternative device identifiers
_DEVICE_ALIASES: dict[str, str] = {
    "headphone": "consumer_headphone_avg",
    "kopfhoerer": "consumer_headphone_avg",
    "kopfhörer": "consumer_headphone_avg",
    "beats": "consumer_headphone_avg",
    "airpod": "airpods",
    "apple_airpods": "airpods",
    "laptop": "laptop_speaker",
    "smartphone": "smartphone_speaker",
    "mobile": "smartphone_speaker",
    "handy": "smartphone_speaker",
    "studio": "studio_monitor",
    "monitor": "studio_monitor",
    "hifi": "hifi_speaker",
    "hi_fi": "hifi_speaker",
    "default": "consumer_headphone_avg",
    "car": "car_sedan_avg",
    "auto": "car_sedan_avg",
    "suv": "car_suv_avg",
    "bluetooth": "bluetooth_speaker_avg",
    "bt_speaker": "bluetooth_speaker_avg",
    "club": "club_pa_system",
    "pa": "club_pa_system",
}

# Maximum correction strength per band (prevents overly aggressive intervention)
_MAX_CORRECTION_DB = 4.0


@dataclass
class PlaybackDeviceProfile:
    """Device profile with frequency-response deviation curve."""

    device_id: str
    display_name: str
    curve_points: list[tuple[float, float]]  # (freq_hz, delta_db)
    max_correction_db: float = _MAX_CORRECTION_DB
    notes: str = ""

    def get_inverse_curve(self) -> list[tuple[float, float]]:
        """Gibt the inverse curve (for translation EQ application) zurück."""
        return [(f, -d) for f, d in self.curve_points]


def _interpolate_db(freq_hz: float, curve: list[tuple[float, float]]) -> float:
    """Log-linear interpolation of the dB deviation at a given frequency."""
    if not curve:
        return 0.0
    if freq_hz <= curve[0][0]:
        return float(curve[0][1])
    if freq_hz >= curve[-1][0]:
        return float(curve[-1][1])
    for i in range(len(curve) - 1):
        f1, d1 = curve[i]
        f2, d2 = curve[i + 1]
        if f1 <= freq_hz <= f2:
            # Log interpolation
            t = (np.log10(freq_hz) - np.log10(f1)) / (np.log10(f2) - np.log10(f1) + 1e-12)
            return float(d1 + t * (d2 - d1))
    return 0.0


def _build_correction_filter(
    curve: list[tuple[float, float]],
    sr: int,
    n_fft: int,
    max_db: float = _MAX_CORRECTION_DB,
) -> np.ndarray:
    """Erstellt an FFT gain vector (n_fft//2+1,) for the translation EQ.

    Negative dB = attenuation; positive dB = boost.
    max_db clamps the correction range.
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    gains = np.zeros(len(freqs), dtype=np.float32)
    for i, f in enumerate(freqs):
        raw_db = _interpolate_db(float(f), curve)
        clamped_db = float(np.clip(raw_db, -max_db, max_db))
        gains[i] = 10.0 ** (clamped_db / 20.0)
    return gains  # type: ignore[no-any-return]


def get_playback_device_profile(device_id: str) -> PlaybackDeviceProfile:
    """Gibt the device profile for the given ID zurück.

    Unknown IDs fall back to `consumer_headphone_avg` as a safe default.
    """
    resolved = _DEVICE_ALIASES.get(device_id.lower(), device_id.lower())
    resolved = _DEVICE_ALIASES.get(resolved, resolved)  # double-resolve aliases
    curve = _DEVICE_CURVES.get(resolved)
    if curve is None:
        logger.debug(
            "PlaybackDevice '%s' unknown → falling back to consumer_headphone_avg",
            device_id,
        )
        resolved = "consumer_headphone_avg"
        curve = _DEVICE_CURVES[resolved]
    return PlaybackDeviceProfile(
        device_id=resolved,
        display_name=resolved.replace("_", " ").title(),
        curve_points=curve,
    )


def apply_translation_eq(
    audio: np.ndarray,
    sr: int,
    profile: PlaybackDeviceProfile,
    *,
    strength: float = 0.5,
) -> np.ndarray:
    """Wendet an: the inverse translation EQ to the audio signal.

    The effect is intentionally gentle (strength ≤ 0.5 default):
    the goal is not a "headphone mix", but a mix that sounds *balanced*
    on consumer devices.

    Args:
        audio:    float32 ndarray (N,) mono or (2, N) stereo
        sr:       sample rate (48 000 Hz expected)
        profile:  PlaybackDeviceProfile
        strength: 0.0 = no intervention; 1.0 = full inverse EQ (max_correction_db)

    Returns:
        float32 ndarray of same shape, ∈ [−1, 1]
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return audio

    audio = np.asarray(audio, dtype=np.float32)
    orig_ndim = audio.ndim
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # (1, N)

    n_channels, n_samples = audio.shape
    if n_samples < 512:
        return audio.squeeze(0) if orig_ndim == 1 else audio

    n_fft = min(4096, 2 ** int(np.ceil(np.log2(n_samples))))
    inverse_curve = profile.get_inverse_curve()
    # Skaliere inverse Kurve mit strength
    scaled_curve = [(f, d * strength) for f, d in inverse_curve]
    gain_vec = _build_correction_filter(
        scaled_curve,
        sr,
        n_fft,
        max_db=profile.max_correction_db * strength,
    )

    audio_out = np.zeros_like(audio)
    hop = n_fft // 2
    for ch in range(n_channels):
        sig = audio[ch]
        # Overlap-Add FFT-EQ (short segments for temporal resolution)
        out = np.zeros(n_samples, dtype=np.float32)
        window = np.hanning(n_fft).astype(np.float32)
        norm = np.zeros(n_samples, dtype=np.float32)
        pos = 0
        while pos < n_samples:
            end = min(pos + n_fft, n_samples)
            seg = np.zeros(n_fft, dtype=np.float32)
            seg[: end - pos] = sig[pos:end]
            seg *= window
            spec = np.fft.rfft(seg)
            spec *= gain_vec
            seg_out = np.fft.irfft(spec).astype(np.float32)
            chunk_len = end - pos  # actual number of samples in this chunk
            out[pos:end] += seg_out[:chunk_len] * window[:chunk_len]
            norm[pos:end] += window[:chunk_len] ** 2
            pos += hop
        norm = np.where(norm > 1e-8, norm, 1.0)
        audio_out[ch] = out / norm

    audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
    audio_out = np.clip(audio_out, -1.0, 1.0)

    if orig_ndim == 1:
        return audio_out[0]  # type: ignore[no-any-return]
    return audio_out  # type: ignore[no-any-return]


def list_device_ids() -> list[str]:
    """Gibt all known device IDs (excluding aliases) zurück."""
    return sorted(_DEVICE_CURVES.keys())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_instance: dict[str, PlaybackDeviceProfile] | None = None
_lock = threading.Lock()


def get_cached_profile(device_id: str) -> PlaybackDeviceProfile:
    """Thread-safe singleton cache for device profiles."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = {}
    if device_id not in _instance:
        with _lock:
            if device_id not in _instance:
                _instance[device_id] = get_playback_device_profile(device_id)
    return _instance[device_id]

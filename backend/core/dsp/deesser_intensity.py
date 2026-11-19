"""Gemeinsame Intensitaets-Schaetzung fuer De-Esser-Phasen.

Leitet aus Defekt-Schwere, Frikativ-Praesenz und spektraler Sibilantenform
einen robusten Intensitaets-Profilektor ab, damit Phase 19 und Phase 43
dieselbe aggressivitaetsadaptive Logik verwenden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeEsserIntensityProfile:
    """Abgeleitete Intensitaetsparameter fuer De-Esser-Steuerung."""

    sibilance_pressure: float
    sibilance_ratio: float
    fricative_drive: float
    affricate_drive: float
    phoneme_drive: float
    breathiness: float
    intensity: float
    control_strength: float
    threshold_db_delta: float
    ratio_multiplier: float
    threshold_ratio_scale: float
    strength_cap: float
    reduction_mix: float


def extract_sibilance_pressure(defect_scores: object) -> float:
    """Liest Sibilance/Harshness severity robust aus heterogenen defect_scores."""
    if not isinstance(defect_scores, dict) or not defect_scores:
        return 0.0

    target_keys = {
        "sibilance",
        "sibilance_excess",
        "vocal_harshness",
    }
    max_pressure = 0.0

    for key, val in defect_scores.items():
        key_name = str(getattr(key, "value", key) or "").strip().lower()
        if key_name not in target_keys:
            continue
        sev_val = float(getattr(val, "severity", val) or 0.0)
        max_pressure = max(max_pressure, sev_val)

    return float(np.clip(max_pressure, 0.0, 1.0))


def _as_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr  # type: ignore[no-any-return]
    if arr.ndim == 2:
        if arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
            return np.asarray(np.mean(arr, axis=0, dtype=np.float32), dtype=np.float32)  # type: ignore[no-any-return]
        return np.asarray(np.mean(arr, axis=1, dtype=np.float32), dtype=np.float32)  # type: ignore[no-any-return]
    return np.ravel(arr).astype(np.float32)  # type: ignore[no-any-return]


def _analyze_sibilance_shape(audio: np.ndarray, sr: int, freq_low: float, freq_high: float) -> tuple[float, float]:
    """Gibt (sibilance_ratio, affricate_drive) aus FFT-Energieverteilung zurueck."""
    mono = _as_mono(audio)
    if mono.size < 512:
        return 0.0, 0.0

    n_fft = int(min(mono.size, 8192))
    start = max(0, (mono.size - n_fft) // 2)
    frame = mono[start : start + n_fft]
    if frame.size < n_fft:
        frame = np.pad(frame, (0, n_fft - frame.size))
    frame = frame * np.hanning(n_fft).astype(np.float32)

    spec = np.abs(np.fft.rfft(frame)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    total_energy = float(np.sum(spec)) + 1e-12

    band_low = float(max(3500.0, min(freq_low, freq_high * 0.92)))
    band_high = float(min(freq_high, sr * 0.5 * 0.98))
    if band_high <= band_low + 200.0:
        band_high = min(sr * 0.5 * 0.98, band_low + 3000.0)

    band_mask = (freqs >= band_low) & (freqs <= band_high)
    if not np.any(band_mask):
        return 0.0, 0.0

    sib_energy = float(np.sum(spec[band_mask]))
    if sib_energy <= 1e-12:
        return 0.0, 0.0

    third = (band_high - band_low) / 3.0
    low_mask = (freqs >= band_low) & (freqs < band_low + third)
    mid_mask = (freqs >= band_low + third) & (freqs < band_low + 2.0 * third)
    low_share = float(np.sum(spec[low_mask])) / sib_energy if np.any(low_mask) else 0.0
    mid_share = float(np.sum(spec[mid_mask])) / sib_energy if np.any(mid_mask) else 0.0
    centroid = float(np.sum(freqs[band_mask] * spec[band_mask]) / sib_energy)
    centroid_norm = float(np.clip((centroid - band_low) / max(band_high - band_low, 1.0), 0.0, 1.0))

    # Affrikate/T-Praesenz: staerkeres Gewicht auf Mid-Band und niedrigeren Zentroid.
    affricate_drive = float(np.clip(0.60 * mid_share + 0.25 * low_share + 0.15 * (1.0 - centroid_norm), 0.0, 1.0))
    sibilance_ratio = float(np.clip(sib_energy / total_energy, 0.0, 1.0))
    return sibilance_ratio, affricate_drive


def _extract_phoneme_drive(phoneme_timeline: object, language_hint: str | None) -> tuple[float, float, float]:
    """Liest frikativ-/plosivlastige Sprachhinweise aus optionaler PhonemeTimeline.

    Return: (phoneme_drive, german_language_boost, explicit_sibilant_boost)
    """
    language = str(getattr(phoneme_timeline, "language", language_hint) or language_hint or "").strip().lower()
    german_language_boost = 0.0
    if language.startswith("de"):
        german_language_boost = 0.18

    timeline_items = getattr(phoneme_timeline, "words", None)
    item_kind = "phoneme_type"
    if not isinstance(timeline_items, list) or not timeline_items:
        timeline_items = getattr(phoneme_timeline, "segments", None)
        item_kind = "phoneme_class"

    if not isinstance(timeline_items, list) or not timeline_items:
        return 0.0, german_language_boost, 0.0

    total = 0.0
    driven = 0.0
    explicit_sibilant_weight = 0.0
    for item in timeline_items:
        phoneme_label = str(getattr(item, item_kind, "mixed") or "mixed").strip().lower()
        phoneme_ipa = str(getattr(item, "phoneme_ipa", "") or "").strip().lower()
        is_stressed = bool(getattr(item, "is_stressed", False))
        weight = 1.0 + (0.25 if is_stressed else 0.0)
        total += weight

        if phoneme_label == "sibilant" or phoneme_ipa in {"s", "z", "ʃ", "ʒ", "ts", "dz", "tʃ", "dʒ"}:
            driven += 1.20 * weight
            explicit_sibilant_weight += weight
        elif phoneme_label == "fricative_stressed":
            driven += 1.05 * weight
        elif "fricative" in phoneme_label:
            driven += 1.00 * weight
        elif phoneme_label == "plosive":
            driven += 0.70 * weight
        elif phoneme_label == "mixed":
            driven += 0.35 * weight

    if total <= 1e-9:
        return 0.0, german_language_boost, 0.0
    explicit_sibilant_boost = float(np.clip(0.12 * (explicit_sibilant_weight / total), 0.0, 0.12))
    return float(np.clip(driven / total, 0.0, 1.0)), german_language_boost, explicit_sibilant_boost


def compute_optimal_deesser_intensity(
    audio: np.ndarray,
    sr: int,
    effective_strength: float,
    defect_scores: object = None,
    fricative_snr_db: float | None = None,
    breathiness: float = 0.0,
    freq_low: float = 5000.0,
    freq_high: float = 11000.0,
    min_strength_cap: float = 0.25,
    language_hint: str | None = None,
    phoneme_timeline: object = None,
) -> DeEsserIntensityProfile:
    """Berechnet ein gemeinsames Intensitaetsprofil fuer De-Esser-Phasen.

    Ziel: starke Sibilanz und T-/CH-artige Frikative sicherer treffen, ohne
    hauchige Stimmen oder natuerliche Artikulation unnoetig zu entstellen.
    """
    eff = float(np.clip(effective_strength, 0.0, 1.0))
    pressure = extract_sibilance_pressure(defect_scores)
    sibilance_ratio, affricate_drive = _analyze_sibilance_shape(audio, sr, freq_low, freq_high)
    (
        phoneme_drive,
        german_language_boost,
        explicit_sibilant_boost,
    ) = _extract_phoneme_drive(phoneme_timeline, language_hint)
    ratio_drive = float(np.clip((sibilance_ratio - 0.03) / 0.22, 0.0, 1.0))

    if fricative_snr_db is None or not np.isfinite(fricative_snr_db):
        fricative_drive = ratio_drive
    else:
        fricative_drive = float(np.clip((float(fricative_snr_db) + 2.0) / 18.0, 0.0, 1.0))

    driver = float(
        np.clip(
            0.38 * pressure
            + 0.20 * ratio_drive
            + 0.18 * fricative_drive
            + 0.14 * affricate_drive
            + 0.10 * phoneme_drive
            + german_language_boost
            + explicit_sibilant_boost,
            0.0,
            1.0,
        )
    )
    intensity = float(
        np.clip(
            max(eff, 0.30 * eff + 0.95 * driver) * (1.0 - 0.30 * np.clip(breathiness, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    if pressure >= 0.55:
        intensity = max(
            intensity,
            float(np.clip(0.55 + 0.35 * ((pressure - 0.55) / 0.45), 0.55, 0.90)),
        )

    control_strength = float(np.clip(max(eff, intensity), 0.0, 1.0))
    threshold_db_delta = float(
        np.clip(
            4.0
            + 8.0 * intensity
            + 1.8 * affricate_drive
            + 1.2 * phoneme_drive
            + 2.0 * german_language_boost
            + 4.0 * explicit_sibilant_boost,
            3.0,
            15.0,
        )
    )
    ratio_multiplier = float(
        np.clip(
            1.0
            + 0.55 * intensity
            + 0.12 * affricate_drive
            + 0.10 * phoneme_drive
            + 0.08 * german_language_boost
            + 0.45 * explicit_sibilant_boost,
            1.0,
            1.95,
        )
    )
    threshold_ratio_scale = float(
        np.clip(
            1.0
            - 0.18 * intensity
            - 0.08 * affricate_drive
            - 0.10 * phoneme_drive
            - 0.06 * german_language_boost
            - 0.20 * explicit_sibilant_boost,
            0.58,
            1.0,
        )
    )
    strength_cap = float(
        np.clip(
            1.0
            - (
                0.50
                + 0.25 * intensity
                + 0.08 * affricate_drive
                + 0.10 * phoneme_drive
                + 0.06 * german_language_boost
                + 0.08 * explicit_sibilant_boost
            )
            * max(eff, 0.35),
            min_strength_cap,
            1.0,
        )
    )
    reduction_mix = float(
        np.clip(
            0.45
            + 0.50 * intensity
            + 0.08 * affricate_drive
            + 0.10 * phoneme_drive
            + 0.05 * german_language_boost
            + 0.10 * explicit_sibilant_boost,
            0.45,
            1.0,
        )
    )

    return DeEsserIntensityProfile(
        sibilance_pressure=pressure,
        sibilance_ratio=sibilance_ratio,
        fricative_drive=fricative_drive,
        affricate_drive=affricate_drive,
        phoneme_drive=phoneme_drive,
        breathiness=float(np.clip(breathiness, 0.0, 1.0)),
        intensity=intensity,
        control_strength=control_strength,
        threshold_db_delta=threshold_db_delta,
        ratio_multiplier=ratio_multiplier,
        threshold_ratio_scale=threshold_ratio_scale,
        strength_cap=strength_cap,
        reduction_mix=reduction_mix,
    )

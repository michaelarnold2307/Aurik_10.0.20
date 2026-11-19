"""
Stereo Field Collapse Detector — Aurik Tier 2
============================================

Detektiert progressiven Stereofeld-Kollaps via:
- Gleitende Interchannel-Korrelation (10 s Fenster, 1 s Hop)
- Korrelation > 0.95 über > 30 s → Warnung
- ΔKorrelation > 0.3 in < 10 s → akuter Kollaps

Registriert als DefectType.STEREO_FIELD_COLLAPSE im DefectScanner.

Author: Aurik Tier 2
"""

import numpy as np


def detect_stereo_collapse(
    audio: np.ndarray,
    sr: int,
    window_s: float = 10.0,
    hop_s: float = 1.0,
) -> tuple[list[tuple[float, float]], float, float]:
    """Detektiert Stereofeld-Kollaps in Stereo-Audio.

    Args:
        audio: (n_samples, 2) Stereo-Audio
        sr: Abtastrate
        window_s: Korrelations-Fensterlänge in Sekunden
        hop_s: Hop-Größe in Sekunden

    Returns:
        (defect_locations, collapse_ratio, confidence)
        - defect_locations: Liste von (start_s, end_s) Tupeln
        - collapse_ratio: Anteil des Audios mit Korrelation > 0.95
        - confidence: Konfidenz-Score 0..1
    """
    if audio.ndim < 2 or audio.shape[1] < 2:
        return [], 0.0, 0.0

    duration = audio.shape[0] / sr
    if duration < window_s:
        return [], 0.0, 0.0

    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)

    win_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)

    n_frames = max(0, (audio.shape[0] - win_samples) // hop_samples + 1)
    if n_frames < 2:
        return [], 0.0, 0.0

    correlations = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + win_samples
        l_seg = left[start:end]
        r_seg = right[start:end]

        l_std = float(np.std(l_seg))
        r_std = float(np.std(r_seg))
        if l_std < 1e-12 or r_std < 1e-12:
            correlations[i] = 1.0  # silence → max correlation
        else:
            l_mean = float(np.mean(l_seg))
            r_mean = float(np.mean(r_seg))
            cov = float(np.mean((l_seg - l_mean) * (r_seg - r_mean)))
            correlations[i] = float(np.clip(cov / (l_std * r_std), -1.0, 1.0))

    # Detect collapse regions: correlation > 0.95
    collapse_mask = correlations > 0.95
    collapse_ratio = float(np.mean(collapse_mask))

    # Find contiguous collapse regions
    locations = []
    in_collapse = False
    collapse_start = 0.0

    for i in range(n_frames):
        t = i * hop_s + window_s / 2.0
        if collapse_mask[i] and not in_collapse:
            in_collapse = True
            collapse_start = max(0.0, t - window_s / 2.0)
        elif not collapse_mask[i] and in_collapse:
            in_collapse = False
            end_t = min(duration, t + window_s / 2.0)
            if end_t - collapse_start >= 30.0:
                locations.append((float(collapse_start), float(end_t)))

    if in_collapse:
        end_t = duration
        if end_t - collapse_start >= 30.0:
            locations.append((float(collapse_start), float(end_t)))

    # Acute collapse detection: Δcorrelation > 0.3 in < 10 s
    acute_hops = int(10.0 / hop_s)
    for i in range(n_frames - 1):
        for j in range(i + 1, min(i + acute_hops + 1, n_frames)):
            delta = correlations[j] - correlations[i]
            if delta > 0.3:
                t_start = i * hop_s + window_s / 2.0
                t_end = j * hop_s + window_s / 2.0
                # Check if not already covered
                already_covered = any(abs(loc[0] - t_start) < 5.0 for loc in locations)
                if not already_covered and t_end - t_start >= 5.0:
                    locations.append((float(max(0, t_start)), float(min(duration, t_end))))

    # Merge overlapping locations
    if locations:
        locations.sort()
        merged = [locations[0]]
        for loc in locations[1:]:
            last = merged[-1]
            if loc[0] <= last[1] + 5.0:
                merged[-1] = (last[0], max(last[1], loc[1]))
            else:
                merged.append(loc)
        locations = merged

    # Confidence
    if collapse_ratio > 0.6:
        confidence = 0.85
    elif collapse_ratio > 0.3:
        confidence = 0.70
    elif collapse_ratio > 0.1:
        confidence = 0.55
    else:
        confidence = 0.3

    # Also consider acute collapse for confidence
    if any((loc[1] - loc[0]) < 30.0 for loc in locations):  # acute events
        confidence = max(confidence, 0.65)

    return locations, collapse_ratio, float(np.clip(confidence, 0.0, 1.0))

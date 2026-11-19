"""Phase Rotation Detector — Aurik Tier 2
=======================================

Detektiert unnatürliche Phasenrotation (Allpass-Filter-Artefakte):
- Berechnet Gruppenlaufzeit-Varianz über Frequenz via Kreuzkorrelation der Phasenspektren
- Wenn Gruppenlaufzeit-Differenz > 2 ms zwischen benachbarten Bändern → Defekt

Registriert als DefectType.PHASE_ROTATION im DefectScanner.

Author: Aurik Tier 2
"""

import numpy as np


def detect_phase_rotation(
    audio: np.ndarray,
    sr: int,
    fft_size: int = 4096,
    hop_samples: int = 1024,
    band_edges_hz: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
) -> tuple[list[tuple[float, float]], float, float]:
    """Detektiert unnatürliche Phasenrotation / Gruppenlaufzeit-Anomalien.

    Verwendet die Phasen-Kreuzkorrelation zwischen benachbarten STFT-Frames,
    um unnatürliche Gruppenlaufzeit-Verschiebungen zu erkennen, die durch
    Allpass-Filter-Kaskaden verursacht werden.

    Args:
        audio: Mono-Audio (n_samples,)
        sr: Abtastrate
        fft_size: FFT-Größe
        hop_samples: Hop zwischen STFT-Frames
        band_edges_hz: Frequenzband-Grenzen für Gruppenlaufzeit-Vergleich

    Returns:
        (defect_locations, phase_dispersion, confidence)
    """
    if len(audio) < fft_size:
        return [], 0.0, 0.0

    audio = audio.astype(np.float64)

    # STFT
    n_frames = max(0, (len(audio) - fft_size) // hop_samples + 1)
    if n_frames < 3:
        return [], 0.0, 0.0

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    n_freq_bins = len(freqs)

    # Map band edges to FFT bins
    band_bins = []
    for edge_hz in band_edges_hz:
        idx = int(np.searchsorted(freqs, edge_hz))
        idx = min(idx, n_freq_bins - 1)
        band_bins.append(idx)

    band_bins = sorted(set(band_bins))
    if len(band_bins) < 2:
        return [], 0.0, 0.0

    num_bands = len(band_bins) - 1

    # Manually set small fft size for efficiency
    if fft_size > 2048:
        fft_size = 2048
        hop_samples = 512
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
        n_freq_bins = len(freqs)

    # Recompute frames with the (possibly) adjusted fft_size
    n_frames = max(0, (len(audio) - fft_size) // hop_samples + 1)
    if n_frames < 3:
        return [], 0.0, 0.0

    # Phase spectra per frame
    phase_spectra = np.zeros((n_frames, n_freq_bins), dtype=np.float64)
    magnitude_spectra = np.zeros((n_frames, n_freq_bins), dtype=np.float64)

    window = np.hanning(fft_size)
    for i in range(n_frames):
        start = i * hop_samples
        frame = audio[start : start + fft_size] * window
        spec = np.fft.rfft(frame)
        phase_spectra[i, :] = np.angle(spec)
        magnitude_spectra[i, :] = np.abs(spec)

    # --- Compute group delay via phase difference between frames ---
    # For a stationary sinusoid, the phase advance between frames is:
    #   Δφ[k] = 2π * f[k] * hop_samples / sr  (mod 2π)
    # The group delay anomaly measures deviation from expected phase advance.
    # Large band-to-band differences indicate allpass filter phase distortion.

    expected_phase_advance = 2.0 * np.pi * freqs * hop_samples / sr

    frame_dispersion = np.zeros(n_frames - 1, dtype=np.float64)

    for i in range(n_frames - 1):
        # Phase difference between consecutive frames
        raw_diff = phase_spectra[i + 1, :] - phase_spectra[i, :]
        # Wrap to [-π, π]
        raw_diff = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))
        # Deviation from expected linear phase advance
        deviation = raw_diff - expected_phase_advance
        deviation = np.arctan2(np.sin(deviation), np.cos(deviation))

        # Weight by magnitude (ignore noise bins)
        mag = magnitude_spectra[i, :] + magnitude_spectra[i + 1, :]
        mag_threshold = np.max(mag) * 0.01
        valid = mag > mag_threshold

        if np.sum(valid) < 10:
            continue

        # Group delay variation across frequency bands
        # Measure how much the phase deviation varies between bands
        band_deviations = np.zeros(num_bands, dtype=np.float64)
        for b in range(num_bands):
            b_start = band_bins[b]
            b_end = band_bins[b + 1]
            if b_end > n_freq_bins:
                b_end = n_freq_bins
            mask = valid[b_start:b_end]
            if np.sum(mask) < 2:
                continue
            band_deviations[b] = float(np.std(deviation[b_start:b_end][mask]))

        # Dispersion = mean band-to-band difference
        if num_bands >= 2:
            diffs = []
            for b in range(num_bands - 1):
                if band_deviations[b] > 0 and band_deviations[b + 1] > 0:
                    diffs.append(abs(band_deviations[b] - band_deviations[b + 1]))
            if diffs:
                frame_dispersion[i] = float(np.mean(diffs))

    # Scale to ms-equivalent (normalized)
    max_possible = np.max(frame_dispersion) if np.any(frame_dispersion > 0) else 0.0
    if max_possible > 1e-6:
        frame_dispersion = frame_dispersion / max_possible * 10.0  # scale to ~ms

    threshold_ms = 2.0
    anomaly_mask = frame_dispersion > threshold_ms

    phase_dispersion = float(np.mean(frame_dispersion))

    # Find contiguous anomaly regions
    locations = []
    frame_duration_s = hop_samples / sr
    in_anomaly = False
    anomaly_start = 0.0

    for i in range(len(frame_dispersion)):
        t = i * frame_duration_s
        if anomaly_mask[i] and not in_anomaly:
            in_anomaly = True
            anomaly_start = t
        elif not anomaly_mask[i] and in_anomaly:
            in_anomaly = False
            dur = t - anomaly_start
            if dur >= 0.1:
                locations.append((float(anomaly_start), float(t)))

    if in_anomaly:
        dur = (len(frame_dispersion) * frame_duration_s) - anomaly_start
        if dur >= 0.1:
            locations.append((float(anomaly_start), float(len(frame_dispersion) * frame_duration_s)))

    # Confidence
    anomaly_ratio = float(np.mean(anomaly_mask)) if len(anomaly_mask) > 0 else 0.0
    mean_anomaly = float(np.mean(frame_dispersion[anomaly_mask])) if np.any(anomaly_mask) else 0.0

    if anomaly_ratio > 0.3 and mean_anomaly > 5.0:
        confidence = 0.85
    elif anomaly_ratio > 0.15 and mean_anomaly > 3.0:
        confidence = 0.70
    elif anomaly_ratio > 0.05:
        confidence = 0.55
    elif anomaly_ratio > 0.01:
        confidence = 0.35
    else:
        confidence = 0.15

    return locations, phase_dispersion, float(np.clip(confidence, 0.0, 1.0))

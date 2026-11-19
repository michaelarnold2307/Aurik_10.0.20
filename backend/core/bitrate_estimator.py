"""
§v10 Bitrate-Estimator + SweetSpot Phase-Wrapper

Zwei finale Optimierungen:
1. MP3/AAC-Bitrate aus spektralem Cutoff schätzen
2. SweetSpot-Check zwischen UV3-Phasen (verhindert interne Rollbacks)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_lossy_bitrate(audio: np.ndarray, sr: int) -> tuple[int, float]:
    """Schätzt die Bitrate eines verlustbehafteten Codecs (MP3/AAC).

    MP3/AAC-Encoder wenden einen Tiefpass an, dessen Frequenz
    mit der Bitrate korreliert:
      320 kbps: >19 kHz | 256: >17 kHz | 192: >15 kHz
      160 kbps: >13.5 kHz | 128: >12 kHz | 96: >10 kHz | 64: >7 kHz

    Returns:
        (estimated_bitrate_kbps, confidence_0_1)
    """
    arr = np.asarray(audio, dtype=np.float64)
    mono = arr.mean(axis=1) if arr.ndim == 2 else arr
    mono = np.atleast_1d(mono).ravel()

    n_fft = 4096
    if len(mono) < n_fft:
        return 320, 0.1

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    energy = np.cumsum(spec**2)
    total = energy[-1] + 1e-12

    # Finde 98%-Energie-Grenzfrequenz (robuster als 99% bei Rauschen)
    cutoff_idx = np.searchsorted(energy, 0.98 * total)
    cutoff_hz = freqs[min(cutoff_idx, len(freqs) - 1)]

    # Auch prüfen: wo fällt das Spektrum steil ab?
    # Brickwall-Filter bei MP3 = plötzlicher Abfall >20dB
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    high_mask = freqs > 8000
    if np.any(high_mask):
        high_spec = spec_db[high_mask]
        high_freqs = freqs[high_mask]
        # Suche nach steilem Abfall
        for i in range(10, len(high_spec) - 1):
            if high_spec[i] - high_spec[i + 1] > 15:
                brickwall_hz = high_freqs[i]
                break
        else:
            brickwall_hz = cutoff_hz
    else:
        brickwall_hz = cutoff_hz

    # Kombiniere beide Schätzungen (niedrigere ist konservativer)
    est_hz = min(cutoff_hz, brickwall_hz)

    # Mapping zu Bitrate
    if est_hz > 19000:
        kbps, conf = 320, 0.90
    elif est_hz > 17500:
        kbps, conf = 256, 0.85
    elif est_hz > 15500:
        kbps, conf = 192, 0.80
    elif est_hz > 14000:
        kbps, conf = 160, 0.75
    elif est_hz > 12500:
        kbps, conf = 128, 0.70
    elif est_hz > 10500:
        kbps, conf = 96, 0.65
    elif est_hz > 8000:
        kbps, conf = 64, 0.60
    else:
        kbps, conf = 32, 0.50

    logger.info(
        "Bitrate estimate: %d kbps (cutoff=%.0fHz brickwall=%.0fHz conf=%.2f)", kbps, cutoff_hz, brickwall_hz, conf
    )

    return kbps, conf


def get_bitrate_aware_limits(material: str, audio: np.ndarray, sr: int) -> dict:
    """Erweitert Medium-Limits mit bitrate-spezifischen Anpassungen."""
    from backend.core.boundary_optimizer import get_media_limits

    limits = get_media_limits(material)

    if "mp3" in material.lower() or "aac" in material.lower():
        kbps, conf = estimate_lossy_bitrate(audio, sr)
        if kbps <= 128:
            # Niedrige Bitrate: deutlich weniger NR, mehr Frequenz-Rekonstruktion
            limits["nr_max_reduction_db"] = min(limits.get("nr_max_reduction_db", 12), 6.0)
            limits["freq_ceiling_hz"] = max(limits.get("freq_ceiling_hz", 15000), 15000)
            logger.info("Bitrate-aware: Low bitrate %dkbps -> reduced NR, extended freq ceiling", kbps)
        limits["estimated_bitrate_kbps"] = kbps
        limits["bitrate_confidence"] = conf

    return limits


def measure_musical_bandwidth(audio: np.ndarray, sr: int) -> dict:
    """Misst die TATSÄCHLICHE musikalische Bandbreite.

    320kbps-MP3 kann 1940er-Musik mit nur 5kHz Bandbreite enthalten.
    Der Encoder füllt alles darüber mit Stille — das ist keine Bandbreite.
    """
    arr = np.asarray(audio, dtype=np.float64)
    mono = arr.mean(axis=1) if arr.ndim == 2 else arr
    mono = np.atleast_1d(mono).ravel()
    n_fft = 8192
    if len(mono) < n_fft:
        return {"musical_ceiling_hz": sr / 2, "energy_bandwidth_hz": sr / 2, "is_bandwidth_limited": False}

    hop = n_fft // 2
    specs = []
    for i in range(0, len(mono) - n_fft, hop):
        specs.append(np.abs(np.fft.rfft(mono[i : i + n_fft] * np.hanning(n_fft))))
    avg_spec = np.mean(specs, axis=0)
    spec_db = 20.0 * np.log10(np.maximum(avg_spec, 1e-12))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    energy = np.cumsum(avg_spec**2)
    total = energy[-1] + 1e-12
    cutoff_99_idx = np.searchsorted(energy, 0.99 * total)
    energy_bw = freqs[min(cutoff_99_idx, len(freqs) - 1)]

    signal_mask = (freqs >= 100) & (freqs <= 5000)
    noise_mask = (freqs >= 8000) & (freqs <= 16000)
    signal_level = np.mean(spec_db[signal_mask]) if np.any(signal_mask) else 0
    noise_level = np.mean(spec_db[noise_mask]) if np.any(noise_mask) else -96
    hf_snr = signal_level - noise_level

    musical_ceiling = sr / 2
    for i in range(len(freqs) - 1, 0, -1):
        if freqs[i] < 5000:
            break
        if spec_db[i] - noise_level > 12 and np.any(noise_mask):
            musical_ceiling = freqs[i]
            break

    is_limited = energy_bw < 15000
    return {
        "musical_ceiling_hz": float(musical_ceiling),
        "energy_bandwidth_hz": float(energy_bw),
        "high_freq_snr_db": float(hf_snr),
        "is_bandwidth_limited": is_limited,
    }

"""WF-CASS: Scrape-Flutter-Restpfad für Cassette — Breitband-Modulations-Kompensation.

Scrape-Flutter = amplitudenmoduliertes Breitband-Artefakt (Bandkante kratzt am
Kopf, Modulationsfrequenzen typ. 5–120 Hz). Getrennt vom gemeinsamen Warp:
`phase_12_wow_flutter_fix` korrigiert Pitch/Geschwindigkeits-Variationen, dieses
Modul kompensiert die danach übrig bleibende **Amplituden**modulation (§SOTA-WF-CASS).

- `detect_scrape_flutter`: Hilbert-Einhüllende des Hochbands (>1,5 kHz) →
  Spektrum → periodische Peaks im Bereich 5–120 Hz → confidence/severity/mod_freqs.
- `compensate_scrape_flutter`: adaptive Hüllkurven-Normalisierung (nur der
  Modulationsanteil wird geglättet, Trägerphase bleibt unberührt) mit
  Soft-Knee-Blend und Never-worsen-Energie-Gate.

Determinismus (§G5 (GEBOTE.md)): keine Zeit-/Zufallsquellen in Entscheidungen.
Stereo-Layout-Invariante: verarbeitet (C, N) und (N, C) und gibt das
Eingabe-Layout zurück.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

logger = logging.getLogger(__name__)

_MOD_MIN_HZ = 5.0
_MOD_MAX_HZ = 120.0
_HIGHPASS_HZ = 1500.0
_LOWPASS_HZ = 8000.0
_ENV_SMOOTH_S = 0.05  # Glättungsfenster der Einhüllenden (~1/20 Hz)
_MAX_CORR = 2.0  # maximaler Hüllkurven-Korrekturfaktor
_MIN_CORR = 0.5
_CONF_ON = 0.55  # Soft-Knee-Start
_CONF_FULL = 0.85  # volle Wirkung


@dataclass
class ScrapeFlutterResult:
    confidence: float
    severity: float
    mod_freqs: list[float] = field(default_factory=list)


def _to_channels(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalisiert auf (C, N); returns (data, was_channels_first)."""
    if x.ndim == 1:
        return x[None, :], True
    if x.shape[0] <= 2 and x.shape[0] < x.shape[1]:
        return x, True  # bereits (C, N)
    return x.T, False  # (N, C) → (C, N)


def _bandpass_env(channel: np.ndarray, sr: int) -> np.ndarray:
    """Hochband-Einhüllende (Hilbert), normiert auf [0, 1] im Median."""
    nyq = 0.5 * sr
    b, a = butter(4, [_HIGHPASS_HZ / nyq, _LOWPASS_HZ / nyq], btype="band")
    hp = filtfilt(b, a, channel)
    env: np.ndarray = np.abs(hilbert(hp))
    med = float(np.median(env)) + 1e-12
    _env_norm: np.ndarray = env / med
    return _env_norm


def detect_scrape_flutter(audio: np.ndarray, sr: int) -> ScrapeFlutterResult:
    """Erkennt breitbandige Amplitudenmodulation (Scrape-Flutter) im Hochband."""
    data, _ = _to_channels(np.asarray(audio, dtype=np.float32))
    confs: list[float] = []
    sevs: list[float] = []
    freqs: set[float] = set()
    for ch in data:
        env = _bandpass_env(ch, sr)
        # Modulationstiefe (Hüllkurven-Varianz)
        sev = float(np.std(env) / (np.mean(env) + 1e-9))
        sevs.append(sev)
        # Periodische Peaks im Modulationsband
        spec = np.abs(np.fft.rfft(env - np.mean(env)))
        f = np.fft.rfftfreq(len(env), 1.0 / sr)
        band = (f >= _MOD_MIN_HZ) & (f <= _MOD_MAX_HZ)
        s_band = spec[band]
        f_band = f[band]
        if s_band.size < 3:
            confs.append(0.0)
            continue
        med = float(np.median(s_band)) + 1e-12
        # Peaks: lokale Maxima über 3× Median
        peaks = [
            float(f_band[i])
            for i in range(1, len(s_band) - 1)
            if s_band[i] > s_band[i - 1] and s_band[i] >= s_band[i + 1] and s_band[i] > 3.0 * med
        ]
        if not peaks:
            confs.append(0.0)
            continue
        peak_energy = float(np.sum(s_band[s_band > 3.0 * med]))
        total_energy = float(np.sum(s_band)) + 1e-12
        confs.append(float(np.clip(peak_energy / total_energy, 0.0, 1.0)))
        freqs.update(round(p, 1) for p in peaks)
    confidence = float(np.mean(confs)) if confs else 0.0
    severity = float(np.mean(sevs)) if sevs else 0.0
    return ScrapeFlutterResult(confidence=confidence, severity=severity, mod_freqs=sorted(freqs))


def compensate_scrape_flutter(audio: np.ndarray, sr: int, result: ScrapeFlutterResult) -> np.ndarray:
    """Kompensiert die Modulation via Hüllkurven-Normalisierung.

    Ohne Befund (confidence < _CONF_ON) wird das Signal unverändert
    zurückgegeben. Soft-Knee-Blend zwischen _CONF_ON und _CONF_FULL (§DSP:
    keine harten Schalter). Never-worsen-Energie-Gate je Kanal.
    """
    data, was_cf = _to_channels(np.asarray(audio, dtype=np.float32))
    if result.confidence < _CONF_ON or result.severity <= 1e-4:
        _passthrough: np.ndarray = np.asarray(audio, dtype=np.float32)
        return _passthrough
    blend = float(np.clip((result.confidence - _CONF_ON) / (_CONF_FULL - _CONF_ON), 0.0, 1.0))
    win = max(3, int(_ENV_SMOOTH_S * sr) | 1)
    out_channels: list[np.ndarray] = []
    for ch in data:
        env = _bandpass_env(ch, sr)
        env_smooth = np.convolve(env, np.ones(win) / win, mode="same")
        gain = np.clip(env_smooth / np.maximum(env, 1e-9), _MIN_CORR, _MAX_CORR)
        # Sanfte Blend des Gain-Faktors (kein hartes Umschalten)
        gain = 1.0 + (gain - 1.0) * blend
        corrected = ch * gain
        # Never-worsen-Energie-Gate: bei Abweichung > ±10 % Original behalten
        e_in = float(np.mean(ch**2)) + 1e-12
        e_out = float(np.mean(corrected**2))
        if not (0.9 * e_in <= e_out <= 1.1 * e_in):
            logger.debug("WF-CASS: Energie-Gate aktiv (%.3f → %.3f) — Kanal unverändert", e_in, e_out)
            out_channels.append(ch)
        else:
            out_channels.append(corrected)
    out = np.stack(out_channels)
    if audio.ndim == 1:
        _mono_out: np.ndarray = out[0]
        return _mono_out
    _stereo_out: np.ndarray = out if was_cf else out.T
    return _stereo_out

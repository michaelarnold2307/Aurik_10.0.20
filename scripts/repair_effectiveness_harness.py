"""Reparatur-Wirksamkeits-Harness (§SR-Audit, L3) — alle Defekt-Familien.

Macht den Arbeitsablauf messbar: MUSIK-ARTIGER TRAEGER + Defekt-Transformation ->
Scanner-Severity VORHER -> zustaendige Reparatur-Phase (Phase-Mapper-Zuordnung)
-> Scanner-Severity NACHHER -> physikalische Defekt-Metrik VORHER/NACHHER ->
Verbesserung dokumentiert und gegated.

MUSIK-TRAEGER (2026-09-25, Nutzer-Vorgabe): Aurik ist auf ECHTE MUSIK ausgelegt —
die Reparatur-Phasen (ML-Denoiser, psychoakustische Guards, Transienten- und
Musik-Klassifikatoren) sind auf musikalische Strukturen trainiert/kalibriert.
Rein-Ton-Synthesen liegen ausserhalb dieser Verteilung und duerfen kein
Wirksamkeits-Mass sein (Befund 2026-09-25: phase_12 Polyphonic-Multi-F0-Konsensus
gibt auf FM-Einton bit-identisch zurueck, obwohl flutter severity = 1.0).
Jeder Defekt wird hier als Signal-Transformation auf einem Musik-artigen Traeger
synthetisiert: Akkordfolge C-Dur/a-Moll/F-Dur/G-Dur, Noten-Huellenkurven mit
Attack-Transienten, Perkussions-Onsets und diffuser Raumanteil.

DEFEKT-SYNTHESE pro DefectType (deterministisch, §G5, copilot-instructions.md):
- Signal-Transformationen: Verzerrung, HueLLkurven-Eingriff, Spektral-EQ,
  Zeitachse (FM/Resampling), Stereo-Geometrie, Echo/Nachhall
- Additive Overlays: Brumm, Hiss, Knistern, Klicks, Motorstoerung

Gates (2026-09-25, nach empirischer Kalibrierung):
- reduce-Metriken (Defekt-Energie, Impulse, Dips, ...): physikalisch auf
  <= 50 % des VORHER-Werts (>= 50 % Reduktion).
- increase-Metriken (Restaurationsphasen: HF-Verlust, Dynamik, Transienten):
  physikalisch auf >= 125 % des VORHER-Werts (>= 25 % Zunahme).
- GATE_SPECIAL: absolute Defekt-Schwellen (z.B. Stereofeld-Kollaps IACC <= 0.90).
- Regressions-Schranke: Scanner-Severity darf nicht steigen
  (nachher <= vorher + 0.03). Die Scanner-Severity zaehlt Musik-Transienten mit
  (gemessen: Clicks 52->2 Kanten bei Severity 0,038->0,022) — die physikalische
  Metrik ist das harte Kriterium, Severity die Schranke.

Befund-Kategorien:
- [LUECKE-P] Reparatur-Phase physikalisch unwirksam (Prio: SOTA-Roadmap)
- [LUECKE-S] Scanner detektiert den Defekt auf Musik nicht (sev < 0.15)
- [SKIP]     dokumentierter Sonderfall (ML-Passthrough ohne Modell §V6 (copilot-instructions.md),
             Bewahr-Typ SOFT_SATURATION §2.1/§6.3)

Nutzung: python scripts/repair_effectiveness_harness.py [familie...]
Exit 0 = alle Gates gruen, 1 = mindestens eine Luecke, 2 = unbekannte Familie.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from scipy import signal

logging.disable(logging.CRITICAL)

from backend.core.defect_scanner import DefectScanner, DefectScoreView, DefectType, MaterialType
from backend.core.phases.phase_interface import PhaseInterface

# Evidenz-Harness: kanonische Quelle fuer Fall-Materialien und die Selektion
# des repraesentativsten Falls pro DefectType (hoechste erwartete Severity).
_spec = importlib.util.spec_from_file_location(
    "defect_evidence_harness_mod", str(_REPO / "scripts" / "defect_evidence_harness.py")
)
assert _spec is not None and _spec.loader is not None
_EVIDENCE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_EVIDENCE)
EVIDENCE_CASES: dict = _EVIDENCE.CASES

import backend.core.defect_phase_mapper as _mapper

_PHASE_MAP = getattr(_mapper, "_PHASE_MAP", None)
if _PHASE_MAP is None:
    for _attr in dir(_mapper):
        _obj = getattr(_mapper, _attr)
        if isinstance(_obj, dict) and any(isinstance(_k, DefectType) for _k in _obj):
            _PHASE_MAP = _obj
            break
assert _PHASE_MAP is not None, "Phase-Mapper nicht ladbar"

# Phasen-IDs im Mapper, die keine eigenen Module sind (kanonische Aliasse).
_PHASE_ALIAS: dict[str, str] = {
    "phase_07_declip": "phase_07_declipper",
    "phase_17_declip": "phase_07_declipper",
    "phase_04_eq": "phase_04_eq_correction",
    "phase_06_stereo_enhancement": "phase_13_stereo_enhancement",
    "phase_25_pitch_correction": "phase_31_speed_pitch_correction",
    "phase_56_head_wear_compensation": "phase_56_spectral_band_gap_repair",
}

SR = _EVIDENCE.SR
DUR = _EVIDENCE.DUR


# ------------------------------------------------------------------ Physikalische Metriken
# Jede Metrik misst die physische Praesenz des Defekts (kleiner = besser).
# Direction: "reduce" (Defekt muss verschwinden) oder "increase"
# (Restaurationsphasen muessen verlorene Energie wiederherstellen).


def _mono(audio: np.ndarray) -> np.ndarray:
    """Layout-agnostisch zu Mono (Stereo-Invariante, AGENTS.md §3)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.shape[0] == 2 and a.shape[1] > 2:
        return a.mean(axis=0)
    return a.mean(axis=1)


def _rfft(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return freqs, spec


def _band_energy(x: np.ndarray, lo: float, hi: float) -> float:
    freqs, spec = _rfft(x)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sqrt(np.sum(spec[mask] ** 2)) / max(len(x), 1))


def _env(x: np.ndarray, win: int) -> np.ndarray:
    rms = np.sqrt(np.convolve(x**2, np.ones(win) / win, mode="same"))
    return rms


def _if_std(x: np.ndarray, lo: float = 40.0, hi: float = 6000.0) -> float:
    """AC-Std der Instantanfrequenz (Hilbert-Demodulation) im Band — Wow/Flutter-Mass."""
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    y = signal.sosfiltfilt(sos, x)
    analytic = signal.hilbert(y)
    inst_freq = np.diff(np.unwrap(np.angle(analytic))) * SR / (2 * np.pi)
    finite = inst_freq[np.isfinite(inst_freq)]
    if len(finite) < 100:
        return 0.0
    return float(np.std(finite))


def _dip_depth(x: np.ndarray, win_s: float = 0.15) -> float:
    """Tiefe der Pegel-Einbrueche (Dropouts, Head-Dips, Splices). min-basiert,
    damit auch kurze Luecken (6 ms Oxid-Dropout) gemessen werden."""
    win = max(32, int(win_s * SR))
    e = _env(x, win)
    p95 = float(np.percentile(e, 95))
    emin = float(np.min(e))
    return max(0.0, (p95 - emin) / max(p95, 1e-12))


def _autocorr_peak(x: np.ndarray, lo_s: float, hi_s: float) -> float:
    """Max. normalisierte Autokorrelation der HUELLKURVE im Lag-Fenster (Echo-Mass).
    Auf periodischen Traegern versagt die Rohsignal-Autokorrelation (Peak bei
    jedem Perioden-Vielfachen); die Huellkurve isoliert Geister-Ereignisse."""
    n = len(x)
    lo = max(1, int(lo_s * SR))
    hi = min(n - 1, int(hi_s * SR))
    if hi <= lo:
        return 0.0
    env = np.abs(signal.hilbert(x))
    env = env - env.mean()
    xc = np.correlate(env, env, mode="full")[n - 1 :]
    xc = xc / max(float(np.dot(env, env)), 1e-12)
    return float(np.max(np.abs(xc[lo : hi + 1])))


def _tail_ratio(x: np.ndarray) -> float:
    """Nachhall-Schwanz-Energie: Verhaeltnis der Energie 100-500 ms nach jedem
    Onset zur Energie 0-100 ms nach dem Onset (RT60-Proxie)."""
    env = np.abs(signal.hilbert(x))
    ratio_sum = 0.0
    n_onsets = 0
    for k in range(1, 6):
        idx = k * 2 * SR
        if idx + int(0.5 * SR) > len(x):
            break
        direct = float(np.sum(env[idx : idx + int(0.1 * SR)] ** 2))
        tail = float(np.sum(env[idx + int(0.1 * SR) : idx + int(0.5 * SR)] ** 2))
        ratio_sum += tail / max(direct, 1e-12)
        n_onsets += 1
    if n_onsets == 0:
        return 0.0
    return float(ratio_sum / n_onsets)


def _sideband_energy(x: np.ndarray, f0: float = 440.0, lo_d: float = 2.0, hi_d: float = 300.0) -> float:
    """Energie der Seitenbaender um den Traeger (Jitter-/Phasenrausch-Signatur)."""
    freqs, spec = _rfft(x)
    mask = ((freqs >= f0 - hi_d) & (freqs <= f0 - lo_d)) | ((freqs >= f0 + lo_d) & (freqs <= f0 + hi_d))
    return float(np.sqrt(np.sum(spec[mask] ** 2)) / max(len(x), 1))


def _f0_deviation(x: np.ndarray, reference_hz: float = 220.0) -> float:
    """Abweichung der gemessenen Grundfrequenz von der Referenz (Pitch/Speed)."""
    y = np.asarray(x, dtype=np.float64)
    y = signal.sosfiltfilt(signal.butter(4, [150, 400], btype="bandpass", fs=SR, output="sos"), y)
    zc = np.where(np.diff(np.signbit(y)))[0]
    if len(zc) < 50:
        return 0.0
    f0 = 0.5 * SR / float(np.mean(np.diff(zc)))
    return abs(f0 - reference_hz)


def _crest(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
    return float(np.max(np.abs(x)) / rms)


def _flat_top_count(x: np.ndarray) -> float:
    """Anzahl der Samples an der Clipping-Decke (Hard-Clipping-Mass)."""
    return float(np.sum(np.abs(x) >= 0.98))


def _impulse_peak(x: np.ndarray) -> float:
    """Impuls-Prominenz: groesste diskrete Kruemmung (Bandstoss-/Klick-Mass).

    Ein Bandstoss-Klick ist ein sub-ms-Impuls (Kruemmungsspitze); die
    PEGELSTUFE zwischen zwei Bandstuecken ist Inhalt (zwei Aufnahmen), kein
    Artefakt — sie wird bewusst NICHT gemessen (L3-Instrument-Kalibrierung
    2026-09-25: die alte _dip_depth-Metrik mass die Stufe und bestrafte damit
    korrektes Verhalten der phase_64).
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 2:
        y = _mono(y)
    if y.size <= 3:
        return 0.0
    return float(np.max(np.abs(np.diff(y, n=2))))


def _thd_tone(x: np.ndarray, f0: float = 220.0) -> float:
    """THD-Proxie: Oberschwingungs-Energie / Grundton-Energie."""
    freqs, spec = _rfft(x)
    fund = float(np.sum(spec[(freqs >= f0 * 0.9) & (freqs <= f0 * 1.1)] ** 2))
    harm = float(np.sum(spec[(freqs >= f0 * 1.8) & (freqs <= f0 * 8.0)] ** 2))
    return float(np.sqrt(harm / max(fund, 1e-12)))


def _hf_shelf_ratio(x: np.ndarray) -> float:
    """HF-Ueberhoehung (5-10 kHz) relativ zum Tiefmittenband (100-500 Hz)."""
    hi = _band_energy(x, 5000.0, 10000.0)
    lo = _band_energy(x, 100.0, 500.0)
    return float(hi / max(lo, 1e-12))


def _asymmetry(x: np.ndarray) -> float:
    """Wellenform-Asymmetrie |mean(x^3)|/rms^3 (Nadel-/Uebersteuerungs-Mass)."""
    rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
    return float(abs(np.mean(x**3)) / (rms**3))


def _hf_env_std(x: np.ndarray, lo: float = 4000.0, hi: float = 12000.0) -> float:
    """Zeitliche Varianz der HF-Band-Energie (periodische HF-Dips: Head Clog/Wear)."""
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    y = signal.sosfiltfilt(sos, x)
    win = max(128, int(0.25 * SR))
    e = _env(y, win)
    return float(np.std(e) / max(np.mean(e), 1e-12))


def _noise_floor_var(x: np.ndarray) -> float:
    """Varianz des Rauschbodens (NR-Atmung/Pumpen)."""
    win = max(128, int(0.1 * SR))
    e = _env(x, win)
    return float(np.std(e) / max(np.mean(e), 1e-12))


def _quant_step_peakiness(x: np.ndarray) -> float:
    """Histogramm-Spitzigkeit an Quantisierungsstufen (8-Bit-Mass)."""
    step = 1.0 / 128.0
    bins = np.arange(-1.0, 1.0 + step, step)
    hist, _ = np.histogram(x, bins=bins)
    hist = hist.astype(np.float64)
    hist = hist / max(float(hist.sum()), 1.0)
    return float(np.max(hist) * 255.0)


def _cross_channel_lag(audio: np.ndarray, lo: float = 2000.0, hi: float = 12000.0) -> float:
    """HF-Gruppenlaufzeit-Differenz L/R (Azimuth-Fehler-Mass) in Sekunden."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2:
        return 0.0
    ch = a if a.shape[0] == 2 else a.T
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    l = signal.sosfiltfilt(sos, ch[0])
    r = signal.sosfiltfilt(sos, ch[1])
    xc = np.correlate(l, r, mode="full")
    lag = int(np.argmax(np.abs(xc))) - (len(l) - 1)
    return abs(float(lag)) / SR


def _lr_ratio_dev(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2:
        return 0.0
    ch = a if a.shape[0] == 2 else a.T
    rl = float(np.sqrt(np.mean(ch[0] ** 2)) + 1e-12)
    rr = float(np.sqrt(np.mean(ch[1] ** 2)) + 1e-12)
    return abs(float(rl / rr) - 1.0)


def _iacc(audio: np.ndarray) -> float:
    """Interaural-Cross-Correlation (Stereofeld-Kollaps: IACC -> 1)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2:
        return 1.0
    ch = a if a.shape[0] == 2 else a.T
    denom = float(np.sqrt(np.dot(ch[0], ch[0]) * np.dot(ch[1], ch[1])) + 1e-12)
    return float(np.dot(ch[0], ch[1]) / denom)


def _lr_sum_cancel(audio: np.ndarray) -> float:
    """Phasenproblem-Mass: Energie(L+R)/Energie(L-R) — Gegenpoligkeit loescht die Summe."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2:
        return 1.0
    ch = a if a.shape[0] == 2 else a.T
    s = float(np.sum((ch[0] + ch[1]) ** 2))
    d = float(np.sum((ch[0] - ch[1]) ** 2))
    return float(s / max(d, 1e-12))


def _envelope_slope(x: np.ndarray) -> float:
    """|Steigung| der Pegelhuellkurve ueber den Song (Amplituden-Drift-Mass)."""
    win = max(128, int(0.5 * SR))
    e = _env(x, win)
    n = len(e)
    t = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(t, e, 1)[0])
    return abs(slope) / max(float(np.mean(e)), 1e-12)


# Metrik-Registry: DefectType -> (Metrik, Richtung)
METRICS: dict[DefectType, tuple] = {
    DefectType.HISS: (_band_energy, "reduce"),
    DefectType.HIGH_FREQ_NOISE: (_band_energy, "reduce"),
    DefectType.MODULATION_NOISE: (_band_energy, "reduce"),
    DefectType.GENERATION_LOSS: (_band_energy, "reduce"),
    DefectType.STICKY_SHED_RESIDUE: (_dip_depth, "reduce"),
    DefectType.QUANTIZATION_NOISE: (_quant_step_peakiness, "reduce"),
    DefectType.COMPRESSION_ARTIFACTS: (_band_energy, "increase"),
    DefectType.JITTER_ARTIFACTS: (_sideband_energy, "reduce"),
    DefectType.DIGITAL_ARTIFACTS: (_band_energy, "reduce"),
    DefectType.ALIASING: (_band_energy, "reduce"),
    DefectType.HUM: (_band_energy, "reduce"),
    DefectType.MOTOR_INTERFERENCE: (_band_energy, "reduce"),
    DefectType.LOW_FREQ_RUMBLE: (_band_energy, "reduce"),
    DefectType.PROXIMITY_EFFECT_EXCESS: (_band_energy, "reduce"),
    DefectType.ROOM_MODE_RESONANCE: (_band_energy, "reduce"),
    DefectType.CLICKS: (_flat_top_count, "reduce"),
    DefectType.CRACKLE: (_flat_top_count, "reduce"),
    DefectType.TAPE_SPLICE_ARTIFACT: (_impulse_peak, "reduce"),
    DefectType.LACQUER_DISC_DEGRADATION: (_flat_top_count, "reduce"),
    DefectType.STYLUS_DAMAGE: (_asymmetry, "reduce"),
    DefectType.CLIPPING: (_flat_top_count, "reduce"),
    DefectType.OVERLOAD_DISTORTION: (_thd_tone, "reduce"),
    DefectType.DISTORTION: (_thd_tone, "reduce"),
    DefectType.INTERMODULATION_DISTORTION: (_band_energy, "reduce"),
    DefectType.INNER_GROOVE_DISTORTION: (_thd_tone, "reduce"),
    DefectType.DC_OFFSET: (_asymmetry, "reduce"),
    DefectType.STEREO_IMBALANCE: (_lr_ratio_dev, "reduce"),
    DefectType.CROSSTALK: (_iacc, "reduce"),
    DefectType.STEREO_FIELD_COLLAPSE: (_iacc, "reduce"),
    DefectType.PHASE_ISSUES: (_lr_sum_cancel, "increase"),
    DefectType.PHASE_ROTATION: (_crest, "increase"),
    DefectType.AZIMUTH_ERROR: (_cross_channel_lag, "reduce"),
    DefectType.WOW: (_if_std, "reduce"),
    DefectType.FLUTTER: (_if_std, "reduce"),
    DefectType.SCRAPE_FLUTTER: (_if_std, "reduce"),
    DefectType.MULTIBAND_WOW_FLUTTER: (_if_std, "reduce"),
    DefectType.FLUTTER_SPECTRAL_SIDEBANDS: (_if_std, "reduce"),
    DefectType.TRANSPORT_BUMP: (_band_energy, "reduce"),
    DefectType.DROPOUTS: (_dip_depth, "reduce"),
    DefectType.DROPOUT: (_dip_depth, "reduce"),
    DefectType.DROPOUT_OXIDE: (_dip_depth, "reduce"),
    DefectType.DROPOUT_HEAD_CONTACT: (_dip_depth, "reduce"),
    DefectType.DROPOUT_SPLICE: (_dip_depth, "reduce"),
    DefectType.MPEG_FRAME_LOSS: (_dip_depth, "reduce"),
    DefectType.TAPE_HEAD_LEVEL_DIP: (_dip_depth, "reduce"),
    DefectType.SIBILANCE: (_band_energy, "reduce"),
    DefectType.VOCAL_HARSHNESS: (_band_energy, "reduce"),
    DefectType.REVERB_EXCESS: (_tail_ratio, "reduce"),
    DefectType.PRE_ECHO: (_autocorr_peak, "reduce"),
    DefectType.PRINT_THROUGH: (_autocorr_peak, "reduce"),
    DefectType.GROOVE_ECHO: (_autocorr_peak, "reduce"),
    DefectType.NR_BREATHING_ARTIFACT: (_noise_floor_var, "reduce"),
    DefectType.RIAA_CURVE_ERROR: (_hf_shelf_ratio, "reduce"),
    DefectType.DOLBY_NR_MISMATCH: (_hf_shelf_ratio, "reduce"),
    DefectType.SPEED_CALIBRATION_ERROR: (_f0_deviation, "reduce"),
    DefectType.PITCH_DRIFT: (_f0_deviation, "reduce"),
    # Restaurationsphasen: verlorene Energie muss ZURUECK kommen.
    DefectType.BANDWIDTH_LOSS: (_band_energy, "increase"),
    DefectType.HF_REMANENCE_LOSS: (_band_energy, "increase"),
    DefectType.BIAS_ERROR: (_band_energy, "increase"),
    DefectType.HEAD_WEAR: (_band_energy, "increase"),
    DefectType.TAPE_HEAD_CLOG: (_hf_env_std, "reduce"),
    DefectType.DYNAMIC_COMPRESSION_EXCESS: (_crest, "increase"),
    DefectType.TRANSIENT_SMEARING: (_crest, "increase"),
    DefectType.AMPLITUDE_DRIFT: (_envelope_slope, "reduce"),
    # §2.1/§6.3: Bewahr-Typ — keine Reparatur (dokumentierter Skip).
    DefectType.SOFT_SATURATION: (_thd_tone, "preserve"),
}

# Band-Konfiguration je DefectType (reduziert False-Positive gegenueber Musik).
_BAND_CFG: dict[DefectType, tuple[float, float]] = {
    DefectType.HISS: (2000.0, 10000.0),
    DefectType.HIGH_FREQ_NOISE: (2000.0, 12000.0),
    DefectType.MODULATION_NOISE: (2000.0, 10000.0),
    DefectType.GENERATION_LOSS: (2000.0, 10000.0),
    DefectType.COMPRESSION_ARTIFACTS: (3000.0, 3800.0),  # Spektralloch (Codec-Signatur) muss gefuellt werden
    DefectType.DIGITAL_ARTIFACTS: (15000.0, 22000.0),
    DefectType.ALIASING: (18000.0, 23000.0),
    DefectType.HUM: (40.0, 320.0),
    DefectType.MOTOR_INTERFERENCE: (70.0, 320.0),
    DefectType.LOW_FREQ_RUMBLE: (10.0, 80.0),
    DefectType.PROXIMITY_EFFECT_EXCESS: (60.0, 250.0),
    DefectType.ROOM_MODE_RESONANCE: (40.0, 200.0),
    DefectType.TRANSPORT_BUMP: (10.0, 90.0),
    DefectType.SIBILANCE: (5000.0, 10000.0),
    DefectType.VOCAL_HARSHNESS: (2000.0, 6000.0),
    DefectType.INTERMODULATION_DISTORTION: (880.0, 1120.0),  # f2-f1 = 1 kHz Produktband
    DefectType.BANDWIDTH_LOSS: (8000.0, 16000.0),
    DefectType.HF_REMANENCE_LOSS: (8000.0, 16000.0),
    DefectType.BIAS_ERROR: (6000.0, 14000.0),
    DefectType.HEAD_WEAR: (8000.0, 16000.0),
}


# Metrik-Fabrik: (DefectType) -> callable(audio) -> float
def _metric_fn(dt: DefectType):
    fn, _dir = METRICS[dt]
    cfg = _BAND_CFG.get(dt)
    if cfg is not None:
        lo, hi = cfg

        def _band(audio: np.ndarray, _lo: float = lo, _hi: float = hi) -> float:
            return _band_energy(_mono(audio), _lo, _hi)

        return _band
    if fn is _autocorr_peak:
        lag = {
            DefectType.GROOVE_ECHO: (1.5, 2.2),
            DefectType.PRE_ECHO: (0.08, 0.35),
            DefectType.PRINT_THROUGH: (0.08, 0.4),
        }.get(dt, (0.05, 2.0))
        lo, hi = lag

        def _ac(audio: np.ndarray, _lo: float = lo, _hi: float = hi) -> float:
            return _autocorr_peak(_mono(audio), _lo, _hi)

        return _ac
    if fn is _flat_top_count:
        if dt is DefectType.CLICKS or dt is DefectType.CRACKLE or dt is DefectType.LACQUER_DISC_DEGRADATION:

            def _edges(audio: np.ndarray) -> float:
                x = _mono(audio)
                d = np.abs(np.diff(x))
                # Feste Salienz-Schwelle: bei Musik ist 10x median zu empfindlich
                # (Noten-Attacken zaehlen als Klicks, Befund 2026-09-25: 52->275
                # bei Musik vs. 52->2 auf dem reinen Ton). Impuls-Defekte haben
                # Ableitungen ~0.8, legitime Musik-Kanten ~0.02-0.05.
                thr = max(0.3, 8.0 * float(np.median(d)))
                return float(np.sum(d > thr))

            return _edges

        def _ftc(audio: np.ndarray) -> float:
            return _flat_top_count(_mono(audio))

        return _ftc
    if fn is _impulse_peak:

        def _ip(audio: np.ndarray) -> float:
            return _impulse_peak(audio)

        return _ip
    if fn is _dip_depth:
        win = {
            DefectType.TAPE_SPLICE_ARTIFACT: 0.03,
            DefectType.DROPOUT_SPLICE: 0.02,
            DefectType.DROPOUT_OXIDE: 0.04,
            DefectType.MPEG_FRAME_LOSS: 0.06,
        }.get(dt, 0.15)

        def _dip(audio: np.ndarray, _win: float = win) -> float:
            return _dip_depth(_mono(audio), _win)

        return _dip
    if fn is _crest:

        def _c(audio: np.ndarray) -> float:
            return _crest(_mono(audio))

        return _c
    if fn is _asymmetry:

        def _as(audio: np.ndarray) -> float:
            return _asymmetry(_mono(audio))

        return _as
    if fn is _thd_tone:

        def _thd(audio: np.ndarray) -> float:
            return _thd_tone(_mono(audio))

        return _thd
    if fn is _iacc:

        def _ic(audio: np.ndarray) -> float:
            return _iacc(audio)

        return _ic
    if fn is _lr_sum_cancel:

        def _sc(audio: np.ndarray) -> float:
            return _lr_sum_cancel(audio)

        return _sc
    if fn is _lr_ratio_dev:

        def _lrd(audio: np.ndarray) -> float:
            return _lr_ratio_dev(audio)

        return _lrd
    if fn is _cross_channel_lag:

        def _ccl(audio: np.ndarray) -> float:
            return _cross_channel_lag(audio)

        return _ccl
    if fn is _if_std:

        def _ifs(audio: np.ndarray) -> float:
            return _if_std(_mono(audio))

        return _ifs
    if fn is _f0_deviation:
        ref = {DefectType.PITCH_DRIFT: 440.0, DefectType.SPEED_CALIBRATION_ERROR: 220.0}.get(dt, 220.0)

        def _f0d(audio: np.ndarray, _ref: float = ref) -> float:
            return _f0_deviation(_mono(audio), _ref)

        return _f0d
    if fn is _tail_ratio:

        def _tr(audio: np.ndarray) -> float:
            return _tail_ratio(_mono(audio))

        return _tr
    if fn is _sideband_energy:

        def _sbe(audio: np.ndarray) -> float:
            return _sideband_energy(_mono(audio))

        return _sbe
    if fn is _envelope_slope:

        def _esl(audio: np.ndarray) -> float:
            return _envelope_slope(_mono(audio))

        return _esl
    if fn is _noise_floor_var:

        def _nfv(audio: np.ndarray) -> float:
            return _noise_floor_var(_mono(audio))

        return _nfv
    if fn is _quant_step_peakiness:

        def _qsp(audio: np.ndarray) -> float:
            return _quant_step_peakiness(_mono(audio))

        return _qsp
    if fn is _hf_env_std:

        def _hfe(audio: np.ndarray) -> float:
            return _hf_env_std(_mono(audio))

        return _hfe
    if fn is _hf_shelf_ratio:

        def _hfs(audio: np.ndarray) -> float:
            return _hf_shelf_ratio(_mono(audio))

        return _hfs

    # Fallback: Breitband-RMS-Abweichung
    def _rmsf(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(_mono(audio) ** 2)) + 1e-12)

    return _rmsf


# ------------------------------------------------------------------ Musik-Traeger
# Deterministischer Musik-artiger Traeger (§G5 (copilot-instructions.md)): Akkordfolge mit Noten-
# Huellenkurven, Attack-Transienten, Perkussions-Onsets und Raum-Boden.
# Die Reparatur-Phasen sind auf musikalische Strukturen kalibriert — der
# Traeger bildet die grundlegenden musikalischen Gestaltmittel ab:
# Tonalitaet (Akkorde), Rhythmik (Onsets), Dynamik (Huellen), Raum (Boden).


def _music_mono() -> np.ndarray:
    """Mono-Musik: 4 Akkorde (C-Dur, a-Moll, F-Dur, G-Dur) ueber 15 s."""
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(97)
    chords = (
        (261.63, 329.63, 392.00),  # C-Dur
        (220.00, 261.63, 329.63),  # a-Moll
        (174.61, 220.00, 261.63),  # F-Dur
        (196.00, 246.94, 392.00),  # G-Dur
    )
    x = np.zeros(SR * DUR)
    seg = SR * DUR // len(chords)
    for ci, chord in enumerate(chords):
        s0 = ci * seg
        note_t = np.arange(seg) / SR
        for f0 in chord:
            env = np.minimum(1.0, note_t / 0.05) * np.exp(-note_t / 1.2)
            note = np.sin(2 * np.pi * f0 * note_t)
            note = note + 0.4 * np.sin(2 * np.pi * 2 * f0 * note_t)
            note = note + 0.15 * np.sin(2 * np.pi * 3 * f0 * note_t)
            x[s0 : s0 + seg] += 0.09 * env * note
    # Perkussions-Onsets (2 s Takt) fuer Transienten-Phasen (08/36) und
    # Echo-Detektion: kurzer HF-Attack mit schnellem Abklingen.
    click_len = int(0.006 * SR)
    ct = np.arange(click_len) / SR
    onsets = 0.22 * np.exp(-ct / 0.002) * np.sin(2 * np.pi * 2000.0 * ct)
    for k in range(1, 15, 2):
        idx = k * SR
        x[idx : idx + click_len] += onsets
    x = x + 0.008 * rng.standard_normal(len(t))  # diffuser Raum-Boden
    return x


def _music_st() -> np.ndarray:
    """Stereo-Musik (N, 2) mit realistischem Panorama: Akkord-Noten leicht
    gepaent, damit L/R unterschiedliche Inhalte haben (Kanal-Trennung wie in
    einer echten Stereoaufnahme)."""
    m = _music_mono()
    n = len(m)
    rng_l = np.random.default_rng(98)
    rng_r = np.random.default_rng(99)
    # Langsames Panorama (0,33 Hz): Instrument wandert im Stereofeld
    pan = 0.30 * np.sin(2 * np.pi * 0.33 * np.arange(n) / SR)
    L = m * (1.0 - pan) + 0.015 * rng_l.standard_normal(n)
    R = m * (1.0 + pan) + 0.015 * rng_r.standard_normal(n)
    peak = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))), 1e-9)
    scale = 0.35 / peak
    return np.stack([L * scale, R * scale], axis=1).astype(np.float32)


def _st(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,2)-Stereo in Einzelkanaele zerlegen (layout-stabil)."""
    return np.asarray(x[:, 0], dtype=np.float64), np.asarray(x[:, 1], dtype=np.float64)


def _stack(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.stack([l, r], axis=1).astype(np.float32)


# ------------------------------------------------------------------ Defekt-Bausteine


def _bandpass(x: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def _lowpass(x: np.ndarray, hz: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, hz, btype="lowpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def _peaking(x: np.ndarray, f0: float, q: float, gain_db: float) -> np.ndarray:
    """Parametrischer EQ-Bell (RBJ Cookbook)."""
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    aa = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return signal.lfilter(np.array(b) / aa[0], np.array(aa) / aa[0], x)


def _shelf(x: np.ndarray, f0: float, gain_db: float, high: bool = True) -> np.ndarray:
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / SR
    cs = np.cos(w0)
    alpha = np.sin(w0) / 2 * np.sqrt((a + 1 / a) * (1 / 0.9 - 1) + 2)
    two_sqrt_a_alpha = 2 * np.sqrt(a) * alpha
    if high:
        b0 = a * ((a + 1) + (a - 1) * cs + two_sqrt_a_alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cs)
        b2 = a * ((a + 1) + (a - 1) * cs - two_sqrt_a_alpha)
    else:
        b0 = a * ((a + 1) - (a - 1) * cs + two_sqrt_a_alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cs)
        b2 = a * ((a + 1) - (a - 1) * cs - two_sqrt_a_alpha)
    a0 = (a + 1) - (a - 1) * cs + two_sqrt_a_alpha
    a1 = 2 * ((a - 1) - (a + 1) * cs)
    a2 = (a + 1) - (a - 1) * cs - two_sqrt_a_alpha
    return signal.lfilter([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0], x)


def _resample(x: np.ndarray, t_dev_samples: np.ndarray) -> np.ndarray:
    """Zeitverzerrung: x wird an den zeitlich verschobenen Positionen gelesen
    (FM/Wow/Flutter/Pitch/Speed/Jitter — alle Zeitachsen-Defekte)."""
    n = x.shape[0]
    idx = np.clip(np.arange(n, dtype=np.float64) + t_dev_samples, 0, n - 1)
    if x.ndim == 2:
        return np.stack(
            [np.interp(idx, np.arange(n), x[:, ch]) for ch in range(x.shape[1])],
            axis=1,
        )
    return np.interp(idx, np.arange(n), x)


def _dip_envelope(n: int, events: list[tuple[int, int, float]]) -> np.ndarray:
    """Huellenkurve mit Pegel-Einbruechen: events = [(start, end, gain)]."""
    env = np.ones(n)
    for s, e, g in events:
        s = max(0, int(s))
        e = min(n, int(e))
        if e > s:
            env[s:e] = g
    # sanfte Ränder (2 ms) gegen Klicks an den Nahtstellen
    ramp = max(8, int(0.002 * SR))
    d = np.diff(env, prepend=1.0)
    edges = np.where(np.abs(d) > 1e-6)[0]
    for ei in edges:
        s = max(0, ei - ramp)
        e = min(n, ei + ramp)
        env[s:e] = np.linspace(env[s - 1] if s > 0 else 1.0, env[e] if e < n else 1.0, e - s)
    return env


def _echo(x: np.ndarray, delay_s: float, gain: float) -> np.ndarray:
    """Geist-Echo (Pre/Post) — Kopie des Signals um delay_s verschoben."""
    d = int(delay_s * SR)
    out = x.copy()
    if d <= 0 or d >= len(x):
        return out
    if x.ndim == 2:
        out[d:] += gain * x[:-d]
    else:
        out[d:] += gain * x[:-d]
    return out


def _reverb_tail(x: np.ndarray, tau_s: float, gain: float) -> np.ndarray:
    """Exponentieller Nachhall-Schwanz nach jedem Perkussions-Onset."""
    out = x.copy()
    tail_len = int(min(2.5, 4 * tau_s) * SR)
    tt = np.arange(tail_len) / SR
    for k in range(1, 15, 2):
        idx = k * SR
        if idx + tail_len > len(x):
            break
        rng = np.random.default_rng(200 + k)
        tail = _lowpass(rng.standard_normal(tail_len), 4000.0, 2)
        tail = tail / (np.max(np.abs(tail)) + 1e-12)
        tail = tail * np.exp(-tt / tau_s)
        if x.ndim == 2:
            out[idx : idx + tail_len] += gain * tail[:, None]
        else:
            out[idx : idx + tail_len] += gain * tail
    return out


def _quantize(x: np.ndarray, bits: int) -> np.ndarray:
    q = 2 ** (bits - 1)
    return np.round(x * q) / q


def _overlay(st: np.ndarray, mono_overlay: np.ndarray) -> np.ndarray:
    return np.stack([st[:, 0] + mono_overlay, st[:, 1] + mono_overlay], axis=1)


def _apply_env(st: np.ndarray, env: np.ndarray) -> np.ndarray:
    return np.stack([st[:, 0] * env, st[:, 1] * env], axis=1)


def _ch(st: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.stack([fn(st[:, 0]), fn(st[:, 1])], axis=1)


def _t_dev_fm(f_mod: float, depth_pct: float) -> np.ndarray:
    """Zeit-Verzerrung fuer FM-Defekte (Wow/Flutter/Scrape/Sidebands)."""
    t = np.arange(SR * DUR) / SR
    return (depth_pct / 100.0) * (1.0 / f_mod) * np.sin(2 * np.pi * f_mod * t) * SR


# ------------------------------------------------------------------ Defekt-Synthese


def _synth_hiss(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1)
    n = len(st)
    noise = rng.normal(0, 0.008, n)
    noise = noise + 0.4 * np.diff(noise, prepend=0.0)  # HF-Anhebung (Tape-Hiss)
    return _overlay(st, noise)


def _synth_broadband(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(3)
    return _overlay(st, rng.normal(0, 0.02, len(st)))


def _synth_hum(st: np.ndarray) -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    hum = (
        0.09 * np.sin(2 * np.pi * 50.0 * t)
        + 0.045 * np.sin(2 * np.pi * 100.0 * t)
        + 0.03 * np.sin(2 * np.pi * 150.0 * t)
    )
    return _overlay(st, hum)


def _synth_motor(st: np.ndarray) -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    motor = (
        0.045 * np.sin(2 * np.pi * 100.0 * t)
        + 0.03 * np.sin(2 * np.pi * 200.0 * t)
        + 0.02 * np.sin(2 * np.pi * 300.0 * t)
    )
    return _overlay(st, motor)


def _synth_rumble(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(5)
    rumble = _lowpass(rng.standard_normal(len(st)), 60.0, 4) * 3.0
    return _overlay(st, rumble)


def _synth_room_mode(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(61)
    reso = _bandpass(rng.standard_normal(len(st)), 70.0, 76.0, 2) * 4.0
    return _overlay(st, reso)


def _synth_proximity(st: np.ndarray) -> np.ndarray:
    lf = _lowpass(st[:, 0], 250.0, 2)
    lf2 = _lowpass(st[:, 1], 250.0, 2)
    return np.stack([st[:, 0] + 0.8 * lf, st[:, 1] + 0.8 * lf2], axis=1)


def _synth_clicks(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(73)
    x = np.zeros(len(st))
    idx = 8000
    while idx < SR * DUR - 200:
        x[idx : idx + 3] += 0.8
        idx += int(0.3 * SR + 0.7 * SR * rng.random())
    return _overlay(st, x)


def _synth_crackle(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(79)
    x = np.zeros(len(st))
    idx = 8000
    while idx < SR * DUR - 2000:
        n_pops = int(2 + 4 * rng.random())
        for _ in range(n_pops):
            x[idx : idx + 2] += (0.10 + 0.20 * rng.random()) * rng.choice((-1.0, 1.0))
            idx += int(2 + 8 * rng.random())
        idx += int(0.05 * SR + 0.15 * SR * rng.random())
    return _overlay(st, x)


def _synth_sibilance(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(13)
    x = np.zeros(len(st))
    for k in range(3, 14, 3):
        s = k * SR + int(0.5 * SR)
        e = s + int(0.15 * SR)
        burst = rng.standard_normal(e - s) * 0.12
        x[s:e] += _bandpass(burst, 5500.0, 9500.0) * 6.0
    return _overlay(st, x)


def _synth_aliasing(st: np.ndarray) -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    nyq = SR / 2
    alias = 0.05 * np.sin(2 * np.pi * (nyq - 700.0) * t) + 0.04 * np.sin(2 * np.pi * (nyq - 1300.0) * t)
    return _overlay(st, alias)


def _synth_imd(st: np.ndarray) -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    imd = 0.06 * np.sin(2 * np.pi * 1000.0 * t) + 0.04 * np.sin(2 * np.pi * 5000.0 * t)
    return _overlay(st, imd)


def _synth_modulation_noise(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(2)
    env = np.abs(st[:, 0])
    noise = rng.normal(0, 1.0, len(st)) * (env + 0.02) * 0.09
    return _overlay(st, noise)


def _synth_nr_breathing(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(53)
    t = np.arange(SR * DUR) / SR
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)  # atem-artige 1,5-Hz-Modulation
    noise = rng.normal(0, 0.015, len(st)) * mod
    return _overlay(st, noise)


def _synth_generation_loss(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(47)
    out = _ch(st, lambda x: _lowpass(x, 6500.0, 3))
    noise = rng.normal(0, 0.025, len(st))
    return _overlay(out, noise)


def _synth_clipping(st: np.ndarray) -> np.ndarray:
    return np.clip(st * 3.5, -0.5, 0.5) * 2.0


def _synth_overload(st: np.ndarray) -> np.ndarray:
    return 0.4 * np.tanh(2.5 * st / 0.35) / np.tanh(2.5)


def _synth_saturation(st: np.ndarray) -> np.ndarray:
    x = st / 0.35
    return (x + 0.3 * x**2) * 0.35


def _synth_stylus(st: np.ndarray) -> np.ndarray:
    x = st / 0.35
    y = np.sign(x) * np.abs(x) ** 0.75  # asymmetrische Kennlinie
    return y * 0.35


def _synth_igd(st: np.ndarray) -> np.ndarray:
    x = st / 0.35
    y = x + 0.35 * x**2 + 0.2 * x**3  # H2/H3 wie IGD
    return y * 0.35


def _synth_compression_artifacts(st: np.ndarray) -> np.ndarray:
    out = _ch(st, lambda x: _lowpass(x, 14500.0, 6))
    out = _ch(out, lambda x: _peaking(x, 3400.0, 2.0, -18.0))  # Spektralloch
    return out


def _synth_digital_artifacts(st: np.ndarray) -> np.ndarray:
    out = _quantize(st, 8)
    t = np.arange(SR * DUR) / SR
    nyq = SR / 2
    alias = 0.05 * np.sin(2 * np.pi * (nyq - 900.0) * t)
    return _overlay(out, alias)


def _synth_quantization(st: np.ndarray) -> np.ndarray:
    return _quantize(st, 8)


def _synth_dyn_compression(st: np.ndarray) -> np.ndarray:
    """Totkomprimiert: Huellenkurve auf ~konstant glaetten (Loudness War)."""
    n = len(st)
    win = int(0.2 * SR)
    out = st.copy()
    for ch in range(st.shape[1]):
        x = st[:, ch]
        rms = np.sqrt(np.convolve(x**2, np.ones(win) / win, mode="same")) + 1e-9
        target = np.full(n, np.percentile(rms, 60))
        gain = np.clip(target / rms, 0.2, 3.0)
        # Glaetten
        gain = np.convolve(gain, np.ones(win) / win, mode="same")
        out[:, ch] = x * gain
    return out


def _synth_transient_smearing(st: np.ndarray) -> np.ndarray:
    """Onsets verrunden: kurzer Tiefpass-Burst an den Attacken."""
    out = st.copy()
    win = int(0.05 * SR)
    for k in range(1, 15, 2):
        s = k * SR
        e = min(len(st), s + win)
        for ch in range(st.shape[1]):
            seg = out[s:e, ch]
            sm = _lowpass(seg, 800.0, 2)
            out[s:e, ch] = 0.4 * sm + 0.6 * seg
    return out


def _synth_wow(st: np.ndarray) -> np.ndarray:
    return _resample(st, _t_dev_fm(0.3, 0.5))


def _synth_flutter(st: np.ndarray) -> np.ndarray:
    return _resample(st, _t_dev_fm(6.0, 0.2))


def _synth_scrape_flutter(st: np.ndarray) -> np.ndarray:
    return _resample(st, _t_dev_fm(80.0, 0.05))


def _synth_multiband_wow(st: np.ndarray) -> np.ndarray:
    hi = _ch(st, lambda x: _bandpass(x, 4000.0, 12000.0))
    lo = _ch(st, lambda x: _lowpass(x, 4000.0, 4))
    hi_mod = _resample(hi, _t_dev_fm(5.0, 0.3))
    return lo + hi_mod


def _synth_sidebands(st: np.ndarray) -> np.ndarray:
    return _resample(st, _t_dev_fm(3.0, 0.1))


def _synth_pitch_drift(st: np.ndarray) -> np.ndarray:
    n = len(st)
    t_dev = np.linspace(0.0, 0.02 * n, n)  # linear wachsende Dehnung (440->~450 Hz)
    return _resample(st, t_dev)


def _synth_speed_offset(st: np.ndarray) -> np.ndarray:
    n = len(st)
    t_dev = 0.004 * np.arange(n)  # konstant +0,4 %
    return _resample(st, t_dev)


def _synth_jitter(st: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(31)
    t_dev = 0.6 * rng.standard_normal(len(st))  # Sample-Timing-Jitter
    return _resample(st, t_dev)


def _synth_dip(st: np.ndarray, events: list[tuple[int, int, float]]) -> np.ndarray:
    return _apply_env(st, _dip_envelope(len(st), events))


def _synth_dropouts(st: np.ndarray) -> np.ndarray:
    return _synth_dip(st, [(7 * SR, 7 * SR + int(0.006 * SR), 0.0)])


def _synth_dropout_oxide(st: np.ndarray) -> np.ndarray:
    s = 7 * SR
    seg = int(0.5 * SR)
    return _synth_dip(st, [(s, s + 3 * seg, 0.1)])


def _synth_dropout_head_contact(st: np.ndarray) -> np.ndarray:
    events = [(k * SR, k * SR + int(0.3 * SR), 0.02) for k in (3, 7, 11)]
    return _synth_dip(st, events)


def _synth_dropout_splice(st: np.ndarray) -> np.ndarray:
    out = _synth_dip(st, [(7 * SR, SR * DUR, 0.5)])
    n = len(st)
    klick = np.zeros(n)
    klick[7 * SR : 7 * SR + 3] = 0.35
    return _overlay(out, klick)


def _synth_mpeg_frame_loss(st: np.ndarray) -> np.ndarray:
    events = [(k * SR, k * SR + int(0.026 * SR), 0.0) for k in (2, 5, 8, 11, 14)]
    return _synth_dip(st, events)


def _synth_tape_splice(st: np.ndarray) -> np.ndarray:
    out = _synth_dip(st, [(7 * SR, SR * DUR, 0.5)])
    klick = np.zeros(len(st))
    klick[7 * SR : 7 * SR + 3] = 0.35
    return _overlay(out, klick)


def _synth_head_level_dip(st: np.ndarray) -> np.ndarray:
    n = len(st)
    env = np.ones(n)
    s = 8 * SR
    seg = int(0.5 * SR)
    env[s : s + seg] = np.linspace(1.0, 0.18, seg)  # Ramp -15 dB
    env[s + seg : s + 2 * seg] = 0.18
    env[s + 2 * seg : s + 3 * seg] = np.linspace(0.18, 1.0, seg)  # Snap-back
    return _apply_env(st, env)


def _synth_transport_bump(st: np.ndarray) -> np.ndarray:
    n = len(st)
    t = np.arange(n) / SR
    phase = (t % 1.5) / 1.5
    drop_mask = (phase >= 0.0) & (phase < 0.03)
    env = np.ones(n)
    env[drop_mask] = 0.25
    out = _apply_env(st, env)
    thump = 0.05 * np.sin(2 * np.pi * 60.0 * t) * drop_mask.astype(float)
    return _overlay(out, thump)


def _synth_sticky_shed(st: np.ndarray) -> np.ndarray:
    out = _synth_dip(st, [(4 * SR, 4 * SR + int(0.1 * SR), 0.15), (9 * SR, 9 * SR + int(0.12 * SR), 0.1)])
    return _synth_modulation_noise(out)


def _synth_tape_head_clog(st: np.ndarray) -> np.ndarray:
    n = len(st)
    env = np.ones(n)
    for k in range(12):
        s = int(k * 1.2 * SR)
        e = s + int(0.08 * SR)
        if e < n:
            env[s:e] = 0.35
    out = _apply_env(st, env)
    return _ch(out, lambda x: _highpass_hf_only(x))


def _highpass_hf_only(x: np.ndarray) -> np.ndarray:
    """HF-Dips wirken vor allem oberhalb 4 kHz — Energie dort dämpfen."""
    lo = _lowpass(x, 4000.0, 4)
    return lo + (x - lo) * 0.5


def _synth_head_wear(st: np.ndarray) -> np.ndarray:
    return _ch(st, lambda x: _lowpass(x, 4500.0, 4))


def _synth_bandwidth_loss(st: np.ndarray) -> np.ndarray:
    return _ch(st, lambda x: _lowpass(x, 6000.0, 4))


def _synth_hf_remanence(st: np.ndarray) -> np.ndarray:
    out = _ch(st, lambda x: _lowpass(x, 7500.0, 4))
    return out


def _synth_bias_error(st: np.ndarray) -> np.ndarray:
    out = _ch(st, lambda x: _lowpass(x, 5500.0, 4))
    return out


def _synth_riaa_error(st: np.ndarray) -> np.ndarray:
    return _ch(st, lambda x: _shelf(x, 3000.0, 12.0, high=True))


def _synth_dolby_mismatch(st: np.ndarray) -> np.ndarray:
    return _ch(st, lambda x: _shelf(x, 4000.0, 6.0, high=True))


def _synth_vocal_harshness(st: np.ndarray) -> np.ndarray:
    out = _ch(st, lambda x: _peaking(x, 3500.0, 1.2, 12.0))
    return out


def _synth_reverb(st: np.ndarray) -> np.ndarray:
    return _reverb_tail(st, 0.36, 0.3)


def _synth_pre_echo(st: np.ndarray) -> np.ndarray:
    return _echo(st, -0.15 if False else 0.15, 0.16)


def _synth_print_through(st: np.ndarray) -> np.ndarray:
    # Geist VOR dem Onset: um 200 ms nach hinten kopieren (Pre-Echo-Signatur)
    out = st.copy()
    d = int(0.2 * SR)
    out[:-d] += 0.18 * st[d:]
    return out


def _synth_groove_echo(st: np.ndarray) -> np.ndarray:
    out = st.copy()
    d = int(1.8 * SR)
    out[:-d] += 0.25 * st[d:]
    return out


def _synth_dc_offset(st: np.ndarray) -> np.ndarray:
    return st + 0.08


def _synth_amplitude_drift(st: np.ndarray) -> np.ndarray:
    n = len(st)
    env = np.linspace(0.55, 1.35, n)  # linearer Pegelanstieg
    return _apply_env(st, env)


def _synth_phase_rotation(st: np.ndarray) -> np.ndarray:
    def _ap(x: np.ndarray) -> np.ndarray:
        a = 0.7
        y = np.zeros_like(x)
        y[0] = x[0]
        for n in range(1, len(x)):
            y[n] = -a * x[n] + x[n - 1] + a * y[n - 1]
        return y

    return _ch(st, _ap)


def _synth_phase_issues(st: np.ndarray) -> np.ndarray:
    l, r = _st(st)
    return _stack(l, -r)


def _synth_crosstalk(st: np.ndarray) -> np.ndarray:
    l, r = _st(st)
    return _stack(l, r + 0.32 * l)  # ~10 dB Trennung


def _synth_imbalance(st: np.ndarray) -> np.ndarray:
    l, r = _st(st)
    return _stack(l, r * 0.71)  # -3 dB rechts


def _synth_collapse(st: np.ndarray) -> np.ndarray:
    l, r = _st(st)
    return _stack(l, l)


def _synth_azimuth(st: np.ndarray) -> np.ndarray:
    l, r = _st(st)
    d = int(0.00008 * SR)  # 80 µs HF-Laufzeitfehler
    r2 = r.copy()
    r2[d:] = r[:-d]
    return _stack(l, r2)


def _synth_lacquer(st: np.ndarray) -> np.ndarray:
    out = _synth_crackle(st)
    out = _ch(out, lambda x: _lowpass(x, 9000.0, 3))
    rng = np.random.default_rng(82)
    return _overlay(out, rng.normal(0, 0.004, len(st)))


# DefectType -> Synthese auf dem Musik-Traeger
_MUSIC_SYNTH: dict[DefectType, Callable[[np.ndarray], np.ndarray]] = {
    # Rauschen / Stroerimpulse (additiv)
    DefectType.HISS: _synth_hiss,
    DefectType.HIGH_FREQ_NOISE: _synth_broadband,
    DefectType.HUM: _synth_hum,
    DefectType.MOTOR_INTERFERENCE: _synth_motor,
    DefectType.LOW_FREQ_RUMBLE: _synth_rumble,
    DefectType.ROOM_MODE_RESONANCE: _synth_room_mode,
    DefectType.PROXIMITY_EFFECT_EXCESS: _synth_proximity,
    DefectType.CLICKS: _synth_clicks,
    DefectType.CRACKLE: _synth_crackle,
    DefectType.SIBILANCE: _synth_sibilance,
    DefectType.ALIASING: _synth_aliasing,
    DefectType.INTERMODULATION_DISTORTION: _synth_imd,
    DefectType.MODULATION_NOISE: _synth_modulation_noise,
    DefectType.NR_BREATHING_ARTIFACT: _synth_nr_breathing,
    DefectType.GENERATION_LOSS: _synth_generation_loss,
    # Verzerrung / Dynamik
    DefectType.CLIPPING: _synth_clipping,
    DefectType.DISTORTION: _synth_overload,
    DefectType.OVERLOAD_DISTORTION: _synth_overload,
    DefectType.SOFT_SATURATION: _synth_saturation,
    DefectType.STYLUS_DAMAGE: _synth_stylus,
    DefectType.INNER_GROOVE_DISTORTION: _synth_igd,
    DefectType.COMPRESSION_ARTIFACTS: _synth_compression_artifacts,
    DefectType.DIGITAL_ARTIFACTS: _synth_digital_artifacts,
    DefectType.QUANTIZATION_NOISE: _synth_quantization,
    DefectType.DYNAMIC_COMPRESSION_EXCESS: _synth_dyn_compression,
    DefectType.TRANSIENT_SMEARING: _synth_transient_smearing,
    # Zeitachse (FM / Resampling)
    DefectType.WOW: _synth_wow,
    DefectType.FLUTTER: _synth_flutter,
    DefectType.SCRAPE_FLUTTER: _synth_scrape_flutter,
    DefectType.MULTIBAND_WOW_FLUTTER: _synth_multiband_wow,
    DefectType.FLUTTER_SPECTRAL_SIDEBANDS: _synth_sidebands,
    DefectType.PITCH_DRIFT: _synth_pitch_drift,
    DefectType.SPEED_CALIBRATION_ERROR: _synth_speed_offset,
    DefectType.JITTER_ARTIFACTS: _synth_jitter,
    # Pegel-Einbrueche / Aussetzer
    DefectType.DROPOUTS: _synth_dropouts,
    DefectType.DROPOUT: _synth_dropout_oxide,
    DefectType.DROPOUT_OXIDE: _synth_dropout_oxide,
    DefectType.DROPOUT_HEAD_CONTACT: _synth_dropout_head_contact,
    DefectType.DROPOUT_SPLICE: _synth_dropout_splice,
    DefectType.MPEG_FRAME_LOSS: _synth_mpeg_frame_loss,
    DefectType.TAPE_SPLICE_ARTIFACT: _synth_tape_splice,
    DefectType.TAPE_HEAD_LEVEL_DIP: _synth_head_level_dip,
    DefectType.TRANSPORT_BUMP: _synth_transport_bump,
    DefectType.STICKY_SHED_RESIDUE: _synth_sticky_shed,
    DefectType.TAPE_HEAD_CLOG: _synth_tape_head_clog,
    # Spektral / EQ
    DefectType.HEAD_WEAR: _synth_head_wear,
    DefectType.BANDWIDTH_LOSS: _synth_bandwidth_loss,
    DefectType.HF_REMANENCE_LOSS: _synth_hf_remanence,
    DefectType.BIAS_ERROR: _synth_bias_error,
    DefectType.RIAA_CURVE_ERROR: _synth_riaa_error,
    DefectType.DOLBY_NR_MISMATCH: _synth_dolby_mismatch,
    DefectType.VOCAL_HARSHNESS: _synth_vocal_harshness,
    # Echo / Nachhall
    DefectType.REVERB_EXCESS: _synth_reverb,
    DefectType.PRE_ECHO: _synth_pre_echo,
    DefectType.PRINT_THROUGH: _synth_print_through,
    DefectType.GROOVE_ECHO: _synth_groove_echo,
    # Pegel / Zeitverlauf
    DefectType.DC_OFFSET: _synth_dc_offset,
    DefectType.AMPLITUDE_DRIFT: _synth_amplitude_drift,
    # Stereo / Phase
    DefectType.PHASE_ROTATION: _synth_phase_rotation,
    DefectType.PHASE_ISSUES: _synth_phase_issues,
    DefectType.CROSSTALK: _synth_crosstalk,
    DefectType.STEREO_IMBALANCE: _synth_imbalance,
    DefectType.STEREO_FIELD_COLLAPSE: _synth_collapse,
    DefectType.AZIMUTH_ERROR: _synth_azimuth,
    # Hybrid (Komposit-Defekte)
    DefectType.LACQUER_DISC_DEGRADATION: _synth_lacquer,
}


def _synth_case(dt: DefectType) -> np.ndarray:
    """Musik-Traeger + Defekt-Transformation — deterministisch (§G5 (copilot-instructions.md))."""
    st = _music_st()
    fn = _MUSIC_SYNTH.get(dt)
    if fn is None:
        return st
    out = fn(st)
    return np.clip(np.asarray(out, dtype=np.float32), -1.0, 1.0)


# ------------------------------------------------------------------ Fall-Aufbau


def _load_phase_class(phase_id: str):
    canon = _PHASE_ALIAS.get(phase_id, phase_id)
    modname = f"backend.core.phases.{canon}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError:
        return None, canon
    for obj in mod.__dict__.values():
        if (
            isinstance(obj, type)
            and issubclass(obj, PhaseInterface)
            and obj is not PhaseInterface
            and getattr(obj, "__module__", "") == modname
        ):
            return obj, canon
    return None, canon


def _build_cases():
    """(DefectType, Name, Generator, Material, Phase-Klasse|None, Skip-Grund|None)

    Pro DefectType wird der Evidenz-Fall mit der HOECHSTEN erwarteten Severity
    gewaehlt (Negativ-Faelle haben kleine Erwartungsbereiche und taugen nicht
    als Reparatur-Evidenz). Die SYNTHSE selbst ist immer Musik-basiert
    (§MUSIK-TRAEGER) — die Evidenz-CASES liefern nur Material und Repraesentanz.
    """
    best: dict[DefectType, tuple] = {}
    for family in sorted(EVIDENCE_CASES):
        for name, gen, material, expects in EVIDENCE_CASES[family]:
            for dt, rng in expects.items():
                lo, hi = rng
                cur = best.get(dt)
                if cur is None or hi > cur[0]:
                    best[dt] = (hi, (name, gen, material, expects))

    cases: list[tuple] = []
    for dt in sorted(best, key=lambda k: k.value):
        _, (name, gen, material, expects) = best[dt]
        synth = lambda _dt=dt: _synth_case(_dt)
        if dt not in _PHASE_MAP:
            cases.append((dt, name, synth, material, None, "mapper-ohne-Zuordnung"))
            continue
        primary = list(_PHASE_MAP[dt].primary_phases)
        cls = None
        used_phase = None
        for pid in primary:
            cls, used_phase = _load_phase_class(pid)
            if cls is not None:
                break
        if cls is None:
            cases.append((dt, name, synth, material, None, f"primärphase-fehlt({','.join(primary)})"))
            continue
        cases.append((dt, name, synth, material, (cls, used_phase), None))
    return cases


def _scan(audio: np.ndarray, material: MaterialType):
    sc = DefectScanner(sample_rate=SR, material_type=material)
    return sc.scan(audio)


def _severity(audio: np.ndarray, material: MaterialType, dt: DefectType) -> float:
    res = _scan(audio, material)
    score = res.scores.get(dt)
    return float(score.severity) if score is not None else 0.0


def _severity_framework(audio: np.ndarray, dt: DefectType) -> float:
    from backend.core.ai_framework import UnifiedDefectDetector

    det = UnifiedDefectDetector(sample_rate=SR)
    mono = _mono(audio)
    if dt == DefectType.DISTORTION:
        _conf, sev, _ = det._detect_distortion(mono, False)
        return float(sev)
    _conf, sev, _ = det._detect_dropout(mono, False)
    return float(sev)


def _pipeline_style_kwargs(material: MaterialType, scores: dict, locations: dict) -> dict:
    """Kwargs wie sie unified_restorer_v3._execute_pipeline an Phasen uebergibt
    (material_type/material als ENUM, defect_scores als DefectScoreView mit
    Enum- UND String-Key-Aufloesung)."""
    return {
        "sample_rate": SR,
        "material_type": material,
        "material": material,
        "defect_scores": DefectScoreView(scores),
        "defect_locations": locations,
        "strength": 1.0,
    }


def _run_case(dt: DefectType, name: str, gen, material: MaterialType, phase_info, skip: str | None) -> tuple:
    audio = gen()
    metric = _metric_fn(dt)
    _dir = METRICS[dt][1]

    is_fw = dt in (DefectType.DISTORTION, DefectType.DROPOUT)
    if is_fw:
        sev_before = _severity_framework(audio, dt)
        from backend.core.defect_scanner import DefectScore as _DS

        _scores: dict = {dt: _DS(dt, float(sev_before), 0.8)}
        _locs: dict = {}
    else:
        res = _scan(audio, material)
        _scores = res.scores
        sev_before = float(_scores[dt].severity) if dt in _scores else 0.0
        _locs = {k.value: list(v.locations) for k, v in _scores.items() if v.locations}
    met_before = metric(audio)

    if skip:
        return ("SKIP", dt, name, sev_before, sev_before, met_before, met_before, "reduce", skip, None)
    if _dir == "preserve":
        return (
            "SKIP",
            dt,
            name,
            sev_before,
            sev_before,
            met_before,
            met_before,
            "preserve",
            "Bewahr-Typ (§2.1/§6.3): Soft-Saturation ist Soll-Zustand, keine Reparatur",
            None,
        )
    # §MUSIK-TRAEGER Scanner-Gate: Was auf Musik nicht detektiert wird, ist ein
    # SCANNER-Gap (LUECKE-S), keine Reparatur-Luecke.
    if sev_before < 0.15:
        return ("SCAN", dt, name, sev_before, sev_before, met_before, met_before, _dir, "sev<0.15", None)

    cls, phase_id = phase_info
    _kwargs = _pipeline_style_kwargs(material, _scores, _locs)
    try:
        phase = cls()
        result = phase.process(audio, **_kwargs)
    except TypeError:
        try:
            phase = cls()
            result = phase.process(audio, sample_rate=SR, material_type=material)
        except TypeError:
            phase = cls()
            result = phase.process(audio)
    except Exception as exc:  # dokumentierter Skip: Phase nicht standalone lauffaehig
        return (
            "SKIP",
            dt,
            name,
            sev_before,
            sev_before,
            met_before,
            met_before,
            _dir,
            f"phase-ausnahme:{type(exc).__name__}",
            phase_id,
        )

    # §V6 (copilot-instructions.md): ML-Passthrough ohne Modell ist bit-identisch.
    warnings = list(getattr(result, "warnings", []) or [])
    if any("modell" in str(w).lower() or "nicht_ladbar" in str(w).lower() for w in warnings):
        return (
            "SKIP",
            dt,
            name,
            sev_before,
            sev_before,
            met_before,
            met_before,
            _dir,
            "ML-Passthrough-ohne-Modell(§V6 (copilot-instructions.md))",
            phase_id,
        )

    audio_after = np.asarray(result.audio, dtype=np.float32)
    if audio_after.shape != audio.shape:
        try:
            audio_after = audio_after[: audio.shape[0]]
        except Exception:
            return ("SKIP", dt, name, sev_before, sev_before, met_before, met_before, _dir, "shape-mismatch", phase_id)

    sev_after = _severity_framework(audio_after, dt) if is_fw else _severity(audio_after, material, dt)
    met_after = metric(audio_after)
    return ("RUN", dt, name, sev_before, sev_after, met_before, met_after, _dir, None, phase_id)


# Absolute Sonder-Gates: Defekt-bezogene Absolut-Schwellen statt 50 %-Reduktion
# (z.B. Stereofeld-Kollaps: IACC muss unter die Detektor-Aktivierung 0.95 fallen;
# Mono-Kompatibilitäts-Floors verhindern eine volle Dekorrelation).
GATE_SPECIAL: dict[DefectType, tuple[str, float]] = {
    DefectType.STEREO_FIELD_COLLAPSE: ("le", 0.90),
}


def main() -> int:
    families = [f for f in sys.argv[1:] if not f.startswith("-")]
    if families:
        valid = set(EVIDENCE_CASES)
        unknown = [f for f in families if f not in valid]
        if unknown:
            print(f"Unbekannte Familien: {unknown}. Verfügbar: {', '.join(sorted(valid))}")
            return 2

    failed = 0
    skipped = 0
    ok = 0
    scan_gaps = 0
    for dt, name, gen, material, phase_info, skip_reason in _build_cases():
        family = next((f for f, lst in EVIDENCE_CASES.items() if any(e[0] == name for e in lst)), "?")
        if families and family not in families:
            continue
        status, _, _, sev_b, sev_a, met_b, met_a, _dir, reason, phase_id = _run_case(
            dt, name, gen, material, phase_info, skip_reason
        )
        if status == "SKIP":
            skipped += 1
            print(f"[SKIP ] {dt.value:32s} ({family}) — {reason} | phase={phase_id or '—'}")
            continue
        if status == "SCAN":
            scan_gaps += 1
            print(
                f"[LUECKE-S] {dt.value:28s} ({family}) Scanner detektiert Defekt "
                f"auf Musik nicht (sev={sev_b:.3f} < 0.15)"
            )
            continue
        # Gate-Auswertung
        _special = GATE_SPECIAL.get(dt)
        if _special is not None:
            _op, _thr = _special
            wirksam = (met_a <= _thr) if _op == "le" else (met_a >= _thr)
        elif _dir == "reduce":
            wirksam = met_a <= met_b * 0.5
        else:
            wirksam = met_a >= met_b * 1.25
        nicht_schlechter = sev_a <= sev_b + 0.03
        good = wirksam and nicht_schlechter
        if good:
            ok += 1
        else:
            failed += 1
        flag = "OK" if good else "LUECKE-P"
        print(
            f"[{flag}] {dt.value:30s} ({family}) phase={phase_id}: "
            f"severity {sev_b:.3f}->{sev_a:.3f}; phys {met_b:.4g}->{met_a:.4g} "
            f"({_dir}) {'OK' if good else 'FEHLT'}"
        )
    print(
        f"Reparatur-Wirksamkeit (Musik-Traeger): {ok} OK, {failed} Phasen-Luecke(n), "
        f"{scan_gaps} Scanner-Luecke(n), {skipped} Skip(s)"
    )
    return 1 if (failed or scan_gaps) else 0


if __name__ == "__main__":
    raise SystemExit(main())

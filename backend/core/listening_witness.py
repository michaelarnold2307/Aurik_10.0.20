"""backend/core/listening_witness.py — Per-Phase-Reinhör-Witness (2026-09-11).

Problem: Die Restaurierungs-Metriken (PMGG/HPI/PQS) bewerteten Läufe als
„hervorragend", die beim Reinhören katastrophal klangen — kontinuierliche
Pitch-Abweichungen, verzerrter Gesang, Lautstärke-Pumpen. Diese Defektklassen
werden von den bestehenden Gates nicht abgebildet (PMGG-Deltas waren ~0.0000).

Lösung (Hörordnung §8a: „Metriken sind Zeugen, die Hör-Instanz entscheidet"):
Nach JEDER abgeschlossenen Phase „hört" Aurik deterministisch in das
Phase-Delta hinein und meldet hör-relevante Regressionen — REPORT-ONLY
(kein Hard-Gate; die Hör-Instanz entscheidet, Entwicklung nutzt die Befunde):

  1. Pitch-Instabilität — Median-|ΔF0| (Cent) über stimmhafte Frames +
     Modulations-Tiefe der F0-Trajektorie (Vibrato-/Wow-Artefakt-Proxy).
  2. Stimm-Verzerrung — HNR-Abfall (de Krom, dB) + Spektral-Flatness-Anstieg
     im Stimmband 2–8 kHz (Rauigkeit/Musical-Noise-Proxy).
  3. Lautstärke-Pumpen — Modulations-Tiefe der Kurzzeit-Lautheit
     (STL 100 ms, Modulationsband 0.2–6 Hz, dB) vorher/nachher.

JND-Referenzen: F0-JND ~5–10 Cent (Töne, 1 kHz); HNR/Harmonizität als
Stimmqualitäts-Proxy (de Krom 1993); Lautheits-Modulation ab ~1 dB
Mittenband hörbar (Moore 2012).

Determinismus (§G5 (GEBOTE.md)): reine numpy/scipy-Operationen, keine ML-Abhängigkeit,
keine Zufallsgrößen. Laufzeit-Ziel: ≤ 2 s pro 30-s-Chunk.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── JND-Schwellen (Report-only; Literaturwerte s. Docstring) ─────────────────
_PITCH_DRIFT_CENTS = 15.0  # Median-|ΔF0| über stimmhafte Frames
_PITCH_MOD_DEPTH_CENTS = 25.0  # Modulations-Tiefe der F0-Trajektorie
_HNR_DROP_DB = 2.0  # Stimm-HNR-Abfall (de Krom)
_HF_FLATNESS_RISE = 0.15  # Spektral-Flatness-Anstieg 2–8 kHz
_FLAT_TOP_RISE = 0.005  # Anteil ±1-gepinnter Samples (Clipping-Proxy)
_LOUD_MOD_RISE_DB = 1.5  # STL-Modulations-Tiefe-Anstieg (0.2–6 Hz)
_BASS_DROP_DB = 1.5  # Relativer Bass-Energie-Anteil-Verlust (20–250 Hz)
_TRANSIENT_SMEAR_RATIO = 0.35  # Relativer Abfall der 95-Perzentil-Hüllkurven-Steigung

_FRAME = 2048
_HOP = 1024
_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 400.0
_VOICED_CORR = 0.7
_VOCAL_LO_HZ = 200.0
_VOCAL_HI_HZ = 4000.0
_HF_LO_HZ = 2000.0
_HF_HI_HZ = 8000.0
_STL_WIN = 0.100
_STL_HOP = 0.050
_MOD_LO_HZ = 0.2
_MOD_HI_HZ = 6.0
_BASS_LO_HZ = 20.0
_BASS_HI_HZ = 250.0
_AIR_LO_HZ = 8000.0
_AIR_HI_HZ = 20000.0
_AIR_LOSS_DB = 2.0
_ROUGHNESS_RISE_ASPER = 0.35
# SUP-F6 (2026-09-16): Der Rauigkeits-Schätzer (roughness_model) ist eine
# RELATIVE Skala (Modulationstiefe, ~10⁴ auf realer Musik) — eine absolute
# 0,35-Schwelle feuerte deshalb bei JEDER Hüllkurven-Änderung (~0-Delta-Phasen:
# +1,16 bei harmlosem 30-Hz-Hochpass auf dem Test-Track-Export). Unter ~35 %
# relativem Anstieg (2× Vassilakis-JND ≈ 17 %) wird auf 0 geklemmt — Finding
# UND Veto-Schwelle (witness_correction_loop) erben EINE Wahrheitsquelle.
_ROUGHNESS_RISE_REL = 0.35
_PRE_ECHO_DB = -12.0
# §Residual-Defekt-Zeugen (2026-09-13): Gedämpfter Gesang + Rest-Verzerrung.
# Klarheitsband des Gesangs (2–6 kHz) — Dämpfung dort = gedämpfter Gesang.
_CLARITY_LO_HZ = 2000.0
_CLARITY_HI_HZ = 6000.0
_VOCAL_MUFFLED_DB = 2.5
# Rest-Clipping nach der zuständigen Phase: Anteil ±1 gepinnter Samples.
_FLAT_TOP_RESIDUAL = 0.0005
_DEFECT_OWNER_PHASES = {
    "phase_07_declip_repair",
    "phase_09_crackle_removal",
    "phase_15_stereo_balance",
    "phase_19_de_esser",
}
_TRANSIENT_WIN = 0.005  # 5-ms-Envelope für Transienten-Steigung


@dataclass
class ListeningWitnessResult:
    """Per-Phase-Hörbefund (Report-only, kein Gate)."""

    phase_id: str
    pitch_drift_cents: float = 0.0
    pitch_mod_depth_cents: float = 0.0
    hnr_drop_db: float = 0.0
    hf_flatness_rise: float = 0.0
    flat_top_rise: float = 0.0
    loud_mod_rise_db: float = 0.0
    bass_drop_db: float = 0.0
    transient_smear_ratio: float = 0.0
    air_gain_db: float = 0.0
    masked_residual_db: float = 0.0
    air_audible: bool = False
    roughness_rise_asper: float = 0.0
    itd_drift_us: float = 0.0
    ild_drift_db: float = 0.0
    iacc_drop: float = 0.0
    pre_echo_db: float = 0.0
    vocal_muffled_db: float = 0.0
    distortion_residual_flat_top: float = 0.0
    findings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "phase_id": self.phase_id,
            "pitch_drift_cents": round(self.pitch_drift_cents, 2),
            "pitch_mod_depth_cents": round(self.pitch_mod_depth_cents, 2),
            "hnr_drop_db": round(self.hnr_drop_db, 2),
            "hf_flatness_rise": round(self.hf_flatness_rise, 4),
            "flat_top_rise": round(self.flat_top_rise, 4),
            "loud_mod_rise_db": round(self.loud_mod_rise_db, 2),
            "bass_drop_db": round(self.bass_drop_db, 2),
            "transient_smear_ratio": round(self.transient_smear_ratio, 3),
            "air_gain_db": round(self.air_gain_db, 2),
            "masked_residual_db": round(self.masked_residual_db, 2),
            "air_audible": bool(self.air_audible),
            "roughness_rise_asper": round(self.roughness_rise_asper, 4),
            "itd_drift_us": round(self.itd_drift_us, 2),
            "ild_drift_db": round(self.ild_drift_db, 2),
            "iacc_drop": round(self.iacc_drop, 4),
            "pre_echo_db": round(self.pre_echo_db, 2),
            "findings": list(self.findings),
        }


def _mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        # Layout-sicher: kanonisch (N, C) nach §11 Spec 02; (C, N) ebenfalls bedienen.
        if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    _witness_ret: np.ndarray = arr
    return _witness_ret


def _channels_first_or_none(audio: np.ndarray) -> np.ndarray | None:
    """Stereo als channels-first (2, N) — None für Mono/unklare Layouts."""
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        return None
    if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        _witness_cf: np.ndarray = arr
        return _witness_cf
    if arr.shape[1] <= 2 and arr.shape[1] < arr.shape[0]:
        _witness_cf_t: np.ndarray = arr.T
        return _witness_cf_t
    return None


def _frame_iter(x: np.ndarray, sr: int, frame: int = _FRAME, hop: int = _HOP):
    for start in range(0, max(len(x) - frame + 1, 1), hop):
        yield x[start : start + frame]


def _frame_f0_hnr(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(f0_hz, voiced, hnr_db, hf_flatness) pro Frame — FFT-Autokorrelation + de-Krom-HNR.

    §V08/§10a-konform: Autokorrelation via FFT statt O(n·lags)-Lag-Loop.
    §PERF-R11 (2026-09-19): Batched FFT über alle Frames (sliding-window +
    axis=1-FFT) statt Python-Frame-Loop — je Frame bit-identisch (gleiche
    FFT-Kernels, gleiche Element-Operationen); Kurzsignale (< _FRAME)
    behalten den Einzel-Frame-Pfad.
    """
    if len(x) < _FRAME:
        return _frame_f0_hnr_loop(x, sr)
    return _frame_f0_hnr_batched(x, sr)


def _frame_f0_hnr_loop(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alter Per-Frame-Pfad (unverändert) — Fallback für Kurzsignale."""
    f0s: list[float] = []
    voiced: list[bool] = []
    hnrs: list[float] = []
    flats: list[float] = []
    nyq = sr / 2.0
    lag_lo = max(int(sr / _F0_MAX_HZ), 1)
    lag_hi = min(int(sr / _F0_MIN_HZ), len(x) - 1)
    for frame in _frame_iter(x, sr):
        if float(np.std(frame)) < 1e-6:
            f0s.append(0.0)
            voiced.append(False)
            hnrs.append(0.0)
            flats.append(0.0)
            continue
        frame = frame - frame.mean()
        e0 = float(np.dot(frame, frame))
        if e0 < 1e-12:
            f0s.append(0.0)
            voiced.append(False)
            hnrs.append(0.0)
            flats.append(0.0)
            continue
        n = len(frame)
        fft_len = 1
        while fft_len < 2 * n:
            fft_len <<= 1
        ac = np.fft.irfft(np.abs(np.fft.rfft(frame, n=fft_len)) ** 2, n=fft_len)[:n]
        ac = ac / max(float(ac[0]), 1e-12)
        lo = min(lag_lo, n - 2)
        hi = min(lag_hi, n - 2)
        if hi <= lo:
            f0s.append(0.0)
            voiced.append(False)
            hnrs.append(0.0)
            flats.append(0.0)
            continue
        seg = ac[lo : hi + 1]
        peak = float(np.max(seg))
        lag = lo + int(np.argmax(seg))
        f0 = float(sr) / float(max(lag, 1))
        is_voiced = peak > _VOICED_CORR
        f0s.append(f0 if is_voiced else 0.0)
        voiced.append(is_voiced)
        hnr = 10.0 * np.log10(max(peak / max(1.0 - peak, 1e-9), 1e-9)) if is_voiced else 0.0
        hnrs.append(float(np.clip(hnr, 0.0, 60.0)))
        # Spektral-Flatness im HF-Stimmband (Rauigkeit/Musical-Noise-Proxy)
        spec = np.abs(np.fft.rfft(frame * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        band = spec[(freqs >= _HF_LO_HZ) & (freqs <= min(_HF_HI_HZ, nyq))]
        if len(band) > 2:
            gm = float(np.exp(np.mean(np.log(band + 1e-12))))
            am = float(np.mean(band))
            flats.append(float(np.clip(gm / max(am, 1e-12), 0.0, 1.0)))
        else:
            flats.append(0.0)
    return (
        np.asarray(f0s, dtype=np.float64),
        np.asarray(voiced, dtype=bool),
        np.asarray(hnrs, dtype=np.float64),
        np.asarray(flats, dtype=np.float64),
    )


def _frame_f0_hnr_batched(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched F0/HNR/HF-Flatness — bit-identisch zu _frame_f0_hnr_loop je Frame."""
    n_total = len(x)
    # Frame-Auswahl wie _frame_iter: range(0, n_total - _FRAME + 1, _HOP)
    _sw = np.lib.stride_tricks.sliding_window_view(x, _FRAME)
    frames = np.ascontiguousarray(_sw[::_HOP])  # (n_frames, _FRAME) float32
    n_frames = frames.shape[0]
    nyq = sr / 2.0
    lag_lo = max(int(sr / _F0_MAX_HZ), 1)
    lag_hi = min(int(sr / _F0_MIN_HZ), n_total - 1)
    n = _FRAME

    out = (
        np.zeros(n_frames, dtype=np.float64),
        np.zeros(n_frames, dtype=bool),
        np.zeros(n_frames, dtype=np.float64),
        np.zeros(n_frames, dtype=np.float64),
    )
    f0s, voiced, hnrs, flats = out

    stds = frames.std(axis=1)
    silent = stds < 1e-6

    centered = frames - frames.mean(axis=1, keepdims=True)
    # e0-Gate: nach dem std-Gate kann e0 < 1e-12 nicht mehr greifen
    # (e0 = n·std² ≥ _FRAME·1e-12 = 2,05e-9) — Rechenweg wie der Loop.

    fft_len = 1
    while fft_len < 2 * n:
        fft_len <<= 1
    spec_ac = np.fft.rfft(centered, n=fft_len, axis=1)
    ac = np.fft.irfft(np.abs(spec_ac) ** 2, n=fft_len, axis=1)[:, :n]
    ac = ac / np.maximum(ac[:, 0], 1e-12)[:, None]

    lo = min(lag_lo, n - 2)
    hi = min(lag_hi, n - 2)
    if hi <= lo:
        return out
    seg = ac[:, lo : hi + 1]
    peak = seg.max(axis=1)
    lag = lo + np.argmax(seg, axis=1)
    f0 = sr / np.maximum(lag, 1).astype(np.float64)
    is_voiced = (peak > _VOICED_CORR) & ~silent
    hnr = np.zeros(n_frames, dtype=np.float64)
    _voiced_idx = np.flatnonzero(is_voiced)
    if _voiced_idx.size:
        _pk = peak[_voiced_idx]
        _hnr = 10.0 * np.log10(np.maximum(_pk / np.maximum(1.0 - _pk, 1e-9), 1e-9))
        hnr[_voiced_idx] = np.clip(_hnr, 0.0, 60.0)

    win_h = np.hanning(n).astype(np.float32)
    spec_f = np.abs(np.fft.rfft(frames * win_h, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band_mask = (freqs >= _HF_LO_HZ) & (freqs <= min(_HF_HI_HZ, nyq))
    band = spec_f[:, band_mask]
    flat = np.zeros(n_frames, dtype=np.float64)
    if band.shape[1] > 2:
        gm = np.exp(np.mean(np.log(band + 1e-12), axis=1))
        am = np.mean(band, axis=1)
        flat = np.clip(gm / np.maximum(am, 1e-12), 0.0, 1.0)

    f0s[:] = np.where(is_voiced, f0, 0.0)
    voiced[:] = is_voiced
    hnrs[:] = hnr
    flats[:] = flat
    return out


def _f0_metrics(f0s: np.ndarray, voiced: np.ndarray, hop_rate_hz: float) -> tuple[float, float]:
    """(Trajektorien-Spread in Cent, Modulations-Tiefe 3–8 Hz in Cent).

    §Witness-Fix (2026-09-11): Die Modulations-Tiefe wird pro ZUSAMMENHÄNGENDEM
    stimmhaftem Segment (Phrase) berechnet und über Phrasen gemittelt (Median) —
    die frühere Konkatenation aller stimmhaften Frames erzeugte an jeder
    Phrasen-Lücke einen künstlichen Cent-Sprung, dessen Spektral-Leckage das
    3–8-Hz-Band dominierte (False-Positives von >100 Cent auf unverändertem
    Signal nach minimalen Rand-Änderungen).
    """
    f0v = f0s[voiced]
    if len(f0v) < 3:
        return 0.0, 0.0
    cents = 1200.0 * np.log2(np.maximum(f0v, 1e-6) / np.median(f0v))
    spread = float(np.median(np.abs(cents - np.median(cents))))
    # Kontinuierliche stimmhafte Runs extrahieren (Phrasen-Grenzen nicht mischen).
    runs: list[np.ndarray] = []
    run_start: int | None = None
    for i in range(1, len(voiced) + 1):
        if i < len(voiced) and voiced[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                _seg = f0s[run_start:i]
                if len(_seg) >= 8:
                    _c = 1200.0 * np.log2(np.maximum(_seg, 1e-6) / np.median(_seg))
                    _c = _c - np.median(_c)
                    runs.append(_c * np.hanning(len(_c)))
                run_start = None
    mod_depths: list[float] = []
    for _c in runs:
        if len(_c) >= 16:
            _spec = np.abs(np.fft.rfft(_c))
            _fr = np.fft.rfftfreq(len(_c), 1.0 / max(hop_rate_hz, 1e-6))
            _band = _spec[(_fr >= 3.0) & (_fr <= 8.0)]
            if len(_band):
                mod_depths.append(float(np.sqrt(np.mean(_band**2)) * 2.0))
    mod_depth = float(np.median(mod_depths)) if mod_depths else 0.0
    return spread, float(mod_depth)


def _loudness_mod_depth_db(x: np.ndarray, sr: int) -> float:
    """Modulations-Tiefe (dB) der STL-Envelope im Band 0.2–6 Hz (Pumping-Proxy).

    §Witness-Fix (2026-09-11):
    - Nahezu stille Fenster (Fade-In/Fade-Out, Pausen) werden ausgeblendet —
      der Silence→Musik-Schritt am Dateirand erzeugte ein Breitband-Leakage,
      das das Modulationsband vollständig dominierte (False-Positives von
      >30 dB auf praktisch unverändertem Signal).
    - Linearer Trend wird entfernt + Hann-Fenster vor der FFT (Standard-
      Spektralschätzung) — unterdrückt Rand-Leakage der aktiven Region.
    """
    win = int(_STL_WIN * sr)
    hop = int(_STL_HOP * sr)
    if len(x) < win:
        return 0.0
    # §PERF-R11: Sliding-Window statt Python-Comprehension — je Fenster
    # bit-identisch (gleiche mean-Reduktion über dasselbe Fenster).
    _sw = np.lib.stride_tricks.sliding_window_view(x, win)
    _rms_win = _sw[::hop]
    rms = np.sqrt(np.mean(_rms_win**2, axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    db = db - np.median(db)
    # Stille-/Randfenster ausblenden: nur Fenster ≥ −50 dB unter dem Peak.
    _peak = float(np.max(db))
    _active = db > (_peak - 50.0)
    _active_idx = np.flatnonzero(_active)
    if len(_active_idx) >= 16:
        db = db[_active_idx[0] : _active_idx[-1] + 1]
    if len(db) < 8:
        return 0.0
    # Trend-Entfernung + Hann-Fenster (Rand-Leakage-Unterdrückung).
    _t = np.linspace(0.0, 1.0, len(db))
    _trend = np.polyfit(_t, db, 1)
    db = (db - np.polyval(_trend, _t)) * np.hanning(len(db))
    spec = np.abs(np.fft.rfft(db))
    freqs = np.fft.rfftfreq(len(db), 1.0 / (1.0 / _STL_HOP))
    band = spec[(freqs >= _MOD_LO_HZ) & (freqs <= _MOD_HI_HZ)]
    if len(band) == 0:
        return 0.0
    return float(np.sqrt(np.mean(band**2)) * 2.0)


_BAND_SPEC_CACHE: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
_BAND_SPEC_CACHE_MAX = 4


def _band_energy_ratio_db(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    """Relativer Energie-Anteil eines Frequenzbands an der Gesamtenergie (20 Hz–20 kHz).

    Ganzes Signal, ein Hann-gefenstertes FFT (statische Tonal-Balance-Proxies).
    Rückgabe in dB relativ zur Gesamtenergie (z. B. −12 dB = Bass trägt 6 %).
    §PERF-R11: Content-keyed Spektrum-Cache (witness ruft 6 Bänder auf 2
    Signalen — gleiches Fenster-Spektrum wird einmal berechnet, bit-identisch).
    """
    if len(x) < 2048:
        return 0.0
    _key = (
        int(len(x)),
        int(sr),
        hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=8).hexdigest(),
    )
    _cached = _BAND_SPEC_CACHE.get(_key)
    if _cached is None:
        _w = np.hanning(len(x))
        _spec = np.abs(np.fft.rfft(x * _w)) ** 2
        _fr = np.fft.rfftfreq(len(x), 1.0 / sr)
        if len(_BAND_SPEC_CACHE) >= _BAND_SPEC_CACHE_MAX:
            del _BAND_SPEC_CACHE[next(iter(_BAND_SPEC_CACHE))]
        _BAND_SPEC_CACHE[_key] = (_spec, _fr)
        _cached = (_spec, _fr)
    _spec, _fr = _cached
    _band = _spec[(_fr >= lo_hz) & (_fr <= hi_hz)]
    _total = _spec[(_fr >= 20.0) & (_fr <= 20000.0)]
    if _total.sum() <= 1e-20 or _band.sum() <= 1e-20:
        return 0.0
    return float(10.0 * np.log10(max(_band.sum(), 1e-20) / max(_total.sum(), 1e-20)))


def _transient_sharpness(x: np.ndarray, sr: int) -> float:
    """95-Perzentil der Hüllkurven-Steigung (dB/s) — Transienten-Schärfe-Proxy.

    5-ms-RMS-Envelope; die steilsten Anstiege (95. Perzentil der positiven
    Steigung) repräsentieren Anschläge/Onsets. Heavy-NR/Verschmierung senkt sie.
    """
    _win = max(64, int(_TRANSIENT_WIN * sr))
    _hop = _win // 2
    if len(x) < _win * 4:
        return 0.0
    # §PERF-R11: Sliding-Window-RMS statt Python-Comprehension (12 k Fenster/30 s).
    _sw = np.lib.stride_tricks.sliding_window_view(x, _win)
    _rms = np.sqrt(np.mean(_sw[::_hop] ** 2, axis=1) + 1e-12)
    _db = 20.0 * np.log10(_rms + 1e-12)
    _slope = np.diff(_db) / (_hop / float(sr))
    _pos = _slope[_slope > 0]
    if len(_pos) < 4:
        return 0.0
    return float(np.percentile(_pos, 95.0))


# ── Audio-Bundle-Zwischenspeicher (bit-identisch, Qualität neutral) ──────────
# Die per-Audio-Witness-Metriken (F0/HNR/Flatness, Loudness-Modulation,
# Flat-Top, Bass, Luftband, Klarheit, Transienten-Schärfe) sind
# deterministische Funktionen des Audio-Inhalts. Im Phasen-Loop ist der
# „before"-Zustand der Phase k+1 eine Kopie des „after"-Zustands der Phase k
# → gleicher Inhalt ⇒ exakt derselbe Wert (keine Näherung).
# Muster: R12-Interaural-Cache (Blake2b-Content-Key, begrenzter LRU).
_WITNESS_BUNDLE_CACHE: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
_WITNESS_BUNDLE_CACHE_MAX = 8
_WITNESS_BUNDLE_LOCK = threading.Lock()


def reset_witness_bundle_cache() -> None:
    """§V8 (copilot-instructions.md) Song-Isolation: leert den Bundle-Cache."""
    with _WITNESS_BUNDLE_LOCK:
        _WITNESS_BUNDLE_CACHE.clear()


def _witness_audio_bundle(x: np.ndarray, sr: int) -> dict[str, Any]:
    """Alle per-Audio-Witness-Metriken für x — content-keyed, bit-identisch."""
    _key = (int(sr), hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=8).hexdigest())
    with _WITNESS_BUNDLE_LOCK:
        _hit = _WITNESS_BUNDLE_CACHE.get(_key)
    if _hit is not None:
        return _hit
    f0, voiced, hnr, flat = _frame_f0_hnr(x, sr)
    bundle: dict[str, object] = {
        "f0": f0,
        "voiced": voiced,
        "hnr": hnr,
        "flat": flat,
        "loud_mod": _loudness_mod_depth_db(x, sr),
        "flat_top": float(np.mean(np.abs(x) > 0.98)),
        "bass": _band_energy_ratio_db(x, sr, _BASS_LO_HZ, _BASS_HI_HZ),
        "air": _band_energy_ratio_db(x, sr, _AIR_LO_HZ, _AIR_HI_HZ),
        "clarity": _band_energy_ratio_db(x, sr, _CLARITY_LO_HZ, _CLARITY_HI_HZ),
        "tr": _transient_sharpness(x, sr),
    }
    with _WITNESS_BUNDLE_LOCK:
        _WITNESS_BUNDLE_CACHE[_key] = bundle
        while len(_WITNESS_BUNDLE_CACHE) > _WITNESS_BUNDLE_CACHE_MAX:
            _WITNESS_BUNDLE_CACHE.popitem(last=False)
    return bundle


def evaluate_listening_witness(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
    phase_id: str,
) -> ListeningWitnessResult:
    """Hört deterministisch in das Phase-Delta hinein — Report-only (§V6 (copilot-instructions.md) am Aufrufer)."""
    a = _mono(audio_before)
    b = _mono(audio_after)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    _bun_a = _witness_audio_bundle(a, sr)
    _bun_b = _witness_audio_bundle(b, sr)

    f0_a = np.asarray(_bun_a["f0"], dtype=np.float64)
    vo_a = np.asarray(_bun_a["voiced"], dtype=bool)
    hnr_a = np.asarray(_bun_a["hnr"], dtype=np.float64)
    flat_a = np.asarray(_bun_a["flat"], dtype=np.float64)
    f0_b = np.asarray(_bun_b["f0"], dtype=np.float64)
    vo_b = np.asarray(_bun_b["voiced"], dtype=bool)
    hnr_b = np.asarray(_bun_b["hnr"], dtype=np.float64)
    flat_b = np.asarray(_bun_b["flat"], dtype=np.float64)

    _hop_rate = float(sr) / float(_HOP)
    _spread_a, mod_a = _f0_metrics(f0_a, vo_a, _hop_rate)
    _spread_b, mod_b = _f0_metrics(f0_b, vo_b, _hop_rate)

    voiced_both = vo_a & vo_b
    hnr_drop = 0.0
    pitch_delta = 0.0
    if voiced_both.any():
        hnr_drop = float(np.median(hnr_a[voiced_both]) - np.median(hnr_b[voiced_both]))
        # Cross-Signal-ΔF0: wie stark hat DIE Phase die Tonhöhe verschoben
        # (Median über stimmhafte Frames, Cent) — die zentrale Pitch-Metrik.
        _ratio = np.maximum(f0_b[voiced_both], 1e-6) / np.maximum(f0_a[voiced_both], 1e-6)
        pitch_delta = float(np.median(np.abs(1200.0 * np.log2(_ratio))))
    flat_rise = float(np.median(flat_b) - np.median(flat_a))

    loud_a = float(_bun_a["loud_mod"])
    loud_b = float(_bun_b["loud_mod"])

    # Flat-Top-Proxy: Anteil an ±1 gepinnter Samples (Clipping/Limiting-Artefakt;
    # periodisch, daher HNR-blind — de-Krom-HNR misst Periodizität, nicht
    # harmonische Verzerrung).
    _flat_a = float(_bun_a["flat_top"])
    _flat_b = float(_bun_b["flat_top"])

    # §Witness-Coverage (2026-09-11): Bass-Präsenz + Transienten-Schärfe —
    # klassische „unangenehm“-Defekte (dünner Klang, verschmierte Anschläge),
    # die Pitch/HNR/Loudness nicht abdecken.
    _bass_a = float(_bun_a["bass"])
    _bass_b = float(_bun_b["bass"])
    _tr_a = float(_bun_a["tr"])
    _tr_b = float(_bun_b["tr"])
    bass_drop = _bass_a - _bass_b
    transient_smear = float((_tr_a - _tr_b) / max(_tr_a, 1e-6)) if _tr_a > 1e-6 else 0.0

    # §Witness↔Goals-Harmonisierung (2026-09-12): Brillianz-Zeuge — signiertes
    # Luftband-Delta (8–20 kHz) in dB. Gleiche Wahrnehmungsdomäne wie das
    # Brillianz-Goal (Musical-Goals), aber delta-basiert (Witness-Rolle:
    # Regression, nicht Zielwert). Positiv = Brillianz-Gewinn, negativ = Verlust.
    _air_a = float(_bun_a["air"])
    _air_b = float(_bun_b["air"])
    air_gain = _air_b - _air_a

    # §Witness-SOTA P1/P2 (2026-09-12, Hörordnung Ebene 2): Audibility statt
    # Mess-Null — Luftband-Delta gegen die Johnston-Maskierungsschwelle; plus
    # Rauigkeits-Delta (Vassilakis-vereinfacht). Determinismus: rein FFT-basiert.
    _masked_residual_db = 0.0
    _air_audible = False
    _roughness_rise = 0.0
    try:
        from backend.core.dsp.masking_model import band_audibility as _ba
        from backend.core.dsp.roughness_model import compute_roughness_asper as _cra
        from backend.core.dsp.roughness_model import roughness_rise_asper as _rra

        _air_aud = _ba(a, b, sr, _AIR_LO_HZ, _AIR_HI_HZ)
        _masked_residual_db = float(_air_aud.get("delta_db", 0.0))
        _air_audible = bool(_air_aud.get("audible", False))
        _roughness_rise = _rra(a, b, sr)
        _roughness_before = float(_cra(a, sr))
        # JND-Kalibrierung (SUP-F6): Unterhalb des relativen JND-Anteils ist
        # der Anstieg nicht als Rauigkeits-Zuwachs hörbar → auf 0 klemmen.
        if _roughness_rise <= max(_ROUGHNESS_RISE_ASPER, _ROUGHNESS_RISE_REL * _roughness_before):
            _roughness_rise = 0.0
    except Exception as _sota_exc:
        logger.debug("§Witness-SOTA P1/P2 nicht verfügbar: %s", _sota_exc)

    # §Witness-SOTA P3/P4 (2026-09-12): Stereo-Kollaps (ITD/ILD/IACC vs. JNDs)
    # und Pre-Echo-Proxy — deterministisch; P4 läuft immer (Mono-Basis),
    # P3 nur bei Stereo-Inputs.
    _itd_drift = 0.0
    _ild_drift = 0.0
    _iacc_drop = 0.0
    _stereo_ok = True
    _pre_echo_db = -200.0
    _cf_a = _channels_first_or_none(audio_before)
    _cf_b = _channels_first_or_none(audio_after)
    if _cf_a is not None and _cf_b is not None:
        try:
            from backend.core.dsp.interaural_cues import interaural_cue_integrity as _ici

            _ic = _ici(_cf_a, _cf_b, sr)
            _itd_drift = float(_ic.itd_drift_us)
            _ild_drift = float(_ic.ild_drift_db)
            _iacc_drop = float(_ic.iacc_delta)
            _stereo_ok = bool(_ic.itd_ok and _ic.ild_ok and _ic.iacc_ok)
        except Exception as _p3_exc:
            logger.debug("§Witness-SOTA P3 nicht verfügbar: %s", _p3_exc)
    try:
        from backend.core.dsp.pre_echo_model import pre_echo_ratio_db as _pe_db

        _pre_echo_db = _pe_db(a, b, sr)
    except Exception as _p4_exc:
        logger.debug("§Witness-SOTA P4 nicht verfügbar: %s", _p4_exc)

    # §Residual-Defekt-Zeugen (2026-09-13): Der Witness soll NICHT delta-only
    # urteilen — gedämpften und verzerrten Gesang erkennt er am RESTZUSTAND:
    # 1. vocal_muffled: Klarheitsband (2–6 kHz) NACH der Phase gedämpft,
    #    UND die Phase hat die Verzerrung nicht reduziert — Dämpfung ohne
    #    Heilung ist das Verbotene (Gesang opfern statt Defekt beheben).
    # 2. vocal_distorted_residual: Rest-Clipping (Flat-Tops) nach der
    #    zuständigen Phase — gewarnt wird nur, wenn der Defekt NOCH da ist.
    _clarity_a = float(_bun_a["clarity"])
    _clarity_b = float(_bun_b["clarity"])
    clarity_drop = _clarity_a - _clarity_b
    _distortion_removed = _flat_b < _flat_a - _FLAT_TOP_RESIDUAL * 0.5
    _muffled = clarity_drop > _VOCAL_MUFFLED_DB and not _distortion_removed
    _residual = _flat_b if phase_id in _DEFECT_OWNER_PHASES else 0.0

    result = ListeningWitnessResult(
        phase_id=phase_id,
        pitch_drift_cents=pitch_delta,
        pitch_mod_depth_cents=max(mod_b - mod_a, 0.0),
        hnr_drop_db=hnr_drop,
        hf_flatness_rise=flat_rise,
        flat_top_rise=max(_flat_b - _flat_a, 0.0),
        loud_mod_rise_db=max(loud_b - loud_a, 0.0),
        bass_drop_db=max(bass_drop, 0.0),
        transient_smear_ratio=max(transient_smear, 0.0),
        air_gain_db=air_gain,
        masked_residual_db=_masked_residual_db,
        air_audible=_air_audible,
        roughness_rise_asper=_roughness_rise,
        itd_drift_us=_itd_drift,
        ild_drift_db=_ild_drift,
        iacc_drop=_iacc_drop,
        pre_echo_db=_pre_echo_db,
        vocal_muffled_db=max(clarity_drop, 0.0),
        distortion_residual_flat_top=_residual,
    )

    if result.pitch_drift_cents > _PITCH_DRIFT_CENTS:
        result.findings.append("pitch_instability")
    if result.pitch_mod_depth_cents > _PITCH_MOD_DEPTH_CENTS:
        result.findings.append("pitch_modulation")
    if (
        result.hnr_drop_db > _HNR_DROP_DB
        or result.hf_flatness_rise > _HF_FLATNESS_RISE
        or result.flat_top_rise > _FLAT_TOP_RISE
    ):
        result.findings.append("vocal_distortion")
    if result.loud_mod_rise_db > _LOUD_MOD_RISE_DB:
        result.findings.append("loudness_pumping")
    if result.bass_drop_db > _BASS_DROP_DB:
        result.findings.append("bass_loss")
    if result.transient_smear_ratio > _TRANSIENT_SMEAR_RATIO:
        result.findings.append("transient_smearing")
    if result.air_gain_db < -_AIR_LOSS_DB:
        result.findings.append("air_loss")
    if result.air_audible and result.air_gain_db < -_AIR_LOSS_DB:
        result.findings.append("air_loss_audible")
    if result.roughness_rise_asper > _ROUGHNESS_RISE_ASPER:
        result.findings.append("roughness_increase")
    if not _stereo_ok:
        result.findings.append("stereo_collapse")
    if result.pre_echo_db > _PRE_ECHO_DB:
        result.findings.append("pre_echo")
    if _muffled:
        result.findings.append("vocal_muffled")
    if _residual > _FLAT_TOP_RESIDUAL:
        result.findings.append("vocal_distorted_residual")
    return result

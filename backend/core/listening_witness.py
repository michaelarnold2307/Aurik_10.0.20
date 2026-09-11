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

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── JND-Schwellen (Report-only; Literaturwerte s. Docstring) ─────────────────
_PITCH_DRIFT_CENTS = 15.0  # Median-|ΔF0| über stimmhafte Frames
_PITCH_MOD_DEPTH_CENTS = 25.0  # Modulations-Tiefe der F0-Trajektorie
_HNR_DROP_DB = 2.0  # Stimm-HNR-Abfall (de Krom)
_HF_FLATNESS_RISE = 0.15  # Spektral-Flatness-Anstieg 2–8 kHz
_FLAT_TOP_RISE = 0.005  # Anteil ±1-gepinnter Samples (Clipping-Proxy)
_LOUD_MOD_RISE_DB = 1.5  # STL-Modulations-Tiefe-Anstieg (0.2–6 Hz)

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


def _frame_iter(x: np.ndarray, sr: int, frame: int = _FRAME, hop: int = _HOP):
    for start in range(0, max(len(x) - frame + 1, 1), hop):
        yield x[start : start + frame]


def _frame_f0_hnr(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(f0_hz, voiced, hnr_db, hf_flatness) pro Frame — FFT-Autokorrelation + de-Krom-HNR.

    §V08/§10a-konform: Autokorrelation via FFT statt O(n·lags)-Lag-Loop —
    für 30-s-Chunks (~1400 Frames) in < 0.3 s, für 224-s-Songs in ~1.5 s.
    """
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


def _f0_metrics(f0s: np.ndarray, voiced: np.ndarray, hop_rate_hz: float) -> tuple[float, float]:
    """(Trajektorien-Spread in Cent, Modulations-Tiefe 3–8 Hz in Cent)."""
    f0v = f0s[voiced]
    if len(f0v) < 3:
        return 0.0, 0.0
    cents = 1200.0 * np.log2(np.maximum(f0v, 1e-6) / np.median(f0v))
    spread = float(np.median(np.abs(cents - np.median(cents))))
    # Modulations-Tiefe: Band 3–8 Hz über die F0-Trajektorie (Wow/Vibrato-Proxy);
    # Trajektorien-Abtastrate = sr / _HOP (46.875 Hz @ 48 kHz).
    mod_depth = spread
    if len(cents) >= 16:
        spec = np.abs(np.fft.rfft(cents - np.median(cents)))
        freqs = np.fft.rfftfreq(len(cents), 1.0 / max(hop_rate_hz, 1e-6))
        band = spec[(freqs >= 3.0) & (freqs <= 8.0)]
        if len(band):
            mod_depth = float(np.sqrt(np.mean(band**2)) * 2.0)
    return spread, float(mod_depth)


def _loudness_mod_depth_db(x: np.ndarray, sr: int) -> float:
    """Modulations-Tiefe (dB) der STL-Envelope im Band 0.2–6 Hz (Pumping-Proxy)."""
    win = int(_STL_WIN * sr)
    hop = int(_STL_HOP * sr)
    if len(x) < win:
        return 0.0
    rms = np.asarray(
        [np.sqrt(np.mean(x[i : i + win] ** 2) + 1e-12) for i in range(0, len(x) - win + 1, hop)],
        dtype=np.float64,
    )
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    db = db - np.median(db)
    if len(db) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(db))
    freqs = np.fft.rfftfreq(len(db), 1.0 / (1.0 / _STL_HOP))
    band = spec[(freqs >= _MOD_LO_HZ) & (freqs <= _MOD_HI_HZ)]
    if len(band) == 0:
        return 0.0
    return float(np.sqrt(np.mean(band**2)) * 2.0)


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

    f0_a, vo_a, hnr_a, flat_a = _frame_f0_hnr(a, sr)
    f0_b, vo_b, hnr_b, flat_b = _frame_f0_hnr(b, sr)

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

    loud_a = _loudness_mod_depth_db(a, sr)
    loud_b = _loudness_mod_depth_db(b, sr)

    # Flat-Top-Proxy: Anteil an ±1 gepinnter Samples (Clipping/Limiting-Artefakt;
    # periodisch, daher HNR-blind — de-Krom-HNR misst Periodizität, nicht
    # harmonische Verzerrung).
    _flat_a = float(np.mean(np.abs(a) > 0.98))
    _flat_b = float(np.mean(np.abs(b) > 0.98))

    result = ListeningWitnessResult(
        phase_id=phase_id,
        pitch_drift_cents=pitch_delta,
        pitch_mod_depth_cents=max(mod_b - mod_a, 0.0),
        hnr_drop_db=hnr_drop,
        hf_flatness_rise=flat_rise,
        flat_top_rise=max(_flat_b - _flat_a, 0.0),
        loud_mod_rise_db=max(loud_b - loud_a, 0.0),
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
    return result

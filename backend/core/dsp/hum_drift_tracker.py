"""§SOTA-HU-V1 — Kalman-getracktes Hum-Drift-Tracking + Subtraktion (phase_02).

Netzfrequenz driftet in Realwelt-Aufnahmen langsam (typ. ±0,2 Hz um 50/60 Hz).
Statische Notch-Filter treffen nur die Momentanfrequenz; bei Drift bleibt der
Brumm hörbar. Dieser Baustein:
  1. misst die Momentanfrequenz des Hum-Grundtons (analytisches Signal nach
     Bandpass, Phasen-Differenz, Median je Fenster),
  2. glättet den Frequenzpfad mit einem Kalman-Filter (Zustand [f, df]),
  3. subtrahiert den getrackten Hum (Grundton + Harmonische) per Fenster-LSQ
     mit Hann-Crossfade und Never-worsen-Energie-Gate je Fenster.

Deterministisch (§G5 (copilot-instructions.md)): reines NumPy/SciPy, keine
Zeit-/Zufallsabhängigkeiten. Never-worsen: kein Fenster darf nach der
Subtraktion mehr Energie haben als vorher.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _bandpass_analytic(x: np.ndarray, sr: int, center: float, bw: float = 2.0) -> Any:
    """Bandpass (2. Ordnung, zero-phase) + analytisches Signal."""
    lo = max(1.0, center - bw)
    hi = min(sr / 2.0 - 1.0, center + bw)
    if hi <= lo:
        return None
    sos = butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    return hilbert(y)


def track_mains_frequency(
    audio: np.ndarray,
    sr: int,
    base_hz: float = 50.0,
    drift_bw: float = 1.0,
    win_seconds: float = 0.5,
    hop_seconds: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman-getrackter Frequenzpfad des Hum-Grundtons.

    Returns (freq_path [Hz, ein Wert je Messfenster], window_times [s]).
    """
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    if mono.size < int(sr * 0.5):
        return np.array([base_hz], dtype=np.float64), np.array([0.0], dtype=np.float64)

    ana = _bandpass_analytic(mono, sr, base_hz)
    if ana is None:
        return np.array([base_hz], dtype=np.float64), np.array([0.0], dtype=np.float64)
    phase = np.unwrap(np.angle(ana))
    inst_f = np.gradient(phase) * sr / (2.0 * np.pi)

    win = max(64, int(sr * win_seconds))
    hop = max(32, int(sr * hop_seconds))
    n_windows = max(1, 1 + (mono.size - win) // hop)
    meas: list[float] = []
    meas_times: list[float] = []
    for i in range(n_windows):
        s = i * hop
        seg = inst_f[s : s + win]
        if seg.size < 64:
            continue
        m = float(np.median(seg))
        if abs(m - base_hz) <= drift_bw:
            meas.append(m)
            meas_times.append((s + win / 2.0) / sr)

    if not meas:
        return np.array([base_hz], dtype=np.float64), np.array([0.0], dtype=np.float64)

    # Kalman: Zustand [f, df], Messung f.
    dt = float(hop_seconds)
    f_mat = np.array([[1.0, dt], [0.0, 1.0]])
    h_mat = np.array([[1.0, 0.0]])
    x = np.array([meas[0], 0.0])
    p_mat = np.diag([0.25, 0.01])
    q_mat = np.diag([1e-4, 1e-4])
    r_mat = np.array([[0.05]])
    path: list[float] = []
    for m in meas:
        x = f_mat @ x
        p_mat = f_mat @ p_mat @ f_mat.T + q_mat
        k = p_mat @ h_mat.T @ np.linalg.inv(h_mat @ p_mat @ h_mat.T + r_mat)
        x = x + k @ (np.array([m]) - h_mat @ x)
        p_mat = (np.eye(2) - k @ h_mat) @ p_mat
        path.append(float(np.clip(x[0], base_hz - drift_bw, base_hz + drift_bw)))
    return np.asarray(path, dtype=np.float64), np.asarray(meas_times, dtype=np.float64)


def _windowed_lsq_subtract(
    x: np.ndarray,
    sr: int,
    f_path: np.ndarray,
    times: np.ndarray,
    harmonic: int,
    base_hz: float,
) -> np.ndarray:
    """Subtrahiert die getrackte Harmonische per Fenster-LSQ (Hann-Crossfade,
    Never-worsen-Energie-Gate je Fenster)."""
    out = cast(np.ndarray, np.asarray(x, dtype=np.float64)).copy()
    win = max(2048, int(sr * 0.5))
    hop = win // 2
    n = x.size
    if n < win or f_path.size < 2:
        return out
    n_windows = 1 + (n - win) // hop
    accum = np.zeros_like(x)
    weight = np.zeros_like(x)
    hann = np.hanning(win)
    for i in range(n_windows):
        s = i * hop
        seg = x[s : s + win]
        t_seg = (np.arange(win) + s) / sr
        f_t = np.interp(t_seg, times, f_path, left=f_path[0], right=f_path[-1]) * harmonic
        phase_t = 2.0 * np.pi * np.cumsum(f_t) / sr
        basis = np.stack([np.cos(phase_t), np.sin(phase_t)], axis=1)
        coef, *_ = np.linalg.lstsq(basis, seg, rcond=None)
        est = basis @ coef
        cand = seg - est
        # Never-worsen-Energie-Gate: Subtraktion nur, wenn sie Energie senkt.
        if np.sum(cand * cand) <= np.sum(seg * seg):
            accum[s : s + win] += cand * hann
            weight[s : s + win] += hann
    mask = weight > _EPS
    out[mask] = accum[mask] / weight[mask]
    return out


def _remove_mono(
    x: np.ndarray,
    sr: int,
    base_hz: float,
    harmonics: tuple[int, ...],
    drift_bw: float,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    path, times = track_mains_frequency(x, sr, base_hz, drift_bw)
    drift_hz = float(np.max(np.abs(path - base_hz))) if path.size else 0.0
    out = cast(np.ndarray, np.asarray(x, dtype=np.float64)).copy()
    for h in harmonics:
        out = _windowed_lsq_subtract(out, sr, path, times, h, base_hz)
    reduction_db = float(10.0 * np.log10((np.sum(x * x) + _EPS) / (np.sum(out * out) + _EPS)))
    return out, {
        "drift_hz": drift_hz,
        "reduction_db": reduction_db,
        "tracked": bool(path.size > 1),
        "path_len": int(path.size),
    }


def remove_drifting_hum(
    audio: np.ndarray,
    sr: int,
    base_hz: float = 50.0,
    harmonics: tuple[int, ...] = (1, 2, 3),
    drift_bw: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    """Kalman-trackt die Netzfrequenz und subtrahiert den driftenden Hum.

    Returns (audio_out [gleiche Form/dtype wie Eingabe], metadata).
    Never-worsen + §V6 (copilot-instructions.md): Fehler ⇒ Eingabe unverändert.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2 and x.shape[0] == 2:
        outs = []
        metas: list[dict[str, float | bool | int]] = []
        for c in range(x.shape[0]):
            o, m = _remove_mono(x[c], sr, base_hz, harmonics, drift_bw)
            outs.append(o)
            metas.append(m)
        out = np.stack(outs)
        meta = {
            "drift_hz": float(max(m.get("drift_hz", 0.0) for m in metas)),
            "reduction_db": float(min(m.get("reduction_db", 0.0) for m in metas)),
            "tracked": bool(any(m.get("tracked", False) for m in metas)),
            "path_len": int(max(m.get("path_len", 0) for m in metas)),
        }
    elif x.ndim == 2 and x.shape[1] == 2:
        out_t = np.stack(
            [_remove_mono(x[:, c], sr, base_hz, harmonics, drift_bw)[0] for c in range(x.shape[1])], axis=1
        )
        _path_mono, _ = track_mains_frequency(x.mean(axis=1), sr, base_hz, drift_bw)
        return np.asarray(out_t, dtype=np.asarray(audio).dtype), {
            "drift_hz": float(np.max(np.abs(_path_mono - base_hz))),
            "reduction_db": 0.0,
            "tracked": bool(_path_mono.size > 1),
            "path_len": int(_path_mono.size),
        }
    else:
        out, meta = _remove_mono(x, sr, base_hz, harmonics, drift_bw)
    return np.asarray(out, dtype=np.asarray(audio).dtype), meta

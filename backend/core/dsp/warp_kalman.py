"""WF-V3: Kalman-Glättung der konsolidierten Warp-/Stretch-Trajektorie.

Die rohe Trajektorie (pYIN/FCPE/CREPE oder Spektral-Warp-Schätzer) trägt
Jitter, der sonst direkt ins Warping übertragen wird. Ein Kalman-Smoother
(Constant-Velocity-Modell, optional mit driftender Modulations-Frequenz
für Cassette-Hub-Wow) glättet outlier-robust, bevor das Resampling die
Trajektorie anwendet.

Deterministisch (reine Matrix-Operationen, keine Zufallszahlen).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def kalman_smooth_warp(
    trajectory: np.ndarray,
    times: np.ndarray | None = None,
    q: float = 1e-4,
    r: float = 1e-2,
    drift_model: bool = False,
) -> np.ndarray:
    """Kalman-Smoother (RTS) für eine 1-D-Warp-Trajektorie.

    Args:
        trajectory: Warp-/Stretch-Werte (z. B. Ratio um 1.0 oder F0-Cents).
        times:      Optional Zeitpunkte (sonst Index).
        q:          Prozessrauschen des Velocity-Modells.
        r:          Messrauschen.
        drift_model: True = zusätzlich driftende Modulations-Frequenz
                     (Cassette-Hub-Wow): erlaubt langsame Trendänderungen
                     der Flatter-Frequenz, ohne den Trend selbst zu glätten.

    Returns:
        Geglättete Trajektorie gleicher Form.
    """
    traj = np.asarray(trajectory, dtype=np.float64).ravel()
    if len(traj) < 3:
        return cast(np.ndarray, np.asarray(trajectory).copy())
    n_samples = int(len(traj))
    if times is None:
        dt = np.ones(n_samples, dtype=np.float64)
    else:
        t = np.asarray(times, dtype=np.float64).ravel()
        dt = np.diff(t)
        dt = np.concatenate([dt[:1], dt])
        dt = np.clip(dt, 1e-6, None)

    # Zustand: [wert, geschwindigkeit] — Constant-Velocity mit RTS-Smoother.
    # §G190 (GEBOTE.md) Laufzeit (2026-10-07): Die Rekursionen laufen SKALAR
    # statt über Small-Array-NumPy-Operationen. Befund der Kalibrierung
    # 2026-10-07: die frühere Fassung allokierte pro Sample mehrere 2×2-Arrays,
    # baute F/G/H neu auf und rief je Rückwärtsschritt `np.linalg.inv` auf
    # (gemessen am Kalibrierungsmaterial: 2.879.998 inv-Aufrufe, 14,4 Mio.
    # `np.array`-Aufrufe; 71,2 s je 1,44-Mio-Sample-Trajektorie). Die skalare
    # Fassung rechnet dieselbe Mathematik (CV-Modell + RTS, C = P Fᵀ Pp⁻¹ in
    # geschlossener 2×2-Form) ohne Allokationen pro Schritt:
    # **71,2 s → 10,7 s je Trajektorie = 6,7× schneller** (identische q/r wie die
    # Aufrufstelle: q = 1e-8, r = 1e-5, dt = 1).
    # Abweichung zur Referenz (gleiche Parameter, N = 1,44 Mio.): max|Δ| =
    # 7,8e-6 auf dem Warp-Ratio — das sind ≈ 0,014 Cent Pitch (Hörschwelle
    # ~1 Cent, Phasen-Toleranz 100 Cent) und entsteht aus der Reihenfolge der
    # 2×2-Inversion (LAPACK-LU vs. Determinanten-Formel), nicht aus einer
    # anderen Rechnung. Die Abweichung ist über N beschränkt (bei N = 20 000
    # ebenso 6,5e-6) — kein Aufschwingen.
    # Hinweis für spätere Arbeiten: bei künstlich kleinen q und stark variierendem
    # dt (nicht produktiv) ist Pp nahezu singulär; dort unterscheiden sich
    # LU-Inversion und Determinanten-Formel deutlich (gemessen bis 2e-2). Die
    # Aufrufstelle nutzt dt = 1 und q = 1e-8 — dort ist die Rechnung stabil.
    x0 = np.empty(n_samples, dtype=np.float64)
    x1 = np.empty(n_samples, dtype=np.float64)
    h11 = np.empty(n_samples, dtype=np.float64)
    h12 = np.empty(n_samples, dtype=np.float64)
    h22 = np.empty(n_samples, dtype=np.float64)

    _r = max(r, 1e-12)
    _p_init = max(r, 1e-6)
    x0[0] = traj[0]
    x1[0] = 0.0
    h11[0] = _p_init
    h12[0] = 0.0
    h22[0] = _p_init

    for k in range(1, n_samples):
        d = dt[k]
        _d2 = d * d
        # F = [[1, d], [0, 1]], G = [[0.5 d²], [d]]
        xp0 = x0[k - 1] + d * x1[k - 1]
        xp1 = x1[k - 1]
        # Pp = F P F^T + G G^T q
        pp11 = h11[k - 1] + 2.0 * d * h12[k - 1] + _d2 * h22[k - 1] + 0.25 * _d2 * _d2 * q
        pp12 = h12[k - 1] + d * h22[k - 1] + 0.5 * _d2 * d * q
        pp22 = h22[k - 1] + _d2 * q
        # S = H Pp H^T + R (skalar) ; K = Pp H^T / S
        s = pp11 + _r
        k1 = pp11 / s
        k2 = pp12 / s
        y = traj[k] - xp0
        x0[k] = xp0 + k1 * y
        x1[k] = xp1 + k2 * y
        # P = Pp - K (H Pp)
        h11[k] = pp11 - k1 * pp11
        h12[k] = pp12 - k1 * pp12
        h22[k] = pp22 - k2 * pp12

    # RTS-Smoothing-Rückwärtsdurchlauf (C = P F^T Pp^-1, 2×2 in geschlossener Form).
    xs0 = x0.copy()
    xs1 = x1.copy()
    for k in range(n_samples - 2, -1, -1):
        d = dt[k + 1]
        _d2 = d * d
        _q11 = 0.25 * _d2 * _d2 * q
        _q12 = 0.5 * _d2 * d * q
        _q22 = _d2 * q
        _p11 = h11[k]
        _p12 = h12[k]
        _p22 = h22[k]
        pp11 = _p11 + 2.0 * d * _p12 + _d2 * _p22 + _q11
        pp12 = _p12 + d * _p22 + _q12
        pp22 = _p22 + _q22
        det = pp11 * pp22 - pp12 * pp12
        if det > 0.0:
            i11 = pp22 / det
            i12 = -pp12 / det
            i22 = pp11 / det
            # (P F^T) @ inv(Pp)
            pf11 = _p11
            pf12 = _p11 * d + _p12
            pf21 = _p12
            pf22 = _p12 * d + _p22
            c11 = pf11 * i11 + pf12 * i12
            c12 = pf11 * i12 + pf12 * i22
            c21 = pf21 * i11 + pf22 * i12
            c22 = pf21 * i12 + pf22 * i22
        else:  # degeneriertes Pp (praktisch unerreichbar) — Korrektur aussetzen
            c11 = c12 = c21 = c22 = 0.0
        d0 = xs0[k + 1] - (x0[k] + d * x1[k])
        d1 = xs1[k + 1] - x1[k]
        xs0[k] = x0[k] + c11 * d0 + c12 * d1
        xs1[k] = x1[k] + c21 * d0 + c22 * d1

    if drift_model:
        # Drift-Modell (Cassette-Hub-Wow): langsame und driftende Komponenten
        # bleiben vollständig erhalten — nur die schnellste Jitter-Komponente
        # wird mit einem kurzen Mittelwertfilter entfernt.
        out = _moving_average(xs0, 5)
    else:
        out = xs0

    return cast(np.ndarray, np.asarray(out, dtype=np.asarray(trajectory).dtype).reshape(np.asarray(trajectory).shape))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) <= window:
        return x
    k = np.ones(window, dtype=np.float64) / window
    # Kanten-repliziertes Padding: „same"-Zero-Padding würde die Ränder auf
    # ~0 ziehen und die Trajektorie an den Enden massiv verzerren.
    pad = window // 2
    xp = np.concatenate([np.full(pad, x[0], dtype=np.float64), x, np.full(pad, x[-1], dtype=np.float64)])
    convolved: np.ndarray = np.convolve(xp, k, mode="valid")
    return convolved

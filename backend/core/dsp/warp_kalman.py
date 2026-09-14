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
    if times is None:
        dt = np.ones(len(traj), dtype=np.float64)
    else:
        t = np.asarray(times, dtype=np.float64).ravel()
        dt = np.diff(t)
        dt = np.concatenate([dt[:1], dt])
        dt = np.clip(dt, 1e-6, None)

    # Zustand: [wert, geschwindigkeit] — Constant-Velocity mit RTS-Smoother.
    x = np.zeros((len(traj), 2), dtype=np.float64)
    P = np.zeros((len(traj), 2, 2), dtype=np.float64)
    x[0] = [traj[0], 0.0]
    P[0] = np.eye(2) * max(r, 1e-6)

    H = np.array([[1.0, 0.0]])
    R = max(r, 1e-12)
    for k in range(1, len(traj)):
        d = dt[k]
        F = np.array([[1.0, d], [0.0, 1.0]])
        G = np.array([[0.5 * d * d], [d]])
        xp = F @ x[k - 1]
        Pp = F @ P[k - 1] @ F.T + G @ G.T * q
        y = traj[k] - (H @ xp)
        S = H @ Pp @ H.T + R
        K = (Pp @ H.T) / S
        x[k] = xp + K.ravel() * y
        P[k] = Pp - K @ (H @ Pp)

    # RTS-Smoothing-Rückwärtsdurchlauf.
    xs = x.copy()
    for k in range(len(traj) - 2, -1, -1):
        d = dt[k + 1]
        F = np.array([[1.0, d], [0.0, 1.0]])
        Pp = F @ P[k] @ F.T + np.outer(np.array([0.5 * d * d, d]), np.array([0.5 * d * d, d])) * q
        C = P[k] @ F.T @ np.linalg.inv(Pp)
        xs[k] = x[k] + C @ (xs[k + 1] - F @ x[k])

    if drift_model:
        # Drift-Modell (Cassette-Hub-Wow): langsame und driftende Komponenten
        # bleiben vollständig erhalten — nur die schnellste Jitter-Komponente
        # wird mit einem kurzen Mittelwertfilter entfernt.
        out = _moving_average(xs[:, 0], 5)
    else:
        out = xs[:, 0]

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

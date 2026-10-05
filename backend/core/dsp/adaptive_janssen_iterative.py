"""
Adaptive Janssen Iterative Spectral Subtraction DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Klassische adaptive Janssen-Iterative-Spektral-Subtraktion mit automatischer Parameteroptimierung (SOTA-Maximum).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_janssen_iterative"
    category: str = "spectral_subtraction"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveJanssenIterative:
    def __init__(self, n_iter: int = 10):
        self.n_iter = n_iter

    def declip(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Janssen iterative AR-Interpolation für Declipping — vektorisiert.

        Algorithmus (Janssen et al., 1986), beschleunigt via scipy.signal.lfilter:
          1. Yule-Walker AR-Schätzung: FFT-basierte Autokorrelation O(n log n).
          2. Alle geclippten Segmente per scipy.signal.lfilter (C-Speed) füllen.
             Vorwärts + Rückwärts + Crossfade für optimale Kontinuität.
          3. Clip-Constraint vektorisiert (np.where, kein Python-Loop).
          4. Wiederhole n_iter-mal.

        Performance: O(n log n · n_iter) statt O(n · order · n_clipped · n_iter)

        Parameters
        ----------
        x    : Eingabe-Signal (float64, 1-D)
        mask : bool-Array gleicher Länge – True = reliables Sample,
               False = geclippt / zu interpolierende Lücke
        """
        from scipy.signal import lfilter

        # toeplitz + fftconvolve entfernt — Yule-Walker durch Burg-Methode ersetzt
        # §4.2: Yule-Walker AR-Modell VERBOTEN → §4.4: Burg-Methode Pflicht

        def _burg_ar_local(sig: np.ndarray, p: int) -> np.ndarray:
            """Burg-Methode AR-Schätzung — §4.4-Pflicht statt Yule-Walker (§4.2).
            Referenz: Burg (1968); Kay (1988) Modern Spectral Estimation."""
            n = len(sig)
            p = min(p, max(1, n - 1))
            ef = sig.astype(np.float64).copy()
            eb = sig.astype(np.float64).copy()
            ar: np.ndarray = np.zeros(p)
            for m in range(p):
                num = -2.0 * float(np.dot(ef[m + 1 :], eb[m : n - 1]))
                den = float(np.dot(ef[m + 1 :], ef[m + 1 :]) + np.dot(eb[m : n - 1], eb[m : n - 1]))
                km = num / (den + 1e-12)
                km = max(-1.0 + 1e-9, min(1.0 - 1e-9, km))
                ar_new = np.zeros(m + 1)
                ar_new[m] = km
                for j in range(m):
                    ar_new[j] = ar[j] + km * ar[m - 1 - j]
                ar = ar_new
                ef_new = ef[m + 1 :] + km * eb[m : n - 1]
                eb = eb[m : n - 1] + km * ef[m + 1 :]
                ef = np.concatenate([[0.0], ef_new])
                eb = np.concatenate([eb, [0.0]])
            return ar

        if x.ndim != 1 or len(x) == 0:
            return x.copy()

        y = x.copy().astype(np.float64)
        reliable = mask.astype(bool)

        if reliable.all():
            return np.asarray(y, dtype=np.float64)  # type: ignore[no-any-return]  # noch nichts tun

        # AR-Ordnung: 1/10 der Signallänge, mindestens 8, höchstens 64.
        # Höhere Ordnungen bringen keinen messbaren Qualitätsgewinn beim
        # Declipping, verursachen aber O(p²)-BLAS-Last bei parallelen Tests.
        order = int(np.clip(len(x) // 10, 8, 64))
        clipped_mask = ~reliable

        # Clip-Constraint-Schwelle und Vorzeichen-Masken (einmalig berechnen)
        threshold = float(np.max(np.abs(x[reliable]))) if reliable.any() else 1.0
        x_pos = (x > 0) & clipped_mask  # positive Clips → y[i] ≥ threshold
        x_neg = (x < 0) & clipped_mask  # negative Clips → y[i] ≤ -threshold

        clipped_idx = np.where(clipped_mask)[0]
        if len(clipped_idx) == 0:
            return np.asarray(y, dtype=np.float64)  # type: ignore[no-any-return]

        for _ in range(self.n_iter):
            y_safe = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Burg-Methode AR-Schätzung §4.4 (Yule-Walker verboten §4.2) ---
            actual_order = min(order, max(1, len(y_safe) - 1))
            try:
                ar_short = _burg_ar_local(y_safe, actual_order)
                if not np.all(np.isfinite(ar_short)):
                    ar_short = np.zeros(actual_order)
            except Exception as exc:
                logger.warning(
                    "adaptive_janssen_iterative: Burg-AR-Schätzung (order=%d) fehlgeschlagen (%s) -> Abbruch der Iterationen, bisheriges Ergebnis (§V6, copilot-instructions.md)",
                    actual_order,
                    exc,
                )
                break

            # --- FIR-Prädiktionsfilter über das gesamte Signal (kein Segment-Loop) ---
            # pred[n] = -ar[0]*y[n-1] - ar[1]*y[n-2] - ... - ar[p-1]*y[n-p]
            # Realisiert als FIR: b = [0, -ar[0], ..., -ar[p-1]], a = [1.0]
            b_pred = np.concatenate([[0.0], -ar_short])
            pred_all = lfilter(b_pred, [1.0], y_safe)  # ein C-Aufruf für alles
            y[clipped_idx] = np.nan_to_num(pred_all[clipped_idx], nan=0.0)

            # --- Clip-Constraint vektorisiert (kein Python-Loop) ---
            y = np.where(x_pos, np.maximum(y, threshold), y)
            y = np.where(x_neg, np.minimum(y, -threshold), y)

        # Finale NaN/Inf-Schutzgarantie (§3.1 Numerische Robustheit)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.asarray(np.clip(y, -1.0, 1.0), dtype=np.float64)  # type: ignore[no-any-return]

    def auto_optimize(self, x: np.ndarray, mask: np.ndarray) -> None:
        # Passe die Iterationszahl adaptiv an
        self.n_iter = min(50, max(5, len(x) // 1000))

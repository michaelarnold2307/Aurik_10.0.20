"""
adaptive_ar_prediction_levinson.py — Adaptive AR-Prädiktion (Levinson-Durbin).

Levinson-Durbin-Rekursion (Proakis & Manolakis) über die (fenster-
gewichtete) Autokorrelation: Koeffizienten aus dem gemittelten
Korrelationsbild des Signals. Gegenüber Burg robuster bei dichter
Mikro-Impuls-Struktur (Knistern, Crackle), weil die Autokorrelation
über viele Impulse mittelt — deshalb der Primär-Pfad des
AutomaticDecrackler (backend/core/dsp/automatic_decrackler.py).

Konvention: a[0] = 1, Prognose x^(n) = -Summe_{j=1..p} a[j] * x(n-j).
Gemeinsame AR-Operatoren (Prognose, Extrapolation, Lückenfüllung)
liegen kanonisch in :mod:`backend.core.dsp.adaptive_ar_prediction_burg`.

Determinismus (§G5 (copilot-instructions.md)): reine Lineare Algebra,
keine Zufallszahlen, keine Zeitstempel in Entscheidungen.
NaN/Inf-Schutz (§0a (copilot-instructions.md)) in jedem Einstiegspunkt.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from backend.core.dsp.adaptive_ar_prediction_burg import (
    DSPContract,
    ar_extrapolate,
    ar_fill_gap,
    ar_predict_one_step,
)

logger = logging.getLogger(__name__)

_EPS = 1e-300
_MAX_REFLECTION = 0.9999  # Stabilitätsgrenze |k| < 1


def levinson_durbin_ar(x: Any, order: int) -> np.ndarray:
    """AR-Koeffizienten a[1..order] via Levinson-Durbin über die Autokorrelation.

    Biased ACF (Rechteckfenster) ⇒ positiv definite Toeplitz-Matrix ⇒
    stabiler Filterkern. Degenerierte Signale liefern Nullkoeffizienten.
    """
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n = sig.size
    order = int(max(1, min(order, n - 2))) if n >= 3 else 0
    if order == 0:
        return np.asarray(np.zeros(0))  # type: ignore[no-any-return]

    # Biased ACF r[0..order]
    r = np.empty(order + 1, dtype=np.float64)
    for lag in range(order + 1):
        r[lag] = float(np.dot(sig[: n - lag], sig[lag:]) / n)

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]
    for m in range(1, order + 1):
        if e <= _EPS:
            break
        acc = r[m] + float(np.dot(a[1:m], r[m - 1 : 0 : -1])) if m > 1 else r[m]
        k = float(np.clip(-acc / e, -_MAX_REFLECTION, _MAX_REFLECTION))
        a_prev = a.copy()
        for j in range(1, m):
            a[j] = a_prev[j] + k * a_prev[m - j]
        a[m] = k
        e *= 1.0 - k * k
    return np.asarray(a[1:])  # type: ignore[no-any-return]


class AdaptiveARPredictionLevinson:
    """Adaptive AR-Prädiktion (Levinson-Durbin) — dichte Mikro-Impulse.

    SOTA-Pfad des Knister-/Crackle-Reparaturwegs; für vereinzelte
    Klicks siehe
    :class:`~core.dsp.adaptive_ar_prediction_burg.AdaptiveARPredictionBurg`.
    """

    contract: DSPContract = DSPContract(id="adaptive_ar_prediction_levinson")

    def __init__(self, order: int = 8):
        self.order = order

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def fit(self, x: np.ndarray) -> np.ndarray:
        """AR-Koeffizienten a[1..order] (Levinson-Durbin)."""
        self.log_contract()
        return levinson_durbin_ar(x, self.order)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Ein-Schritt-AR-Prognose des Signals (gleiche Länge)."""
        self.log_contract()
        return ar_predict_one_step(x, self.fit(x))

    def extrapolate(self, x: np.ndarray, steps: int, reverse: bool = False) -> np.ndarray:
        """AR-Fortsetzung um `steps` Samples (reverse: chronologisch davor)."""
        self.log_contract()
        ctx = x[::-1] if reverse else x
        ext = ar_extrapolate(ctx, self.fit(ctx), steps)
        return ext[::-1] if reverse else ext

    def interpolate_gap(self, x: np.ndarray, start: int, end: int) -> np.ndarray:
        """Beidseitige AR-Lückenfüllung x[start:end) mit Crossfade."""
        self.log_contract()
        return ar_fill_gap(x, start, end, order=self.order, fit=levinson_durbin_ar)

    def auto_optimize(self, x: np.ndarray) -> None:
        """Passt die Ordnung adaptiv an die Signal-Länge an."""
        self.order = min(16, max(2, len(x) // 1000))

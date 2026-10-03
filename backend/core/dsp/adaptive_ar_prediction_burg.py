"""
adaptive_ar_prediction_burg.py — Adaptive AR-Prädiktion (Burg) für Aurik.

Burg-Methode (Burg 1967/1975): Reflexionskoeffizienten direkt aus den
Vorwärts-/Rückwärts-Prognosefehlern, Koeffizienten-Rekursion nach
Proakis & Modulator. Gegenüber der reinen Autokorrelations-Methode
stabil bei kurzen Kontexten — deshalb der Primär-Pfad für einzelne
Klicks und kurze Lücken (declick_core).

Konvention: a[0] = 1, Prognose x^(n) = -Summe_{j=1..p} a[j] * x(n-j).

Einsatz: musikalische Lücken-/Klick-Reparatur (backend/core/dsp/declick_core.py).
Reparatur-Artefakte müssen unter der psychoakustischen Maskierungs-
schwelle bleiben (Hörordnung Ebene 2,
`.github/instructions/hoerordnung.instructions.md`) — AR-Extrapolation
setzt die Fortsetzung am Spektrum des Kontexts an statt mit hartem
Median-Ersatz eine Kante zu erzeugen.

Determinismus (§G5 (copilot-instructions.md)): reine Lineare Algebra,
keine Zufallszahlen, keine Zeitstempel in Entscheidungen.
NaN/Inf-Schutz (§0a (copilot-instructions.md)) in jedem Einstiegspunkt.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

_EPS = 1e-300
_MAX_REFLECTION = 0.9999  # Stabilitätsgrenze |k| < 1


@dataclass(frozen=True)
class DSPContract:
    """DSPContract — die eine AR-Contract-Implementierung der Domain dsp/ar.

    Symbol-Duplikat-Regel (repo_graph --duplicates): genau eine
    Implementierung pro Domain. Diese Klasse ist kanonisch (wie die
    geteilten AR-Operatoren); die Levinson-Durbin-Variante importiert
    sie und führt sich als distincte `id`-Variante
    (`adaptive_ar_prediction_levinson`).
    """

    id: str = "adaptive_ar_prediction_burg"
    category: str = "ar_prediction"
    version: str = "2.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


def burg_ar_coefficients(x: Any, order: int) -> np.ndarray:
    """AR-Koeffizienten a[1..order] (Burg) für ein 1-D-Signal.

    Rückgabe shape (order,) — a[0] = 1 ist implizit. Degenerierte
    Signale (zu kurz, konstante Ebene) liefern Nullkoeffizienten
    (Determinismus statt Zufalls-Rest).
    """
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n = sig.size
    order = int(max(1, min(order, n - 2))) if n >= 3 else 0
    if order == 0:
        return np.asarray(np.zeros(0))  # type: ignore[no-any-return]

    # e_f^{m-1}(n) = x(n), e_b^{m-1}(n-1) = x(n-1), n = 1..N-1
    ef = sig[1:].copy()
    eb = sig[:-1].copy()
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    for m in range(1, order + 1):
        num = -2.0 * float(np.dot(ef, eb))
        den = float(np.dot(ef, ef) + np.dot(eb, eb))
        if den <= _EPS:
            break
        k = float(np.clip(num / den, -_MAX_REFLECTION, _MAX_REFLECTION))
        a_prev = a.copy()
        for j in range(1, m):
            a[j] = a_prev[j] + k * a_prev[m - j]
        a[m] = k
        # Fehler-Rekursion (Proakis): e_f^m = e_f^{m-1} + k*e_b^{m-1},
        # e_b^m = e_b^{m-1} + k*e_f^{m-1}; Arrays schrumpfen um je 1.
        ef, eb = (ef + k * eb)[1:], (eb + k * ef)[:-1]
        if ef.size == 0:
            break
    return np.asarray(a[1:])  # type: ignore[no-any-return]


def ar_predict_one_step(x: Any, a: np.ndarray) -> np.ndarray:
    """Ein-Schritt-AR-Prognose x^(n) = -Summe_j a[j]*x(n-j) (gleiche Länge).

    Die ersten p Samples nutzen den jeweils verfügbaren kürzeren Kontext
    (keine Stilllegung am Rand).
    """
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty_like(sig)
    p = int(a.size)
    for n in range(sig.size):
        j_max = min(p, n)
        out[n] = -float(np.dot(a[:j_max], sig[n - j_max : n][::-1])) if j_max else 0.0
    return np.asarray(out)  # type: ignore[no-any-return]


def ar_extrapolate(x: Any, a: np.ndarray, steps: int) -> np.ndarray:
    """Fortsetzung des Signals um `steps` Samples per rekursiver AR-Prognose.

    Rückwärts-Prognose über Umkehrung des Kontexts (siehe ar_fill_gap):
    die Operatoren bleiben zeitorientiert, die Richtung steckt im Kontext.
    """
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    steps = int(max(0, steps))
    if steps == 0 or sig.size == 0:
        return np.asarray(np.zeros(0))  # type: ignore[no-any-return]
    hist = [float(v) for v in sig]
    out = np.empty(steps, dtype=np.float64)
    p = int(a.size)
    for i in range(steps):
        j_max = min(p, len(hist))
        pred = -float(np.dot(a[:j_max], np.asarray(hist[len(hist) - j_max :][::-1]))) if j_max else 0.0
        out[i] = pred
        hist.append(pred)
    return np.asarray(out)  # type: ignore[no-any-return]


def ar_fill_gap(
    x: Any,
    start: int,
    end: int,
    order: int = 8,
    ctx: int = 32,
    fit: Callable[[Any, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Füllt die Lücke x[start:end) musikalisch (beidseitige AR-Extrapolation).

    Vorwärts-Prognose aus dem linken, Rückwärts-Prognose aus dem rechten
    Kontext, linear überblendet (Mitte = gemittelt, Ränder = je die
    nähere Seite). Rückgabe: Kopie mit gefüllter Lücke. Degenerierte
    Kontexte fallen deterministisch auf lineare Interpolation zurück
    (§V6 (VERBOTEN.md): Fallback wird geloggt).
    """
    fit_fn = fit if fit is not None else burg_ar_coefficients
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    out = sig.copy()
    start = int(max(0, start))
    end = int(min(sig.size, end))
    length = end - start
    if length <= 0:
        return np.asarray(out)  # type: ignore[no-any-return]

    left = sig[max(0, start - ctx) : start]
    right = sig[end : min(sig.size, end + ctx)]
    fwd = bwd = None
    if left.size >= 4:
        a_l = fit_fn(left, order)
        if a_l.size > 0:
            fwd = ar_extrapolate(left, a_l, length)
    if right.size >= 4:
        # Rückwärts-Kontext: Fortsetzung am "Ende" der Zeitumkehr = Lücke
        # von rechts nach links; [::-1] stellt die Zeitrichtung wieder her.
        a_r = fit_fn(right[::-1], order)
        if a_r.size > 0:
            bwd = ar_extrapolate(right[::-1], a_r, length)[::-1]

    if fwd is not None and bwd is not None:
        w = np.linspace(1.0, 0.0, length + 2)[1:-1]
        out[start:end] = w * fwd + (1.0 - w) * bwd
    elif fwd is not None:
        out[start:end] = fwd
    elif bwd is not None:
        out[start:end] = bwd
    else:
        logger.warning(
            "AR-Lückenfüllung ohne ausreichenden Kontext (start=%d, end=%d) — §V6-Fallback: lineare Interpolation",
            start,
            end,
        )
        edge_left = sig[start - 1] if start > 0 else (sig[end] if end < sig.size else 0.0)
        edge_right = sig[end] if end < sig.size else edge_left
        out[start:end] = np.linspace(edge_left, edge_right, length)
    return np.asarray(out)  # type: ignore[no-any-return]


class AdaptiveARPredictionBurg:
    """Adaptive AR-Prädiktion (Burg) — kurze Kontexte, einzelne Klicks.

    SOTA-Pfad der Declicker-Familie für vereinzelte Impulse; für dichte
    Mikro-Impulse (Knistern) steht die Levinson-Durbin-Variante bereit
    (:class:`~core.dsp.adaptive_ar_prediction_levinson.AdaptiveARPredictionLevinson`).
    """

    contract: DSPContract = DSPContract()

    def __init__(self, order: int = 8):
        self.order = order

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def fit(self, x: Any) -> np.ndarray:
        """AR-Koeffizienten a[1..order] (Burg)."""
        self.log_contract()
        return burg_ar_coefficients(x, self.order)

    def predict(self, x: Any) -> np.ndarray:
        """Ein-Schritt-AR-Prognose des Signals (gleiche Länge)."""
        self.log_contract()
        return ar_predict_one_step(x, self.fit(x))

    def extrapolate(self, x: Any, steps: int, reverse: bool = False) -> np.ndarray:
        """AR-Fortsetzung um `steps` Samples (reverse: chronologisch davor)."""
        self.log_contract()
        ctx = x[::-1] if reverse else x
        ext = ar_extrapolate(ctx, self.fit(ctx), steps)
        return ext[::-1] if reverse else ext

    def interpolate_gap(self, x: Any, start: int, end: int) -> np.ndarray:
        """Beidseitige AR-Lückenfüllung x[start:end) mit Crossfade."""
        self.log_contract()
        return ar_fill_gap(x, start, end, order=self.order, fit=burg_ar_coefficients)

    def auto_optimize(self, x: Any) -> None:
        """Passt die Ordnung adaptiv an die Signal-Länge an."""
        self.order = min(16, max(2, len(x) // 1000))

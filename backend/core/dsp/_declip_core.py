"""
_declip_core.py – Gemeinsame Kern-Implementierung für alle Aurik-Declipper.

Algorithmus: Janssen AR-Iterative Interpolation (Janssen et al., 1986)
  1. Geclippte Samples via Amplituden-Schwellenwert erkennen.
  2. AR(order)-Modell via Yule-Walker auf nicht-geclippten Samples schätzen.
  3. Geclippte Samples via AR-Vorwärtsvorhersage rekonstruieren + Clip-Constraint.
  4. n_iter Durchgänge.

Exportierte Funktion: ar_declip(audio, sr, **kwargs) -> np.ndarray
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# toeplitz entfernt — Yule-Walker durch Burg-Methode ersetzt (§4.2: Yule-Walker verboten)


def _burg_ar(sig: np.ndarray, order: int) -> np.ndarray:
    """Burg-Methode AR-Schätzung — §4.4-Pflicht statt Yule-Walker (verboten §4.2).

    Referenz: Burg (1968); Kay (1988) 'Modern Spectral Estimation',
    Prentice-Hall. Numerisch stabiler als Yule-Walker auf geclippten Signalen.
    """
    n = len(sig)
    order = min(order, max(1, n - 1))
    ef = sig.astype(np.float64).copy()
    eb = sig.astype(np.float64).copy()
    ar: np.ndarray = np.zeros(order)
    for m in range(order):
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


def ar_declip(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.95,
    order: int = 64,
    n_iter: int = 10,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    bp_low_hz: float | None = None,
    bp_high_hz: float | None = None,
) -> np.ndarray:
    """
    AR-basiertes Declipping via Janssen-Algorithmus.

    Parameters
    ----------
    audio      : 1-D oder 2-D (samples × channels) float-Array
    sr         : Sample-Rate in Hz
    threshold  : Clipping-Schwellenwert relativ zum Peak (0.80–0.99)
    order      : AR-Modellordnung (Taps)
    n_iter     : Iterationszahl
    lowpass_hz : Tiefpass-Grenzfrequenz (Hz) – für Bass-Variante
    highpass_hz: Hochpass-Grenzfrequenz (Hz)
    bp_low_hz  : Bandpass untere Grenze
    bp_high_hz : Bandpass obere Grenze
    """
    audio = np.asarray(audio, dtype=np.float64)

    # --- Stereo-Support: kanalweise ---
    if audio.ndim == 2:
        return np.stack(  # type: ignore[no-any-return]
            [
                ar_declip(
                    audio[:, ch],
                    sr,
                    threshold=threshold,
                    order=order,
                    n_iter=n_iter,
                    lowpass_hz=lowpass_hz,
                    highpass_hz=highpass_hz,
                    bp_low_hz=bp_low_hz,
                    bp_high_hz=bp_high_hz,
                )
                for ch in range(audio.shape[1])
            ],
            axis=1,
        )

    if audio.ndim != 1 or len(audio) == 0:
        return audio.copy()

    # --- Optionale Filtervorverarbeitung ---
    from scipy.signal import butter, sosfilt

    y_orig = audio.copy()
    y = audio.copy()

    if lowpass_hz is not None and lowpass_hz < sr / 2:
        sos = butter(4, lowpass_hz / (sr / 2), btype="low", output="sos")
        y = sosfilt(sos, y)

    if highpass_hz is not None and highpass_hz < sr / 2:
        sos = butter(4, highpass_hz / (sr / 2), btype="high", output="sos")
        y = sosfilt(sos, y)

    if bp_low_hz is not None and bp_high_hz is not None:
        lo = np.clip(bp_low_hz / (sr / 2), 1e-4, 0.999)
        hi = np.clip(bp_high_hz / (sr / 2), 1e-4, 0.999)
        if lo < hi:
            sos = butter(4, [lo, hi], btype="band", output="sos")
            y = sosfilt(sos, y)

    # --- Clipping-Maske ---
    peak: float = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return y_orig.copy()

    clip_thresh = peak * float(threshold)
    reliable = np.abs(y) < clip_thresh  # True = verlässliches Sample

    if reliable.all():
        return y_orig.copy()

    # --- Janssen AR-Iterationen (Burg-Methode §4.4 — Yule-Walker verboten §4.2) ---
    order = int(np.clip(order, 4, min(256, len(y) // 4)))
    clipped_idx = np.where(~reliable)[0]

    # Janssen-Initialisierung (Janssen et al. 1986): Geclippte Samples vor der
    # ersten Burg-AR-Schätzung mit linearer Interpolation belegen, damit der
    # Schätzer keine Flat-Tops sieht und korrekte AR-Koeffizienten lernt.
    if len(clipped_idx) > 0:
        for ci in clipped_idx:
            l_nb, r_nb = ci - 1, ci + 1
            while l_nb >= 0 and not reliable[l_nb]:
                l_nb -= 1
            while r_nb < len(y) and not reliable[r_nb]:
                r_nb += 1
            if l_nb >= 0 and r_nb < len(y):
                y[ci] = y[l_nb] + (y[r_nb] - y[l_nb]) * (ci - l_nb) / (r_nb - l_nb)
            elif l_nb >= 0:
                y[ci] = y[l_nb]
            elif r_nb < len(y):
                y[ci] = y[r_nb]
            # else: Index am Rand ohne Nachbarn — bleibt wie es ist

    for _ in range(n_iter):
        y_safe = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            ar = _burg_ar(y_safe, order)
            if not np.all(np.isfinite(ar)):
                ar = np.zeros(order)
        except Exception as exc:
            logger.warning(
                "_declip_core: Burg-AR-Schätzung (order=%d) fehlgeschlagen (%s) -> Abbruch der AR-Iterationen, bisheriges Ergebnis (§V6, copilot-instructions.md)",
                order,
                exc,
            )
            break

        # AR-Vorwärtsvorhersage für geclippte Samples
        for i in clipped_idx:
            if i < order:
                left = y_safe[max(0, i - 1)] if i > 0 else 0.0
                right = y_safe[i + 1] if i + 1 < len(y) else 0.0
                pred = 0.5 * (left + right)
            else:
                context = y_safe[i - order : i][::-1]
                pred = float(np.dot(ar, context))
                if not np.isfinite(pred):
                    pred = y_safe[i - 1] if i > 0 else 0.0
            # Clip-Constraint: rekonstruierter Wert muss >= clip_thresh
            if y_orig[i] >= clip_thresh:
                y[i] = max(pred, clip_thresh)
            elif y_orig[i] <= -clip_thresh:
                y[i] = min(pred, -clip_thresh)
            else:
                y[i] = pred

    return np.asarray(np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0))  # type: ignore[no-any-return]


def multiband_ar_declip(
    audio: np.ndarray,
    sr: int,
    n_bands: int = 3,
    threshold: float = 0.95,
    order: int = 32,
    n_iter: int = 8,
) -> np.ndarray:
    """
    Multiband AR-Declipping: Signal in n_bands logarithmisch aufgeteilt,
    jedes Band separat deklipt, dann addiert.
    """
    from scipy.signal import butter, sosfilt

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or len(audio) == 0:
        return audio.copy()

    freqs = np.logspace(np.log10(60), np.log10(sr / 2 * 0.95), n_bands + 1)
    result = np.zeros_like(audio)

    for i in range(n_bands):
        lo = freqs[i] / (sr / 2)
        hi = freqs[i + 1] / (sr / 2)
        lo = np.clip(lo, 1e-4, 0.999)
        hi = np.clip(hi, 1e-4, 0.999)
        if lo >= hi:
            continue
        try:
            sos = butter(4, [lo, hi], btype="band", output="sos")
            band = sosfilt(sos, audio)
            band_dec = ar_declip(band, sr, threshold=threshold, order=order, n_iter=n_iter)
            result += band_dec
        except Exception as exc:
            logger.warning(
                "_declip_core: Band-Deklipping %d (%.0f–%.0f Hz) fehlgeschlagen (%s) -> Originalband addiert (§V6, copilot-instructions.md)",
                i,
                freqs[i],
                freqs[i + 1],
                exc,
            )
            result += audio

    # Normalisieren: Energie-Verhältnis erhalten
    orig_rms = np.sqrt(np.mean(audio**2))
    out_rms = np.sqrt(np.mean(result**2))
    if out_rms > 1e-8:
        result *= orig_rms / out_rms

    return np.asarray(np.clip(result, -1.0, 1.0))  # type: ignore[no-any-return]

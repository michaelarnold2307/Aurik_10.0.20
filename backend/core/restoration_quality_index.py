"""RestorationQualityIndex (RQI) — §GEBOT-G52.

Misst, ob eine Restaurierung das Signal TATSÄCHLICH verbessert hat,
unabhängig von Studio-Referenzmetriken die Abweichung bestrafen.

Komponenten (gewichtet):
  1. Defekt-Reduktion    (0.40): Wurden Klicks/Knacken/Rauschen reduziert?
  2. Bandbreiten-Gewinn  (0.30): Wurde die effektive Bandbreite erweitert?
  3. Natürlichkeit       (0.30): Klingt das Ergebnis natürlicher?

RQI ∈ [0.0, 1.0]: 0.0 = verschlechtert, 0.5 = neutral, 1.0 = perfekt verbessert.

Verwendung: RQI > 0.5 → Quality-Gate-Warnings auf INFO herabstufen
            RQI < 0.3 → tatsächliches Qualitätsproblem
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# RQI-Schwellwerte (§GEBOT-G54)
# ---------------------------------------------------------------------------
RQI_GOOD = 0.50  # oberhalb: Restaurierung erfolgreich → Warnings unterdrücken
RQI_BAD = 0.30  # unterhalb: echtes Qualitätsproblem → WARNING berechtigt


def compute_rqi(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    defect_severity_before: float = 0.5,
    defect_severity_after: float | None = None,
    bandwidth_before_hz: float = 8000.0,
    bandwidth_after_hz: float | None = None,
) -> dict:
    """Berechnet RestorationQualityIndex aus Vorher/Nachher-Messungen.

    Args:
        original:            Degradiertes Original-Audio (mono oder stereo)
        restored:            Restauriertes Audio
        sr:                  Sample-Rate
        defect_severity_before: Geschätzte Defekt-Schwere vorher [0,1]
        defect_severity_after:  Geschätzte Defekt-Schwere nachher [0,1] (None→schätze)
        bandwidth_before_hz:    Effektive Bandbreite vorher (Hz)
        bandwidth_after_hz:     Effektive Bandbreite nachher (Hz) (None→schätze)

    Returns:
        dict mit keys: rqi, defect_reduction, bandwidth_gain, naturalness,
                       interpretation, suppress_warnings
    """
    # Mono machen für Analyse
    orig = original if original.ndim == 1 else original.mean(axis=0)
    rest = restored if restored.ndim == 1 else restored.mean(axis=0)
    orig = np.asarray(orig, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    n = min(len(orig), len(rest))
    orig = orig[:n]
    rest = rest[:n]

    # ── 1. Defekt-Reduktion (0.40) ──────────────────────────────────────
    # Schätze Defektreduktion via spektraler Flachheit (flacher = weniger Impulsdefekte)
    # und Hochfrequenz-Energie-Ratio (Klicks/Knacken → HF-Energie)
    if defect_severity_after is None:
        # Proxy: RMS-Reduktion im 4-16kHz Band (Klick-Energie)
        _fft_orig = np.abs(np.fft.rfft(orig * np.hanning(n)))
        _fft_rest = np.abs(np.fft.rfft(rest * np.hanning(n)))
        _freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        _hf_mask = (_freqs >= 4000) & (_freqs <= 16000)
        _hf_orig = float(np.mean(_fft_orig[_hf_mask] ** 2))
        _hf_rest = float(np.mean(_fft_rest[_hf_mask] ** 2))
        if _hf_orig > 1e-20:
            _defect_after = float(np.clip(1.0 - (_hf_rest / _hf_orig), 0.0, 1.0))
        else:
            _defect_after = 0.0
    else:
        _defect_after = defect_severity_after

    defect_reduction = float(np.clip(defect_severity_before - _defect_after, 0.0, 1.0))

    # ── 2. Bandbreiten-Gewinn (0.30) ────────────────────────────────────
    if bandwidth_after_hz is None:
        # Spektral-Rolloff 90% schätzen
        _cumsum_o = np.cumsum(_fft_orig)
        _cumsum_r = np.cumsum(_fft_rest)
        _total_o = _cumsum_o[-1] + 1e-20
        _total_r = _cumsum_r[-1] + 1e-20
        _bw_before = float(_freqs[int(np.searchsorted(_cumsum_o, 0.90 * _total_o))])
        _bw_after = float(_freqs[int(np.searchsorted(_cumsum_r, 0.90 * _total_r))])
    else:
        _bw_before = bandwidth_before_hz
        _bw_after = bandwidth_after_hz

    _bw_nyquist = sr / 2.0
    bw_gain = float(np.clip((_bw_after - _bw_before) / max(_bw_nyquist - _bw_before, 1.0), 0.0, 1.0))

    # ── 3. Natürlichkeit (0.30) ─────────────────────────────────────────
    # Proxy: Spektrale Glattheit (MFCC-basiert) + HNR-Änderung
    # Glatteres Spektrum ohne Rausch-Spikes → natürlicher
    from scipy.signal import welch

    _f_w, _pxx_o = welch(orig, sr, nperseg=2048)
    _, _pxx_r = welch(rest, sr, nperseg=2048)

    # Spektrale Rauheit: Varianz der logarithmierten PSD
    _log_o = np.log10(np.maximum(_pxx_o, 1e-20))
    _log_r = np.log10(np.maximum(_pxx_r, 1e-20))
    _roughness_o = float(np.std(np.diff(_log_o)))
    _roughness_r = float(np.std(np.diff(_log_r)))

    if _roughness_o > 1e-6:
        _smoothness_gain = float(np.clip(1.0 - (_roughness_r / _roughness_o), 0.0, 1.0))
    else:
        _smoothness_gain = 0.5

    # HNR-Proxy: Peak-to-Mean-Ratio im Spektrum
    _hnr_o = float(np.max(_pxx_o) / (np.mean(_pxx_o) + 1e-20))
    _hnr_r = float(np.max(_pxx_r) / (np.mean(_pxx_r) + 1e-20))
    if _hnr_o > 1e-6:
        _hnr_gain = float(np.clip((_hnr_r / _hnr_o) - 1.0, 0.0, 1.0))
    else:
        _hnr_gain = 0.5

    naturalness = float(np.clip(0.5 * _smoothness_gain + 0.5 * _hnr_gain, 0.0, 1.0))

    # ── RQI ─────────────────────────────────────────────────────────────
    rqi = float(
        np.clip(
            0.40 * defect_reduction + 0.30 * bw_gain + 0.30 * naturalness,
            0.0,
            1.0,
        )
    )

    # ── Interpretation ───────────────────────────────────────────────────
    if rqi >= RQI_GOOD:
        interpretation = "Restaurierung erfolgreich — Signal hörbar verbessert"
        suppress_warnings = True
    elif rqi >= RQI_BAD:
        interpretation = "Restaurierung neutral — leichte Verbesserung"
        suppress_warnings = True  # §GEBOT-G54: ab 0.3 unterdrücken
    else:
        interpretation = "Restaurierung problematisch — mögliche Verschlechterung"
        suppress_warnings = False

    return {
        "rqi": float(rqi),
        "defect_reduction": float(defect_reduction),
        "bandwidth_gain": float(bw_gain),
        "naturalness": float(naturalness),
        "bandwidth_before_hz": float(_bw_before),
        "bandwidth_after_hz": float(_bw_after),
        "interpretation": interpretation,
        "suppress_warnings": suppress_warnings,
    }

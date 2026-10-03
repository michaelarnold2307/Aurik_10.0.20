"""modulation_transfer.py — MTF/STI-Verständlichkeits-Zweitstimme (Punkt 6).

Norm-Bezug: IEC 60268-16 (Speech Transmission Index) über die indirekte
Modulationsübertragungsfunktion: pro Oktavband wird gemessen, wie viel
Amplituden-Modulation (Sprach-/Gesangsrhythmus 0,63–12,5 Hz) durch die
Verarbeitung erhalten bleibt. Verlust an Modulationstiefe = Verlust an
Verständlichkeit (Silbenrhythmus, Konsonantenansätze).

Zwei Ausgaben:
  - ``sti``: STI-artiger Index [0, 1] über die sechs Oktavbänder 250 Hz–8 kHz,
    gewichtet nach IEC-Oktavband-Wichtigkeit (auf Summe der MESSBAREN Bänder
    normiert — ein stilles Band ist „nicht bewertbar", nicht „schlecht").
  - ``consonant_modulation``: Modulationserhalt 4–12,5 Hz in den Bändern
    2–4 kHz — die Konsonantenklarheits-Sicht (Hörordnung §3, Schwelle ≥ 0,85
    als Never-worsen-Erwartung an die Reparatur).

Messvertrag (kalibriert 2026-10-03):
  - Modulationstiefe m = 2·|FFT(env)|/DC der Hüllkurve; der Fourier-Bin wird
    als Maximum in ±2 Bins um die Nominalrate gesucht (Raster-Detuning und
    Leakage sonst fatal: 6,0-Hz-AM landet sonst im Nachbarbin).
  - Ein Band gilt nur als bewertbar, wenn seine Hüllkurven-Energie ≥ 1 % der
    stärksten Band-Hülle trägt — sonst wären numerische Ripple in leeren
    Bändern „perfekt messbar" und würden den Index verfälschen.

Rolle (Hörordnung §8): **Zeuge, nie Richter**. Diese Messung ist die
normnahe Zweitstimme neben dem `IntelligibilityScorer` (vocal_analysis) —
bewusst als eigenständige, nachrechenbare DSP-Messung ohne ML-Abhängigkeit.

Deterministisch (§G5 (GEBOTE.md)), NaN-sicher (§0a (copilot-instructions.md)),
Layout-sicher über `audio_layout.mono_mix` (Stereo-Layout-Invariante,
AGENTS.md), numerisch fail-closed: Fehler werden nicht verschwiegen, sondern
als Exception geworfen — Auswerter entscheiden (§V6 (VERBOTEN.md)).
"""

from __future__ import annotations

import logging

import numpy as np

from backend.core.audio_layout import mono_mix

logger = logging.getLogger(__name__)

#: Oktavband-Mitten 250 Hz … 8 kHz (IEC 60268-16 STI-Bänder).
OCTAVE_CENTER_HZ: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

#: IEC-Oktavband-Wichtigkeiten (Richtung; normiert über messbare Bänder).
STI_ALPHA: tuple[float, ...] = (0.085, 0.127, 0.230, 0.233, 0.309, 0.224)

#: IEC-Modulationsraten (Hz).
MOD_RATES_HZ: tuple[float, ...] = (
    0.63,
    0.8,
    1.0,
    1.25,
    1.6,
    2.0,
    2.5,
    3.15,
    4.0,
    5.0,
    6.3,
    8.0,
    10.0,
    12.5,
)

#: Konsonantenfenster: Bänder und Raten für die Klartext-Sicht.
CONSONANT_BANDS_HZ: tuple[float, ...] = (2000.0, 4000.0)
CONSONANT_RATES_HZ: tuple[float, ...] = (4.0, 5.0, 6.3, 8.0, 10.0, 12.5)

_MIN_SECONDS = 1.0
_EPS = 1e-12
_BIN_SEARCH = 2  # ±2 Fourier-Bins Toleranz um die Nominalrate
_BAND_ENERGY_FRACTION = 0.01  # Band muss ≥ 1 % der stärksten Band-Hülle tragen
_MIN_MOD_DEPTH = 1e-3  # unterhalb ist die Rate nicht anregbar genug


def _band_envelope(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    """Oktavband @fc → analytische Amplituden-Hüllkurve (deterministisch)."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    lo = fc / np.sqrt(2.0)
    hi = min(fc * np.sqrt(2.0), 0.49 * sr)
    sos = butter(4, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="bandpass", output="sos")
    band = sosfiltfilt(sos, x)
    return np.asarray(np.abs(hilbert(band)))  # type: ignore[no-any-return]


def _mod_depth(env: np.ndarray, rate_hz: float, sr: int) -> float:
    """Modulationstiefe m einer Hüllkurve bei rate_hz (Fourier-Peak ±2 Bins)."""
    if env.size < 32:
        return 0.0
    spec = np.abs(np.fft.rfft(env))
    k0 = int(round(rate_hz * env.size / sr))
    if k0 <= 0:
        return 0.0
    lo = max(1, k0 - _BIN_SEARCH)
    hi = min(spec.size - 1, k0 + _BIN_SEARCH)
    if hi < lo:
        return 0.0
    dc = float(spec[0])
    if dc <= _EPS:
        return 0.0
    peak = float(np.max(spec[lo : hi + 1]))
    return float(np.clip(2.0 * peak / dc, 0.0, 1.0))


def _ti_from_ratio(ratio: float) -> float:
    """Modulationsverhältnis → Transmission Index (IEC 60268-16, SNR_eff ±15 dB)."""
    r = float(np.clip(ratio, 0.0, 1.0))
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return 1.0
    snr_eff = 10.0 * np.log10(r / (1.0 - r))
    return float((np.clip(snr_eff, -15.0, 15.0) + 15.0) / 30.0)


def measure_modulation_transfer(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
) -> dict[str, object]:
    """Misst den Modulationserhalt original → restored (Kette, nicht Signal!).

    Rückgabe: ``sti`` (0–1), ``consonant_modulation`` (0–1), ``bands``
    (je Oktavband: center_hz/alpha/ti/mean_ratio; ti = None wenn nicht
    bewertbar) und ``rates_hz``. Level-Änderungen ohne Tiefenänderung sind
    KEIN Verlust (m ist normiert) — analog zur Hörordnungs-Kalibrierung
    „gleichfrequente Pegeländerung ist kein Fund".
    """
    if sr <= 0:
        raise ValueError("sr muss > 0 sein")
    orig = np.nan_to_num(np.asarray(mono_mix(original), dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    rest = np.nan_to_num(np.asarray(mono_mix(restored), dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(min(orig.size, rest.size))
    if n < int(_MIN_SECONDS * sr):
        raise ValueError(f"Signal zu kurz für MTF (min {_MIN_SECONDS:.0f} s)")
    orig, rest = orig[:n], rest[:n]

    envs_in = {fc: _band_envelope(orig, sr, fc) for fc in OCTAVE_CENTER_HZ}
    strongest = max((float(e.mean()) for e in envs_in.values()), default=0.0)
    energy_floor = _BAND_ENERGY_FRACTION * strongest

    bands: list[dict[str, object]] = []
    ti_values: list[float | None] = []
    cons_ratios: list[float] = []
    for fc, alpha in zip(OCTAVE_CENTER_HZ, STI_ALPHA):
        env_in = envs_in[fc]
        env_out = _band_envelope(rest, sr, fc)
        ratios: list[float] = []
        if float(env_in.mean()) >= energy_floor:
            for rate in MOD_RATES_HZ:
                m_in = _mod_depth(env_in, rate, sr)
                m_out = _mod_depth(env_out, rate, sr)
                if m_in <= _MIN_MOD_DEPTH:
                    continue  # Rate in diesem Band nicht anregbar — nicht bewertbar
                ratios.append(float(np.clip(m_out / m_in, 0.0, 1.0)))
                if fc in CONSONANT_BANDS_HZ and rate in CONSONANT_RATES_HZ:
                    cons_ratios.append(ratios[-1])
        ti: float | None = float(np.mean([_ti_from_ratio(r) for r in ratios])) if ratios else None
        ti_values.append(ti)
        bands.append(
            {
                "center_hz": float(fc),
                "alpha": float(alpha),
                "ti": round(ti, 4) if ti is not None else None,
                "mean_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
            }
        )

    # α-Normierung nur über MESSBARE Bänder — ein stilles Band ist „nicht
    # bewertbar", nicht „schlecht" (Musik trägt nicht alle Oktaven).
    weighed = [(a, t) for a, t in zip(STI_ALPHA, ti_values) if t is not None]
    alpha_sum = float(sum(a for a, _ in weighed)) or 1.0
    sti = float(sum(a * t for a, t in weighed) / alpha_sum) if weighed else 0.0
    consonant = float(np.mean(cons_ratios)) if cons_ratios else 0.0
    return {
        "sti": round(sti, 4),
        "consonant_modulation": round(consonant, 4),
        "bands": bands,
        "rates_hz": list(MOD_RATES_HZ),
    }


__all__ = [
    "CONSONANT_BANDS_HZ",
    "CONSONANT_RATES_HZ",
    "MOD_RATES_HZ",
    "OCTAVE_CENTER_HZ",
    "STI_ALPHA",
    "measure_modulation_transfer",
]

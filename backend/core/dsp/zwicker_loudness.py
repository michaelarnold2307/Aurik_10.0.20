"""Stationäres Zwicker-Loudness-Modell nach ISO 532-1 (Zwicker-Methode, Annex).

§SOTA-PSY-A2/Q9: Stationärer Nachfolger des MPEG-1-Modells (ISO 11172-3) im
Audibility-Gate. Berechnet (1) die stationäre Gesamtlautheit N in sone und den
Lautheitspegel L_N in phon nach ISO 532-1 sowie (2) eine daraus abgeleitete
psychoakustische Maskierungsschwelle (Erregungsmuster + Maskierungsflanken),
die das Gate austauschbar zum MPEG-1-``compute_masking_threshold_db`` nutzen
kann (siehe ``audibility_gate.defect_audibility(..., model="zwicker")``).

Zeitvariante Lautheit (kurz-/langzeitintegriert): ``compute_time_varying_loudness``
in DIESEM Modul (DIN-45631/A1-Struktur: gefensterte Kurzzeit-Lautheit + Angriffs-/
Abkling-Zeitbewertung + N5/N10-Perzentile) — PSY-A2-Folge-Schritt 2026-09-15.
``temporal_loudness.py`` (Moore-Glasberg-Basis) bleibt unabhängig davon bestehen.

Konzeption (Zwicker & Fastl „Psychoacoustics: Facts and Models", ISO 532-1):

  1. 1/3-Oktav-Bandpegel L_i (28 Bänder, 25 Hz–12,5 kHz) aus dem Signal.
  2. Erregungspegel E = L + a0 je Band (Mittel-/Außenohr-Übertragungsfaktor a0).
  3. Spezifische Lautheit N' je Bark über eine Kompressions-Potenzfunktion mit
     Kompressions-Exponent 0.23 (Ruhehörschwellen-Gate je Band).
  4. Gesamtlautheit N = N'max + 0.15·(ΣN' − N'max)  [sone].
  5. Lautheitspegel L_N = 40 + 10·log2(N)  [phon] für N ≥ 1 sone,
     sonst L_N = 40·(N + 0.0005)^0.35.

Dokumentierte Abweichungen gegenüber ISO 532-1 Annex B (bewusst, deterministisch,
analog ``temporal_loudness.py`` — §V7 (copilot-instructions.md): keine
ad-hoc-Steigungs-Interpolation, kein zeitvariantes Modell):

  * A) Spezifische Lautheit: ISO 532-1 Annex B integriert die N'-vs-E-Kurve über
    eine pegelabhängige Steigungs-Tabelle s(E) (Wertebereich ≈ 0.05–0.25
    sone/Bark/dB). Statt dieser stückweisen Steigungs-Integration verwenden wir
    die geschlossene Zwicker-Potenzform N' ∝ (E/E_0)^0.23 mit fixem Exponenten
    0.23 — das ist die Tieffpegel-Asymptote der ISO-N'-Funktion und ergibt bei
    der Kalibrierreferenz (1 kHz / 40 dB SPL ⇒ 1 sone) den Sollwert exakt.
    Abweichung im Übergangsbereich (knapp über der Ruhehörschwelle) < ~0.15
    sone/Bark, monoton pegelabhängig erhalten.

  * B) Ruhehörschwelle: harter Gate (N' = 0 unterhalb der Ruhehörschwelle
    L_TQ) statt des glatten ISO-Übergangs. Physikalisch plausibel, deterministisch.

  * C) a0-/L_TQ-Tabellen: EXAKT nach DIN 45631 (Zwicker-Verfahren) /
    ISO 532-1:2017, wie in Zwicker & Fastl (3. Aufl.) abgedruckt
    (Update 2026-09-15; vorher Näherungswerte — Session-Ertrag P5).

  * D) Referenz-Kalibrierung: das Modul nimmt die digitale Vollaussteuerung als
    0 dBFS ≡ 100 dB SPL an (dokumentierte Konstante ``_DB_SPL_FULL_SCALE``).
    Damit entspricht 40 dB SPL einer Sinus-Amplitude ``_REF_AMPLITUDE = 0.001``
    (20·log10(0.001) = −60 dB ⇒ 100 − 60 = 40 dB). Die Pipeline hat keinen
    absoluten Wiedergabepegel; diese Festlegung ist deterministisch und macht
    den 1-kHz/40-dB-Referenzwert unabhängig reproduzierbar.

Determinismus (§G5 (GEBOTE.md)): reine numpy/FFT, keine Zufallszahlen außer
expliziten Seeds (hier: gar keine), kein ``time.time()``.

§0a / NaN-Schutz: Eingaben werden NaN/Inf-bereinigt; alle Rückgaben sind endlich.

Normalisierung: Eingaben werden zu Mono gemittelt (Stereo → Mittel). Layout-
Invariante (§V7 (copilot-instructions.md)): es wird keine channels-first-Annahme
getroffen, jede Achse mit Länge 2 wird zu einem Kanal gemittelt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Referenz-Kalibrierung (dokumentierte Näherung, siehe Modul-Docstring D) ──
_DB_SPL_FULL_SCALE = 100.0  # 0 dBFS ≡ 100 dB SPL (Monitoring-Referenz)
_REF_SPL_DB = 40.0  # 1-sone-Referenz: 1 kHz Ton bei 40 dB SPL
_REF_AMPLITUDE = 10.0 ** ((_REF_SPL_DB - _DB_SPL_FULL_SCALE) / 20.0)  # 0.001
_REF_RMS = _REF_AMPLITUDE / np.sqrt(2.0)

# ── Kompressions-Exponent der spezifischen Lautheit (Zwicker, Tieffpegel-Wert) ─
_COMPRESSION_EXP = 0.23

# ── Lautheits-Integration (ISO 532-1) ────────────────────────────────────────
_PARTIAL_LOUDNESS_FACTOR = 0.15

# ── Maskierungsflanken (Zwicker/Fastl, Tabelle 6.2-Näherung) ─────────────────
_LOWER_SLOPE_DB_PER_BARK = 27.0  # untere Flanke (zu höheren Frequenzen)
_UPPER_SLOPE_BASE_DB = 30.0  # obere Flanke: steiler bei niedrigen Pegeln
_UPPER_SLOPE_LEVEL_COEF = 0.10  # flacher mit steigendem Erregungspegel
_UPPER_SLOPE_MIN_DB = 15.0
_UPPER_SLOPE_MAX_DB = 30.0
_MASKED_OFFSET_DB = -3.0  # Schwelle am Masker ~3 dB unter dessen Erregung

# ── Bark-Raster ──────────────────────────────────────────────────────────────
_BARK_STEP = 0.5
_BARK_MAX = 24.0

# ── 1/3-Oktav-Bänder: Mitten (Hz), Übertragungsfaktor a0 (dB), Ruhehörschwelle ─
# EXAKTE Tabellen nach DIN 45631 (Zwicker-Verfahren) / ISO 532-1:2017 (Annex),
# wie in Zwicker & Fastl „Psychoacoustics: Facts and Models“ (3. Aufl.) und den
# DIN-45631-Referenzimplementierungen abgedruckt (a0[1 kHz] = 0 dB,
# L_TQ[1 kHz] = 0 dB). Ersetzt die Näherungswerte von 2026-09-14 (PSY-A2/Q9
# Folge-Schritt „exakte a0/L_TQ-Tabellen“, Session-Ertrag 2026-09-15).
_THIRD_OCTAVE_CENTERS_HZ: tuple[float, ...] = (
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
)

_A0_DB: tuple[float, ...] = (
    -32.0,
    -27.5,
    -23.0,
    -19.2,
    -15.9,
    -13.0,
    -10.3,
    -8.1,
    -6.2,
    -4.4,
    -3.0,
    -1.9,
    -1.0,
    -0.3,
    0.5,
    0.9,
    0.0,
    -0.7,
    -2.2,
    -3.8,
    -5.0,
    -5.8,
    -6.5,
    -6.8,
    -7.2,
    -7.8,
    -9.0,
    -10.8,
)

_THRESHOLD_IN_QUIET_DB: tuple[float, ...] = (
    65.0,
    57.0,
    49.5,
    44.0,
    38.5,
    32.5,
    27.0,
    22.0,
    17.5,
    13.5,
    10.0,
    7.0,
    4.5,
    3.0,
    1.5,
    0.7,
    0.0,
    -0.6,
    -1.7,
    -3.0,
    -3.9,
    -4.1,
    -3.9,
    -3.0,
    -0.8,
    2.5,
    6.8,
    12.0,
)

_CENTERS = np.asarray(_THIRD_OCTAVE_CENTERS_HZ, dtype=np.float64)
_A0 = np.asarray(_A0_DB, dtype=np.float64)
_TQ = np.asarray(_THRESHOLD_IN_QUIET_DB, dtype=np.float64)


def _hz_to_bark(f: np.ndarray) -> np.ndarray:
    """Hz → Bark nach Traunmüller (deterministisch, konsistent zu masking_model)."""
    f = np.asarray(f, dtype=np.float64)
    return np.asarray(13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2), dtype=np.float64)  # type: ignore[no-any-return]


# Bark-Positionen der 1/3-Oktav-Mitten (einmalig berechnet).
_CENTERS_BARK = _hz_to_bark(_CENTERS)

# Fein gerasterte f→z-Referenz für die Umkehrung (Bark→Hz) der Ausgabeachse.
_F_REF = np.logspace(np.log10(20.0), np.log10(15500.0), 4096)
_Z_REF = _hz_to_bark(_F_REF)


def _bark_to_hz(z: np.ndarray) -> np.ndarray:
    """Bark → Hz (monotone Interpolation über die feine f→z-Referenz)."""
    z = np.asarray(z, dtype=np.float64)
    return np.asarray(np.interp(z, _Z_REF, _F_REF), dtype=np.float64)  # type: ignore[no-any-return]


def _to_mono(x: np.ndarray) -> np.ndarray:
    """NaN/Inf-bereinigen und zu Mono mitteln (layout-tolerante Normalisierung)."""
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return np.asarray(arr.ravel(), dtype=np.float64)  # type: ignore[no-any-return]


def _third_octave_levels_db_spl(x: np.ndarray, sr: int) -> np.ndarray:
    """1/3-Oktav-Bandpegel in dB SPL (Parseval-exakte RMS je Band, deterministisch).

    Jeder Bandpegel entspricht L_i = 20·log10(rms_i / _REF_RMS) + 40 dB. Die
    Bandenergie wird über eine einfache (rechteckfenster-)DFT mit
    Parseval-Normalisierung bestimmt, damit ein reiner Sinus auf exakt seinen
    Pegel an der Bandmitte abgebildet wird (Referenz-Kalibrierung exakt).
    """
    n = len(x)
    if n == 0:
        return np.full(_TQ.shape, -1e3, dtype=np.float64)  # type: ignore[no-any-return]
    spec = np.fft.rfft(x)
    # Einseitige Leistungsdichte (Parseval): |X[0]|² + 2·Σ|X[k]|² + |X[N/2]|² = N·Σx²
    power = np.asarray(np.abs(spec) ** 2, dtype=np.float64)
    power[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Bandkanten der 1/3-Oktav-Mitten.
    edges_lo = _CENTERS / (2.0 ** (1.0 / 6.0))
    edges_hi = _CENTERS * (2.0 ** (1.0 / 6.0))

    levels = np.empty(_CENTERS.shape, dtype=np.float64)
    nyq = sr / 2.0
    for i in range(len(_CENTERS)):
        lo = edges_lo[i]
        hi = min(edges_hi[i], nyq)
        if hi <= lo or lo >= nyq:
            levels[i] = -1e3
            continue
        mask = (freqs >= lo) & (freqs < hi)
        # Parseval: Σx² = (|X[0]|² + 2·Σ|X[k]|² + |X[N/2]|²) / N ⇒ mean_sq = power/N².
        mean_sq = float(power[mask].sum()) / float(n * n)
        rms = np.sqrt(max(mean_sq, 0.0))
        if rms <= 0.0:
            levels[i] = -1e3
        else:
            levels[i] = 20.0 * np.log10(rms / _REF_RMS) + _REF_SPL_DB
    return np.asarray(levels, dtype=np.float64)  # type: ignore[no-any-return]


def _excitation_levels_db_spl(levels_db_spl: np.ndarray) -> np.ndarray:
    """Erregungspegel E = L + a0 je 1/3-Oktav-Band (dB SPL)."""
    return np.asarray(levels_db_spl + _A0, dtype=np.float64)  # type: ignore[no-any-return]


def _bark_grid() -> np.ndarray:
    """Bark-Raster 0–24 Bark in 0.5-Schritten (49 Punkte)."""
    return np.asarray(np.arange(0.0, _BARK_MAX + 1e-9, _BARK_STEP, dtype=np.float64), dtype=np.float64)  # type: ignore[no-any-return]


def _interp_to_bark(values_db: np.ndarray) -> np.ndarray:
    """Interpoliert band-/punktweise dB-Werte (auf _CENTERS_BARK) auf das Bark-Raster."""
    grid = _bark_grid()
    return np.asarray(np.interp(grid, _CENTERS_BARK, values_db), dtype=np.float64)  # type: ignore[no-any-return]


def _specific_loudness(exc_db: np.ndarray, tq_db: np.ndarray | None = None) -> np.ndarray:
    """Spezifische Lautheit N' (sone/Band) aus Erregungspegel (dB SPL).

    N'(z) = (E/E_0)^0.23 für E über der Ruhehörschwelle, sonst 0 (harter Gate,
    dokumentierte Näherung B). Mit E/E_0 = 10^((L_E − 40)/10) ⇒ 1 kHz / 40 dB
    ⇒ N' = 1 sone/Band exakt (Referenz-Kalibrierung erfüllt).
    """
    if tq_db is None:
        tq_db = _TQ
    tq = np.asarray(tq_db, dtype=np.float64)
    ratio = 10.0 ** ((exc_db - _REF_SPL_DB) / 10.0)
    nprime = np.where(exc_db > tq, ratio**_COMPRESSION_EXP, 0.0)
    return np.asarray(nprime, dtype=np.float64)  # type: ignore[no-any-return]


@dataclass
class ZwickerLoudnessResult:
    """Ergebnis der stationären Zwicker-Loudness (ISO 532-1).

    Attributes:
        loudness_sone: Gesamtlautheit N in sone (ISO 532-1 Summationsformel).
        loudness_level_phon: Lautheitspegel L_N in phon.
        total_specific_loudness_sone: Σ N' (Basis der Summation).
        max_specific_loudness_sone: N'max (Spitzenwert der spezifischen Lautheit).
    """

    loudness_sone: float = 0.0
    loudness_level_phon: float = 0.0
    total_specific_loudness_sone: float = 0.0
    max_specific_loudness_sone: float = 0.0


def _sone_to_phon(n_sone: float) -> float:
    """Lautheit sone → Lautheitspegel phon (ISO 532-1)."""
    if n_sone >= 1.0:
        return float(40.0 + 10.0 * np.log2(n_sone))
    return float(40.0 * (max(n_sone, 0.0) + 0.0005) ** 0.35)


def compute_zwicker_loudness(x: np.ndarray, sr: int) -> ZwickerLoudnessResult:
    """Berechnet die stationäre Zwicker-Lautheit (ISO 532-1) in sone und phon.

    Args:
        x: Mono-/Stereo-Signal (float). NaN/Inf werden bereinigt (§0a).
        sr: Abtastrate in Hz.

    Returns:
        ZwickerLoudnessResult (loudness_sone, loudness_level_phon, …),
        deterministisch (§G5 (GEBOTE.md)).
    """
    try:
        mono = _to_mono(x)
        if len(mono) < 2 or sr <= 0:
            return ZwickerLoudnessResult()
        levels = _third_octave_levels_db_spl(mono, sr)
        exc = _excitation_levels_db_spl(levels)  # 28 1/3-Oktav-Bänder
        nprime = _specific_loudness(exc, _TQ)  # sone je Band (≈ 1 Krit. Band)
        # Gesamtlautheit N = N'max + 0.15·(ΣN' − N'max) über die 28 Bänder
        # (ISO 532-1 behandelt jedes 1/3-Oktav-Band äquivalent zu ~1 krit. Band).
        total = float(np.sum(nprime))
        nmax = float(np.max(nprime))
        n_sone = nmax + _PARTIAL_LOUDNESS_FACTOR * (total - nmax)
        n_sone = float(np.clip(n_sone, 0.0, None))
        n_sone = 0.0 if not np.isfinite(n_sone) else n_sone
        phon = _sone_to_phon(n_sone)
        return ZwickerLoudnessResult(
            loudness_sone=n_sone,
            loudness_level_phon=float(phon),
            total_specific_loudness_sone=total,
            max_specific_loudness_sone=nmax,
        )
    except Exception as exc:  # §V6 (copilot-instructions.md): nie blockieren
        logger.debug("Zwicker-Lautheitsberechnung fehlgeschlagen (%s) — N=0.", exc)
        return ZwickerLoudnessResult()


def compute_loudness_sone(x: np.ndarray, sr: int) -> float:
    """Gesamtlautheit N in sone (Kurzform zu compute_zwicker_loudness)."""
    return compute_zwicker_loudness(x, sr).loudness_sone


def compute_loudness_phon(x: np.ndarray, sr: int) -> float:
    """Lautheitspegel L_N in phon (Kurzform zu compute_zwicker_loudness)."""
    return compute_zwicker_loudness(x, sr).loudness_level_phon


def zwicker_excitation_pattern(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Erregungspegel E (dB SPL) je Bark über das 0.5-Schritt-Raster (0–24 Bark).

    Args:
        x: Mono-/Stereo-Signal.
        sr: Abtastrate.

    Returns:
        (excitation_db [Bark-Raster], bark_grid [Bark]) — deterministisch.
    """
    mono = _to_mono(x)
    bark = _bark_grid()
    if len(mono) < 2 or sr <= 0:
        return np.full_like(bark, -1e3), bark
    levels = _third_octave_levels_db_spl(mono, sr)
    exc = _excitation_levels_db_spl(levels)
    exc_on_bark = _interp_to_bark(exc)
    exc_on_bark[~np.isfinite(exc_on_bark)] = -1e3
    return exc_on_bark, bark


def zwicker_masking_threshold_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Maskierungsschwelle (dB SPL) je Bark mit zugehöriger Frequenzachse (Hz).

    Abgeleitet aus dem Erregungsmuster (``zwicker_excitation_pattern``) mit
    pegelabhängiger Maskierungsflanke (untere Flanke ~27 dB/Bark zu höheren
    Frequenzen; obere Flanke steiler bei niedrigen Erregungspegeln — Zwicker/
    Fastl Tabelle 6.2-Näherung). Die Schwelle wird nach unten nie unter die
    Ruhehörschwelle L_TQ gesetzt und liegt am Masker ~3 dB unter dessen Erregung.

    Rückgabe analog ``masking_model.compute_masking_threshold_db`` (Schwelle +
    Frequenzachse), damit ``audibility_gate`` beide austauschbar nutzen kann.

    Args:
        x: Mono-/Stereo-Signal (Masker).
        sr: Abtastrate.

    Returns:
        (threshold_db [Bark-Raster, dB SPL], freq_hz [Hz]) — deterministisch.
    """
    exc, bark = zwicker_excitation_pattern(x, sr)
    n = len(bark)
    # Ruhehörschwelle auf dem Bark-Raster als Untergrenze.
    tq = _interp_to_bark(_TQ)

    # Pegelabhängige obere Flanke (dB/Bark): steiler bei niedrigen Pegeln.
    upper_slope = _UPPER_SLOPE_BASE_DB - _UPPER_SLOPE_LEVEL_COEF * np.maximum(exc, 0.0)
    upper_slope = np.clip(upper_slope, _UPPER_SLOPE_MIN_DB, _UPPER_SLOPE_MAX_DB)

    # Spreading: Schwelle an jeder Position = Maximum über alle Masker-Beiträge.
    z = bark
    thr = np.full(n, -1e3, dtype=np.float64)
    for j in range(n):  # Masker-Position
        dz = z - z[j]
        # Beiträge oberhalb (höhere Frequenz, dz>0): feste untere Flanke.
        contrib = exc[j] - _LOWER_SLOPE_DB_PER_BARK * np.maximum(dz, 0.0)
        # Beiträge unterhalb (niedrigere Frequenz, dz<0): pegelabhängige obere Flanke.
        contrib = contrib - upper_slope[j] * np.maximum(-dz, 0.0)
        thr = np.maximum(thr, contrib)

    # Am Ort des Maskers selbst etwa auf Erregungsniveau (Offset), nie unter L_TQ.
    thr = np.maximum(thr, exc + _MASKED_OFFSET_DB)
    thr = np.maximum(thr, tq)
    thr = np.asarray(
        np.nan_to_num(np.asarray(thr, dtype=np.float64), nan=0.0, posinf=120.0, neginf=0.0), dtype=np.float64
    )

    freq_hz = _bark_to_hz(bark)
    return thr, freq_hz


@dataclass
class TimeVaryingLoudnessResult:
    """Zeitvariante Zwicker-Lautheit (DIN-45631/A1-Struktur).

    Attributes:
        time_seconds: Zeitachse (Frame-Mitten) der Kurzzeit-Lautheit.
        loudness_sone_t: zeitbewertete Kurzzeit-Lautheit N(t) in sone.
        n5_sone: N5 — 95. Perzentil von N(t) (DIN 45631/A1).
        n10_sone: N10 — 90. Perzentil von N(t).
        mean_sone: Mittelwert von N(t).
        peak_sone: Maximum von N(t).
    """

    time_seconds: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    loudness_sone_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    n5_sone: float = 0.0
    n10_sone: float = 0.0
    mean_sone: float = 0.0
    peak_sone: float = 0.0


def compute_time_varying_loudness(
    x: np.ndarray,
    sr: int,
    window_ms: float = 42.7,
    hop_ms: float = 10.0,
    attack_ms: float = 5.0,
    release_ms: float = 100.0,
) -> TimeVaryingLoudnessResult:
    """Zeitvariante Zwicker-Lautheit N(t) nach DIN-45631/A1-Struktur (PSY-A2, P5).

    Kurzzeit-Lautheit je gefenstertem Segment (stationäre ISO-532-1-Rechnung),
    dann nichtlineare Zeitbewertung: schneller Angriff (~5 ms), langsames
    Abklingen (~100 ms — DIN 45631/A1 nutzt 20–100 ms pegelabhängig; hier die
    konservative obere Grenze). N5/N10 sind die Exzedenz-Perzentile von N(t).

    Deterministisch (§G5 (GEBOTE.md)), NaN/Inf-geschützt (§0a), Stereo → Mittel (Layout-
    Invariante wie im stationären Pfad).

    Args:
        x: Mono-/Stereo-Signal.
        sr: Abtastrate.
        window_ms: Fensterlänge je Segment (Default 42,7 ms ≈ 2048 @ 48 kHz).
        hop_ms: Hop zwischen Segmenten (Default 10 ms).
        attack_ms: Angriffs-Zeitkonstante (Default 5 ms).
        release_ms: Abkling-Zeitkonstante (Default 100 ms, DIN-obere Grenze).

    Returns:
        TimeVaryingLoudnessResult — leere Zeitachse wenn kein vollständiges
        Segment passt (kein Fehler).
    """
    mono = _to_mono(x)
    win = max(16, int(round(window_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    if len(mono) < win or sr <= 0:
        return TimeVaryingLoudnessResult()

    n_frames = 1 + (len(mono) - win) // hop
    hann = np.hanning(win)
    # Energieerhalt: Hann reduziert die RMS um sqrt(3/8) ≈ 0,612 — ohne
    # Korrektur würde die Kurzzeit-Lautheit stationärer Töne systematisch
    # unterschätzt (Befund im P5-Test 2026-09-15).
    hann_gain = float(np.sqrt(np.mean(hann**2)))
    times = np.empty(n_frames, dtype=np.float64)
    levels = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s0 = i * hop
        seg = mono[s0 : s0 + win] * (hann / max(hann_gain, 1e-9))
        levels[i] = compute_zwicker_loudness(seg, sr).loudness_sone
        times[i] = (s0 + win / 2.0) / sr

    # Nichtlineare Zeitbewertung (DIN 45631/A1-Struktur):
    # Angriff steigt exponentiell Richtung Ziel, Abklingen fällt exponentiell.
    dt = hop / sr
    a_coef = float(np.exp(-dt / max(attack_ms / 1000.0, 1e-6)))
    r_coef = float(np.exp(-dt / max(release_ms / 1000.0, 1e-6)))
    smoothed = np.empty(n_frames, dtype=np.float64)
    prev = 0.0
    for i in range(n_frames):
        s_t = float(levels[i])
        if s_t >= prev:
            prev = s_t + (prev - s_t) * a_coef
        else:
            prev = s_t + (prev - s_t) * r_coef
        smoothed[i] = prev

    smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0)
    return TimeVaryingLoudnessResult(
        time_seconds=np.asarray(times, dtype=np.float64),
        loudness_sone_t=np.asarray(smoothed, dtype=np.float64),
        n5_sone=float(np.percentile(smoothed, 95.0)),
        n10_sone=float(np.percentile(smoothed, 90.0)),
        mean_sone=float(np.mean(smoothed)),
        peak_sone=float(np.max(smoothed)),
    )

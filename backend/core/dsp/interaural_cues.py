#!/usr/bin/env python3
"""backend/core/dsp/interaural_cues.py — HRTF-/interaurale Cue-Messung (SOTA-Basis).

Hörfähigkeits-SOTA (psychoakustisch validierte JNDs, Quellen s. u.):
  - ITD (interaural time difference):  JND ~10–20 µs (Clicks), ~20–40 µs (Töne
    1 kHz) — Klumpp & Eady (1956), Mossop & Culling (2008).
  - ILD (interaural level difference): JND ~1 dB (mittelbandig) — Mills (1960).
  - IACC (interaural cross-correlation): Δ > 0.08–0.10 als „hörbar breiter/
    schmaler" — Mason, Ford & Rumsey (2005).
  - BMLD (binaural masking level difference): N0S0 vs NπS0 liefert bis zu
    ~15 dB Schwellwert-Vorteil bei 500 Hz — Levitt & Rabiner (1967), Durlach
    (1972). Erklärt, warum mono-kollabierte Restaurierungen selbst bei
    unverändertem SNR hörbar schlechter wirken.

Modelle (deterministisch, bit-reproduzierbar — §G5 copilot-instructions.md):
  - ITD: Kreuzkorrelation mit parabolischer Sub-Sample-Interpolation
    (Jacovitti & Scarano 1993), Fenster-weise Median für Robustheit.
  - ILD: mittelbandige Pegeldifferenz (1/3-Oktav-Bandenergie, 250 Hz–4 kHz,
    gewichtet) in dB.
  - IACC: Maximum der normalisierten Kreuzkorrelation (|ρ|), Standarddefinition.
  - Kopfschatten: sphärisches Kopfmodell nach Duda & Martens (1998)
    „Range dependence of the response of a spherical head model" — ILD(f, θ).
  - Woodworth-Formel: ITD(θ) = (a/c)·(θ + sin θ), a = 8.75 cm Kopfdistanz —
    Woodworth (1938), Kuhn (1977) für den Schallschatten-Term.

Ehrlicher Scope (dokumentiert): Dies ist das deterministische Sphärenkopf-Modell
(First-Order-HRTF-Cues ITD/ILD), KEIN gemessenes individualisiertes HRIR-Set
(CIPIC/KEMAR). Individualisierte HRIR-Faltung ist ein Plug-in-Thema
(plugins/_vendor_*); dieses Modul liefert die normativen psychoakustischen
Cue-Metriken und den Guard-Datentyp, an dem jede HRIR-Erweiterung gemessen wird.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

from backend.core.audio_layout import mono_mix, to_channels_first

logger = logging.getLogger(__name__)

# ── Psychoakustische JND-Konstanten (Literaturwerte, s. Modul-Docstring) ────
ITD_JND_US: float = 30.0  # konservativ (Töne ~20–40 µs; Clicks ~10–20 µs)
ITD_HARD_FAIL_US: float = 60.0  # 2× JND — hörbare Verschiebung der Phantomquelle
ILD_JND_DB: float = 1.0
ILD_HARD_FAIL_DB: float = 4.0
IACC_JND: float = 0.08
IACC_HARD_FAIL: float = 0.15

# Kopfgeometrie (Kuhn 1977 / Duda & Martens 1998)
HEAD_RADIUS_M: float = 0.0875
SPEED_OF_SOUND_M_S: float = 343.0


@dataclass
class InterauralCueProfile:
    """Interaurale Cue-Werte eines Signals (referenzierbar in Guards)."""

    is_stereo: bool
    itd_us: float  # ITD = Ankunft links − rechts; negativ = rechts eilt nach
    ild_db: float  # mittelbandige L/R-Pegeldifferenz (L − R)
    iacc: float  # interaurale Kreuzkorrelation |ρ| ∈ [0, 1]
    itd_jitter_us: float  # Fenster-zu-Fenster-Streuung des ITD (Robustheit)
    lowband_corr: float  # Korrelation 250 Hz–1 kHz (BMLD-relevantes Band)


@dataclass
class InterauralIntegrityResult:
    """Vergleich Original vs. restauriert gegen JND-Schwellen."""

    original: InterauralCueProfile
    restored: InterauralCueProfile
    itd_drift_us: float  # |restored − original| — positiv = verschlechtert
    ild_drift_db: float
    iacc_delta: float  # restored − original (positiv = Richtung Mono)
    bmld_advantage_kept_db: float  # ≥0: BMLD-Vorteil erhalten
    itd_ok: bool
    ild_ok: bool
    iacc_ok: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "original": {
                "is_stereo": self.original.is_stereo,
                "itd_us": round(self.original.itd_us, 3),
                "ild_db": round(self.original.ild_db, 3),
                "iacc": round(self.original.iacc, 4),
            },
            "restored": {
                "is_stereo": self.restored.is_stereo,
                "itd_us": round(self.restored.itd_us, 3),
                "ild_db": round(self.restored.ild_db, 3),
                "iacc": round(self.restored.iacc, 4),
            },
            "itd_drift_us": round(self.itd_drift_us, 3),
            "ild_drift_db": round(self.ild_drift_db, 3),
            "iacc_delta": round(self.iacc_delta, 4),
            "bmld_advantage_kept_db": round(self.bmld_advantage_kept_db, 3),
            "itd_ok": bool(self.itd_ok),
            "ild_ok": bool(self.ild_ok),
            "iacc_ok": bool(self.iacc_ok),
        }


def _itd_from_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[float, float]:
    """ITD in µs + Peak-Korrelation via parabolischer Sub-Sample-Interpolation.

    Jacovitti & Scarano (1993): Diskrete Kreuzkorrelations-Peaks quadratisch
    interpolieren → Sub-Sample-ITD ohne Resampling, bit-deterministisch.
    """
    n = min(len(x), len(y))
    if n < 4:
        return 0.0, 1.0
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom < 1e-12:
        return 0.0, 1.0
    # §V08/§10a-konform: FFT-basierte Kreuzkorrelation statt O(n²) np.correlate.
    # np.correlate hat ab ~10k Samples einen harten Kosten-Cliff (Direkt-Faltung):
    # 30 s @ 48 kHz (1.44 M Samples) ≈ 2·10¹² MACs → 30–60 min je Aufruf.
    # Produktionsbefund 2026-09-11: Pipeline-Hänger nach der ersten ausgeführten
    # Phase im §2.51a-Stereo-Guard (Stack: np.correlate auf Vollsignal-IACC).
    if n > 8192:
        from backend.core.core_utils import fft_crosscorr

        corr = fft_crosscorr(x, y) / denom
    else:
        corr = np.correlate(x, y, mode="full") / denom
    lag_axis = np.arange(-(n - 1), n, dtype=np.int64)
    max_lag = min(max_lag, n - 1)
    mask = np.abs(lag_axis) <= max_lag
    idx = int(np.argmax(np.abs(corr[mask])))
    peak = float(corr[mask][idx])
    lag_f = float(lag_axis[mask][idx])
    # Parabolische Interpolation um den diskreten Peak (nur innere Lags)
    if 0 < idx < int(mask.sum()) - 1:
        c_prev = float(corr[mask][idx - 1])
        c_here = float(corr[mask][idx])
        c_next = float(corr[mask][idx + 1])
        denom_p = c_prev - 2.0 * c_here + c_next
        if abs(denom_p) > 1e-12:
            lag_f = lag_f + 0.5 * (c_prev - c_next) / denom_p
    return lag_f, float(peak)


def compute_interaural_profile(audio: np.ndarray, sr: int) -> InterauralCueProfile:
    """Berechnet ITD/ILD/IACC eines Stereo- oder Mono-Signals.

    §V7 (copilot-instructions.md): layout-sicher via audio_layout — kein
    hartes (N,2)/(2,N)-Annehmen. Mono → itd=0, ild=0, iacc=1.
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return InterauralCueProfile(False, 0.0, 0.0, 1.0, 0.0, 1.0)
    if arr.ndim != 2:
        return InterauralCueProfile(False, 0.0, 0.0, 1.0, 0.0, 1.0)
    ch = to_channels_first(arr)
    if ch.shape[0] < 2:
        mono = mono_mix(ch)
        return InterauralCueProfile(False, 0.0, 0.0, 1.0, 0.0, _corr_coef(mono, mono))
    left = ch[0]
    right = ch[1]

    # ── IACC: Maximum der normalisierten Kreuzkorrelation (Standarddefinition) ──
    max_lag_samples = int(round(0.001 * sr))  # ±1 ms Suchfenster
    itd_samples, iacc = _itd_from_cross_correlation(left, right, max_lag_samples)
    iacc = abs(iacc)

    # ── ITD über Fenster-Median (robust gegen einzelne fehlgetriggerte Fenster) ──
    win = max(int(round(0.05 * sr)), 256)  # 50 ms Fenster
    hop = max(win // 2, 1)
    n = len(left)
    itds: list[float] = []
    for start in range(0, max(n - win + 1, 1), hop):
        _l = left[start : start + win]
        _r = right[start : start + win]
        if float(np.std(_l)) < 1e-8 or float(np.std(_r)) < 1e-8:
            continue
        _lag, _ = _itd_from_cross_correlation(_l, _r, max_lag_samples)
        itds.append(_lag / sr * 1e6)
    if itds:
        itd_us = float(np.median(itds))
        itd_jitter_us = float(np.median(np.abs(np.asarray(itds) - itd_us)))
    else:
        itd_us = itd_samples / sr * 1e6
        itd_jitter_us = 0.0

    # ── ILD: mittelbandige Pegeldifferenz (250 Hz–4 kHz, 1/3-Oktav-artig) ──
    ild_db = _band_limited_ild_db(left, right, sr)

    # ── Lowband-Korrelation (BMLD-relevantes Band 250 Hz–1 kHz) ──
    lowband_corr = _band_limited_corr(left, right, sr, 250.0, 1000.0)

    return InterauralCueProfile(
        is_stereo=True,
        itd_us=float(itd_us),
        ild_db=float(ild_db),
        iacc=float(iacc),
        itd_jitter_us=float(itd_jitter_us),
        lowband_corr=float(lowband_corr),
    )


def _band_limited_ild_db(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """Mittelbandige ILD = 10·log10(Σ E_L(band)/Σ E_R(band)) über 1/3-Oktaven.

    Bänder: Mitten 315, 500, 800, 1250, 2000, 3150 Hz (je 1/3 Oktave),
    linear gewichtet — ILD-JND ~1 dB gilt mittelbandig (Mills 1960).
    """
    centers = [315.0, 500.0, 800.0, 1250.0, 2000.0, 3150.0]
    e_l = 0.0
    e_r = 0.0
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    for fc in centers:
        if fc * 2 ** (1 / 6) >= nyq:
            continue
        lo = fc / (2 ** (1 / 6))
        hi = fc * (2 ** (1 / 6))
        try:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            e_l += float(np.sum(sosfiltfilt(sos, left) ** 2))
            e_r += float(np.sum(sosfiltfilt(sos, right) ** 2))
        except ValueError:
            continue
    if e_l < 1e-18 and e_r < 1e-18:
        return 0.0
    return float(10.0 * np.log10((e_l + 1e-12) / (e_r + 1e-12)))


def _band_limited_corr(left: np.ndarray, right: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    """Pearson-Korrelation in einem Frequenzband (für BMLD-Proxy)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if f_hi >= nyq or f_lo <= 0:
        return 0.0
    try:
        sos = butter(4, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
        l_b = sosfiltfilt(sos, left)
        r_b = sosfiltfilt(sos, right)
    except ValueError:
        return 0.0
    return _corr_coef(l_b, r_b)


def _corr_coef(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-12:
        return 1.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def interaural_cue_integrity(original: np.ndarray, restored: np.ndarray, sr: int) -> InterauralIntegrityResult:
    """Original vs. restauriert gegen psychoakustische JNDs prüfen.

    Delta-basiert (AGENTS.md Guard-Kalibrierung): Nur eine VERSCHLECHTERUNG
    gegenüber dem Input ist ein Befund — absolute Werte (z. B. ITD aus
    analogem Träger-Versatz) sind kein Hard-Fail.
    """
    prof_orig = compute_interaural_profile(original, sr)
    prof_rest = compute_interaural_profile(restored, sr)

    if not prof_rest.is_stereo:
        # Mono-Kollaps: alle Cues verloren → maximaler Befund.
        return InterauralIntegrityResult(
            original=prof_orig,
            restored=prof_rest,
            itd_drift_us=1e6,
            ild_drift_db=1e3,
            iacc_delta=1.0,
            bmld_advantage_kept_db=0.0,
            itd_ok=False,
            ild_ok=False,
            iacc_ok=False,
        )

    itd_drift = abs(prof_rest.itd_us - prof_orig.itd_us)
    ild_drift = abs(prof_rest.ild_db - prof_orig.ild_db)
    iacc_delta = prof_rest.iacc - prof_orig.iacc  # >0 → Richtung Mono

    # BMLD-Proxy: Lowband-Korrelation 250 Hz–1 kHz muss erhalten bleiben.
    # N0S0 (korrelierter Maskierer) vs NπS0 (unkorrelierter) — der
    # Freisetzungseffekt sinkt monoton mit steigender Interaural-Korrelation.
    _bml_orig = 15.0 * (1.0 - abs(prof_orig.lowband_corr))
    _bml_rest = 15.0 * (1.0 - abs(prof_rest.lowband_corr))
    bmld_kept = float(np.clip(_bml_rest - _bml_orig, -15.0, 15.0))

    return InterauralIntegrityResult(
        original=prof_orig,
        restored=prof_rest,
        itd_drift_us=float(itd_drift),
        ild_drift_db=float(ild_drift),
        iacc_delta=float(iacc_delta),
        bmld_advantage_kept_db=bmld_kept,
        itd_ok=bool(itd_drift <= ITD_JND_US),
        ild_ok=bool(ild_drift <= ILD_JND_DB),
        iacc_ok=bool(iacc_delta <= IACC_JND),
    )


def compute_itd_us(audio: np.ndarray, sr: int) -> float:
    """ITD in µs — positiv = rechtes Signal eilt voraus, negativ = rechts eilt nach."""
    return compute_interaural_profile(audio, sr).itd_us


def compute_ild_db(audio: np.ndarray, sr: int) -> float:
    """Mittelbandige ILD = L − R in dB (250 Hz–4 kHz, 1/3-Oktaven)."""
    return compute_interaural_profile(audio, sr).ild_db


def compute_iacc(audio: np.ndarray, sr: int) -> float:
    """Interaurale Kreuzkorrelation |ρ| ∈ [0, 1] (Maximum über ±1 ms)."""
    return compute_interaural_profile(audio, sr).iacc


def bmld_advantage_db(iacc_ref: float) -> float:
    """BMLD-Proxy: Maskierungs-Freisetzung in dB aus der Interaural-Korrelation.

    N0S0 (korrelierter Maskierer, IACC→1) → Vorteil ~0 dB; NπS0
    (interaural invertierter Maskierer, IACC→0) → bis ~15 dB (Levitt 1971,
    Durlach 1972). Linearer Proxy: 15·(1−|ρ|) — deterministisch, monoton.
    """
    return float(np.clip(15.0 * (1.0 - abs(float(iacc_ref))), 0.0, 15.0))


# ── Sphärisches Kopfmodell (Duda & Martens 1998) — First-Order-HRTF-Cues ─────


def woodworth_itd_us(azimuth_deg: float, head_radius_m: float = HEAD_RADIUS_M) -> float:
    """ITD(θ) nach Woodworth (1938)/Kuhn (1977): (a/c)·(θ + sin θ), µs.

    θ in Radiant, 0° = frontal. Deterministisch, keine Messdaten.
    """
    theta = float(np.deg2rad(azimuth_deg))
    return float(1e6 * (head_radius_m / SPEED_OF_SOUND_M_S) * (theta + np.sin(theta)))


def head_shadow_ild_db(azimuth_deg: float, freq_hz: float, head_radius_m: float = HEAD_RADIUS_M) -> float:
    """Kopfschatten-ILD aus dem Sphärenmodell (Duda & Martens 1998).

    Vereinfachte geschlossene Form: ILD(f, θ) = 20·log10(1 + α(f)·|sin θ|·k·a)
    mit k = 2πf/c; α skaliert den Schattenwurf monoton mit der Frequenz.
    Deterministisch und für jede Frequenz reproduzierbar.
    """
    k = 2.0 * np.pi * float(freq_hz) / SPEED_OF_SOUND_M_S
    ka = k * head_radius_m
    shadow = np.clip(np.log1p(ka) / np.log1p(30.0), 0.0, 1.0)  # 0 @ 0 Hz → 1 @ HF
    return float(-abs(np.sin(np.deg2rad(azimuth_deg))) * shadow * 20.0)


def apply_interaural_cues(
    mono: np.ndarray,
    sr: int,
    azimuth_deg: float = 0.0,
    distance_m: float = 1.5,
    hrir_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Platziert Mono-Quelle mit Kopfmodell-Cues: ITD + ILD + Distanzgesetz.

    Returns stereo (2, N) channels-first. Mit ``hrir_pair`` (aus
    ``load_sofa_hrir``) wird statt des Sphärenmodells das gemessene
    HRIR-Paar gefaltet (FFT-Konvolution, deterministisch) — der
    individualisierte HRTF-Pfad. Ohne HRIR: Woodworth-ITD als
    Sub-Sample-Delay (lineare Interpolation → kein Filter-Ringing) und die
    Sphären-ILD gemittelt über die kritischen Bänder 500/1000/2000 Hz.
    """
    x = np.asarray(mono, dtype=np.float32)
    if x.ndim != 1:
        x = mono_mix(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if hrir_pair is not None:
        out = apply_hrir_pair(x, hrir_pair[0], hrir_pair[1])
        dist_gain = float(np.clip(1.0 / max(distance_m, 0.1), 0.02, 1.0))
        return cast(np.ndarray, np.clip(out * dist_gain, -1.0, 1.0))

    theta = float(azimuth_deg)
    itd_s = woodworth_itd_us(theta) / 1e6
    ild_db = float(np.mean([head_shadow_ild_db(theta, f) for f in (500.0, 1000.0, 2000.0)]))
    # Standard-Azimut: positiv = rechts. Quelle rechts → rechtes Ohr führt und
    # ist lauter → das LINK (kontralaterale) Ohr wird um |ITD| verzögert und
    # gedämpft (Kopfschatten). Quelle links spiegelbildlich.
    if theta > 0:
        _delay_left = abs(itd_s)
        _delay_right = 0.0
        _gain_left = 10.0 ** ((ild_db / 2.0) / 20.0)
        _gain_right = 10.0 ** ((-ild_db / 2.0) / 20.0)
    elif theta < 0:
        _delay_left = 0.0
        _delay_right = abs(itd_s)
        _gain_left = 10.0 ** ((-ild_db / 2.0) / 20.0)
        _gain_right = 10.0 ** ((ild_db / 2.0) / 20.0)
    else:
        _delay_left = 0.0
        _delay_right = 0.0
        _gain_left = 1.0
        _gain_right = 1.0

    dist_gain = float(np.clip(1.0 / max(distance_m, 0.1), 0.02, 1.0))
    _gain_left *= dist_gain
    _gain_right *= dist_gain

    n = len(x)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    if _delay_left <= 0.0:
        left = x * _gain_left
    else:
        dl = int(np.floor(_delay_left * sr))
        frac = float(_delay_left * sr) - dl
        left[dl:] = x[: n - dl] * (1.0 - frac) + x[1 : n - dl + 1] * frac
        left = left * _gain_left
    if _delay_right <= 0.0:
        right = x * _gain_right
    else:
        dr = int(np.floor(_delay_right * sr))
        frac = float(_delay_right * sr) - dr
        right[dr:] = x[: n - dr] * (1.0 - frac) + x[1 : n - dr + 1] * frac
        right = right * _gain_right

    out = np.stack([left, right], axis=0).astype(np.float32)
    return cast(np.ndarray, np.clip(out, -1.0, 1.0))


# ── Gemessene HRIRs (SOFA, z. B. CIPIC/KEMAR) — Level-2-HRTF-Pfad ────────────


def apply_hrir_pair(mono: np.ndarray, hrir_l: np.ndarray, hrir_r: np.ndarray) -> np.ndarray:
    """Faltet Mono mit einem HRIR-Paar (FFT-Konvolution, deterministisch).

    Returns stereo (2, N) channels-first mit der Signallänge des Mono-Inputs.
    """
    x = np.asarray(mono, dtype=np.float64)
    if x.ndim != 1:
        x = mono_mix(x).astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    hl = np.nan_to_num(np.asarray(hrir_l, dtype=np.float64), nan=0.0)
    hr = np.nan_to_num(np.asarray(hrir_r, dtype=np.float64), nan=0.0)
    if hl.size < 2 or hr.size < 2:
        out = np.stack([x.astype(np.float32), x.astype(np.float32)], axis=0)
        return cast(np.ndarray, out)
    n_fft = 1
    while n_fft < len(x) + max(len(hl), len(hr)) - 1:
        n_fft *= 2
    X = np.fft.rfft(x, n=n_fft)
    L = np.fft.rfft(hl, n=n_fft) * X
    R = np.fft.rfft(hr, n=n_fft) * X
    left = np.fft.irfft(L, n=n_fft)[: len(x)]
    right = np.fft.irfft(R, n=n_fft)[: len(x)]
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-9)
    out = np.stack([left, right], axis=0).astype(np.float32) / max(peak, 1e-9)
    return cast(np.ndarray, np.clip(out, -1.0, 1.0))


def load_sofa_hrir(
    path: str, azimuth_deg: float = 0.0, elevation_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liest ein HRIR-Paar aus einer SOFA-Datei (SimpleFreeFieldHRIR).

    Unterstützt netCDF-3-SOFA (scipy.io.netcdf_file); netCDF-4/HDF5-Dateien
    oder Parse-Fehler liefern ``None`` — Aufrufer fallen dann auf das
    deterministische Sphärenmodell zurück (§V6 (copilot-instructions.md): Logging mit Begründung).
    Nächster Azimut zu ``azimuth_deg`` bei gegebener Elevation wird gewählt.
    """
    try:
        from scipy.io import netcdf_file

        nc = netcdf_file(path, mode="r", mmap=False)
        try:
            ir = np.asarray(nc.variables["Data.IR"][:], dtype=np.float64)
            pos = np.asarray(nc.variables["SourcePosition"][:], dtype=np.float64)
            float(np.asarray(nc.variables["Data.SamplingRate"][:]).ravel()[0])
        finally:
            nc.close()
        if ir.ndim != 3 or ir.shape[1] < 2:
            logger.warning(
                "§V6 (copilot-instructions.md) SOFA: unerwartetes Data.IR-Format %s in %s — Sphärenmodell-Ersatzpfad",
                ir.shape,
                path,
            )
            return None
        az = pos[:, 0]
        el = pos[:, 1]
        el_diff = np.abs(el - elevation_deg)
        cand = np.where(el_diff <= 2.5)[0]
        if len(cand) == 0:
            cand = np.argsort(el_diff)[:4]
        _d = np.abs(((az[cand] - azimuth_deg + 180.0) % 360.0) - 180.0)
        m = int(cand[int(np.argmin(_d))])
        hrir_l = ir[m, 0, :]
        hrir_r = ir[m, 1, :]
        # DC-Offset entfernen (HRIR-Messung enthält oft Sub-Hz-Anteile)
        hrir_l -= hrir_l.mean()
        hrir_r -= hrir_r.mean()
        return (hrir_l.astype(np.float32), hrir_r.astype(np.float32))
    except Exception as exc:
        logger.warning(
            "§V6 (copilot-instructions.md) SOFA-Lesen fehlgeschlagen für %s — Sphärenmodell-Ersatzpfad: %s", path, exc
        )
        return None


__all__ = [
    "ITD_JND_US",
    "ITD_HARD_FAIL_US",
    "ILD_JND_DB",
    "ILD_HARD_FAIL_DB",
    "IACC_JND",
    "IACC_HARD_FAIL",
    "InterauralCueProfile",
    "InterauralIntegrityResult",
    "compute_interaural_profile",
    "compute_itd_us",
    "compute_ild_db",
    "compute_iacc",
    "bmld_advantage_db",
    "interaural_cue_integrity",
    "woodworth_itd_us",
    "head_shadow_ild_db",
    "apply_interaural_cues",
    "apply_hrir_pair",
    "load_sofa_hrir",
]

"""C1–C3-Rekombinations-Gates vor der Stem-Summe (§SLR-1f).

Quelle: docs/REKOMBINATION_ZEITPUNKT_ANALYSE.md §4 Option C / §6 (2026-09-12).
Ein Rekombinationspunkt bleibt; unmittelbar VOR der Summe laufen:

- **C2 Alignment-Korrektur** (W2): Hüllkurven-Kreuzkorrelation
  Vokal↔Instrumental auf einem 60-s-Fenster → Sample-Korrektur des
  Instrumental-Stems (Kammfilter-Präkursor). Deterministisch (FFT-xcorr,
  Parabel-Refinement, 4-kHz-Hüllkurven-Downsample).
- **C1 Residuum-Gate** (W1): der Separations-Verlust
  `mix − vocal_raw − instr_raw` (nicht die NR-/Enhancement-Deltas der
  Final-Stems!) wird nicht pauschal verworfen. Pro Bark-Band: Residuum-RMS
  vs. lokale Maskierungsschwelle (backend/core/dsp/masking_model.py,
  Hörordnung Ebene 2) — liegt das Residuum darüber, wird es spektral
  bedarfsweise zurückgemischt (STFT-Maske, Hann-OLA, deterministisch).
  Unter der Schwelle bleibt es verworfen (perzeptuell identisch).
  Beabsichtigte Stem-Verarbeitung (KIM2/KIM-Inst/Air-Presence/DFN) bleibt
  unangetastet — das Gate stellt nur den Separations-Fehler wieder her.
- **C3 Stereo-Check** (W3): `interaural_cue_integrity` Original vs. Remix
  am Nahtpunkt — ITD/ILD/IACC gegen Hör-JNDs (30/60 µs, 1/2 dB, 0.08/0.15).

Alle Gates sind report-only (Hörordnung §8a); der Remix bleibt der einzige
Rekombinationspunkt. Layout-sicher: intern channels-first (C, N), Rückgabe
im Referenz-Layout des Mix-Originals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_ALIGN_WIN_S = 60.0  # Korrelations-Analysefenster (Mitte des Signals)
_ALIGN_MAX_S = 0.05  # ±50 ms Lag (deckt ML-Blocklatenz)
_ENV_DS_HZ = 4000.0  # Hüllkurven-Downsample für die xcorr
_MIN_OFFSET_SAMPLES = 1
_MIN_ALIGN_CORR = 0.25  # Hüllkurven müssen echt korrelieren (kein xcorr-Rauschen)
_N_FFT_BAND = 2048  # Band-Energie-Raster (nicht-überlappend)
_STFT_RATIO_S = 0.04  # Remix-STFT n_fft ≈ 40 ms
_RESIDUE_FLOOR_DB = (
    80.0  # numerischer Guard: Residuum < 80 dB unter Signal-RMS = Float32-Rauschen, kein Separations-Verlust
)

# ── §III.12 Pflicht-Witnesses (Rev. v10.3.11), report-only ──────────
_DOUBLE_PROC_DB = 0.5  # ab dieser Band-Δ-Energie gilt ein Stem als verändert (C4)
_LEVEL_FRAME_S = 0.020  # 20-ms-Raster für die Pegel-Kontinuität (C5)
_LAG_WIN_S = 0.250  # nicht-überlappendes Fenster für die Laufzeit-Kontinuität (W6)
_LAG_HOP_S = 0.250
_LAG_SEARCH_S = 0.005  # ±5 ms Suchbereich des Korrelations-Lags


@dataclass
class RecombinationGateResult:
    """Ergebnis der C1–C3-Gates und der §III.12-Pflicht-Witnesses (C4/C5/W4/W5/W6).

    Die Witnesses sind **report-only**: Sie belegen Eigenschaften der Rekombination,
    sie verändern kein Signal. W6 und C5 sind hier als Zeit-Kontinuität gemessen — die
    Rekombination ist eine sample-genaue Summe ohne Segment-Naht (kein Concat/Crossfade),
    eine „Naht"-Größe im engeren Sinn existiert daher nicht und wird nicht vorgetäuscht.
    """

    offset_samples: int = 0
    alignment_corr: float = 1.0
    residue_bands_reused: int = 0
    residue_db_max_over: float = 0.0
    itd_drift_us: float = 0.0
    ild_drift_db: float = 0.0
    iacc_drop: float = 0.0
    stereo_ok: bool = True
    # §III.12 Pflicht-Witnesses (Rev. v10.3.11) — report-only
    ripple_depth_db: float = 0.0  # W4: Kammfilter-Ripple-Tiefe
    leakage_corr: float = 0.0  # W5: Stem-Leakage (Geister-Anteile)
    double_processed_bands: int = 0  # C4: Bänder mit Veränderung in BEIDEN Stems
    level_step_db: float = 0.0  # C5: max. Pegel-Sprung Mix→Remix (Zeit-Kontinuität)
    lag_step_us: float = 0.0  # W6: max. Laufzeit-Sprung zwischen Nachbarfenstern
    witness: dict = field(default_factory=dict)

    def build_witness(self) -> dict:
        """Report-only Zusammenfassung (StemContext.witness_reports["recombination"])."""
        self.witness = {
            "alignment_offset_samples": int(self.offset_samples),
            "alignment_corr": round(float(self.alignment_corr), 4),
            "residue_bands_reused": int(self.residue_bands_reused),
            "residue_db_max_over": round(float(self.residue_db_max_over), 2),
            "itd_drift_us": round(float(self.itd_drift_us), 2),
            "ild_drift_db": round(float(self.ild_drift_db), 2),
            "iacc_drop": round(float(self.iacc_drop), 4),
            "stereo_ok": bool(self.stereo_ok),
            "ripple_depth_db": round(float(self.ripple_depth_db), 3),
            "leakage_corr": round(float(self.leakage_corr), 4),
            "double_processed_bands": int(self.double_processed_bands),
            "level_step_db": round(float(self.level_step_db), 3),
            "lag_step_us": round(float(self.lag_step_us), 2),
            "continuity_note": (
                "W6/C5 als Zeit-Kontinuität gemessen — die Rekombination ist eine sample-genaue "
                "Summe ohne Segment-Naht (kein Concat/Crossfade)."
            ),
        }
        return self.witness


def _to_cn(x: np.ndarray) -> np.ndarray:
    """Layout-Normalisierung: channels-first (C, N) float32; Mono → (1, N)."""
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        return arr[None, :]  # type: ignore[no-any-return]
    if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        return arr  # type: ignore[no-any-return]
    if arr.shape[1] <= 2 and arr.shape[1] < arr.shape[0]:
        return arr.T  # type: ignore[no-any-return]
    return arr  # type: ignore[no-any-return]


def _mono(x_cn: np.ndarray) -> np.ndarray:
    _mono_mean: np.ndarray = np.mean(x_cn, axis=0)
    return _mono_mean.astype(np.float32)  # type: ignore[no-any-return]


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Analytische Hüllkurve, auf ~4 kHz dezimiert (deterministisch)."""
    from scipy.signal import hilbert  # pylint: disable=import-outside-toplevel

    _x = x - float(np.mean(x))
    env = np.abs(hilbert(_x)).astype(np.float32)
    factor = max(1, int(sr / _ENV_DS_HZ))
    env_ds: np.ndarray = env[::factor].copy()
    env_ds = env_ds - float(np.mean(env_ds))  # DC trägt keine Timing-Info
    _rms = float(np.sqrt(np.mean(env_ds**2))) + 1e-12
    _env_out: np.ndarray = env_ds / _rms
    return _env_out


def _xcorr_offset(env_vocal: np.ndarray, env_instr: np.ndarray, max_lag: int) -> tuple[float, float]:
    """FFT-xcorr der Hüllkurven → Versatz in DS-Samples + normierter Koeffizient.

    cc[k] = Σ ev[n+k]·ei[n] (zirkulär, Zero-Padding N ≥ 2m): Ist ei um d
    verzögert, liegt der Peak bei k = −d. Negative Lags liegen am
    ARRAY-ENDE (Indizes N−lag..N−1), nicht bei m−lag.
    """
    m = len(env_vocal)
    n_fft = 1 << (2 * m - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(env_vocal, n_fft) * np.conj(np.fft.rfft(env_instr, n_fft)), n_fft)
    lag = min(max_lag, m - 1)
    # Kandidaten: positive Lags [0, lag] und negative Lags am Array-Ende.
    idx = np.concatenate([np.arange(0, lag + 1), np.arange(n_fft - lag, n_fft)])
    peak_i = int(np.argmax(cc[idx]))
    k = int(idx[peak_i])
    if k > n_fft // 2:
        k -= n_fft
    # Parabel-Refinement (Jacovitti & Scarano 1993 — wie interaural_cues).
    kp = k - 1
    kn = k + 1
    c_prev, c_here, c_next = float(cc[kp % n_fft]), float(cc[k % n_fft]), float(cc[kn % n_fft])
    denom = c_prev - 2.0 * c_here + c_next
    k_ref: float = float(k)
    if abs(denom) > 1e-12:
        k_ref = float(k) + 0.5 * (c_prev - c_next) / denom
    corr = float(np.clip(c_here / m, -1.0, 1.0))  # irfft ist bereits 1/N-normalisiert; Envs RMS-normiert
    return k_ref, corr


def _shift(x_cn: np.ndarray, tau: int) -> np.ndarray:
    """Kanalweise Verschiebung um tau Samples (tau>0 = Verzögerung), Kanten genullt."""
    if tau == 0:
        return x_cn
    out = np.zeros_like(x_cn)
    if tau > 0:
        out[:, tau:] = x_cn[:, : x_cn.shape[1] - tau]
    else:
        out[:, : x_cn.shape[1] + tau] = x_cn[:, -tau:]
    return out  # type: ignore[no-any-return]


def _band_energy_db(x: np.ndarray, sr: int, edges: np.ndarray) -> np.ndarray:
    """RMS-Energie pro Bark-Band (dB) über ein nicht-überlappendes Raster."""
    n_bands = len(edges) - 1
    if len(x) < 64:
        return np.zeros(n_bands, dtype=np.float64)  # type: ignore[no-any-return]
    n_fft = _N_FFT_BAND
    if len(x) < n_fft:
        n_fft = max(64, 1 << (len(x) - 1).bit_length())
    hop = n_fft
    n_frames = max(1, (len(x) - n_fft) // hop + 1)
    win = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    out = np.zeros(n_bands, dtype=np.float64)
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if mask.any():
            _mean_e = float(np.mean(np.sum(spec[:, mask], axis=1)))
            out[b] = 10.0 * np.log10(max(_mean_e, 1e-20))
    out_arr: np.ndarray = out
    return out_arr  # type: ignore[no-any-return]


def _residue_reuse_mask(ref_cn: np.ndarray, residue_cn: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """C1: Pro Bark-Band — Residuum-RMS vs. minimale Maskierungsschwelle.

    Referenz = die Summe OHNE Residuum (das tatsächlich Gehörte) — der Mix
    als Referenz würde verlorenen Inhalt durch sich selbst maskieren.

    Returns:
        (reuse_mask (n_bands,), db_max_over) — Reuse dort, wo das Residuum
        die (konservativ minimale) Schwelle überschreitet.
    """
    from backend.core.dsp.masking_model import (  # pylint: disable=import-outside-toplevel
        bark_band_edges,
        compute_masking_threshold_db,
    )

    edges = bark_band_edges(sr)
    thr_db, _ = compute_masking_threshold_db(_mono(ref_cn), sr)
    thr_min = np.min(thr_db, axis=0)
    res_db = _band_energy_db(_mono(residue_cn), sr, edges)
    _ref_rms_db = 10.0 * np.log10(float(np.mean(_mono(ref_cn) ** 2)) + 1e-20)
    _floor_db = _ref_rms_db - _RESIDUE_FLOOR_DB
    over = res_db - thr_min
    mask: np.ndarray = (over > 0.0) & (res_db > _floor_db)
    db_max_over = float(np.max(over)) if over.size else 0.0
    return mask, db_max_over


def _bandpass_residue(residue_cn: np.ndarray, reuse_mask: np.ndarray, sr: int) -> np.ndarray:
    """C1: Residuum nur in wiederverwendeten Bändern zurückmischen (Hann-OLA)."""
    if not bool(np.any(reuse_mask)):
        return np.zeros_like(residue_cn)  # type: ignore[no-any-return]
    from backend.core.dsp.masking_model import bark_band_edges  # pylint: disable=import-outside-toplevel

    edges = bark_band_edges(sr)
    n_fft = max(512, 1 << int(np.log2(sr * _STFT_RATIO_S)).bit_length())
    hop = max(1, n_fft // 4)
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bin_mask = np.zeros(len(freqs), dtype=bool)
    for b in range(len(edges) - 1):
        if reuse_mask[b]:
            bin_mask[(freqs >= edges[b]) & (freqs < edges[b + 1])] = True
    if not bool(bin_mask.any()):
        return np.zeros_like(residue_cn)  # type: ignore[no-any-return]

    out = np.zeros_like(residue_cn)
    norm = np.zeros(residue_cn.shape[1], dtype=np.float32)
    for start in range(0, residue_cn.shape[1] - n_fft + 1, hop):
        frames = residue_cn[:, start : start + n_fft] * win
        spec = np.fft.rfft(frames, n=n_fft, axis=1)
        spec[:, ~bin_mask] = 0.0
        rec = np.fft.irfft(spec, n=n_fft, axis=1) * win
        out[:, start : start + n_fft] += rec.astype(np.float32)
        norm[start : start + n_fft] += win * win
    norm = np.maximum(norm, 1e-9)
    out_norm: np.ndarray = out / norm[None, :]
    return out_norm  # type: ignore[no-any-return]


def _bark_edges(sr: int) -> np.ndarray:
    """Bark-Band-Kanten aus dem zentralen Maskierungsmodell (keine eigene Tabelle, §G9 copilot-instructions.md)."""
    from backend.core.dsp.masking_model import bark_band_edges  # pylint: disable=import-outside-toplevel

    return bark_band_edges(sr)


def _ripple_depth_db(mix_cn: np.ndarray, remix_cn: np.ndarray, sr: int) -> float:
    """W4 — Kammfilter-Ripple-Tiefe: Streuung des Band-Energie-Verhältnisses Remix/Mix (dB).

    Eine Rest-Gruppenlaufzeit-Differenz erzeugt periodische Einbrüche; im Verhältnis
    über Bark-Bänder zeigt sich das als größere Streuung. Report-only.
    """
    try:
        _edges = _bark_edges(sr)
        _delta = _band_energy_db(_mono(remix_cn), sr, _edges) - _band_energy_db(_mono(mix_cn), sr, _edges)
        _finite = _delta[np.isfinite(_delta)]
        return float(np.std(_finite)) if _finite.size else 0.0
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("§W4 Ripple-Witness nicht verfügbar: %s", _exc)
        return 0.0


def _stem_leakage_corr(v_raw_cn: np.ndarray, i_raw_cn: np.ndarray, sr: int) -> float:
    """W5 — Stem-Leakage: |Korrelation| der Hüllkurven beider RAW-Stems.

    Sauber getrennte Quellen ⇒ unkorrelierte Hüllkurven (≈ 0). Geister-Anteile —
    dieselbe Quelle in beiden Stems — heben die Korrelation. Report-only.
    """
    try:
        _ev = _envelope(_mono(v_raw_cn), sr)
        _ei = _envelope(_mono(i_raw_cn), sr)
        _n = min(_ev.size, _ei.size)
        if _n < 8:
            return 0.0
        _c = float(np.corrcoef(_ev[:_n], _ei[:_n])[0, 1])
        return 0.0 if not np.isfinite(_c) else float(abs(_c))
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("§W5 Leakage-Witness nicht verfügbar: %s", _exc)
        return 0.0


def _double_processed_bands(
    v_raw_cn: np.ndarray,
    vocal_cn: np.ndarray,
    i_raw_cn: np.ndarray,
    instr_cn: np.ndarray,
    sr: int,
) -> int:
    """C4 — Bark-Bänder, in denen BEIDE Stems verändert wurden (Kamm-/Phantom-Risiko).

    Überlappende Spektralanteile, die in beiden Stems bearbeitet wurden, können sich
    bei der Summe als Kamm- oder Phantom-Artefakt niederschlagen. Report-only.
    """
    try:
        _edges = _bark_edges(sr)
        _dv = _band_energy_db(_mono(vocal_cn), sr, _edges) - _band_energy_db(_mono(v_raw_cn), sr, _edges)
        _di = _band_energy_db(_mono(instr_cn), sr, _edges) - _band_energy_db(_mono(i_raw_cn), sr, _edges)
        return int(np.sum((np.abs(_dv) > _DOUBLE_PROC_DB) & (np.abs(_di) > _DOUBLE_PROC_DB)))
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("§C4 Doppelverarbeitungs-Witness nicht verfügbar: %s", _exc)
        return 0


def _level_step_db(mix_cn: np.ndarray, remix_cn: np.ndarray, sr: int) -> float:
    """C5 — max. Pegel-Sprung (dB) zwischen Mix und Remix über 20-ms-Frames.

    Ersetzt „Pegel-Kontinuität am Nahtpunkt": Die Summe hat keine Naht, geprüft wird
    deshalb die Kontinuität über die Zeit. Report-only.
    """
    try:
        _hop = max(int(_LEVEL_FRAME_S * sr), 1)
        _m, _r = _mono(mix_cn), _mono(remix_cn)
        _n = min(_m.size, _r.size)
        if _n < 2 * _hop:
            return 0.0
        _frames = _n // _hop
        _lvl_m = 10.0 * np.log10(np.mean(_m[: _frames * _hop].reshape(_frames, _hop) ** 2, axis=1) + 1e-20)
        _lvl_r = 10.0 * np.log10(np.mean(_r[: _frames * _hop].reshape(_frames, _hop) ** 2, axis=1) + 1e-20)
        _diff = np.abs(_lvl_r - _lvl_m)
        _finite = _diff[np.isfinite(_diff)]
        return float(np.max(_finite)) if _finite.size else 0.0
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("§C5 Pegel-Kontinuitäts-Witness nicht verfügbar: %s", _exc)
        return 0.0


def _lag_step_us(mix_cn: np.ndarray, remix_cn: np.ndarray, sr: int) -> float:
    """W6 — max. Sprung (µs) des Korrelations-Lags zwischen Nachbarfenstern.

    Ersetzt „Seam-Gruppenlaufzeit-Sprung": Ohne Naht wird die Kontinuität der
    Gruppenlaufzeit über die Zeit geprüft. Report-only.
    """
    try:
        _win = int(_LAG_WIN_S * sr)
        _hop = max(int(_LAG_HOP_S * sr), 1)
        _m, _r = _mono(mix_cn), _mono(remix_cn)
        _n = min(_m.size, _r.size)
        if _win < 64 or _n < 3 * _win:
            return 0.0
        _max_lag = max(1, int(_LAG_SEARCH_S * sr))
        _lags: list[float] = []
        _frames = (_n - _win) // _hop + 1
        for _i in range(int(_frames)):
            _s = _i * _hop
            _lag, _corr = _xcorr_offset(_m[_s : _s + _win], _r[_s : _s + _win], _max_lag)
            if _corr >= _MIN_ALIGN_CORR:
                _lags.append(float(_lag))
        if len(_lags) < 2:
            return 0.0
        _steps = np.abs(np.diff(np.asarray(_lags, dtype=np.float64)))
        return float(np.max(_steps) / max(int(sr), 1) * 1e6)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("§W6 Laufzeit-Kontinuitäts-Witness nicht verfügbar: %s", _exc)
        return 0.0


def recombine_stems_with_gates(
    mix_original: np.ndarray,
    vocal_stem_raw: np.ndarray,
    instr_stem_raw: np.ndarray,
    vocal_stem_final: np.ndarray,
    instr_stem_final: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, RecombinationGateResult]:
    """C2 → C1 → C3, dann die Summe — der eine Rekombinationspunkt.

    Returns:
        (remix in channels-first (C, N), gate result)
    """
    sr = int(sample_rate)
    mix = _to_cn(mix_original)
    v_raw = _to_cn(vocal_stem_raw)
    i_raw = _to_cn(instr_stem_raw)
    vocal = _to_cn(vocal_stem_final)
    instr = _to_cn(instr_stem_final)
    n = min(mix.shape[1], vocal.shape[1], instr.shape[1], v_raw.shape[1], i_raw.shape[1])
    mix, vocal, instr = mix[:, :n], vocal[:, :n], instr[:, :n]
    v_raw, i_raw = v_raw[:, :n], i_raw[:, :n]

    # C2 — Alignment: Instrumental-Hüllkurve gegen Vokal ausrichten.
    offset = 0
    corr = 1.0
    win = min(int(_ALIGN_WIN_S * sr), n)
    if win >= 2 * sr:
        try:
            start = max(0, (n - win) // 2)
            _ev = _envelope(_mono(vocal)[start : start + win], sr)
            _ei = _envelope(_mono(instr)[start : start + win], sr)
            max_lag_ds = max(1, int(_ALIGN_MAX_S * _ENV_DS_HZ))
            offset_ds, corr = _xcorr_offset(_ev, _ei, max_lag_ds)
            ds_factor = max(1, int(sr / _ENV_DS_HZ))
            offset = int(round(offset_ds * ds_factor))
        except Exception as _c2_exc:  # pylint: disable=broad-except
            logger.debug("§C2 Alignment nicht verfügbar: %s", _c2_exc)
    if abs(offset) >= _MIN_OFFSET_SAMPLES and corr >= _MIN_ALIGN_CORR:
        instr = _shift(instr, offset)  # cc[k]=Σev·ei[n−k]: Peak bei −d → shift um offset=−d rückt instr vor
    elif abs(offset) >= _MIN_OFFSET_SAMPLES:
        logger.debug("§C2 Alignment verworfen: corr=%.3f < %.2f (Hüllkurven unkorreliert)", corr, _MIN_ALIGN_CORR)
        offset = 0  # nicht angewendet → als 0 berichten

    # C1 — Separations-Verlust (Raw-Stems) gegen die Maskierungsschwelle.
    # Referenz für die Schwelle: die Summe OHNE Residuum (das Gehörte).
    residue = mix - v_raw - i_raw
    reuse_mask, db_max_over = _residue_reuse_mask(vocal + instr, residue, sr)
    residue_gated = _bandpass_residue(residue, reuse_mask, sr)
    remix = vocal + instr + residue_gated

    # C3 — Stereo-Check am Nahtpunkt (JNDs aus interaural_cues).
    # Nur für echte Stereo-Inputs — Mono ist kein Kollaps.
    itd, ild, iacc = 0.0, 0.0, 0.0
    stereo_ok = True
    if mix.shape[0] >= 2:
        try:
            from backend.core.dsp.interaural_cues import (  # pylint: disable=import-outside-toplevel
                interaural_cue_integrity,
            )

            _ic = interaural_cue_integrity(mix, remix, sr)
            itd, ild, iacc = float(_ic.itd_drift_us), float(_ic.ild_drift_db), float(_ic.iacc_delta)
            stereo_ok = bool(_ic.itd_ok and _ic.ild_ok and _ic.iacc_ok)
        except Exception as _c3_exc:  # pylint: disable=broad-except
            logger.debug("§C3 Stereo-Prüfung nicht verfügbar: %s", _c3_exc)
    if not stereo_ok:
        logger.warning("§C3 Stereo-Drift am Nahtpunkt: ITD=%.1f µs ILD=%.1f dB IACC=%.3f", itd, ild, iacc)

    # §III.12 Pflicht-Witnesses (Rev. v10.3.11) — report-only, kein Eingriff ins Signal.
    _w4_ripple = _ripple_depth_db(mix, remix, sr)
    _w5_leak = _stem_leakage_corr(v_raw, i_raw, sr)
    _c4_double = _double_processed_bands(v_raw, vocal, i_raw, instr, sr)
    _c5_level = _level_step_db(mix, remix, sr)
    _w6_lag = _lag_step_us(mix, remix, sr)
    result = RecombinationGateResult(
        offset_samples=offset,
        alignment_corr=corr,
        residue_bands_reused=int(np.sum(reuse_mask)),
        residue_db_max_over=db_max_over,
        itd_drift_us=itd,
        ild_drift_db=ild,
        iacc_drop=iacc,
        stereo_ok=stereo_ok,
        ripple_depth_db=_w4_ripple,
        leakage_corr=_w5_leak,
        double_processed_bands=_c4_double,
        level_step_db=_c5_level,
        lag_step_us=_w6_lag,
    )
    result.build_witness()
    remix_cn: np.ndarray = remix.astype(np.float32)
    return remix_cn, result

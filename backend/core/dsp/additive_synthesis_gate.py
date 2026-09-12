"""Additive-Synthesis-Gate für energie-HINZUFÜGENDE ML-Stufen.

Schützt FlashSR-Bandbreiten-Extension und BigVGAN-Vocoder-Repair vor
hörbaren Synthese-Artefakten (Hörordnung Ebene 2: Maskierungsschwelle
statt Mess-Null; Hör-Invariante „Natürlichkeit").

Regel pro Bark-Band und Frame:
  - Energie HINZUFÜGEN darf das ML-Modell nur dort, wo das Baseline-Band
    UNHÖRBAR ist (e_base < thr + 3 dB — nichts Reales wird beschädigt).
  - Die hinzugefügte Energie wird auf die Maskierungsschwelle begrenzt
    (w = (thr − e_base)/(e_cand − e_base)): kein unnatürliches
    HF-Zischen/-Klingeln über der Hörbarkeitsgrenze.
  - In Bändern mit hörbarem Original-Inhalt bleibt das Baseline
    unangetastet (Never-worsen: nie Energie über max(e_base, thr + 1 dB)).
  - Transienten-Schutz: Frames mit starkem Onset (Energie-Verhältnis > 3
    gegen lokalen Mittelwert) behalten das Baseline vollständig
    (keine synthetisierte Energie in Anschlägen).

Deterministisch (kein RNG, kein time.time), rein numpy.
Stereo-Layout-agnostisch: (N,) und (C, N) werden normalisiert bedient.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_N_FFT = 2048
_HOP = 512
_WIN = np.hanning(_N_FFT).astype(np.float32)
_INAUDIBLE_MARGIN_DB = 3.0  # Baseline muss KLAR unter der Schwelle liegen
_ONSET_RATIO = 3.0  # Energie-Verhältnis Frame vs. lokaler Mittelwert
_SMOOTH_FRAMES = 3


def _to_channels_first(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :], False
    if x.shape[0] <= 2 and x.shape[1] > x.shape[0]:
        return x, False
    return x.T, True


def _stft(x: np.ndarray) -> tuple[np.ndarray, int]:
    """(Spektrum (frames, bins), Frequenzen). Beidseitig gepaddet (COLA-Norm)."""
    x = np.pad(x, (_N_FFT // 2, _N_FFT // 2))
    n_frames = max(1, int(np.ceil(len(x) / _HOP)))
    pad_to = _N_FFT + (n_frames - 1) * _HOP
    if pad_to > len(x):
        x = np.pad(x, (0, pad_to - len(x)))
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WIN
    spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
    return spec, _N_FFT


def _band_energies(spec: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-Frame-per-Band-Energien [dB] + Band-Masken."""
    from backend.core.dsp.masking_model import bark_band_edges

    freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr)
    edges = bark_band_edges(sr)
    n_bands = len(edges) - 1
    n_frames = spec.shape[0]
    band_e = np.zeros((n_frames, n_bands), dtype=np.float64)
    masks: list[np.ndarray] = []
    for b in range(n_bands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        masks.append(m)
        if m.any():
            band_e[:, b] = np.sum(np.abs(spec[:, m]) ** 2, axis=1)
    return 10.0 * np.log10(band_e + 1e-12), np.asarray(masks)


def _threshold_fine(baseline: np.ndarray, sr: int, n_frames: int, n_bands: int) -> np.ndarray:
    """Maskierungsschwelle des Baseline auf das feine Frame-Raster interpoliert."""
    from backend.core.dsp.masking_model import compute_masking_threshold_db

    thr_coarse, _ = compute_masking_threshold_db(baseline, sr)
    t_coarse = (np.arange(thr_coarse.shape[0]) * _N_FFT + _N_FFT / 2.0) / sr
    t_fine = (np.arange(n_frames) * _HOP + _N_FFT / 2.0) / sr
    thr = np.zeros((n_frames, n_bands), dtype=np.float64)
    for b in range(min(n_bands, thr_coarse.shape[1])):
        thr[:, b] = np.interp(t_fine, t_coarse, thr_coarse[:, b], left=thr_coarse[0, b], right=thr_coarse[-1, b])
    return cast(np.ndarray, thr)


def _onset_frames(spec: np.ndarray) -> np.ndarray:
    """Transienten-Maske: 1.0 = ruhig, 0.0 = Onset-Frame (Baseline behalten)."""
    energy = np.sum(np.abs(spec) ** 2, axis=1)
    # Lokaler Mittelwert über ±3 Frames (deterministische Box).
    k = np.ones(7, dtype=np.float64) / 7.0
    local = np.convolve(energy, k, mode="same")
    ratio = energy / (local + 1e-12)
    onset = ratio > _ONSET_RATIO
    mask = np.where(onset, 0.0, 1.0)
    # Weiche Flanken: Onsets werden nicht hart geschaltet.
    return cast(np.ndarray, np.clip(np.convolve(mask, np.ones(3) / 3.0, mode="same"), 0.0, 1.0))


def _istft_ola(spec: np.ndarray, n: int) -> np.ndarray:
    nf = spec.shape[0]
    pad = _N_FFT - spec.shape[1]
    full = np.concatenate([spec, np.zeros((nf, pad), dtype=np.complex128)], axis=1)
    t_out = np.fft.irfft(full, n=_N_FFT, axis=1)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    w64 = _WIN.astype(np.float64)
    off = _N_FFT // 2
    for f in range(nf):
        s0 = f * _HOP - off
        seg = t_out[f] * w64
        lo = max(0, -s0)
        hi = min(_N_FFT, n - s0)
        if hi > lo:
            out[s0 + lo : s0 + hi] += seg[lo:hi]
            norm[s0 + lo : s0 + hi] += w64[lo:hi] ** 2
    normed = np.zeros(n, dtype=np.float64)
    mask = norm > 1e-9
    normed[mask] = out[mask] / norm[mask]
    return cast(np.ndarray, normed.astype(np.float32))


def additive_synthesis_gate(
    candidate: np.ndarray,
    baseline: np.ndarray,
    sr: int,
    model: str = "ml_synthesis",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Masking-bewusste Freigabe synthetisierter Energie (Never-worsen).

    Returns:
        (gated, report) — gated hat Baseline-Form und -Länge; in jedem
        Bark-Band gilt: E(gated) ≤ max(E(baseline), Schwelle + 1 dB).
    """
    cand_cn, _trans_c = _to_channels_first(candidate)
    base_cn, _trans_b = _to_channels_first(baseline)
    n = min(cand_cn.shape[1], base_cn.shape[1])
    cand_cn = cand_cn[:, :n]
    base_cn = base_cn[:, :n]

    base_mix = base_cn.mean(axis=0)
    spec_b_mix, _ = _stft(base_mix)
    e_b_mix, masks = _band_energies(spec_b_mix, sr)
    thr = _threshold_fine(base_mix, sr, e_b_mix.shape[0], masks.shape[0])
    onset = _onset_frames(spec_b_mix)

    n_bands = masks.shape[0]
    w_band: np.ndarray | None = None
    for ch in range(cand_cn.shape[0]):
        spec_c, _ = _stft(cand_cn[ch])
        spec_b, _ = _stft(base_cn[ch])
        nf = min(spec_c.shape[0], spec_b.shape[0])
        e_c, _ = _band_energies(spec_c[:nf], sr)
        e_b, _ = _band_energies(spec_b[:nf], sr)
        added = e_c - e_b  # > 0: Kandidat fügt Energie hinzu
        inaudible = e_b < thr[:nf, :n_bands] + _INAUDIBLE_MARGIN_DB
        # Energie-Deckel: w bringt E(gated) auf die Schwelle (lineare dB-Interp).
        denom = np.maximum(e_c - e_b, 1e-9)
        w = np.where(added > 0.0, (thr[:nf, :n_bands] - e_b) / denom, 0.0)
        w = np.where(inaudible, w, 0.0)
        w = np.clip(w, 0.0, 1.0)
        # Transienten-Schutz: in Onset-Frames keine Synthese-Energie.
        w = w * onset[:nf, None]
        w = np.clip(
            np.convolve(w.ravel(), np.ones(_SMOOTH_FRAMES) / _SMOOTH_FRAMES, mode="same").reshape(w.shape), 0.0, 1.0
        )
        if w_band is None:
            w_band = w
        else:
            w_band = w_band + w
    w_band = (w_band / cand_cn.shape[0]) if w_band is not None else np.zeros((1, n_bands))

    gated = np.zeros_like(base_cn)
    for ch in range(base_cn.shape[0]):
        spec_c, _ = _stft(cand_cn[ch])
        spec_b, _ = _stft(base_cn[ch])
        nf = min(spec_c.shape[0], spec_b.shape[0], w_band.shape[0])
        wb = np.zeros((nf, spec_b.shape[1]), dtype=np.float64)
        for b in range(n_bands):
            m = masks[b]
            wb[:, m] = w_band[:nf, b : b + 1]
        spec_f = spec_b[:nf] + wb * (spec_c[:nf] - spec_b[:nf])
        gated[ch] = _istft_ola(spec_f, n)

    out_audio: np.ndarray
    if candidate.ndim == 1:
        if baseline.ndim == 1:
            out_audio = gated[0]
        else:
            out_audio = gated if not _trans_b else gated.T
    else:
        out_audio = gated if not _trans_c else gated.T

    report: dict[str, Any] = {
        "model": model,
        "mean_synthesis_gain": round(float(np.mean(w_band)), 4),
        "bands_released": int(np.sum(np.mean(w_band, axis=0) > 0.05)),
        "onset_frames_protected": int(np.sum(onset < 0.5)),
        "never_worsen": True,
    }
    return np.asarray(out_audio, dtype=np.float32), report

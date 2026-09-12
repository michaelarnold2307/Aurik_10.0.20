"""H1+H2: Masking-threshold-bewusste Denoise-Fusion + Musical-Noise-Gate.

§Witness-SOTA / Hörordnung Ebene 2 (Maskierungsschwelle statt Mess-Null):
Der ML-Denoiser (DeepFilterNet/EAR-VAE) darf sein Ergebnis nur dort
übernehmen, wo die Änderung gegenüber dem DSP-Baseline (OMLSA) oberhalb
der Bark-Maskierungsschwelle LIEGT und das Band leiser macht
(Never-worsen: nie Energie hinzufügen). Danach dämpft das
Musical-Noise-Gate hörbares Restrauschen (oberhalb der Schwelle) auf die
Schwelle herunter — nie in „tote Stille".

Deterministisch (kein RNG, kein time.time), rein numpy/scipy.
Stereo-Layout-agnostisch: (N,) und (C, N) werden normalisiert bedient
(Layout-Invariante, AGENTS.md §3).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_N_FFT = 2048
_HOP = 512
_WIN = np.hanning(_N_FFT).astype(np.float32)
_ALPHA = 0.5  # Musical-Noise-Gate: Kompressions-Exponent
_GATE_FLOOR_DB = 6.0  # nur Bänder, die klar HÖRBAR über der Schwelle liegen, gaten
_SMOOTH_FRAMES = 3  # Box-Glättung der Band-Gewichte (deterministisch)


def _to_channels_first(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalisiert auf (C, N); Rückgabe (audio_cn, war_transponiert)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :], False
    if x.shape[0] <= 2 and x.shape[1] > x.shape[0]:
        return x, False
    return x.T, True


def _stft_bands(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Spektrum (frames, bins), Band-Energien [dB] (frames, bands), Band-Masken).

    Das Signal wird null-gepaddet, sodass die Frames das Signalende voll
    abdecken (Tail-Abdeckung für die Rücktransformation).
    """
    from backend.core.dsp.masking_model import bark_band_edges

    x = np.asarray(x, dtype=np.float32)
    # Beidseitiges Halbfenster-Padding: die Rücktransformation erhält dadurch
    # volle Fenster-Norm bis zur ersten/letzten echten Sample (COLA-konstant).
    x = np.pad(x, (_N_FFT // 2, _N_FFT // 2))
    n_frames = max(1, int(np.ceil(len(x) / _HOP)))
    pad_to = _N_FFT + (n_frames - 1) * _HOP
    if pad_to > len(x):
        x = np.pad(x, (0, pad_to - len(x)))
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WIN
    spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
    freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr)
    edges = bark_band_edges(sr)
    n_bands = len(edges) - 1
    band_e = np.zeros((n_frames, n_bands), dtype=np.float64)
    masks: list[np.ndarray] = []
    for b in range(n_bands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        masks.append(m)
        if m.any():
            band_e[:, b] = np.sum(np.abs(spec[:, m]) ** 2, axis=1)
    band_e_db: np.ndarray = 10.0 * np.log10(band_e + 1e-12)
    return spec, band_e_db, np.asarray(masks)


def _istft_ola(spec: np.ndarray, n: int) -> np.ndarray:
    """Overlap-Add-Rücktransformation (Hanning, COLA-Norm) mit Head-Offset.

    ``_stft_bands`` padet beidseitig um _N_FFT//2; hier wird der Offset
    zurückgerechnet, sodass Fenster-Norm an den Signalkanten nicht
    kollabiert (Head/Tail-Nullstellen des Hann-Fensters).
    """
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


def _smooth_w(w: np.ndarray) -> np.ndarray:
    """Box-Glättung über die Zeit (deterministisch, _SMOOTH_FRAMES)."""
    if w.shape[0] <= _SMOOTH_FRAMES:
        return w
    k = np.ones(_SMOOTH_FRAMES, dtype=np.float64) / _SMOOTH_FRAMES
    out = np.zeros_like(w)
    for b in range(w.shape[1]):
        out[:, b] = np.convolve(w[:, b], k, mode="same")
    return cast(np.ndarray, np.clip(out, 0.0, 1.0))


def _band_delta_db_full(candidate: np.ndarray, baseline: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Exakte globale Band-Energie-Deltas via Parseval (kein Fenster!).

    Fenster-FFTs erzeugen Kreuzterm-Auslöschungen, die eine reine
    Energie-HINZUFÜGUNG als Phantom-„leiser“-Band erscheinen lassen
    (Median hilft nicht — die Auslöschung ist strukturiert). Die
    Band-Energie des VOLLEN Signals (Σ|X[k]|², Parseval) ist dagegen
    exakt additiv: E(cand) ≥ E(base) für jede reine Hinzufügung.
    Rückgabe: (delta_db (n_bands,), bark_centers). Zeitkonstant — die
    Maskierungsschwelle wirkt weiterhin Frame-weise als Gate.
    """
    from backend.core.dsp.masking_model import bark_band_edges

    cand = np.asarray(candidate, dtype=np.float32)
    base = np.asarray(baseline, dtype=np.float32)
    n = min(len(cand), len(base))
    xc = np.fft.rfft(cand[:n])
    xb = np.fft.rfft(base[:n])
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    edges = bark_band_edges(sr)
    delta = np.zeros(len(edges) - 1, dtype=np.float64)
    for b in range(len(edges) - 1):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        ec: float = float(np.sum(np.abs(xc[m]) ** 2))
        eb: float = float(np.sum(np.abs(xb[m]) ** 2))
        delta[b] = 10.0 * np.log10(ec / eb) if eb > 1e-20 and ec > 0.0 else 0.0
    return delta, edges


def masked_denoise_fusion(
    candidate: np.ndarray,
    baseline: np.ndarray,
    sr: int,
    floor_db: float = -80.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """H1: Bark-band-weise, masking-threshold-bewusste Fusion ML vs. DSP.

    Regel pro Band b (Johnston-Schwelle des Baseline):
      delta = E_candidate − E_baseline [dB]
      w = 0                                   wenn delta ≥ −3 dB (Never-worsen;
                                              3 dB = JND für Band-Energie nach
                                              Zwicker — darunter hört das Ohr
                                              keinen Unterschied; unterdrückt
                                              zugleich FFT-Leakage-Kreuzterme)
      w = clip(0.5·(1 + tanh((|delta| − thr)/2))) sonst
    → Kandidat wird nur dort übernommen, wo er HÖRBAR leiser ist
    (Restrausch entfernt); unhörbare Unterschiede bleiben beim Baseline.

    Returns:
        (fused, report) — fused hat Baseline-Form und -Länge.
    """
    cand_cn, _trans_c = _to_channels_first(candidate)
    base_cn, _trans_b = _to_channels_first(baseline)
    n = min(cand_cn.shape[1], base_cn.shape[1])
    cand_cn = cand_cn[:, :n]
    base_cn = base_cn[:, :n]

    base_mix = base_cn.mean(axis=0)
    _, e_b, masks = _stft_bands(base_mix, sr)
    thr = _threshold_fine(base_mix, sr, e_b.shape[0], e_b.shape[1])

    n_bands = e_b.shape[1]
    # Exakte Band-Deltas (Parseval, zeitkonstant) — pro Kanal gemittelt.
    delta_all = np.zeros(n_bands, dtype=np.float64)
    for ch in range(cand_cn.shape[0]):
        _d, _edges = _band_delta_db_full(cand_cn[ch], base_cn[ch], sr)
        delta_all += _d[:n_bands]
    delta_all /= max(1, cand_cn.shape[0])

    w_band: np.ndarray | None = None
    for ch in range(cand_cn.shape[0]):
        cand_mix_ch = cand_cn[ch] if cand_cn.shape[0] == 1 else cand_cn[ch]
        spec_c, e_c, _m2 = _stft_bands(cand_mix_ch, sr)
        spec_b, e_bb, _m3 = _stft_bands(base_cn[ch], sr)
        nf = min(spec_c.shape[0], spec_b.shape[0])
        # Hörbarkeit: |delta| muss die Maskierungsschwelle ÜBERSCHREITEN und
        # mindestens 3 dB betragen (JND Band-Energie, Zwicker).
        margin = np.where(delta_all[None, :] < -3.0, (-delta_all[None, :]) - thr[:nf, :n_bands], -1e3)
        w = 0.5 * (1.0 + np.tanh(margin / 2.0))
        w = np.clip(w, 0.0, 1.0)
        w = _smooth_w(w)
        if w_band is None:
            w_band = w
        else:
            w_band = w_band + w  # Mittel über Kanäle → Stereo-kohärente Gewichte
    w_band = (w_band / cand_cn.shape[0]) if w_band is not None else np.zeros((1, n_bands))

    # Per-Bin-Gewichte aus Band-Masken broadcasten → komplexe Fusion mit Phase.
    fused = np.zeros_like(base_cn)
    for ch in range(base_cn.shape[0]):
        spec_c, e_c, _m4 = _stft_bands(cand_cn[ch], sr)
        spec_b, e_bb, _m5 = _stft_bands(base_cn[ch], sr)
        nf = min(spec_c.shape[0], spec_b.shape[0], w_band.shape[0])
        wb = np.zeros((nf, spec_b.shape[1]), dtype=np.float64)
        for b in range(n_bands):
            m = masks[b]
            wb[:, m] = w_band[:nf, b : b + 1]
        spec_f = spec_b[:nf] + wb * (spec_c[:nf] - spec_b[:nf])
        fused[ch] = _istft_ola(spec_f, n)

    report: dict[str, Any] = {
        "mean_blend": round(float(np.mean(w_band)), 4),
        "bands_taken": int(np.sum(np.mean(w_band, axis=0) > 0.5)),
        "max_delta_db": round(float(np.min(delta_all)), 2),
        "never_worsen": True,
    }
    out_audio: np.ndarray
    if candidate.ndim == 1:
        if baseline.ndim == 1:
            out_audio = fused[0]
        else:
            out_audio = fused if not _trans_b else fused.T
    else:
        out_audio = fused if not _trans_c else fused.T
    return np.asarray(out_audio, dtype=np.float32), report


def musical_noise_gate(
    x: np.ndarray,
    sr: int,
    alpha: float = _ALPHA,
    gate_floor_db: float = _GATE_FLOOR_DB,
    floor_db: float = -80.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """H2: Musical-Noise-Gate — hörbares Restrauschen auf die Schwelle dämpfen.

    Pro Bark-Band: Minimum-Tracking liefert den Rausch-Floor. Gegatet wird
    nur, wo das Band KLAR HÖRBAR über der Maskierungsschwelle liegt
    (e > thr + gate_floor_db); Ziel = max(floor, thr), Gain g = (Ziel/e)^α,
    g ∈ [0.1, 1] → nie unter die Schwelle, nie „tote Stille", nie lauter.
    """
    x = np.asarray(x, dtype=np.float32)
    x_cn, trans = _to_channels_first(x)
    n = x_cn.shape[1]
    mix = x_cn.mean(axis=0)
    spec_m, e_m, masks = _stft_bands(mix, sr)
    nf = spec_m.shape[0]
    thr = _threshold_fine(mix, sr, nf, masks.shape[0])

    # Minimum-Tracking Floor (deterministisch, langsamer Anstieg).
    floor_db_lin = np.power(10.0, np.clip(e_m, -200.0, 0.0) / 10.0)
    track = floor_db_lin[0].copy()
    tracked = [track]
    for f in range(1, nf):
        track = np.minimum(track * 1.01, floor_db_lin[f])
        tracked.append(track.copy())
    floor_e = np.asarray(tracked)

    # Gain pro Band: nur klar hörbares Restrauschen gaten.
    thr_lin = np.power(10.0, thr / 10.0)
    gate_mask = e_m > thr + gate_floor_db
    target = np.maximum(floor_e, thr_lin)
    g_band = np.where(gate_mask, np.clip(np.power(target / np.maximum(floor_db_lin, 1e-12), alpha), 0.1, 1.0), 1.0)
    g_band = _smooth_w(g_band)

    out = np.zeros_like(x_cn)
    for ch in range(x_cn.shape[0]):
        spec_c, _, _ = _stft_bands(x_cn[ch], sr)
        gb = np.zeros((min(nf, spec_c.shape[0]), spec_c.shape[1]), dtype=np.float64)
        for b in range(masks.shape[0]):
            m = masks[b]
            gb[:, m] = g_band[: gb.shape[0], b : b + 1]
        spec_g = spec_c[: gb.shape[0]] * gb
        out[ch] = _istft_ola(spec_g, n)

    report: dict[str, Any] = {
        "mean_gain": round(float(np.mean(g_band)), 4),
        "bands_gated": int(np.sum(np.mean(g_band, axis=0) < 0.95)),
        "never_worsen": True,
    }
    out_audio = out[0] if x.ndim == 1 else (out if not trans else out.T)
    return np.asarray(out_audio, dtype=np.float32), report

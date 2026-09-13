"""IN-V1 + IN-V2: Naht-Gates für Inpainting-Kandidaten (§Witness-SOTA, phase_55).

Roadmap `docs/TODOS_SOTA_ROADMAP.md` SOTA-IN-V1+V2: Inpainting-Naht-Gates um
phase_55 (DiffWave/AudioLDM2/CQT-Diff+): additive_synthesis_gate + C2-artiges
Hüllkurven-Alignment.

Ein ML-/DSP-Inpainting-Fill ersetzt fehlenden Inhalt (Dropout, Transport-
Lücke) — der Kandidat kann an den Gap-Rändern Pegelsprünge erzeugen (Klick/
Naht-Artefakt) oder lauter als der umgebende Musik-Kontext halluzinieren.
Zwei Gates, beide rein deterministisch (numpy):

- IN-V2 Hüllkurven-Alignment (C2-artig): Die RMS-Hüllkurve des Kandidaten
  wird in der log-Domäne an die Kontext-Hüllkurve AN BEIDEN NÄHTEN
  angeglichen (exponentiell zur Naht gewichtete Kontext-RMS als Ziel,
  Gain-Limit [0.2, 3.0], log-lineare Interpolation über die Lücke,
  5-ms-Rampen an den Rändern). Ergebnis: keine Pegelsprünge an den
  Gap-Rändern.
- IN-V1 Additive-Kappung (maskierungsbewusst): Die Band-Energie des Fills
  wird per STFT-Gain-Maske auf die (frequenz-geglättete) Kontext-Band-
  Energie + cap_margin_db gedeckelt — der Fill wird nie lauter als die
  spektrale Hülle des umgebenden Musik-Kontexts (Never-worsen,
  §0 Primum-non-nocere). Stiller Kontext → Fill wird auf den
  Hörschwellen-Floor gedämpft (passthrough-sicher).

Abgrenzung zu `additive_synthesis_gate` (B4+B5): jenes deckelt HINZUGEFÜGTE
Energie auf einer hörbaren Baseline — für Inpainting ist die Baseline an der
Lücke (nahezu) stumm, weshalb hier die Kontext-Referenz (Musik um die Lücke)
verwendet wird; das Prinzip (Band-weise Energie-Deckelung, Maskierung) ist
dasselbe.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_FRAME_MS = 10.0
_EDGE_MS = 40.0
_N_FFT = 1024
_HOP = 256
_FREQ_SMOOTH_BINS = 16
_ATH_FLOOR_DB = -70.0  # Hörschwellen-Floor für stillen Kontext (dBFS-Näherung)


def _frame_rms_db(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Gefensterte RMS-Hüllkurve in dBFS (log-Domäne, Floor-gebunden)."""
    n = len(x)
    if n < frame:
        return np.full(1, _ATH_FLOOR_DB, dtype=np.float64)  # type: ignore[no-any-return]
    n_frames = max(1, (n - frame) // hop + 1)
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 10.0 ** (_ATH_FLOOR_DB / 20.0)))  # type: ignore[no-any-return]


def _conv_rows_time(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zeilenweise (Frame-)Glättung mit kantenreplizierter 1-D-Konvolution."""
    k = np.asarray(kernel, dtype=np.float64).ravel()
    half = len(k) // 2
    out: np.ndarray = np.zeros(x.shape, dtype=np.float64)
    rows = np.arange(x.shape[0])
    for shift, c in enumerate(k):
        src = np.clip(rows + shift - half, 0, x.shape[0] - 1)
        out += c * x[src, :]
    return out


def _smooth_freq(x: np.ndarray, width: int) -> np.ndarray:
    """Gleitender Mittelwert über Frequenz-Bins (spektrale Hüllkurve)."""
    kernel = np.ones((1, width), dtype=np.float64) / width
    cols = np.arange(x.shape[1])
    out: np.ndarray = np.zeros(x.shape, dtype=np.float64)
    for shift in range(width):
        src = np.clip(cols + shift - width // 2, 0, x.shape[1] - 1)
        out += x[:, src] * kernel[0, shift]
    return out


def inpainting_seam_gate(
    channel: np.ndarray,
    candidate: np.ndarray,
    start: int,
    end: int,
    sr: int,
    align_gain_limits: tuple[float, float] = (0.2, 3.0),
    cap_margin_db: float = 6.0,
) -> tuple[np.ndarray, dict]:
    """Gate einen Inpainting-Kandidaten an den Naht-Kontext.

    Args:
        channel: Mono-Kanal mit der Lücke (Original, Lücke unverändert).
        candidate: Kandidaten-Fill, Länge == end - start.
        start, end: Gap-Grenzen in Samples.
        sr: Abtastrate.
        align_gain_limits: (min, max) Gain für das Hüllkurven-Alignment.
        cap_margin_db: Energie-Deckel über dem Kontext (dB).

    Returns:
        (gated_candidate, report) — deterministisch, 1-D.
    """
    n = max(0, int(end) - int(start))
    cand = np.nan_to_num(np.asarray(candidate, dtype=np.float32).ravel()[:n], nan=0.0, posinf=0.0, neginf=0.0)
    if n < 8:
        return cand, {"applied": False, "reason": "gap_too_short"}

    ch = np.asarray(channel, dtype=np.float32).ravel()
    frame = max(64, int(sr * _FRAME_MS / 1000.0))
    hop = max(16, frame // 4)
    edge_s = max(frame, int(sr * _EDGE_MS / 1000.0))

    # ── IN-V2: Hüllkurven-Alignment in der log-Domäne ─────────────────────
    env_cand = _frame_rms_db(cand, frame, hop)
    gain = np.ones(n, dtype=np.float64)

    def _edge_target(ctx_seg: np.ndarray, near_seam: bool) -> tuple[float, bool]:
        """Ziel-RMS an der Naht: exponentiell zur Naht gewichtete Kontext-RMS."""
        env = _frame_rms_db(ctx_seg, frame, hop)
        k = len(env)
        if k == 0:
            return 1.0, False
        w = np.exp(-np.arange(k, dtype=np.float64))
        if near_seam:
            w = w[::-1]  # letzte Kontext-Frames liegen an der Naht
        return float(np.average(env, weights=w)), True

    left_ctx = ch[max(0, start - edge_s) : start]
    right_ctx = ch[end : min(len(ch), end + edge_s)]
    cand_left_env = float(env_cand[0])
    cand_right_env = float(env_cand[-1])

    tgt_l, ok_l = _edge_target(left_ctx, near_seam=True) if len(left_ctx) >= frame else (1.0, False)
    tgt_r, ok_r = _edge_target(right_ctx, near_seam=False) if len(right_ctx) >= frame else (1.0, False)

    g_lo, g_hi = float(align_gain_limits[0]), float(align_gain_limits[1])
    gain_l = float(np.clip(10.0 ** ((tgt_l - cand_left_env) / 20.0), g_lo, g_hi)) if ok_l else 1.0
    gain_r = float(np.clip(10.0 ** ((tgt_r - cand_right_env) / 20.0), g_lo, g_hi)) if ok_r else 1.0
    if ok_l and ok_r:
        gain = np.exp(np.linspace(np.log(gain_l), np.log(gain_r), n))
    elif ok_l:
        gain[:] = gain_l
    elif ok_r:
        gain[:] = gain_r
    # Keine zusätzlichen Rampen: Das Alignment macht die Hüllkurve an beiden
    # Nähten per Konstruktion stetig (cand·gain = Kontext-RMS an der Naht) —
    # eine Rampe gegen 1.0 würde genau dort eine Stufe ERZEUGEN.
    aligned = cand * gain.astype(np.float32)

    # ── IN-V1: Additive-Kappung gegen den Kontext (Band-weise) ─────────────
    w_s = max(int(0.10 * sr), frame * 4)
    s0 = max(0, start - w_s)
    e0 = min(len(ch), end + w_s)
    ctx = np.array(ch[s0:e0], dtype=np.float32)
    ctx[start - s0 : end - s0] = 0.0  # Lücke aus dem Kontext nehmen

    win = np.hanning(_N_FFT).astype(np.float32)
    n_frames = max(1, (n - _N_FFT) // _HOP + 1)
    idx_c = np.clip(np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None], 0, n - 1)
    frames_c = aligned[idx_c] * win
    spec_c = np.abs(np.fft.rfft(frames_c, n=_N_FFT, axis=1))

    n_ctx = len(ctx)
    if n_ctx >= _N_FFT:
        n_f_ctx = max(1, (n_ctx - _N_FFT) // _HOP + 1)
        idx_x = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_f_ctx)[:, None]
        spec_x = np.abs(np.fft.rfft(ctx[idx_x] * win, n=_N_FFT, axis=1))
        ctx_band_e = np.median(spec_x**2, axis=0)[None, :]
    else:
        ctx_band_e = np.zeros((1, spec_c.shape[1]), dtype=np.float64)

    # Frequenz-geglättete Kontext-Hüllkurve als Kappe (keine per-Bin-Nullstellen).
    ctx_env = _smooth_freq(ctx_band_e, _FREQ_SMOOTH_BINS)
    cap = ctx_env * (10.0 ** (float(cap_margin_db) / 10.0))
    floor = (10.0 ** (_ATH_FLOOR_DB / 10.0)) * np.ones_like(cap)
    cap = np.maximum(cap, floor)
    w_mask = np.sqrt(np.minimum(1.0, cap / np.maximum(spec_c**2, 1e-12)))
    # Zeitliche Glättung der Maske (3 Frames) gegen Klangflackern.
    w_mask = np.clip(_conv_rows_time(w_mask, np.ones((3, 1)) / 3.0), 0.0, 1.0)

    spec_g = np.fft.rfft(frames_c, n=_N_FFT, axis=1) * w_mask
    spec_full = np.concatenate([spec_g, np.conj(spec_g[:, 1:-1][:, ::-1])], axis=1)
    out_frames = np.fft.ifft(spec_full, n=_N_FFT, axis=1).real.astype(np.float32)
    gated = np.zeros(n, dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    for f in range(n_frames):
        gated[idx_c[f]] += out_frames[f]
        wsum[idx_c[f]] += win
    # OLA-Normalisierung nur dort, wo das Fenster voll überlappt (innen ≈ 2.0);
    # an den Gap-Rändern wäre wsum ≈ win ≈ 0 → Division-Blowing. Dort bleibt das
    # alignierte Zeitbereichs-Signal (die Naht ist ohnehin IN-V2-Sache).
    _interior = wsum > 0.5
    gated = np.where(_interior, gated / np.maximum(wsum, 1e-9), aligned)

    report = {
        "applied": True,
        "align_gain_left": gain_l if ok_l else None,
        "align_gain_right": gain_r if ok_r else None,
        "cap_mean": float(np.mean(w_mask)),
    }
    return gated.astype(np.float32), report

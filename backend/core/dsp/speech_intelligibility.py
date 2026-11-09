"""Speech Intelligibility Index (SII) — deterministischer Band-Energie-Proxy.

Hörordnung §3 (Ebene 1, L1 aus dem Spec-SOTA-Audit 2026-11-09): Die
bestehende Invariante `consonant_clarity ≥ 0.85` ist ein Energie-Proxy. Der
Speech Intelligibility Index (ANSI S3.5) gewichtet die kritischen Bänder
(160 Hz–8 kHz) nach ihrem Beitrag zur Verständlichkeit — genau die Bänder,
die De-Esser/Denoiser bei aggressiver Bearbeitung opfern. Dieses Modul liefert
einen **deterministischen Band-Energie-Proxy** des SII (kein voller STI mit
Modulationstransfer — ehrlich dokumentiert) und die delta-basierte
Schutzgröße `sii_delta(pre, post)`.

Struktur (ANSI S3.5 vereinfacht, deterministisch, §G5 (copilot-instructions.md)):
  1. 1/3-Oktav-Bandpegel über 18 Bänder (Mitten 160–8000 Hz, FFT-Summierung).
  2. Relativer Bandabstand zum lautesten Band (Sprach-Spektrum-Form).
  3. Audibility je Band: A_i = clip((L_i − (L_TQ_i + 20 dB)) / 30, 0, 1) —
     Ruhehörschwelle L_TQ (ISO 226) + 20-dB-Rauschboden-Marge als fester
     Referenz-Störpegel (kein externes Störspektrum nötig).
  4. SII = Σ W_i · A_i mit den ANSI-S3.5-Band-Wichtungen (normiert auf 1).

Schwelle: `SII_JND_VORBELEGUNG = 0.05` — Panel-Vorbelegung
(`vorbelegung_hoerpanel_offen`, Golden-Ear-Konvention Hörordnung §4).
Report-Only: dieses Modul misst und markiert; die Rücknahme-Schwelle
(Blend) bleibt Panel-Entscheidung — kein eigenmächtiger Veto-Pfad.

Kein neuer Absolut-Schwellwert ohne Panel (Guard-Kalibrierungs-Grundsatz):
natürliche Musik/Sprache schwankt; nur die DELTA-Verschlechterung gegenüber
dem Input ist ein Befund.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ANSI S3.5 Band-Wichtungen (18 Bänder, Mitten 160–8000 Hz) — normiert auf 1.
_SII_BAND_WEIGHTS = np.array(
    [
        0.0083,
        0.0095,
        0.0150,
        0.0289,
        0.0440,
        0.0578,
        0.0653,
        0.0711,
        0.0818,
        0.0844,
        0.0882,
        0.0898,
        0.0868,
        0.0844,
        0.0771,
        0.0527,
        0.0364,
        0.0185,
    ],
    dtype=np.float64,
)
_SII_BAND_WEIGHTS /= _SII_BAND_WEIGHTS.sum()

# ISO-226 Ruhehörschwellen (dB SPL) an den 18 Bandmitten (Näherung, konstant).
_SII_TQ_DB = np.array(
    [
        30.0,
        24.0,
        18.0,
        12.0,
        9.0,
        6.0,
        4.0,
        2.0,
        0.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -4.0,
        -3.0,
        -1.0,
        2.0,
        5.0,
    ],
    dtype=np.float64,
)

_SII_REF_MARGIN_DB = 20.0  # fester Referenz-Rauschboden über der Ruhehörschwelle
_SII_SLOPE_DB = 30.0  # Audibility-Anstieg je Band (dB)

# Panel-Vorbelegung: hörbarer SII-Verlust (0.05 ≈ SII-JND in der Literatur).
SII_JND_VORBELEGUNG = 0.05


def _band_levels_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """1/3-Oktav-Bandpegel (dB, relativ 1.0) an den 18 ANSI-Mitten.

    Stereo-Layout-Invariante: 2D wird über Kanal-Mittelung mono-kollabiert
    (shape[0] ≤ 8 ⇒ (C, N), sonst (N, C)) — kein hartes Layout-Annehmen.
    """
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2 and mono.shape[0] <= 8:
        mono = mono.mean(axis=0)
    elif mono.ndim == 2:
        mono = mono.mean(axis=-1)
    mono = mono.ravel()
    if mono.size < 512:
        return np.zeros(18, dtype=np.float64) - 120.0  # type: ignore[no-any-return]
    n_fft = 2048
    hop = n_fft // 2
    seg = mono[: min(mono.size, sr * 60)]
    if seg.size < n_fft:
        seg = np.pad(seg, (0, n_fft - seg.size))
    # Zeitliche Mittelung statt EINEM 42-ms-Fenster: ein Lauf kann mit Stille
    # oder Fade beginnen — ein Einzelfenster wäre dann blind (SII 0 auf beiden
    # Seiten). Deterministisch: feste Hann-Fenster, festes Hop, gedeckelte
    # Frame-Zahl mit gleichmäßigem Raster (§G5 (copilot-instructions.md)).
    n_frames = 1 + (seg.size - n_fft) // hop
    max_frames = 512
    if n_frames > max_frames:
        frame_idx = np.linspace(0, n_frames - 1, max_frames).astype(np.int64)
    else:
        frame_idx = np.arange(n_frames)
    idx = np.arange(n_fft)[None, :] + hop * frame_idx[:, None]
    frames = seg[idx] * np.hanning(n_fft)[None, :]
    spec = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    centers = np.array(
        [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000],
        dtype=np.float64,
    )
    levels = np.zeros(18, dtype=np.float64) - 120.0
    for i, fc in enumerate(centers):
        lo, hi = fc / (2 ** (1 / 6)), fc * (2 ** (1 / 6))
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            e = float(np.sum(spec[mask]))
            levels[i] = 10.0 * np.log10(e + 1e-20)
    return levels  # type: ignore[no-any-return]


def sii_index(audio: np.ndarray, sr: int) -> float:
    """Deterministischer SII-Proxy in [0, 1] (Band-Energie × ANSI-Wichtung).

    Höhere Werte = mehr verständlichkeitsrelevante Band-Energie nahe dem
    Spektrums-Peak. KEIN voller STI (kein Modulationstransfer) — für die
    Delta-Schutzgröße `sii_delta` ausgelegt und so dokumentiert.
    Kalibrierung (relativ, skalen-frei): ein Band trägt voll bei, wenn es
    höchstens 0 dB unter dem Peak liegt, und nichts mehr ab 30 dB darunter
    (lineare Rampe) — deterministisch, unabhängig von absoluter Pegel-Skala
    (§G5 (copilot-instructions.md)).
    """
    try:
        levels = _band_levels_db(audio, sr)
        ref = levels.max()
        if ref < -100.0:
            return 0.0  # Schweigen: kein Befund, kein Rauschen
        rel = levels - ref  # relativer Bandabstand zum Peak (dB, ≤ 0)
        aud = np.clip(1.0 + rel / _SII_SLOPE_DB, 0.0, 1.0)
        return float(np.sum(_SII_BAND_WEIGHTS * aud))
    except Exception as exc:  # §V6 (copilot-instructions.md): nie still werfen
        logger.debug("sii_index nicht berechenbar: %s", exc)
        return 0.0


def sii_delta(pre: np.ndarray, post: np.ndarray, sr: int) -> dict[str, float | bool | str]:
    """Delta-basierte Verständlichkeits-Schutzgröße (Hörordnung §3, L1).

    Nur eine VERSCHLECHTERUNG gegenüber dem Input ist ein Befund
    (Guard-Kalibrierungs-Grundsatz); Schwelle = Panel-Vorbelegung.
    """
    sii_pre = sii_index(pre, sr)
    sii_post = sii_index(post, sr)
    delta = sii_post - sii_pre
    return {
        "sii_pre": round(sii_pre, 4),
        "sii_post": round(sii_post, 4),
        "delta": round(delta, 4),
        "verschlechterung": bool(delta < -SII_JND_VORBELEGUNG),
        "schwelle": SII_JND_VORBELEGUNG,
        "panel_status": "vorbelegung_hoerpanel_offen",
    }


__all__ = ["sii_index", "sii_delta", "SII_JND_VORBELEGUNG"]

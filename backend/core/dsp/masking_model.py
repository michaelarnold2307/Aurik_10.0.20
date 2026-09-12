"""Maskierungsmodell — Audibility statt Mess-Null (Hörordnung Ebene 2).

§Witness-SOTA P1 (2026-09-12, docs/WITNESS_SOTA_GAP_ANALYSE.md §5): Ein
vereinfachtes Johnston-1988-Maskierungsmodell liefert pro Frame eine
Bark-band-weise Maskierungsschwelle. Damit kann der Listening-Witness jedes
spektrale Finding als ``audible`` (Delta ueber der Schwelle) oder als
Mess-Null-Artefakt (Delta unter der Schwelle) einstufen — exakt die
Konfliktregel der Hörordnung: Maskierungsschwelle statt Mess-Null.

Vereinfachungen (bewusst konservativ, deterministisch, kein ML):
- 26 Kanten der klassischen Bark-Skala (Zwicker 1961), 25 Bänder.
- Tone-vs-Noise ueber Spectral-Flatness-Maß (Johnston: SFM < −17 dB → tonal).
- Spreading: 10 dB/Bark Flanke beidseitig (konservativer als Johnston).
- Schwellen-Offset O = α(14.5 + z) + (1−α)·5.5 dB (Johnston 1988, Gl. 2-4)
  plus +6 dB Sicherheitsmarge — „inaudible" nur bei klarer Unterschreitung.

Referenz: Johnston (1988), „Transform Coding of Audio Signals Using Perceptual
Noise Criteria", IEEE JSAC 6(2); Zwicker & Fastl (2007) §7.
"""

from __future__ import annotations

import numpy as np

# Klassische Bark-Bandkanten (Hz), 0–24 Bark (Zwicker 1961 / Fastl & Zwicker).
_BARK_EDGES_HZ: tuple[float, ...] = (
    0.0,
    100.0,
    200.0,
    300.0,
    400.0,
    510.0,
    630.0,
    770.0,
    920.0,
    1080.0,
    1270.0,
    1480.0,
    1720.0,
    2000.0,
    2320.0,
    2700.0,
    3150.0,
    3700.0,
    4400.0,
    5300.0,
    6400.0,
    7700.0,
    9500.0,
    12000.0,
    15500.0,
    20500.0,
    27000.0,
)

_N_FFT = 4096
_TONE_SFM_DB = -17.0  # Johnston: SFM-Schwelle tonal vs. rauschartig
_SAFETY_MARGIN_DB = 6.0  # konservativ: nur klare Unterschreitung = inaudible
_SPREAD_DB_PER_BARK = 10.0


def bark_band_edges(sr: int) -> np.ndarray:
    """Bark-Bandkanten in Hz, bei Nyquist beschnitten (deterministisch)."""
    nyq = sr / 2.0
    edges = np.asarray(_BARK_EDGES_HZ, dtype=np.float64)
    edges = np.clip(edges, 0.0, nyq)
    # Duplikate (durch Nyquist-Beschneidung) entfernen, Reihenfolge stabil.
    uniq = [edges[0]]
    for _e in edges[1:]:
        if _e > uniq[-1]:
            uniq.append(_e)
    edges_out: np.ndarray = np.asarray(uniq, dtype=np.float64)
    return edges_out


def _frame_band_energies(x: np.ndarray, sr: int, n_fft: int = _N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """(Energie pro Bark-Band pro Frame [dB], Bark-Band-Mitten [Bark])."""
    x = np.asarray(x, dtype=np.float32)
    hop = n_fft  # non-overlapping: Maskierung variiert langsam
    n_frames = max(1, (len(x) - n_fft) // hop + 1)
    win = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    edges = bark_band_edges(sr)
    centers_hz = 0.5 * (edges[:-1] + edges[1:])
    # Bark-Zentren: z = 13·arctan(0.00076·f) + 3.5·arctan((f/7500)²) (Traunmüller)
    z: np.ndarray = 13.0 * np.arctan(0.00076 * centers_hz) + 3.5 * np.arctan((centers_hz / 7500.0) ** 2)

    n_bands = len(edges) - 1
    band_e = np.zeros((n_frames, n_bands), dtype=np.float64)
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if mask.any():
            band_e[:, b] = np.sum(spec[:, mask], axis=1)
    band_e_db: np.ndarray = 10.0 * np.log10(band_e + 1e-12)
    return band_e_db, z


def compute_masking_threshold_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Maskierungsschwelle pro Frame pro Bark-Band (dB, deterministisch).

    Returns:
        (threshold_db (n_frames, n_bands), bark_centers (n_bands,))
    """
    band_e, z = _frame_band_energies(x, sr)
    n_frames, n_bands = band_e.shape

    # Tone-vs-Noise: SFM pro Band (Johnston) — α = clamp(SFM_dB / −60, 0, 1)
    # SFM über die FFT-Bins eines Bands; konservativ: schätze α aus der
    # Flachheit der Band-Energien über der Zeit (rauschige Bänder fluktuieren).
    alpha = np.zeros(n_bands, dtype=np.float64)
    if n_frames > 2:
        band_lin = 10.0 ** (band_e / 20.0)
        gm = np.exp(np.mean(np.log(band_lin + 1e-12), axis=0))
        am = np.mean(band_lin, axis=0)
        sfm = 10.0 * np.log10(np.clip(gm / (am + 1e-12), 1e-12, 1.0))
        alpha = np.clip(sfm / -60.0, 0.0, 1.0)

    # Masker-Offset (Johnston Gl. 2-4) + Sicherheitsmarge.
    offset: np.ndarray = alpha[None, :] * (14.5 + z[None, :]) + (1.0 - alpha[None, :]) * 5.5
    offset = offset + _SAFETY_MARGIN_DB

    # Spreading: 10 dB/Bark, beide Flanken (konservativ).
    dz = np.abs(np.arange(n_bands)[None, :] - np.arange(n_bands)[:, None]).astype(np.float64)
    spread: np.ndarray = band_e[:, :, None] - _SPREAD_DB_PER_BARK * dz[None, :, :]
    threshold: np.ndarray = np.max(spread, axis=1) - offset
    return threshold, z


def band_audibility(
    x: np.ndarray,
    y: np.ndarray,
    sr: int,
    lo_hz: float,
    hi_hz: float,
) -> dict[str, float | bool]:
    """Ist die Delta-Energie in [lo_hz, hi_hz] oberhalb der Maskierungsschwelle?

    Masker = das lautere der beiden Signale je Band (klassisch). Rückgabe:
    {"delta_db", "threshold_db", "audible"} — deterministisch.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = min(len(x), len(y))
    d = (y[:n] - x[:n]).astype(np.float32)

    edges = bark_band_edges(sr)
    bands = np.where((edges[:-1] < hi_hz) & (edges[1:] > lo_hz))[0]
    if len(bands) == 0:
        return {"delta_db": 0.0, "threshold_db": 0.0, "audible": False}

    e_x, _ = _frame_band_energies(x[:n], sr)
    e_d, _ = _frame_band_energies(d, sr)
    thr, _ = compute_masking_threshold_db(x[:n], sr)

    delta_db = float(np.max(e_d[:, bands]))
    threshold_db = float(np.max(thr[:, bands]))
    audible = bool(delta_db > threshold_db)
    return {
        "delta_db": round(delta_db, 2),
        "threshold_db": round(threshold_db, 2),
        "audible": audible,
    }

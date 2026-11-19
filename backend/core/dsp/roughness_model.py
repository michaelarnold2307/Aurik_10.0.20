"""Rauigkeit (Roughness) — deterministischer Fluktuations-Schätzer.

§Witness-SOTA P2 (2026-09-12, docs/WITNESS_SOTA_GAP_ANALYSE.md §5): Rauigkeit
ist die Wahrnehmungsgröße für „harsch/kratzig" (Amplitudenmodulation 15–300 Hz).
Vereinfachung auf eine deterministische Hüllkurven-Fluktuationsmessung:
Hilbert-Hüllkurve → 400-Hz-Raster → Modulationsspektrum 20–150 Hz → relative
Modulationstiefe. Band-weise Vassilakis-Verfeinerung bleibt dokumentierter
Folgeschritt (die hier gewählte Breitband-Variante ist für das Witness-Delta
ausreichend und O(n·log n)).

Referenz: Vassilakis (2001) „Perceptual and Physical Properties of Amplitude
Fluctuation"; Zwicker & Fastl (2007) §11.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert as _hilbert

_AM_LO_HZ = 20.0
_AM_HI_HZ = 150.0
_ENV_SR = 400.0  # Hüllkurven-Raster (Hz)


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Hilbert-Hüllkurve auf 400-Hz-Raster (deterministisch)."""
    xa = np.asarray(x, dtype=np.float32)
    env_full = np.abs(_hilbert(xa)).astype(np.float64)
    win = max(1, int(sr / _ENV_SR))
    n_win = len(env_full) // win
    env: np.ndarray = env_full[: n_win * win].reshape(n_win, win).mean(axis=1)
    return env


def compute_roughness_asper(x: np.ndarray, sr: int) -> float:
    """Rauigkeit in Asper (relative Skala, deterministisch).

    Returns:
        float >= 0 — höher = rauer. Relative Größe (kalibriert an der
        Modulationstiefe der Hüllkurve), kein absoluter Vassilakis-Wert.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < int(sr * 0.5):
        return 0.0
    env = _envelope(x, sr)
    if len(env) < 8:
        return 0.0
    env = env - float(np.mean(env))
    am = np.abs(np.fft.rfft(env)) ** 2
    freqs = np.fft.rfftfreq(len(env), d=1.0 / _ENV_SR)
    mask = (freqs >= _AM_LO_HZ) & (freqs <= _AM_HI_HZ)
    if not mask.any():
        return 0.0
    depth = float(np.sqrt(np.sum(am[mask]) + 1e-12))
    mean_amp = float(np.mean(np.abs(env)) + 1e-12)
    return depth / mean_amp


def roughness_rise_asper(x: np.ndarray, y: np.ndarray, sr: int) -> float:
    """Rauigkeits-Zuwachs von x → y (signed, Asper). Negativ = glatter."""
    n = min(len(x), len(y))
    r_x = compute_roughness_asper(x[:n], sr)
    r_y = compute_roughness_asper(y[:n], sr)
    return float(r_y - r_x)

"""backend/core/dsp/ssm_segmentation.py — kanonische SSM-Segmentierung (Foote 2000).

§SOTA-Analogie-Korrektur 2026-09-17: Die §2.52b-Songstruktur nutzte eine
agglomerative k-Heuristik (1 Grenze / 30 s) als Grenz-Detektor, während die
definierende Evidenz — die Novelty-Kurve der Self-Similarity-Matrix — bereits
im §2.17-MusicalStructureAnalyzer implementiert war. Zwei Analysatoren, zwei
Methoden, potenziell widersprüchliche Sektions-Karten in EINEM Pipeline-Lauf
(§V7 (copilot-instructions.md): eine Lösung pro Rolle). Dieses Modul ist die
EINE kanonische Methode; beide Analysatoren delegieren hierher.

Referenz: Foote (2000), „Automatic Audio Segmentation using a Measure of
Audio Novelty“; Müller FMP §4.4. Deterministisch (§G5 (GEBOTE.md)), rein numpy.
"""

from __future__ import annotations

import numpy as np


def checkerboard_novelty(ssm: np.ndarray, kernel_half_size: int = 8) -> np.ndarray:
    """Schachbrett-Kernel auf die SSM → Novelty-Kurve (Foote 2000).

    Kernel: +1 auf den Diagonal-Blöcken, −1 außerhalb (Gauß-tapered).
    """
    n = ssm.shape[0]
    k = max(1, int(kernel_half_size))
    novelty = np.zeros(n, dtype=np.float32)

    g = np.exp(-(np.arange(-k, k + 1) ** 2) / (2.0 * (k / 2.0) ** 2))
    kernel = np.outer(g, g)
    mask = np.ones((2 * k + 1, 2 * k + 1), dtype=np.float32)
    mask[:k, k + 1 :] = -1.0
    mask[k + 1 :, :k] = -1.0
    mask[:k, :k] = 1.0
    mask[k + 1 :, k + 1 :] = 1.0
    mask[k, :] = 0.0
    mask[:, k] = 0.0
    kernel = kernel * mask

    for t in range(k, n - k):
        block = ssm[t - k : t + k + 1, t - k : t + k + 1]
        novelty[t] = float(np.sum(block * kernel))

    return np.clip(novelty, 0.0, None)  # type: ignore[no-any-return]


def gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1-D-Gauß-Glättung via Faltung."""
    if sigma <= 0:
        return x
    half = max(1, int(3 * sigma))
    t = np.arange(-half, half + 1)
    kernel = np.exp(-(t**2) / (2 * sigma**2)).astype(np.float32)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same").astype(np.float32)  # type: ignore[no-any-return]


def pick_novelty_peaks(novelty: np.ndarray, min_dist: int = 8) -> list[int]:
    """Peak-Picking mit Mindestabstand (deterministisch)."""
    n = len(novelty)
    if n == 0:
        return []
    threshold = float(novelty.mean() + 0.5 * novelty.std())
    peaks: list[int] = []
    last_peak = -min_dist - 1
    for i in range(1, n - 1):
        if (
            novelty[i] > threshold
            and novelty[i] >= novelty[i - 1]
            and novelty[i] >= novelty[i + 1]
            and (i - last_peak) >= min_dist
        ):
            peaks.append(i)
            last_peak = i
    return peaks


def ssm_boundaries_from_chroma(
    chroma: np.ndarray,
    hop: int,
    sr: int,
    duration_s: float,
    kernel_half_size: int = 8,
    sigma: float = 3.0,
    min_seg_s: float = 4.0,
) -> tuple[list[int], float]:
    """Segment-Grenzen (Samples) aus Chroma via SSM + Checkerboard-Novelty.

    Args:
        chroma:      (12, T) Chroma-Features (beliebige Skala — wird intern
                     frame-weise auf Einheitslänge normalisiert).
        hop:         Hop der Chroma-Frames in Samples.
        sr:          Abtastrate.
        duration_s:  Gesamtdauer (für die Konfidenz-Skalierung).
        kernel_half_size/sigma/min_seg_s: SSM-Parameter (Foote 2000).

    Returns:
        (boundary_samples, confidence) — Grenzen aufsteigend, inklusive 0 und
        letztem Sample; Konfidenz in [0, 1]. Deterministisch.
    """
    n_frames = chroma.shape[1]
    if n_frames < 2 * kernel_half_size + 4:
        return [0, chroma.shape[1] * hop], 0.0

    col_norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
    chroma_n = chroma / col_norms  # (12, T)

    ssm = chroma_n.T @ chroma_n  # Kosinus ∈ [−1, 1]
    ssm = np.clip((ssm + 1.0) / 2.0, 0.0, 1.0)

    novelty = checkerboard_novelty(ssm, kernel_half_size)
    novelty = gauss_smooth(novelty, sigma)

    min_seg_frames = max(2, int(min_seg_s * sr / hop))
    boundary_frames = pick_novelty_peaks(novelty, min_dist=min_seg_frames)
    boundary_frames = sorted({0, *boundary_frames, n_frames})
    boundary_frames = [b for b in boundary_frames if 0 <= b <= n_frames]

    n_total = int(round(n_frames * hop))
    bounds: list[int] = []
    for f in boundary_frames:
        samp = min(n_total, max(0, int(f * hop)))
        bounds.append(samp)
    if not bounds or bounds[-1] != n_total:
        bounds.append(n_total)
    if bounds[0] != 0:
        bounds.insert(0, 0)
    bounds = sorted(set(bounds))

    if float(novelty.std()) > 0:
        conf = float(np.clip(float(novelty.std()) * 4.0 + 0.5, 0.0, 1.0))
    else:
        conf = 0.4
    conf = float(np.clip(conf + min(0.3, duration_s / 180.0), 0.0, 1.0))
    return bounds, conf

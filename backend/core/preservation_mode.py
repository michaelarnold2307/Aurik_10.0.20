"""PreservationMode — §INCREMENTAL #11.

Nur analysieren, nichts ändern. Vertrauensbildende Maßnahme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreservationReport:
    material: str = ""
    era: int = 0
    defects_found: dict[str, float] = field(default_factory=dict)
    would_apply_strategies: list[str] = field(default_factory=list)
    estimated_improvement: float = 0.0
    recommendation: str = ""


def analyze_only(audio: np.ndarray, sr: int, material: str = "unknown", era: int = 0) -> PreservationReport:
    """Analysiert ohne zu ändern. Sagt was es tun WÜRDE."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)

    # Defekt-Erkennung (vereinfacht)
    defects = {}
    n_fft = min(4096, len(mono))
    spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Hiss
    log_mean = np.exp(np.mean(np.log(spec + 1e-10)))
    arith_mean = np.mean(spec)
    noise_ratio = float(log_mean / max(arith_mean, 1e-10))
    if noise_ratio > 0.6:
        defects["hiss"] = noise_ratio

    # Clicks
    hf = np.sum(spec[freqs >= 6000] ** 2) / max(np.sum(spec**2), 1e-10)
    if hf > 0.05:
        defects["clicks"] = float(hf)

    # Hum
    hum_e = sum(np.sum(spec[(freqs >= lo) & (freqs <= hi)] ** 2) for lo, hi in [(45, 65), (95, 125)])
    hum_r = hum_e / max(np.sum(spec**2), 1e-10)
    if hum_r > 0.1:
        defects["hum"] = float(hum_r)

    # Clipping
    clip_pct = float(np.mean(np.abs(mono) > 0.95))
    if clip_pct > 0.01:
        defects["clipping"] = clip_pct

    # Strategie-Empfehlung
    strategies = []
    if defects:
        strategies.append("light")
    if len(defects) >= 2:
        strategies.append("balanced")
    if len(defects) >= 3:
        strategies.append("deep")

    improvement = min(3.0, len(defects) * 0.5)
    rec = "Restaurierung empfohlen" if defects else "Keine Restaurierung nötig — Audio ist bereits sauber"

    return PreservationReport(
        material=material,
        era=era,
        defects_found=defects,
        would_apply_strategies=strategies,
        estimated_improvement=round(improvement, 1),
        recommendation=rec,
    )

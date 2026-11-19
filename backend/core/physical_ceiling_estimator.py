"""
backend/core/physical_ceiling_estimator.py
Aurik 10.0.0 -- Spec §2.33: PhysicalCeilingEstimator

Schaetzt informationstheoretische Qualitaets-Obergrenzen
aus Quell-SNR und Bandbreite.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HEADROOM_THRESHOLD: float = 0.03  # < 3 % -> keine weiteren Iterationen


@dataclass
class PhysicalCeilingResult:
    """Spec §2.33: Informationstheoretische Qualitaets-Obergrenzen."""

    ceiling: dict[str, float]
    snr_profile_db: np.ndarray
    effective_bandwidth_hz: float
    headroom_per_goal: dict[str, float]
    further_optimization_worthwhile: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialisierung fuer Logging und API-Antworten (§2.22, §3.6)."""
        return {
            "ceiling": dict(self.ceiling),
            "effective_bandwidth_hz": float(self.effective_bandwidth_hz),
            "further_optimization_worthwhile": bool(self.further_optimization_worthwhile),
            "headroom_per_goal": dict(self.headroom_per_goal),
            "snr_profile_db": self.snr_profile_db.tolist(),
        }


class PhysicalCeilingEstimator:
    """Spec §2.33: Schaetzt informationstheoretische Qualitaetsgrenzen.

    Algorithmus:
        1. SNR pro Bark-Band (IMCRA-Minima-Schaetzung, 24 Baender)
        2. Effektive Bandbreite: hoechstes Bark-Band mit SNR >= -10 dB
        3. Musical-Goal-Ceiling-Mapping
        4. further_optimization_worthwhile Flag
    """

    HEADROOM_THRESHOLD: float = HEADROOM_THRESHOLD

    def __init__(self) -> None:
        self._last_ceiling: dict[str, float] | None = None

    BARK_EDGES_HZ = [
        20,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ]

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        current_goal_scores: dict[str, float],
        material: str = "unknown",
    ) -> PhysicalCeilingResult:
        """Schaetzt physikalische Qualitaetsdecke.

        Args:
            audio: float32 ndarray
            sr: Sample-Rate in Hz
            current_goal_scores: Aktuelle Musical-Goal-Scores
            material: MaterialType-Label

        Returns:
            PhysicalCeilingResult
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio.copy()

        # 1. SNR pro Bark-Band
        snr_profile = self._compute_bark_snr(mono, sr)

        # 2. Effektive Bandbreite
        bw_hz = self._effective_bandwidth(mono, sr, snr_profile)

        # 3. Ceiling-Mapping
        mean_snr = float(np.mean(snr_profile))
        ceiling = self._compute_ceiling(mean_snr, bw_hz, mono, sr, material)

        # NaN-Guard
        for g in ceiling:
            if not math.isfinite(ceiling[g]):
                ceiling[g] = 0.98

        # Cache for ceiling_avg()
        self._last_ceiling = ceiling

        # 4. Headroom + Flag
        headroom: dict[str, float] = {}
        for g, ceil_v in ceiling.items():
            cur = current_goal_scores.get(g, 0.0)
            if not math.isfinite(cur):
                cur = 0.0
            headroom[g] = max(0.0, ceil_v - cur)

        worthwhile = any(h >= self.HEADROOM_THRESHOLD for h in headroom.values())

        return PhysicalCeilingResult(
            ceiling=ceiling,
            snr_profile_db=snr_profile,
            effective_bandwidth_hz=bw_hz,
            headroom_per_goal=headroom,
            further_optimization_worthwhile=worthwhile,
        )

    def ceiling_avg(self) -> float:
        """Gibt arithmetic mean of all Musical-Goal ceiling values from last estimate() call zurück.

        Spec §2.33: ceiling_avg() is used to derive headroom tiers 1.00 / 0.93 / 0.85 / 0.75.
        Returns 0.85 if estimate() has not been called yet (conservative fallback).
        """
        if self._last_ceiling is None:
            return 0.85
        values = [v for v in self._last_ceiling.values() if math.isfinite(v)]
        return float(np.mean(values)) if values else 0.85

    def _compute_bark_snr(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """SNR pro Bark-Band (24 Baender), float32 [24]."""
        n = min(len(mono), 65536)
        if n < 512:
            return np.full(24, 20.0, dtype=np.float32)  # type: ignore[no-any-return]

        spec = np.abs(np.fft.rfft(mono[:n])) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        bark_snr = np.zeros(24, dtype=np.float32)
        edges = self.BARK_EDGES_HZ

        for b in range(24):
            lo, hi = edges[b], edges[b + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                bark_snr[b] = -30.0
                continue
            band_energy = spec[mask]
            band_energy_sorted = np.sort(band_energy)
            n_floor = max(1, len(band_energy_sorted) // 10)
            noise_e = float(np.mean(band_energy_sorted[:n_floor])) + 1e-20
            signal_e = float(np.mean(band_energy_sorted[-n_floor:])) + 1e-20
            snr = 10.0 * math.log10(signal_e / noise_e)
            bark_snr[b] = float(np.clip(snr, -60.0, 80.0))

        return bark_snr  # type: ignore[no-any-return]

    def _effective_bandwidth(self, mono: np.ndarray, _sr: int, snr_profile: np.ndarray) -> float:
        """Effektive Bandbreite: hoechstes Bark-Band mit SNR >= -10 dB."""
        edges = self.BARK_EDGES_HZ
        _ = mono
        bw: float = float(edges[1])  # Minimum
        for b in range(24):
            if snr_profile[b] >= -10.0:
                bw = float(edges[b + 1])
        return bw

    def _compute_ceiling(
        self,
        mean_snr: float,
        bw_hz: float,
        mono: np.ndarray,
        _sr: int,
        material: str,
    ) -> dict[str, float]:
        """Musical-Goal-Ceiling-Mapping (Spec §2.33)."""

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        # Natuerlichkeit: sigmoid((mean_snr - 5) / 5) * 0.97 + 0.03
        nat = sigmoid((mean_snr - 5.0) / 5.0) * 0.97 + 0.03

        # Brillanz: sigmoid((bw_hz - 8000) / 2000) * 0.98 + 0.02
        brill = sigmoid((bw_hz - 8000.0) / 2000.0) * 0.98 + 0.02

        # Stereo-Dekorrelation (vereinfacht)
        if mono.ndim == 1 or len(mono) < 100:
            stereo_decor = 0.0
        else:
            # Proxy: Spektral-Flatness als Stereo-Proxy
            spec = np.abs(np.fft.rfft(mono[: min(len(mono), 8192)])) ** 2 + 1e-15
            gm = math.exp(float(np.mean(np.log(spec))))
            am = float(np.mean(spec))
            stereo_decor = float(np.clip(gm / (am + 1e-10), 0.0, 1.0))

        spatial = sigmoid(stereo_decor * 10.0) * 0.92

        # Wow/Flutter -> GrooveMetric
        groove = 0.97  # Standard-Decke

        # TonalCenter -> SNR-abhaengig
        tonal_center = sigmoid(mean_snr * 2.0) * 0.98

        ceiling = {
            "natuerlichkeit": min(0.99, nat),
            "brillanz": min(0.98, brill),
            "spatial_depth": min(0.95, spatial),
            "groove": min(0.98, groove),
            "tonal_center": min(0.99, tonal_center),
            # Alle anderen: konservativ 0.99 (material-spezifische Caps unten)
            "waerme": 0.99,
            "authentizitaet": 0.99,
            "emotionalitaet": 0.99,
            "transparenz": 0.99,
            "bass_kraft": 0.99,
            "timbre_authentizitaet": 0.99,
            "micro_dynamics": 0.99,
            "separation_fidelity": 0.99,
            "artikulation": 0.99,
            "transient_energie": 0.99,
        }

        # Shellac/Wachswalze: kuenstliche Deckel (physikalische Limitierung)
        if material in {"shellac", "wax_cylinder"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.75)
            ceiling["natuerlichkeit"] = min(ceiling["natuerlichkeit"], 0.85)
            for _g in ceiling:
                ceiling[_g] = min(ceiling[_g], 0.95)

        # Vinyl: moderate Deckel
        elif material in {"vinyl", "vinyl_lp"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.90)

        # MP3 (Verlustbehaftet): Codec-Artefakt-bedingter Brillanz-Deckel
        elif material in {"mp3_low", "mp3_high", "mp3"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.85)

        # Tape/Kassette: Deckel für HF-Metriken — rekalibriert mit material-adaptiver Formel
        # §9.12.7 [BUG-FIX v10.0.0]: Ceiling an neue BrillanzMetric-Formeldynamik angepasst.
        # Mit material-adaptiver Kalibration (offset=0.10, divisor=1.20) gibt tape crest_peak≈8
        # score≈0.67, crest_peak≈12 score≈0.82. Physikalische Obergrenze: 0.78/0.50.
        elif material in {"tape", "cassette"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.78)
            ceiling["transparenz"] = min(ceiling["transparenz"], 0.50)
        elif material in {"reel_tape"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.85)
            ceiling["transparenz"] = min(ceiling["transparenz"], 0.62)

        return ceiling


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: PhysicalCeilingEstimator | None = None
_lock = threading.Lock()


def get_physical_ceiling_estimator() -> PhysicalCeilingEstimator:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance  # pylint: disable=global-statement  # §3.2 Pflicht-Singleton-Pattern
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PhysicalCeilingEstimator()
    return _instance


def estimate_physical_ceiling(
    audio: np.ndarray,
    sr: int,
    current_goal_scores: dict[str, float],
    material: str = "unknown",
) -> PhysicalCeilingResult:
    """Convenience-Wrapper."""
    return get_physical_ceiling_estimator().estimate(audio, sr, current_goal_scores, material)


__all__ = [
    "HEADROOM_THRESHOLD",
    "PhysicalCeilingEstimator",
    "PhysicalCeilingResult",
    "estimate_physical_ceiling",
    "get_physical_ceiling_estimator",
]

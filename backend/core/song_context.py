"""SongContext — §STRATEGIC: Vorausschauende Phasen-Steuerung.

Jede Phase sieht, was VOR ihr und NACH ihr im Song passiert.
Ergebnis: Kohärenz ist eingebaut, nicht nachträglich gefixt.

Fließt durch die Pipeline:
  Pre-Analysis → SongContext → Phase 1..66 → Export
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SongContext:
    """Kontext-Information, die jede Phase für kohärente Entscheidungen nutzt."""

    # ── Look-Ahead/Look-Behind Fenster ────────────────────────────────
    audio_before: np.ndarray | None = None  # ~1s Audio VOR diesem Segment
    audio_after: np.ndarray | None = None  # ~1s Audio NACH diesem Segment

    # ── Spektrale Referenz ────────────────────────────────────────────
    reference_centroid_hz: float = 2000.0  # Soll-Centroid (vom Nachbar)
    reference_rms_db: float = -20.0  # Soll-RMS (vom Nachbar)
    reference_brightness: float = 0.5  # Soll-Brightness

    # ── Strategie-Kohärenz ────────────────────────────────────────────
    neighbor_strategies: list[str] = field(default_factory=list)  # Strategien der Nachbarn
    max_gap_to_neighbors: int = 0  # 0=gleiche Strategie, 2=max erlaubt

    # ── Phasen-Entscheidungshilfe ─────────────────────────────────────
    should_preserve_transients: bool = True  # Nachbar hat viele Transienten
    should_match_tonality: bool = False  # Nachbar hat andere Tonalität
    target_lufs: float = -18.0  # Ziel-LUFS vom Album-Target

    # ── Validierung ───────────────────────────────────────────────────
    coherence_score: float = 1.0  # 0–1 (1=perfekt kohärent)


class SongContextBuilder:
    """Baut SongContext aus Segment-Informationen vor der Pipeline."""

    LOOKAHEAD_S: float = 1.0  # 1 Sekunde Look-Ahead/Behind

    @staticmethod
    def build(
        audio: np.ndarray,
        sr: int,
        segment_index: int = 0,
        total_segments: int = 1,
        neighbor_strategies: list[str] | None = None,
        album_target_lufs: float = -18.0,
    ) -> SongContext:
        """Erstellt SongContext für ein Segment.

        Args:
            audio: Das aktuelle Segment
            sr: Sample-Rate
            segment_index: Position im Song (0=Anfang)
            total_segments: Gesamtzahl Segmente
            neighbor_strategies: Strategien der Nachbar-Segmente
            album_target_lufs: Ziel-LUFS für Album-Konsistenz

        Returns:
            SongContext mit Look-Ahead/Behind Information
        """
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)

        # Referenz-Metriken dieses Segments
        n_fft = min(4096, len(mono))
        spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        total_e = float(np.sum(spec**2)) + 1e-10
        centroid = float(np.sum(freqs * spec**2) / total_e)
        rms = float(np.sqrt(np.mean(mono**2))) + 1e-10

        # Brightness via spectral flatness
        log_mean = np.exp(np.mean(np.log(spec + 1e-10)))
        arith_mean = np.mean(spec)
        brightness = float(np.clip(1.0 - log_mean / max(arith_mean, 1e-10), 0.0, 1.0))

        # Strategie-Gap zu Nachbarn
        strategy_order = {"passthrough": 0, "light": 1, "balanced": 2, "deep": 3, "full": 4}
        neighbor_strategies = neighbor_strategies or []
        max_gap = 0
        if neighbor_strategies:
            my_order = strategy_order.get("balanced", 2)
            for ns in neighbor_strategies:
                gap = abs(strategy_order.get(ns, 2) - my_order)
                max_gap = max(max_gap, gap)

        return SongContext(
            reference_centroid_hz=centroid,
            reference_rms_db=20.0 * np.log10(rms),
            reference_brightness=brightness,
            neighbor_strategies=neighbor_strategies,
            max_gap_to_neighbors=max_gap,
            should_match_tonality=max_gap > 1,
            should_preserve_transients=segment_index > 0,
            target_lufs=album_target_lufs,
            coherence_score=1.0 - max_gap * 0.25,
        )

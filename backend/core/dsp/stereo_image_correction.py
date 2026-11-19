"""
stereo_image_correction.py — Stereo-Image-Korrektur für Aurik.

M/S-basierte Korrektur der Stereobreite. Layout-sicher
(Stereo-Layout-Invariante, AGENTS.md): beide Layouts (C, N) und
(N, C) werden bedient, Mono und Mehrkanal bleiben unverändert.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.core.audio_layout import is_channels_first, is_samples_first

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "stereo_image_correction"
    category: str = "spatial"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
stereo_image_correction_contract = DSPContract(
    io={
        "channels": "stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"target_width": 1.0},
        "safe_ranges": {"target_width": {"min": 0.0, "max": 2.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[
        {
            "risk": "Phasenauslöschung",
            "expected_when": "target_width > 1.5",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["stereo_width"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class StereoImageCorrection:
    """
    Stereo-Image-Korrektur:
    - Korrigiert Stereobreite per M/S-Skalierung (Phasenlage bleibt erhalten)
    """

    def __init__(self, target_width: float = 1.0):
        self.target_width = target_width

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """M/S-basierte Stereo-Image-Korrektur.

        1. L/R  -> M/S Transformation
        2. Side-Kanal mit target_width skalieren (< 1.0 = enger, > 1.0 = breiter)
        3. M/S  -> L/R Rücktransformation
        4. Mono/Mehrkanal/ambiges Layout: direkte Rückgabe (kein Stereo-Processing möglich)
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        # Mono- und Nicht-Stereo-Fallback
        if audio.ndim != 2 or 1 in audio.shape or 2 not in audio.shape:
            return audio
        channels_first = is_channels_first(audio)
        if not channels_first and not is_samples_first(audio):
            logger.warning("Ambiges Kanal-Layout — StereoImageCorrection bleibt unverändert (§V6 (VERBOTEN.md))")
            return audio
        if channels_first:
            L = audio[0].astype(np.float64)
            R = audio[1].astype(np.float64)
        else:
            L = audio[:, 0].astype(np.float64)
            R = audio[:, 1].astype(np.float64)
        M = (L + R) * 0.5
        S = (L - R) * 0.5
        # Stereobreite skalieren: target_width=1.0 -> unverändert
        S_scaled = S * float(self.target_width)
        L_out = M + S_scaled
        R_out = M - S_scaled
        # Pegelkompensation: Energie erhalten
        gain = 1.0 / max(1e-9, np.sqrt(0.5 * (1.0 + float(self.target_width) ** 2)))
        if channels_first:
            out = np.stack([L_out * gain, R_out * gain], axis=0)
        else:
            out = np.stack([L_out * gain, R_out * gain], axis=1)
        return out.astype(audio.dtype)  # type: ignore[no-any-return]

"""Vocal-Gender (regelbasierte Fassade) — delegiert an den kanonischen Kern.

Historisch war dies eine eigenständige Regel-Heuristik (FFT-Peak-Pitch + eigene
LPC-Formanten mit eigenen Schwellen). Seit 2026-10-06 gibt es nur noch EINEN
Detektor-Pfad: die Multi-Evidenz-Fusion in
``backend.core.vocal_ai_enhancement.GenderDetector`` (§G9 copilot-instructions.md).
Diese Klasse bleibt als API-kompatible Fassade für Batch-/Eval-Aufrufer erhalten
und enthält KEINE eigene Klassifikationslogik mehr — sonst entstünde ein zweiter,
abweichend kalibrierter Gender-Begriff im Projekt.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.forensics.gender_detection import GenderDetector as _KanonFassade

logger = logging.getLogger(__name__)


class RuleBasedGenderDetector:
    """API-kompatible Fassade über den kanonischen SOTA-Gender-Pfad (§19)."""

    def __init__(self, sr: int = 16000) -> None:
        self.sr = int(sr)
        self._detector = _KanonFassade(sample_rate=self.sr)

    def detect_gender(self, audio_file: Any, panns_tags: Any = None) -> str:
        """Erkennt ``male``/``female``/``child``/``unknown`` (kanonischer Pfad)."""
        return self._detector.detect_gender(audio_file, panns_tags=panns_tags)

    def detect_gender_array(self, audio: Any, sample_rate: int | None = None, panns_tags: Any = None) -> str:
        """Array-Pfad (bevorzugt) — dieselbe kanonische Multi-Evidenz-Fusion."""
        return self._detector.detect_gender_array(audio, sample_rate or self.sr, panns_tags=panns_tags)

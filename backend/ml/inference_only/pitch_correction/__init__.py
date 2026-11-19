"""
AURIK v8.2 - Conservative Pitch Correction Module

Provides SOTA-level pitch correction with HIPS compliance and epistemic safety.

Features:
- CREPE-based pitch detection (state-of-the-art neural pitch tracking)
- Conservative correction (only unambiguous errors > 25 cents)
- Vibrato preservation (never touches intentional pitch variation)
- Glissando detection (never touches slides)
- Formant preservation (mandatory for natural sound)

Philosophy:
This module follows the "First, do no harm" principle. It will REJECT correction
when there is epistemic ambiguity about whether a pitch deviation is an error
or musical expression.

HIPS Compliance:
- Kontextbewusstsein: ✅ Analyzes 2s windows for vibrato/glissando detection
- Nebenwirkungen: ✅ Modeled (formant shift, robotic sound, transient loss)
- Reversibilität: ✅ Original audio always preserved
- Auditierbarkeit: ✅ All correction decisions logged
- Steuerbarkeit: ✅ Configurable thresholds and correction strength
"""

from .conservative_corrector import ConservativePitchCorrector
from .pitch_detector import CREPEPitchDetector

__all__ = ["CREPEPitchDetector", "ConservativePitchCorrector"]

__version__ = "9.15.0"

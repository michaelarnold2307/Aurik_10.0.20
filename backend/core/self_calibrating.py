"""SelfCalibrating — §INCREMENTAL #12.

PhaseImpactRecorder kalibriert Schwellwerte automatisch.
Nach 100 Songs: SKIP_THRESHOLD = datengetrieben statt −0.15.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibratedThresholds:
    skip_threshold: float = -0.15
    reduce_threshold: float = -0.05
    strong_apply_threshold: float = 0.10
    n_samples: int = 0
    calibrated: bool = False


def calibrate() -> CalibratedThresholds:
    """Kalibriert Schwellwerte aus gesammelten Impact-Daten."""
    try:
        from backend.core.phase_impact_recorder import get_phase_impact_recorder

        rec = get_phase_impact_recorder()

        if not rec._session_impacts:
            return CalibratedThresholds()

        deltas = [i.quality_delta for i in rec._session_impacts if abs(i.quality_delta) < 1.0]
        if len(deltas) < 10:
            return CalibratedThresholds(n_samples=len(deltas))

        # Perzentil-basierte Kalibrierung
        p10 = float(np.percentile(deltas, 10))  # Schlechteste 10%
        p25 = float(np.percentile(deltas, 25))  # Untere 25%
        p75 = float(np.percentile(deltas, 75))  # Obere 25%

        return CalibratedThresholds(
            skip_threshold=round(p10, 3),
            reduce_threshold=round(p25, 3),
            strong_apply_threshold=round(p75, 3),
            n_samples=len(deltas),
            calibrated=True,
        )
    except Exception as e:
        logger.debug("SelfCalibrating: %s", e)
        return CalibratedThresholds()

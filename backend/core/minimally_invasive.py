"""MinimallyInvasive — §INCREMENTAL #8.

Starte mit Passthrough, füge Phasen nur hinzu wenn ΔMOS > Schwelle.
Minimaler Eingriff, maximaler Erhalt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MIN_DELTA_THRESHOLD: float = 0.03


@dataclass
class MinimalPlan:
    strategies: list[str] = field(default_factory=list)
    expected_delta: float = 0.0
    total_phases: int = 0


def compute_plan(material: str, era: int = 0, mode: str = "restoration") -> MinimalPlan:
    """Erstellt Minimal-Eingriffs-Plan: nur Strategien die sicher helfen."""
    try:
        from backend.core.phase_impact_predictor import get_phase_impact_predictor

        pred = get_phase_impact_predictor()

        all_strategies = ["passthrough", "light", "balanced", "deep", "full"]
        selected = ["passthrough"]
        total_delta = 0.0

        for strat in all_strategies[1:]:  # Starte nach passthrough
            p = pred.predict(material=material, era=era, phase_id=strat, mode=mode)
            if p.predicted_delta > MIN_DELTA_THRESHOLD and p.confidence > 0.4:
                selected.append(strat)
                total_delta += p.predicted_delta

        return MinimalPlan(
            strategies=selected,
            expected_delta=round(total_delta, 3),
            total_phases=len(selected) - 1,  # Minus passthrough
        )
    except Exception as e:
        logger.debug("MinimallyInvasive: %s", e)
        return MinimalPlan(strategies=["passthrough"])

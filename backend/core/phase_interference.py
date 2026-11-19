"""PhaseInterferenceDetector — §INCREMENTAL #7.

Erkennt: Manche Phasen-Kombinationen interagieren schlecht.
Lernt aus PhaseImpactRecorder: „Phase 04+19 = −0.3 MOS".
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InterferenceReport:
    problematic_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    n_checked: int = 0


def detect(material: str = "", era: int = 0) -> InterferenceReport:
    """Erkennt Phasen-Paare mit negativem Interaktionseffekt."""
    try:
        from backend.core.phase_impact_recorder import get_phase_impact_recorder

        rec = get_phase_impact_recorder()
        impacts = rec._session_impacts

        # Gruppiere nach Phase-Paaren (gleicher Song)
        pairs = defaultdict(list)
        for imp in impacts:
            key = (imp.phase_id, imp.material, imp.era)
            pairs[key].append(imp.quality_delta)

        problematic = []
        for (p1, mat, era_val), deltas in pairs.items():
            if len(deltas) >= 3:
                avg = sum(deltas) / len(deltas)
                if avg < -0.1:
                    # Checke: welche andere Phase lief im gleichen Kontext?
                    for (p2, mat2, era2), d2 in pairs.items():
                        if p1 != p2 and mat == mat2 and era_val == era2:
                            avg2 = sum(d2) / len(d2)
                            if avg2 > 0 and abs(avg + avg2) < 0.05:
                                problematic.append((p1, p2, avg))

        return InterferenceReport(
            problematic_pairs=list(set(problematic))[:20],
            n_checked=len(pairs),
        )
    except Exception as e:
        logger.debug("PhaseInterference: %s", e)
        return InterferenceReport()

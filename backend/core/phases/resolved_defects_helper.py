"""
§v10.18 resolved_defects helper — konsistente Berechnung residualer Defekt-Severities.

Jede Phase, die einen Defekt behebt, ruft nach der Reparatur:

    resolved = compute_resolved_defects(
        defect_type="CLICKS",
        original_severity=defect_severity_map.get("CLICKS", 0.0),
        reduction_ratio=0.95,  # 95% der Klicks entfernt
    )
    return create_phase_result(..., resolved_defects=resolved)

Die residuale Severity wird als `min(original * (1 - reduction_ratio), 0.5)`
berechnet, wobei 0.3 das Maximum für "teilweise behoben" ist.
"""

from __future__ import annotations

import numpy as np


def compute_resolved_defects(
    defect_type: str,
    original_severity: float,
    reduction_ratio: float,
    *,
    max_residual: float = 0.3,
    min_original: float = 0.01,
) -> dict[str, float]:
    """Berechnet {defect_type: residual_severity} nach Reparatur.

    Args:
        defect_type:       DefectType-Name (z.B. "CLICKS", "HIGH_FREQ_NOISE")
        original_severity: Severity AUS DER PRE-ANALYSE (0.0–1.0)
        reduction_ratio:   Wie viel wurde behoben? (0.0 = nichts, 1.0 = alles)
        max_residual:      Maximale residuale Severity (Default 0.3)
        min_original:      Minimale Original-Severity, unter der nichts gemeldet wird

    Returns:
        {defect_type: residual_severity} oder leeres Dict wenn original zu gering
    """
    if original_severity < min_original:
        return {}

    residual = float(np.clip(original_severity * max(0.0, 1.0 - reduction_ratio), 0.0, max_residual))
    return {defect_type: residual}


def compute_resolved_defects_multi(
    defect_map: dict[str, tuple[float, float]],
    *,
    max_residual: float = 0.3,
    min_original: float = 0.01,
) -> dict[str, float]:
    """Berechnet resolved_defects für mehrere Defekte gleichzeitig.

    Args:
        defect_map: {defect_type: (original_severity, reduction_ratio)}
    """
    result: dict[str, float] = {}
    for defect_type, (original_severity, reduction_ratio) in defect_map.items():
        resolved = compute_resolved_defects(
            defect_type,
            original_severity,
            reduction_ratio,
            max_residual=max_residual,
            min_original=min_original,
        )
        result.update(resolved)
    return result

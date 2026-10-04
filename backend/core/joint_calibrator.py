"""§2.70 Joint-Calibration — defect-evidence first, goal-aware for enhancement.

Reparaturphasen folgen ihren gemessenen Defekt-Posteriors. Enhancement-Phasen
werden nur bei passenden Musical-Goal-Gaps aktiviert; deren Beitrag wird auf
die jeweilige Phase normalisiert.

Keine phase-spezifischen Magic-Numbers. Kein `if pid == "phase_X"`.
Der Denker entscheidet für JEDEN Song individuell.
"""

from __future__ import annotations

import numpy as np

# Goal-Gewichte: Welche Goals sind perceptuell am wichtigsten?
GOAL_WEIGHTS: dict[str, float] = {
    "natuerlichkeit": 1.2,
    "authentizitaet": 1.1,
    "transparenz": 1.0,
    "waerme": 1.0,
    "artikulation": 0.9,
    "groove": 0.9,
    "emotionalitaet": 0.8,
    "brillanz": 0.8,
    "tonal_center": 0.7,
    "micro_dynamics": 0.7,
    "transient_energie": 0.7,
    "bass_kraft": 0.6,
    "separation_fidelity": 0.6,
    "spatial_depth": 0.5,
    "timbre_authentizitaet": 0.5,
}


def joint_calibrate(
    phase_ids: list[str],
    goal_proxies: dict[str, float],
    goal_targets: dict[str, float],
    *,
    material: str = "vinyl",
    panns_singing: float = 0.0,
    codec_avg_discount: float = 1.0,
    terminal_codec: str | None = None,
    min_strength: float | None = None,
    restorability_score: float | None = None,
    default_strength: float = 0.85,
    transfer_chain_depth: int | None = None,
    defect_scores: dict | None = None,
) -> dict[str, float]:
    """Berechnet Reparaturstärken aus Defekt-Posteriors und Enhancement aus Goal-Gaps.

    Keine hartcodierten Phasen-Regeln. Jede Entscheidung ist aus den
    Daten ableitbar und im Log nachvollziehbar.

    Args:
        phase_ids: Ausgewählte Phasen
        goal_proxies: Aktuelle Goal-Proxies
        goal_targets: Zielwerte pro Goal
        material: Trägermedium (nur zur Anwendbarkeit)
        panns_singing: PANNs Singing-Konfidenz
        codec_avg_discount: ∅ Diskont-Faktor aus Codec-Kette
        terminal_codec: Terminal-Codec-Typ oder None
        min_strength: Legacy-Parameter; fixed strength floors are ignored (§G188).
        restorability_score: Legacy context; it does not scale measured defect depth.
        default_strength: Legacy default; phases without evidence return zero.
        defect_scores: DefectScanner scores, used as severity × detector confidence.

    Returns:
        {phase_id: calibrated_strength}
    """
    from backend.core.defect_phase_mapper import get_phase_defect_severity, get_reverse_phase_map
    from backend.core.phase_effect_catalog import PHASE_EFFECT_CATALOG

    # Legacy priors remain accepted for API compatibility but cannot cap or
    # lift measured correction strength (§G188–§G189 (GEBOTE.md)).
    _ = (
        min_strength,
        restorability_score,
        default_strength,
        transfer_chain_depth,
        panns_singing,
        codec_avg_discount,
        terminal_codec,
    )
    defect_scores = defect_scores if isinstance(defect_scores, dict) else {}
    reverse_phase_map = get_reverse_phase_map()

    # ── 1. Goal-Gaps ───────────────────────────────────────────
    gaps: dict[str, float] = {}
    for goal, target in goal_targets.items():
        current = float(goal_proxies.get(goal, target))
        gap = target - current
        if gap > 0.001:
            gaps[goal] = gap

    # ── 2. Per-Phase evidence + goal applicability ─────────────
    results: dict[str, float] = {}
    for pid in phase_ids:
        profile = PHASE_EFFECT_CATALOG.get(pid)
        if profile is None or not hasattr(profile, "goal_impact"):
            results[pid] = 0.0
            continue

        utility = 0.0
        applicable_impact = 0.0
        for goal, impact in profile.goal_impact.items():
            gap = gaps.get(goal, 0.0)
            weight = GOAL_WEIGHTS.get(goal, 0.7)
            weighted_impact = abs(float(impact)) * weight
            if gap > 0.0 and weighted_impact > 0.0:
                utility += weighted_impact * gap
                applicable_impact += weighted_impact

        if material in getattr(profile, "unsupported_materials", frozenset()):
            results[pid] = 0.0
            continue

        if pid in reverse_phase_map:
            # Defect phases use measured defect depth × detector confidence.
            strength = get_phase_defect_severity(pid, defect_scores)
        else:
            # Normalize by the applicable phase impacts so coefficients do not
            # become arbitrary strength multipliers. No measured gap means 0.
            strength = utility / applicable_impact if applicable_impact > 0.0 else 0.0
        results[pid] = float(np.clip(strength, 0.0, 1.0))

    return results

"""Spec-Upgrade Gate (§2.81 / §MG-UPG).

Dieses Modul entscheidet, ob eine bessere Implementierung als
normativer Spec-Upgrade-Kandidat promoted werden darf.

Invarianten:
- Kein Upgrade ohne Safety-Pass (artifact_freedom >= 0.95)
- Kein Upgrade bei deutlicher Goal-Regression
- Bei Vokal-Material (panns_singing >= 0.35): VQI darf nicht sinken
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.core.song_goal_importance import ALL_GOAL_NAMES

_EPS = 1e-9


@dataclass(frozen=True)
class SpecUpgradeDecision:
    """Ergebnis der Spec-Upgrade-Pruefung."""

    promoted: bool
    improved_goals_count: int
    non_degraded_goals_count: int
    degraded_goals_count: int
    safety_ok: bool
    vqi_ok: bool
    reason: str


def evaluate_spec_upgrade_candidate(
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
    *,
    artifact_freedom: float,
    panns_singing: float = 0.0,
    vqi_before: float | None = None,
    vqi_after: float | None = None,
) -> SpecUpgradeDecision:
    """Bewertet, ob ein Kandidat als Spec-Upgrade promoted werden darf.

    Kriterien gemaess §2.81 / §MG-UPG:
    - improved_goals_count >= 1
    - non_degraded_goals_count >= 14
    - artifact_freedom >= 0.95
    - falls panns_singing >= 0.35: vqi_after >= vqi_before
    """
    improved = 0
    non_degraded = 0
    degraded = 0

    for goal in ALL_GOAL_NAMES:
        before = float(baseline_scores.get(goal, 0.0))
        after = float(candidate_scores.get(goal, 0.0))
        delta = after - before
        if delta > _EPS:
            improved += 1
            non_degraded += 1
        elif delta >= -_EPS:
            non_degraded += 1
        else:
            degraded += 1

    safety_ok = float(artifact_freedom) >= 0.95

    vqi_ok = True
    if float(panns_singing) >= 0.35:
        if vqi_before is None or vqi_after is None:
            vqi_ok = False
        else:
            vqi_ok = float(vqi_after) + _EPS >= float(vqi_before)

    promoted = (improved >= 1) and (non_degraded >= 14) and safety_ok and vqi_ok

    if not safety_ok:
        reason = "safety_fail_artifact_freedom"
    elif not vqi_ok:
        reason = "vqi_regression_or_missing"
    elif non_degraded < 14:
        reason = "too_many_goal_regressions"
    elif improved < 1:
        reason = "no_goal_improvement"
    else:
        reason = "promote_to_spec"

    return SpecUpgradeDecision(
        promoted=promoted,
        improved_goals_count=improved,
        non_degraded_goals_count=non_degraded,
        degraded_goals_count=degraded,
        safety_ok=safety_ok,
        vqi_ok=vqi_ok,
        reason=reason,
    )

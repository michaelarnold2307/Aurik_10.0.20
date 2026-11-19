"""
backend/core/feedback_integrator.py — Integrates expert and community feedback
===============================================================================

Blends expert and community aggregates with equal weight (0.5/0.5).
The test asserts exactalgorithmically:
    integrated["brillanz"] == 0.5 * expert["brillanz"] + 0.5 * community["brillanz"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntegratedFeedback:
    """Typed integrated feedback payload."""

    scores: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.scores.get(key, default))


class FeedbackIntegrator:
    """Mischt expert and community feedback aggregates.

    Parameters
    ----------
    expert_system:
        An ``ExpertFeedbackSystem`` instance.
    community_platform:
        A ``CommunityRatingPlatform`` instance.
    expert_weight:
        Weight for expert scores (community weight = 1 - expert_weight).
    """

    def __init__(
        self,
        expert_system: Any,
        community_platform: Any,
        expert_weight: float = 0.5,
    ) -> None:
        self.expert_system = expert_system
        self.community_platform = community_platform
        self.expert_weight = expert_weight

    def integrate(self) -> IntegratedFeedback:
        """Gibt a weighted blend of expert and community aggregates zurück."""
        expert_agg = self.expert_system.aggregate()
        community_agg = self.community_platform.aggregate()

        all_keys = set(expert_agg.scores) | set(community_agg.scores)
        result: dict[str, float] = {}
        for k in all_keys:
            e = expert_agg.get(k, 0.0)
            c = community_agg.get(k, 0.0)
            result[k] = self.expert_weight * e + (1.0 - self.expert_weight) * c
        return IntegratedFeedback(result)


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_feedback_integrator_instance = None
_feedback_integrator_lock = _threading.Lock()


def get_feedback_integrator(
    expert_system=None,
    community_platform=None,
) -> FeedbackIntegrator:
    """Gibt the process-wide singleton ``FeedbackIntegrator`` instance zurück."""
    global _feedback_integrator_instance
    if _feedback_integrator_instance is None:
        with _feedback_integrator_lock:
            if _feedback_integrator_instance is None:
                from backend.core.community_rating_platform import CommunityRatingPlatform
                from backend.core.expert_feedback_system import ExpertFeedbackSystem

                es = expert_system or ExpertFeedbackSystem()
                cp = community_platform or CommunityRatingPlatform()
                _feedback_integrator_instance = FeedbackIntegrator(es, cp)
    return _feedback_integrator_instance

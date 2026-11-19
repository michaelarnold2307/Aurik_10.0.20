from __future__ import annotations

"""Normative CI gate for performance budget invariants.

This test module enforces stable, machine-checkable constraints for
performance budget configuration without running long ML-heavy pipelines.
"""


import pytest

from backend.core.performance_guard import PerformanceGuard, QualityMode
from backend.core.unified_restorer_v3 import RestorationConfig


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_performance_guard_budget_constants_are_within_policy_bounds() -> None:
    """Performance constants must stay within release policy bounds.

    Spec §2.38 KMV: LIMIT_FAST = 8.0 for real-audio proof/headroom,
    LIMIT_BALANCED/QUALITY/MAXIMUM = 32.0, RT8_EXCELLENCE_BUDGET = 32.0.
    Old 3× FAST caused noisy real-audio validation warnings despite successful
    bounded processing.
    """
    assert PerformanceGuard.LIMIT_FAST <= 8.0
    assert PerformanceGuard.LIMIT_BALANCED <= 32.0
    assert PerformanceGuard.LIMIT_3X_RT <= 32.0
    assert PerformanceGuard.RT8_EXCELLENCE_BUDGET <= 32.0


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_performance_guard_target_mapping_is_consistent() -> None:
    """Target RT mapping per quality mode must remain deterministic."""
    fast_guard = PerformanceGuard(mode=QualityMode.FAST, enforce_limit=True, enable_adaptive_skipping=True)
    balanced_guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True, enable_adaptive_skipping=True)
    quality_guard = PerformanceGuard(mode=QualityMode.QUALITY, enforce_limit=True, enable_adaptive_skipping=True)

    assert fast_guard.target_rt_factor <= 8.0
    assert balanced_guard.target_rt_factor <= 32.0
    assert quality_guard.target_rt_factor >= balanced_guard.target_rt_factor


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_musical_excellence_phases_are_never_skippable_by_priority() -> None:
    """All musical excellence phases must keep priority 9+ (effectively non-skippable)."""
    priorities = PerformanceGuard.PHASE_PRIORITIES

    missing = [p for p in PerformanceGuard.MUSICAL_EXCELLENCE_PHASES if p not in priorities]
    assert not missing, f"Missing musical excellence priorities: {missing}"

    downgraded = [p for p in PerformanceGuard.MUSICAL_EXCELLENCE_PHASES if priorities[p] < 9]
    assert not downgraded, f"Musical excellence phases downgraded below priority 9: {downgraded}"


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_restoration_config_defaults_keep_rt_enforcement_disabled() -> None:
    """Default config: RT enforcement opt-in only; standard paths use no_rt_limit=True + UI watchdog."""
    cfg = RestorationConfig()
    assert cfg.enable_performance_guard is True
    assert cfg.enforce_3x_rt is False
    assert cfg.enable_adaptive_skipping is False

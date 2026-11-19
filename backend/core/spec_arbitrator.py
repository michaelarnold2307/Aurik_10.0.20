"""
spec_arbitrator.py — v10 Spec-Arbitrator: Automatic Spec Upgrades
====================================================================

Closes the Continuous Improvement loop: when code consistently exceeds
spec thresholds, the Arbitrator proposes and applies spec upgrades.

Three stages:
  1. EVALUATE: Compare this run's metrics against current spec thresholds
  2. DECIDE: If 5+ consecutive runs exceed a spec by >5%, propose upgrade
  3. ACT: Generate spec upgrade recommendation (human- or auto-applied)

Integration:
  from backend.core.spec_arbitrator import get_arbitrator
  arb = get_arbitrator()
  upgrade = arb.evaluate(comparison_result)
  if upgrade.is_ready():
      print(upgrade.recommendation)

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
CONSECUTIVE_RUNS_FOR_UPGRADE: int = 5
MIN_EXCEED_PCT: float = 0.05  # 5% over spec target
MIN_CONFIDENCE_FOR_AUTO: float = 0.85

# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class SpecUpgradeProposal:
    """A concrete proposal to raise a spec threshold."""

    metric: str
    current_spec: float
    proposed_spec: float
    avg_achieved: float
    consecutive_runs: int
    confidence: float  # 0-1
    recommendation: str

    def is_ready(self) -> bool:
        return self.consecutive_runs >= CONSECUTIVE_RUNS_FOR_UPGRADE

    def is_auto_apply(self) -> bool:
        return self.is_ready() and self.confidence >= MIN_CONFIDENCE_FOR_AUTO


@dataclass
class ArbitratorReport:
    """Full arbitrator output after evaluating a pipeline run."""

    proposals: list[SpecUpgradeProposal] = field(default_factory=list)
    ready_upgrades: list[SpecUpgradeProposal] = field(default_factory=list)
    auto_upgrades: list[SpecUpgradeProposal] = field(default_factory=list)
    total_runs: int = 0
    summary: str = ""


# ── SpecArbitrator ──────────────────────────────────────────────────────


class SpecArbitrator:
    """Monitors code-vs-spec over multiple runs and proposes spec upgrades."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, list[float]] = {}  # metric -> [values]
        self._consecutive_counts: dict[str, int] = {}  # metric -> consecutive exceeds
        self._run_count: int = 0

    def evaluate(
        self,
        comparison: Any,  # SpecComparisonResult
        *,
        material: str = "unknown",
    ) -> ArbitratorReport:
        """Evaluate one pipeline run and return upgrade proposals.

        Args:
            comparison: Result from spec_improvement_loop.compare_spec_vs_code()
            material: Material type for context

        Returns:
            ArbitratorReport with ready and auto-upgrade proposals
        """
        report = ArbitratorReport()

        with self._lock:
            self._run_count += 1
            report.total_runs = self._run_count

            for metric in getattr(comparison, "metrics", []):
                name = getattr(metric, "name", "")
                achieved = getattr(metric, "code_achieved", 0.0)
                target = getattr(metric, "spec_target", 0.0)
                exceeds = getattr(metric, "exceeds", False)

                if name not in self._history:
                    self._history[name] = []
                    self._consecutive_counts[name] = 0

                self._history[name].append(achieved)
                if len(self._history[name]) > CONSECUTIVE_RUNS_FOR_UPGRADE * 2:
                    self._history[name] = self._history[name][-CONSECUTIVE_RUNS_FOR_UPGRADE:]

                if exceeds and achieved > target * (1.0 + MIN_EXCEED_PCT):
                    self._consecutive_counts[name] += 1
                else:
                    self._consecutive_counts[name] = 0

                # Check if ready for upgrade
                if self._consecutive_counts[name] >= CONSECUTIVE_RUNS_FOR_UPGRADE:
                    recent = self._history[name][-CONSECUTIVE_RUNS_FOR_UPGRADE:]
                    if len(recent) >= CONSECUTIVE_RUNS_FOR_UPGRADE:
                        avg = sum(recent) / len(recent)
                        proposed = avg * 0.95  # 5% safety margin below average
                        proposed = max(proposed, target + 0.02)  # minimum 0.02 increase

                        confidence = min(1.0, 0.5 + self._consecutive_counts[name] * 0.05)

                        proposal = SpecUpgradeProposal(
                            metric=name,
                            current_spec=target,
                            proposed_spec=round(proposed, 4),
                            avg_achieved=round(avg, 4),
                            consecutive_runs=self._consecutive_counts[name],
                            confidence=round(confidence, 3),
                            recommendation=(
                                f"UPGRADE {name}: {target:.3f} -> {proposed:.3f} "
                                f"(Code achieves ~{avg:.3f} over {self._consecutive_counts[name]} runs, "
                                f"confidence={confidence:.0%})"
                            ),
                        )
                        report.proposals.append(proposal)

                        if proposal.is_ready():
                            report.ready_upgrades.append(proposal)
                        if proposal.is_auto_apply():
                            report.auto_upgrades.append(proposal)

        # Summary
        if report.auto_upgrades:
            metrics = [p.metric for p in report.auto_upgrades]
            report.summary = f"AUTO-UPGRADE: {len(report.auto_upgrades)} specs ready: {', '.join(metrics)}"
        elif report.ready_upgrades:
            metrics = [p.metric for p in report.ready_upgrades]
            report.summary = f"UPGRADE-READY: {len(report.ready_upgrades)} specs: {', '.join(metrics)}"
        elif report.proposals:
            report.summary = f"PROPOSAL: {len(report.proposals)} upgrades pending"
        else:
            report.summary = f"NO-UPGRADE: all metrics within spec after {self._run_count} runs"

        return report

    def generate_spec_patch(self, proposal: SpecUpgradeProposal) -> str:
        """Generate a human-readable spec upgrade recommendation."""
        return (
            f"# Spec Upgrade: {proposal.metric}\n"
            f"# Current: {proposal.current_spec:.3f}\n"
            f"# Proposed: {proposal.proposed_spec:.3f}\n"
            f"# Evidence: {proposal.consecutive_runs} runs, avg={proposal.avg_achieved:.3f}\n"
            f"# Confidence: {proposal.confidence:.0%}\n"
            f"# {proposal.recommendation}\n"
        )

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._consecutive_counts.clear()
            self._run_count = 0

    @property
    def run_count(self) -> int:
        return self._run_count


# ── Singleton ───────────────────────────────────────────────────────────
_arbitrator: SpecArbitrator | None = None
_arb_lock = threading.Lock()


def get_arbitrator() -> SpecArbitrator:
    global _arbitrator
    with _arb_lock:
        if _arbitrator is None:
            _arbitrator = SpecArbitrator()
    return _arbitrator

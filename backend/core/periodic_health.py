"""
§2.59 Periodic Health Report (2026-07-09)

Sammelt über Batch-Runs hinweg Pipeline-Gesundheitsmetriken und
gibt alle N Runs einen Snapshot aus. Ermöglicht Trend-Erkennung
bevor Degradation hörbar wird.

Wird von UV3.restore() pro Run aufgerufen.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunHealth:
    """Gesundheits-Snapshot eines einzelnen Runs."""

    run_id: int = 0
    timestamp: float = 0.0
    duration_s: float = 0.0
    contract_ok: bool = True
    silent_excepts: int = 0
    phases_executed: int = 0
    global_scalar: float = 1.0
    artifact_freedom: float = 1.0
    ram_gb: float = 0.0
    material: str = "unknown"


class HealthReportCollector:
    """Thread-sicherer Collector für Run-Health-Metriken.

    Usage:
        collector = get_health_collector()
        collector.record_run(RunHealth(...))
    """

    def __init__(self, report_interval: int = 50) -> None:
        self._lock = threading.Lock()
        self._runs: list[RunHealth] = []
        self._report_interval = report_interval
        self._total_runs = 0
        self._start_time = time.monotonic()

    def record_run(self, health: RunHealth) -> None:
        """Zeichnet einen Run auf und gibt ggf. Report aus."""
        with self._lock:
            self._total_runs += 1
            health.run_id = self._total_runs
            health.timestamp = time.time()
            self._runs.append(health)

            # Nur die letzten N Runs behalten
            if len(self._runs) > self._report_interval * 2:
                self._runs = self._runs[-self._report_interval :]

            # Report alle N Runs
            if self._total_runs % self._report_interval == 0:
                self._emit_report()

    def _emit_report(self) -> None:
        """Gibt den periodischen Gesundheitsbericht aus."""
        if not self._runs:
            return

        recent = self._runs[-self._report_interval :]
        n = len(recent)

        contracts_ok = sum(1 for r in recent if r.contract_ok)
        total_excepts = sum(r.silent_excepts for r in recent)
        phases_avg = sum(r.phases_executed for r in recent) / max(n, 1)
        scalar_avg = sum(r.global_scalar for r in recent) / max(n, 1)
        artifact_avg = sum(r.artifact_freedom for r in recent) / max(n, 1)
        ram_avg = sum(r.ram_gb for r in recent) / max(n, 1)
        dur_avg = sum(r.duration_s for r in recent) / max(n, 1)

        uptime_h = (time.monotonic() - self._start_time) / 3600.0

        logger.info(
            "📊 PeriodicHealth #%d (last %d runs, uptime %.1fh): "
            "ContractOK=%d/%d | SilentExcepts=%d | "
            "PhasesAvg=%.1f | GlobalScalarAvg=%.3f | "
            "ArtifactFreedomAvg=%.3f | RAMAvg=%.1fGB | DurAvg=%.0fs",
            self._total_runs,
            n,
            uptime_h,
            contracts_ok,
            n,
            total_excepts,
            phases_avg,
            scalar_avg,
            artifact_avg,
            ram_avg,
            dur_avg,
        )

        # Warnung bei Trends
        if artifact_avg < 0.90:
            logger.warning(
                "📊 Health: ArtifactFreedom Ø=%.3f < 0.90 — mögliche Qualitäts-Degradation über die letzten %d Runs",
                artifact_avg,
                n,
            )
        if total_excepts > 0:
            logger.warning(
                "📊 Health: %d silent excepts in %d Runs — Fehler werden unterdrückt, siehe debug-Log",
                total_excepts,
                n,
            )

        # §2.59 Trend-Erkennung: Degradation über halbe Report-Periode
        if len(self._runs) >= self._report_interval:
            older = self._runs[-self._report_interval : -self._report_interval // 2]
            newer = self._runs[-self._report_interval // 2 :]
            if older and newer:
                old_artifact = sum(r.artifact_freedom for r in older) / len(older)
                new_artifact = sum(r.artifact_freedom for r in newer) / len(newer)
                if new_artifact < old_artifact - 0.05:
                    logger.warning(
                        "📊 Health TREND: ArtifactFreedom sinkt — %.3f → %.3f (Δ=%.3f über %d Runs)",
                        old_artifact,
                        new_artifact,
                        old_artifact - new_artifact,
                        self._report_interval // 2,
                    )
                old_scalar = sum(r.global_scalar for r in older) / len(older)
                new_scalar = sum(r.global_scalar for r in newer) / len(newer)
                if new_scalar < old_scalar - 0.03:
                    logger.warning(
                        "📊 Health TREND: GlobalScalar sinkt — %.3f → %.3f (Δ=%.3f)",
                        old_scalar,
                        new_scalar,
                        old_scalar - new_scalar,
                    )

    def get_summary(self) -> dict[str, Any]:
        """Gibt eine Zusammenfassung für den aktuellen Run-Kontext."""
        with self._lock:
            if not self._runs:
                return {"runs": 0}
            last = self._runs[-1]
            return {
                "runs": self._total_runs,
                "last_contract_ok": last.contract_ok,
                "last_silent_excepts": last.silent_excepts,
                "last_phases": last.phases_executed,
                "last_global_scalar": last.global_scalar,
            }


# ── Singleton ────────────────────────────────────────────────────────────────

_HEALTH_COLLECTOR: HealthReportCollector | None = None
_LOCK = threading.Lock()


def get_health_collector(report_interval: int = 50) -> HealthReportCollector:
    """Thread-sicherer Singleton-Accessor."""
    global _HEALTH_COLLECTOR
    with _LOCK:
        if _HEALTH_COLLECTOR is None:
            _HEALTH_COLLECTOR = HealthReportCollector(report_interval=report_interval)
    return _HEALTH_COLLECTOR

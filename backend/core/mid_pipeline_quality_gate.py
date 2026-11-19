"""
backend/core/mid_pipeline_quality_gate.py — HPE Quality Gate (§v10.10)
=======================================================================

Prüft alle N Phasen den HPE-Score (Harmonic-Preservation-Energy).
Bei Verschlechterung: Selbstkalibrierung passt nachfolgende Phasen-Strengths an.

Synergie Preset × Selbstkalibrierung:
    Preset = Start-Schwellwert (z.B. -5% HPE-Toleranz)
    Selbstkalibrierung = Dynamische Anpassung basierend auf tatsächlichem HPE-Delta

Usage:
    from backend.core.mid_pipeline_quality_gate import HpeQualityGate
    gate = HpeQualityGate(interval=8, hpe_drop_tolerance_pct=5.0)
    decision = gate.check(phase_idx, hpe_before, hpe_after)
    if decision.calibrate:
        kwargs["strength"] *= decision.strength_scalar
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HpeGateDecision:
    """Ergebnis einer HPE-Gate-Prüfung."""

    phase_id: str
    phase_idx: int
    hpe_before: float
    hpe_after: float
    hpe_delta_pct: float
    passed: bool
    severity: str = "ok"  # ok | warn | critical
    calibrate: bool = False  # Selbstkalibrierung aktiv?
    strength_scalar: float = 1.0  # Multiplikator für Folgephasen-Strength
    recommendation: str = ""


@dataclass
class HpeGateConfig:
    """Konfiguration des HPE Quality Gates — Preset-Seite."""

    check_interval: int = 8  # Alle N Phasen prüfen
    hpe_drop_warn_pct: float = 3.0  # WARNING ab -3%
    hpe_drop_critical_pct: float = 7.0  # CRITICAL ab -7%
    max_strength_scalar: float = 0.60  # Maximale Strength-Reduktion bei Kalibrierung
    min_consecutive_pass: int = 2  # Erst nach N aufeinanderfolgenden OKs Kalibrierung lösen
    learning_rate: float = 0.15  # Wie stark Selbstkalibrierung von Ist-Werten lernt


class HpeQualityGate:
    """Mid-Pipeline HPE-Wächter mit Selbstkalibrierung.

    Selbstkalibrierung:
        Wenn HPE fällt, werden Folgephasen proportional gedämpft.
        Wenn HPE stabil bleibt, wird die Dämpfung langsam zurückgenommen (Learning).
        So kalibriert sich die Pipeline automatisch auf den optimalen Arbeitspunkt.

    Preset-Integration:
        Das Preset definiert die initiale Toleranz (hpe_drop_warn_pct).
        Die Selbstkalibrierung passt dynamisch an — lernt vom tatsächlichen Verhalten.
    """

    def __init__(self, config: HpeGateConfig | None = None) -> None:
        self._cfg = config or HpeGateConfig()
        self._hpe_history: list[tuple[int, float, float]] = []  # (idx, before, after)
        self._consecutive_ok: int = 0
        self._cumulative_drop_pct: float = 0.0
        self._active_calibration: float = 1.0  # 1.0 = volle Strength, <1.0 = gedämpft

    def check(
        self,
        phase_id: str,
        phase_idx: int,
        hpe_before: float,
        hpe_after: float,
    ) -> HpeGateDecision:
        """Prüft HPE-Delta und gibt Kalibrierungsentscheidung zurück.

        Args:
            phase_id: Phase-ID (z.B. 'phase_07_harmonic_restoration').
            phase_idx: Index in der Pipeline (0-basiert).
            hpe_before: HPE-Score VOR der Phase.
            hpe_after: HPE-Score NACH der Phase.

        Returns:
            HpeGateDecision mit Pass/Fail und Kalibrierungs-Skalar.
        """
        _delta = hpe_after - hpe_before
        _delta_pct = (_delta / max(hpe_before, 0.01)) * 100.0

        self._hpe_history.append((phase_idx, hpe_before, hpe_after))

        # Nur an Check-Intervallen prüfen
        if phase_idx % self._cfg.check_interval != 0:
            return HpeGateDecision(
                phase_id=phase_id,
                phase_idx=phase_idx,
                hpe_before=hpe_before,
                hpe_after=hpe_after,
                hpe_delta_pct=_delta_pct,
                passed=True,
            )

        # Prüfung
        _passed = _delta_pct > -self._cfg.hpe_drop_warn_pct
        _severity = "ok"
        _calibrate = False
        _scalar = 1.0
        _rec = ""

        if _delta_pct <= -self._cfg.hpe_drop_critical_pct:
            _severity = "critical"
            _calibrate = True
            _cumulative = abs(_delta_pct) / 100.0
            _scalar = float(np.clip(1.0 - _cumulative * 0.8, self._cfg.max_strength_scalar, 1.0))
            _rec = f"HPE critical drop ({_delta_pct:+.1f}%) — Selbstkalibrierung: Folgephasen-Strength ×{_scalar:.2f}"
            self._consecutive_ok = 0
        elif _delta_pct <= -self._cfg.hpe_drop_warn_pct:
            _severity = "warn"
            _calibrate = True
            _scalar = float(np.clip(1.0 - abs(_delta_pct) / 200.0, 0.80, 1.0))
            _rec = f"HPE warning ({_delta_pct:+.1f}%) — leichte Selbstkalibrierung: ×{_scalar:.2f}"
            self._consecutive_ok = 0
        else:
            self._consecutive_ok += 1
            # Selbstkalibrierung lernt: nach N aufeinanderfolgenden OKs langsam zurück zur vollen Strength
            if self._consecutive_ok >= self._cfg.min_consecutive_pass:
                self._active_calibration = float(
                    np.clip(
                        self._active_calibration + self._cfg.learning_rate,
                        0.0,
                        1.0,
                    )
                )
                _scalar = self._active_calibration
                if _scalar < 1.0:
                    _rec = f"HPE stabil — Selbstkalibrierung Recovery → ×{_scalar:.2f}"

        if _calibrate:
            self._active_calibration = _scalar
            self._cumulative_drop_pct += abs(_delta_pct)

        _log_fn = logger.error if _severity == "critical" else logger.warning if _severity == "warn" else logger.info
        _log_fn(
            "🛡️ HPE-Gate %s [%d]: %.4f→%.4f (Δ=%+.1f%%) | %s | scalar=%.2f | cum_drop=%.1f%%",
            phase_id,
            phase_idx,
            hpe_before,
            hpe_after,
            _delta_pct,
            _severity,
            _scalar,
            self._cumulative_drop_pct,
        )

        return HpeGateDecision(
            phase_id=phase_id,
            phase_idx=phase_idx,
            hpe_before=hpe_before,
            hpe_after=hpe_after,
            hpe_delta_pct=_delta_pct,
            passed=_passed,
            severity=_severity,
            calibrate=_calibrate,
            strength_scalar=_scalar,
            recommendation=_rec,
        )

    def get_preset_snapshot(self) -> dict:
        """Exportiert aktuellen Kalibrierungsstand als Preset — für Preset-Learning."""
        return {
            "cumulative_drop_pct": round(self._cumulative_drop_pct, 2),
            "active_calibration": round(self._active_calibration, 3),
            "consecutive_ok": self._consecutive_ok,
            "phase_count": len(self._hpe_history),
            "avg_hpe_delta": round(
                float(np.mean([a - b for _, b, a in self._hpe_history])) if self._hpe_history else 0.0,
                4,
            ),
        }

    @property
    def calibration_active(self) -> bool:
        return self._active_calibration < 0.99

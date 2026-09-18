"""§PERF-R R7 (2026-09-18) — Hör-Impact-Schätzer für die RT-Deferral-Entscheidung.

Der PerformanceGuard entscheidet unter Budgetdruck (RT-Limit), welche Phasen
deferred werden. R7 macht diese Entscheidung hör-bewusst (Hörordnung:
Audibility statt Mess-Null) — ohne die DSP-Parameter je Phase anzutasten:

    impact(phase) = max über alle der Phase zugeordneten DefectTypes
                    (canonical `DefectPhaseMapper._PHASE_MAP`) der
                    defect_score (0..1)

  - impact == 0.0 → die Phase repariert auf DIESEM Audio nichts Hörbares
    (Null-Impact). PSY-A1-gegatete Phasen laufen bei Null-Impact ohnehin als
    Passthrough — ihr Deferral ist eine reine Laufzeit-Ersparnis ohne
    Hör-Änderung.
  - impact  > 0.6 → die Phase repariert deutlich Hörbares — unter Budgetdruck
    SCHÜTZEN; der Skip trifft zuerst Null-Impact-Phasen statt nach fester
    Prioritäts-Reihenfolge die falsche Phase.
  - impact == None → keine Zuordnung oder keine Daten → bisherige
    Fix-Priorität (Verhalten unverändert, bit-identisch).

Deterministisch (§G5 (copilot-instructions.md)): reine Funktion über feste Daten,
keine Zufalls-Tiebreaks, keine versteckte Zeit- oder Reihenfolge-Abhängigkeit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# §PERF-R R7: Schwellen — konservativ gewählt; zwischen den Schwellen bleibt
# die bisherige Fix-Prioritäts-Logik unangetastet.
ZERO_IMPACT_THRESHOLD: float = 0.05
HIGH_IMPACT_THRESHOLD: float = 0.60

_RESTORATION_MODE: str = "restoration"


def _to_defect_type(key: Any) -> Any | None:
    """Normalisiert Score-Schlüssel (str/Enum) auf `DefectType` — ohne Crash."""
    try:
        from backend.core.defect_scanner import DefectType

        if isinstance(key, DefectType):
            return key
        if isinstance(key, str):
            return DefectType(key)  # Enum-Werte sind die Lowercase-Strings
    except (ImportError, ValueError):
        return None
    return None


def estimate_phase_hearing_impact(
    phase_id: str,
    defect_scores: Mapping[Any, float] | None,
    mode: str = _RESTORATION_MODE,
) -> float | None:
    """Erwarteter Hör-Gewinn der Phase auf diesem Audio (0..1) oder None.

    None bedeutet „keine Daten/keine Zuordnung" — der Aufrufer behält dann
    die bisherige Fix-Prioritäts-Logik (bit-identisch zum Status quo).
    """
    if not defect_scores:
        return None
    try:
        from backend.core.defect_phase_mapper import DefectPhaseMapper
    except ImportError:
        logger.debug("R7: DefectPhaseMapper nicht verfügbar — Impact unbekannt")
        return None

    _mapper = DefectPhaseMapper()
    _max_impact = 0.0
    _mapped_any = False
    for _key, _raw in defect_scores.items():
        try:
            _score = float(_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if _score < 0.0:
            continue
        _dt = _to_defect_type(_key)
        if _dt is None:
            continue
        try:
            _phases = _mapper.get_all_phases(_dt, mode=mode)
        except Exception as _map_exc:  # Bug 9 (anti-regression): kein stummer except
            logger.debug("R7: Phasen-Zuordnung für %s nicht verfügbar (%s)", _dt, _map_exc)
            continue
        if phase_id in _phases:
            _mapped_any = True
            _max_impact = max(_max_impact, min(1.0, _score))
    if not _mapped_any:
        return None
    return _max_impact


def classify_hearing_impact(impact: float | None) -> str:
    """Einheitliche R7-Klassifikation: 'zero' | 'high' | 'normal' | 'unknown'."""
    if impact is None:
        return "unknown"
    if impact <= ZERO_IMPACT_THRESHOLD:
        return "zero"
    if impact >= HIGH_IMPACT_THRESHOLD:
        return "high"
    return "normal"

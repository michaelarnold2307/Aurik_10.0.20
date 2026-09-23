"""§v10.303.19 Phase-Effectiveness-Memory: Lernt welche Phasen pro Material wirken.

Speichert PMGG-Deltas pro Phase und Material über mehrere Runs.
Nach 3+ Runs mit |Δ| < 0.001 wird die Phase automatisch geskippt.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# ── In-Memory Cache ───────────────────────────────────────────────────
# {(scope, material): {phase_id: {"runs": N, "abs_delta_sum": S}}}
# scope isoliert Songs voneinander (§V8/§G1 (copilot-instructions.md) Song-Isolation):
# UV3 nutzt einen Instanz-Scope, damit Lernzustand niemals zwischen Songs durchschlägt.
_cache: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
_lock = threading.Lock()


def record_phase_deltas(material: str, deltas: dict[str, float], scope: str = "global") -> None:
    """Zeichnet einen Lauf auf. Absolut-Deltas pro Phase (Multi-Goal-Aggregat)."""
    with _lock:
        _mat_entry = _cache.setdefault((scope, material.lower()), {})
        for _pid, _delta in deltas.items():
            _entry = _mat_entry.setdefault(_pid, {"runs": 0.0, "abs_delta_sum": 0.0})
            _entry["runs"] += 1.0
            _entry["abs_delta_sum"] += abs(float(_delta))
        _n = len(deltas)
        logger.debug(
            "Verarbeitungsschritt-Effectiveness: %d Verarbeitungsschritt(n) für material=%s scope=%s aufgezeichnet",
            _n,
            material,
            scope,
        )


def should_skip_phase(
    material: str,
    phase_id: str,
    scope: str = "global",
    min_runs: int = 3,
    max_avg_abs_delta: float = 0.001,
) -> bool:
    """§v10.303.19: Prüft ob Phase historisch wirkungslos war (im Song-Scope)."""
    with _lock:
        _mat_entry = _cache.get((scope, material.lower()), {})
        _phase_entry = _mat_entry.get(phase_id)
        if _phase_entry is None:
            return False
        _runs = int(_phase_entry["runs"])
        if _runs < min_runs:
            return False
        _avg_abs_delta = _phase_entry["abs_delta_sum"] / _runs
        if _avg_abs_delta < max_avg_abs_delta:
            logger.info(
                "§v10.303.19 Verarbeitungsschritt-Effectiveness: %s auf %s → ueberspringen (Ø|Δ|=%.4f nach %d Runs)",
                phase_id,
                material,
                _avg_abs_delta,
                _runs,
            )
            return True
        return False


def reset_scope(scope: str = "global") -> None:
    """§V8/§G1 (copilot-instructions.md) Song-Isolation: Löscht den Lernzustand eines Songs."""
    with _lock:
        for _key in [k for k in _cache if k[0] == scope]:
            _cache.pop(_key, None)


def get_effectiveness_stats(material: str, scope: str = "global") -> dict[str, dict[str, float]]:
    """Gibt Statistiken für Debug/UI zurück."""
    with _lock:
        return dict(_cache.get((scope, material.lower()), {}))

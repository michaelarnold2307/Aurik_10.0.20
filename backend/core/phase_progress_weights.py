"""D-K3-69 — Dauergewichteter Phasenfortschritt (Selbstkalibrierung je Lauf).

Befund (Tiefenanalyse 2026-10-09): Der an die GUI gemeldete UV3-Fortschritt
(`pct`) war eine **Zähl-Fraktion abgeschlossener Phasen** in festem Band 20–85
(`_idx_ex / _n_sel`). Die realen Phasen-Wand-Zeiten streuen jedoch ca. **11:1**
(Referenzlauf 10.12.25: `phase_12_wow_flutter_fix` 57× vs. `phase_17` 5×
Heartbeat-Anteile; Kalibrierung: phase_12 = 250–440 s) — der Balken sprintete
bei Kurzphasen und fror bei Langphasen ein.

Lösung ohne Fantasiegewichte (§V7 copilot-instructions.md): Die Gewichte sind
die **gemessenen Wand-Zeiten dieses Laufs** (Selbstkalibrierung). Bis eine
Phase gemessen ist, gilt der Median der bereits gemessenen Phasen als Prior
(neutraler Startwert 1,0 s). Rein anzeigend — kein Einfluss auf Audio oder
Entscheidungen; deterministisch je Lauf-Zustand (§G5 copilot-instructions.md
gilt für den Output, nicht für Wand-Zeit-Anzeigen).

Reine Logik ohne Seiteneffekte (headless-testbar).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Untergrenze je Phasengewicht — verhindert Null-/Negativgewichte bei
# instantanen Phasen und hält den Nenner stabil.
_PRIOR_FLOOR_S = 0.5


def _prior_from_measured(durations: Mapping[str, float]) -> float:
    """Median der gemessenen Phasen-Zeiten (Prior für noch unbekannte Phasen)."""
    vals = sorted(float(v) for v in durations.values() if isinstance(v, (int, float)) and float(v) > 0.0)
    if not vals:
        return 1.0
    return max(_PRIOR_FLOOR_S, vals[len(vals) // 2])


def _dedupe(ids: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(p) for p in ids))


def weighted_progress_fraction(
    completed_ids: Iterable[object],
    durations: Mapping[str, float],
    n_sel: int,
) -> float:
    """Dauergewichteter Anteil [0..1] der abgeschlossenen Phasen.

    Gewicht einer abgeschlossenen Phase = gemessene Wand-Zeit (Fallback: Prior);
    verbleibende Phasen zählen mit dem Prior (Median der gemessenen Phasen).
    """
    done = _dedupe(completed_ids)
    prior = _prior_from_measured(durations)
    w_done = sum(max(_PRIOR_FLOOR_S, float(durations.get(pid, prior) or 0.0)) for pid in done)
    n_rem = max(0, int(n_sel) - len(done))
    w_rem = n_rem * prior
    total = w_done + w_rem
    if total <= 0.0:
        return 0.0
    return max(0.0, min(1.0, w_done / total))


def weighted_phase_span(
    completed_ids: Iterable[object],
    durations: Mapping[str, float],
    current_phase_id: object,
    n_sel: int,
) -> tuple[float, float]:
    """Dauergewichtete [Start, Ende]-Spanne (Fraktion 0..1) der AKTUELLEN Phase.

    Sub-Progress-Callbacks der Phase werden in diesen Bereich gemappt; die
    Grenzen stimmen mit `weighted_progress_fraction` an den Phasen-Grenzen
    überein (Start = Fraktion vor der Phase, Ende = Fraktion danach).
    """
    done = _dedupe(completed_ids)
    prior = _prior_from_measured(durations)
    w_done = sum(max(_PRIOR_FLOOR_S, float(durations.get(pid, prior) or 0.0)) for pid in done)
    w_cur = max(_PRIOR_FLOOR_S, float(durations.get(str(current_phase_id), prior) or 0.0))
    n_rem_after = max(0, int(n_sel) - len(done) - 1)
    total = w_done + w_cur + n_rem_after * prior
    if total <= 0.0:
        return 0.0, 1.0
    start = max(0.0, min(1.0, w_done / total))
    end = max(start, min(1.0, (w_done + w_cur) / total))
    return start, end


__all__ = ["weighted_progress_fraction", "weighted_phase_span"]

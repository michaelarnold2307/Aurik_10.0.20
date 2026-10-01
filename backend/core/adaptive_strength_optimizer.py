"""§ADAPTIVE Adaptive Phase Strength Optimizer — §Messen→Kalibrieren→Restaurieren.

Ersetzt das binäre "Phase an/aus"-Denken durch kontinuierliche Stärke-Modulation.
Jede Phase wird bei MINIMALER Stärke gestartet und durch Messung iterativ
optimiert — nicht durch Vorentscheidung eliminiert.

Prinzip:
  1. Starte Phase bei strength = strength_floor (z.B. 0.10)
  2. Messe Qualitäts-Delta (HPI, Crest, AFG)
  3. Wenn Verbesserung → erhöhe Stärke um step (z.B. +0.12)
  4. Wenn Verschlechterung → reduziere Stärke um step
  5. Wenn keine Änderung → konvergiert, nächsthöhere Stärke testen
  6. Erst wenn KEINE Stärke hilft → Phase skippen

Die Stärke-Abstufung ist fein genug (0.05-0.10 Schritte) um das Optimum
zu finden, ohne in zu viele Iterationen zu verfallen.

§V25-konform: Alle Schwellen aus Messungen abgeleitet.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveStrengthResult:
    """Ergebnis einer adaptiven Phasen-Optimierung."""

    phase_id: str
    optimal_strength: float
    was_executed: bool
    was_skipped: bool
    iterations: int
    best_delta: float
    strength_history: list[tuple[float, float]] = field(default_factory=list)
    # (strength, quality_delta) pro Iteration
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Konfiguration — ALLE Werte aus Pre-Analysis-Messwerten abgeleitet (§V25)
# ═══════════════════════════════════════════════════════════════════════════


def _adaptive_config(restorability_score: float, transfer_chain_depth: int, bandwidth_loss: float) -> dict[str, Any]:
    """Leitet adaptive Optimierungs-Parameter aus Pre-Analysis-Messwerten ab.

    Returns:
        Dict mit floor, ceiling, step, max_iterations — alle kontinuierlich.
    """
    rs = float(np.clip(restorability_score, 10.0, 100.0))
    depth = max(1, int(transfer_chain_depth))
    bw = float(np.clip(bandwidth_loss, 0.0, 1.0))

    # ── Stärke-Bereich ──
    # Je schlechter das Material, desto niedriger starten und desto feiner die Schritte
    # rs=30 → floor=0.03, step=0.04
    # rs=70 → floor=0.12, step=0.10
    # rs=95 → floor=0.25, step=0.12
    strength_floor = 0.03 + (rs - 10.0) / 400.0  # Kontinuierlich: 0.03 → 0.26
    strength_ceiling = 0.15 + (rs - 10.0) / 150.0  # Kontinuierlich: 0.15 → 0.75
    strength_step = 0.04 + (rs - 10.0) / 600.0  # Kontinuierlich: 0.04 → 0.19

    # ── Iterationen ──
    # Tiefe Ketten brauchen mehr Iterationen (mehr Unsicherheit)
    max_iterations = 5 + depth  # depth=1 → 6, depth=4 → 9

    # ── Bandwidth-Loss-Modifikator ──
    # Hoher bw_loss → noch konservativer starten. §G188 (GEBOTE.md): Dämpfung folgt dem
    # gemessenen Bandwidth-Verlust, nicht einem festen Literal (§7.4d (06_phases_system.md)).
    # Bei vollem Verlust (bw ≥ 1) exakt die bisherigen Kappen: floor ×0.6, ceiling/step ×0.7.
    if bw > 0.6:
        excess = min((bw - 0.6) / 0.4, 1.0)  # 0 an der Schwelle → 1 bei vollem Verlust
        strength_floor *= 1.0 - 0.4 * excess  # max. 0.6×
        strength_ceiling *= 1.0 - 0.3 * excess  # max. 0.7×
        strength_step *= 1.0 - 0.3 * excess  # max. 0.7×

    # Clipping
    strength_floor = float(np.clip(strength_floor, 0.02, 0.30))
    strength_ceiling = float(np.clip(strength_ceiling, 0.10, 0.85))
    strength_step = float(np.clip(strength_step, 0.03, 0.20))
    max_iterations = int(np.clip(max_iterations, 4, 12))

    return {
        "floor": round(strength_floor, 3),
        "ceiling": round(strength_ceiling, 3),
        "step": round(strength_step, 3),
        "max_iterations": max_iterations,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Schnelle Qualitäts-Messung (für Iterationen)
# ═══════════════════════════════════════════════════════════════════════════


def _quick_quality_delta(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sample_rate: int = 48000,
    *,
    before_result: Any = None,
) -> float:
    """Wohlklang-Delta für adaptive Iterationen (Hörordnung §1–§3).

    §Wohlklang-Optimum 2026-09-26: Diese Zielfunktion bewertete früher
    Signal-Ähnlichkeit (identisch = 1,0 = best, Baseline −0.95) — jede
    echte Reparatur SENKTE den Zielfert, die Suche blieb bei Default-Stärken
    (Produktionsbefund „0 boosted, 0 damped", Musik wurde nicht optimiert).
    Maßgeblich ist jetzt das HPE-Delta (maximaler Wohlklang, ±0.03 =
    hörbare Veränderung); die Signaltreue wirkt nur noch als
    Hör-Invarianten-Wächter (Ebene 1, Kollaps = harte Ablehnung).

    Kosten: zwei HPE-Messungen je Kandidat (before_result wiederverwendbar).
    """
    from backend.core.human_pleasantness_estimator import (
        wohlklang_objective_delta as _wohlklang_delta,
    )

    return _wohlklang_delta(audio_before, audio_after, sample_rate, original_result=before_result)


# ═══════════════════════════════════════════════════════════════════════════
# Adaptiver Phasen-Optimierer
# ═══════════════════════════════════════════════════════════════════════════

# Hörordnung JND (GEBOTE.md §G124): PMGG-Messungenauigkeit ±0.03 ist die
# kanonische Hörbarkeitsschwelle für HPE-Wohlklang-Deltas — dieselbe Semantik
# wie das HPE-Gate in per_phase_musical_goals_gate.py (< -0.03 ⇒ skip,
# [-0.03, 0) ⇒ tolerabel). Alle Skip-/Probe-Entscheidungen in diesem Modul
# beziehen sich auf diesen einen Wert — keine privaten Schwellen
# (§V7 Workaround-Verbot, §G9 Spec-Zitat).
_HPE_JND = 0.03


def optimize_phase_strength(
    *,
    phase_id: str,
    audio_input: np.ndarray,
    sample_rate: int,
    phase_runner: Callable[[np.ndarray, float], np.ndarray],
    # Funktion die (audio, strength) → processed_audio ausführt
    restorability_score: float = 50.0,
    transfer_chain_depth: int = 1,
    bandwidth_loss: float = 0.0,
    is_repair_phase: bool = False,
    is_enhancement_phase: bool = False,
    is_risky_phase: bool = False,
) -> AdaptiveStrengthResult:
    """Findet die optimale Stärke für EINE Phase durch iteratives Messen.

    Startet bei strength_floor und tastet sich in steps zum Optimum.
    Bricht ab wenn:
    - Keine Verbesserung mehr bei höherer Stärke
    - Max Iterationen erreicht
    - Verschlechterung bei ALLEN getesteten Stärken

    Args:
        phase_id: ID der Phase (für Logging)
        audio_input: Audio VOR der Phase
        sample_rate: Sample-Rate
        phase_runner: Funktion die Phase mit gegebener Stärke ausführt
        restorability_score: 0-100
        transfer_chain_depth: 1-5
        bandwidth_loss: 0-1
        is_repair_phase: Reparatur-Phase (darf stärker sein)
        is_enhancement_phase: Enhancement-Phase (moderat)
        is_risky_phase: Riskante Phase (muss sehr konservativ sein)

    Returns:
        AdaptiveStrengthResult mit optimaler Stärke und History.
    """
    cfg = _adaptive_config(restorability_score, transfer_chain_depth, bandwidth_loss)

    # Phasen-Typ-Modifikatoren
    floor = cfg["floor"]
    ceiling = cfg["ceiling"]
    step = cfg["step"]

    if is_repair_phase:
        floor *= 1.5  # Reparatur darf stärker starten
        ceiling = min(ceiling * 1.2, 0.90)
    elif is_enhancement_phase:
        floor *= 0.8  # Enhancement vorsichtiger
        ceiling *= 0.8
    elif is_risky_phase:
        floor *= 0.4  # Riskant SEHR vorsichtig
        ceiling *= 0.5
        step *= 0.7

    floor = float(np.clip(floor, 0.02, 0.50))
    ceiling = float(np.clip(ceiling, 0.08, 0.90))
    step = float(np.clip(step, 0.03, 0.20))
    max_iter = cfg["max_iterations"]

    # ── Iterative Optimierung ──
    history: list[tuple[float, float]] = []
    best_strength = 0.0
    best_delta = -999.0
    current = floor
    iteration = 0
    improving = True
    tried_zero = False

    # Wohlklang-Vorher-Bewertung EINMAL je Suche (Hörordnung): alle
    # Kandidaten werden gegen dieselbe Referenz gemessen.
    try:
        from backend.core.human_pleasantness_estimator import (
            compute_pleasantness as _compute_pleasantness,
        )

        _hpe_before: Any = _compute_pleasantness(audio_input, sample_rate)
    except Exception:
        _hpe_before = None

    while iteration < max_iter and current <= ceiling + 0.001:
        iteration += 1

        # Phase ausführen
        try:
            audio_after = phase_runner(audio_input, current)
        except Exception as e:
            logger.debug("AdaptiveStrength %s @ %.3f fehlgeschlagen: %s", phase_id, current, e)
            current += step
            continue

        # Wohlklang-Delta messen (Optimumsziel, nicht Signal-Ähnlichkeit)
        delta = _quick_quality_delta(audio_input, audio_after, sample_rate, before_result=_hpe_before)
        history.append((current, round(delta, 5)))

        if delta > best_delta:
            best_delta = delta
            best_strength = current
            if delta > 0.01:
                # Verbesserung → nächsthöhere Stärke testen
                current += step
                continue
            else:
                # Keine signifikante Verbesserung mehr → konvergiert
                improving = False
        else:
            # Verschlechterung gegenüber bestem Wert
            improving = False

        # Wenn wir am Floor sind und schon Verschlechterung → noch niedriger testen
        if not tried_zero and best_delta < -_HPE_JND and current <= floor + 0.001:
            current = max(0.02, floor * 0.5)
            tried_zero = True
            improving = True
            continue

        if not improving:
            # Konvergiert — prüfe ob nächsthöhere Stärke noch was bringt
            next_up = best_strength + step
            if next_up <= ceiling and iteration < max_iter:
                iteration += 1
                try:
                    audio_up = phase_runner(audio_input, next_up)
                    delta_up = _quick_quality_delta(audio_input, audio_up, sample_rate, before_result=_hpe_before)
                    history.append((next_up, round(delta_up, 5)))
                    if delta_up > best_delta + 0.005:
                        best_delta = delta_up
                        best_strength = next_up
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            break

    # ── Entscheidung ──
    # Hörordnung §2 (GEBOTE.md §G124): KEINE getestete Stärke bringt hörbare
    # Verbesserung (best Δ < −JND ±0.03) ⇒ Phase überspringen — dieselbe
    # Semantik wie das PMGG-HPE-Gate. Schließt die Dead Zone zwischen −0.05
    # und −HPE_JND, in der eine Phase mit best_delta ≈ −0.04 (≒ 1.3× JND)
    # trotzdem bei Floor-Stärke lief (hörbare Verschlechterung pro Song).
    was_skipped = best_delta < -_HPE_JND

    if was_skipped:
        logger.info(
            "§ADAPTIVE ueberspringen %s: best_delta=%.4f < -HPE_JND(-%.2f) @ strength=%.3f → Verarbeitungsschritt übersprungen",
            phase_id,
            best_delta,
            _HPE_JND,
            best_strength,
        )
    else:
        logger.info(
            "§ADAPTIVE %s: optimal_strength=%.3f delta=%.4f iterations=%d range=[%.2f-%.2f]",
            phase_id,
            best_strength,
            best_delta,
            iteration,
            floor,
            ceiling,
        )

    return AdaptiveStrengthResult(
        phase_id=phase_id,
        optimal_strength=round(best_strength, 3),
        was_executed=not was_skipped and best_strength > 0.01,
        was_skipped=was_skipped,
        iterations=iteration,
        best_delta=round(best_delta, 5),
        strength_history=history,
        reason=(
            f"Keine Stärke brachte Verbesserung (best Δ={best_delta:.4f})"
            if was_skipped
            else f"Optimal bei {best_strength:.3f} (Δ={best_delta:.4f})"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Batch-Optimierer: Alle Phasen adaptiv optimieren
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptivePipelineResult:
    """Ergebnis der adaptiven Optimierung über alle Phasen."""

    results: list[AdaptiveStrengthResult]
    total_phases_selected: int
    total_phases_executed: int
    total_phases_skipped: int
    total_iterations: int
    summary: str


def optimize_pipeline(
    *,
    phase_ids: list[str],
    audio_input: np.ndarray,
    sample_rate: int,
    phase_runner_factory: Callable[[str], Callable[[np.ndarray, float], np.ndarray] | None],
    restorability_score: float = 50.0,
    transfer_chain_depth: int = 1,
    bandwidth_loss: float = 0.0,
    repair_families: frozenset[str] | None = None,
    enhancement_families: frozenset[str] | None = None,
    risky_families: frozenset[str] | None = None,
) -> AdaptivePipelineResult:
    """Optimiert ALLE Phasen adaptiv — keine wird vorab eliminiert.

    Jede Phase bekommt ihre Chance bei minimaler Stärke.
    Nur wenn MESSUNG zeigt, dass keine Stärke hilft, wird sie übersprungen.
    """
    from backend.core.aurik_orchestrator import _family_from_phase_id

    repair = repair_families or frozenset()
    enhance = enhancement_families or frozenset()
    risky = risky_families or frozenset()

    results: list[AdaptiveStrengthResult] = []
    current_audio = audio_input

    for pid in phase_ids:
        runner = phase_runner_factory(pid)
        if runner is None:
            logger.debug("§ADAPTIVE: %s — kein Runner verfügbar, übersprungen", pid)
            continue

        family = _family_from_phase_id(pid)
        is_repair = family in repair
        is_enhance = family in enhance
        is_risky = family in risky

        t0 = time.monotonic()
        result = optimize_phase_strength(
            phase_id=pid,
            audio_input=current_audio,
            sample_rate=sample_rate,
            phase_runner=runner,
            restorability_score=restorability_score,
            transfer_chain_depth=transfer_chain_depth,
            bandwidth_loss=bandwidth_loss,
            is_repair_phase=is_repair,
            is_enhancement_phase=is_enhance,
            is_risky_phase=is_risky,
        )
        elapsed = time.monotonic() - t0
        results.append(result)

        if result.was_executed and not result.was_skipped:
            # Audio für nächste Phase aktualisieren
            try:
                current_audio = runner(current_audio, result.optimal_strength)
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                pass

        logger.debug(
            "§ADAPTIVE %s: %.3f @ %.3fs (%d iter)",
            pid,
            result.optimal_strength,
            elapsed,
            result.iterations,
        )

    executed = sum(1 for r in results if r.was_executed)
    skipped = sum(1 for r in results if r.was_skipped)
    total_iter = sum(r.iterations for r in results)

    summary = (
        f"Adaptive Optimierung: {executed} Phasen ausgeführt, "
        f"{skipped} übersprungen (von {len(phase_ids)}). "
        f"{total_iter} Iterationen gesamt."
    )
    logger.info("§ADAPTIVE PIPELINE: %s", summary)

    return AdaptivePipelineResult(
        results=results,
        total_phases_selected=len(phase_ids),
        total_phases_executed=executed,
        total_phases_skipped=skipped,
        total_iterations=total_iter,
        summary=summary,
    )

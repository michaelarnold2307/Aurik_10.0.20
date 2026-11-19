"""§2.60.1 Fahrplan — Denker als Dirigent, nicht als Schalter.

Der Denker erstellt einen strukturierten Ausführungsplan (Fahrplan),
der NICHT nur sagt WELCHE Phasen laufen, sondern WO, WIE STARK,
in welcher REIHENFOLGE, und mit welchem ZIEL.

Prinzipien:
  1. Per-Segment: Nicht alle Phasen überall gleich stark.
     → SectionGoalAdapter liefert Sektionen, Fahrplan skaliert pro Sektion.
  2. Goal-Priorität: Die schwächsten Goals zuerst angehen.
     → SongGoalImportance liefert Gewichte, Fahrplan ordnet Phasen danach.
  3. Phasen-Substitution: Wenn Risk-Guard eine Phase entfernt,
     → Fahrplan wählt Ersatz (z.B. phase_07→phase_23 für Harmonik).
  4. Perceptual-Budget: Mehr Verarbeitung in psychoakustisch kritischen Bändern.
     → 2-5kHz (Präsenz) mehr Budget als <100Hz oder >15kHz.
  5. Physical-Ceiling: Keine Frequenzen restoren die nie da waren.
     → SourceFidelityReconstructor gibt target_bw vor, Fahrplan respektiert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# §2.60.1 Perceptual-Budget: Psychoakustisch kritische Bänder bekommen mehr Verarbeitung.
# Verteilung basiert auf Fletcher-Munson: 2-5kHz (Präsenz) hat höchste Priorität.
PERCEPTUAL_BUDGET: dict[str, float] = {
    "sub_bass": 0.15,  # < 60 Hz
    "bass": 0.20,  # 60-250 Hz
    "low_mid": 0.15,  # 250-500 Hz
    "mid": 0.10,  # 500-2000 Hz
    "presence": 0.25,  # 2000-5000 Hz
    "high": 0.10,  # 5000-10000 Hz
    "air": 0.05,  # > 10000 Hz
}

# §2.60.1 Phasen-Substitutionen: Wenn Risk-Guard eine Phase entfernt, wählt der Fahrplan Ersatz.
PHASE_SUBSTITUTIONS: dict[str, str] = {
    "phase_07_harmonic_restoration": "phase_23_spectral_repair",
    "phase_23_spectral_repair": "phase_07_harmonic_restoration",
    "phase_02_hum_removal": "phase_03_denoise",
    "phase_03_denoise": "phase_02_hum_removal",
}


@dataclass
class PhaseSegmentInstruction:
    """Anweisung für eine Phase in einer bestimmten Sektion."""

    phase_id: str
    section_idx: int  # 0-basierter Sektions-Index
    strength_mod: float = 1.0  # Multiplikator auf Basis-Stärke
    priority: int = 5  # 1-10, höher = früher ausführen
    skip: bool = False  # True = in dieser Sektion überspringen
    reason: str = ""  # Begründung (für Debugging)


@dataclass
class Fahrplan:
    """Strukturierter Ausführungsplan für UV3.

    Ersetzt die flache Phasenliste durch eine 2D-Matrix:
    Phasen × Sektionen, mit per-Zelle Anweisungen.
    """

    # Phasen in Ausführungsreihenfolge (goal-priorisiert)
    phase_order: list[str] = field(default_factory=list)
    # Sektions-Liste von SectionGoalAdapter (start_s, end_s, label)
    sections: list[tuple[float, float, str]] = field(default_factory=list)
    # Per-Sektion-Phasen-Anweisungen
    instructions: dict[str, list[PhaseSegmentInstruction]] = field(default_factory=dict)
    # Phasen die global laufen (nicht per Sektion)
    global_phases: list[str] = field(default_factory=list)
    # Phasen-Substitutionen (entfernt → Ersatz)
    substitutions: dict[str, str] = field(default_factory=dict)
    # Physical Ceiling: maximale Zielfrequenz
    physical_ceiling_hz: float = 20000.0
    # Perceptual-Budget-Verteilung (Frequenzband → Budget-Anteil)
    perceptual_budget: dict[str, float] = field(default_factory=dict)
    # Goal-Prioritäten (goal → priority 1-10)
    goal_priorities: dict[str, int] = field(default_factory=dict)
    # Metadaten
    total_segments: int = 0
    note: str = ""

    @property
    def calibration(self) -> dict[str, float]:
        """Flat dict: phase_id → average strength across all segments."""
        result = {}
        for pid in self.phase_order:
            instrs = self.instructions.get(pid, [])
            if instrs:
                strengths = [i.strength_mod for i in instrs if not i.skip]
                result[pid] = sum(strengths) / max(len(strengths), 1) if strengths else 1.0
            else:
                result[pid] = 1.0
        return result


def build_fahrplan(
    phase_ids: list[str],
    sections: list[tuple[float, float, str]],
    *,
    goal_priorities: dict[str, float],
    physical_ceiling_hz: float = 20000.0,
    removed_risk_phases: list[str] | None = None,
    phase_effect_catalog: dict[str, Any] | None = None,
    audio_ctx: dict[str, Any] | None = None,
) -> Fahrplan:
    """Erstellt einen strukturierten Fahrplan aus Denker-Wissen.

    Args:
        phase_ids: Ausgewählte Phasen (von PhaseInteractionDenker)
        sections: Sektionen von SectionGoalAdapter [(start_s, end_s, label), ...]
        goal_priorities: Goal-Gewichte von SongGoalImportance
        physical_ceiling_hz: Maximale wiederherstellbare Frequenz
        removed_risk_phases: Vom Risk-Guard entfernte Phasen
        phase_effect_catalog: PhaseEffectProfile-Dict
        audio_ctx: Audio-Kontext für Kalibrierung

    Returns:
        Fahrplan mit per-Segment-Anweisungen
    """
    sections = sections or [(0.0, 1.0, "full")]
    audio_ctx = audio_ctx or {}

    plan = Fahrplan(
        sections=sections,
        total_segments=len(sections),
        physical_ceiling_hz=physical_ceiling_hz,
        perceptual_budget=dict(PERCEPTUAL_BUDGET),
        goal_priorities={
            g: int(p * 10) for g, p in sorted(goal_priorities.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        if goal_priorities
        else {},
    )

    # ── 1. Substitutionen auflösen ──────────────────────────
    removed = set(removed_risk_phases or [])
    for removed_phase in removed:
        substitute = PHASE_SUBSTITUTIONS.get(removed_phase)
        if substitute and substitute not in phase_ids and substitute not in removed:
            plan.substitutions[removed_phase] = substitute
            phase_ids = [p if p != removed_phase else substitute for p in phase_ids]

    # ── 2. Goal-priorisierte Reihenfolge ────────────────────
    # Schwächste Goals → deren Phasen zuerst
    if goal_priorities:
        # Sortiere Phasen nach der Priorität ihrer Goal-Impacts
        def _phase_goal_score(pid: str) -> float:
            if phase_effect_catalog and pid in phase_effect_catalog:
                profile = phase_effect_catalog[pid]
                score = 0.0
                for goal, impact in profile.goal_impact.items():
                    priority = goal_priorities.get(goal, 0.5)
                    # Negative Impacts (Risiken) bei schwachen Goals → Phase später
                    if impact < 0 and priority < 0.3:
                        score -= abs(impact) * (1.0 - priority)
                    else:
                        score += abs(impact) * priority
                return score
            return 0.0

        plan.phase_order = sorted(phase_ids, key=_phase_goal_score, reverse=True)
    else:
        plan.phase_order = list(phase_ids)

    # ── 3. Per-Segment-Anweisungen ──────────────────────────
    for pid in plan.phase_order:
        plan.instructions[pid] = []
        for s_idx, (s_start, s_end, s_label) in enumerate(sections):
            instr = PhaseSegmentInstruction(
                phase_id=pid,
                section_idx=s_idx,
                strength_mod=1.0,
                priority=5,
            )

            # Segment-spezifische Regeln:
            s_label_lower = s_label.lower()

            # Stille-Sektionen → Click/Hum-Phasen überspringen
            if "silence" in s_label_lower and pid in (
                "phase_01_click_removal",
                "phase_02_hum_removal",
                "phase_09_crackle_removal",
                "phase_27_click_pop_removal",
            ):
                instr.skip = True
                instr.reason = "silence_segment"

            # Intro/Outro → weniger invasive Eingriffe
            if any(w in s_label_lower for w in ("intro", "outro", "fade")):
                instr.strength_mod *= 0.6
                instr.reason = "boundary_segment"

            # Strophe (Verse) → Gesangs-Phasen verstärken
            if "verse" in s_label_lower or "strophe" in s_label_lower:
                if pid in ("phase_19_de_esser", "phase_03_denoise"):
                    instr.strength_mod *= 1.15
                    instr.reason = "vocal_segment"
                    instr.priority += 1

            # Refrain → Dynamik-Phasen verstärken (lauter = mehr Energie)
            if "chorus" in s_label_lower or "refrain" in s_label_lower:
                if pid in ("phase_26_dynamic_range_expansion", "phase_08_transient_preservation"):
                    instr.strength_mod *= 1.1
                    instr.reason = "chorus_energy"

            # Bridge → konservativer (oft ruhiger, detailreich)
            if "bridge" in s_label_lower:
                instr.strength_mod *= 0.7
                instr.reason = "bridge_delicate"

            plan.instructions[pid].append(instr)

    # ── 4. Global vs. Per-Segment klassifizieren ─────────────
    for pid in plan.phase_order:
        instrs = plan.instructions.get(pid, [])
        # Wenn alle Sektionen gleiche Stärke → global
        strengths = {i.strength_mod for i in instrs}
        if len(strengths) == 1 and not any(i.skip for i in instrs):
            plan.global_phases.append(pid)

    # ── 5. Physical-Ceiling-Anpassung ────────────────────────
    if physical_ceiling_hz < 15000:
        # Bei starkem Bandbreitenverlust: Air/Ultra-Phasen drosseln
        for pid in ("phase_39_air_band_enhancement", "phase_38_presence_boost"):
            if pid in plan.phase_order:
                for instr in plan.instructions.get(pid, []):
                    instr.strength_mod *= max(0.2, physical_ceiling_hz / 15000)
                    instr.reason = f"physical_ceiling_{physical_ceiling_hz:.0f}hz"

    plan.note = (
        f"{len(plan.phase_order)} Phasen, {len(plan.sections)} Sektionen, "
        f"{len(plan.global_phases)} global, "
        f"{len(plan.substitutions)} Substitutionen, "
        f"Ceiling={physical_ceiling_hz:.0f}Hz"
    )

    return plan


def fahrplan_to_log(fahrplan: Fahrplan) -> str:
    """Lesbare Zusammenfassung für Logging."""
    lines = [
        f"🚗 FAHRPLAN: {fahrplan.note}",
        f"   Reihenfolge: {' → '.join(fahrplan.phase_order[:8])}..."
        if len(fahrplan.phase_order) > 8
        else f"   Reihenfolge: {' → '.join(fahrplan.phase_order)}",
    ]
    if fahrplan.substitutions:
        for old, new in fahrplan.substitutions.items():
            lines.append(f"   🔄 {old} → {new} (Substitution)")
    if fahrplan.goal_priorities:
        goals = sorted(fahrplan.goal_priorities.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"   🎯 Prioritäten: {', '.join(f'{g}={p}' for g, p in goals[:3])}")
    return "\n".join(lines)

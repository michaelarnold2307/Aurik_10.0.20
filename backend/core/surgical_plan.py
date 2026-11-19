"""§2.59.14 Surgical Plan — Präzisions-Chirurgie durch Denker-Orchestrierung.

Statt neuer Lightweight-Funktionen werden die EXISTIERENDEN,
kampferprobten Phasen im chirurgischen Modus ausgeführt:
nur auf den vom DefectScanner lokalisierten Zeitfenstern,
mit voller Psychoakustik, PMGG und Safety-Clamps.

Fluss:
  PhaseInteractionDenker.plan()
    → SurgicalPlan (Defekt → Phase, Zeitfenster, Stärke)
    → UV3.restore(precomputed_surgical_plan=...)
    → PhaseInterface.execute(time_range=(start_s, end_s))
    → Safety-Clamp: output ∈ [−2× input, +2× input]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Mapping: DefectType → Phase-ID die diesen Defekt chirurgisch behandeln kann
# Nur Phasen mit bekannter, stabiler per-Instance-Reparatur.
SURGICAL_DEFECT_TO_PHASE: dict[str, str] = {
    # Transienten → Click-Phasen
    "clicks": "phase_01_click_removal",
    # Crackle/Knistern → Surface-Noise-Phase (arbeitet per-Instance)
    "crackle": "phase_28_surface_noise_profiling",
    # Dropouts → Dropout-Repair (arbeitet per-Instance via Gap-Detektion)
    "dropouts": "phase_24_dropout_repair",
    "dropout_oxide": "phase_24_dropout_repair",
    "dropout_head_contact": "phase_24_dropout_repair",
    "dropout_splice": "phase_24_dropout_repair",
    # Wow/Flutter → Phase 12 (arbeitet per-Fenster)
    "wow": "phase_12_wow_flutter_fix",
    "flutter": "phase_12_wow_flutter_fix",
    "transport_bump": "phase_12_wow_flutter_fix",
    "scrape_flutter": "phase_12_wow_flutter_fix",
    "multiband_wow_flutter": "phase_12_wow_flutter_fix",
    # Tape-Splice → Click-Phase (Klick-Anteil) + danach Gain-Ausgleich
    "tape_splice_artifact": "phase_01_click_removal",
    # Sibilance → De-Esser (arbeitet per-Instanz via Band-Detektion)
    "sibilance": "phase_19_de_esser",
    # Motor-Interferenz → Hum-Phase (arbeitet per-Frequenz, adaptiv)
    "motor_interference": "phase_02_hum_removal",
    # Modulation-Noise → Tape-Hiss-Phase
    "modulation_noise": "phase_29_tape_hiss_reduction",
    # Pre-Echo / Print-Through → Reverb-Reduction (temporale Artefakt-Unterdrückung)
    "pre_echo": "phase_20_reverb_reduction",
    "print_through": "phase_20_reverb_reduction",
    # MPEG-Frame-Loss → Dropout-Repair
    "mpeg_frame_loss": "phase_24_dropout_repair",
    # Sticky-Shed → Oberflächenrausch-Phase
    "sticky_shed_residue": "phase_28_surface_noise_profiling",
    # Tape-Head-Clog → Dropout-Repair (Pegel-Kompensation)
    "tape_head_clog": "phase_24_dropout_repair",
    # DC-Offset → EQ-Phase (Subsonic-Filter)
    "dc_offset": "phase_05_rumble_filter",
    # Groove-Echo → Reverb-Reduction
    "groove_echo": "phase_20_reverb_reduction",
    # Inner-Groove-Distortion → Spektrale Reparatur (HF-Entzerrung)
    "inner_groove_distortion": "phase_23_spectral_repair",
    # Transient-Smearing → Transient-Preservation
    "transient_smearing": "phase_08_transient_preservation",
}


@dataclass
class SurgicalInstruction:
    """Eine einzelne chirurgische Anweisung: Defekt → Phase → Zeitfenster."""

    defect_type: str
    phase_id: str
    start_s: float
    end_s: float
    severity: float = 1.0
    # Pro-Instanz-Metadaten (Magnituden etc.) vom DefectScanner
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurgicalPlan:
    """Kompletter chirurgischer Plan für einen Song.

    Wird vom PhaseInteractionDenker erstellt und an UV3.restore()
    als precomputed_surgical_plan übergeben.
    """

    instructions: list[SurgicalInstruction] = field(default_factory=list)
    # Metadaten
    total_zones: int = 0
    total_defect_types: int = 0
    audio_duration_s: float = 0.0
    # Safety
    max_amplitude_ratio: float = 2.0  # Output nie > 2× Input

    def add(
        self,
        defect_type: str,
        phase_id: str,
        start_s: float,
        end_s: float,
        severity: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fügt eine chirurgische Anweisung hinzu."""
        # Validiere: kein Placeholder (0–duration)
        dur = end_s - start_s
        if self.audio_duration_s > 0 and dur > self.audio_duration_s * 0.5:
            return  # Skip Placeholder
        if dur < 0.0001:  # < 0.1ms
            return  # Skip zu kurze Events
        self.instructions.append(
            SurgicalInstruction(
                defect_type=defect_type,
                phase_id=phase_id,
                start_s=start_s,
                end_s=end_s,
                severity=severity,
                metadata=metadata or {},
            )
        )

    def by_phase(self) -> dict[str, list[SurgicalInstruction]]:
        """Gruppiert Anweisungen nach Phase-ID."""
        result: dict[str, list[SurgicalInstruction]] = {}
        for inst in self.instructions:
            result.setdefault(inst.phase_id, []).append(inst)
        return result

    def summary(self) -> str:
        """Lesbare Zusammenfassung."""
        by_defect: dict[str, int] = {}
        for inst in self.instructions:
            by_defect[inst.defect_type] = by_defect.get(inst.defect_type, 0) + 1
        parts = [f"{d}={c}×" for d, c in sorted(by_defect.items())]
        return ", ".join(parts) if parts else "leer"

    def __len__(self) -> int:
        return len(self.instructions)

    def __bool__(self) -> bool:
        return len(self.instructions) > 0

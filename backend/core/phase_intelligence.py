"""§U: Phase-Ordering-Intelligence — akustische Interaktion zwischen Phasen.

Der PhaseInteractionDenker (PID) ordnet Phasen aktuell nach Defekt-Schwere und
Material.  Aber bestimmte Phasen-Sequenzen klingen besser als andere:
- Denoise VOR Dereverb = natürlicher (Rauschen maskiert Hall-Artifakte)
- EQ NACH Dynamik-Kompression = ausgewogener (Kompression ändert Spektrum)
- Stereo-Enhancement VOR Bass-Boost = breiteres Stereobild ohne Matsch

PhaseOrderIntelligence analysiert die PID-Phase-Liste, erkennt akustische
Kopplungen und schlägt eine optimierte Reihenfolge vor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Akustische Kopplungs-Regeln ───────────────────────────────────────────────
# Format: ("phase_A", "phase_B", "before"|"after", reason)
# "phase_A before phase_B" bedeutet: A sollte VOR B ausgeführt werden.
_ACOUSTIC_COUPLING_RULES: list[tuple[str, str, str, str]] = [
    # Denoise → Dereverb: Rauschen maskiert Hall-Artefakte
    (
        "phase_03_denoise",
        "phase_49_advanced_dereverb",
        "before",
        "Rauschen maskiert Hall-Artefakte — Denoise zuerst für natürlicheren Klang",
    ),
    # Dereverb → Denoise: saubere Reflexionen helfen dem Denoiser
    ("phase_49_advanced_dereverb", "phase_03_denoise", "after", "Saubere Reflexionen verbessern Denoiser-Erkennung"),
    # Dynamics → EQ: Kompression ändert Spektrum — EQ danach
    (
        "phase_54_transparent_dynamics",
        "phase_16_final_eq",
        "before",
        "Kompression ändert Frequenzbalance — EQ muss danach kommen",
    ),
    # EQ → Kompression: EQ vor Kompression vermeidet Pumping
    ("phase_16_final_eq", "phase_54_transparent_dynamics", "before", "EQ vor Kompression verhindert Pumping-Artefakte"),
    # Stereo → Bass: Breites Stereobild vor Bass-Boost = kein Matsch
    (
        "phase_13_stereo_enhancement",
        "phase_37_bass_enhancement",
        "before",
        "Stereo-Breite vor Bass-Boost — verhindert Mono-Kompatibilitäts-Probleme",
    ),
    # Bass → Stereo: Bass-Fundament stabilisiert Stereobild
    ("phase_37_bass_enhancement", "phase_13_stereo_enhancement", "after", "Bass-Fundament stabilisiert Stereobild"),
    # Transienten → EQ: Transienten-Shaping beeinflusst Höhenwahrnehmung
    (
        "phase_08_transient_preservation",
        "phase_16_final_eq",
        "before",
        "Transienten formen Höhenwahrnehmung — EQ danach abstimmen",
    ),
    # De-Esser → Presence: Zischlaute erst entfernen, dann Presence boosten
    ("phase_43_ml_deesser", "phase_38_presence_boost", "before", "Erst Zischlaute entfernen, dann Präsenz anheben"),
    # Click/Crackle → Denoise: Transiente Störer vor Breitband-Denoise
    (
        "phase_01_click_removal",
        "phase_03_denoise",
        "before",
        "Transiente Störer vor Breitband-Rauschunterdrückung entfernen",
    ),
    (
        "phase_09_crackle_removal",
        "phase_03_denoise",
        "before",
        "Knistern vor Denoise — sonst wird Knistern als Rauschen interpretiert",
    ),
    # Hum → Denoise: Brummen vor Denoise entfernen
    (
        "phase_02_hum_removal",
        "phase_03_denoise",
        "before",
        "Brummen vor Denoise — Denoiser arbeitet sonst gegen stationäres Brummen",
    ),
    # Wow/Flutter → Speed/Pitch: Erst Geschwindigkeit korrigieren, dann Pitch
    (
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
        "before",
        "Erst Gleichlauf, dann Geschwindigkeit — kumulative Korrektur vermeiden",
    ),
]


@dataclass
class PhaseOrderResult:
    """Ergebnis der Phasen-Ordnungs-Optimierung."""

    original_order: list[str] = field(default_factory=list)
    optimized_order: list[str] = field(default_factory=list)
    changes: list[dict[str, str]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0


class PhaseOrderIntelligence:
    """Optimiert die Phasen-Reihenfolge basierend auf akustischen Kopplungen.

    Verwendet den PID-Plan als Input und wendet _ACOUSTIC_COUPLING_RULES an.
    """

    def __init__(self) -> None:
        self._rules = list(_ACOUSTIC_COUPLING_RULES)

    def optimize(self, phases: list[str]) -> PhaseOrderResult:
        """Optimiert die Reihenfolge. Gibt original + optimiert zurück."""
        result = PhaseOrderResult(original_order=list(phases))
        if len(phases) <= 2:
            result.optimized_order = list(phases)
            return result

        # Schritt 1: Sammle anwendbare Regeln
        phase_set = set(phases)
        applicable: list[tuple[int, int, str, str]] = []
        for phase_a, phase_b, relation, reason in self._rules:
            if phase_a in phase_set and phase_b in phase_set:
                idx_a = phases.index(phase_a)
                idx_b = phases.index(phase_b)
                if relation == "before" and idx_a > idx_b:
                    applicable.append((idx_a, idx_b, phase_a, reason))
                    result.reasons.append(reason)
                elif relation == "after" and idx_a < idx_b:
                    applicable.append((idx_b, idx_a, phase_b, reason))
                    result.reasons.append(reason)

        if not applicable:
            result.optimized_order = list(phases)
            return result

        # Schritt 2: Wende Regeln an (topologische Sortierung light)
        optimized = list(phases)
        for high_idx, low_idx, phase_name, reason in applicable:
            # Verschiebe phase_name von high_idx vor low_idx
            if high_idx > low_idx and phase_name in optimized:
                optimized.remove(phase_name)
                optimized.insert(low_idx, phase_name)
                result.changes.append(
                    {
                        "phase": phase_name,
                        "from_pos": str(high_idx + 1),
                        "to_pos": str(low_idx + 1),
                        "reason": reason,
                    }
                )

        result.optimized_order = optimized
        result.score = 1.0 - (len(result.changes) / max(1, len(phases)))
        return result

    def recommend_adjacent_phases(self, phase_id: str, all_phases: list[str]) -> list[str]:
        """Empfiehlt Phasen, die NACH dieser Phase kommen sollten."""
        phase_set = set(all_phases)
        recommendations = []
        for phase_a, phase_b, relation, reason in self._rules:
            if phase_a == phase_id and phase_b in phase_set and relation == "before":
                recommendations.append(phase_b)
        return recommendations

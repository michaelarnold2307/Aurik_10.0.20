"""§v10.15 Pipeline Plausibility Guard
======================================
Cross-validates phase selection, material detection, era classification
against each other and against known-good configurations.

Prüft:
- Wurde das korrekte Material erkannt? (passt es zur Dateiendung?)
- Sind die ausgewählten Phasen für das Material angemessen?
- Fehlen Pflicht-Phasen?
- Wurden widersprüchliche Phasen ausgewählt?
- Ist die Phasen-Anzahl im erwarteten Bereich?

Kommuniziert Funde über die Narrative Engine an den Nutzer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Expected phase counts per material ────────────────────────────
EXPECTED_PHASE_COUNTS: dict[str, tuple[int, int]] = {
    # (min, max) expected phases for restoration mode
    "shellac": (30, 55),
    "wax_cylinder": (30, 55),
    "wire_recording": (28, 52),
    "vinyl": (25, 48),
    "reel_tape": (22, 45),
    "tape": (22, 45),
    "cassette": (28, 50),
    "lacquer_disc": (25, 48),
    "minidisc": (15, 35),
    "mp3_low": (12, 30),
    "mp3_high": (10, 28),
    "cd_digital": (8, 22),
    "streaming": (5, 18),
    "unknown": (15, 45),
}

# ── Pflicht-Phasen pro Material (müssen vorhanden sein) ───────────
MANDATORY_PHASES: dict[str, list[str]] = {
    "shellac": ["phase_05_rumble_filter", "phase_09_crackle_removal", "phase_23_spectral_repair"],
    "vinyl": ["phase_05_rumble_filter", "phase_09_crackle_removal"],
    "cassette": ["phase_12_wow_flutter_fix", "phase_29_tape_hiss_reduction"],
    "reel_tape": ["phase_12_wow_flutter_fix", "phase_29_tape_hiss_reduction", "phase_64_tape_splice_repair"],
    "tape": ["phase_12_wow_flutter_fix", "phase_29_tape_hiss_reduction"],
}

# ── Widersprüchliche Phasen (dürfen NICHT gemeinsam vorkommen) ───
CONTRADICTORY_PHASE_PAIRS: list[tuple[str, str]] = [
    ("phase_60_inner_groove_distortion_repair", "phase_29_tape_hiss_reduction"),
    ("phase_61_groove_echo_cancellation", "phase_29_tape_hiss_reduction"),
]

# ── Material-Profil: erwartete Eigenschaften ─────────────────────
MATERIAL_PROFILES: dict[str, dict[str, Any]] = {
    "vinyl": {"typical_snr_range": (15, 55), "typical_bw_hz": (8000, 20000), "has_surface_noise": True},
    "shellac": {"typical_snr_range": (5, 30), "typical_bw_hz": (3000, 8000), "has_surface_noise": True},
    "cassette": {"typical_snr_range": (10, 45), "typical_bw_hz": (5000, 16000), "has_surface_noise": False},
    "reel_tape": {"typical_snr_range": (15, 55), "typical_bw_hz": (8000, 22000), "has_surface_noise": False},
    "cd_digital": {"typical_snr_range": (40, 96), "typical_bw_hz": (15000, 22050), "has_surface_noise": False},
}

# ── Dateiendung → erwartetes Material ────────────────────────────
EXTENSION_MATERIAL_MAP: dict[str, str] = {
    ".mp3": "mp3_low",
    ".flac": "cd_digital",
    ".wav": "cd_digital",
    ".aac": "mp3_high",
    ".ogg": "mp3_high",
    ".wma": "mp3_high",
}


@dataclass
class PlausibilityReport:
    """Ergebnis der Plausibilitätsprüfung."""

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    material_detected: str = ""
    material_expected_from_extension: str = ""
    material_confidence: float = 0.0

    phase_count: int = 0
    phase_count_expected_range: tuple[int, int] = (0, 0)
    missing_mandatory: list[str] = field(default_factory=list)
    contradictory_found: list[tuple[str, str]] = field(default_factory=list)

    era_detected: int = 0
    era_plausible: bool = True

    snr_detected: float = 0.0
    bandwidth_detected: float = 0.0

    narrative_summary: str = ""


class PipelinePlausibilityGuard:
    """Prüft die Pipeline-Konfiguration auf Plausibilität."""

    def __init__(self) -> None:
        """Initialize the plausibility guard with default thresholds."""
        self._min_confidence: float = 0.6
        self._warn_bandwidth_hz: float = 8000.0
        self._max_phases: int = 64
        self._issues: list[str] = []

    def check(
        self,
        material: str,
        material_confidence: float,
        file_extension: str,
        selected_phases: list[str],
        era: int,
        snr_db: float,
        bandwidth_hz: float,
        carrier_chain: list[str] | None = None,
    ) -> PlausibilityReport:
        """Führt alle Plausibilitäts-Prüfungen durch."""
        report = PlausibilityReport()
        report.material_detected = material
        report.material_confidence = material_confidence
        report.phase_count = len(selected_phases)
        report.era_detected = era
        report.snr_detected = snr_db
        report.bandwidth_detected = bandwidth_hz

        _mat_key = str(material).lower().replace("-", "_").replace(" ", "_")

        # 1. Material vs Dateiendung
        _ext = str(file_extension).lower()
        report.material_expected_from_extension = EXTENSION_MATERIAL_MAP.get(_ext, "unknown")
        if _ext == ".mp3" and material not in ("mp3_low", "mp3_high", "unknown"):
            if material_confidence < 0.4:
                report.warnings.append(
                    f"MP3-Datei, aber Material wurde als '{material}' erkannt "
                    f"(Konfidenz: {material_confidence:.0%}). "
                    "Analoge Artefakte in MP3 sind unwahrscheinlich."
                )

        # 2. Material-Konfidenz
        if material_confidence < 0.15 and material != "unknown":
            report.warnings.append(
                f"Material-Erkennung sehr unsicher ({material_confidence:.0%}). '{material}' könnte falsch sein."
            )

        # 3. Phasen-Anzahl
        _expected = EXPECTED_PHASE_COUNTS.get(_mat_key, EXPECTED_PHASE_COUNTS["unknown"])
        report.phase_count_expected_range = _expected
        if report.phase_count < _expected[0]:
            report.warnings.append(
                f"Nur {report.phase_count} Phasen ausgewählt (erwartet: {_expected[0]}–{_expected[1]}). "
                "Möglicherweise fehlen wichtige Restaurations-Schritte."
            )
        elif report.phase_count > _expected[1]:
            report.info.append(
                f"{report.phase_count} Phasen ausgewählt (mehr als übliche {_expected[1]}). "
                "Gründliche Restauration — alle Eventualitäten abgedeckt."
            )

        # 4. Pflicht-Phasen
        _mandatory = MANDATORY_PHASES.get(_mat_key, [])
        _missing = [p for p in _mandatory if p not in selected_phases]
        report.missing_mandatory = _missing
        if _missing:
            report.errors.append(
                f"Pflicht-Phasen für '{material}' fehlen: {', '.join(_missing)}. "
                "Die Restauration wird unvollständig sein."
            )
            report.passed = False

        # 5. Widersprüchliche Phasen
        _contra = [(a, b) for a, b in CONTRADICTORY_PHASE_PAIRS if a in selected_phases and b in selected_phases]
        report.contradictory_found = _contra
        if _contra:
            for a, b in _contra:
                report.errors.append(f"Widersprüchliche Phasen: '{a}' + '{b}'. Eine davon sollte entfernt werden.")
            report.passed = False

        # 6. Material-Profil vs gemessene Werte
        _profile = MATERIAL_PROFILES.get(_mat_key)
        if _profile:
            _snr_range = _profile.get("typical_snr_range", (0, 100))
            if snr_db < _snr_range[0] - 5:
                report.info.append(
                    f"SNR ({snr_db:.1f} dB) ist ungewöhnlich niedrig für '{material}'. Stärkere Entrauschung nötig."
                )
            _bw_range = _profile.get("typical_bw_hz", (0, 48000))
            if bandwidth_hz < _bw_range[0] * 0.5:
                report.warnings.append(
                    f"Bandbreite ({bandwidth_hz:.0f} Hz) ist sehr niedrig für '{material}'. "
                    "Frequenz-Erweiterung wird kritisch sein."
                )

        # 7. Ära-Plausibilität
        report.era_plausible = True
        if material in ("vinyl", "shellac") and era >= 1990:
            report.era_plausible = False
            report.warnings.append(
                f"'{material}'-Material mit Ära {era} ist ungewöhnlich. "
                "Möglicherweise ein digitalisiertes Archiv-Stück."
            )
        if material in ("cd_digital", "streaming") and era < 1980:
            report.era_plausible = False
            report.info.append(f"Digitales Material mit Ära {era} — wahrscheinlich eine spätere Digitalisierung.")

        # 8. Carrier-Chain-Kohärenz
        if carrier_chain:
            _chain_analog = [c for c in carrier_chain if c in ("vinyl", "shellac", "reel_tape", "cassette", "tape")]
            _chain_digital = [c for c in carrier_chain if c in ("mp3_low", "mp3_high", "cd_digital")]
            if len(_chain_analog) >= 2 and len(_chain_digital) >= 2:
                report.info.append(
                    "Lange Tonträger-Kette mit mehreren analogen und digitalen Stufen. "
                    "Jede Stufe hat eigene Artefakte hinterlassen."
                )

        # 9. Narrative Zusammenfassung
        report.narrative_summary = self._build_narrative(report)

        return report

    def _build_narrative(self, report: PlausibilityReport) -> str:
        """Erzeugt eine narrative Zusammenfassung der Prüfung."""
        parts = []

        if report.passed and not report.warnings and not report.errors:
            parts.append(
                f"Die automatische Analyse hat '{report.material_detected}' "
                f"als Tonträger erkannt — alle {report.phase_count} Phasen "
                "sind stimmig ausgewählt. Die Pipeline-Konfiguration ist plausibel."
            )
            return " ".join(parts)

        if report.errors:
            parts.append(f"⚡ {len(report.errors)} kritische Probleme gefunden.")
        if report.warnings:
            parts.append(f"⚠️ {len(report.warnings)} Unstimmigkeiten entdeckt.")
        if report.info and not report.errors:
            parts.append(f"ℹ️ {len(report.info)} interessante Beobachtungen.")

        return " ".join(parts)

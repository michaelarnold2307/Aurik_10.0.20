"""
§2.59 DefectManifest — Zentrale Defekt-Registry (2026-07-09)

Kanonische, einzige Quelle für:
  - DefectType → behandelnde Phase(n)
  - DefectType → beeinflusste Musical Goals
  - DefectType → empfohlene Repair-Stärke-Kategorie

Module (PhasePruner, SongGoalImportance, DefectPrecisionEnhancer, etc.)
lesen aus dieser Registry statt eigene, potenziell divergierende Listen
zu pflegen. Neue DefectTypes werden hier registriert und sind automatisch
in allen Modulen sichtbar.

Usage:
    from backend.core.defect_manifest import DEFECT_MANIFEST

    phases = DEFECT_MANIFEST.get_phases_for_defect("clicks")
    goals = DEFECT_MANIFEST.get_goals_for_defect("bandwidth_loss")
    strength = DEFECT_MANIFEST.get_strength_category("wow")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Strength Categories ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RepairStrength:
    """Empfohlene Repair-Stärke für einen Defekt-Typ."""

    category: str  # "gentle", "moderate", "aggressive"
    min_strength: float = 0.2
    max_strength: float = 1.0
    severity_multiplier: float = 1.0  # Multiplikator auf Defekt-Severity


GENTLE = RepairStrength("gentle", 0.2, 0.8, 1.2)
MODERATE = RepairStrength("moderate", 0.3, 1.0, 1.3)
AGGRESSIVE = RepairStrength("aggressive", 0.3, 1.0, 1.5)
NEUTRAL = RepairStrength("neutral", 0.2, 1.0, 1.0)


# ── Defect Entry ───────────────────────────────────────────────────────────


@dataclass
class DefectEntry:
    """Vollständige Beschreibung eines Defekts: Phasen, Goals, Stärke."""

    defect_value: str  # DefectType.value
    phases: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    strength: RepairStrength = NEUTRAL
    description: str = ""


# ── Manifest ───────────────────────────────────────────────────────────────


class DefectManifest:
    """Zentrale Registry aller Defekt→Phase/Goal/Strength-Mappings."""

    def __init__(self) -> None:
        self._entries: dict[str, DefectEntry] = {}
        self._populate()

    def _populate(self) -> None:
        """Registriert alle DefectType-Werte mit ihren Mappings."""
        # Import hier, um Circular Imports zu vermeiden
        from backend.core.defect_scanner import DefectType

        for dt in DefectType:
            entry = DefectEntry(defect_value=dt.value)
            self._entries[dt.value] = entry

        # ── Phase-Mappings ──────────────────────────────────────────
        self._map_phases()

        # ── Goal-Mappings ───────────────────────────────────────────
        self._map_goals()

        # ── Strength-Mappings ───────────────────────────────────────
        self._map_strengths()

    def _map_phases(self) -> None:
        """Definiert, welche Phase(n) jeden Defekt behandeln."""
        pm = self._phase_map = {
            # --- Grundreinigung ---
            "clicks": ["phase_01_click_removal", "phase_27_click_pop_removal"],
            "crackle": [
                "phase_03_denoise",
                "phase_09_crackle_removal",
                "phase_28_surface_noise_profiling",
                "phase_67_crackle_texture_removal",
            ],
            "hum": ["phase_02_hum_removal"],
            "motor_interference": ["phase_02_hum_removal"],
            "high_freq_noise": [
                "phase_03_denoise",
                "phase_18_noise_gate",
                "phase_29_tape_hiss_reduction",
                "phase_66_stem_targeted_nr",
            ],
            "modulation_noise": [
                "phase_03_denoise",
                "phase_29_tape_hiss_reduction",
                "phase_59_modulation_noise_reduction",
                "phase_66_stem_targeted_nr",
            ],
            "quantization_noise": ["phase_03_denoise"],
            "aliasing": ["phase_03_denoise"],
            "low_freq_rumble": ["phase_05_rumble_filter"],
            # --- Frequenz / Harmonik ---
            "clipping": ["phase_06_frequency_restoration", "phase_07_harmonic_restoration", "phase_11_limiting"],
            "bandwidth_loss": [
                "phase_06_frequency_restoration",
                "phase_23_spectral_repair",
                "phase_39_air_band_enhancement",
                "phase_55_diffusion_inpainting",
                "phase_56_spectral_band_gap_repair",
            ],
            "overload_distortion": ["phase_07_harmonic_restoration"],
            "intermodulation_distortion": ["phase_63_intermodulation_reduction"],
            "transient_smearing": ["phase_08_transient_preservation"],
            # --- Transport ---
            "wow": ["phase_12_wow_flutter_fix"],
            "flutter": ["phase_12_wow_flutter_fix"],
            "multiband_wow_flutter": ["phase_12_wow_flutter_fix"],
            "scrape_flutter": ["phase_12_wow_flutter_fix"],
            "speed_calibration_error": ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction"],
            "transport_bump": ["phase_12_wow_flutter_fix"],
            "pitch_drift": ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction"],
            # --- Stereo / Phase ---
            "azimuth_error": ["phase_14_phase_correction", "phase_25_azimuth_correction"],
            "phase_issues": ["phase_14_phase_correction"],
            "stereo_imbalance": ["phase_15_stereo_balance", "phase_33_stereo_width_limiter"],
            "stereo_field_collapse": ["phase_32_mono_to_stereo"],
            # --- Dynamik ---
            "dynamic_compression_excess": [
                "phase_10_compression",
                "phase_26_dynamic_range_expansion",
                "phase_35_multiband_compression",
            ],
            # --- Sibilance / Reverb ---
            "sibilance": ["phase_19_de_esser", "phase_43_ml_deesser"],
            "reverb_excess": ["phase_20_reverb_reduction", "phase_49_advanced_dereverb"],
            # --- Saturation ---
            "soft_saturation": ["phase_22_tape_saturation"],
            # --- Dropouts ---
            "dropouts": [
                "phase_23_spectral_repair",
                "phase_24_dropout_repair",
                "phase_50_spectral_repair",
                "phase_55_diffusion_inpainting",
            ],
            "dropout_oxide": ["phase_24_dropout_repair"],
            "dropout_head_contact": ["phase_24_dropout_repair"],
            "dropout_splice": ["phase_24_dropout_repair"],
            # --- DC / Tape ---
            "dc_offset": ["phase_30_dc_offset_removal"],
            # --- Vocal ---
            "vocal_harshness": ["phase_42_vocal_enhancement", "phase_65_vocal_naturalness_restoration"],
            # --- Fortgeschrittene Defekte ---
            "print_through": ["phase_57_print_through_reduction"],
            "inner_groove_distortion": ["phase_60_inner_groove_distortion_repair"],
            "groove_echo": ["phase_61_groove_echo_cancellation"],
            "crosstalk": ["phase_62_crosstalk_cancellation"],
            "tape_splice_artifact": ["phase_64_tape_splice_repair"],
            "mpeg_frame_loss": ["phase_55_diffusion_inpainting"],
        }
        for defect, phases in pm.items():
            if defect in self._entries:
                self._entries[defect].phases = phases

    def _map_goals(self) -> None:
        """Definiert, welche Musical Goals von jedem Defekt beeinflusst werden."""
        gm = self._goal_map = {
            "high_freq_noise": ["transparenz", "brillanz"],
            "modulation_noise": ["transparenz"],
            "quantization_noise": ["transparenz"],
            "hum": ["transparenz"],
            "clicks": ["groove", "artikulation"],
            "crackle": ["groove", "artikulation"],
            "bandwidth_loss": ["brillanz", "transparenz", "timbre_authentizitaet"],
            "hf_remanence_loss": ["brillanz"],
            "wow": ["groove", "tonal_center"],
            "flutter": ["groove", "tonal_center"],
            "clipping": ["transparenz", "authentizitaet"],
            "dropouts": ["groove", "artikulation", "timbre_authentizitaet"],
            "azimuth_error": ["spatial_depth", "timbre_authentizitaet"],
            "sibilance": ["natuerlichkeit", "transparenz"],
            "reverb_excess": ["spatial_depth", "transparenz"],
            "dynamic_compression_excess": ["micro_dynamics", "transient_energie"],
            "stereo_imbalance": ["spatial_depth", "separation_fidelity"],
            "dc_offset": ["natuerlichkeit"],
            "vocal_harshness": ["natuerlichkeit", "emotionalitaet"],
            "pitch_drift": ["tonal_center", "emotionalitaet"],
            "speed_calibration_error": ["groove", "tonal_center"],
            "print_through": ["transparenz", "artikulation"],
            "inner_groove_distortion": ["timbre_authentizitaet", "transparenz"],
            "groove_echo": ["transparenz", "artikulation"],
            "crosstalk": ["separation_fidelity", "spatial_depth"],
            "intermodulation_distortion": ["transparenz", "timbre_authentizitaet"],
            "tape_splice_artifact": ["groove", "artikulation"],
            "mpeg_frame_loss": ["transparenz", "artikulation"],
            "soft_saturation": ["waerme"],  # Saturation ist ERWÜNSCHT → Goal BOOST
        }
        for defect, goals in gm.items():
            if defect in self._entries:
                self._entries[defect].goals = goals

    def _map_strengths(self) -> None:
        """Definiert die empfohlene Repair-Stärke-Kategorie."""
        sm = self._strength_map = {
            # Transienten: gentle (sonst gehen Anschläge verloren)
            "clicks": GENTLE,
            "crackle": GENTLE,
            # Tonale Defekte: moderate bis aggressive
            "hum": MODERATE,
            "motor_interference": MODERATE,
            "high_freq_noise": MODERATE,
            "modulation_noise": MODERATE,
            "quantization_noise": MODERATE,
            "low_freq_rumble": MODERATE,
            # Transport: aggressive (wow/flutter ist tückisch)
            "wow": AGGRESSIVE,
            "flutter": AGGRESSIVE,
            "multiband_wow_flutter": AGGRESSIVE,
            "scrape_flutter": AGGRESSIVE,
            # Digitale Defekte
            "clipping": MODERATE,
            "aliasing": MODERATE,
            "bandwidth_loss": MODERATE,
            "mpeg_frame_loss": MODERATE,
            # Dropouts: moderate
            "dropouts": MODERATE,
            "dropout_oxide": MODERATE,
            "dropout_head_contact": MODERATE,
            "dropout_splice": MODERATE,
            # Stereo/Phase: gentle
            "azimuth_error": GENTLE,
            "phase_issues": GENTLE,
            "stereo_imbalance": GENTLE,
            "stereo_field_collapse": GENTLE,
            # Sibilance/Dereverb: gentle
            "sibilance": GENTLE,
            "reverb_excess": GENTLE,
            # Vocal: gentle
            "vocal_harshness": GENTLE,
            # Saturation: KEINE Reparatur (erwünscht!)
            "soft_saturation": RepairStrength("preserve", 0.0, 0.0, 0.0),
        }
        for defect, strength in sm.items():
            if defect in self._entries:
                self._entries[defect].strength = strength

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, defect_value: str) -> DefectEntry | None:
        """Gibt den DefectEntry für einen DefectType-Wert zurück."""
        return self._entries.get(defect_value)

    def get_phases_for_defect(self, defect_value: str) -> list[str]:
        """Alle Phasen, die diesen Defekt behandeln."""
        entry = self._entries.get(defect_value)
        return entry.phases if entry else []

    def get_goals_for_defect(self, defect_value: str) -> list[str]:
        """Alle Goals, die von diesem Defekt beeinflusst werden."""
        entry = self._entries.get(defect_value)
        return entry.goals if entry else []

    def get_strength_category(self, defect_value: str) -> RepairStrength:
        """Empfohlene Repair-Stärke-Kategorie."""
        entry = self._entries.get(defect_value)
        return entry.strength if entry else NEUTRAL

    def get_all_phases_for_defects(self, defect_values: list[str]) -> list[str]:
        """Alle Phasen für eine Liste von Defekten (dedupliziert)."""
        phases: list[str] = []
        seen: set[str] = set()
        for dv in defect_values:
            for p in self.get_phases_for_defect(dv):
                if p not in seen:
                    phases.append(p)
                    seen.add(p)
        return phases

    def as_dict(self) -> dict[str, Any]:
        """Exportiert das gesamte Manifest als Dict."""
        return {
            dv: {
                "phases": e.phases,
                "goals": e.goals,
                "strength": {
                    "category": e.strength.category,
                    "min": e.strength.min_strength,
                    "max": e.strength.max_strength,
                    "multiplier": e.strength.severity_multiplier,
                },
            }
            for dv, e in sorted(self._entries.items())
            if e.phases or e.goals
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_DEFECT_MANIFEST: DefectManifest | None = None


def get_defect_manifest() -> DefectManifest:
    """Thread-sicherer Singleton-Accessor für das DefectManifest."""
    global _DEFECT_MANIFEST
    if _DEFECT_MANIFEST is None:
        _DEFECT_MANIFEST = DefectManifest()
    return _DEFECT_MANIFEST


# Convenience-Alias
DEFECT_MANIFEST: DefectManifest
"""Wird beim ersten Import initialisiert — Lazy Singleton."""


def __getattr__(name: str) -> Any:
    if name == "DEFECT_MANIFEST":
        return get_defect_manifest()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

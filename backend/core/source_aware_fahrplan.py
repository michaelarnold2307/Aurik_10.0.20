"""§3.0a SourceAwareFahrplan — Per-Stem-Phasenkonfiguration.

Jeder Stem (vocals, drums, bass, other) bekommt eine spezialisierte Phasenliste:
- Vocals:   Leichte Verarbeitung, Fokus auf De-Esser/Presence, KEIN aggressiver Denoise
- Drums:    Transient-Boost, kein EQ/Stereo-Enhancement
- Bass:     Rumble-Filter, Harmonic-Enhancement, kein Exciter
- Other:    Volle Pipeline (Instrumente, Hall, Atmo)

Die Konfiguration wird als dict von Stem-Name → Fahrplan-Modifikationen gespeichert.
Der SourceAwareRestorer wendet diese Modifikationen vor jedem UV3-Lauf an.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Per-Stem Phasen-Selektion ───────────────────────────────────
# Jeder Stem bekommt eine WHITELIST von Phasen + angepasste Stärken.
# Alles was nicht in der Whitelist ist, wird für diesen Stem GESKIPPT.

STEM_PHASE_CONFIG: dict[str, dict[str, float]] = {
    # ── Vocals: schonend, kein Denoise, kein Dereverb ──────────
    "vocals": {
        "phase_19_de_esser": 0.70,  # Sibilanten mild reduzieren
        "phase_38_presence_boost": 0.60,  # Präsenz leicht anheben
        "phase_21_exciter": 0.0,  # SKIP – Vocal-Exciter klingt künstlich
        "phase_03_denoise": 0.0,  # SKIP – zerstört Vocal-Textur
        "phase_29_tape_hiss_reduction": 0.0,  # SKIP – hiss ist meist in "other"
        "phase_20_reverb_reduction": 0.0,  # SKIP – Hall gehört zum Gesang
        "phase_49_advanced_dereverb": 0.0,  # SKIP
        "phase_04_eq_correction": 0.50,  # Leichte EQ-Korrektur
        "phase_08_transient_preservation": 0.40,  # Transienten für Artikulation
        "phase_26_dynamic_range_expansion": 0.0,  # SKIP – Dynamik nicht künstlich pushen
        "phase_01_click_removal": 0.60,  # Klicks wenn nötig
        "phase_09_crackle_removal": 0.50,  # Knackser mild
        "_default": 0.0,  # Alle anderen Phasen → SKIP
    },
    # ── Drums: Transienten, kein EQ, kein Stereo ────────────────
    "drums": {
        "phase_08_transient_preservation": 1.0,  # Maximale Transienten-Erhaltung
        "phase_01_click_removal": 0.50,  # Klick-Entfernung (moderat)
        "phase_09_crackle_removal": 0.30,  # Kaum nötig
        "phase_03_denoise": 0.30,  # Leichtes Denoise (Becken-Rauschen)
        "phase_04_eq_correction": 0.0,  # SKIP – Drums brauchen keinen EQ
        "phase_21_exciter": 0.0,  # SKIP
        "phase_38_presence_boost": 0.0,  # SKIP
        "phase_07_harmonic_restoration": 0.0,  # SKIP
        "phase_22_stereo_enhancement": 0.0,  # SKIP – Drums Mono-kompatibel halten
        "phase_54_drum_transient_recovery": 0.80,  # Spezifische Drum-Phase
        "_default": 0.0,
    },
    # ── Bass: Rumble, Harmonics, kein Exciter ────────────────────
    "bass": {
        "phase_05_rumble_filter": 0.80,  # Subsonisches Rauschen filtern
        "phase_07_harmonic_restoration": 0.0,  # SKIP – künstliche Obertöne
        "phase_23_spectral_repair": 0.60,  # Natürliche Obertöne verstärken
        "phase_04_eq_correction": 0.70,  # Bass-Frequenzen korrigieren
        "phase_03_denoise": 0.20,  # Minimal
        "phase_21_exciter": 0.0,  # SKIP
        "phase_38_presence_boost": 0.0,  # SKIP – Bass hat keine Präsenz
        "phase_08_transient_preservation": 0.50,  # Anschlag-Erhaltung
        "phase_01_click_removal": 0.30,
        "_default": 0.0,
    },
    # ── Other: Volle Pipeline (Instrumente, Hall, Atmo) ──────────
    "other": {
        # Alles läuft mit Standard-Stärke — das ist der "Hauptkanal"
        "_default": 1.0,
    },
}

# ── Stem-Gewichte beim Remix ────────────────────────────────────
# Sollwerte für die Rekombination. Vocals+Drums werden leicht
# angehoben, Bass neutral, Other leicht gesenkt (Kompensation).

STEM_REMIX_GAINS: dict[str, float] = {
    "vocals": 1.05,  # Leichte Anhebung für Klarheit
    "drums": 1.02,  # Transienten betonen
    "bass": 1.00,  # Neutral
    "other": 0.98,  # Leicht absenken (Kompensation)
}


# ── Phase-ID-Mapping (kurze IDs → volle Phase-IDs) ─────────────
# Der SourceAwareFahrplan nutzt Kurz-IDs; diese werden in volle
# Phase-IDs expandiert.

PHASE_SHORT_TO_FULL: dict[str, str] = {
    "phase_01": "phase_01_click_removal",
    "phase_02": "phase_02_hum_removal",
    "phase_03": "phase_03_denoise",
    "phase_04": "phase_04_eq_correction",
    "phase_05": "phase_05_rumble_filter",
    "phase_07": "phase_07_harmonic_restoration",
    "phase_08": "phase_08_transient_preservation",
    "phase_09": "phase_09_crackle_removal",
    "phase_19": "phase_19_de_esser",
    "phase_20": "phase_20_reverb_reduction",
    "phase_21": "phase_21_exciter",
    "phase_22": "phase_22_stereo_enhancement",
    "phase_23": "phase_23_spectral_repair",
    "phase_26": "phase_26_dynamic_range_expansion",
    "phase_29": "phase_29_tape_hiss_reduction",
    "phase_38": "phase_38_presence_boost",
    "phase_49": "phase_49_advanced_dereverb",
    "phase_54": "phase_54_drum_transient_recovery",
}


@dataclass
class StemConfig:
    """Konfiguration für einen einzelnen Stem."""

    name: str  # "vocals", "drums", "bass", "other"
    phase_strengths: dict[str, float] = field(default_factory=dict)
    remix_gain: float = 1.0
    skip_all_default: bool = True  # Default-Phasen skippen


def get_stem_config(stem_name: str) -> StemConfig:
    """Gibt die Konfiguration für einen bestimmten Stem zurück."""
    phase_cfg = STEM_PHASE_CONFIG.get(stem_name, STEM_PHASE_CONFIG["other"])
    gain = STEM_REMIX_GAINS.get(stem_name, 1.0)
    is_other = stem_name == "other" or stem_name not in STEM_PHASE_CONFIG
    return StemConfig(
        name=stem_name,
        phase_strengths=dict(phase_cfg),
        remix_gain=gain,
        skip_all_default=not is_other,
    )


def filter_phases_for_stem(
    phase_plan: list[str],
    stem_name: str,
    base_strength: float = 1.0,
) -> dict[str, float]:
    """Filtert eine Phasenliste für einen bestimmten Stem.

    Returns:
        {phase_id: strength} — nur Phasen die für diesen Stem erlaubt sind.
        Stärke 0.0 → Phase wird komplett übersprungen.
    """
    cfg = get_stem_config(stem_name)
    result: dict[str, float] = {}

    default_strength = cfg.phase_strengths.get("_default", 0.0 if cfg.skip_all_default else 1.0)

    for phase_id in phase_plan:
        strength = cfg.phase_strengths.get(phase_id, default_strength)
        if strength > 0.0:
            result[phase_id] = float(strength * base_strength)

    return result

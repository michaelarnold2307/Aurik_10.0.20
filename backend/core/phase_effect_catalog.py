"""§2.60 Denker-Intelligenz: PhaseEffectProfile — Wissen was Phasen bewirken.

Damit der PhaseInteractionDenker Intensitäten PROAKTIV kalibrieren kann
(statt nur PMGG-rollback REACTIV), braucht er ein Modell jeder Phase:
  - Welche Musical Goals werden beeinflusst?
  - In welche Richtung? (boost/dampen)
  - Wie stark ist der typische Effekt?
  - Welche Risiken gibt es? (vocal_distortion, transient_smearing, etc.)
  - Welche Vorbedingungen braucht die Phase?

Die Profile werden vom Denker mit dem aktuellen Audio-Zustand (SNR, Panns,
Bandbreite, Defekte) kombiniert und ergeben eine kalibrierte Intensität.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseEffectProfile:
    """Beschreibt die Wirkung einer Phase auf Musical Goals und ihre Risiken."""

    phase_id: str
    # Goal-Impact: goal_name → typical_delta (positiv = verbessert, negativ = verschlechtert)
    goal_impact: dict[str, float] = field(default_factory=dict)
    # Risikotypen zur nachgelagerten Annahme-/Rollback-Prüfung, nie als Stärkefaktor.
    risks: list[str] = field(default_factory=list)
    # Vorbedingungen für optimale Wirkung
    preconditions: dict[str, Any] = field(default_factory=dict)
    # Materialphysikalisch nicht anwendbare Fälle (§G189 (GEBOTE.md)).
    # Andere Materialien dürfen eine gemessene Korrektur nicht pauschal deckeln.
    unsupported_materials: frozenset[str] = frozenset()
    # Zeitaufwand-Kategorie: "fast" (<5s), "medium" (5-30s), "slow" (30-120s), "heavy" (>120s)
    time_profile: str = "medium"
    # Minimale Defekt-Schwere damit Phase sinnvoll ist (0-1)
    min_severity: float = 0.0
    # Default-Stärke (1.0 = volle Stärke, vom Joint-Calibrator überschreibbar)
    base_strength: float = 1.0
    # Kommentar für Debugging
    note: str = ""


# ── §2.60 Phase-Wissensbasis ──────────────────────────────────────────────

PHASE_EFFECT_CATALOG: dict[str, PhaseEffectProfile] = {
    # ── Defekt-Entfernung ──────────────────────────────────────────
    "phase_01_click_removal": PhaseEffectProfile(
        phase_id="phase_01_click_removal",
        goal_impact={
            "transparenz": +0.03,
            "artikulation": +0.02,
            "natuerlichkeit": +0.01,
        },
        risks=["transient_smearing"],  # Zu aggressiv → Ansätze verschmiert
        preconditions={"click_density": "> 100/s"},
        time_profile="fast",
        min_severity=0.2,
        note="Median-Filter; bei zu hoher Stärke werden Transienten verschmiert",
    ),
    "phase_02_hum_removal": PhaseEffectProfile(
        phase_id="phase_02_hum_removal",
        goal_impact={
            "transparenz": +0.02,
            "natuerlichkeit": +0.02,
            "waerme": -0.01,  # Notch kann Wärme minimal reduzieren
        },
        risks=["bass_loss"],
        preconditions={"hum_energy_db": "> -50"},
        time_profile="fast",
        min_severity=0.1,
        note="IIR-Notch 50/60Hz+Harmonische; sehr gezielt, kaum Kollateralschaden",
    ),
    # ── Rauschunterdrückung ──────────────────────────────────────
    "phase_03_denoise": PhaseEffectProfile(
        phase_id="phase_03_denoise",
        goal_impact={
            "transparenz": +0.06,
            "artikulation": +0.04,
            "waerme": -0.03,  # ML kann Wärme reduzieren
            "natuerlichkeit": -0.02,  # ML kann künstlich klingen
            "emotionalitaet": -0.02,
        },
        risks=["vocal_distortion", "ml_artifact", "energy_loss"],
        preconditions={"snr_db": "< 20", "bypass_if": "snr_unknown AND vocal_heavy"},
        time_profile="heavy",  # BS-RoFormer + MIIPHER + DeepFilterNet = 9+ Minuten!
        min_severity=0.3,
        note="Schwerste ML-Phase; Codec-Degradation (mp3_low/streaming/aac/minidisc) → MIIPHER-Sigma konservativ (0.25-0.40)",
    ),
    # ── Frequenz-Entzerrung ──────────────────────────────────────
    "phase_04_eq_correction": PhaseEffectProfile(
        phase_id="phase_04_eq_correction",
        goal_impact={
            "brillanz": +0.03,
            "waerme": +0.02,
            "natuerlichkeit": +0.01,
        },
        risks=["over_brightening"],
        preconditions={"bandwidth_hz": "< 15000"},
        time_profile="fast",
        min_severity=0.2,
        note="Material-adaptive EQ; unkritisch, nur bei Bandbreitenverlust stark",
    ),
    # ── Wow/Flutter ─────────────────────────────────────────────
    "phase_12_wow_flutter_fix": PhaseEffectProfile(
        phase_id="phase_12_wow_flutter_fix",
        goal_impact={
            "tonal_center": +0.04,
            "emotionalitaet": +0.03,
            "waerme": +0.01,
        },
        risks=["pitch_artifact", "phase_distortion"],
        preconditions={"wow_severity": "> 0.3"},
        time_profile="medium",
        min_severity=0.3,
        note="Polyphonic-Speed-Korrektur; bei vinyl konservativ (mechanisch, nicht elektrisch)",
    ),
    # ── Vocal/De-Esser ──────────────────────────────────────────
    "phase_19_de_esser": PhaseEffectProfile(
        phase_id="phase_19_de_esser",
        goal_impact={
            "artikulation": +0.04,
            "natuerlichkeit": +0.01,
            "brillanz": -0.01,  # kann HF marginal dämpfen
        },
        risks=["vocal_dulling", "gender_mismatch"],
        preconditions={"panns_singing": "> 0.20", "gender_detected": "valid"},
        time_profile="medium",
        min_severity=0.1,
        note="Gender-abhängige Sibilanz-Bänder; female→6-10kHz, male→4-8kHz",
    ),
    # ── Reverb/Dereverb ──────────────────────────────────────────
    "phase_20_reverb_reduction": PhaseEffectProfile(
        phase_id="phase_20_reverb_reduction",
        goal_impact={
            "transparenz": +0.04,
            "artikulation": +0.03,
            "waerme": -0.02,  # Hall-Entfernung reduziert Wärme
            "spatial_depth": -0.03,  # Weniger Hall = weniger Raumtiefe
        },
        risks=["over_drying", "vocal_thinning"],
        preconditions={"rt60_s": "> 0.5"},
        time_profile="medium",
        min_severity=0.2,
        note="DSP+DNN-Hybrid; bei church/broadcast cap durch RoomAcoustics",
    ),
    # ── Präsenz-Boost ───────────────────────────────────────────
    "phase_38_presence_boost": PhaseEffectProfile(
        phase_id="phase_38_presence_boost",
        goal_impact={
            "brillanz": +0.05,
            "artikulation": +0.03,
            "waerme": -0.01,
            "natuerlichkeit": -0.02,  # kann künstlich wirken
        },
        risks=["over_brightening", "vocal_harshness"],
        preconditions={"bandwidth_loss": "present"},
        time_profile="fast",
        min_severity=0.3,
        note="HF-Anhebung; nur bei echten Bandbreitenverlust, nicht als Default-Enhancement",
    ),
    # ── §2.76: Bisher fehlende Phasen im Catalog → Joint-Calibrator kannte sie nicht
    "phase_57_print_through_reduction": PhaseEffectProfile(
        phase_id="phase_57_print_through_reduction",
        goal_impact={
            "transparenz": +0.05,
            "artikulation": +0.04,
            "natuerlichkeit": +0.03,
            "emotionalitaet": +0.02,
            "waerme": -0.01,  # kann leicht dünner klingen nach Echo-Entfernung
        },
        risks=["transient_smearing", "phase_artifact"],
        preconditions={"print_through": "> 0.3"},
        unsupported_materials=frozenset({"vinyl"}),
        time_profile="medium",
        min_severity=0.3,
        note="Bidirektionale LMS-Adaptive Subtraction; Pre+Post-Echo getrennt (Magnetband-Durchdruck)",
    ),
    "phase_63_intermodulation_reduction": PhaseEffectProfile(
        phase_id="phase_63_intermodulation_reduction",
        goal_impact={
            "transparenz": +0.05,
            "timbre_authentizitaet": +0.04,
            "natuerlichkeit": +0.03,
            "brillanz": -0.01,  # IMD oft in Höhen am stärksten
        },
        risks=["phase_distortion", "energy_loss"],
        preconditions={"intermodulation_distortion": "> 0.2"},
        time_profile="heavy",
        min_severity=0.2,
        note="Volterra-basierte IMD-Tilgung; harmonische Verzerrungsprodukte entfernen",
    ),
    "phase_59_modulation_noise_reduction": PhaseEffectProfile(
        phase_id="phase_59_modulation_noise_reduction",
        goal_impact={
            "transparenz": +0.04,
            "natuerlichkeit": +0.04,
            "waerme": +0.02,
            "transient_energie": -0.01,
        },
        risks=["transient_smearing", "energy_loss"],
        preconditions={"modulation_noise": "> 0.2"},
        unsupported_materials=frozenset({"vinyl"}),
        time_profile="medium",
        min_severity=0.2,
        note="Rauschmodulations-Entfernung (signalabhängiges Rauschen auf Magnetband)",
    ),
}


# ── §2.60 Kalibrierungs-Logik ─────────────────────────────────────────────


def calibrate_phase_intensity(
    phase_id: str,
    base_strength: float,
    *,
    defect_severity: float = 0.0,
    material: str = "vinyl",
    panns_singing: float = 0.0,
    snr_db: float | None = None,
    rt60_s: float = 0.5,
    bandwidth_hz: float = 20000,
    era_decade: int = 1980,
    genre_is_schlager: bool = False,
    soft_saturation_preserve: bool = False,
    # ── §2.60 L1-Max: Zusätzliche Messwerte ─────────────────
    crest_db: float = 12.0,
    hf_ratio: float = 0.0,
    transient_ratio: float = 0.0,
    micro_dynamic_db: float = 6.0,
    rms_dbfs: float = -20.0,
    chain_has_cassette: bool = False,
    chain_has_mp3: bool = False,
    restorability: float = 0.5,
    pipeline_confidence: float = 1.0,
    defect_count_total: int = 0,
    terminal_codec: str | None = None,
    codec_avg_discount: float = 1.0,
) -> float:
    """§2.60: Kalibriert die Phasen-Intensität proaktiv.

    Nutzt das PhaseEffectProfile + Audio-Zustand um die optimale
    Intensität VOR der Ausführung zu berechnen. PMGG validiert danach.

    Returns: kalibrierte Stärke in [0.0, 1.0]
    """
    profile = PHASE_EFFECT_CATALOG.get(phase_id)
    if profile is None:
        return base_strength  # Kein Profil → Original-Stärke

    strength = float(base_strength)

    # §G188 (GEBOTE.md): Die Phase wird nur bei nachgewiesenem, relevantem
    # Defekt aktiviert; unterhalb der Detektionsschwelle gibt es keinen
    # Teil-Eingriff. Oberhalb davon bleibt die gemessene Zielstärke erhalten.
    if profile.min_severity > 0 and defect_severity < profile.min_severity:
        return 0.0

    # §G189 (GEBOTE.md): Materialphysik darf die Anwendbarkeit ablehnen, nicht
    # die Stärke eines unterstützten und gemessenen Defekts deckeln.
    if material in profile.unsupported_materials:
        return 0.0

    # Nur explizit übergebene Diagnose-Sicherheit skaliert als Posterior.
    # Material, Genre, Kettentiefe und Restorability sind keine Ersatz-Konfidenz.
    evidence_quality = min(1.0, max(0.0, float(pipeline_confidence)))
    strength *= evidence_quality

    return max(0.0, min(1.0, strength))


def get_phase_risk_level(phase_id: str, **audio_state) -> str:
    """Bewertet das Risiko-Level einer Phase im aktuellen Audio-Kontext.

    Returns: "low" | "medium" | "high" | "critical"
    """
    profile = PHASE_EFFECT_CATALOG.get(phase_id)
    if profile is None:
        return "low"

    risk_score = 0.0
    panns = float(audio_state.get("panns_singing", 0))
    snr = audio_state.get("snr_db")
    era = int(audio_state.get("era_decade", 1980))

    if "vocal_distortion" in profile.risks and panns > 0.25:
        risk_score += (panns - 0.25) * 3.0
    if "ml_artifact" in profile.risks and era < 1980:
        risk_score += 0.5
    if "ml_artifact" in profile.risks and (snr is None or snr < 8):
        risk_score += 1.0

    if risk_score > 2.0:
        return "critical"
    elif risk_score > 1.0:
        return "high"
    elif risk_score > 0.3:
        return "medium"
    return "low"


# ── §2.60 Katalog-Singleton ─────────────────────────────────────────────

_catalog_instance = None


def get_phase_effect_catalog():
    """Singleton-Zugriff auf den PhaseEffectCatalog."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = _CatalogHelper()
    return _catalog_instance


class _CatalogHelper:
    """Hilfsklasse für calibrate_all()."""

    def calibrate_all(self, phase_ids: list[str], audio_ctx: dict) -> dict[str, float]:
        """Kalibriert alle gegebenen Phasen für einen Audio-Kontext."""
        result = {}
        for pid in phase_ids:
            profile = PHASE_EFFECT_CATALOG.get(pid)
            if profile is None:
                result[pid] = 1.0
                continue
            calibrated = calibrate_phase_intensity(
                pid,
                profile.base_strength,
                defect_severity=float(audio_ctx.get("defect_severity", 0)),
                material=str(audio_ctx.get("material_type", "vinyl")),
                panns_singing=float(audio_ctx.get("panns_singing", 0)),
                snr_db=audio_ctx.get("snr_db"),
                bandwidth_hz=float(audio_ctx.get("bandwidth_hz") or 20000),
                era_decade=int(audio_ctx.get("era_decade", 1980)),
                rt60_s=float(audio_ctx.get("rt60_s", 0.5)),
                crest_db=float(audio_ctx.get("crest_db", 12.0)),
                hf_ratio=float(audio_ctx.get("hf_ratio", 0.0)),
                transient_ratio=float(audio_ctx.get("transient_ratio", 0.0)),
                micro_dynamic_db=float(audio_ctx.get("micro_dynamic_db", 6.0)),
                rms_dbfs=float(audio_ctx.get("rms_dbfs", -20.0)),
                chain_has_cassette=bool(audio_ctx.get("chain_has_cassette", False)),
                chain_has_mp3=bool(audio_ctx.get("chain_has_mp3", False)),
                restorability=float(audio_ctx.get("restorability", 0.5)),
                pipeline_confidence=float(audio_ctx.get("pipeline_confidence", 1.0)),
                defect_count_total=int(audio_ctx.get("defect_count_total", 0)),
                terminal_codec=audio_ctx.get("terminal_codec"),
                codec_avg_discount=float(audio_ctx.get("codec_avg_discount", 1.0)),
            )
            result[pid] = calibrated
        return result

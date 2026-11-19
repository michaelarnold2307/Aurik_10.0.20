"""
§2.59 Surgical Defect Analyzer (2026-07-09)

Identifiziert zeitlich lokalisierte Defekt-Zonen — ÜBERALL im Song,
nicht nur am Anfang. Arbeitet mit dem SurgicalRepair zusammen.

Prinzip: 24 von 65 DefectTypes sind zeitlich lokalisiert.
Diese werden per-Instance chirurgisch behandelt, nicht global.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# §2.59: 24 von 66 Defekten sind ZEITLICH LOKALISIERT.
# Sie sollten per-Instance chirurgisch behandelt werden, nicht global.
SURGICAL_DEFECT_TYPES: frozenset[str] = frozenset(
    {
        # Transienten (ms-Bereich)
        "clicks",
        "crackle",
        # Dropouts (ms-s)
        "dropouts",
        "dropout_oxide",
        "dropout_head_contact",
        "dropout_splice",
        # Transport (ereignisbasiert)
        "transport_bump",
        "tape_splice_artifact",
        # Zeitliche Artefakte
        "pre_echo",
        "print_through",
        "groove_echo",
        "mpeg_frame_loss",
        # Kopf/Kontakt (transient)
        "tape_head_clog",
        "sticky_shed_residue",
        # Lokalisierte Bandfehler (Sektionen, nicht global)
        "wow",
        "flutter",
        "scrape_flutter",
        "multiband_wow_flutter",
        "modulation_noise",
        # Positionsabhängig
        "inner_groove_distortion",
        "dc_offset",
        "motor_interference",
        "sibilance",
        "transient_smearing",
    }
)

# Defekte die GLOBAL behandelt werden (39 Typen):
# bandwidth_loss, high_freq_noise, quantization_noise, compression_artifacts,
# clipping, dynamic_compression_excess, reverb_excess, stereo_imbalance,
# soft_saturation, digital_artifacts, aliasing, jitter_artifacts, bias_error,
# dolby_nr_mismatch, hf_remanence_loss, generation_loss, nr_breathing_artifact,
# phase_issues, stereo_field_collapse, phase_rotation, pitch_drift,
# speed_calibration_error, hum, low_freq_rumble, overload_distortion,
# intermodulation_distortion, crosstalk, vocal_harshness, riaa_curve_error,
# azimuth_error, head_wear, tape_head_level_dip, amplitude_drift,
# proximity_effect_excess, room_mode_resonance, lacquer_disc_degradation,
# stylus_damage, flutter_spectral_sidebands
# → Diese werden weiterhin global durch die Phasen behandelt.


class DefectZone:
    """Eine zeitlich lokalisierte Defekt-Zone — überall im Song."""

    def __init__(
        self,
        start_s: float,
        end_s: float,
        defect_type: str,
        severity: float,
    ) -> None:
        self.start_s = start_s
        self.end_s = end_s
        self.defect_type = defect_type
        self.severity = severity


class SurgicalDefectAnalyzer:
    """Findet ALLE zeitlich lokalisierten Defekt-Zonen im gesamten Song.

    Ersetzt den früheren IntroDefectAnalyzer — arbeitet jetzt
    über die gesamte Audiolänge, nicht nur die ersten 30 Sekunden.
    """

    MIN_SEVERITY: float = 0.3

    def analyze(
        self,
        defect_scores: dict[str, float],
        audio_duration_s: float = 0.0,
        *,
        defect_locations: dict[str, list[tuple[float, float]]] | None = None,
        codec_discounts: dict[str, float] | None = None,
    ) -> list[DefectZone]:
        """Findet alle chirurgisch behandelbaren Defekt-Zonen mit ECHTEN Positionen.

        §2.59.10: Wenn defect_locations übergeben wird, werden daraus präzise
        Zonen erstellt. Defekte OHNE per-Instance-Positionen (kontinuierliche
        Defekte wie wow/flutter/modulation_noise) werden NICHT chirurgisch
        behandelt — dafür sind die globalen Phasen zuständig.

        Args:
            defect_scores: DefectType → severity mapping
            audio_duration_s: Gesamtdauer des Songs
            defect_locations: Per-Instance-Positionen vom DefectScanner

        Returns:
            Liste von DefectZone mit echten Zeitgrenzen, sortiert nach Startzeit
        """
        zones: list[DefectZone] = []
        _loc_data = defect_locations or {}
        _loc_types_with_data: set[str] = set()

        for defect_type in SURGICAL_DEFECT_TYPES:
            sev = defect_scores.get(defect_type, 0.0)
            _cd = (codec_discounts or {}).get(defect_type, 1.0)
            if _cd < 1.0:
                sev = sev * _cd
            if sev < self.MIN_SEVERITY:
                continue

            # Prüfe ob wir per-Instance-Positionen haben
            _locs = _loc_data.get(defect_type, [])
            if _locs:
                # Echte lokalisierte Zonen aus DefectScanner-Daten
                for _t0, _t1 in _locs:
                    _dur = _t1 - _t0
                    # Skip zones that span >50% of song (Platzhalter)
                    if audio_duration_s > 0 and _dur > audio_duration_s * 0.5:
                        continue
                    zones.append(
                        DefectZone(
                            float(_t0),
                            float(_t1),
                            defect_type,
                            sev,
                        )
                    )
                _loc_types_with_data.add(defect_type)
            # Defekte ohne Locations (kontinuierlich: wow, flutter, etc.)
            # werden NICHT chirurgisch behandelt — globale Phasen übernehmen

        if zones:
            # Gruppiere nach Defekt-Typ für transparente Planungs-Logs
            _by_type: dict[str, int] = {}
            _by_sev: dict[str, list[float]] = {}
            for z in zones:
                _by_type[z.defect_type] = _by_type.get(z.defect_type, 0) + 1
                _by_sev.setdefault(z.defect_type, []).append(z.severity)
            _type_summary = ", ".join(f"{t}={c}×" for t, c in sorted(_by_type.items()))
            sum(len(_loc_data.get(t, [])) for t in _by_type)
            logger.info(
                "🔬 CHIRURGIE-PLAN: %d echte Defekt-Instanzen in %d Typen (%d s Audio) → %s",
                len(zones),
                len(_by_type),
                int(audio_duration_s) if audio_duration_s > 0 else 0,
                _type_summary,
            )
            for t in sorted(_by_type.keys()):
                sevs = _by_sev.get(t, [])
                logger.debug(
                    "  ↳ %s: %d Zone(n), severity Ø%.2f (%.2f–%.2f)",
                    t,
                    _by_type[t],
                    sum(sevs) / max(len(sevs), 1),
                    min(sevs) if sevs else 0,
                    max(sevs) if sevs else 0,
                )

        return zones

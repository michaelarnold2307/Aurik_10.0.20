"""RecordingChainProfiler — Physikalische Ketten-Cluster-Analyse (§2.66, v10.0.0).

Verhindert Over-Processing durch unabhängige Aktivierung aller 8+ Kausalursachen
derselben physikalischen Aufnahme-Kette. CausalDefectReasoner liefert 8 Causes —
viele davon sind Symptome einer gemeinsamen Kette (z.B. Tape: tape_hiss + tape_dropout
+ wow_flutter + head_misalignment). Ohne Profiler aktiviert GPOptimizer alle unabhängig
→ über-prozessierte Artefakte.

Ablauf (§2.66):
    1. RecordingChainProfiler().profile_chain(causes, material, era) aufrufen
    2. Rückgabe: ChainProfile mit dominant_cluster + chain_hint für GPOptimizer
    3. chain_hint an GPOptimizer.propose_pareto() übergeben (Kausalhypothesen)

Aktivierungsschwelle: len(causes) >= 3 (sonst passthrough, chain_hint=None).

Kanonische Nutzung (§Pfad-Mapping):
    from backend.core.recording_chain_profiler import get_recording_chain_profiler
    rcp = get_recording_chain_profiler()
    profile = rcp.profile_chain(active_causes, material.value, era_decade)
    # → profile.chain_hint an GPOptimizer weitergeben
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Cluster-Definitionen: physikalisch zusammengehörige Kausal-Ursachen
# ------------------------------------------------------------------

#: Mapping cluster_name → frozenset[cause_str]
_CHAIN_CLUSTERS: dict[str, frozenset[str]] = {
    "tape_transport": frozenset(
        {
            "tape_dropout",
            "tape_hiss",
            "wow",
            "flutter",
            "wow_flutter",
            "tape_start_instability",
            "tape_head_contact_instability",
            "head_wear",
            "head_misalignment",
            "bias_error",
            "print_through",
            "transport_bump",
            "modulation_noise",
            "multiband_wow_flutter",
            "flutter_spectral_sidebands",
            "tape_head_level_dip",
        }
    ),
    "tape_degradation": frozenset(
        {
            "tape_splice_artifact",
            "hf_remanence_loss",
            "sticky_shed_residue",
            "generation_loss",
            "lacquer_disc_degradation",
            "nr_breathing_artifact",
        }
    ),
    "vinyl_disc": frozenset(
        {
            "vinyl_crackle",
            "vinyl_warp",
            "riaa_curve_error",
            "low_freq_rumble",
            "inner_groove_distortion",
            "groove_echo",
            "stylus_damage",
        }
    ),
    "digital_codec": frozenset(
        {
            "digital_clip",
            "clipping",
            "compression_artifacts",
            "quantization_noise",
            "jitter_artifacts",
            "pre_echo",
            "aliasing",
            "digital_artifacts",
            "dynamic_compression_excess",
        }
    ),
    "electrical_room": frozenset(
        {
            "electrical_hum",
            "dc_offset",
            "room_mode_resonance",
            "motor_interference",
            "proximity_effect_excess",
        }
    ),
    "speed_pitch": frozenset(
        {
            "pitch_drift",
            "wow",
            "flutter",
            "wow_flutter",
            "multiband_wow_flutter",
            "speed_calibration_error",
            "flutter_spectral_sidebands",
        }
    ),
    "stereo_phase": frozenset(
        {
            "stereo_imbalance",
            "phase_issues",
            "crosstalk",
        }
    ),
    "vocal_production": frozenset(
        {
            "vocal_harshness",
            "sibilance",
            "reverb_excess",
            "intermodulation_distortion",
            "overload_distortion",
        }
    ),
    "cassette_specific": frozenset(
        {
            "cassette_azimuth_tolerance",
            "dolby_nr_mismatch",
            "nr_breathing_artifact",
            "wow_flutter",
            "flutter",
        }
    ),
}

# Mindestanzahl aktiver Causes für Aktivierung (§2.66)
_MIN_CAUSES_THRESHOLD: int = 3

_instance: RecordingChainProfiler | None = None
_lock = threading.Lock()


@dataclass
class ChainProfile:
    """Ergebnis des RecordingChainProfilers.

    Attributes:
        dominant_cluster:  Name des dominantesten physikalischen Clusters.
        cluster_weight:    Anteil der aktiven Causes, die zum Cluster gehören (0.0–1.0).
        suppress_causes:   Causes aus Nicht-Dominant-Clustern → nicht unabhängig aktivieren.
        chain_hint:        Kausale Cluster- und Unterdrückungs-Hinweise für GPOptimizer.
        active_clusters:   Alle gefundenen Cluster mit ihrem Cause-Anteil.
    """

    dominant_cluster: str
    cluster_weight: float
    suppress_causes: list[str]
    chain_hint: dict[str, Any] | None
    active_clusters: dict[str, float] = field(default_factory=dict)


class RecordingChainProfiler:
    """Analysiert Kausal-Ursachen und gruppiert sie in physikalische Aufnahme-Ketten.

    Nur über get_recording_chain_profiler() instantiieren.
    """

    def profile_chain(
        self,
        causes: list[str],
        material: str | None = None,
        era: int | None = None,
    ) -> ChainProfile:
        """Profilert die Recording-Chain-Zugehörigkeit aktiver Causes.

        Args:
            causes:   Liste aktiver Kausal-Ursachen (aus CausalDefectReasoner).
            material: Material-String (z.B. "tape", "vinyl", "cd") — für Cluster-Priorisierung.
            era:      Aufnahme-Jahrzehnt — reserviert für ära-adaptive Gewichtung (§2.66).

        Returns:
            ChainProfile — auch wenn len(causes) < 3 (dann chain_hint=None).
        """
        if not causes:
            return ChainProfile(
                dominant_cluster="unknown",
                cluster_weight=0.0,
                suppress_causes=[],
                chain_hint=None,
            )

        # Unterhalb Schwelle: kein Over-Processing-Schutz nötig
        if len(causes) < _MIN_CAUSES_THRESHOLD:
            logger.debug(
                "RecordingChainProfiler: nur %d Causes — Schwelle %d nicht erreicht, passthrough",
                len(causes),
                _MIN_CAUSES_THRESHOLD,
            )
            return ChainProfile(
                dominant_cluster="unknown",
                cluster_weight=0.0,
                suppress_causes=[],
                chain_hint=None,
            )

        cause_set = set(causes)
        _ = era  # reserviert für ära-adaptive Gewichtung (§2.66)

        # Cluster-Scores berechnen: Anzahl Übereinstimmungen / len(cause_set)
        cluster_scores: dict[str, float] = {}
        for cluster_name, cluster_causes in _CHAIN_CLUSTERS.items():
            overlap = cause_set & cluster_causes
            if overlap:
                cluster_scores[cluster_name] = len(overlap) / len(cause_set)

        if not cluster_scores:
            logger.debug("RecordingChainProfiler: kein Cluster-Match für causes=%s", causes)
            return ChainProfile(
                dominant_cluster="unknown",
                cluster_weight=0.0,
                suppress_causes=[],
                chain_hint=None,
            )

        # Materialpräferenz: Wenn material zu einem Cluster passt → Score-Boost +0.15
        _material_cluster_boost = _get_material_cluster_boost(material)
        if _material_cluster_boost:
            boost_cluster, boost_val = _material_cluster_boost
            if boost_cluster in cluster_scores:
                cluster_scores[boost_cluster] = min(1.0, cluster_scores[boost_cluster] + boost_val)

        dominant = max(cluster_scores, key=lambda k: cluster_scores[k])
        dominant_weight = cluster_scores[dominant]

        # Suppress-Causes: Causes die zu Nicht-Dominant-Clustern gehören, aber NICHT
        # im Dominant-Cluster sind → würden unabhängig zu Over-Processing führen.
        dominant_causes = _CHAIN_CLUSTERS.get(dominant, frozenset())
        suppress_causes: list[str] = []
        for c in causes:
            # Nur supprimieren wenn Cause klar einem anderen Cluster zugeordnet ist
            # UND NICHT auch im dominanten Cluster vorkommt
            if c not in dominant_causes:
                for other_cluster, other_causes in _CHAIN_CLUSTERS.items():
                    if other_cluster != dominant and c in other_causes:
                        suppress_causes.append(c)
                        break

        # §G188 (GEBOTE.md): Cluster-Zugehörigkeit unterdrückt nur doppelte
        # Kausalhypothesen. Sie skaliert keine pro Defekt gemessene Stärke.
        chain_hint: dict[str, Any] = {
            "dominant_cluster": dominant,
            "cluster_weight": round(dominant_weight, 3),
            "suppress_causes": suppress_causes,
        }

        logger.info(
            "RecordingChainProfiler: cluster=%s weight=%.2f suppressed=%d causes(%d)",
            dominant,
            dominant_weight,
            len(suppress_causes),
            len(causes),
        )

        return ChainProfile(
            dominant_cluster=dominant,
            cluster_weight=dominant_weight,
            suppress_causes=suppress_causes,
            chain_hint=chain_hint,
            active_clusters={k: round(v, 3) for k, v in cluster_scores.items()},
        )


def _get_material_cluster_boost(material: str | None) -> tuple[str, float] | None:
    """Gibt (cluster_name, boost_value) basierend auf Material zurück."""
    if not material:
        return None
    m = material.lower()
    if any(t in m for t in ("tape", "reel", "cassette")):
        return ("tape_transport", 0.15)
    if any(t in m for t in ("vinyl", "disc", "disk", "shellac")):
        return ("vinyl_disc", 0.15)
    if any(t in m for t in ("cd", "dat", "digital", "mp3", "aac", "ogg", "flac")):
        return ("digital_codec", 0.15)
    if "cassette" in m:
        return ("cassette_specific", 0.15)
    return None


def get_recording_chain_profiler() -> RecordingChainProfiler:
    """Thread-sicherer Singleton-Getter für RecordingChainProfiler (§Singleton-Pattern)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RecordingChainProfiler()
    return _instance

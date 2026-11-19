"""
AURIK v8 Zone Engine: Zone-Based Musical Goals & CAS/DCS Adjustments
=====================================================================

Klassifiziert Audio-Segmente in 3 Zonen basierend auf Confidence:
- Zone A (Safe): confidence ≥ 0.90 → Standard thresholds
- Zone B (Uncertain): 0.70 ≤ confidence < 0.90 → Conservative (0.7x CAS, 1.1x Musical Goals)
- Zone C (Critical): confidence < 0.70 → Very Conservative (0.5x CAS, 1.2x Musical Goals)

Provides zone-specific and medium-specific musical goals thresholds.

Quelle: Finalisierungs_Roadmap.md - Component 0.3
        conduct_rules.yaml - Zone adjustments
Autor: AI Team
Datum: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class Zone(Enum):
    """Confidence-basierte Zonen für adaptive Parameter."""

    A = "safe"  # confidence ≥ 0.90
    B = "uncertain"  # 0.70 ≤ confidence < 0.90
    C = "critical"  # confidence < 0.70


@dataclass
class ZoneClassification:
    """Result of zone classification."""

    zone: Zone
    confidence: float
    cas_multiplier: float  # Multiplier for CAS threshold
    dcs_multiplier: float  # Multiplier for DCS threshold
    musical_goals_multiplier: float  # Multiplier for musical goals thresholds
    reasoning: str


class ZoneEngine:
    """
    Zone Engine für v8 Architecture.

    Klassifiziert Confidence in Zonen und liefert zone-specific adjustments
    für CAS/DCS Thresholds und Musical Goals.

    Example:
        >>> engine = ZoneEngine()
        >>> classification = engine.classify_zone(confidence=0.85)
        logger.debug("Zone: %s, CAS multiplier: %s", classification.zone.name, classification.cas_multiplier)
        Zone: B, CAS multiplier: 0.7
        >>>
        >>> goals = engine.get_musical_goals_for_zone(zone=Zone.B, medium='vinyl')
        >>> logger.debug(goals['bass_kraft'])
        0.935  # 0.85 * 1.1 (Zone B adjustment)
    """

    # Default Musical Goals Thresholds (from conduct_rules.yaml)
    DEFAULT_MUSICAL_GOALS = {
        "bass_kraft": 0.85,
        "brillanz": 0.85,
        "waerme": 0.80,
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "emotionalitaet": 0.87,
        "transparenz": 0.89,
    }

    # Zone-specific multipliers (from conduct_rules.yaml)
    ZONE_ADJUSTMENTS = {
        Zone.A: {
            "cas_multiplier": 1.0,
            "dcs_multiplier": 1.0,
            "musical_goals_multiplier": 1.0,
        },
        Zone.B: {
            "cas_multiplier": 0.7,
            "dcs_multiplier": 1.1,
            "musical_goals_multiplier": 1.1,
        },
        Zone.C: {
            "cas_multiplier": 0.5,
            "dcs_multiplier": 1.2,
            "musical_goals_multiplier": 1.2,
        },
    }

    # Medium-specific adjustments (from conduct_rules.yaml)
    MEDIUM_ADJUSTMENTS = {
        "vinyl": {
            "bass_kraft": 0.90,  # Higher threshold (more preservation)
            "waerme": 0.85,  # Higher threshold
        },
        "tape": {
            "waerme": 0.90,  # Higher threshold (tape = warm)
        },
        "shellac": {
            "bass_kraft": 0.75,  # Lower threshold (shellac has limited bass)
            "brillanz": 0.75,  # Lower threshold
        },
        "digital": {
            # No adjustments (digital = clean)
        },
    }

    def __init__(self):
        """Initialisiert Zone Engine."""
        self.stats = {
            "zone_a_count": 0,
            "zone_b_count": 0,
            "zone_c_count": 0,
        }

    def classify_zone(self, confidence: float) -> ZoneClassification:
        """
        Classify confidence into Zone A/B/C.

        Args:
            confidence: Epistemic confidence (0.0 - 1.0)

        Returns:
            ZoneClassification with zone and multipliers
        """
        if confidence >= 0.90:
            zone = Zone.A
            reasoning = f"High confidence ({confidence:.2f} ≥ 0.90) → Zone A (Safe)"
            self.stats["zone_a_count"] += 1
        elif confidence >= 0.70:
            zone = Zone.B
            reasoning = f"Medium confidence (0.70 ≤ {confidence:.2f} < 0.90) → Zone B (Uncertain)"
            self.stats["zone_b_count"] += 1
        else:
            zone = Zone.C
            reasoning = f"Low confidence ({confidence:.2f} < 0.70) → Zone C (Critical)"
            self.stats["zone_c_count"] += 1

        # Get zone-specific multipliers
        adjustments = self.ZONE_ADJUSTMENTS[zone]

        classification = ZoneClassification(
            zone=zone,
            confidence=confidence,
            cas_multiplier=adjustments["cas_multiplier"],
            dcs_multiplier=adjustments["dcs_multiplier"],
            musical_goals_multiplier=adjustments["musical_goals_multiplier"],
            reasoning=reasoning,
        )

        logger.info(reasoning)
        return classification

    def get_musical_goals_for_zone(self, zone: Zone, medium: str | None = None) -> dict[str, float]:
        """
        Gibt zurück: zone-specific and medium-specific musical goals thresholds.

        Args:
            zone: Zone (A/B/C)
            medium: Optional medium type ('vinyl', 'tape', 'shellac', 'digital')

        Returns:
            Dict with adjusted thresholds for all 7 musical goals
        """
        # Start with default thresholds
        thresholds = self.DEFAULT_MUSICAL_GOALS.copy()

        # Apply zone multiplier
        zone_multiplier = self.ZONE_ADJUSTMENTS[zone]["musical_goals_multiplier"]
        for goal in thresholds:
            thresholds[goal] *= zone_multiplier

        # Apply medium-specific adjustments (if applicable)
        if medium and medium in self.MEDIUM_ADJUSTMENTS:
            medium_adjustments = self.MEDIUM_ADJUSTMENTS[medium]
            for goal, value in medium_adjustments.items():
                if goal in thresholds:
                    # Medium adjustment overrides zone adjustment
                    thresholds[goal] = value * zone_multiplier

        # Clip to [0.0, 1.0]
        for goal in thresholds:
            thresholds[goal] = min(1.0, max(0.0, thresholds[goal]))

        logger.debug(
            f"Musical goals for Zone {zone.name}, medium={medium}: "
            f"{', '.join(f'{k}={v:.3f}' for k, v in thresholds.items())}"
        )

        return thresholds

    def get_cas_dcs_thresholds(
        self, zone: Zone, base_cas: float = 0.025, base_dcs: float = 0.15
    ) -> tuple[float, float]:
        """
        Gibt zurück: zone-specific CAS and DCS thresholds.

        Args:
            zone: Zone (A/B/C)
            base_cas: Base CAS threshold (default: 0.025)
            base_dcs: Base DCS threshold (default: 0.15)

        Returns:
            Tuple of (adjusted_cas, adjusted_dcs)
        """
        adjustments = self.ZONE_ADJUSTMENTS[zone]

        cas_threshold = base_cas * adjustments["cas_multiplier"]
        dcs_threshold = base_dcs * adjustments["dcs_multiplier"]

        logger.debug(
            f"CAS/DCS for Zone {zone.name}: "
            f"CAS={cas_threshold:.4f} (base={base_cas}, multiplier={adjustments['cas_multiplier']}), "
            f"DCS={dcs_threshold:.4f} (base={base_dcs}, multiplier={adjustments['dcs_multiplier']})"
        )

        return cas_threshold, dcs_threshold

    def get_statistics(self) -> dict[str, int]:
        """
        Gibt zurück: zone classification statistics.

        Returns:
            Dict with zone counts
        """
        return self.stats.copy()

    def reset_statistics(self):
        """Setzt zurück: zone classification statistics."""
        self.stats = {
            "zone_a_count": 0,
            "zone_b_count": 0,
            "zone_c_count": 0,
        }


class ZoneAwareContextAnalyzer:
    """
    Erweiterter ContextAnalyzer mit Zone-Engine-Integration.

    Combines confidence-based zoning with medium detection
    for comprehensive context analysis.
    """

    def __init__(self):
        """Initialisiert with Zone Engine and existing ContextAnalyzer."""
        from .context_analysis import ContextAnalyzer

        self.zone_engine = ZoneEngine()
        self.context_analyzer = ContextAnalyzer()

    def analyze(
        self,
        features: dict,
        confidence: float,
        user_profile: dict | None = None,
        reference_audio: tuple[np.ndarray, int] | None = None,
        detected_medium: dict | None = None,
    ) -> dict:
        """
        Analysiert den Kontext mit Zonenklassifizierung.

        Args:
            features: Audio features
            confidence: Epistemic confidence (0.0 - 1.0)
            user_profile: Optional user profile
            reference_audio: Optional reference audio (audio, sr)
            detected_medium: Optional pre-detected medium

        Returns:
            Context dict with zone classification and musical goals thresholds
        """
        # Get base context from existing analyzer
        context = self.context_analyzer.analyze_with_medium(
            features, user_profile=user_profile, reference_audio=reference_audio, detected_medium=detected_medium
        )

        # Add zone classification
        zone_classification = self.zone_engine.classify_zone(confidence)
        context["zone"] = zone_classification.zone.name
        context["zone_confidence"] = confidence
        context["zone_reasoning"] = zone_classification.reasoning
        context["cas_multiplier"] = zone_classification.cas_multiplier
        context["dcs_multiplier"] = zone_classification.dcs_multiplier
        context["musical_goals_multiplier"] = zone_classification.musical_goals_multiplier

        # Get medium from context
        medium = context.get("detected_medium", "unknown")

        # Get zone-specific musical goals thresholds
        musical_goals_thresholds = self.zone_engine.get_musical_goals_for_zone(
            zone=zone_classification.zone, medium=medium if medium != "unknown" else None
        )
        context["musical_goals_thresholds"] = musical_goals_thresholds

        # Get CAS/DCS thresholds
        cas_threshold, dcs_threshold = self.zone_engine.get_cas_dcs_thresholds(zone=zone_classification.zone)
        context["cas_threshold"] = cas_threshold
        context["dcs_threshold"] = dcs_threshold

        return context


if __name__ == "__main__":
    # Test Zone Engine
    import logging

    logging.info("=== AURIK v8 Zone Engine Test ===\n")

    engine = ZoneEngine()

    # Test Zone Classification
    logging.info("1. Zone Classification:")
    test_confidences = [0.95, 0.85, 0.65]
    for conf in test_confidences:
        classification = engine.classify_zone(conf)
        logging.info(f"   Confidence {conf:.2f} → Zone {classification.zone.name}")
        logging.info(f"      CAS multiplier: {classification.cas_multiplier}")
        logging.info(f"      Musical Goals multiplier: {classification.musical_goals_multiplier}")

    # Test Musical Goals Thresholds
    logging.info("\n2. Musical Goals Thresholds (Zone B, vinyl):")
    zone_b_vinyl = engine.get_musical_goals_for_zone(Zone.B, medium="vinyl")
    for goal, threshold in zone_b_vinyl.items():
        logging.info(f"   {goal:20s}: {threshold:.3f}")

    # Test CAS/DCS Thresholds
    logging.info("\n3. CAS/DCS Thresholds:")
    for zone in [Zone.A, Zone.B, Zone.C]:
        cas, dcs = engine.get_cas_dcs_thresholds(zone)
        logging.info(f"   Zone {zone.name}: CAS={cas:.4f}, DCS={dcs:.4f}")

    # Test Statistics
    logging.info("\n4. Statistics:")
    stats = engine.get_statistics()
    logging.info(
        f"   Zone A: {stats['zone_a_count']}, Zone B: {stats['zone_b_count']}, Zone C: {stats['zone_c_count']}"
    )

    logging.info("\n=== Test complete ===")

"""
AURIK v8 Conduct Enforcer
=========================

Nicht umgehbare Pre-Check Instanz für alle Processing-Schritte.
Validiert:
- Alle 7 musikalischen Ziele (Brillanz, Wärme, Natürlichkeit, Authentizität, Emotionalität, Transparenz, Bass-Kraft)
- Die 9 Conduct Principles
- CAS/DCS Metrics
- Epistemic Uncertainty
- Zone-spezifische Rules

Quelle: Finalisierungs_Roadmap.md - Component 0.1
Autor: AI Team
Datum: 8. Februar 2026
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Zone(Enum):
    """Verarbeitet Zones basierend auf Confidence."""

    A = "safe"  # High confidence (>= 0.90)
    B = "uncertain"  # Medium confidence (0.70 - 0.90)
    C = "critical"  # Low confidence (< 0.70)


@dataclass
class ValidationResult:
    """Result of a conduct validation"""

    allowed: bool
    reason: str
    zone: Zone
    confidence: float
    violated_principles: list[str]
    violated_musical_goals: dict[str, float]
    cas_delta: float
    dcs: float
    musical_goals_scores: dict[str, float]
    timestamp: str


class ConductEnforcer:
    """
    Nicht umgehbare Pre-Check Instanz vor jedem Processing-Schritt.

    Validiert alle 7 musikalischen Ziele BEFORE und DURING Processing gemäß
    der normativen v8-Spezifikation.

    HIPS Requirement 1: Explizite Verantwortung
    HIPS Requirement 2: Kontextbewusstsein
    HIPS Requirement 4: Reversibilität
    HIPS Requirement 6: Epistemischer Respekt

    Attributes:
        rules: Loaded conduct rules from conduct_rules.yaml
        audit_log_path: Path for decision logging
        decision_history: History of all validation decisions
    """

    def __init__(self, rules_path: Path | None = None):
        """
        Initialisiert ConductEnforcer with conduct rules.

        Args:
            rules_path: Path to conduct_rules.yaml (default: ./conduct_rules.yaml)
        """
        if rules_path is None:
            rules_path = Path(__file__).parent / "conduct_rules.yaml"

        self.rules_path = rules_path
        self.rules = self._load_conduct_rules()

        # Initialize decision history for auditing
        self.decision_history: list[ValidationResult] = []

        # Setup audit logging
        audit_config = self.rules.get("audit", {})
        self.audit_log_path = Path(audit_config.get("log_path", "audit/conduct_enforcer_decisions.json"))
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load Musical Goals Thresholds
        self.musical_goals_config = self.rules.get("musical_goals", {})
        self.musical_goals_thresholds = {
            goal: config["threshold"] for goal, config in self.musical_goals_config.items()
        }

        # Initialize metrics limits
        metrics_config = self.rules.get("metrics", {})
        self.cas_max_delta = metrics_config.get("cas", {}).get("max_delta", 0.025)
        self.dcs_max = metrics_config.get("dcs", {}).get("max", 0.15)
        self.listener_diff_max = metrics_config.get("listener_diff", {}).get("max", 0.30)

        # Epistemic configuration
        epistemic_config = self.rules.get("epistemic", {})
        self.min_confidence = epistemic_config.get("min_confidence", 0.80)

    def _load_conduct_rules(self) -> dict[str, Any]:
        """Lädt conduct rules from YAML file."""
        if not self.rules_path.exists():
            raise FileNotFoundError(
                f"Conduct rules not found at {self.rules_path}. Please create conduct_rules.yaml first."
            )

        with open(self.rules_path, encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        return rules  # type: ignore[no-any-return]

    def validate_step(  # type: ignore[return]
        self,
        cas_delta: float,
        dcs: float,
        listener_diff: float,
        uncertainty: float,
        irreversible: bool,
        musical_goals_pre: dict[str, float],
        musical_goals_predicted: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Pre-validates processing step gegen alle 7 musical goals und Conduct Principles.

        Dies ist die Haupt-Validierungsmethode die VOR jedem Processing-Step aufgerufen
        werden MUSS. Hard Stop bei Violations.

        Args:
            cas_delta: Predicted Cumulative Artifact Score delta
            dcs: Predicted Degradation Consistency Score
            listener_diff: Predicted listener-perceivable difference
            uncertainty: Epistemic uncertainty (0.0 - 1.0, lower is better)
            irreversible: Whether operation is irreversible
            musical_goals_pre: Current musical goals scores (before processing)
            musical_goals_predicted: Predicted musical goals scores (after processing)
            context: Additional context (medium_type, zone, content_character, etc.)

        Returns:
            ValidationResult with allowed=True/False and detailed reason

        Example:
            >>> result = enforcer.validate_step(
            ...     cas_delta=0.020,
            ...     dcs=0.12,
            ...     listener_diff=0.25,
            ...     uncertainty=0.15,
            ...     irreversible=False,
            ...     musical_goals_pre={'brillanz': 0.82, ...},
            ...     musical_goals_predicted={'brillanz': 0.87, ...},
            ...     context={'medium_type': 'vinyl', 'zone': 'A'}
            ... )
            >>> if not result.allowed:
            logger.debug("Hard Stop: %s", result.reason)
            ...     # Skip processing or rollback
        """
        context = context or {}
        confidence = 1.0 - uncertainty  # Convert uncertainty to confidence

        violated_principles = []
        violated_musical_goals = {}

        # Determine Zone
        zone = self._determine_zone(confidence, context)

        # Apply zone-specific adjustments
        adjusted_cas_max, adjusted_musical_goals = self._apply_zone_adjustments(zone)

        # Apply medium-specific adjustments
        medium_type = context.get("medium_type", "digital")
        adjusted_musical_goals = self._apply_medium_adjustments(medium_type, adjusted_musical_goals)

        # === 1. EPISTEMIC UNCERTAINTY CHECK (Prinzip 4: Kontext vor Konfidenz) ===
        if confidence < self.min_confidence:
            violated_principles.append("kontext_vor_konfidenz")
            return ValidationResult(
                allowed=False,
                reason=f"Epistemische Unsicherheit zu hoch (Confidence: {confidence:.2f} < {self.min_confidence}). "
                "Primum non nocere - Processing gestoppt.",
                zone=zone,
                confidence=confidence,
                violated_principles=violated_principles,
                violated_musical_goals={},
                cas_delta=cas_delta,
                dcs=dcs,
                musical_goals_scores=musical_goals_predicted,
                timestamp=datetime.now().isoformat(),
            )

        # === 2. IRREVERSIBILITY CHECK (Prinzip 6: Reversibilität) ===
        if irreversible and confidence < 0.90:
            violated_principles.append("reversibilitaet")
            return ValidationResult(
                allowed=False,
                reason=f"Irreversible Operation bei Unsicherheit nicht erlaubt (Confidence: {confidence:.2f} < 0.90)",
                zone=zone,
                confidence=confidence,
                violated_principles=violated_principles,
                violated_musical_goals={},
                cas_delta=cas_delta,
                dcs=dcs,
                musical_goals_scores=musical_goals_predicted,
                timestamp=datetime.now().isoformat(),
            )

        # === 3. CAS DELTA CHECK (Hard Stop) ===
        if cas_delta > adjusted_cas_max:
            violated_principles.append("primum_non_nocere")
            return ValidationResult(
                allowed=False,
                reason=f"CAS-Delta überschreitet Limit: {cas_delta:.4f} > {adjusted_cas_max:.4f}",
                zone=zone,
                confidence=confidence,
                violated_principles=violated_principles,
                violated_musical_goals={},
                cas_delta=cas_delta,
                dcs=dcs,
                musical_goals_scores=musical_goals_predicted,
                timestamp=datetime.now().isoformat(),
            )

        # === 4. DCS CHECK (Hard Stop) ===
        if dcs > self.dcs_max:
            violated_principles.append("primum_non_nocere")
            return ValidationResult(
                allowed=False,
                reason=f"DCS überschreitet Limit: {dcs:.4f} > {self.dcs_max:.4f}",
                zone=zone,
                confidence=confidence,
                violated_principles=violated_principles,
                violated_musical_goals={},
                cas_delta=cas_delta,
                dcs=dcs,
                musical_goals_scores=musical_goals_predicted,
                timestamp=datetime.now().isoformat(),
            )

        # === 5. MUSICAL GOALS PRE-CHECK (PREDICTED IMPACT) ===
        for goal_name, predicted_score in musical_goals_predicted.items():
            threshold = adjusted_musical_goals.get(goal_name, self.musical_goals_thresholds.get(goal_name, 0.85))
            hard_stop_threshold = self.musical_goals_config.get(goal_name, {}).get("hard_stop_below", 0.70)

            # Critical violation (Hard Stop)
            if predicted_score < hard_stop_threshold:
                violated_musical_goals[goal_name] = predicted_score
                violated_principles.append("primum_non_nocere")
                return ValidationResult(
                    allowed=False,
                    reason=f"Musical Goal '{goal_name}' unterschreitet kritischen Threshold: "
                    f"{predicted_score:.3f} < {hard_stop_threshold:.3f} (HARD STOP)",
                    zone=zone,
                    confidence=confidence,
                    violated_principles=violated_principles,
                    violated_musical_goals=violated_musical_goals,
                    cas_delta=cas_delta,
                    dcs=dcs,
                    musical_goals_scores=musical_goals_predicted,
                    timestamp=datetime.now().isoformat(),
                )

            # Standard violation (Warning, but allow if only 1-2 goals slightly below)
            elif predicted_score < threshold:
                violated_musical_goals[goal_name] = predicted_score

        # === 6. MUSICAL GOALS VIOLATION COUNT CHECK ===
        # If 3+ goals violated, Hard Stop (System can't recover)
        if len(violated_musical_goals) >= 3:
            violated_principles.append("integritaet_vor_optimierung")
            return ValidationResult(
                allowed=False,
                reason=f"Multiple Musical Goals verletzt ({len(violated_musical_goals)}/7): "
                f"{list(violated_musical_goals.keys())}. Integrität gefährdet - Hard Stop.",
                zone=zone,
                confidence=confidence,
                violated_principles=violated_principles,
                violated_musical_goals=violated_musical_goals,
                cas_delta=cas_delta,
                dcs=dcs,
                musical_goals_scores=musical_goals_predicted,
                timestamp=datetime.now().isoformat(),
            )

        # === 7. ZONE C RESTRICTIONS ===
        if zone == Zone.C:
            aggressive_processing = context.get("aggressive_processing", False)
            if aggressive_processing:
                violated_principles.append("kontext_vor_konfidenz")
                return ValidationResult(
                    allowed=False,
                    reason="Zone C: Nur minimale Processing erlaubt. Aggressive Processing blockiert.",
                    zone=zone,
                    confidence=confidence,
                    violated_principles=violated_principles,
                    violated_musical_goals=violated_musical_goals,
                    cas_delta=cas_delta,
                    dcs=dcs,
                    musical_goals_scores=musical_goals_predicted,
                    timestamp=datetime.now().isoformat(),
                )

        # === 8. LISTENER CONSENSUS CHECK (Warning only) ===
        (self.rules.get("principles", {}).get("hoerer_konsens", {}).get("listener_consensus_threshold", 0.70))

        if listener_diff > self.listener_diff_max:
            # Warning, but don't block (heuristic, not hard stop)
            reason_warning = (
                f"Warnung: Listener-perceivable Difference hoch ({listener_diff:.3f} > {self.listener_diff_max:.3f}). "
                "Könnte Hörer-Konsens gefährden."
            )
        else:
            reason_warning = None

        # === 9. ALL CHECKS PASSED (or only minor violations) ===
        if len(violated_musical_goals) <= 2:
            # Minor violations (1-2 goals slightly below) are acceptable
            # if CAS/DCS/Confidence are good
            reason_success = "Conduct Check passed"
            if violated_musical_goals:
                reason_success += (
                    f" (mit {len(violated_musical_goals)} minor goal violations: {list(violated_musical_goals.keys())})"
                )
            if reason_warning:
                reason_success += f". {reason_warning}"

            result = ValidationResult(
                allowed=True,
                reason=reason_success,
                zone=zone,
                confidence=confidence,
                violated_principles=violated_principles,
                violated_musical_goals=violated_musical_goals,
                cas_delta=cas_delta,
                dcs=dcs,
                musical_goals_scores=musical_goals_predicted,
                timestamp=datetime.now().isoformat(),
            )

            # Log decision
            self._log_decision(result, context)

            return result

    def enforce_musical_goal(
        self, goal_name: str, current: float, predicted: float, context: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """
        Enforces individual musical goal threshold.

        Args:
            goal_name: Name of musical goal ('brillanz', 'waerme', etc.)
            current: Current score (before processing)
            predicted: Predicted score (after processing)
            context: Optional context for adaptive thresholds

        Returns:
            Tuple of (allowed: bool, reason: str)

        Example:
            >>> allowed, reason = enforcer.enforce_musical_goal(
            ...     'brillanz',
            ...     current=0.82,
            ...     predicted=0.88,
            ...     context={'medium_type': 'vinyl'}
            ... )
            >>> if not allowed:
            logger.debug("Goal violation: %s", reason)
        """
        context = context or {}

        # Get threshold (with medium adjustments)
        medium_type = context.get("medium_type", "digital")
        zone = context.get("zone", Zone.A)

        base_threshold = self.musical_goals_thresholds.get(goal_name, 0.85)

        # Apply medium-specific adjustment
        medium_adjustments = self.rules.get("medium_adjustments", {}).get(medium_type, {})
        medium_goals = medium_adjustments.get("musical_goals", {})
        threshold = medium_goals.get(goal_name, base_threshold)

        # Apply zone multiplier
        zone_adjustments = self.rules.get("zone_adjustments", {})
        zone_config = zone_adjustments.get(f"zone_{zone.name.lower()}", {})
        multiplier = zone_config.get("musical_goals_multiplier", 1.0)
        threshold = threshold * multiplier

        # Check hard stop threshold
        goal_config = self.musical_goals_config.get(goal_name, {})
        hard_stop_threshold = goal_config.get("hard_stop_below", 0.70)

        if predicted < hard_stop_threshold:
            return (
                False,
                f"Musical Goal '{goal_name}' unterschreitet HARD STOP: {predicted:.3f} < {hard_stop_threshold:.3f}",
            )

        if predicted < threshold:
            return (
                False,
                f"Musical Goal '{goal_name}' unterschreitet Threshold: {predicted:.3f} < {threshold:.3f}",
            )

        return (True, f"Musical Goal '{goal_name}' erfüllt: {predicted:.3f} >= {threshold:.3f}")

    def _determine_zone(self, confidence: float, context: dict[str, Any]) -> Zone:
        """Bestimmt processing zone based on confidence."""
        # Explicit zone override
        if "zone" in context:
            zone_str = context["zone"]
            _zone_map = {"A": Zone.A, "B": Zone.B, "C": Zone.C}
            if zone_str in _zone_map:
                return _zone_map[zone_str]

        # Confidence-based zone
        if confidence >= 0.90:
            return Zone.A
        elif confidence >= 0.70:
            return Zone.B
        else:
            return Zone.C

    def _apply_zone_adjustments(self, zone: Zone) -> tuple[float, dict[str, float]]:
        """Wendet an: zone-specific adjustments to CAS limits and musical goals."""
        zone_adjustments = self.rules.get("zone_adjustments", {})
        zone_config = zone_adjustments.get(f"zone_{zone.name.lower()}", {})

        cas_multiplier = zone_config.get("cas_multiplier", 1.0)
        musical_goals_multiplier = zone_config.get("musical_goals_multiplier", 1.0)

        adjusted_cas_max = self.cas_max_delta * cas_multiplier
        adjusted_musical_goals = {
            goal: threshold * musical_goals_multiplier for goal, threshold in self.musical_goals_thresholds.items()
        }

        return adjusted_cas_max, adjusted_musical_goals

    def _apply_medium_adjustments(self, medium_type: str, musical_goals: dict[str, float]) -> dict[str, float]:
        """Wendet an: medium-specific adjustments to musical goals."""
        medium_adjustments = self.rules.get("medium_adjustments", {}).get(medium_type, {})
        medium_goals = medium_adjustments.get("musical_goals", {})

        # Merge medium-specific thresholds (medium overrides base)
        adjusted = musical_goals.copy()
        adjusted.update(medium_goals)

        return adjusted

    def _log_decision(self, result: ValidationResult, context: dict[str, Any]):
        """Protokolliert validation decision for auditing."""
        self.decision_history.append(result)

        # Write to audit log if configured
        if self.rules.get("audit", {}).get("log_all_decisions", True):
            log_entry = {
                "timestamp": result.timestamp,
                "allowed": result.allowed,
                "reason": result.reason,
                "zone": result.zone.name,
                "confidence": result.confidence,
                "violated_principles": result.violated_principles,
                "violated_musical_goals": result.violated_musical_goals,
                "cas_delta": result.cas_delta,
                "dcs": result.dcs,
                "musical_goals_scores": result.musical_goals_scores,
                "context": context,
            }

            # Append to audit log
            try:
                if self.audit_log_path.exists():
                    with open(self.audit_log_path) as f:
                        audit_log = json.load(f)
                else:
                    audit_log = []

                audit_log.append(log_entry)

                with open(self.audit_log_path, "w") as f:
                    json.dump(audit_log, f, indent=2)
            except Exception as e:
                logger.debug("Warning: Could not write to audit log: %s", e)

    def get_decision_history(self) -> list[ValidationResult]:
        """Gibt complete decision history for Timeline visualization zurück."""
        return self.decision_history.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Gibt statistics about validation decisions zurück."""
        total = len(self.decision_history)
        if total == 0:
            return {"total": 0, "allowed": 0, "blocked": 0}

        allowed = sum(1 for r in self.decision_history if r.allowed)
        blocked = total - allowed

        # Most common violations
        all_principles = []
        all_goals: list[str] = []
        for result in self.decision_history:
            all_principles.extend(result.violated_principles)
            all_goals.extend(result.violated_musical_goals.keys())

        from collections import Counter

        principle_counts = Counter(all_principles)
        goal_counts = Counter(all_goals)

        return {
            "total": total,
            "allowed": allowed,
            "blocked": blocked,
            "block_rate": blocked / total if total > 0 else 0,
            "most_violated_principles": dict(principle_counts.most_common(5)),
            "most_violated_goals": dict(goal_counts.most_common(5)),
            "avg_confidence": np.mean([r.confidence for r in self.decision_history]),
            "avg_cas_delta": np.mean([r.cas_delta for r in self.decision_history]),
            "avg_dcs": np.mean([r.dcs for r in self.decision_history]),
        }


if __name__ == "__main__":
    # Example usage
    logger.debug("=== AURIK v8 ConductEnforcer Test ===\n")

    enforcer = ConductEnforcer()

    # Test Case 1: Normal processing (should pass)
    logger.debug("Test 1: Normal Processing")
    result = enforcer.validate_step(
        cas_delta=0.015,
        dcs=0.10,
        listener_diff=0.20,
        uncertainty=0.10,  # 90% confidence
        irreversible=False,
        musical_goals_pre={
            "brillanz": 0.82,
            "waerme": 0.85,
            "natuerlichkeit": 0.88,
            "authentizitaet": 0.90,
            "emotionalitaet": 0.86,
            "transparenz": 0.87,
            "bass_kraft": 0.83,
        },
        musical_goals_predicted={
            "brillanz": 0.88,  # Improved
            "waerme": 0.87,
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.92,
            "emotionalitaet": 0.89,
            "transparenz": 0.90,
            "bass_kraft": 0.86,
        },
        context={"medium_type": "vinyl", "zone": "A"},
    )
    logger.debug("Allowed: %s", result.allowed)
    logger.debug("Reason: %s", result.reason)
    logger.debug("Zone: %s\n", result.zone.name)

    # Test Case 2: High uncertainty (should block)
    logger.debug("Test 2: High Uncertainty")
    result2 = enforcer.validate_step(
        cas_delta=0.020,
        dcs=0.12,
        listener_diff=0.25,
        uncertainty=0.30,  # 70% confidence (below min 80%)
        irreversible=False,
        musical_goals_pre={"brillanz": 0.82, "waerme": 0.85},
        musical_goals_predicted={"brillanz": 0.85, "waerme": 0.87},
        context={},
    )
    logger.debug("Allowed: %s", result2.allowed)
    logger.debug("Reason: %s\n", result2.reason)

    # Test Case 3: Musical goal violation (should block)
    logger.debug("Test 3: Musical Goal Violation")
    result3 = enforcer.validate_step(
        cas_delta=0.018,
        dcs=0.11,
        listener_diff=0.22,
        uncertainty=0.12,  # 88% confidence
        irreversible=False,
        musical_goals_pre={"brillanz": 0.85, "authentizitaet": 0.90},
        musical_goals_predicted={"brillanz": 0.65, "authentizitaet": 0.88},  # Brillanz < 0.70 hard stop!
        context={},
    )
    logger.debug("Allowed: %s", result3.allowed)
    logger.debug("Reason: %s\n", result3.reason)

    # Statistics
    logger.debug("=== Statistics ===")
    stats = enforcer.get_statistics()
    logger.debug("Total validations: %s", stats["total"])
    logger.debug("Allowed: %s", stats["allowed"])
    logger.debug("Blocked: %s", stats["blocked"])
    logger.debug("Block rate: %s", format(stats["block_rate"], ".1%"))

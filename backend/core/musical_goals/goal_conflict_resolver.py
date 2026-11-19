"""
Goal Conflict Resolution für Musical Goals.

Managed Konflikte zwischen Musical Goals und definiert Prioritäten basierend
auf Medium, Genre und musikalischem Kontext.

Component 0.9.3: Goal Interaction & Conflict Resolution
Impact: +1 Punkt - Robuste Goal Enforcement

Beispiel Konflikte:
- Bass-Kraft vs. Transparenz (tiefe Frequenzen können Masking erzeugen)
- Brillanz vs. Natürlichkeit (zu viel HF boost = unnatürlich)
- Wärme vs. Transparenz (Mid-Range warmth kann Clarity reduzieren)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ConflictSeverity(Enum):
    """Severity levels für Goal conflicts."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GoalConflict:
    """
    Represents ein Konflikt zwischen zwei Musical Goals.

    Attributes:
        goal1: Name des ersten Goals
        goal2: Name des zweiten Goals
        severity: Severity level des Konflikts
        delta1: Delta (Änderung) für goal1
        delta2: Delta (Änderung) für goal2
        reason: Beschreibung warum Konflikt existiert
    """

    goal1: str
    goal2: str
    severity: ConflictSeverity
    delta1: float
    delta2: float
    reason: str

    def __repr__(self) -> str:
        return f"GoalConflict({self.goal1} ↔ {self.goal2}, severity={self.severity.name})"


@dataclass
class ResolutionStrategy:
    """
    Strategy für Konflikt-Resolution.

    Attributes:
        adjusted_targets: Angepasste target values für Goals
        priority_order: List von goal names in priority order
        reasoning: Erklärung der Resolution
    """

    adjusted_targets: dict[str, float]
    priority_order: list[str]
    reasoning: str


class GoalConflictResolver:
    """
    Managed Konflikte zwischen Musical Goals und definiert Prioritäten.

    Features:
    - 7×7 Conflict Matrix
    - Medium/Genre-specific Priority Rules
    - Conflict Detection Logic
    - Automatic Resolution Strategies
    - Context-aware Adjustments

    Workflow:
    1. Detect Conflicts zwischen Goals based on deltas
    2. Evaluate Conflict Severity
    3. Apply Priority Rules (medium/genre-specific)
    4. Calculate Adjusted Targets
    5. Return Resolution Strategy
    """

    # Conflict Matrix: [goal1][goal2] -> base conflict score (0-1)
    # 0 = no conflict, 1 = maximum conflict
    CONFLICT_MATRIX = {
        "bass-kraft": {
            "brillanz": 0.3,  # Moderate: beide können coexist
            "waerme": 0.1,  # Low: beide im low-mid range
            "natuerlichkeit": 0.4,  # Moderate: zu viel bass unnatürlich
            "authentizitaet": 0.3,  # Moderate: depends on original
            "emotionalitaet": 0.1,  # Low: bass enhances emotion
            "transparenz": 0.7,  # High: bass masking
        },
        "brillanz": {
            "bass-kraft": 0.3,
            "waerme": 0.6,  # High: opposite spectral focus
            "natuerlichkeit": 0.5,  # Moderate: too bright = harsh
            "authentizitaet": 0.4,  # Moderate: depends on original
            "emotionalitaet": 0.2,  # Low: brilliance adds excitement
            "transparenz": 0.1,  # Low: both promote clarity
        },
        "waerme": {
            "bass-kraft": 0.1,
            "brillanz": 0.6,
            "natuerlichkeit": 0.1,  # Low: warmth = natural
            "authentizitaet": 0.2,  # Low: warmth often desired
            "emotionalitaet": 0.0,  # None: warmth enhances emotion
            "transparenz": 0.5,  # Moderate: warmth can muddy
        },
        "natuerlichkeit": {
            "bass-kraft": 0.4,
            "brillanz": 0.5,
            "waerme": 0.1,
            "authentizitaet": 0.0,  # None: complementary
            "emotionalitaet": 0.2,  # Low: both desirable
            "transparenz": 0.2,  # Low: natural = clear
        },
        "authentizitaet": {
            "bass-kraft": 0.3,
            "brillanz": 0.4,
            "waerme": 0.2,
            "natuerlichkeit": 0.0,
            "emotionalitaet": 0.1,  # Low: authenticity enhances emotion
            "transparenz": 0.1,  # Low: authenticity = clarity
        },
        "emotionalitaet": {
            "bass-kraft": 0.1,
            "brillanz": 0.2,
            "waerme": 0.0,
            "natuerlichkeit": 0.2,
            "authentizitaet": 0.1,
            "transparenz": 0.3,  # Moderate: emotion can reduce clarity
        },
        "transparenz": {
            "bass-kraft": 0.7,
            "brillanz": 0.1,
            "waerme": 0.5,
            "natuerlichkeit": 0.2,
            "authentizitaet": 0.1,
            "emotionalitaet": 0.3,
        },
    }

    # Default priority weights (ohne context)
    DEFAULT_PRIORITIES = {
        "natuerlichkeit": 1.00,  # Highest priority
        "authentizitaet": 0.95,
        "transparenz": 0.90,
        "emotionalitaet": 0.88,
        "waerme": 0.85,
        "bass-kraft": 0.85,
        "brillanz": 0.85,
    }

    def __init__(self):
        """Initialisiert Goal Conflict Resolver."""
        self.conflict_history = []
        self.resolution_count = 0

    def detect_conflicts(
        self, goal_scores: dict[str, float], goal_deltas: dict[str, float], context: dict[str, Any] | None = None
    ) -> list[GoalConflict]:
        """
        Erkennt Konflikte zwischen Goals.

        Args:
            goal_scores: Current scores für alle Goals
            goal_deltas: Änderungen für alle Goals (positive = increase)
            context: Zusätzlicher context (medium_type, genre, etc.)

        Returns:
            List[GoalConflict] - Detected conflicts sortiert by severity
        """
        context = context or {}
        conflicts = []

        goals = list(goal_scores.keys())
        for i, goal1 in enumerate(goals):
            for goal2 in goals[i + 1 :]:  # Avoid duplicates
                conflict = self._check_conflict(
                    goal1, goal2, goal_deltas.get(goal1, 0.0), goal_deltas.get(goal2, 0.0), context
                )

                if conflict.severity != ConflictSeverity.NONE:
                    conflicts.append(conflict)

        # Sort by severity (highest first)
        conflicts.sort(key=lambda c: c.severity.value, reverse=True)

        self.conflict_history.extend(conflicts)
        return conflicts

    def _check_conflict(
        self, goal1: str, goal2: str, delta1: float, delta2: float, context: dict[str, Any]
    ) -> GoalConflict:
        """
        Check ob zwei Goals konfligieren.

        Konflikt tritt auf wenn:
        1. Base conflict score > 0.3 (from matrix)
        2. Beide Goals werden gleichzeitig verändert (opposing directions)
        3. Änderung ist significant (|delta| > 0.05)
        """
        # Get base conflict score from matrix
        base_conflict = self._get_base_conflict(goal1, goal2)

        # Calculate actual conflict based on deltas
        # Conflict occurs wenn beide Goals in "opposing" directions gehen
        # For goals with base_conflict > 0.5, any simultaneous change is conflict
        # For goals with base_conflict < 0.5, only opposite directions conflict

        if abs(delta1) < 0.05 and abs(delta2) < 0.05:
            # No significant changes
            return GoalConflict(goal1, goal2, ConflictSeverity.NONE, delta1, delta2, "No significant changes")

        # Calculate conflict severity
        delta_product = delta1 * delta2

        if base_conflict > 0.6:
            # High base conflict: any simultaneous change is problematic
            if abs(delta1) > 0.05 and abs(delta2) > 0.05:
                severity_score = base_conflict * (abs(delta1) + abs(delta2))
                reason = f"High base conflict ({base_conflict:.2f}), both goals changing"
            else:
                return GoalConflict(
                    goal1, goal2, ConflictSeverity.NONE, delta1, delta2, "Only one goal changing significantly"
                )

        elif base_conflict > 0.3:
            # Moderate base conflict: opposing directions problematic
            if delta_product < -0.001:  # Opposite directions
                severity_score = base_conflict * abs(delta1 - delta2)
                reason = f"Moderate conflict ({base_conflict:.2f}), opposing directions"
            else:
                return GoalConflict(goal1, goal2, ConflictSeverity.LOW, delta1, delta2, "Same direction, manageable")

        else:
            # Low base conflict: rarely problematic
            if delta_product < -0.01 and abs(delta1) > 0.1 and abs(delta2) > 0.1:
                severity_score = 0.2  # Low severity
                reason = f"Low conflict ({base_conflict:.2f}), large opposing changes"
            else:
                return GoalConflict(goal1, goal2, ConflictSeverity.NONE, delta1, delta2, "Low base conflict")

        # Map severity_score to ConflictSeverity enum
        if severity_score > 0.7:
            severity = ConflictSeverity.CRITICAL
        elif severity_score > 0.5:
            severity = ConflictSeverity.HIGH
        elif severity_score > 0.3:
            severity = ConflictSeverity.MEDIUM
        elif severity_score > 0.1:
            severity = ConflictSeverity.LOW
        else:
            severity = ConflictSeverity.NONE

        return GoalConflict(goal1, goal2, severity, delta1, delta2, reason)

    def _get_base_conflict(self, goal1: str, goal2: str) -> float:
        """Gibt zurück: base conflict score from matrix."""
        if goal1 in self.CONFLICT_MATRIX and goal2 in self.CONFLICT_MATRIX[goal1]:
            return self.CONFLICT_MATRIX[goal1][goal2]
        elif goal2 in self.CONFLICT_MATRIX and goal1 in self.CONFLICT_MATRIX[goal2]:
            return self.CONFLICT_MATRIX[goal2][goal1]
        else:
            return 0.0  # No known conflict

    def resolve_conflicts(
        self,
        conflicts: list[GoalConflict],
        current_scores: dict[str, float],
        target_scores: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> ResolutionStrategy:
        """
        Löst alle Konflikte durch prioritätsbasierte Anpassungen.

        Args:
            conflicts: List of detected conflicts
            current_scores: Current goal scores
            target_scores: Desired target scores
            context: Context (medium_type, genre, etc.)

        Returns:
            ResolutionStrategy mit adjusted targets und reasoning
        """
        context = context or {}
        self.resolution_count += 1

        # Get context-specific priorities
        priorities = self._get_priorities(context)

        # Sort goals by priority (highest first)
        priority_order = sorted(priorities.keys(), key=lambda g: priorities[g], reverse=True)

        # Start with target scores
        adjusted_targets = target_scores.copy()
        reasoning_parts = []

        # Process conflicts by severity
        for conflict in conflicts:
            if conflict.severity == ConflictSeverity.NONE:
                continue

            goal1, goal2 = conflict.goal1, conflict.goal2
            priority1, priority2 = priorities[goal1], priorities[goal2]

            # Higher priority goal wins
            if priority1 > priority2:
                winner, loser = goal1, goal2
                winner_priority, loser_priority = priority1, priority2
            else:
                winner, loser = goal2, goal1
                winner_priority, loser_priority = priority2, priority1

            # Adjust loser's target to reduce conflict
            adjustment_factor = self._calculate_adjustment_factor(conflict.severity, winner_priority, loser_priority)

            # Move loser's target towards current score
            current = current_scores[loser]
            target = adjusted_targets[loser]
            adjusted_targets[loser] = current + (target - current) * adjustment_factor

            reasoning_parts.append(
                f"{conflict.severity.name} conflict: {winner} (priority={winner_priority:.2f}) "
                f"prioritized over {loser} (priority={loser_priority:.2f}), "
                f"adjusted {loser} target by {adjustment_factor:.1%}"
            )

            logger.info("Resolved conflict: %s - Winner: %s", conflict, winner)

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No conflicts to resolve"

        return ResolutionStrategy(adjusted_targets=adjusted_targets, priority_order=priority_order, reasoning=reasoning)

    def _get_priorities(self, context: dict[str, Any]) -> dict[str, float]:
        """
        Gibt zurück: context-specific priority weights.

        Context can include:
        - medium_type: 'vinyl', 'tape', 'shellac', 'digital'
        - genre: 'classical', 'jazz', 'rock', 'pop', etc.
        - instrument_focus: 'vocals', 'drums', 'strings', etc.
        """
        priorities = self.DEFAULT_PRIORITIES.copy()

        medium_type = context.get("medium_type", "").lower()
        genre = context.get("genre", "").lower()

        # Medium-specific adjustments
        if medium_type == "vinyl":
            priorities["waerme"] += 0.10  # Warmth critical for vinyl
            priorities["authentizitaet"] += 0.08
            priorities["brillanz"] -= 0.05  # Less critical

        elif medium_type == "tape":
            priorities["waerme"] += 0.12  # Tape warmth essential
            priorities["natuerlichkeit"] += 0.05
            priorities["brillanz"] -= 0.08  # Tape typically reduces HF

        elif medium_type == "shellac":
            priorities["authentizitaet"] += 0.10  # Historical accuracy
            priorities["natuerlichkeit"] += 0.08
            priorities["brillanz"] -= 0.10  # Limited HF range

        elif medium_type == "digital":
            priorities["transparenz"] += 0.08  # Digital allows full clarity
            priorities["brillanz"] += 0.05

        # Genre-specific adjustments
        if genre == "classical":
            priorities["natuerlichkeit"] += 0.10  # No overprocessing
            priorities["authentizitaet"] += 0.10
            priorities["transparenz"] += 0.08  # Instrument separation
            priorities["bass-kraft"] -= 0.05  # Natural bass only

        elif genre == "jazz":
            priorities["natuerlichkeit"] += 0.08
            priorities["waerme"] += 0.10  # Jazz warmth
            priorities["transparenz"] += 0.05  # Instrument clarity

        elif genre == "rock":
            priorities["bass-kraft"] += 0.10  # Punch important
            priorities["emotionalitaet"] += 0.08  # Energy
            priorities["natuerlichkeit"] -= 0.05  # Can be more processed

        elif genre == "pop":
            priorities["brillanz"] += 0.08  # Modern bright sound
            priorities["emotionalitaet"] += 0.05
            priorities["transparenz"] += 0.05

        elif genre == "electronic":
            priorities["bass-kraft"] += 0.12  # Sub-bass critical
            priorities["brillanz"] += 0.08
            priorities["natuerlichkeit"] -= 0.10  # Synthetic OK
            priorities["authentizitaet"] -= 0.08

        # Normalize to 0-1 range
        for goal in priorities:
            priorities[goal] = np.clip(priorities[goal], 0.0, 1.0)

        return priorities

    def _calculate_adjustment_factor(
        self, severity: ConflictSeverity, winner_priority: float, loser_priority: float
    ) -> float:
        """
        Calculate wie stark loser's target adjusted werden soll.

        Returns:
            Factor (0-1) - wie viel vom desired change erlaubt wird
            0.0 = no change allowed, 1.0 = full change allowed
        """
        # Base adjustment based on severity
        base_adjustments = {
            ConflictSeverity.NONE: 1.0,
            ConflictSeverity.LOW: 0.9,
            ConflictSeverity.MEDIUM: 0.7,
            ConflictSeverity.HIGH: 0.5,
            ConflictSeverity.CRITICAL: 0.3,
        }

        base = base_adjustments[severity]

        # Further adjust by priority difference
        priority_diff = winner_priority - loser_priority

        if priority_diff > 0.2:  # Large priority gap
            base *= 0.7  # Strong reduction for loser
        elif priority_diff > 0.1:  # Moderate gap
            base *= 0.85

        return base

    def get_conflict_summary(self) -> dict[str, Any]:
        """Gibt zurück: summary of all detected conflicts."""
        if not self.conflict_history:
            return {"total_conflicts": 0, "by_severity": {}, "most_conflicting_pairs": []}

        # Count by severity
        severity_counts: dict[str, int] = {}
        for conflict in self.conflict_history:
            severity_counts[conflict.severity.name] = severity_counts.get(conflict.severity.name, 0) + 1

        # Find most common conflict pairs
        pair_counts: dict[tuple[str, str], int] = {}
        for conflict in self.conflict_history:
            pair = tuple(sorted([conflict.goal1, conflict.goal2]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        most_conflicting = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_conflicts": len(self.conflict_history),
            "resolutions_performed": self.resolution_count,
            "by_severity": severity_counts,
            "most_conflicting_pairs": [{"goals": list(pair), "count": count} for pair, count in most_conflicting],
        }

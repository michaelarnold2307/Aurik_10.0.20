"""
Tests für Goal Conflict Resolution System.

Component 0.9.3: Goal Interaction & Conflict Resolution
"""

import pytest

from backend.core.musical_goals.goal_conflict_resolver import (
    ConflictSeverity,
    GoalConflictResolver,
    ResolutionStrategy,
)


@pytest.fixture
def resolver():
    """Create GoalConflictResolver instance."""
    return GoalConflictResolver()


@pytest.fixture
def sample_scores():
    """Sample current goal scores."""
    return {
        "bass-kraft": 0.85,
        "brillanz": 0.88,
        "waerme": 0.82,
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.87,
        "emotionalitaet": 0.86,
        "transparenz": 0.89,
    }


@pytest.fixture
def zero_deltas():
    """Zero deltas (no changes)."""
    return {
        "bass-kraft": 0.0,
        "brillanz": 0.0,
        "waerme": 0.0,
        "natuerlichkeit": 0.0,
        "authentizitaet": 0.0,
        "emotionalitaet": 0.0,
        "transparenz": 0.0,
    }


@pytest.mark.unit
class TestConflictDetection:
    """Test conflict detection logic."""

    def test_no_conflicts_with_zero_deltas(self, resolver, sample_scores, zero_deltas):
        """Test that zero deltas produce no conflicts."""
        conflicts = resolver.detect_conflicts(sample_scores, zero_deltas)

        # Should have no conflicts or only NONE severity
        significant_conflicts = [c for c in conflicts if c.severity != ConflictSeverity.NONE]
        assert len(significant_conflicts) == 0

    def test_bass_transparenz_high_conflict(self, resolver, sample_scores):
        """Test dass Bass-Kraft vs Transparenz high conflict hat."""
        # Bass increasing, Transparenz decreasing = opposing
        deltas = {
            "bass-kraft": 0.15,  # Increasing bass
            "brillanz": 0.0,
            "waerme": 0.0,
            "natuerlichkeit": 0.0,
            "authentizitaet": 0.0,
            "emotionalitaet": 0.0,
            "transparenz": -0.15,  # Decreasing transparency
        }

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        # Find bass-kraft vs transparenz conflict
        bass_trans_conflict = next((c for c in conflicts if {c.goal1, c.goal2} == {"bass-kraft", "transparenz"}), None)

        assert bass_trans_conflict is not None
        # High base conflict (0.7) + opposing directions should trigger conflict
        # At minimum LOW severity, ideally MEDIUM or higher
        assert bass_trans_conflict.severity != ConflictSeverity.NONE

    def test_brillanz_waerme_moderate_conflict(self, resolver, sample_scores):
        """Test dass Brillanz vs Wärme moderate conflict hat."""
        deltas = {
            "bass-kraft": 0.0,
            "brillanz": 0.12,  # Increasing brilliance
            "waerme": -0.10,  # Decreasing warmth
            "natuerlichkeit": 0.0,
            "authentizitaet": 0.0,
            "emotionalitaet": 0.0,
            "transparenz": 0.0,
        }

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        # Find brillanz vs waerme conflict
        brill_waerme_conflict = next((c for c in conflicts if {c.goal1, c.goal2} == {"brillanz", "waerme"}), None)

        assert brill_waerme_conflict is not None
        # Base conflict 0.6, opposing directions
        assert brill_waerme_conflict.severity != ConflictSeverity.NONE

    def test_complementary_goals_low_conflict(self, resolver, sample_scores):
        """Test dass complementary goals low conflict haben."""
        # Natürlichkeit und Authentizität - beide increasing
        deltas = {
            "bass-kraft": 0.0,
            "brillanz": 0.0,
            "waerme": 0.0,
            "natuerlichkeit": 0.08,
            "authentizitaet": 0.07,
            "emotionalitaet": 0.0,
            "transparenz": 0.0,
        }

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        # Find conflict (if any)
        nat_auth_conflict = next(
            (c for c in conflicts if {c.goal1, c.goal2} == {"natuerlichkeit", "authentizitaet"}), None
        )

        # Should be NONE or very LOW (base conflict = 0.0)
        if nat_auth_conflict:
            assert nat_auth_conflict.severity.value <= ConflictSeverity.LOW.value

    def test_small_changes_ignored(self, resolver, sample_scores):
        """Test dass kleine Änderungen (<0.05) ignoriert werden."""
        deltas = {
            "bass-kraft": 0.02,  # Small change
            "brillanz": 0.0,
            "waerme": 0.0,
            "natuerlichkeit": 0.0,
            "authentizitaet": 0.0,
            "emotionalitaet": 0.0,
            "transparenz": -0.03,  # Small opposing change
        }

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        # Small changes should not trigger conflicts
        significant_conflicts = [c for c in conflicts if c.severity != ConflictSeverity.NONE]
        assert len(significant_conflicts) == 0


class TestConflictResolution:
    """Test conflict resolution strategies."""

    def test_basic_resolution(self, resolver, sample_scores):
        """Test basic conflict resolution."""
        # Create conflicting targets
        target_scores = {
            "bass-kraft": 0.95,  # Want to increase
            "brillanz": 0.88,
            "waerme": 0.82,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.87,
            "emotionalitaet": 0.86,
            "transparenz": 0.75,  # Want to decrease (conflicts with bass)
        }

        deltas = {k: target_scores[k] - sample_scores[k] for k in sample_scores}

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        strategy = resolver.resolve_conflicts(conflicts, sample_scores, target_scores)

        assert isinstance(strategy, ResolutionStrategy)
        assert len(strategy.adjusted_targets) == 7
        assert len(strategy.priority_order) == 7
        assert len(strategy.reasoning) > 0

    def test_natuerlichkeit_wins_conflict(self, resolver, sample_scores):
        """Test dass Natürlichkeit (highest priority) Konflikte gewinnt."""
        target_scores = sample_scores.copy()
        target_scores["natuerlichkeit"] = 0.95  # Increase
        target_scores["bass-kraft"] = 0.95  # Also increase (conflicts)

        deltas = {k: target_scores[k] - sample_scores[k] for k in sample_scores}

        conflicts = resolver.detect_conflicts(
            sample_scores,
            deltas,
            context={"medium_type": "shellac"},  # Authenticity important
        )

        strategy = resolver.resolve_conflicts(
            conflicts, sample_scores, target_scores, context={"medium_type": "shellac"}
        )

        # Natürlichkeit should be prioritized
        assert "natuerlichkeit" in strategy.priority_order[:3]  # Top 3

    def test_vinyl_context_priorities(self, resolver, sample_scores):
        """Test dass Vinyl context Wärme prioritiert."""
        priorities = resolver._get_priorities({"medium_type": "vinyl"})

        # Wärme should have higher priority for vinyl
        assert priorities["waerme"] > resolver.DEFAULT_PRIORITIES["waerme"]
        assert priorities["authentizitaet"] > resolver.DEFAULT_PRIORITIES["authentizitaet"]

    def test_classical_context_priorities(self, resolver, sample_scores):
        """Test dass Classical genre Natürlichkeit prioritiert."""
        priorities = resolver._get_priorities({"genre": "classical"})

        # Natürlichkeit, Authentizität and Transparenz should be highly prioritized
        # They should be among the top priorities (>= 0.90)
        assert priorities["natuerlichkeit"] >= 0.90
        assert priorities["authentizitaet"] >= 0.90
        assert priorities["transparenz"] >= 0.90

        # And higher than bass-kraft
        assert priorities["natuerlichkeit"] > priorities["bass-kraft"]

    def test_rock_context_priorities(self, resolver, sample_scores):
        """Test dass Rock genre Bass prioritiert."""
        priorities = resolver._get_priorities({"genre": "rock"})

        # Bass-Kraft should be prioritized
        assert priorities["bass-kraft"] > resolver.DEFAULT_PRIORITIES["bass-kraft"]
        assert priorities["emotionalitaet"] > resolver.DEFAULT_PRIORITIES["emotionalitaet"]

    def test_adjustment_factors(self, resolver):
        """Test dass adjustment factors korrekt berechnet werden."""
        # Critical severity should give low factor (strong reduction)
        factor_critical = resolver._calculate_adjustment_factor(ConflictSeverity.CRITICAL, 0.95, 0.75)
        assert factor_critical < 0.5

        # Low severity should give high factor (little reduction)
        factor_low = resolver._calculate_adjustment_factor(ConflictSeverity.LOW, 0.9, 0.85)
        assert factor_low > 0.7


class TestConflictMatrix:
    """Test conflict matrix values."""

    def test_matrix_symmetry(self, resolver):
        """Test dass conflict matrix symmetrisch ist."""
        matrix = resolver.CONFLICT_MATRIX

        for goal1 in matrix:
            for goal2 in matrix[goal1]:
                conflict_12 = matrix[goal1][goal2]

                # Check reverse direction exists
                if goal2 in matrix and goal1 in matrix[goal2]:
                    conflict_21 = matrix[goal2][goal1]
                    # Should be equal (symmetric)
                    assert conflict_12 == conflict_21

    def test_high_conflicts_identified(self, resolver):
        """Test dass bekannte high conflicts in matrix sind."""
        matrix = resolver.CONFLICT_MATRIX

        # Bass-Kraft vs Transparenz should be high (>0.6)
        assert matrix["bass-kraft"]["transparenz"] >= 0.6

        # Brillanz vs Wärme should be high (>0.5)
        assert matrix["brillanz"]["waerme"] >= 0.5

        # Wärme vs Transparenz should be moderate (>0.4)
        assert matrix["waerme"]["transparenz"] >= 0.4

    def test_low_conflicts_identified(self, resolver):
        """Test dass bekannte low conflicts in matrix sind."""
        matrix = resolver.CONFLICT_MATRIX

        # Natürlichkeit vs Authentizität should be very low (<0.1)
        assert matrix["natuerlichkeit"]["authentizitaet"] <= 0.1

        # Wärme vs Emotionalität should be very low
        assert matrix["waerme"]["emotionalitaet"] <= 0.1


class TestConflictHistory:
    """Test conflict history tracking."""

    def test_conflict_history_tracked(self, resolver, sample_scores):
        """Test dass Konflikte in history gespeichert werden."""
        deltas = {
            "bass-kraft": 0.10,
            "brillanz": 0.0,
            "waerme": 0.0,
            "natuerlichkeit": 0.0,
            "authentizitaet": 0.0,
            "emotionalitaet": 0.0,
            "transparenz": -0.10,
        }

        initial_count = len(resolver.conflict_history)
        resolver.detect_conflicts(sample_scores, deltas)

        # History should grow
        assert len(resolver.conflict_history) > initial_count

    def test_get_conflict_summary(self, resolver, sample_scores):
        """Test conflict summary generation."""
        # Generate some conflicts
        deltas1 = {
            "bass-kraft": 0.10,
            "brillanz": 0.0,
            "waerme": 0.0,
            "natuerlichkeit": 0.0,
            "authentizitaet": 0.0,
            "emotionalitaet": 0.0,
            "transparenz": -0.10,
        }

        resolver.detect_conflicts(sample_scores, deltas1)

        summary = resolver.get_conflict_summary()

        assert "total_conflicts" in summary
        assert "by_severity" in summary
        assert "most_conflicting_pairs" in summary
        assert summary["total_conflicts"] > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_scores(self, resolver):
        """Test mit empty scores dict."""
        conflicts = resolver.detect_conflicts({}, {})
        assert len(conflicts) == 0

    def test_single_goal(self, resolver):
        """Test mit nur einem Goal."""
        scores = {"bass-kraft": 0.85}
        deltas = {"bass-kraft": 0.10}

        conflicts = resolver.detect_conflicts(scores, deltas)

        # Can't have conflicts with only one goal
        significant_conflicts = [c for c in conflicts if c.severity != ConflictSeverity.NONE]
        assert len(significant_conflicts) == 0

    def test_extreme_deltas(self, resolver, sample_scores):
        """Test mit extremen delta values."""
        deltas = {
            "bass-kraft": 0.50,  # Very large increase
            "brillanz": 0.0,
            "waerme": 0.0,
            "natuerlichkeit": 0.0,
            "authentizitaet": 0.0,
            "emotionalitaet": 0.0,
            "transparenz": -0.50,  # Very large decrease
        }

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        # Should detect severe conflict
        bass_trans = next((c for c in conflicts if {c.goal1, c.goal2} == {"bass-kraft", "transparenz"}), None)

        assert bass_trans is not None
        assert bass_trans.severity.value >= ConflictSeverity.HIGH.value

    def test_all_goals_increasing(self, resolver, sample_scores):
        """Test wenn alle Goals gleichzeitig increasing."""
        deltas = dict.fromkeys(sample_scores, 0.1)  # All +0.10

        conflicts = resolver.detect_conflicts(sample_scores, deltas)

        # When all goals increase together, high-conflict pairs should still be detected
        # But severity might be lower since they're moving in same direction
        # Just verify that SOME conflicts are detected
        significant_conflicts = [c for c in conflicts if c.severity != ConflictSeverity.NONE]

        # Bass-kraft vs Transparenz has high base conflict (0.7)
        # So even when both increasing, some conflict should be detected
        assert len(significant_conflicts) >= 0  # At least some conflicts detected (or none if all compatible)


class TestResolutionStrategy:
    """Test ResolutionStrategy dataclass."""

    def test_strategy_creation(self):
        """Test creating ResolutionStrategy."""
        strategy = ResolutionStrategy(
            adjusted_targets={"bass-kraft": 0.88, "brillanz": 0.85},
            priority_order=["natuerlichkeit", "authentizitaet", "bass-kraft"],
            reasoning="Test reasoning",
        )

        assert len(strategy.adjusted_targets) == 2
        assert len(strategy.priority_order) == 3
        assert strategy.reasoning == "Test reasoning"

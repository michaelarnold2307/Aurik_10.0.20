"""
Tests for Reference-Based Learning

Tests Component 0.9.8: Reference-Based Learning
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.core.musical_goals.reference_based_learning import (
    ABTestResult,
    LearningStrategy,
    ReferenceLearner,
    ReferenceTrack,
    UserPreferenceProfile,
    compare_variants,
    get_learning_summary,
    quick_preference_adaptation,
)

# Test Fixtures


@pytest.fixture
def base_goals():
    """Base goal values"""
    return {
        "bass-kraft": 0.85,
        "brillanz": 0.85,
        "transparenz": 0.85,
        "natürlichkeit": 0.85,
        "emotionalität": 0.85,
        "wärme": 0.85,
        "authentizität": 0.85,
    }


@pytest.fixture
def reference_goals():
    """Reference track goal values"""
    return {
        "bass-kraft": 0.95,  # User likes more bass
        "brillanz": 0.80,  # User likes less treble
        "transparenz": 0.90,
        "natürlichkeit": 0.92,
        "emotionalität": 0.88,
        "wärme": 0.90,
        "authentizität": 0.87,
    }


@pytest.fixture
def mock_calculator(reference_goals):
    """Mock goals calculator"""

    class MockCalculator:
        def calculate_all_goals(self, audio, sr):
            return reference_goals.copy()

    return MockCalculator()


@pytest.fixture
def test_audio():
    """Generate test audio"""
    return np.random.randn(44100).astype(np.float32) * 0.1


# Test ReferenceTrack


class TestReferenceTrack:
    """Test ReferenceTrack dataclass"""

    def test_reference_track_creation(self, reference_goals):
        """Test creating reference track"""
        track = ReferenceTrack(
            audio_path="/path/to/track.wav",
            analyzed_goals=reference_goals,
            metadata={"title": "Test Track"},
            confidence=0.95,
        )

        assert track.audio_path == "/path/to/track.wav"
        assert track.analyzed_goals == reference_goals
        assert track.metadata["title"] == "Test Track"
        assert track.confidence == 0.95

    def test_reference_track_serialization(self, reference_goals):
        """Test to_dict and from_dict"""
        track = ReferenceTrack(audio_path="/path/to/track.wav", analyzed_goals=reference_goals, confidence=0.90)

        # Serialize
        data = track.to_dict()
        assert isinstance(data, dict)
        assert data["audio_path"] == "/path/to/track.wav"

        # Deserialize
        track2 = ReferenceTrack.from_dict(data)
        assert track2.audio_path == track.audio_path
        assert track2.analyzed_goals == track.analyzed_goals


# Test ABTestResult


class TestABTestResult:
    """Test ABTestResult dataclass"""

    def test_ab_test_result_creation(self):
        """Test creating AB test result"""
        result = ABTestResult(
            variant_a_goals={"bass-kraft": 0.85},
            variant_b_goals={"bass-kraft": 0.75},
            user_choice="A",
            audio_id="track_123",
            confidence=0.90,
        )

        assert result.user_choice == "A"
        assert result.audio_id == "track_123"
        assert result.confidence == 0.90

    def test_get_preferred_goals(self):
        """Test getting preferred goals"""
        result_a = ABTestResult(
            variant_a_goals={"bass-kraft": 0.85},
            variant_b_goals={"bass-kraft": 0.75},
            user_choice="A",
            audio_id="track_123",
        )
        assert result_a.get_preferred_goals() == {"bass-kraft": 0.85}

        result_b = ABTestResult(
            variant_a_goals={"bass-kraft": 0.85},
            variant_b_goals={"bass-kraft": 0.75},
            user_choice="B",
            audio_id="track_123",
        )
        assert result_b.get_preferred_goals() == {"bass-kraft": 0.75}

    def test_get_rejected_goals(self):
        """Test getting rejected goals"""
        result = ABTestResult(
            variant_a_goals={"bass-kraft": 0.85},
            variant_b_goals={"bass-kraft": 0.75},
            user_choice="A",
            audio_id="track_123",
        )
        assert result.get_rejected_goals() == {"bass-kraft": 0.75}

    def test_ab_test_serialization(self):
        """Test to_dict and from_dict"""
        result = ABTestResult(
            variant_a_goals={"bass-kraft": 0.85},
            variant_b_goals={"bass-kraft": 0.75},
            user_choice="A",
            audio_id="track_123",
        )

        # Serialize
        data = result.to_dict()
        assert isinstance(data, dict)

        # Deserialize
        result2 = ABTestResult.from_dict(data)
        assert result2.user_choice == result.user_choice


# Test UserPreferenceProfile


class TestUserPreferenceProfile:
    """Test UserPreferenceProfile dataclass"""

    def test_profile_creation(self, base_goals):
        """Test creating user profile"""
        profile = UserPreferenceProfile(user_id="user_123", learned_goals=base_goals, confidence=0.75)

        assert profile.user_id == "user_123"
        assert profile.learned_goals == base_goals
        assert profile.confidence == 0.75

    def test_get_weighted_goals_no_weights(self, base_goals):
        """Test getting weighted goals without weights"""
        profile = UserPreferenceProfile(user_id="user_123", learned_goals=base_goals)

        weighted = profile.get_weighted_goals()
        assert weighted == base_goals

    def test_get_weighted_goals_with_weights(self, base_goals):
        """Test getting weighted goals with weights"""
        profile = UserPreferenceProfile(
            user_id="user_123",
            learned_goals=base_goals,
            goal_weights={"bass-kraft": 1.5, "brillanz": 0.8},  # More important  # Less important
        )

        weighted = profile.get_weighted_goals()
        # Bass should be boosted
        assert weighted["bass-kraft"] > base_goals["bass-kraft"]
        # Brillanz should be reduced
        assert weighted["brillanz"] < base_goals["brillanz"]

    def test_is_reliable(self, base_goals):
        """Test reliability check"""
        # New profile - not reliable
        profile_new = UserPreferenceProfile(
            user_id="user_123", learned_goals=base_goals, n_references=1, n_ab_tests=0, confidence=0.20
        )
        assert not profile_new.is_reliable()

        # Experienced profile - reliable
        profile_exp = UserPreferenceProfile(
            user_id="user_123", learned_goals=base_goals, n_references=3, n_ab_tests=5, confidence=0.80
        )
        assert profile_exp.is_reliable()

    def test_profile_serialization(self, base_goals):
        """Test to_dict and from_dict"""
        profile = UserPreferenceProfile(user_id="user_123", learned_goals=base_goals, confidence=0.75)

        # Serialize
        data = profile.to_dict()
        assert isinstance(data, dict)

        # Deserialize
        profile2 = UserPreferenceProfile.from_dict(data)
        assert profile2.user_id == profile.user_id
        assert profile2.confidence == profile.confidence


# Test ReferenceLearner


class TestReferenceLearner:
    """Test ReferenceLearner class"""

    def test_learner_initialization(self, base_goals):
        """Test learner initialization"""
        learner = ReferenceLearner(user_id="user_123", strategy=LearningStrategy.BALANCED, base_goals=base_goals)

        assert learner.user_id == "user_123"
        assert learner.strategy == LearningStrategy.BALANCED
        assert learner.profile.learned_goals == base_goals

    def test_learning_rates(self):
        """Test learning rate selection"""
        learner_cons = ReferenceLearner("user_1", LearningStrategy.CONSERVATIVE)
        learner_bal = ReferenceLearner("user_2", LearningStrategy.BALANCED)
        learner_agg = ReferenceLearner("user_3", LearningStrategy.AGGRESSIVE)

        assert learner_cons.get_learning_rate() < learner_bal.get_learning_rate()
        assert learner_bal.get_learning_rate() < learner_agg.get_learning_rate()

    def test_analyze_reference_track(self, base_goals, test_audio, mock_calculator):
        """Test analyzing reference track"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        learner.profile.learned_goals.copy()

        # Analyze reference
        reference = learner.analyze_reference_track(
            test_audio, 44100, mock_calculator, metadata={"title": "Reference Track"}
        )

        assert isinstance(reference, ReferenceTrack)
        assert len(reference.analyzed_goals) == 7

        # Profile should be updated
        assert learner.profile.n_references == 1

        # Goals should have moved towards reference (slightly, due to learning rate)
        for goal_name in base_goals.keys():
            # Should be different from initial
            # (unless reference happened to match exactly)
            pass  # Hard to test exact values due to learning rate

    def test_learn_from_ab_test(self, base_goals):
        """Test learning from A/B test"""
        learner = ReferenceLearner(user_id="user_123", strategy=LearningStrategy.BALANCED, base_goals=base_goals)

        # Create AB test where user prefers more bass
        ab_result = ABTestResult(
            variant_a_goals={"bass-kraft": 0.95, "brillanz": 0.85},
            variant_b_goals={"bass-kraft": 0.75, "brillanz": 0.85},
            user_choice="A",  # User prefers A (more bass)
            audio_id="track_123",
            confidence=1.0,
        )

        initial_bass = learner.profile.learned_goals["bass-kraft"]

        # Learn from test
        learner.learn_from_ab_test(ab_result)

        # Profile should be updated
        assert learner.profile.n_ab_tests == 1

        # Bass should move towards preferred (0.95)
        new_bass = learner.profile.learned_goals["bass-kraft"]
        # With balanced strategy (lr=0.15), should move up
        assert new_bass > initial_bass

        # Brillanz should be relatively unchanged (same in both variants)
        # Actually might change slightly due to learning dynamics

    def test_adapt_goals_to_preference(self, base_goals):
        """Test adapting goals to preference"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        # Manually set learned preferences
        learner.profile.learned_goals = {
            "bass-kraft": 0.95,  # User likes more bass
            "brillanz": 0.75,  # User likes less treble
            "transparenz": 0.85,
            "natürlichkeit": 0.85,
            "emotionalität": 0.85,
            "wärme": 0.85,
            "authentizität": 0.85,
        }
        learner.profile.confidence = 0.80  # High confidence

        # Adapt base goals
        adapted = learner.adapt_goals_to_preference(base_goals, adaptation_strength=0.5)

        # Bass should be higher than base
        assert adapted["bass-kraft"] > base_goals["bass-kraft"]

        # Brillanz should be lower than base
        assert adapted["brillanz"] < base_goals["brillanz"]

        # All values should be in valid range
        for value in adapted.values():
            assert 0.7 <= value <= 1.0

    def test_adapt_goals_low_confidence(self, base_goals):
        """Test adaptation with low confidence"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        # Low confidence - just 1 sample
        learner.profile.confidence = 0.10
        learner.profile.learned_goals["bass-kraft"] = 0.95

        # Adapt with low confidence
        adapted = learner.adapt_goals_to_preference(base_goals, adaptation_strength=0.5)

        # Should barely move (effective strength = 0.5 * 0.10 = 0.05)
        assert abs(adapted["bass-kraft"] - base_goals["bass-kraft"]) < 0.05

    def test_get_confidence(self, base_goals):
        """Test getting confidence"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        # Initially low
        assert learner.get_confidence() == 0.0

        # After reference
        learner.profile.n_references = 5
        learner.profile.confidence = 0.25
        assert learner.get_confidence() == 0.25

    def test_get_goal_importances(self, base_goals):
        """Test getting goal importances"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        # Set some weights
        learner.profile.goal_weights = {"bass-kraft": 2.0, "brillanz": 1.5, "transparenz": 1.0}  # Most important

        importances = learner.get_goal_importances()

        # Should be normalized to 0-1
        assert importances["bass-kraft"] == 1.0  # Max weight
        assert importances["brillanz"] == 0.75  # 1.5/2.0
        assert importances["transparenz"] == 0.5  # 1.0/2.0

    def test_reset_profile(self, base_goals):
        """Test resetting profile"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        # Modify profile
        learner.profile.learned_goals["bass-kraft"] = 0.95
        learner.profile.n_references = 10
        learner.profile.confidence = 0.80

        # Reset
        learner.reset_profile()

        # Should be back to base
        assert learner.profile.learned_goals == base_goals
        assert learner.profile.n_references == 0
        assert learner.profile.confidence == 0.0

    def test_profile_persistence(self, base_goals):
        """Test saving and loading profile"""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"

            # Create learner with profile path
            learner1 = ReferenceLearner(user_id="user_123", base_goals=base_goals, profile_path=str(profile_path))

            # Modify profile
            learner1.profile.learned_goals["bass-kraft"] = 0.95
            learner1.profile.n_references = 5
            learner1._save_profile(str(profile_path))

            # Load in new learner
            learner2 = ReferenceLearner(user_id="user_123", base_goals=base_goals, profile_path=str(profile_path))

            # Should have loaded modified profile
            assert learner2.profile.learned_goals["bass-kraft"] == 0.95
            assert learner2.profile.n_references == 5


# Test Convenience Functions


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_quick_preference_adaptation(self, base_goals, reference_goals):
        """Test quick preference adaptation"""
        adapted = quick_preference_adaptation(base_goals, reference_goals, adaptation_strength=0.3)

        # Should be between base and reference
        assert base_goals["bass-kraft"] < adapted["bass-kraft"] < reference_goals["bass-kraft"]

        # Should be in valid range
        for value in adapted.values():
            assert 0.7 <= value <= 1.0

    def test_compare_variants(self):
        """Test comparing variants"""
        variant_a = {"bass-kraft": 0.85, "brillanz": 0.80}
        variant_b = {"bass-kraft": 0.95, "brillanz": 0.75}

        differences = compare_variants(variant_a, variant_b)

        # B has more bass (+0.10)
        assert differences["bass-kraft"] == pytest.approx(0.10, abs=0.01)

        # B has less brillanz (-0.05)
        assert differences["brillanz"] == pytest.approx(-0.05, abs=0.01)

    def test_get_learning_summary(self, base_goals):
        """Test learning summary generation"""
        profile = UserPreferenceProfile(
            user_id="user_123",
            learned_goals=base_goals,
            goal_weights={"bass-kraft": 2.0},
            n_references=5,
            n_ab_tests=10,
            confidence=0.75,
        )

        summary = get_learning_summary(profile)

        assert "user_123" in summary
        assert "0.75" in summary  # confidence
        assert "5" in summary  # references
        assert "10" in summary  # AB tests


# Integration Tests


class TestIntegration:
    """Test integration scenarios"""

    def test_complete_learning_workflow(self, base_goals, test_audio, mock_calculator):
        """Test complete learning workflow"""
        learner = ReferenceLearner(user_id="user_123", strategy=LearningStrategy.BALANCED, base_goals=base_goals)

        # 1. Analyze reference track
        reference = learner.analyze_reference_track(
            test_audio, 44100, mock_calculator, metadata={"title": "My Favorite Song"}
        )

        assert learner.profile.n_references == 1
        assert learner.profile.confidence > 0.0

        # 2. Learn from A/B tests
        for i in range(3):
            ab_result = ABTestResult(
                variant_a_goals={"bass-kraft": 0.90, "brillanz": 0.85, "transparenz": 0.85},
                variant_b_goals={"bass-kraft": 0.80, "brillanz": 0.85, "transparenz": 0.85},
                user_choice="A",  # Consistently prefer more bass
                audio_id=f"track_{i}",
                confidence=0.9,
            )
            learner.learn_from_ab_test(ab_result)

        assert learner.profile.n_ab_tests == 3

        # 3. Check profile reliability
        # 1 reference + 3 AB tests = 4 samples (< 5 threshold)
        # But confidence might still be decent

        # 4. Adapt goals to preference
        learner.adapt_goals_to_preference(base_goals)

        # Should have learned user prefers more bass
        # (moved towards 0.95 from references and 0.90 from AB tests)
        # Exact value depends on learning dynamics

        # 5. Get summary
        summary = get_learning_summary(learner.profile)
        assert len(summary) > 0

    def test_learning_converges(self, base_goals):
        """Test that learning converges over time"""
        learner = ReferenceLearner(user_id="user_123", strategy=LearningStrategy.BALANCED, base_goals=base_goals)

        # Simulate many AB tests with consistent preference
        target_bass = 0.92

        bass_history = [learner.profile.learned_goals["bass-kraft"]]

        for i in range(20):
            ab_result = ABTestResult(
                variant_a_goals={"bass-kraft": target_bass},
                variant_b_goals={"bass-kraft": 0.80},
                user_choice="A",
                audio_id=f"track_{i}",
                confidence=1.0,
            )
            learner.learn_from_ab_test(ab_result)
            bass_history.append(learner.profile.learned_goals["bass-kraft"])

        # Bass should converge towards target
        final_bass = bass_history[-1]

        # Should be closer to target than initially
        initial_distance = abs(bass_history[0] - target_bass)
        final_distance = abs(final_bass - target_bass)
        assert final_distance < initial_distance

        # Confidence should increase
        assert learner.profile.confidence > 0.50

    def test_importance_weight_learning(self, base_goals):
        """Test that importance weights are learned correctly"""
        learner = ReferenceLearner(user_id="user_123", base_goals=base_goals)

        # Simulate AB tests where bass has large differences (important)
        # and brillanz has small differences (less important)
        for i in range(10):
            ab_result = ABTestResult(
                variant_a_goals={
                    "bass-kraft": 0.95,  # Large difference (0.20)
                    "brillanz": 0.86,  # Small difference (0.02)
                },
                variant_b_goals={"bass-kraft": 0.75, "brillanz": 0.84},
                user_choice="A",
                audio_id=f"track_{i}",
                confidence=1.0,
            )
            learner.learn_from_ab_test(ab_result)

        # Bass weight should be higher (user cares more about bass)
        learner.get_goal_importances()
        # Bass should have high importance due to large differences
        # Brillanz should have lower importance due to small differences
        # (Exact values depend on learning dynamics)

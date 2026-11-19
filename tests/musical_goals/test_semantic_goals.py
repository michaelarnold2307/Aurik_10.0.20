"""
Tests für Semantic Musical Goals (Component 0.9.6)

Testet Instrument- und Segment-spezifische Musical Goals Anpassung.
"""

# Import from backend
import sys

import numpy as np
import pytest

sys.path.insert(0, "/mnt/1846D15B46D139E8/Aurik_Standalone")
from backend.core.musical_goals.semantic_goals import (
    GoalProfile,
    InstrumentCategory,
    InstrumentProfileLibrary,
    SegmentProfileLibrary,
    SegmentType,
    SemanticContext,
    SemanticGoalsEngine,
    get_instrument_profile,
    get_segment_profile,
)


@pytest.mark.unit
class TestEnums:
    """Test enum definitions"""

    def test_instrument_category_values(self):
        """Test InstrumentCategory enum has all expected values"""
        expected = [
            "vocals",
            "strings",
            "brass",
            "woodwinds",
            "percussion",
            "drums",
            "bass",
            "guitar",
            "keyboard",
            "electronic",
            "ensemble",
            "unknown",
        ]

        values = [cat.value for cat in InstrumentCategory]

        for exp in expected:
            assert exp in values

    def test_segment_type_values(self):
        """Test SegmentType enum has all expected values"""
        expected = ["intro", "verse", "chorus", "bridge", "outro", "solo", "breakdown", "build_up", "drop", "unknown"]

        values = [seg.value for seg in SegmentType]

        for exp in expected:
            assert exp in values


class TestGoalProfile:
    """Test GoalProfile dataclass"""

    def test_goal_profile_creation(self):
        """Test creating a goal profile"""
        profile = GoalProfile(
            name="Test",
            goals={"bass-kraft": 0.90, "brillanz": 0.85},
            priorities={"bass-kraft": 1.0, "brillanz": 0.9},
            description="Test profile",
        )

        assert profile.name == "Test"
        assert profile.goals["bass-kraft"] == 0.90
        assert profile.priorities["bass-kraft"] == 1.0
        assert profile.description == "Test profile"

    def test_apply_to_base_goals(self):
        """Test applying profile to base goals"""
        profile = GoalProfile(
            name="Test", goals={"bass-kraft": 0.95, "brillanz": 0.80}, priorities={"bass-kraft": 1.0, "brillanz": 0.9}
        )

        base_goals = {"bass-kraft": 0.85, "brillanz": 0.85, "waerme": 0.80}

        adjusted = profile.apply_to_base_goals(base_goals)

        # Should be weighted average: 70% profile + 30% base
        expected_bass = 0.7 * 0.95 + 0.3 * 0.85  # = 0.92
        assert abs(adjusted["bass-kraft"] - expected_bass) < 0.01

        expected_brillanz = 0.7 * 0.80 + 0.3 * 0.85  # = 0.815
        assert abs(adjusted["brillanz"] - expected_brillanz) < 0.01

        # waerme not in profile, should remain unchanged
        assert adjusted["waerme"] == 0.80

    def test_apply_clamps_to_valid_range(self):
        """Test that adjustments are clamped to [0.7, 1.0]"""
        profile = GoalProfile(
            name="Test",
            goals={"bass-kraft": 1.10},
            priorities={"bass-kraft": 1.0},  # Would exceed 1.0
        )

        base_goals = {"bass-kraft": 0.60}  # Below 0.7

        adjusted = profile.apply_to_base_goals(base_goals)

        # Should be clamped
        assert 0.7 <= adjusted["bass-kraft"] <= 1.0


class TestSemanticContext:
    """Test SemanticContext dataclass"""

    def test_semantic_context_creation(self):
        """Test creating semantic context"""
        context = SemanticContext(
            dominant_instrument=InstrumentCategory.VOCALS,
            all_instruments=[InstrumentCategory.VOCALS, InstrumentCategory.GUITAR],
            segment_type=SegmentType.CHORUS,
            segment_position=0.5,
            confidence=0.9,
        )

        assert context.dominant_instrument == InstrumentCategory.VOCALS
        assert len(context.all_instruments) == 2
        assert context.segment_type == SegmentType.CHORUS
        assert context.segment_position == 0.5
        assert context.confidence == 0.9

    def test_semantic_context_defaults(self):
        """Test semantic context with default values"""
        context = SemanticContext(dominant_instrument=InstrumentCategory.BASS)

        assert context.all_instruments == []
        assert context.segment_type == SegmentType.UNKNOWN
        assert context.segment_position == 0.5
        assert context.confidence == 1.0


class TestInstrumentProfileLibrary:
    """Test instrument profile library"""

    def test_library_initialization(self):
        """Test library initializes all profiles"""
        lib = InstrumentProfileLibrary()

        # Should have profile for each category
        for category in InstrumentCategory:
            profile = lib.get_profile(category)
            assert profile is not None
            assert isinstance(profile, GoalProfile)

    def test_vocals_profile(self):
        """Test vocals profile prioritizes natürlichkeit and emotionalität"""
        lib = InstrumentProfileLibrary()
        profile = lib.get_profile(InstrumentCategory.VOCALS)

        # Check key characteristics
        assert profile.goals["natuerlichkeit"] >= 0.95
        assert profile.goals["emotionalitaet"] >= 0.95
        assert profile.goals["transparenz"] >= 0.90

        # Check priorities
        assert profile.priorities["natuerlichkeit"] >= 0.95
        assert profile.priorities["emotionalitaet"] >= 0.95

    def test_bass_profile(self):
        """Test bass profile prioritizes bass-kraft and wärme"""
        lib = InstrumentProfileLibrary()
        profile = lib.get_profile(InstrumentCategory.BASS)

        # Check key characteristics
        assert profile.goals["bass-kraft"] >= 0.92
        assert profile.goals["waerme"] >= 0.88

        # Check priorities
        assert profile.priorities["bass-kraft"] >= 0.95
        assert profile.priorities["waerme"] >= 0.90

    def test_drums_profile(self):
        """Test drums profile prioritizes transparenz and definition"""
        lib = InstrumentProfileLibrary()
        profile = lib.get_profile(InstrumentCategory.DRUMS)

        # Check key characteristics
        assert profile.goals["transparenz"] >= 0.92
        assert profile.goals["bass-kraft"] >= 0.88  # Kick drum
        assert profile.goals["brillanz"] >= 0.88  # Cymbals

        # Check priorities
        assert profile.priorities["transparenz"] >= 0.95

    def test_electronic_profile(self):
        """Test electronic profile has different weighting"""
        lib = InstrumentProfileLibrary()
        profile = lib.get_profile(InstrumentCategory.ELECTRONIC)

        # Electronic should prioritize brightness and bass over warmth
        assert profile.goals["brillanz"] >= profile.goals["waerme"]
        assert profile.goals["bass-kraft"] >= 0.90

        # Less emphasis on "natural" qualities
        assert profile.priorities["natuerlichkeit"] < 0.75

    def test_unknown_uses_base_goals(self):
        """Test unknown category uses base goals"""
        lib = InstrumentProfileLibrary()
        profile = lib.get_profile(InstrumentCategory.UNKNOWN)

        # Should match base goals
        for goal_name, value in InstrumentProfileLibrary.BASE_GOALS.items():
            assert profile.goals[goal_name] == value


class TestSegmentProfileLibrary:
    """Test segment profile library"""

    def test_library_initialization(self):
        """Test library initializes all profiles"""
        lib = SegmentProfileLibrary()

        # Should have profile for each segment type
        for segment in SegmentType:
            profile = lib.get_profile(segment)
            assert profile is not None
            assert isinstance(profile, GoalProfile)

    def test_chorus_profile(self):
        """Test chorus profile prioritizes emotional peak"""
        lib = SegmentProfileLibrary()
        profile = lib.get_profile(SegmentType.CHORUS)

        # Chorus should have high values across the board
        assert profile.goals["emotionalitaet"] >= 0.92
        assert profile.goals["bass-kraft"] >= 0.90
        assert profile.goals["brillanz"] >= 0.90
        assert profile.goals["transparenz"] >= 0.90

        # Check priorities
        assert profile.priorities["emotionalitaet"] >= 0.95

    def test_intro_profile(self):
        """Test intro profile emphasizes clarity"""
        lib = SegmentProfileLibrary()
        profile = lib.get_profile(SegmentType.INTRO)

        # Intro should emphasize transparency
        assert profile.goals["transparenz"] >= 0.90
        assert profile.priorities["transparenz"] >= 0.95

        # Lower emotional intensity (building up)
        assert profile.goals["emotionalitaet"] < 0.90

    def test_outro_profile(self):
        """Test outro profile emphasizes warmth and resolution"""
        lib = SegmentProfileLibrary()
        profile = lib.get_profile(SegmentType.OUTRO)

        # Outro should emphasize warmth
        assert profile.goals["waerme"] >= 0.88
        assert profile.priorities["waerme"] >= 0.95

        # Less brightness (winding down)
        assert profile.goals["brillanz"] < 0.88

    def test_solo_profile(self):
        """Test solo profile maximizes clarity and expression"""
        lib = SegmentProfileLibrary()
        profile = lib.get_profile(SegmentType.SOLO)

        # Solo should have high transparency and expressiveness
        assert profile.goals["transparenz"] >= 0.93
        assert profile.goals["emotionalitaet"] >= 0.92
        assert profile.goals["natuerlichkeit"] >= 0.92

        # Check priorities
        assert profile.priorities["transparenz"] >= 0.95
        assert profile.priorities["emotionalitaet"] >= 0.95

    def test_drop_profile(self):
        """Test drop profile maximizes bass and impact"""
        lib = SegmentProfileLibrary()
        profile = lib.get_profile(SegmentType.DROP)

        # Drop should have maximum bass
        assert profile.goals["bass-kraft"] >= 0.93
        assert profile.goals["emotionalitaet"] >= 0.93

        # Check priorities
        assert profile.priorities["bass-kraft"] >= 0.95


class TestSemanticGoalsEngine:
    """Test semantic goals engine"""

    def test_engine_initialization(self):
        """Test engine initialization without ML models"""
        engine = SemanticGoalsEngine()

        assert engine.instrument_library is not None
        assert engine.segment_library is not None
        # ML models should be None (not available)
        assert engine.instrument_detector is None
        assert engine.structure_analyzer is not None  # madmom installed

    def test_detect_instruments_fallback(self):
        """Test fallback instrument detection"""
        engine = SemanticGoalsEngine()

        # Generate test audio
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Low frequency (bass-like)
        audio_bass = np.sin(2 * np.pi * 100 * t)

        dominant, all_inst, conf = engine.detect_instruments(audio_bass, sr)

        # Should detect as bass
        assert dominant == InstrumentCategory.BASS
        assert conf > 0  # Should have some confidence

    def test_analyze_structure_fallback(self):
        """Test fallback structure analysis"""
        engine = SemanticGoalsEngine()

        # Generate test audio
        sr = 44100
        duration = 60.0  # 60 seconds
        audio = np.random.randn(int(sr * duration)) * 0.1

        segments = engine.analyze_structure(audio, sr)

        # Should return at least intro/main/outro
        assert len(segments) >= 3

        # Check segment structure
        for start, end, seg_type in segments:
            assert start < end
            assert isinstance(seg_type, SegmentType)

    def test_get_semantic_context(self):
        """Test getting semantic context from audio"""
        engine = SemanticGoalsEngine()

        # Generate test audio
        sr = 44100
        duration = 30.0
        audio = np.random.randn(int(sr * duration)) * 0.1

        context = engine.get_semantic_context(audio, sr, timestamp=15.0)

        assert isinstance(context, SemanticContext)
        assert isinstance(context.dominant_instrument, InstrumentCategory)
        assert isinstance(context.segment_type, SegmentType)
        assert 0.0 <= context.segment_position <= 1.0
        assert 0.0 <= context.confidence <= 1.0

    def test_adjust_goals_for_vocals_context(self):
        """Test goal adjustment for vocals context"""
        engine = SemanticGoalsEngine()

        base_goals = {
            "bass-kraft": 0.85,
            "brillanz": 0.85,
            "waerme": 0.80,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.87,
            "transparenz": 0.89,
        }

        context = SemanticContext(
            dominant_instrument=InstrumentCategory.VOCALS, segment_type=SegmentType.CHORUS, confidence=1.0
        )

        adjusted = engine.adjust_goals_for_context(base_goals, context)

        # Vocals in chorus should boost natürlichkeit, emotionalität
        assert adjusted["natuerlichkeit"] > base_goals["natuerlichkeit"]
        assert adjusted["emotionalitaet"] > base_goals["emotionalitaet"]

        # Transparenz should be boosted for clarity
        assert adjusted["transparenz"] >= base_goals["transparenz"]

    def test_adjust_goals_for_bass_context(self):
        """Test goal adjustment for bass context"""
        engine = SemanticGoalsEngine()

        base_goals = {
            "bass-kraft": 0.85,
            "brillanz": 0.85,
            "waerme": 0.80,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.87,
            "transparenz": 0.89,
        }

        context = SemanticContext(
            dominant_instrument=InstrumentCategory.BASS, segment_type=SegmentType.VERSE, confidence=1.0
        )

        adjusted = engine.adjust_goals_for_context(base_goals, context)

        # Bass should boost bass-kraft and wärme
        assert adjusted["bass-kraft"] > base_goals["bass-kraft"]
        assert adjusted["waerme"] > base_goals["waerme"]

        # Brillanz should be reduced
        assert adjusted["brillanz"] <= base_goals["brillanz"]

    def test_adjust_goals_for_drums_context(self):
        """Test goal adjustment for drums context"""
        engine = SemanticGoalsEngine()

        base_goals = {
            "bass-kraft": 0.85,
            "brillanz": 0.85,
            "waerme": 0.80,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.87,
            "transparenz": 0.89,
        }

        context = SemanticContext(
            dominant_instrument=InstrumentCategory.DRUMS, segment_type=SegmentType.DROP, confidence=1.0
        )

        adjusted = engine.adjust_goals_for_context(base_goals, context)

        # Drums should boost transparenz, bass-kraft, brillanz
        assert adjusted["transparenz"] > base_goals["transparenz"]
        assert adjusted["bass-kraft"] > base_goals["bass-kraft"]
        assert adjusted["brillanz"] > base_goals["brillanz"]

    def test_confidence_affects_adjustment(self):
        """Test that low confidence reduces adjustment strength"""
        engine = SemanticGoalsEngine()

        base_goals = {
            "bass-kraft": 0.85,
            "brillanz": 0.85,
            "waerme": 0.80,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.87,
            "transparenz": 0.89,
        }

        # High confidence context
        context_high = SemanticContext(
            dominant_instrument=InstrumentCategory.VOCALS, segment_type=SegmentType.CHORUS, confidence=1.0
        )

        # Low confidence context
        context_low = SemanticContext(
            dominant_instrument=InstrumentCategory.VOCALS, segment_type=SegmentType.CHORUS, confidence=0.3
        )

        adjusted_high = engine.adjust_goals_for_context(base_goals, context_high)
        adjusted_low = engine.adjust_goals_for_context(base_goals, context_low)

        # High confidence should adjust more
        for goal_name in base_goals:
            diff_high = abs(adjusted_high[goal_name] - base_goals[goal_name])
            diff_low = abs(adjusted_low[goal_name] - base_goals[goal_name])

            # High confidence adjustment should be larger
            assert diff_high >= diff_low

    def test_adjusted_goals_stay_in_bounds(self):
        """Test that adjusted goals stay in [0.7, 1.0]"""
        engine = SemanticGoalsEngine()

        base_goals = {
            "bass-kraft": 0.85,
            "brillanz": 0.85,
            "waerme": 0.80,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.87,
            "transparenz": 0.89,
        }

        # Test with various contexts
        contexts = [
            SemanticContext(InstrumentCategory.VOCALS, segment_type=SegmentType.CHORUS),
            SemanticContext(InstrumentCategory.BASS, segment_type=SegmentType.DROP),
            SemanticContext(InstrumentCategory.DRUMS, segment_type=SegmentType.INTRO),
        ]

        for context in contexts:
            adjusted = engine.adjust_goals_for_context(base_goals, context)

            for goal_name, value in adjusted.items():
                assert 0.7 <= value <= 1.0, f"{goal_name} = {value} out of bounds for {context.dominant_instrument}"


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_instrument_profile(self):
        """Test get_instrument_profile function"""
        profile = get_instrument_profile(InstrumentCategory.VOCALS)

        assert isinstance(profile, GoalProfile)
        assert profile.name == "Vocals"

    def test_get_segment_profile(self):
        """Test get_segment_profile function"""
        profile = get_segment_profile(SegmentType.CHORUS)

        assert isinstance(profile, GoalProfile)
        assert profile.name == "Chorus"


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_complete_semantic_adjustment_workflow(self):
        """Test complete workflow from audio to adjusted goals"""
        engine = SemanticGoalsEngine()

        # Generate test audio (60 seconds)
        sr = 44100
        duration = 60.0
        t = np.linspace(0, duration, int(sr * duration))

        # Mix of frequencies (simulate full mix)
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)  # Bass
            + 0.3 * np.sin(2 * np.pi * 440 * t)  # Midrange
            + 0.2 * np.sin(2 * np.pi * 2000 * t)  # Highs
            + 0.1 * np.random.randn(len(t))  # Noise
        )

        base_goals = InstrumentProfileLibrary.BASE_GOALS.copy()

        # Test at different timestamps
        timestamps = [5.0, 15.0, 30.0, 50.0]  # intro, verse, mid, outro

        for ts in timestamps:
            # Get context
            context = engine.get_semantic_context(audio, sr, ts)

            # Adjust goals
            adjusted = engine.adjust_goals_for_context(base_goals, context)

            # Verify adjustments
            assert len(adjusted) == len(base_goals)
            for goal_name in base_goals.keys():
                assert 0.7 <= adjusted[goal_name] <= 1.0

    def test_segment_progression_affects_goals(self):
        """Test that goals change across segment progression"""
        engine = SemanticGoalsEngine()

        sr = 44100
        duration = 90.0
        audio = np.random.randn(int(sr * duration)) * 0.1

        base_goals = InstrumentProfileLibrary.BASE_GOALS.copy()

        # Sample at different points
        intro_context = engine.get_semantic_context(audio, sr, 5.0)
        mid_context = engine.get_semantic_context(audio, sr, 45.0)
        outro_context = engine.get_semantic_context(audio, sr, 85.0)

        intro_goals = engine.adjust_goals_for_context(base_goals, intro_context)
        mid_goals = engine.adjust_goals_for_context(base_goals, mid_context)
        outro_goals = engine.adjust_goals_for_context(base_goals, outro_context)

        # Goals should differ across segments
        # (exact differences depend on segment detection, but should not be all identical)
        all_same = all(intro_goals[g] == mid_goals[g] == outro_goals[g] for g in base_goals.keys())
        assert not all_same, "Goals should vary across segments"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

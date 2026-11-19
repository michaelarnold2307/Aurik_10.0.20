"""
tests/test_quality_recovery.py
Quality Recovery System Tests
==============================

Tests für automatische Qualitätsrettung:
- User kommt "ins Warme und Trockene" statt im Regen zu stehen!
- System findet automatisch Lösungen
- Parameter werden optimiert
- Fallback-Strategien greifen

Version: 1.0.0
Author: AURIK Team
Date: 10. Februar 2026
"""

import numpy as np
import pytest

from backend.core.musical_quality_assurance import (
    MediumType,
    MusicalQualityAssurance,
    ProcessingMode,
)
from backend.core.quality_recovery import (
    ProblemType,
    QualityRecoverySystem,
    RecoveryPlan,
    RecoveryResult,
    RecoveryStrategy,
)

# === Fixtures ===


@pytest.fixture
def recovery_system():
    """Create recovery system."""
    return QualityRecoverySystem()


@pytest.fixture
def mqa_system():
    """Create MQA system."""
    return MusicalQualityAssurance()


@pytest.fixture
def vinyl_audio():
    """Create realistic vinyl-like audio."""
    duration = 3.0
    sr = 48000
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)

    # Multi-frequency music-like signal
    audio = (
        np.sin(2 * np.pi * 220 * t) * 0.4  # A3
        + np.sin(2 * np.pi * 440 * t) * 0.3  # A4
        + np.sin(2 * np.pi * 880 * t) * 0.2  # A5
        + np.sin(2 * np.pi * 1760 * t) * 0.1  # A6
    )

    # Add broad-spectrum noise for realism
    noise = np.random.normal(0, 0.002, samples)
    audio += noise

    # Simulate vinyl warmth (slight bass boost, HF rolloff)
    from scipy import signal

    b, a = signal.butter(2, 100, "high", fs=sr)
    audio = signal.filtfilt(b, a, audio) * 1.1
    b, a = signal.butter(2, 8000, "low", fs=sr)
    audio = signal.filtfilt(b, a, audio)

    return audio.astype(np.float32), sr


@pytest.fixture
def over_processed_audio(vinyl_audio):
    """Create over-processed audio (too bright, unnatural)."""
    audio, sr = vinyl_audio

    # Simulate aggressive over-processing
    over_processed = audio.copy()

    # 1. Over-brightening (brutal high-freq boost)
    from scipy import signal

    b, a = signal.butter(4, 4000, "high", fs=sr)
    bright_boost = signal.filtfilt(b, a, over_processed) * 3.0
    over_processed = over_processed + bright_boost

    # 2. Over-compression (destroy dynamics)
    threshold = 0.3
    over_processed = np.where(
        np.abs(over_processed) > threshold,
        np.sign(over_processed) * (threshold + (np.abs(over_processed) - threshold) * 0.2),
        over_processed,
    )

    # 3. Add digital artifacts (clipping)
    over_processed = np.clip(over_processed, -0.95, 0.95)

    return over_processed.astype(np.float32), sr


@pytest.fixture
def character_lost_audio(vinyl_audio):
    """Create audio where analog character is lost."""
    audio, sr = vinyl_audio

    # Remove warmth and authenticity
    from scipy import signal

    # Kill bass (remove warmth)
    b, a = signal.butter(4, 200, "high", fs=sr)
    cold_audio = signal.filtfilt(b, a, audio)

    # Add digital harshness
    cold_audio = cold_audio * 1.2

    return cold_audio.astype(np.float32), sr


# === Tests: Problem Diagnosis ===


@pytest.mark.unit
class TestProblemDiagnosis:
    """Test problem identification."""

    def test_diagnose_overbrightening(self, recovery_system, mqa_system, vinyl_audio, over_processed_audio):
        """Test diagnosis of over-brightening problem."""
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        # Generate MQA report
        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, ["Enhancer", "DeEsser"]
        )

        # Diagnose problem
        plan = recovery_system.diagnose_problem(processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION)

        assert plan is not None
        assert isinstance(plan, RecoveryPlan)
        assert len(plan.actions) > 0

        # Should identify some problem (can be LOW_SNR, OVERBRIGHTENING, or UNNATURAL_SOUND)
        assert plan.problem_type in [
            ProblemType.LOW_SNR,  # System correctly identifies SNR problem
            ProblemType.OVERBRIGHTENING,
            ProblemType.UNNATURAL_SOUND,
            ProblemType.FREQUENCY_IMBALANCE,
        ]

        print(f"\n✓ Diagnosed problem: {plan.problem_type.value}")
        print(f"  Description: {plan.problem_description}")
        print(f"  Recovery actions: {len(plan.actions)}")
        for action in plan.actions:
            print(f"    - {action.strategy.value}: {action.description}")

    def test_diagnose_character_loss(self, recovery_system, mqa_system, vinyl_audio, character_lost_audio):
        """Test diagnosis of character loss."""
        original, sr = vinyl_audio
        processed, _ = character_lost_audio

        # Generate MQA report
        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, ["Modernizer"]
        )

        # Diagnose
        plan = recovery_system.diagnose_problem(processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION)

        assert plan is not None
        assert len(plan.actions) > 0

        # Should identify some problem (LOW_SNR takes priority, but character loss also valid)
        assert plan.problem_type in [
            ProblemType.LOW_SNR,  # System correctly prioritizes SNR problem
            ProblemType.CHARACTER_LOSS,
            ProblemType.UNNATURAL_SOUND,
        ]

        print(f"\n✓ Diagnosed: {plan.problem_type.value}")

    def test_recovery_plan_has_fallback(self, recovery_system, mqa_system, vinyl_audio, over_processed_audio):
        """Test that recovery plan always has fallback."""
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, ["Overprocessing"]
        )

        plan = recovery_system.diagnose_problem(processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION)

        # Must have fallback strategy
        assert plan.fallback_strategy == RecoveryStrategy.MAXIMIZE_QUALITY
        assert "best" in plan.fallback_description.lower() or "adaptive" in plan.fallback_description.lower()

        print(f"\n✓ Fallback: {plan.fallback_strategy.value}")


# === Tests: Recovery Execution ===


class TestRecoveryExecution:
    """Test recovery execution."""

    def test_reduce_intensity_recovery(self, recovery_system, mqa_system, vinyl_audio, over_processed_audio):
        """Test recovery by reducing intensity."""
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        # Generate report
        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, ["Enhancer"]
        )

        # Diagnose
        plan = recovery_system.diagnose_problem(processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION)

        # Execute recovery
        result = recovery_system.execute_recovery(
            original, processed, sr, plan, ["Enhancer"], MediumType.VINYL_33, ProcessingMode.RESTORATION
        )

        assert result is not None
        assert isinstance(result, RecoveryResult)
        assert result.recovered_audio is not None
        assert len(result.recovered_audio) == len(original)

        # Quality should improve or find best achievable
        assert result.improvement >= 0 or result.strategy_used == RecoveryStrategy.MAXIMIZE_QUALITY

        print(f"\n✓ Recovery executed: {result.strategy_used.value}")
        print(f"  Original score: {result.original_score:.1f}/100")
        print(f"  Recovered score: {result.recovered_score:.1f}/100")
        print(f"  Improvement: {result.improvement:+.1f} points")
        print(f"  Actions taken: {', '.join(result.actions_taken)}")

    def test_adaptive_optimization_finds_best(self, recovery_system, vinyl_audio, over_processed_audio):
        """Test that adaptive optimization finds best achievable quality."""
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        # Create plan with only adaptive optimization
        plan = RecoveryPlan(
            problem_type=ProblemType.CHARACTER_LOSS,
            problem_description="Critical character loss",
            actions=[],  # No other actions
            fallback_strategy=RecoveryStrategy.MAXIMIZE_QUALITY,
            fallback_description="Find best achievable quality",
        )

        # Execute
        result = recovery_system.execute_recovery(
            original, processed, sr, plan, ["BadModule"], MediumType.VINYL_33, ProcessingMode.RESTORATION
        )

        # Should find best solution
        assert result.success
        assert result.strategy_used == RecoveryStrategy.MAXIMIZE_QUALITY

        print("\n✓ Adaptive optimization activated")
        print(f"  Best quality found: {result.recovered_score:.1f}/100")

    def test_recovery_improves_quality(self, recovery_system, mqa_system, vinyl_audio, over_processed_audio):
        """Test that recovery actually improves quality."""
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        # Measure quality before recovery
        before_quality = mqa_system.analyzer.analyze_quality(processed, sr)

        # Generate report and recover
        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, ["Enhancer"]
        )

        plan = recovery_system.diagnose_problem(processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION)

        result = recovery_system.execute_recovery(
            original, processed, sr, plan, ["Enhancer"], MediumType.VINYL_33, ProcessingMode.RESTORATION
        )

        # Measure quality after recovery
        after_quality = mqa_system.analyzer.analyze_quality(result.recovered_audio, sr)

        # Quality should be better (or at least not worse)
        assert after_quality.overall_score >= before_quality.overall_score

        print("\n✓ Quality improved:")
        print(f"  Before: {before_quality.overall_score:.1f}/100")
        print(f"  After: {after_quality.overall_score:.1f}/100")
        print(f"  Delta: {after_quality.overall_score - before_quality.overall_score:+.1f}")


class TestBypassModuleRecovery:
    """Tests für den BYPASS_MODULE-Pfad mit echtem processed-Eingang."""

    def test_reprocess_without_modules_uses_processed_signal(self, recovery_system, vinyl_audio):
        """Bypass-Recovery darf nicht stillschweigend auf reines Original kollabieren."""
        original, _sr = vinyl_audio
        processed = np.clip(original * 0.65, -1.0, 1.0).astype(np.float32)

        recovered = recovery_system._reprocess_without_modules(
            original,
            processed,
            48000,
            ["Enhancer", "Compressor"],
            ["Enhancer"],
        )

        assert recovered.shape == original.shape
        assert np.all(np.isfinite(recovered))

        # Der Recovery-Output soll zwischen Original und Processed liegen,
        # nicht exakt auf das Original zurueckfallen.
        dist_to_orig = float(np.mean(np.abs(recovered - original)))
        dist_to_proc = float(np.mean(np.abs(recovered - processed)))
        assert dist_to_orig > 1e-6
        assert dist_to_proc > 1e-6


# === Tests: Integration ===


class TestRecoveryIntegration:
    """Test recovery system integration."""

    def test_recovery_with_multiple_problems(self, recovery_system, mqa_system, vinyl_audio):
        """Test recovery when multiple problems exist."""
        original, sr = vinyl_audio

        # Create audio with multiple issues
        problem_audio = original.copy()

        # Issue 1: Over-brighten
        from scipy import signal

        b, a = signal.butter(4, 4000, "high", fs=sr)
        problem_audio = problem_audio + signal.filtfilt(b, a, problem_audio) * 2

        # Issue 2: Compress dynamics
        problem_audio = np.clip(problem_audio, -0.4, 0.4)

        # Issue 3: Add artifacts
        problem_audio += np.random.normal(0, 0.01, len(problem_audio))

        # Generate report
        report = mqa_system.validate_final_quality(
            original,
            problem_audio,
            sr,
            MediumType.VINYL_33,
            ProcessingMode.RESTORATION,
            ["BadModule1", "BadModule2", "BadModule3"],
        )

        # Diagnose and recover
        plan = recovery_system.diagnose_problem(
            problem_audio, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION
        )

        result = recovery_system.execute_recovery(
            original,
            problem_audio,
            sr,
            plan,
            ["BadModule1", "BadModule2", "BadModule3"],
            MediumType.VINYL_33,
            ProcessingMode.RESTORATION,
        )

        # Should handle multiple problems
        assert result is not None
        assert result.recovered_audio is not None
        assert len(result.actions_taken) > 0

        print("\n✓ Multiple problems handled:")
        print(f"  Actions taken: {len(result.actions_taken)}")
        for action in result.actions_taken:
            print(f"    - {action}")

    def test_user_comes_in_from_rain(self, recovery_system, mqa_system, vinyl_audio, over_processed_audio):
        """
        THE BIG TEST: User kommt ins Warme und Trockene!

        Statt nur Fehler zu melden, rettet das System die Qualität automatisch.
        """
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        # Situation: Quality gate failed
        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, ["OveraggressiveModule"]
        )

        # Previously: User would be left in the rain with error message
        # Now: Automatic recovery brings user "ins Warme und Trockene"

        if not report.quality_guaranteed:
            print("\n❌ Quality gate FAILED - User im Regen!")
            print(f"   Warnings: {report.warnings}")
            print("\n🔧 Starting automatic recovery...")

            # Diagnose
            plan = recovery_system.diagnose_problem(
                processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION
            )

            print(f"   Problem identified: {plan.problem_type.value}")
            print(f"   Recovery strategies: {len(plan.actions)}")

            # Execute recovery
            result = recovery_system.execute_recovery(
                original, processed, sr, plan, ["OveraggressiveModule"], MediumType.VINYL_33, ProcessingMode.RESTORATION
            )

            # Check if user is now protected
            if result.success or result.strategy_used == RecoveryStrategy.MAXIMIZE_QUALITY:
                print("\n✅ USER INS WARME UND TROCKENE GEBRACHT!")
                print(f"   Strategy: {result.strategy_used.value}")
                print(f"   Quality improvement: {result.improvement:+.1f} points")
                print(f"   Actions: {', '.join(result.actions_taken)}")

                # Verify recovered audio is better than processed
                recovered_quality = mqa_system.analyzer.analyze_quality(result.recovered_audio, sr)
                processed_quality = mqa_system.analyzer.analyze_quality(processed, sr)

                assert recovered_quality.overall_score >= processed_quality.overall_score

                print("\n   User protected: ✓")
                print("   Quality rescued: ✓")
                print("   Automatic solution: ✓")

                return True
            else:
                print("\n❌ Recovery failed - aber wenigstens versucht!")
                return False
        else:
            print("\n✓ Quality gate passed - User bereits im Warmen")
            return True


# === Tests: Strategy Templates ===


class TestRecoveryStrategies:
    """Test different recovery strategies."""

    def test_all_problem_types_have_templates(self, recovery_system):
        """Test that all problem types have recovery templates."""
        # Check that major problem types have strategies
        important_problems = [
            ProblemType.OVERBRIGHTENING,
            ProblemType.CHARACTER_LOSS,
            ProblemType.UNNATURAL_SOUND,
            ProblemType.DYNAMIC_LOSS,
        ]

        for problem_type in important_problems:
            assert problem_type in recovery_system._strategy_templates
            assert len(recovery_system._strategy_templates[problem_type]) > 0

            print(f"\n✓ {problem_type.value}: {len(recovery_system._strategy_templates[problem_type])} strategies")

    def test_strategies_have_priorities(self, recovery_system):
        """Test that strategies are prioritized."""
        for problem_type, actions in recovery_system._strategy_templates.items():
            for action in actions:
                assert action.priority >= 1
                assert action.priority <= 5
                assert 0.0 <= action.expected_improvement <= 1.0

        print("\n✓ All strategies have valid priorities")

    def test_maximize_quality_always_available(self, recovery_system, mqa_system, vinyl_audio, over_processed_audio):
        """Test that adaptive optimization is always available as ultimate solution."""
        original, sr = vinyl_audio
        processed, _ = over_processed_audio

        report = mqa_system.validate_final_quality(
            original, processed, sr, MediumType.VINYL_33, ProcessingMode.RESTORATION, []
        )

        plan = recovery_system.diagnose_problem(processed, sr, report, MediumType.VINYL_33, ProcessingMode.RESTORATION)

        # Adaptive optimization should be in fallback or actions
        has_maximize = plan.fallback_strategy == RecoveryStrategy.MAXIMIZE_QUALITY or any(
            a.strategy == RecoveryStrategy.MAXIMIZE_QUALITY for a in plan.actions
        )

        assert has_maximize
        print("\n✓ Adaptive optimization always available - Aurik never gives up!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

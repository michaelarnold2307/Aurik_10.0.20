"""
tests/test_module_coordinator.py
Test Suite for Advanced Module Coordinator
==========================================

Tests:
- Module registration and dependency management
- Topological sort with staging
- Parallel vs sequential execution
- ML-based parameter optimization
- Quality prediction
- Error recovery
- Performance profiling

Author: AURIK Team
"""

import time

import numpy as np
import pytest

from backend.core.module_communication import ModuleCommunicationBus
from backend.core.module_coordinator import (
    ExecutionStrategy,
    ModuleCoordinator,
    ModulePriority,
    create_coordinator,
)
from backend.core.processing_context import ProcessingContext

# === Mock Modules ===


class MockForensicAnalyzer:
    """Mock forensic analyzer."""

    def analyze(self, audio, sr):
        return {
            "medium_type": "VINYL",
            "quality_assessment": "GOOD",
            "defects_detected": {"clicks": True, "crackle": True},
        }

    def get_metrics(self):
        return {"confidence": 0.95}


class MockTapeSpecialist:
    """Mock tape specialist."""

    def process(self, audio, sr, strength=0.5, **kwargs):
        # Simulate processing (accept extra kwargs from ML parameter optimization)
        time.sleep(0.01)
        return audio * 0.99  # Slight change

    def get_metrics(self):
        return {"noise_reduced_db": 12.5, "confidence": 0.88}


class MockClickRemover:
    """Mock click removal."""

    def process(self, audio, sr, threshold=0.02):
        time.sleep(0.005)
        return audio * 0.98

    def get_metrics(self):
        return {"clicks_removed": 45, "confidence": 0.92}


class MockEqualizer:
    """Mock equalizer."""

    def process(self, audio, sr):
        time.sleep(0.003)
        return audio

    def get_metrics(self):
        return {"confidence": 0.85}


class FailingModule:
    """Module that always fails."""

    def process(self, audio, sr):
        raise RuntimeError("Simulated failure")


# === Test Fixtures ===


@pytest.fixture
def audio():
    """Test audio fixture — Sinus-Signal mit hohem SNR (überschreitet Quality-Gate 48 dB)."""
    t = np.linspace(0, 1.0, 48000, dtype=np.float32)
    # 440 Hz Sinus + minimales Rauschen → SNR ≈ 54 dB (> 48 dB Schwelle)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.001 * np.random.default_rng(42).standard_normal(48000).astype(np.float32)
    return (signal + noise).astype(np.float32)


@pytest.fixture
def context():
    """Processing context fixture."""
    return ProcessingContext("test_session")


@pytest.fixture
def bus():
    """Communication bus fixture."""
    return ModuleCommunicationBus()


@pytest.fixture
def coordinator(context, bus):
    """Module coordinator fixture."""
    return create_coordinator(context, bus, max_workers=4)


# === Test Cases ===


class TestModuleRegistration:
    """Test module registration."""

    def test_register_module(self, coordinator):
        """Test registering a module."""
        coordinator.register_module(
            name="ForensicAnalyzer",
            module_class=MockForensicAnalyzer,
            priority=ModulePriority.HIGH,
            provides=["forensic_analysis"],
        )

        modules = coordinator.get_registered_modules()
        assert "ForensicAnalyzer" in modules

    def test_register_with_dependencies(self, coordinator):
        """Test registering modules with dependencies."""
        coordinator.register_module(
            name="ForensicAnalyzer", module_class=MockForensicAnalyzer, priority=ModulePriority.HIGH
        )

        coordinator.register_module(
            name="TapeSpecialist",
            module_class=MockTapeSpecialist,
            dependencies=["ForensicAnalyzer"],
            categories=["tape"],
        )

        modules = coordinator.get_registered_modules()
        assert len(modules) == 2
        assert "ForensicAnalyzer" in modules
        assert "TapeSpecialist" in modules

    def test_unregister_module(self, coordinator):
        """Test unregistering a module."""
        coordinator.register_module(name="ForensicAnalyzer", module_class=MockForensicAnalyzer)

        coordinator.unregister_module("ForensicAnalyzer")

        modules = coordinator.get_registered_modules()
        assert "ForensicAnalyzer" not in modules

    def test_register_multiple_modules(self, coordinator):
        """Test registering multiple modules."""
        modules_to_register = [
            ("ForensicAnalyzer", MockForensicAnalyzer, ModulePriority.HIGH, []),
            ("TapeSpecialist", MockTapeSpecialist, ModulePriority.NORMAL, ["ForensicAnalyzer"]),
            ("ClickRemover", MockClickRemover, ModulePriority.NORMAL, ["ForensicAnalyzer"]),
            ("Equalizer", MockEqualizer, ModulePriority.LOW, ["TapeSpecialist", "ClickRemover"]),
        ]

        for name, cls, priority, deps in modules_to_register:
            coordinator.register_module(name=name, module_class=cls, priority=priority, dependencies=deps)

        registered = coordinator.get_registered_modules()
        assert len(registered) == 4


class TestDependencyResolution:
    """Test dependency resolution and execution planning."""

    def test_simple_dependency_chain(self, coordinator):
        """Test simple A -> B -> C dependency chain."""
        coordinator.register_module("A", MockForensicAnalyzer, ModulePriority.HIGH, [])
        coordinator.register_module("B", MockTapeSpecialist, ModulePriority.NORMAL, ["A"])
        coordinator.register_module("C", MockEqualizer, ModulePriority.NORMAL, ["B"])

        plan = coordinator.build_execution_plan()

        # Should have 3 stages (A, then B, then C)
        assert len(plan.stages) == 3
        assert plan.stages[0][0].name == "A"
        assert plan.stages[1][0].name == "B"
        assert plan.stages[2][0].name == "C"

    def test_parallel_execution_plan(self, coordinator):
        """Test parallel execution planning."""
        # A is root, B and C depend on A, D depends on B and C
        coordinator.register_module("A", MockForensicAnalyzer, ModulePriority.HIGH, [])
        coordinator.register_module("B", MockTapeSpecialist, ModulePriority.NORMAL, ["A"])
        coordinator.register_module("C", MockClickRemover, ModulePriority.NORMAL, ["A"])
        coordinator.register_module("D", MockEqualizer, ModulePriority.NORMAL, ["B", "C"])

        plan = coordinator.build_execution_plan()

        # Should have 3 stages: [A], [B, C], [D]
        assert len(plan.stages) == 3
        assert len(plan.stages[0]) == 1  # A
        assert len(plan.stages[1]) == 2  # B and C in parallel
        assert len(plan.stages[2]) == 1  # D

        # Check parallel opportunities
        assert plan.parallel_opportunities >= 1

    def test_priority_ordering(self, coordinator):
        """Test that modules are ordered by priority within stages."""
        coordinator.register_module("A", MockTapeSpecialist, ModulePriority.LOW, [])
        coordinator.register_module("B", MockClickRemover, ModulePriority.HIGH, [])
        coordinator.register_module("C", MockEqualizer, ModulePriority.NORMAL, [])

        plan = coordinator.build_execution_plan()

        # All have no dependencies, so should be in one stage
        assert len(plan.stages) == 1

        # Should be ordered by priority (HIGH, NORMAL, LOW)
        names = [m.name for m in plan.stages[0]]
        assert names.index("B") < names.index("C")  # HIGH before NORMAL
        assert names.index("C") < names.index("A")  # NORMAL before LOW

    def test_critical_path_identification(self, coordinator):
        """Test critical path identification."""
        # Create a diamond dependency: A -> B -> D, A -> C -> D
        coordinator.register_module("A", MockForensicAnalyzer, ModulePriority.HIGH, [])
        coordinator.register_module("B", MockTapeSpecialist, ModulePriority.NORMAL, ["A"])
        coordinator.register_module("C", MockClickRemover, ModulePriority.NORMAL, ["A"])
        coordinator.register_module("D", MockEqualizer, ModulePriority.NORMAL, ["B", "C"])

        plan = coordinator.build_execution_plan()

        # Critical path should include A and D at minimum
        assert "A" in plan.critical_path_modules
        assert "D" in plan.critical_path_modules


class TestExecution:
    """Test module execution."""

    def test_sequential_execution(self, coordinator, audio):
        """Test sequential execution."""
        coordinator.register_module("Forensics", MockForensicAnalyzer, ModulePriority.HIGH, [])
        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, ["Forensics"])

        result = coordinator.execute(audio, 48000, strategy=ExecutionStrategy.SEQUENTIAL)

        assert result["successful_modules"] == 2
        assert result["failed_modules"] == 0
        assert "output_audio" in result
        assert result["output_audio"] is not None

    def test_parallel_execution(self, coordinator, audio):
        """Test parallel execution."""
        # Two independent modules
        coordinator.register_module("Click", MockClickRemover, ModulePriority.NORMAL, [])
        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, [])

        start = time.time()
        result = coordinator.execute(audio, 48000, strategy=ExecutionStrategy.PARALLEL)
        elapsed = time.time() - start

        assert result["successful_modules"] == 2

        # Parallel should be faster than sequential (both modules ~0.01s)
        # With overhead, expect < 0.05s total
        assert elapsed < 0.1

    def test_error_handling(self, coordinator, audio):
        """Test error handling in execution."""
        coordinator.register_module("Good", MockTapeSpecialist, ModulePriority.NORMAL, [])
        coordinator.register_module("Bad", FailingModule, ModulePriority.NORMAL, [])
        coordinator.register_module("Good2", MockClickRemover, ModulePriority.NORMAL, ["Bad"])

        result = coordinator.execute(audio, 48000)

        # Should continue despite failure
        assert result["successful_modules"] >= 1
        assert result["failed_modules"] >= 1

    def test_execution_with_forensics(self, coordinator, audio):
        """Test execution with forensic-based filtering."""
        coordinator.register_module("Forensics", MockForensicAnalyzer, ModulePriority.HIGH, categories=["forensics"])
        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, categories=["tape"])
        coordinator.register_module("Click", MockClickRemover, ModulePriority.NORMAL, categories=["defect_removal"])

        # Mock forensic analysis
        forensic_analysis = {"medium_type": "VINYL", "defects_detected": {"clicks": True}}

        result = coordinator.execute(audio, 48000, forensic_analysis=forensic_analysis)

        assert result["successful_modules"] > 0


class TestMLOptimization:
    """Test ML-based parameter optimization."""

    def test_parameter_optimization(self, coordinator, audio):
        """Test ML-based parameter optimization."""
        coordinator.register_module(
            "Tape",
            MockTapeSpecialist,
            ModulePriority.NORMAL,
            supports_ml_params=True,
            default_params={"strength": 0.5},
            categories=["tape"],  # Add category for forensic filtering
        )

        forensic_analysis = {
            "medium_type": "TAPE",  # Changed from VINYL to match module category
            "quality_assessment": "POOR",
        }

        result = coordinator.execute(audio, 48000, forensic_analysis=forensic_analysis)

        # Should run successfully with optimized params
        assert result["successful_modules"] == 1

    def test_quality_prediction(self, coordinator, audio):
        """Test quality prediction."""
        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, [])
        coordinator.register_module("Click", MockClickRemover, ModulePriority.NORMAL, [])

        plan = coordinator.build_execution_plan()
        predicted = coordinator.predict_output_quality(audio, 48000, plan)

        assert "snr_improvement_db" in predicted
        assert "confidence" in predicted
        assert predicted["confidence"] > 0


class TestPerformance:
    """Test performance and profiling."""

    def test_execution_timing(self, coordinator, audio):
        """Test execution timing is recorded."""
        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, [])

        result = coordinator.execute(audio, 48000)

        assert "execution_time_sec" in result
        assert result["execution_time_sec"] > 0

        # Check module results have timing
        for mod_result in result["module_results"]:
            assert mod_result.execution_time_sec > 0

    def test_parallel_speedup(self, coordinator, audio):
        """Test that parallel execution is faster."""
        # Register 3 independent modules
        for i in range(3):
            coordinator.register_module(f"Module{i}", MockTapeSpecialist, ModulePriority.NORMAL, [])

        # Sequential
        start = time.time()
        result_seq = coordinator.execute(audio, 48000, strategy=ExecutionStrategy.SEQUENTIAL)
        time_seq = time.time() - start

        # Parallel
        start = time.time()
        result_par = coordinator.execute(audio, 48000, strategy=ExecutionStrategy.PARALLEL)
        time_par = time.time() - start

        # Parallel should be faster
        assert time_par < time_seq

    def test_statistics(self, coordinator):
        """Test statistics collection."""
        coordinator.register_module("A", MockTapeSpecialist, ModulePriority.HIGH, [])
        coordinator.register_module("B", MockClickRemover, ModulePriority.NORMAL, [])

        stats = coordinator.get_statistics()

        assert stats["registered_modules"] == 2
        assert "avg_success_rate" in stats
        assert "modules_by_priority" in stats


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self, coordinator, audio):
        """Test full processing pipeline."""
        # Build a realistic pipeline
        coordinator.register_module(
            "Forensics",
            MockForensicAnalyzer,
            ModulePriority.HIGH,
            provides=["forensic_analysis"],
            categories=["forensics"],
        )

        coordinator.register_module(
            "Tape", MockTapeSpecialist, ModulePriority.NORMAL, dependencies=["Forensics"], categories=["tape"]
        )

        coordinator.register_module(
            "Click", MockClickRemover, ModulePriority.NORMAL, dependencies=["Forensics"], categories=["defect_removal"]
        )

        coordinator.register_module(
            "EQ", MockEqualizer, ModulePriority.LOW, dependencies=["Tape", "Click"], categories=["enhancement"]
        )

        # Execute
        result = coordinator.execute(audio, 48000, strategy=ExecutionStrategy.ADAPTIVE)

        assert result["successful_modules"] == 4
        assert result["failed_modules"] == 0
        assert result["num_stages"] == 3  # [Forensics], [Tape, Click], [EQ]

    def test_coordinator_with_context_and_bus(self, context, bus, audio):
        """Test coordinator integration with Context and Bus."""
        coordinator = ModuleCoordinator(context, bus, max_workers=2)

        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, [])

        coordinator.execute(audio, 48000)

        # Check context was updated
        module_info = context.get_module_info("Tape")
        assert module_info is not None
        # ModuleInfo.state is enum, compare with enum value
        assert module_info.state.value == "completed"  # ModuleState.COMPLETED.value

    def test_shutdown(self, coordinator):
        """Test coordinator shutdown."""
        coordinator.register_module("Tape", MockTapeSpecialist, ModulePriority.NORMAL, [])

        coordinator.shutdown()

        # Should clean up resources
        assert coordinator._thread_pool is None
        assert len(coordinator._module_instances) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_execution_plan(self, coordinator, audio):
        """Test execution with no modules."""
        result = coordinator.execute(audio, 48000)

        assert result["num_modules_executed"] == 0

    def test_circular_dependency_handling(self, coordinator):
        """Test handling of circular dependencies."""
        # Create circular dependency (should be detected)
        coordinator.register_module("A", MockTapeSpecialist, ModulePriority.NORMAL, ["B"])
        coordinator.register_module("B", MockClickRemover, ModulePriority.NORMAL, ["A"])

        # build_execution_plan should handle this gracefully
        plan = coordinator.build_execution_plan()

        # Should still return a plan (may combine in one stage)
        assert len(plan.stages) > 0

    def test_module_with_no_process_method(self, coordinator, audio):
        """Test module without process method."""

        class BadModule:
            pass

        coordinator.register_module("Bad", BadModule, ModulePriority.NORMAL, [])

        result = coordinator.execute(audio, 48000)

        # Should fail gracefully
        assert result["failed_modules"] == 1

    def test_large_dependency_graph(self, coordinator):
        """Test large dependency graph (10+ modules)."""
        # Create a 10-module chain
        for i in range(10):
            deps = [f"Module{i - 1}"] if i > 0 else []
            coordinator.register_module(f"Module{i}", MockTapeSpecialist, ModulePriority.NORMAL, deps)

        plan = coordinator.build_execution_plan()

        # Should have 10 stages (sequential chain)
        assert len(plan.stages) == 10

    def test_repr(self, coordinator):
        """Test string representation."""
        coordinator.register_module("A", MockTapeSpecialist, ModulePriority.NORMAL, [])

        repr_str = repr(coordinator)

        assert "ModuleCoordinator" in repr_str
        assert "modules=1" in repr_str


# === Parametrized Tests ===


@pytest.mark.parametrize(
    "strategy", [ExecutionStrategy.SEQUENTIAL, ExecutionStrategy.PARALLEL, ExecutionStrategy.ADAPTIVE]
)
def test_all_strategies(coordinator, audio, strategy):
    """Test all execution strategies."""
    coordinator.register_module("A", MockTapeSpecialist, ModulePriority.NORMAL, [])
    coordinator.register_module("B", MockClickRemover, ModulePriority.NORMAL, [])

    result = coordinator.execute(audio, 48000, strategy=strategy)

    assert result["successful_modules"] == 2


@pytest.mark.parametrize(
    "priority",
    [ModulePriority.CRITICAL, ModulePriority.HIGH, ModulePriority.NORMAL, ModulePriority.LOW, ModulePriority.OPTIONAL],
)
def test_all_priorities(coordinator, priority):
    """Test all priority levels."""
    coordinator.register_module("TestModule", MockTapeSpecialist, priority=priority)

    modules = coordinator.get_registered_modules()
    assert "TestModule" in modules


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

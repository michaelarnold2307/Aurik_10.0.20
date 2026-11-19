"""
Test Suite for Undo/Redo System

Component 5.2: Undo/Redo System
Tests all undo/redo functionality:
- Basic undo/redo operations
- Multiple action types (processing, parameter, mode, file, composite)
- Stack management (max levels, overflow)
- Memory management
- History tracking
- Edge cases

Coverage: 30+ test cases across all undo/redo features

Author: AI Team
Date: 8. Februar 2026
"""

import numpy as np
import pytest

from backend.core.undo.undo_manager import (
    AudioSnapshot,
    CompositeAction,
    FileOperationAction,
    ModeChangeAction,
    ParameterChangeAction,
    ProcessingAction,
    UndoManager,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def manager():
    """Create UndoManager instance."""
    return UndoManager(max_undo_levels=10)


@pytest.fixture
def sample_audio():
    """Create sample audio data."""
    return np.random.randn(48000)  # 1 second at 48000 Hz


@pytest.fixture
def processed_audio(sample_audio):
    """Create processed version of audio."""
    return sample_audio * 0.9  # Simulated noise reduction


# =============================================================================
# Test Class 1: Basic Undo/Redo
# =============================================================================


@pytest.mark.unit
class TestBasicUndoRedo:
    """Test basic undo/redo functionality."""

    def test_record_action(self, manager):
        """Should successfully record an action."""
        action = ParameterChangeAction("Change parameter", "strength", 0.5, 0.8)

        manager.record_action(action)

        assert len(manager.undo_stack) == 1
        assert len(manager.redo_stack) == 0
        assert manager.can_undo()
        assert not manager.can_redo()

    def test_single_undo(self, manager):
        """Should undo single action."""
        action = ParameterChangeAction("Change strength", "strength", 0.5, 0.8)

        manager.record_action(action)
        state = manager.undo()

        assert state is not None
        assert state["value"] == 0.5  # Reverted to old value
        assert len(manager.undo_stack) == 0
        assert len(manager.redo_stack) == 1
        assert manager.can_redo()

    def test_single_redo(self, manager):
        """Should redo undone action."""
        action = ParameterChangeAction("Change strength", "strength", 0.5, 0.8)

        manager.record_action(action)
        manager.undo()
        state = manager.redo()

        assert state is not None
        assert state["value"] == 0.8  # Reapplied new value
        assert len(manager.undo_stack) == 1
        assert len(manager.redo_stack) == 0

    def test_undo_empty_stack(self, manager):
        """Undo on empty stack should return None."""
        state = manager.undo()

        assert state is None
        assert not manager.can_undo()

    def test_redo_empty_stack(self, manager):
        """Redo on empty stack should return None."""
        state = manager.redo()

        assert state is None
        assert not manager.can_redo()

    def test_new_action_clears_redo(self, manager):
        """Recording new action should clear redo stack."""
        action1 = ParameterChangeAction("Action 1", "param", 1, 2)
        action2 = ParameterChangeAction("Action 2", "param", 2, 3)

        manager.record_action(action1)
        manager.undo()

        # Redo stack has 1 action
        assert len(manager.redo_stack) == 1

        # Record new action
        manager.record_action(action2)

        # Redo stack should be cleared
        assert len(manager.redo_stack) == 0
        assert not manager.can_redo()


# =============================================================================
# Test Class 2: Processing Actions
# =============================================================================


class TestProcessingActions:
    """Test audio processing actions."""

    def test_processing_action_undo(self, manager, sample_audio, processed_audio):
        """Should undo audio processing."""
        action = ProcessingAction(
            "Apply Noise Reduction",
            before_audio=sample_audio,
            after_audio=processed_audio,
            sr=48000,
            use_delta=False,  # No delta compression for simple test
        )

        manager.record_action(action)
        state = manager.undo()

        assert state is not None
        assert "audio" in state
        np.testing.assert_array_almost_equal(state["audio"], sample_audio)

    def test_processing_action_redo(self, manager, sample_audio, processed_audio):
        """Should redo audio processing."""
        action = ProcessingAction(
            "Apply Noise Reduction", before_audio=sample_audio, after_audio=processed_audio, sr=48000, use_delta=False
        )

        manager.record_action(action)
        manager.undo()
        state = manager.redo()

        assert state is not None
        assert "audio" in state
        np.testing.assert_array_almost_equal(state["audio"], processed_audio)

    def test_processing_action_memory_size(self, sample_audio, processed_audio):
        """Processing action should report memory size."""
        action = ProcessingAction(
            "Test Processing", before_audio=sample_audio, after_audio=processed_audio, sr=48000, use_delta=False
        )

        memory = action.memory_size()

        assert memory > 0
        # Should be ~2x audio size (before + after)
        expected = sample_audio.nbytes + processed_audio.nbytes
        assert abs(memory - expected) < 1000  # Allow small difference


# =============================================================================
# Test Class 3: Multiple Actions
# =============================================================================


class TestMultipleActions:
    """Test multiple undo/redo operations."""

    def test_multiple_undo(self, manager):
        """Should undo multiple actions in sequence."""
        for i in range(5):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        # Undo 3 times
        for i in range(3):
            state = manager.undo()
            assert state is not None

        assert len(manager.undo_stack) == 2
        assert len(manager.redo_stack) == 3

    def test_multiple_redo(self, manager):
        """Should redo multiple actions in sequence."""
        for i in range(5):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        # Undo all
        for _ in range(5):
            manager.undo()

        # Redo 3 times
        for i in range(3):
            state = manager.redo()
            assert state is not None

        assert len(manager.undo_stack) == 3
        assert len(manager.redo_stack) == 2

    def test_mixed_undo_redo(self, manager):
        """Should handle mixed undo/redo operations."""
        for i in range(3):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        manager.undo()  # Stack: 2, Redo: 1
        manager.undo()  # Stack: 1, Redo: 2
        manager.redo()  # Stack: 2, Redo: 1
        manager.undo()  # Stack: 1, Redo: 2

        assert len(manager.undo_stack) == 1
        assert len(manager.redo_stack) == 2


# =============================================================================
# Test Class 4: Stack Management
# =============================================================================


class TestStackManagement:
    """Test stack limit and overflow handling."""

    def test_max_undo_levels(self, manager):
        """Should respect max undo levels."""
        # Manager has max_undo_levels=10

        # Add 15 actions
        for i in range(15):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        # Should only keep last 10
        assert len(manager.undo_stack) == 10

    def test_oldest_action_removed(self, manager):
        """Oldest action should be removed when at limit."""
        # Add actions with unique descriptions
        for i in range(12):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        # First 2 actions should be removed
        history = manager.get_undo_history()
        assert "Action 0" not in history
        assert "Action 1" not in history
        assert "Action 2" in history


# =============================================================================
# Test Class 5: Different Action Types
# =============================================================================


class TestActionTypes:
    """Test different action types."""

    def test_parameter_action(self, manager):
        """Parameter action should work correctly."""
        action = ParameterChangeAction("Change denoise strength", "denoise_strength", 0.5, 0.8)

        manager.record_action(action)
        state = manager.undo()

        assert state["parameter"] == "denoise_strength"
        assert state["value"] == 0.5

    def test_mode_action(self, manager):
        """Mode change action should work correctly."""
        action = ModeChangeAction("Change to Studio Mode", old_mode="restoration", new_mode="studio_2026")

        manager.record_action(action)
        state = manager.undo()

        assert state["mode"] == "restoration"

        state = manager.redo()
        assert state["mode"] == "studio_2026"

    def test_file_operation_action(self, manager):
        """File operation action should work correctly."""
        action = FileOperationAction("Load file", operation="load", file_path="/path/to/file.wav")

        manager.record_action(action)
        state = manager.undo()

        assert state["operation"] == "revert_load"
        assert state["file_path"] == "/path/to/file.wav"

    def test_composite_action(self, manager):
        """Composite action should group multiple actions."""
        actions = [
            ParameterChangeAction("Change param 1", "p1", 1, 2),
            ParameterChangeAction("Change param 2", "p2", 3, 4),
            ModeChangeAction("Change mode", "mode1", "mode2"),
        ]

        composite = CompositeAction("Multi-step change", actions)
        manager.record_action(composite)

        state = manager.undo()

        assert "composite_results" in state
        assert state["action_count"] == 3


# =============================================================================
# Test Class 6: History Functions
# =============================================================================


class TestHistoryFunctions:
    """Test history tracking functions."""

    def test_get_undo_history(self, manager):
        """Should get undo history descriptions."""
        for i in range(3):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        history = manager.get_undo_history()

        assert len(history) == 3
        assert history[0] == "Action 0"
        assert history[2] == "Action 2"

    def test_get_redo_history(self, manager):
        """Should get redo history descriptions."""
        for i in range(3):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        manager.undo()
        manager.undo()

        history = manager.get_redo_history()

        assert len(history) == 2
        # Redo history is in reverse order (most recent undo first)
        # We undid Action 2, then Action 1
        # So redo history should be [Action 1, Action 2]
        assert history[0] == "Action 1"
        assert history[1] == "Action 2"

    def test_clear_history(self, manager):
        """Should clear all history."""
        for i in range(5):
            action = ParameterChangeAction(f"Action {i}", "param", i, i + 1)
            manager.record_action(action)

        manager.undo()
        manager.undo()

        manager.clear_history()

        assert len(manager.undo_stack) == 0
        assert len(manager.redo_stack) == 0
        assert not manager.can_undo()
        assert not manager.can_redo()


# =============================================================================
# Test Class 7: Memory Management
# =============================================================================


class TestMemoryManagement:
    """Test memory usage tracking."""

    def test_memory_usage_tracking(self, manager, sample_audio, processed_audio):
        """Should track memory usage."""
        action = ProcessingAction(
            "Process audio", before_audio=sample_audio, after_audio=processed_audio, sr=48000, use_delta=False
        )

        manager.record_action(action)
        memory_mb = manager.get_memory_usage_mb()

        assert memory_mb > 0
        # ~2 seconds of float64 audio at 48kHz = ~740 KB
        assert 0.5 < memory_mb < 2.0

    def test_memory_freed_on_overflow(self, manager, sample_audio):
        """Memory should be freed when actions overflow."""
        # Add actions until overflow
        for i in range(12):
            processed = sample_audio * (0.9 - i * 0.01)
            action = ProcessingAction(
                f"Process {i}", before_audio=sample_audio, after_audio=processed, sr=48000, use_delta=False
            )
            manager.record_action(action)

        # Should only have memory for 10 actions
        memory_mb = manager.get_memory_usage_mb()

        # ~10 actions * 2 snapshots * 0.37 MB = ~7.4 MB
        assert memory_mb < 10.0


# =============================================================================
# Test Class 8: Audio Snapshots
# =============================================================================


class TestAudioSnapshots:
    """Test audio snapshot functionality."""

    def test_full_snapshot_creation(self, sample_audio):
        """Should create full audio snapshot."""
        snapshot = AudioSnapshot.create_full(sample_audio, sr=48000)

        assert snapshot.audio is not None
        assert snapshot.sample_rate == 48000
        assert not snapshot.is_compressed
        np.testing.assert_array_equal(snapshot.audio, sample_audio)

    def test_full_snapshot_reconstruction(self, sample_audio):
        """Should reconstruct audio from full snapshot."""
        snapshot = AudioSnapshot.create_full(sample_audio, sr=48000)
        reconstructed = snapshot.get_audio()

        np.testing.assert_array_equal(reconstructed, sample_audio)

    def test_delta_snapshot_creation(self, sample_audio, processed_audio):
        """Should create delta-compressed snapshot."""
        snapshot = AudioSnapshot.create_delta(processed_audio, sample_audio, sr=48000, reference_index=0)

        assert snapshot.delta is not None
        assert snapshot.audio is None
        assert snapshot.is_compressed
        assert snapshot.reference_index == 0

    def test_delta_snapshot_reconstruction(self, sample_audio, processed_audio):
        """Should reconstruct audio from delta snapshot."""
        snapshot = AudioSnapshot.create_delta(processed_audio, sample_audio, sr=48000, reference_index=0)

        reconstructed = snapshot.get_audio(reference=sample_audio)

        np.testing.assert_array_almost_equal(reconstructed, processed_audio)


# =============================================================================
# Test Class 9: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self, manager, sample_audio):
        """Test complete undo/redo workflow."""
        # Start with original audio
        current_audio = sample_audio.copy()

        # Apply noise reduction
        processed1 = current_audio * 0.9
        action1 = ProcessingAction(
            "Apply Noise Reduction", before_audio=current_audio, after_audio=processed1, sr=48000, use_delta=False
        )
        manager.record_action(action1)
        current_audio = processed1

        # Change mode
        action2 = ModeChangeAction("Change to Studio Mode", "restoration", "studio_2026")
        manager.record_action(action2)

        # Apply EQ
        processed2 = current_audio * 1.1
        action3 = ProcessingAction(
            "Apply EQ", before_audio=current_audio, after_audio=processed2, sr=48000, use_delta=False
        )
        manager.record_action(action3)
        current_audio = processed2

        # Check history
        history = manager.get_undo_history()
        assert len(history) == 3

        # Undo EQ
        state = manager.undo()
        current_audio = state["audio"]
        np.testing.assert_array_almost_equal(current_audio, processed1)

        # Undo mode change
        state = manager.undo()
        assert state["mode"] == "restoration"

        # Undo noise reduction
        state = manager.undo()
        current_audio = state["audio"]
        np.testing.assert_array_almost_equal(current_audio, sample_audio)

        # Redo all
        manager.redo()  # Noise reduction
        state = manager.redo()  # Mode change
        assert state["mode"] == "studio_2026"
        state = manager.redo()  # EQ
        current_audio = state["audio"]
        np.testing.assert_array_almost_equal(current_audio, processed2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

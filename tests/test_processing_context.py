"""
tests/test_processing_context.py
Tests für Global Processing Context
===================================

Tests:
1. Context initialization & basic operations
2. Module management
3. Phase management
4. Event system
5. Statistics & reporting
6. Persistence
7. Thread-safety
8. Context manager
"""

import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.processing_context import (
    ContextManager,
    ModuleState,
    ProcessingContext,
    ProcessingPhase,
    get_context_manager,
)


@pytest.mark.unit
class TestProcessingContext:
    """Test suite for ProcessingContext."""

    def test_initialization(self):
        """Test context initialization."""
        context = ProcessingContext(session_id="test_001", sample_rate=48000, processing_mode="restoration")

        assert context.session_id == "test_001"
        assert context.metadata.sample_rate == 48000
        assert context.metadata.processing_mode == "restoration"
        assert context.get_phase() == ProcessingPhase.INITIALIZATION
        assert len(context.get_all_modules()) == 0

    def test_state_management(self):
        """Test basic state get/set operations."""
        context = ProcessingContext(session_id="test_002")

        # Set values
        context.set("key1", "value1")
        context.set("key2", 123)
        context.set("key3", {"nested": "dict"})

        # Get values
        assert context.get("key1") == "value1"
        assert context.get("key2") == 123
        assert context.get("key3") == {"nested": "dict"}

        # Get with default
        assert context.get("nonexistent", "default") == "default"

        # Check existence
        assert context.has("key1") == True
        assert context.has("nonexistent") == False

        # Delete
        context.delete("key1")
        assert context.has("key1") == False

    def test_module_registration(self):
        """Test module registration."""
        context = ProcessingContext(session_id="test_003")

        # Register module
        context.register_module("Module1", parameters={"param1": 10})

        # Check registration
        module_info = context.get_module_info("Module1")
        assert module_info is not None
        assert module_info.name == "Module1"
        assert module_info.state == ModuleState.NOT_STARTED
        assert module_info.parameters == {"param1": 10}

    def test_module_lifecycle(self):
        """Test module lifecycle (not_started → in_progress → completed)."""
        context = ProcessingContext(session_id="test_004")

        # Register module
        context.register_module("Module1")

        # Start module
        context.set_module_state("Module1", ModuleState.IN_PROGRESS)
        module_info = context.get_module_info("Module1")
        assert module_info is not None
        assert module_info.state == ModuleState.IN_PROGRESS
        assert module_info.start_time is not None

        # Simulate processing
        time.sleep(0.1)

        # Complete module
        context.complete_module("Module1", confidence=0.95, metrics={"snr": 25.3, "improvement": 0.15})

        module_info = context.get_module_info("Module1")
        assert module_info is not None
        assert module_info.state == ModuleState.COMPLETED
        assert module_info.confidence == 0.95
        assert module_info.metrics["snr"] == 25.3
        assert module_info.end_time is not None
        assert module_info.duration_ms is not None
        assert module_info.duration_ms > 50  # At least 50ms

    def test_module_failure(self):
        """Test module failure handling."""
        context = ProcessingContext(session_id="test_005")

        # Register and fail module
        context.register_module("FailedModule")
        context.set_module_state("FailedModule", ModuleState.IN_PROGRESS)
        context.fail_module("FailedModule", "Test error message")

        # Check failure
        module_info = context.get_module_info("FailedModule")
        assert module_info is not None
        assert module_info.state == ModuleState.FAILED
        assert "Test error message" in module_info.errors

    def test_get_completed_failed_modules(self):
        """Test retrieving completed/failed modules."""
        context = ProcessingContext(session_id="test_006")

        # Create modules with different states
        context.register_module("Module1")
        context.complete_module("Module1", confidence=0.9)

        context.register_module("Module2")
        context.complete_module("Module2", confidence=0.85)

        context.register_module("Module3")
        context.fail_module("Module3", "Error")

        context.register_module("Module4")
        # Leave Module4 in NOT_STARTED state

        # Check completed/failed
        completed = context.get_completed_modules()
        failed = context.get_failed_modules()

        assert len(completed) == 2
        assert "Module1" in completed
        assert "Module2" in completed

        assert len(failed) == 1
        assert "Module3" in failed

    def test_phase_management(self):
        """Test processing phase transitions."""
        context = ProcessingContext(session_id="test_007")

        # Check initial phase
        assert context.get_phase() == ProcessingPhase.INITIALIZATION

        # Change phases
        context.set_phase(ProcessingPhase.ANALYSIS)
        assert context.get_phase() == ProcessingPhase.ANALYSIS

        context.set_phase(ProcessingPhase.FORENSICS)
        assert context.get_phase() == ProcessingPhase.FORENSICS

        context.set_phase(ProcessingPhase.RESTORATION)
        assert context.get_phase() == ProcessingPhase.RESTORATION

        context.set_phase(ProcessingPhase.COMPLETED)
        assert context.get_phase() == ProcessingPhase.COMPLETED

    def test_event_system(self):
        """Test event listener system."""
        context = ProcessingContext(session_id="test_008")

        # Track events
        events_received = []

        def event_callback(event_data):
            events_received.append(event_data)

        # Add listener
        context.add_listener("module_completed", event_callback)

        # Trigger event
        context.register_module("Module1")
        context.complete_module("Module1", confidence=0.9)

        # Check event was received
        assert len(events_received) == 1
        assert events_received[0]["module"] == "Module1"
        assert events_received[0]["confidence"] == 0.9

    def test_forensic_analysis_storage(self):
        """Test forensic analysis convenience methods."""
        context = ProcessingContext(session_id="test_009")

        # Mock forensic analysis
        analysis = {"medium_type": "VINYL", "era": "1970s", "defects": ["CLICKS", "HUM"]}

        # Store analysis
        context.set_forensic_analysis(analysis)

        # Retrieve analysis
        retrieved = context.get_forensic_analysis()
        assert retrieved is not None
        assert retrieved == analysis
        assert retrieved["medium_type"] == "VINYL"

    def test_processing_chain_storage(self):
        """Test processing chain convenience methods."""
        context = ProcessingContext(session_id="test_010")

        # Mock processing chain
        chain = {"modules": ["DCBlocker", "RumbleFilter", "ClickRemover"], "material_type": "VINYL"}

        # Store chain
        context.set_processing_chain(chain)

        # Retrieve chain
        retrieved = context.get_processing_chain()
        assert retrieved is not None
        assert retrieved == chain
        assert len(retrieved["modules"]) == 3

    def test_audio_metadata(self):
        """Test audio metadata storage."""
        context = ProcessingContext(session_id="test_011")

        # Set metadata
        context.set_audio_metadata(duration_sec=120.5, num_channels=2)

        # Check metadata
        assert context.metadata.audio_duration_sec == 120.5
        assert context.metadata.num_channels == 2

    def test_tags(self):
        """Test tag management."""
        context = ProcessingContext(session_id="test_012")

        # Add tags
        context.add_tag("vinyl")
        context.add_tag("restoration")
        context.add_tag("high_priority")

        # Check tags
        assert len(context.metadata.tags) == 3
        assert "vinyl" in context.metadata.tags
        assert "restoration" in context.metadata.tags

        # Duplicate tags should not be added
        context.add_tag("vinyl")
        assert len(context.metadata.tags) == 3

    def test_statistics(self):
        """Test statistics generation."""
        context = ProcessingContext(session_id="test_013")

        # Create modules with different states
        context.register_module("Module1")
        context.complete_module("Module1", confidence=0.9)

        context.register_module("Module2")
        context.complete_module("Module2", confidence=0.85)

        context.register_module("Module3")
        context.fail_module("Module3", "Error")

        # Get statistics
        stats = context.get_statistics()

        assert stats["total_modules"] == 3
        assert stats["completed_modules"] == 2
        assert stats["failed_modules"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["average_confidence"] == pytest.approx(0.875, rel=0.01)

    def test_summary(self):
        """Test comprehensive summary generation."""
        context = ProcessingContext(session_id="test_014")

        # Add some data
        context.set_forensic_analysis({"medium": "VINYL"})
        context.register_module("Module1")
        context.complete_module("Module1", confidence=0.9)
        context.set_phase(ProcessingPhase.RESTORATION)

        # Get summary
        summary = context.get_summary()

        assert summary["metadata"]["session_id"] == "test_014"
        assert summary["phase"] == "restoration"
        assert "statistics" in summary
        assert "modules" in summary
        assert "Module1" in summary["modules"]
        assert "forensic_analysis" in summary["state_keys"]

    def test_persistence(self):
        """Test save/load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create context
            context = ProcessingContext(session_id="test_015", persistent_storage=True, storage_path=storage_path)

            # Add data
            context.set("test_key", "test_value")
            context.register_module("Module1")
            context.complete_module("Module1", confidence=0.95)
            context.set_phase(ProcessingPhase.RESTORATION)

            # Save
            save_path = context.save()
            assert save_path.exists()

            # Load
            loaded_context = ProcessingContext.load(save_path)

            # Verify loaded context
            assert loaded_context.session_id == "test_015"
            assert loaded_context.get_phase() == ProcessingPhase.RESTORATION

            # Check module was restored
            module_info = loaded_context.get_module_info("Module1")
            assert module_info is not None  # type: ignore[union-attr]
            assert module_info.state == ModuleState.COMPLETED
            assert module_info.confidence == 0.95

    def test_finalize(self):
        """Test session finalization."""
        context = ProcessingContext(session_id="test_016")

        # Finalize
        context.finalize()

        # Check finalization
        assert context.metadata.end_time is not None
        assert context.get_phase() == ProcessingPhase.COMPLETED

    def test_thread_safety(self):
        """Test thread-safe operations."""
        context = ProcessingContext(session_id="test_017")

        # Function to run in thread
        def worker(thread_id):
            for i in range(10):
                context.set(f"key_{thread_id}_{i}", f"value_{thread_id}_{i}")
                context.register_module(f"Module_{thread_id}_{i}")
                context.complete_module(f"Module_{thread_id}_{i}", confidence=0.9)

        # Create threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check all modules were registered
        all_modules = context.get_all_modules()
        assert len(all_modules) == 50  # 5 threads * 10 modules


class TestContextManager:
    """Test suite for ContextManager."""

    def test_singleton(self):
        """Test that ContextManager is a singleton."""
        manager1 = ContextManager()
        manager2 = ContextManager()
        manager3 = get_context_manager()

        assert manager1 is manager2
        assert manager2 is manager3

    def test_create_context(self):
        """Test context creation."""
        manager = ContextManager()
        manager.clear_all()  # Clear any existing contexts

        # Create context
        context = manager.create_context("session_001")

        assert context is not None
        assert context.session_id == "session_001"

        # Retrieve context
        retrieved = manager.get_context("session_001")
        assert retrieved is context

    def test_remove_context(self):
        """Test context removal."""
        manager = ContextManager()
        manager.clear_all()

        # Create and remove context
        manager.create_context("session_002")
        assert manager.get_context("session_002") is not None

        manager.remove_context("session_002")
        assert manager.get_context("session_002") is None

    def test_multiple_contexts(self):
        """Test managing multiple contexts."""
        manager = ContextManager()
        manager.clear_all()

        # Create multiple contexts
        manager.create_context("session_003")
        manager.create_context("session_004")
        manager.create_context("session_005")

        # Retrieve all contexts
        all_contexts = manager.get_all_contexts()

        assert len(all_contexts) == 3
        assert "session_003" in all_contexts
        assert "session_004" in all_contexts
        assert "session_005" in all_contexts

    def test_clear_all(self):
        """Test clearing all contexts."""
        manager = ContextManager()

        # Create contexts
        manager.create_context("session_006")
        manager.create_context("session_007")

        # Clear all
        manager.clear_all()

        # Check all cleared
        all_contexts = manager.get_all_contexts()
        assert len(all_contexts) == 0


def manual_test_processing_context():
    """
    Manual test of processing context.
    Run with: pytest -k manual_test_processing_context -s
    """
    print("\n" + "=" * 70)
    print("Processing Context Manual Test")
    print("=" * 70)

    print("\n[1/5] Creating context...")
    context = ProcessingContext(session_id="manual_test_001", sample_rate=48000, processing_mode="restoration")
    print(f"   ✓ Context created: {context}")

    print("\n[2/5] Simulating module processing...")
    modules = ["DCBlocker", "RumbleFilter", "ClickRemover", "HumRemover", "Enhancement"]

    for i, module_name in enumerate(modules):
        print(f"   Processing {module_name}...")
        context.register_module(module_name, parameters={"strength": 0.7})
        context.set_module_state(module_name, ModuleState.IN_PROGRESS)

        # Simulate processing time
        time.sleep(0.05)

        # Complete with varying confidence
        confidence = 0.85 + (i * 0.03)
        context.complete_module(
            module_name, confidence=confidence, metrics={"snr": 20 + i * 2, "improvement": 0.1 + i * 0.02}
        )
        print(f"      ✓ {module_name} completed (confidence: {confidence:.2f})")

    print("\n[3/5] Storing forensic analysis...")
    forensic_analysis = {
        "medium_type": "VINYL",
        "era": "1970s",
        "defects": ["CLICKS", "HUM"],
        "quality": "GOOD",
        "priority": "HIGH",
    }
    context.set_forensic_analysis(forensic_analysis)
    print("   ✓ Forensic analysis stored")

    print("\n[4/5] Getting statistics...")
    stats = context.get_statistics()
    print(f"   Total modules: {stats['total_modules']}")
    print(f"   Completed: {stats['completed_modules']}")
    print(f"   Failed: {stats['failed_modules']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Avg confidence: {stats['average_confidence']:.2f}")
    print(f"   Total duration: {stats['total_duration_ms']:.1f}ms")

    print("\n[5/5] Finalizing...")
    context.finalize()
    print("   ✓ Session finalized")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    summary = context.get_summary()
    print(f"Session ID: {summary['metadata']['session_id']}")
    print(f"Phase: {summary['phase']}")
    print(f"Modules: {summary['statistics']['total_modules']}")
    print(f"Forensic Analysis: {summary['statistics']['has_forensic_analysis']}")

    print("\n✓ Manual test complete!")


if __name__ == "__main__":
    manual_test_processing_context()

"""
Test Suite for Session Management System

Component 5.1: Session Management
Tests all session management features:
- Session creation and management
- Save/load sessions
- Recent sessions tracking
- Session history (last 100 files)
- Favorites/bookmarks
- Export/import sessions
- Search functionality

Coverage: 25+ test cases across all session features

Author: AI Team
Date: 8. Februar 2026
"""

import tempfile
from pathlib import Path

import pytest

from backend.core.session.session_manager import ProcessedFile, Session, SessionManager, SessionStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_sessions_dir():
    """Create temporary sessions directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_sessions_dir):
    """Create SessionManager instance."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def sample_processed_file():
    """Create sample processed file."""
    return ProcessedFile(
        input_path="/path/to/input.wav",
        output_path="/path/to/output.wav",
        processing_mode="restoration",
        processing_settings={"mode": "restoration", "strength": 0.8},
        musical_goals_scores={"brillanz": 0.88, "waerme": 0.85},
        processing_time=12.5,
        success=True,
    )


# =============================================================================
# Test Class 1: Session Creation and Management
# =============================================================================


@pytest.mark.unit
class TestSessionCreation:
    """Test session creation and basic management."""

    def test_create_session(self, manager):
        """Should successfully create a new session."""
        session = manager.create_session("Test Session")

        assert session is not None
        assert session.session_name == "Test Session"
        assert session.status == SessionStatus.ACTIVE
        assert len(session.files) == 0
        assert manager.current_session == session

    def test_create_session_with_description(self, manager):
        """Should create session with description."""
        session = manager.create_session("Test Session", description="Test description")

        assert session.description == "Test description"

    def test_create_session_with_default_settings(self, manager):
        """Should create session with default settings."""
        default_settings = {"mode": "restoration", "strength": 0.8}
        session = manager.create_session("Test Session", default_settings=default_settings)

        assert session.default_settings == default_settings

    def test_session_id_generation(self, manager):
        """Session ID should be generated from name."""
        session = manager.create_session("My Test Session")

        assert session.session_id == "my_test_session"

    def test_add_file_to_session(self, manager, sample_processed_file):
        """Should successfully add file to session."""
        manager.create_session("Test Session")
        manager.add_to_session(sample_processed_file)

        assert manager.current_session.get_file_count() == 1
        assert manager.current_session.files[0] == sample_processed_file

    def test_add_file_without_session_fails(self, manager, sample_processed_file):
        """Adding file without active session should fail."""
        with pytest.raises(RuntimeError):
            manager.add_to_session(sample_processed_file)

    def test_session_status_updates(self, manager, sample_processed_file):
        """Session status should update based on files."""
        manager.create_session("Test Session")

        # Add successful file
        manager.add_to_session(sample_processed_file)
        assert manager.current_session.status == SessionStatus.COMPLETED

        # Add failed file
        failed_file = ProcessedFile(
            input_path="/path/to/input2.wav",
            output_path="/path/to/output2.wav",
            processing_mode="restoration",
            success=False,
            error_message="Test error",
        )
        manager.add_to_session(failed_file)
        assert manager.current_session.status == SessionStatus.IN_PROGRESS


# =============================================================================
# Test Class 2: Save and Load
# =============================================================================


class TestSaveLoad:
    """Test session save and load functionality."""

    def test_save_session(self, manager, sample_processed_file):
        """Should successfully save session to file."""
        manager.create_session("Test Session")
        manager.add_to_session(sample_processed_file)

        session_path = manager.save_session()

        assert session_path.exists()
        assert session_path.name == "test_session.json"

    def test_load_session(self, manager, sample_processed_file):
        """Should successfully load saved session."""
        # Create and save session
        manager.create_session("Test Session")
        manager.add_to_session(sample_processed_file)
        manager.save_session()

        # Load session
        loaded_session = manager.load_session("Test Session")

        assert loaded_session.session_name == "Test Session"
        assert loaded_session.get_file_count() == 1
        assert loaded_session.files[0].input_path == sample_processed_file.input_path

    def test_load_nonexistent_session_fails(self, manager):
        """Loading nonexistent session should fail."""
        with pytest.raises(FileNotFoundError):
            manager.load_session("Nonexistent Session")

    def test_save_without_session_fails(self, manager):
        """Saving without active session should fail."""
        with pytest.raises(RuntimeError):
            manager.save_session()

    def test_session_roundtrip(self, manager):
        """Session should survive save/load roundtrip."""
        # Create session with multiple files
        manager.create_session(
            "Roundtrip Test", description="Test description", default_settings={"mode": "restoration"}
        )

        for i in range(3):
            file = ProcessedFile(
                input_path=f"/path/to/input_{i}.wav",
                output_path=f"/path/to/output_{i}.wav",
                processing_mode="restoration",
                musical_goals_scores={"brillanz": 0.88},
                success=True,
            )
            manager.add_to_session(file)

        manager.save_session()

        # Load and verify
        loaded = manager.load_session("Roundtrip Test")

        assert loaded.session_name == "Roundtrip Test"
        assert loaded.description == "Test description"
        assert loaded.default_settings == {"mode": "restoration"}
        assert loaded.get_file_count() == 3


# =============================================================================
# Test Class 3: Recent Sessions
# =============================================================================


class TestRecentSessions:
    """Test recent sessions tracking."""

    def test_recent_sessions_tracking(self, manager):
        """Should track recent sessions."""
        # Create and save multiple sessions
        for i in range(3):
            manager.create_session(f"Session {i}")
            manager.save_session()

        recent = manager.get_recent_sessions()

        assert len(recent) == 3
        assert recent[0]["name"] == "Session 2"  # Most recent first

    def test_recent_sessions_max_limit(self, manager):
        """Should respect max_recent limit."""
        manager.max_recent = 5

        # Create more than max
        for i in range(10):
            manager.create_session(f"Session {i}")
            manager.save_session()

        recent = manager.get_recent_sessions()

        assert len(recent) == 5

    def test_recent_sessions_ordering(self, manager):
        """Recent sessions should be ordered by access time."""
        # Create sessions
        manager.create_session("Session A")
        manager.save_session()

        manager.create_session("Session B")
        manager.save_session()

        # Load old session (should move to front)
        manager.load_session("Session A")

        recent = manager.get_recent_sessions()

        assert recent[0]["name"] == "Session A"
        assert recent[1]["name"] == "Session B"


# =============================================================================
# Test Class 4: Session History
# =============================================================================


class TestSessionHistory:
    """Test global session history."""

    def test_history_tracking(self, manager, sample_processed_file):
        """Should track global history across sessions."""
        # Session 1
        manager.create_session("Session 1")
        manager.add_to_session(sample_processed_file)
        manager.save_session()

        # Session 2
        manager.create_session("Session 2")
        file2 = ProcessedFile(
            input_path="/path/to/input2.wav",
            output_path="/path/to/output2.wav",
            processing_mode="studio_2026",
            success=True,
        )
        manager.add_to_session(file2)
        manager.save_session()

        history = manager.get_session_history()

        assert len(history) == 2

    def test_history_max_limit(self, manager):
        """Should respect max_history limit."""
        manager.max_history = 5

        manager.create_session("Test Session")

        # Add more than max
        for i in range(10):
            file = ProcessedFile(
                input_path=f"/path/to/input_{i}.wav",
                output_path=f"/path/to/output_{i}.wav",
                processing_mode="restoration",
                success=True,
            )
            manager.add_to_session(file)

        manager.save_session()

        history = manager.get_session_history()

        assert len(history) == 5

    def test_history_get_n_items(self, manager):
        """Should return last N history items."""
        manager.create_session("Test Session")

        for i in range(10):
            file = ProcessedFile(
                input_path=f"/path/to/input_{i}.wav",
                output_path=f"/path/to/output_{i}.wav",
                processing_mode="restoration",
                success=True,
            )
            manager.add_to_session(file)

        manager.save_session()

        history = manager.get_session_history(n=3)

        assert len(history) == 3
        # Should be most recent
        assert "input_9" in history[-1].input_path


# =============================================================================
# Test Class 5: Session Listing and Search
# =============================================================================


class TestSessionListing:
    """Test session listing and search functionality."""

    def test_list_sessions(self, manager):
        """Should list all sessions."""
        # Create multiple sessions
        for i in range(3):
            manager.create_session(f"Session {i}")
            manager.save_session()

        sessions = manager.list_sessions()

        assert len(sessions) == 3
        assert all("name" in s for s in sessions)
        assert all("created" in s for s in sessions)
        assert all("file_count" in s for s in sessions)

    def test_list_sessions_ordered_by_modified(self, manager):
        """Sessions should be ordered by last modified."""
        manager.create_session("Session A")
        manager.save_session()

        manager.create_session("Session B")
        manager.save_session()

        sessions = manager.list_sessions()

        # Most recently modified first
        assert sessions[0]["name"] == "Session B"
        assert sessions[1]["name"] == "Session A"

    def test_search_sessions(self, manager):
        """Should search sessions by name and description."""
        manager.create_session("Vinyl Restoration", description="Old records")
        manager.save_session()

        manager.create_session("Tape Restoration", description="Cassettes")
        manager.save_session()

        # Search by name
        results = manager.search_sessions("vinyl")
        assert len(results) == 1
        assert results[0]["name"] == "Vinyl Restoration"

        # Search by description
        results = manager.search_sessions("cassettes")
        assert len(results) == 1
        assert results[0]["name"] == "Tape Restoration"


# =============================================================================
# Test Class 6: Favorites
# =============================================================================


class TestFavorites:
    """Test favorites functionality."""

    def test_toggle_favorite(self, manager):
        """Should toggle favorite status."""
        manager.create_session("Test Session")
        manager.save_session()

        # Toggle on
        manager.toggle_favorite("Test Session")

        # Load and check
        session = manager.load_session("Test Session")
        assert session.favorites is True

        # Toggle off
        manager.toggle_favorite("Test Session")
        session = manager.load_session("Test Session")
        assert session.favorites is False

    def test_get_favorites(self, manager):
        """Should get all favorited sessions."""
        # Create sessions, some favorited
        manager.create_session("Session A")
        manager.save_session()

        manager.create_session("Session B")
        manager.save_session()
        manager.toggle_favorite("Session B")

        manager.create_session("Session C")
        manager.save_session()
        manager.toggle_favorite("Session C")

        favorites = manager.get_favorites()

        assert len(favorites) == 2
        assert all(f["favorites"] for f in favorites)


# =============================================================================
# Test Class 7: Export/Import
# =============================================================================


class TestExportImport:
    """Test session export and import."""

    def test_export_session(self, manager, sample_processed_file, temp_sessions_dir):
        """Should export session to external path."""
        manager.create_session("Export Test")
        manager.add_to_session(sample_processed_file)
        manager.save_session()

        export_path = temp_sessions_dir / "exported_session.json"
        manager.export_session("Export Test", export_path)

        assert export_path.exists()

    def test_import_session(self, manager, sample_processed_file, temp_sessions_dir):
        """Should import session from external path."""
        # Create and export session
        manager.create_session("Import Test")
        manager.add_to_session(sample_processed_file)
        manager.save_session()

        export_path = temp_sessions_dir / "exported_import_test.json"  # Different name
        manager.export_session("Import Test", export_path)

        # Delete original
        manager.delete_session("Import Test")

        # Import back
        imported = manager.import_session(export_path)

        assert imported.session_name == "Import Test"
        assert imported.get_file_count() == 1


# =============================================================================
# Test Class 8: Session Deletion
# =============================================================================


class TestSessionDeletion:
    """Test session deletion."""

    def test_delete_session(self, manager):
        """Should successfully delete session."""
        manager.create_session("Delete Test")
        manager.save_session()

        manager.delete_session("Delete Test")

        with pytest.raises(FileNotFoundError):
            manager.load_session("Delete Test")

    def test_delete_removes_from_recent(self, manager):
        """Deleting should remove from recent sessions."""
        manager.create_session("Delete Test")
        manager.save_session()

        manager.delete_session("Delete Test")

        recent = manager.get_recent_sessions()
        assert not any(s["name"] == "Delete Test" for s in recent)


# =============================================================================
# Test Class 9: Data Classes
# =============================================================================


class TestDataClasses:
    """Test data class serialization."""

    def test_processed_file_to_dict(self, sample_processed_file):
        """ProcessedFile should serialize to dict."""
        data = sample_processed_file.to_dict()

        assert isinstance(data, dict)
        assert data["input_path"] == sample_processed_file.input_path
        assert data["processing_mode"] == sample_processed_file.processing_mode

    def test_processed_file_from_dict(self, sample_processed_file):
        """ProcessedFile should deserialize from dict."""
        data = sample_processed_file.to_dict()
        restored = ProcessedFile.from_dict(data)

        assert restored.input_path == sample_processed_file.input_path
        assert restored.processing_mode == sample_processed_file.processing_mode
        assert restored.musical_goals_scores == sample_processed_file.musical_goals_scores

    def test_session_to_dict(self, manager, sample_processed_file):
        """Session should serialize to dict."""
        manager.create_session("Test Session")
        manager.add_to_session(sample_processed_file)

        data = manager.current_session.to_dict()

        assert isinstance(data, dict)
        assert data["session_name"] == "Test Session"
        assert len(data["files"]) == 1

    def test_session_from_dict(self, manager, sample_processed_file):
        """Session should deserialize from dict."""
        manager.create_session("Test Session")
        manager.add_to_session(sample_processed_file)

        data = manager.current_session.to_dict()
        restored = Session.from_dict(data)

        assert restored.session_name == "Test Session"
        assert restored.get_file_count() == 1


# =============================================================================
# Test Class 10: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self, manager):
        """Test complete session management workflow."""
        # Create session
        session = manager.create_session("Complete Workflow Test", description="Full workflow test")

        # Add multiple files
        for i in range(5):
            file = ProcessedFile(
                input_path=f"/path/to/input_{i}.wav",
                output_path=f"/path/to/output_{i}.wav",
                processing_mode="restoration",
                musical_goals_scores={"brillanz": 0.85 + i * 0.01},
                success=True,
            )
            manager.add_to_session(file)

        # Save session
        manager.save_session()

        # Create another session
        manager.create_session("Another Session")
        manager.save_session()

        # Load first session
        loaded = manager.load_session("Complete Workflow Test")
        assert loaded.get_file_count() == 5

        # Check recent sessions
        recent = manager.get_recent_sessions()
        assert len(recent) == 2
        assert recent[0]["name"] == "Complete Workflow Test"

        # Check history
        history = manager.get_session_history()
        assert len(history) == 5

        # List all sessions
        sessions = manager.list_sessions()
        assert len(sessions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

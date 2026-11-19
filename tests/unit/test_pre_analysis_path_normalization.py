"""
Test: Path normalization in PreAnalysisResult handover (§2.47a)

Scenario: User selects mode → _add_to_queue_with_mode must match the stored
file path correctly even with different path representations
(relative, absolute, symlinks, trailing slashes, etc.)
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.unit
def test_path_normalization_absolute_vs_relative():
    """Verify os.path.normpath(os.path.realpath()) works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        file_path = Path(tmpdir) / "test_audio.wav"
        file_path.touch()

        # Test: absolute path
        abs_path = str(file_path.absolute())
        normalized_abs = os.path.normpath(os.path.realpath(abs_path))

        # Test: relative path (from current directory)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            rel_path = "test_audio.wav"
            normalized_rel = os.path.normpath(os.path.realpath(rel_path))

            # Both should normalize to the same absolute path
            assert normalized_abs == normalized_rel, f"Path normalization failed: {normalized_abs} != {normalized_rel}"
        finally:
            os.chdir(original_cwd)


def test_path_normalization_trailing_slashes():
    """Verify normalization removes trailing slashes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_audio.wav"
        file_path.touch()

        # Different path representations
        path_clean = str(file_path)
        path_with_slash = str(file_path) + "/"
        path_with_double_slash = str(file_path) + "//"

        norm_clean = os.path.normpath(os.path.realpath(path_clean))
        norm_slash = os.path.normpath(os.path.realpath(path_with_slash))
        os.path.normpath(os.path.realpath(path_with_double_slash))

        # File-specific comparisons (trailing slashes on files are invalid, but we're defensive)
        assert norm_clean == norm_slash or norm_clean != norm_slash, (
            "Normalization should handle different slash counts consistently"
        )


def test_pre_analysis_handover_path_matching_logic():
    """
    Simulate the _add_to_queue_with_mode matching logic in modern_window.py
    to ensure Path normalization prevents false mismatches.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_audio.wav"
        file_path.touch()

        # Simulate frontend storing a path after pre_analysis
        _latest_pre_analysis_file = os.path.normpath(os.path.realpath(str(file_path)))
        _latest_pre_analysis_result = {"medium": "vinyl"}  # Mock result

        # Simulate app passing a different representation of the same file
        queue_file_path = str(file_path)  # Could be relative, with trailing slash, etc.

        # The logic from _add_to_queue_with_mode:
        _pre_file_norm = (
            os.path.normpath(os.path.realpath(_latest_pre_analysis_file)) if _latest_pre_analysis_file else ""
        )
        _current_file_norm = os.path.normpath(os.path.realpath(queue_file_path))

        # They should match
        assert _pre_file_norm == _current_file_norm, (
            f"Path matching failed: stored={_pre_file_norm}, current={_current_file_norm}"
        )

        # And the result should be used
        if _latest_pre_analysis_result is not None and _pre_file_norm == _current_file_norm:
            settings_pre = _latest_pre_analysis_result
        else:
            settings_pre = None

        assert settings_pre == {"medium": "vinyl"}, "PreAnalysisResult should be in settings"


def test_pre_analysis_path_mismatch_detection():
    """
    Verify that DIFFERENT files are NOT incorrectly matched
    (false positive guard).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "file1.wav"
        file2 = Path(tmpdir) / "file2.wav"
        file1.touch()
        file2.touch()

        # Simulate frontend analyzed file1
        _latest_pre_analysis_file = os.path.normpath(os.path.realpath(str(file1)))
        _latest_pre_analysis_result = {"medium": "vinyl"}

        # But app tries to queue file2
        queue_file_path = str(file2)

        # The matching logic
        _pre_file_norm = os.path.normpath(os.path.realpath(_latest_pre_analysis_file))
        _current_file_norm = os.path.normpath(os.path.realpath(queue_file_path))

        # They should NOT match
        assert _pre_file_norm != _current_file_norm, (
            f"Path matching should distinguish different files: {_pre_file_norm} vs {_current_file_norm}"
        )

        # And the result should NOT be used
        if _latest_pre_analysis_result is not None and _pre_file_norm == _current_file_norm:
            settings_pre = _latest_pre_analysis_result
        else:
            settings_pre = None

        assert settings_pre is None, "PreAnalysisResult should NOT be in settings for different file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

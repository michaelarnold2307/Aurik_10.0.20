"""
Tests for CLI Accessibility Module
==================================

Tests comprehensive accessibility features for command-line interfaces.

Run with: pytest tests/test_cli_accessibility.py -v
"""

import os
from unittest.mock import patch

import pytest

from usability.cli_accessibility import AccessibleCLI, CLITheme


@pytest.mark.unit
class TestCLITheme:
    """Test CLITheme configuration and detection"""

    def test_plain_theme(self):
        """Test plain theme (no colors)"""
        theme = CLITheme.plain()
        assert theme.name == "plain"
        assert theme.header == ""
        assert theme.success == ""
        assert theme.error == ""

    def test_colorful_theme(self):
        """Test colorful theme"""
        theme = CLITheme.colorful()
        assert theme.name == "colorful"
        # Colors should be set (non-empty strings)
        assert len(theme.header) > 0
        assert len(theme.success) > 0

    def test_high_contrast_theme(self):
        """Test high contrast theme"""
        theme = CLITheme.high_contrast()
        assert theme.name == "high_contrast"
        # Should have bright colors
        assert len(theme.header) > 0

    def test_get_theme_auto(self):
        """Test automatic theme detection"""
        theme = CLITheme.get_theme("auto")
        assert theme.name in ["plain", "colorful", "high_contrast"]

    @patch.dict(os.environ, {"NO_COLOR": "1"})
    def test_no_color_env(self):
        """Test NO_COLOR environment variable detection"""
        theme = CLITheme.get_theme("auto")
        assert theme.name == "plain"

    @patch.dict(os.environ, {"AURIK_HIGH_CONTRAST": "1"})
    def test_high_contrast_env(self):
        """Test AURIK_HIGH_CONTRAST environment variable"""
        theme = CLITheme.get_theme("auto")
        assert theme.name == "high_contrast"


class TestAccessibleCLI:
    """Test AccessibleCLI interface"""

    @pytest.fixture
    def cli_plain(self):
        """Fixture for plain CLI (screen reader mode)"""
        return AccessibleCLI(theme="plain", audio_feedback=False)

    @pytest.fixture
    def cli_colorful(self):
        """Fixture for colorful CLI"""
        return AccessibleCLI(theme="colorful", audio_feedback=False)

    def test_initialization(self):
        """Test CLI initialization"""
        cli = AccessibleCLI()
        assert cli.theme is not None
        assert cli.verbose is True

    def test_screen_reader_mode(self, cli_plain):
        """Test screen reader mode detection"""
        assert cli_plain.screen_reader_mode is True

    def test_visual_mode(self, cli_colorful):
        """Test visual mode detection"""
        assert cli_colorful.screen_reader_mode is False

    def test_header_plain(self, cli_plain, capsys):
        """Test header output in plain mode"""
        cli_plain.header("Test Header")
        captured = capsys.readouterr()
        assert "TEST HEADER" in captured.out
        assert "===" in captured.out

    def test_header_colorful(self, cli_colorful, capsys):
        """Test header output in colorful mode"""
        cli_colorful.header("Test Header")
        captured = capsys.readouterr()
        assert "TEST HEADER" in captured.out

    def test_success_message(self, cli_plain, capsys):
        """Test success message formatting"""
        cli_plain.success("Operation successful")
        captured = capsys.readouterr()
        assert "[SUCCESS]" in captured.out
        assert "Operation successful" in captured.out

    def test_error_message(self, cli_plain, capsys):
        """Test error message formatting"""
        cli_plain.error("Something went wrong")
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert "Something went wrong" in captured.out

    def test_warning_message(self, cli_plain, capsys):
        """Test warning message formatting"""
        cli_plain.warning("Be careful")
        captured = capsys.readouterr()
        assert "[WARNING]" in captured.out
        assert "Be careful" in captured.out

    def test_info_message(self, cli_plain, capsys):
        """Test info message formatting"""
        cli_plain.info("Just so you know")
        captured = capsys.readouterr()
        assert "[INFO]" in captured.out
        assert "Just so you know" in captured.out

    def test_dim_message(self, cli_plain, capsys):
        """Test dimmed message formatting"""
        cli_plain.dim("Metadata text")
        captured = capsys.readouterr()
        assert "Metadata text" in captured.out

    def test_progress_plain(self, cli_plain, capsys):
        """Test progress bar in plain mode"""
        cli_plain.progress("Processing", 50, 100)
        captured = capsys.readouterr()
        assert "Processing: 50/100" in captured.out
        assert "50.0%" in captured.out

    def test_progress_colorful(self, cli_colorful, capsys):
        """Test progress bar in colorful mode"""
        cli_colorful.progress("Processing", 50, 100)
        captured = capsys.readouterr()
        assert "Processing" in captured.out
        assert "50.0%" in captured.out
        assert "█" in captured.out  # Progress bar character

    def test_progress_complete(self, cli_plain, capsys):
        """Test progress bar at 100%"""
        cli_plain.progress("Complete", 100, 100)
        captured = capsys.readouterr()
        assert "100/100" in captured.out
        assert "100.0%" in captured.out

    def test_list_options_plain(self, cli_plain, capsys):
        """Test options list in plain mode"""
        cli_plain.list_options({"1": "First option", "2": "Second option"}, title="Menu")
        captured = capsys.readouterr()
        assert "Press 1: First option" in captured.out
        assert "Press 2: Second option" in captured.out

    def test_list_options_colorful(self, cli_colorful, capsys):
        """Test options list in colorful mode"""
        cli_colorful.list_options({"a": "Option A", "b": "Option B"})
        captured = capsys.readouterr()
        assert "a" in captured.out
        assert "Option A" in captured.out

    def test_table_plain(self, cli_plain, capsys):
        """Test table formatting in plain mode"""
        cli_plain.table(headers=["Name", "Age", "City"], rows=[["Alice", "30", "NYC"], ["Bob", "25", "LA"]])
        captured = capsys.readouterr()
        assert "[TABLE: Name, Age, City]" in captured.out
        assert "Alice" in captured.out
        assert "Bob" in captured.out
        assert "[END TABLE]" in captured.out

    def test_table_colorful(self, cli_colorful, capsys):
        """Test table formatting in colorful mode"""
        cli_colorful.table(
            headers=["File", "Size"],
            rows=[["audio.wav", "2.5 MB"], ["music.mp3", "3.1 MB"]],
            alignments=["left", "right"],
        )
        captured = capsys.readouterr()
        assert "File" in captured.out
        assert "Size" in captured.out
        assert "audio.wav" in captured.out

    def test_separator(self, cli_plain, capsys):
        """Test separator output"""
        cli_plain.separator()
        captured = capsys.readouterr()
        # In plain mode, separator is just a newline
        assert len(captured.out) > 0

    @patch("builtins.input", return_value="test_input")
    def test_prompt_basic(self, mock_input, cli_plain):
        """Test basic prompt"""
        result = cli_plain.prompt("What's your name?")
        assert result == "test_input"

    @patch("builtins.input", return_value="")
    def test_prompt_with_default(self, mock_input, cli_plain):
        """Test prompt with default value"""
        result = cli_plain.prompt("What's your name?", default="John")
        assert result == "John"

    @patch("builtins.input", side_effect=["invalid", "y"])
    def test_prompt_with_validation(self, mock_input, cli_plain):
        """Test prompt with valid choices validation"""
        result = cli_plain.prompt("Continue?", valid_choices=["y", "n"])
        assert result == "y"

    @patch("builtins.input", return_value="y")
    def test_confirm_yes(self, mock_input, cli_plain):
        """Test confirmation with yes"""
        result = cli_plain.confirm("Are you sure?")
        assert result is True

    @patch("builtins.input", return_value="n")
    def test_confirm_no(self, mock_input, cli_plain):
        """Test confirmation with no"""
        result = cli_plain.confirm("Are you sure?")
        assert result is False

    @patch("builtins.input", return_value="")
    def test_confirm_default_true(self, mock_input, cli_plain):
        """Test confirmation with default=True"""
        result = cli_plain.confirm("Continue?", default=True)
        assert result is True

    @patch("builtins.input", return_value="")
    def test_confirm_default_false(self, mock_input, cli_plain):
        """Test confirmation with default=False"""
        result = cli_plain.confirm("Continue?", default=False)
        assert result is False

    def test_play_sound(self, cli_plain):
        """Test audio feedback (should not crash)"""
        # Enable audio feedback
        cli_plain.audio_feedback = True

        # These should not raise exceptions
        cli_plain.play_sound("success")
        cli_plain.play_sound("error")
        cli_plain.play_sound("warning")
        cli_plain.play_sound("info")

    def test_play_sound_disabled(self, cli_plain):
        """Test audio feedback when disabled"""
        cli_plain.audio_feedback = False

        # Should do nothing (not raise exception)
        cli_plain.play_sound("success")


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_cli_confirm_import(self):
        """Test cli_confirm can be imported"""
        from usability.cli_accessibility import cli_confirm

        assert callable(cli_confirm)

    def test_cli_print_progress_import(self):
        """Test cli_print_progress can be imported"""
        from usability.cli_accessibility import cli_print_progress

        assert callable(cli_print_progress)


class TestEnvironmentVariables:
    """Test environment variable detection"""

    @patch.dict(os.environ, {"NO_COLOR": "1"})
    def test_no_color_standard(self):
        """Test standard NO_COLOR environment variable"""
        cli = AccessibleCLI(theme="auto")
        assert cli.screen_reader_mode is True

    @patch.dict(os.environ, {"AURIK_NO_COLOR": "1"})
    def test_aurik_no_color(self):
        """Test AURIK_NO_COLOR environment variable"""
        cli = AccessibleCLI(theme="auto")
        assert cli.screen_reader_mode is True

    @patch.dict(os.environ, {"AURIK_AUDIO_FEEDBACK": "1"})
    def test_audio_feedback_env(self):
        """Test AURIK_AUDIO_FEEDBACK environment variable"""
        cli = AccessibleCLI()
        assert cli.audio_feedback is True

    @patch.dict(os.environ, {"AURIK_HIGH_CONTRAST": "1"}, clear=True)
    def test_high_contrast_detection(self):
        """Test high contrast mode detection"""
        cli = AccessibleCLI(theme="auto")
        assert cli.theme.name == "high_contrast"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_queue_progress(self, capsys):
        """Test progress bar with 0 total"""
        cli = AccessibleCLI(theme="plain")
        cli.progress("Empty", 0, 0)
        captured = capsys.readouterr()
        assert "0/0" in captured.out

    def test_table_empty_rows(self, capsys):
        """Test table with empty rows"""
        cli = AccessibleCLI(theme="plain")
        cli.table(["Col1", "Col2"], [])
        captured = capsys.readouterr()
        assert "Col1" in captured.out
        assert "[END TABLE]" in captured.out

    def test_table_mismatched_columns(self, capsys):
        """Test table with mismatched column counts"""
        cli = AccessibleCLI(theme="plain")
        # Should not crash even with mismatched columns
        cli.table(headers=["A", "B", "C"], rows=[["1", "2"]])  # Only 2 values for 3 headers
        captured = capsys.readouterr()
        assert "A" in captured.out

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_prompt_keyboard_interrupt(self, mock_input):
        """Test prompt handles keyboard interrupt"""
        cli = AccessibleCLI(theme="plain")
        with pytest.raises(SystemExit):
            cli.prompt("Input:")

    def test_very_long_table_cell(self, capsys):
        """Test table with very long cell content"""
        cli = AccessibleCLI(theme="plain")
        long_text = "A" * 200
        cli.table(["Header"], [[long_text]])
        captured = capsys.readouterr()
        assert long_text in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

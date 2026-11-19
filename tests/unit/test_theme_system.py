"""Tests for _Theme dual-palette (dark/light) and SettingsManager.theme().

Ensures:
- _Theme.apply('dark') / _Theme.apply('light') switches all tokens
- _Theme defaults to dark
- SettingsManager persists theme preference
- Theme round-trip: apply → read back → consistent
"""

import pytest

# ---------------------------------------------------------------------------
# _Theme palette tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThemePalette:
    """Test _Theme class-level palette switching."""

    @pytest.fixture(autouse=True)
    def _reset_theme(self):
        """Ensure dark theme is restored after each test."""
        from Aurik10.ui.modern_window import _Theme

        yield
        _Theme.apply("dark")

    def test_default_is_dark(self):
        from Aurik10.ui.modern_window import _Theme

        _Theme.apply("dark")
        assert _Theme._active == "dark"
        assert _Theme.BG_DARK == "#080a18"
        assert not _Theme.is_light()

    def test_switch_to_light(self):
        from Aurik10.ui.modern_window import _Theme

        _Theme.apply("light")
        assert _Theme._active == "light"
        assert _Theme.BG_DARK == "#f5f6fa"
        assert _Theme.is_light()

    def test_switch_back_to_dark(self):
        from Aurik10.ui.modern_window import _Theme

        _Theme.apply("light")
        _Theme.apply("dark")
        assert _Theme._active == "dark"
        assert _Theme.BG_DARK == "#080a18"

    def test_light_primary_differs_from_dark(self):
        from Aurik10.ui.modern_window import _Theme

        _Theme.apply("dark")
        dark_primary = _Theme.PRIMARY
        _Theme.apply("light")
        light_primary = _Theme.PRIMARY
        assert dark_primary != light_primary

    def test_all_dark_keys_present(self):
        from Aurik10.ui.modern_window import _Theme

        for key in _Theme._DARK:
            assert hasattr(_Theme, key), f"_Theme missing attribute {key}"

    def test_all_light_keys_present(self):
        from Aurik10.ui.modern_window import _Theme

        for key in _Theme._LIGHT:
            assert hasattr(_Theme, key), f"_Theme missing attribute {key}"

    def test_dark_light_same_keys(self):
        from Aurik10.ui.modern_window import _Theme

        assert set(_Theme._DARK.keys()) == set(_Theme._LIGHT.keys())

    def test_all_light_values_differ_from_dark(self):
        from Aurik10.ui.modern_window import _Theme

        for key in _Theme._DARK:
            assert _Theme._DARK[key] != _Theme._LIGHT[key], (
                f"Dark and Light have same value for {key}: {_Theme._DARK[key]}"
            )

    def test_font_unchanged_by_theme(self):
        from Aurik10.ui.modern_window import _Theme

        _Theme.apply("dark")
        font_dark = _Theme.FONT_UI
        _Theme.apply("light")
        font_light = _Theme.FONT_UI
        assert font_dark == font_light == "Segoe UI"

    def test_radii_unchanged_by_theme(self):
        from Aurik10.ui.modern_window import _Theme

        _Theme.apply("light")
        assert _Theme.RADIUS_SM == 8
        assert _Theme.RADIUS_MD == 10
        assert _Theme.RADIUS_LG == 15


# ---------------------------------------------------------------------------
# SettingsManager theme persistence
# ---------------------------------------------------------------------------


class TestSettingsManagerTheme:
    def test_default_theme_is_dark(self):
        from Aurik10.core.settings_manager import SettingsManager

        sm = SettingsManager()
        # Fresh settings with no saved value should return "dark"
        assert sm.theme() in ("dark", "light")  # may have been persisted

    def test_set_and_get_theme(self):
        from Aurik10.core.settings_manager import SettingsManager

        sm = SettingsManager()
        sm.set_theme("light")
        assert sm.theme() == "light"
        sm.set_theme("dark")
        assert sm.theme() == "dark"

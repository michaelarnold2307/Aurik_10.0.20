from __future__ import annotations

"""Tests for Aurik10/core/settings_manager.py — persistent app settings."""


import pytest

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global singleton before each test."""
    import Aurik10.core.settings_manager as sm

    sm._instance = None
    yield
    sm._instance = None


@pytest.fixture
def settings(tmp_path):
    """Create a SettingsManager backed by a temporary QSettings file."""
    from Aurik10.core.settings_manager import SettingsManager

    s = SettingsManager.__new__(SettingsManager)
    # Use QSettings with a custom file path
    from PyQt5.QtCore import QSettings

    s._qs = QSettings(str(tmp_path / "test_settings.ini"), QSettings.Format.IniFormat)
    return s


# ── Singleton ────────────────────────────────────────────────────────────


def test_singleton_returns_same_instance():
    from Aurik10.core.settings_manager import get_settings_manager

    a = get_settings_manager()
    b = get_settings_manager()
    assert a is b


def test_singleton_thread_safe():
    import threading

    from Aurik10.core.settings_manager import get_settings_manager

    instances = []

    def _get():
        instances.append(get_settings_manager())

    threads = [threading.Thread(target=_get) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len({id(i) for i in instances}) == 1


# ── Window Geometry ──────────────────────────────────────────────────────


def test_window_geometry_roundtrip(settings):
    from PyQt5.QtCore import QByteArray

    geo = QByteArray(b"fake_geometry_data")
    state = QByteArray(b"fake_state_data")
    settings.save_window_geometry(geo, state)
    assert settings.window_geometry() == geo
    assert settings.window_state() == state


def test_window_geometry_defaults_none(settings):
    assert settings.window_geometry() is None
    assert settings.window_state() is None


def test_window_maximized_roundtrip(settings):
    assert settings.window_maximized() is False
    settings.save_window_maximized(True)
    assert settings.window_maximized() is True
    settings.save_window_maximized(False)
    assert settings.window_maximized() is False


# ── Recent Files ─────────────────────────────────────────────────────────


def test_recent_files_empty_initially(settings):
    assert settings.recent_files() == []


def test_add_recent_file(settings, tmp_path):
    f1 = tmp_path / "song1.wav"
    f1.touch()
    result = settings.add_recent_file(str(f1))
    assert str(f1.resolve()) in result
    assert len(result) == 1


def test_recent_files_moves_duplicate_to_front(settings, tmp_path):
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    f1.touch()
    f2.touch()
    settings.add_recent_file(str(f1))
    settings.add_recent_file(str(f2))
    result = settings.add_recent_file(str(f1))
    assert result[0] == str(f1.resolve())
    assert result[1] == str(f2.resolve())


def test_recent_files_max_10(settings, tmp_path):
    files = []
    for i in range(15):
        f = tmp_path / f"track{i:02d}.wav"
        f.touch()
        files.append(f)
        settings.add_recent_file(str(f))
    result = settings.recent_files()
    assert len(result) <= 10


def test_recent_files_excludes_deleted(settings, tmp_path):
    f1 = tmp_path / "exists.wav"
    f1.touch()
    settings.add_recent_file(str(f1))
    f1.unlink()
    result = settings.recent_files()
    assert len(result) == 0


def test_clear_recent_files(settings, tmp_path):
    f1 = tmp_path / "song.wav"
    f1.touch()
    settings.add_recent_file(str(f1))
    settings.clear_recent_files()
    assert settings.recent_files() == []


# ── Language ─────────────────────────────────────────────────────────────


def test_language_default_de(settings):
    assert settings.language() == "de"


def test_language_roundtrip(settings):
    settings.set_language("en")
    assert settings.language() == "en"


# ── Export Format ────────────────────────────────────────────────────────


def test_default_export_format(settings):
    assert settings.default_export_format() == "flac_24"


def test_export_format_roundtrip(settings):
    settings.set_default_export_format("wav_16")
    assert settings.default_export_format() == "wav_16"


# ── Processing Mode ──────────────────────────────────────────────────────


def test_default_processing_mode(settings):
    assert settings.default_processing_mode() == "RESTORATION"


def test_processing_mode_roundtrip(settings):
    settings.set_default_processing_mode("STUDIO_2026")
    assert settings.default_processing_mode() == "STUDIO_2026"


# ── Last Directories ─────────────────────────────────────────────────────


def test_last_open_dir_empty_default(settings):
    assert settings.last_open_dir() == ""


def test_last_open_dir_roundtrip(settings, tmp_path):
    settings.set_last_open_dir(str(tmp_path))
    assert settings.last_open_dir() == str(tmp_path)


def test_last_open_dir_invalid_returns_empty(settings):
    settings.set_last_open_dir("/nonexistent/path/12345")
    assert settings.last_open_dir() == ""


def test_last_export_dir_roundtrip(settings, tmp_path):
    settings.set_last_export_dir(str(tmp_path))
    assert settings.last_export_dir() == str(tmp_path)


# ── i18n Keys ────────────────────────────────────────────────────────────


def test_i18n_keys_exist():
    """Verify that the new i18n keys for recent files, help, and tray are present."""
    from Aurik10.i18n import set_language, t

    set_language("de")
    expected_keys = [
        "recent.title",
        "recent.empty",
        "recent.clear",
        "help.shortcuts",
        "help.user_guide",
        "help.troubleshooting",
        "help.configuration",
        "help.about",
        "tray.batch_done",
        "tray.batch_ok",
        "tray.batch_mixed",
        "tray.batch_failed",
        "tray.show_window",
        "tray.quit",
    ]
    for key in expected_keys:
        val = t(key)
        assert val != key, f"i18n key '{key}' not found (returned key itself)"

    set_language("en")
    for key in expected_keys:
        val = t(key)
        assert val != key, f"i18n key '{key}' not found in English"


def test_i18n_tray_batch_ok_format():
    from Aurik10.i18n import set_language, t

    set_language("de")
    result = t("tray.batch_ok", count=5)
    assert "5" in result


def test_i18n_tray_batch_mixed_format():
    from Aurik10.i18n import set_language, t

    set_language("de")
    result = t("tray.batch_mixed", ok=3, failed=2)
    assert "3" in result and "2" in result

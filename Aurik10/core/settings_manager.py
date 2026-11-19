"""Aurik10/core/settings_manager.py — Persistent app settings via QSettings.

Singleton access: ``get_settings_manager()``

Persists window geometry, recent files, language, default export format,
default processing mode, and last-used directories across sessions.
Storage: platform-native (Linux: ~/.config/AURIK/AURIK Professional.conf).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PyQt5.QtCore import QByteArray, QSettings

logger = logging.getLogger(__name__)

_instance: SettingsManager | None = None
_lock = threading.Lock()

_MAX_RECENT_FILES = 10


def get_settings_manager() -> SettingsManager:
    """Thread-safe singleton access (double-checked locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SettingsManager()
    return _instance


class SettingsManager:
    """Kapselt QSettings for all persistent Aurik preferences."""

    def __init__(self) -> None:
        self._qs = QSettings("AURIK", "AURIK Professional")

    # ── Window Geometry ───────────────────────────────────────────────────

    def save_window_geometry(self, geometry: QByteArray, state: QByteArray) -> None:
        self._qs.setValue("window/geometry", geometry)
        self._qs.setValue("window/state", state)

    def window_geometry(self) -> QByteArray | None:
        val = self._qs.value("window/geometry")
        return val if isinstance(val, QByteArray) else None

    def window_state(self) -> QByteArray | None:
        val = self._qs.value("window/state")
        return val if isinstance(val, QByteArray) else None

    def save_window_maximized(self, maximized: bool) -> None:
        self._qs.setValue("window/maximized", maximized)

    def window_maximized(self) -> bool:
        return self._qs.value("window/maximized", False, type=bool)

    # ── Recent Files ──────────────────────────────────────────────────────

    def recent_files(self) -> list[str]:
        raw = self._qs.value("recent_files", [])
        if isinstance(raw, list):
            return [str(p) for p in raw if p and Path(str(p)).exists()][:_MAX_RECENT_FILES]
        return []

    def add_recent_file(self, path: str) -> list[str]:
        files = [str(p) for p in self._qs.value("recent_files", []) or [] if p]
        path = str(Path(path).resolve())
        if path in files:
            files.remove(path)
        files.insert(0, path)
        files = files[:_MAX_RECENT_FILES]
        self._qs.setValue("recent_files", files)
        return [f for f in files if Path(f).exists()]

    def clear_recent_files(self) -> None:
        self._qs.setValue("recent_files", [])

    # ── Language ──────────────────────────────────────────────────────────

    def language(self) -> str:
        return str(self._qs.value("app/language", "de"))

    def set_language(self, lang: str) -> None:
        self._qs.setValue("app/language", lang)

    # ── Default Export Format ─────────────────────────────────────────────

    def default_export_format(self) -> str:
        return str(self._qs.value("export/default_format", "flac_24"))

    def set_default_export_format(self, fmt: str) -> None:
        self._qs.setValue("export/default_format", fmt)

    # ── Default Processing Mode ───────────────────────────────────────────

    def default_processing_mode(self) -> str:
        return str(self._qs.value("processing/default_mode", "RESTORATION"))

    def set_default_processing_mode(self, mode: str) -> None:
        self._qs.setValue("processing/default_mode", mode)

    # ── Last Directories ──────────────────────────────────────────────────

    def last_open_dir(self) -> str:
        val = str(self._qs.value("dirs/last_open", ""))
        return val if val and Path(val).is_dir() else ""

    def set_last_open_dir(self, path: str) -> None:
        self._qs.setValue("dirs/last_open", str(path))

    def last_export_dir(self) -> str:
        val = str(self._qs.value("dirs/last_export", ""))
        return val if val and Path(val).is_dir() else ""

    def set_last_export_dir(self, path: str) -> None:
        self._qs.setValue("dirs/last_export", str(path))

    # ── Theme ─────────────────────────────────────────────────────────────

    def theme(self) -> str:
        """Gibt current theme name: 'dark' (default) or 'light' zurück."""
        return str(self._qs.value("app/theme", "dark"))

    def set_theme(self, theme: str) -> None:
        self._qs.setValue("app/theme", theme)

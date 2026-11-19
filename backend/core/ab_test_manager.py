"""
Deprecated: ABTestManager → ``core.ab_compare_manager.ABCompareManager``.

Dieses Modul ist veraltet seit v10.0.0.
Alle Fähigkeiten sind in :class:`~core.ab_compare_manager.ABCompareManager`
enthalten, der Thread-sicher, persistent (JSON-Sidecar) und mit vollständiger
Metrik-Unterstützung ausgestattet ist.

Migration::

    # Alt (veraltet):
    from backend.core.ab_test_manager import ABTestManager
    mgr = ABTestManager()

    # Neu (kanonisch):
    from backend.core.ab_compare_manager import get_ab_manager, store_ab_session
    session_id = store_ab_session(original, restored, sample_rate, material)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

# Re-export kanonische Symbole für Backward-Compatibility

logger = logging.getLogger(__name__)


class ABTestManager:
    """Deprecated Stub — Nachfolger: :class:`~core.ab_compare_manager.ABCompareManager`.

    .. deprecated:: 9.10.45
        ``ABTestManager`` lieferte immer leere Listen, da bei jedem Aufruf eine
        frische Instanz ohne geteilten State erzeugt wurde.
        Bitte ``get_ab_manager()`` aus ``core.ab_compare_manager`` verwenden.
    """

    def __init__(self) -> None:
        warnings.warn(
            "ABTestManager ist veraltet (seit v10.0.0). "
            "Bitte 'from backend.core.ab_compare_manager import get_ab_manager' verwenden.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tests: list[dict] = []
        self.results: list[dict] = []
        self.listeners: list[Callable[[dict], None]] = []

    def submit_test(self, original_audio, processed_audio, metadata: dict) -> None:
        """Stub — ohne Funktion. Nutze ``ABCompareManager.compare_audio()``."""

    def submit_result(self, test_id: int, winner: str, feedback: str = "") -> None:
        """Stub — ohne Funktion. Nutze ``ABCompareManager.store()``."""

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """Stub — ohne Funktion."""

    def _notify(self, test: dict) -> None:
        """Stub — ohne Funktion."""

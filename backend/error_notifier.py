"""error_notifier.py — Minimaler Fehler-Benachrichtigungs-Stub für Aurik 10.0.0.

Stellt `setup_error_notifier()` bereit, das von `backend/logging_config.py`
benötigt wird. Keine externe Netzwerkverbindung — reine lokale Protokollierung.
"""

from __future__ import annotations

import logging

__all__ = ["setup_error_notifier"]

_logger = logging.getLogger(__name__)


def setup_error_notifier() -> None:
    """Richtet den internen Fehler-Benachrichtiger ein.

    Aktuell: stille Stub-Implementierung — alle Fehler gehen über
    das Standard-Python-Logging-System. Keine externen Verbindungen.
    """
    _logger.debug("Error notifier initialised (internal logging only).")

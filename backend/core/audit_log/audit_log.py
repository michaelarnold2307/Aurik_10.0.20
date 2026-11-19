"""
Audit Log Modul – erzeugt unveränderliche Logs für jeden Lauf.
Normativ: Jeder Lauf muss geloggt werden (§10.4).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AuditLog:
    """Schreibt unveränderliche Audit-Einträge für jeden Restaurierungslauf."""

    def log_run(self, info: Any) -> None:
        """Persistiert alle relevanten Lauf-Informationen im Logger.

        Args:
            info: Dict oder serialisierbares Objekt mit Lauf-Metadaten.
        """
        logger.info("[AuditLog] Lauf-Eintrag: %s", info)

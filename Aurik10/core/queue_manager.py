"""
QueueManager — Aurik 10.x.x

Verwaltet die Warteschlange für Batch-Verarbeitung in der PyQt5-GUI.
Thread-sicher, NaN/Inf-frei, Singleton-Pattern (§3.2).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueueStatus(Enum):
    """Status eines Warteschlangen-Eintrags."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """Ein einzelner Eintrag in der Restaurierungs-Warteschlange."""

    id: str
    input_file: str
    output_file: str
    settings: dict[str, Any]
    status: QueueStatus = QueueStatus.PENDING
    progress: int = 0
    error_message: str | None = None


class QueueManager:
    """Thread-sichere Verwaltung der Restaurierungs-Warteschlange.

    Verwendung:
        manager = QueueManager()
        item = manager.add_item("/input.wav", "/output.flac", settings={})
        manager.update_item_status(item.id, QueueStatus.PROCESSING, 50)
        manager.remove_item(item.id)
        manager.clear_queue(clear_completed=True)
    """

    def __init__(self) -> None:
        self._items: dict[str, QueueItem] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Öffentliche API                                                      #
    # ------------------------------------------------------------------ #

    def add_item(
        self,
        input_file: str,
        output_file: str,
        settings: dict[str, Any] | None = None,
    ) -> QueueItem:
        """Fügt eine neue Datei der Warteschlange hinzu.

        Args:
            input_file:  Absoluter Pfad zur Quelldatei.
            output_file: Absoluter Pfad zur Zieldatei.
            settings:    Restaurierungseinstellungen (GP-Parameter, Modus, …).

        Returns:
            Neu erstelltes QueueItem mit eindeutiger ID.
        """
        item = QueueItem(
            id=str(uuid.uuid4()),
            input_file=input_file,
            output_file=output_file,
            settings=settings or {},
        )
        with self._lock:
            self._items[item.id] = item
        return item

    def update_item_status(
        self,
        item_id: str,
        status: QueueStatus,
        progress: int = 0,
        error_message: str | None = None,
    ) -> bool:
        """Aktualisiert Status und Fortschritt eines Eintrags.

        Args:
            item_id:       ID des Eintrags.
            status:        Neuer QueueStatus.
            progress:      Fortschritt 0–100.
            error_message: Optionale Fehlermeldung (bei FAILED).

        Returns:
            True wenn Eintrag gefunden und aktualisiert, False sonst.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False
            item.status = status
            item.progress = max(0, min(100, progress))
            if error_message is not None:
                item.error_message = error_message
        return True

    def remove_item(self, item_id: str) -> bool:
        """Entfernt einen Eintrag aus der Warteschlange.

        Args:
            item_id: ID des zu entfernenden Eintrags.

        Returns:
            True wenn Eintrag gefunden und entfernt, False sonst.
        """
        with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                return True
        return False

    def clear_queue(self, clear_completed: bool = False) -> None:
        """Leert die Warteschlange.

        Args:
            clear_completed: True → auch abgeschlossene Einträge entfernen.
                             False → nur wartende Einträge entfernen.
        """
        with self._lock:
            if clear_completed:
                self._items.clear()
            else:
                to_remove = [item_id for item_id, item in self._items.items() if item.status == QueueStatus.PENDING]
                for item_id in to_remove:
                    del self._items[item_id]

    def get_items(self) -> list[QueueItem]:
        """Gibt alle Einträge in Einfügereihenfolge zurück.

        Returns:
            Kopie der aktuellen Warteschlange als Liste.
        """
        with self._lock:
            return list(self._items.values())

    def get_item(self, item_id: str) -> QueueItem | None:
        """Gibt einen einzelnen Eintrag zurück oder None.

        Args:
            item_id: ID des gesuchten Eintrags.
        """
        with self._lock:
            return self._items.get(item_id)

    def get_pending_items(self) -> list[QueueItem]:
        """Gibt alle wartenden (PENDING) Einträge zurück."""
        with self._lock:
            return [item for item in self._items.values() if item.status == QueueStatus.PENDING]

    def reorder_items(self, new_order: list[str]) -> bool:
        """Ordnet die Queue-Einträge entsprechend der übergebenen ID-Reihenfolge um.

        Unbekannte IDs werden ignoriert; fehlende IDs bleiben am Ende erhalten.
        Thread-sicher (Lock).

        Args:
            new_order: Liste von Item-IDs in gewünschter neuer Reihenfolge.

        Returns:
            True wenn mindestens ein Eintrag die Position gewechselt hat.
        """
        with self._lock:
            # Build reordered dict: known IDs first, remaining IDs appended.
            known = {iid: self._items[iid] for iid in new_order if iid in self._items}
            rest = {iid: item for iid, item in self._items.items() if iid not in known}
            new_dict = {**known, **rest}
            if list(new_dict.keys()) == list(self._items.keys()):
                return False  # no change
            self._items = new_dict
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

"""§D-K3-58 (2026-10-07): Log-Health-Guard — Backend-FileHandler-Verlust heilt sich.

Produktionsbefund (überwachter Lauf „Trio Schweizer – 13 Tage"): Um 22:42
verstummte `logs/aurik_backend.log` mitten im Lauf; der Prozess hielt danach
KEINEN offenen Handle mehr auf die Datei (fd-Prüfung am laufenden Prozess),
stdout lief weiter — Monitoring war blind (§G8 (copilot-instructions.md)).
Fix: `ensure_file_handler_attached()` in `backend/logging_config` (eine Quelle
für die Handler-Parameter, §G9 (copilot-instructions.md)), aufgerufen pro
Phase in `_profiled_phase_call` (UV3).

Gepinnt: Verlust → Re-Attach (True) → idempotent (False); ein GESCHLOSSENER
Stream zählt NICHT als aktiv und wird ersetzt; die Heilung wird mit WARNING
protokolliert (§V6 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from backend import logging_config as lc

_TARGET = os.path.abspath(lc.LOG_FILE)


def _active_handlers() -> list[RotatingFileHandler]:
    root = logging.getLogger()
    return [
        h
        for h in root.handlers
        if isinstance(h, RotatingFileHandler)
        and os.path.abspath(getattr(h, "baseFilename", "")) == _TARGET
        and not getattr(getattr(h, "stream", None), "closed", False)
    ]


def _restore(saved: list[RotatingFileHandler]) -> None:
    """Sessionzustand wiederherstellen: Guard-Handler weg, Originale zurück."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if (
            isinstance(h, RotatingFileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == _TARGET
            and h not in saved
        ):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                logging.getLogger(__name__).debug("Test-Handler close fehlgeschlagen", exc_info=True)
    for h in saved:
        root.addHandler(h)


def test_verlust_wird_geheilt_und_ist_idempotent(caplog) -> None:
    assert _active_handlers(), "Vorbedingung: FileHandler aktiv (Import von logging_config)"
    saved = _active_handlers()
    for h in saved:
        logging.getLogger().removeHandler(h)
    try:
        with caplog.at_level(logging.WARNING, logger="backend.logging_config"):
            assert lc.ensure_file_handler_attached() is True
        assert _active_handlers(), "Guard hat nicht re-attached"
        assert lc.ensure_file_handler_attached() is False
        assert any("D-K3-58" in r.message for r in caplog.records)
    finally:
        _restore(saved)


def test_geschlossener_stream_zaehlt_nicht_als_aktiv() -> None:
    saved = _active_handlers()
    for h in saved:
        logging.getLogger().removeHandler(h)
    h_closed = lc._make_file_handler()
    h_closed.stream.close()  # defekter Handler: Stream geschlossen
    logging.getLogger().addHandler(h_closed)
    try:
        assert lc._file_handler_attached(logging.getLogger()) is False
        assert lc.ensure_file_handler_attached() is True
        assert _active_handlers(), "defekter Handler wurde nicht ersetzt"
    finally:
        _restore(saved)

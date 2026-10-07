import logging
import os
from logging.handlers import RotatingFileHandler

from .error_notifier import setup_error_notifier

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "aurik_backend.log")

# §D-K3-58: kanonische FileHandler-Parameter (eine Quelle, §G9 (copilot-instructions.md))
_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 5
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def _make_file_handler() -> RotatingFileHandler:
    """§D-K3-58: erzeugt den kanonischen Backend-FileHandler (5 MB × 5 Backups)."""
    handler = RotatingFileHandler(LOG_FILE, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def _file_handler_attached(root: logging.Logger) -> bool:
    """§D-K3-58: True, wenn ein AKTIVER FileHandler auf LOG_FILE am Root hängt.

    Anders als die frühere Prüfung zählt ein Handler mit GESCHLOSSENEM Stream
    nicht als aktiv (Produktionsbefund: Log verstummte, Prozess hielt keinen
    offenen Datei-Handle mehr — fd-Prüfung am laufenden Prozess 2026-10-07).
    """
    _abs_target = os.path.abspath(LOG_FILE)
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == _abs_target:
            stream = getattr(h, "stream", None)
            if stream is not None and not getattr(stream, "closed", False):
                return True
    return False


def ensure_file_handler_attached() -> bool:
    """§D-K3-58 (2026-10-07): Selbstheilung für den Backend-FileHandler.

    Produktionsbefund (überwachter Lauf „Trio Schweizer – 13 Tage"): Um 22:42
    verstummte `logs/aurik_backend.log` mitten im Lauf — der Prozess hielt
    danach KEINEN offenen Handle mehr auf die Datei (fd-Prüfung), stdout lief
    weiter (Monitoring war blind, §G8 (copilot-instructions.md)). Die Ursache des Verlusts ließ sich
    nicht abschließend lokalisieren (kein removeHandler/shutdown im Song-Pfad
    auffindbar). Dieser Guard heilt: fehlt der FileHandler — oder ist sein
    Stream geschlossen — wird er ersetzt, neu registriert und mit
    WARNING protokolliert (§V6 (copilot-instructions.md)). Rückgabe: True wenn (re-)attached, False wenn
    bereits aktiv. Aufruf: pro Phase in `_profiled_phase_call` (UV3).
    """
    root = logging.getLogger()
    if _file_handler_attached(root):
        return False
    _abs_target = os.path.abspath(LOG_FILE)
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == _abs_target:
            try:
                root.removeHandler(h)
            except Exception:
                logging.getLogger(__name__).debug("§D-K3-58: defekter Handler nicht entfernbar", exc_info=True)
    root.addHandler(_make_file_handler())
    logging.getLogger(__name__).warning(
        "§D-K3-58: Backend-FileHandler fehlte/defekt — neu registriert (%s); das Log war bis hier blind (§G8 (copilot-instructions.md)).",
        LOG_FILE,
    )
    return True


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Guard: only register the FileHandler if no other handler for the same file is
# already active on the root logger (Aurik10/main.py registers one first).
# §D-K3-58: inkl. Closed-Stream-Erkennung via _file_handler_attached().
if not _file_handler_attached(root_logger):
    root_logger.addHandler(_make_file_handler())

# Fehlerbenachrichtigung aktivieren (E-Mail), falls konfiguriert
setup_error_notifier()

# Optional: Fehler-Alerts (z.B. per E-Mail) können hier ergänzt werden


def get_logger(name=None, level: int = logging.INFO) -> logging.Logger:
    """
    Gibt einen konfigurierten Logger zurück.
    Kombiniert globales Logfile-Setup mit individuellem Level-Support
    (portiert aus backend.core.regulator.logging_config).
    """
    lg = logging.getLogger(name)
    lg.setLevel(level)
    return lg

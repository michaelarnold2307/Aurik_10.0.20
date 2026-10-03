"""
aurik6.dsp.logging
SOTA-konformes Logging für Musikrestaurierung und Audit-Trails
"""

import datetime


class AuditLogger:
    """Schreibt Audit-Logs für alle relevanten Verarbeitungsschritte."""

    def __init__(self, logfile: str = "audit_log.txt"):
        self.logfile = logfile

    def log(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.logfile, "a") as f:
            f.write(f"[{timestamp}] {message}\n")


class InMemoryLogger:
    """Speichert Logs im Speicher für schnelle Analyse und Tests."""

    def __init__(self):
        self.entries = []

    def log(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        self.entries.append(f"[{timestamp}] {message}")

    def get_entries(self):
        return self.entries

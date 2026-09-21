"""
Aurik 10 — Weltklasse-Audio-Restaurierung
Weltweit führendes kognitiv-perceptuelles Audio-Restaurierungssystem mit chirurgischer Präzision.
"""

_FALLBACK_VERSION = "10.1.0"  # Letzter bekannter Stand; §v10.802: die Bridge ist die Quelle

try:
    # §V4 (copilot-instructions.md) + §v10.802 GUI-Sync: Version ausschließlich über
    # die Bridge beziehen — das Frontend darf backend/core nie direkt importieren.
    from backend.api.bridge import get_aurik_version

    __version__ = get_aurik_version()
except Exception:  # pragma: no cover — Fallback, wenn das Backend nicht importierbar ist
    __version__ = _FALLBACK_VERSION

__author__ = "Aurik Team"

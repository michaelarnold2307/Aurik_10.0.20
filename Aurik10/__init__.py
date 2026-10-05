"""
Aurik 10 — Weltklasse-Audio-Restaurierung
Weltweit führendes kognitiv-perceptuelles Audio-Restaurierungssystem mit chirurgischer Präzision.
"""

# Fallback nur für den Fall, dass die Bridge nicht importierbar ist (§V4 (copilot-instructions.md)).
# §v10.802: keine zweite driftende Nummer — Gleichlauf mit backend/core/version.py
# wird von scripts/version_guard.py (Warnung) und
# tests/unit/test_version_checker_and_ux.py (fail-closed) erzwungen.
_FALLBACK_VERSION = "10.3.6"

try:
    # §V4 (copilot-instructions.md) + §v10.802 GUI-Sync: Version ausschließlich über
    # die Bridge beziehen — das Frontend darf backend/core nie direkt importieren.
    from backend.api.bridge import get_aurik_version

    __version__ = get_aurik_version()
except Exception:  # pragma: no cover — Fallback, wenn das Backend nicht importierbar ist
    __version__ = _FALLBACK_VERSION

__author__ = "Aurik Team"

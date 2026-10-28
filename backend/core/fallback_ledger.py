"""Fallback-Ledger — ML→DSP-Rückfälle zählen (Pfeiler 5 der SOTA-Roadmap).

§V6 (copilot-instructions.md) verlangt für jeden ML→DSP-Fallback eine
``logger.warning``-Meldung mit Begründung. Diese Meldungen zu LOGGEN macht
die Lücke sichtbar — sie zu ZÄHLEN macht sie zur Qualitäts-KPI (B1 des
Wohlklang-Vertrags, ``.github/WOHLKLANG_CLAIMS.md``):

* **Budget:** 0 ML-Fallbacks im Referenz-Korpus (Release-Bedingung 3 der
  GO/NO-GO-Checkliste, ``docs/guides/GO_NO_GO_DECISION_PROTOCOL.md``).
* **Release-Gate:** Fallback-Quote > 0 blockiert den Flag-Flip.

Zwei Ebenen:
1. ``record_fallback(component, reason)`` — explizite Registrierung für
   neue Code-Stellen.
2. ``install_log_counter(logger_name="backend")`` — zählt WARNING+-Records
   mit Fallback-Merkmal im Text; fängt bestehende §V6-Stellen ab, ohne deren
   Code anzufassen.

Determinismus (§G5 (copilot-instructions.md)): reine Buchführung ohne
Zeitstempel; Zählung pro Prozess (Kind-Prozesse melden ihren Snapshot über
den Payload an den Elternprozess).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}
_ART: dict[str, int] = {}
_TOTAL = 0

# Geplante Rückfälle (KEINE Defekte): bewusst gesperrte Flags
# (§v10.802 (copilot-instructions.md)) oder offene Zeugen-Modelle.
# Regeln: (Logger-Endung, Meldungs-Teilstring) → Art "geplant".
PLANNED_RULES: tuple[tuple[str, str], ...] = (
    ("phase_49_advanced_dereverb", "ML→DSP-Ersatzpfad"),  # SGMSE+ gesperrt (D-K0-3)
    ("music_mos_oracle", "SingMOS Pro nicht verfügbar"),  # Zeugen-Modell KP-12 offen
)
ARTEN = ("defekt", "geplant")

# Marker für ML→DSP-Rückfälle in Log-Meldungen (case-insensitive).
FALLBACK_MARKERS: tuple[str, ...] = ("fallback", "ersatzpfad", "dsp-ersatz")


def record_fallback(component: str, reason: str = "", art: str = "defekt") -> None:
    """Explizite Registrierung eines ML→DSP-Rückfalls (art: defekt | geplant)."""
    global _TOTAL
    key = str(component) or "unbekannt"
    art = art if art in ARTEN else "defekt"
    with _LOCK:
        _COUNTS[key] = _COUNTS.get(key, 0) + 1
        _ART[art] = _ART.get(art, 0) + 1
        _TOTAL += 1
    logging.getLogger(__name__).debug("Fallback registriert: %s (%s)", key, reason or "ohne Grund")


def reset() -> None:
    """Zähler zurücksetzen (zwischen Songs/Läufen — Song-Isolation §G1 (copilot-instructions.md))."""
    global _TOTAL
    with _LOCK:
        _COUNTS.clear()
        _ART.clear()
        _TOTAL = 0


def snapshot() -> dict[str, Any]:
    """Konsistenter Momentanstand: ``{"total": n, "by_component": {...}}``."""
    with _LOCK:
        return {"total": _TOTAL, "by_component": dict(_COUNTS), "by_art": dict(_ART)}


class FallbackLogCounter(logging.Handler):
    """Zählt WARNING+-Records mit Fallback-Merkmal (bestehende §V6-Stellen)."""

    def __init__(self, component: str = "ml") -> None:
        super().__init__(level=logging.WARNING)
        self.component = component

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging-Callback
        try:
            message = record.getMessage().lower()
        except Exception:
            # Bug 9 / §V6 (copilot-instructions.md): kein stummer Abbruch; Debug-Level
            # verhindert Rekursion in den eigenen WARNING-Zaehler.
            logging.getLogger(__name__).debug("Log-Record nicht auswertbar", exc_info=True)
            return
        if any(marker in message for marker in FALLBACK_MARKERS):
            art = (
                "geplant"
                if any(record.name.endswith(rule[0]) and rule[1] in message for rule in PLANNED_RULES)
                else "defekt"
            )
            record_fallback(self.component, record.getMessage()[:120], art=art)


def install_log_counter(logger_name: str = "backend", component: str = "ml") -> FallbackLogCounter:
    """Log-Counter an einen Logger-Hierarchie-Ast hängen (idempotent pro Ast)."""
    counter = FallbackLogCounter(component=component)
    logging.getLogger(logger_name).addHandler(counter)
    return counter


def fallback_report() -> dict[str, object]:
    """Report-Feld für Analyse-/Release-Output (Pfeiler 5)."""
    snap = snapshot()
    arten = snap["by_art"] or {}
    ungeplant = int(arten.get("defekt", 0))
    return {
        "fallback_count": snap["total"],
        "fallback_by_component": snap["by_component"],
        "fallback_by_art": arten,
        "budget": 0,
        "release_gate": "gesperrt" if ungeplant > 0 else "erfuellt (geplant ausgewiesen)",
    }

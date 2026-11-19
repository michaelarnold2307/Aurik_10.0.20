"""Defekt-Fortschritts-Modell — EINE Zählquelle für alle GUI-Defektanzeigen.

Befund (2026-10-09, Tiefenanalyse): Die Defekt-Rückwärtszählung wurde in
`modern_window.py` von **vier konkurrierenden Wächrungen** gespeist —
Typ-Zählung, Severity-Heuristik (x100), Scan-Ereigniszählung und
Engine-Behebungszählung — und von drei Stellen proporzional überschrieben
(Chirurgie-Ratio, Waveform-Marker, Prozess-Ratio). Ergebnis: springende,
rückwärtslaufende und zwischen den Anzeigen divergierende Zähler.

Dieses Modul kanonisiert die Zählung (§G9 copilot-instructions.md: eine
Messung, eine Quelle):

- **Initial** = Ereigniszählung des Scans (nur wachsende Universen: ein
  späterer, spärlicherer Scan verkleinert den Bestand nicht).
- **Behoben** = kumulative Engine-Zählung (`resolved_counts`, §v10.802),
  plus monotonen Untergrenzen aus Waveform-Markern und Chirurgie-Report.
- **Aliase** (`pops`~`clipping`, `noise`~`noise_level`, `dropout`~`dropouts`)
  werden gruppiert — kein Doppelzählen in Summen, keine Doppel-Subtraktion.
- **Monoton**: `remaining` sinkt nie, steigt nie über `total`; nie negativ.

Reine Logik ohne Qt-Import (headless-testbar, §G5 copilot-instructions.md:
deterministisch, kein Zufall, keine Uhrzeit).
"""

from __future__ import annotations

from typing import Any

# Alias-Gruppen: kanonischer Schlüssel zuerst; alle Gruppenmitglieder zeigen
# auf denselben Zähler. Deckt die in der GUI/Song-Prognose verwendeten Namen ab.
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("pops", "clipping"),
    ("noise", "noise_level"),
    ("dropout", "dropouts"),
)

_KANON: dict[str, str] = {}
for _gruppe in _ALIAS_GROUPS:
    for _name in _gruppe:
        _KANON[_name] = _gruppe[0]


def kanonischer_schluessel(name: Any) -> str:
    """Normalisiert einen Defekttyp-Namen (lowercase, Alias → Gruppenkopf)."""
    key = str(name or "").strip().lower()
    return _KANON.get(key, key)


class DefectProgressModel:
    """Kanonisches Zählmodell für Defekte eines Songs (pro Song eine Instanz).

    Alle Mutationen sind idempotent und monoton; `snapshot()` liefert den
    Anzeigezustand, den Chips, Gesamtzählung, Badge und Live-Label teilen.
    """

    def __init__(self) -> None:
        self._initial: dict[str, int] = {}
        self._resolved_exact: dict[str, int] = {}
        self._resolved_floor: dict[str, int] = {}
        self._initialized = False

    # ── Aufbau ────────────────────────────────────────────────────────────

    def set_initial(self, counts: dict[str, Any] | None) -> None:
        """Setzt/erweitert die initialen Ereigniszählungen (niemals verkleinernd).

        Erster Aufruf definiert das Universum; spätere Aufrufe dürfen nur
        neue Typen ergänzen oder größere Zählungen desselben Typs melden
        (spärliche Re-Scans verkleinern den Bestand nicht — Befund 2026-10-09).
        """
        if not counts:
            return
        for roh_key, roh_val in counts.items():
            key = kanonischer_schluessel(roh_key)
            try:
                val = int(roh_val)
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            self._initial[key] = max(self._initial.get(key, 0), val)
        if self._initial:
            self._initialized = True

    def apply_exact(self, resolved_counts: dict[str, Any] | None) -> None:
        """Kumulative Engine-Behebungen (§v10.802) — exakt, monoton."""
        if not resolved_counts:
            return
        for roh_key, roh_val in resolved_counts.items():
            key = kanonischer_schluessel(roh_key)
            try:
                val = int(roh_val)
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            self._resolved_exact[key] = max(self._resolved_exact.get(key, 0), val)

    def apply_resolved_types(self, types: list[Any] | tuple[Any, ...] | None) -> None:
        """Typ-Diff (`resolved`) als Untergrenze: je gemeldetem Typ ≥ 1 behoben.

        Nur Untergrenze (kein Zähler-Fabrizieren): hebt die Behebungszahl des
        Typs auf mindestens 1, falls Engine/Marker noch nichts meldeten.
        """
        for typ in types or []:
            key = kanonischer_schluessel(typ)
            if key in self._initial:
                self._resolved_floor[key] = max(self._resolved_floor.get(key, 0), 1)

    def apply_marker_resolved(self, resolved_by_type: dict[str, Any] | None) -> None:
        """Waveform-Marker (segmentgenau) als Untergrenze je Typ."""
        if not resolved_by_type:
            return
        for roh_key, roh_val in resolved_by_type.items():
            key = kanonischer_schluessel(roh_key)
            try:
                val = int(roh_val)
            except (TypeError, ValueError):
                continue
            if val > 0 and key in self._initial:
                self._resolved_floor[key] = max(self._resolved_floor.get(key, 0), val)

    # Hinweis (§WP-GUI-2026-10-09): Der Chirurgie-Live-Countdown („X/Y Defekte
    # repariert") hat ein EIGENES Label und speist diesen Hauptcounter NICHT
    # mehr — zwei Totale über denselben Begriff waren eine der Divergenz-
    # Quellen der Tiefenanalyse (keine proportionale Verteilung, Befund 2026-10-09).

    # ── Abfrage ───────────────────────────────────────────────────────────

    def has_initial(self) -> bool:
        return self._initialized and bool(self._initial)

    @property
    def initial_counts(self) -> dict[str, int]:
        return dict(self._initial)

    def _resolved_per_key(self, key: str) -> int:
        total = self._initial.get(key, 0)
        resolved = max(
            self._resolved_exact.get(key, 0),
            self._resolved_floor.get(key, 0),
        )
        return max(0, min(total, resolved))

    def remaining_counts(self) -> dict[str, int]:
        """Restzähler je kanonischem Schlüssel (nie negativ)."""
        if not self._initial:
            return {}
        return {key: self._initial[key] - self._resolved_per_key(key) for key in self._initial}

    def snapshot(self) -> dict[str, int]:
        """Anzeigezustand: total/remaining/resolved/resolved_pct (alias-frei).

        Invarianz: `remaining == sum(remaining_counts().values())` — Chips,
        Gesamtzählung, Badge und Live-Label können nicht mehr divergieren.
        """
        if not self._initial:
            return {"total": 0, "remaining": 0, "resolved": 0, "resolved_pct": 0}
        total = int(sum(self._initial.values()))
        remaining = int(sum(self.remaining_counts().values()))
        resolved = max(0, total - remaining)
        pct = int(round(100.0 * resolved / max(1, total)))
        return {"total": total, "remaining": remaining, "resolved": resolved, "resolved_pct": pct}


__all__ = ["DefectProgressModel", "kanonischer_schluessel"]

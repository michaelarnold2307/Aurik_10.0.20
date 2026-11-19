"""Aurik10/ui/live_narrative.py — die Stimme von Aurik während der Arbeit.

**Befund 2026-10-07 (gemessen, nicht geschätzt).** Der Nutzer sah im langen
Mittelteil einer Restaurierung **einen Satz**: `_long_phase_reassure_text`
wählte aus **drei** festen Texten, gesteuert allein über das Fortschrittsband
(`<12 %`, `<92 %`, sonst). Bei einer 30-Minuten-Phase wiederholte sich derselbe
Satz im 8-Sekunden-Takt — rund 200-mal — und im Toast alle 35 s noch einmal.
Eine Varianten- oder Rotationsmechanik existierte im gesamten Frontend **nicht**
(Suche über `modern_window.py`: kein Treffer). Zwei Schönheitsfehler steckten
ausgerechnet im meistgesehenen Satz: „ - " statt Gedankenstrich und das Wort
„Rechenintensive".

**Warum das ein Befund ist und keine Geschmacksfrage.** §v10.305
(copilot-instructions.md) verlangt „kontextbewusste Kommunikation: Jeder Song
bekommt individuelle Statusmeldungen". Die dafür nötigen Signale liegen an der
Aufrufstelle längst vor — Phase, Reparatur-Namen, aktive Störungen,
Defekt-Fortschritt, Restzeit, Material/Ära/Genre —, wurden aber für die
Beruhigungszeile **nicht** benutzt: Der informationsärmste Text stand an der
Stelle mit der längsten Wartezeit. Dieselbe Klasse wie D-K3-9 („Prüfung, die
nichts prüft"): die Zusage steht da, die Umsetzung fehlt.

**Was dieses Modul tut.** Es komponiert die Live-Zeile aus drei Teilen —

* **was gerade geschieht** (rotierender Satz je Band, 4–6 Fassungen),
* **was erreicht ist** (Störungen behoben, sobald Zahlen vorliegen),
* **wie lange noch** (vorhandener Restzeit-Text),

und liefert zusätzlich **Meilenstein-Sätze** (ein Viertel, Halbzeit, drei
Viertel, alle Störungen), die genau **einmal** erscheinen. Damit trägt jede
Meldung eine neue Information, statt denselben Zustand zu wiederholen.

**Determinismus (§G5 copilot-instructions.md).** Die Auswahl ist eine reine
Funktion des **Meldungszählers** (`index`) und des Bandes — kein Zufall, keine
Uhrzeit, kein `time.time()`. Derselbe Lauf erzeugt dieselbe Folge; Wiederholung
entsteht nur, wenn alle Fassungen durchlaufen sind, und dann in anderer
Reihenfolge als beim vorherigen Zyklus (Versatz um 1), damit zwei aufeinander
folgende Zyklen nicht identisch klingen.

Rein rechnend und ohne Qt — deshalb direkt testbar
(`tests/unit/test_live_narrative.py`).
"""

from __future__ import annotations

from collections.abc import Callable

#: Bandgrenzen — identisch zu den bisherigen Schwellen (12 % / 92 %), damit sich
#: der Zeitpunkt des Tonwechsels nicht verschiebt.
BAND_ANALYSIS_MAX = 12.0
BAND_WORKING_MAX = 92.0

#: Katalogschlüssel je Band. Mehrere Fassungen je Band sind der Kern gegen die
#: Wiederholung; jede Fassung ist ein eigener Übersetzungsschlüssel (de/en).
BAND_VARIANTS: dict[str, tuple[str, ...]] = {
    "analysis": (
        "narrative.analysis.1",
        "narrative.analysis.2",
        "narrative.analysis.3",
        "narrative.analysis.4",
    ),
    "working": (
        "narrative.working.1",
        "narrative.working.2",
        "narrative.working.3",
        "narrative.working.4",
        "narrative.working.5",
        "narrative.working.6",
    ),
    "finalize": (
        "narrative.finalize.1",
        "narrative.finalize.2",
        "narrative.finalize.3",
    ),
}

#: Meilensteine des Störungs-Fortschritts in Prozent → Katalogschlüssel.
MILESTONES: tuple[tuple[int, str], ...] = (
    (25, "narrative.milestone.quarter"),
    (50, "narrative.milestone.half"),
    (75, "narrative.milestone.three_quarters"),
    (100, "narrative.milestone.all"),
)
MILESTONE_KEYS: tuple[str, ...] = tuple(key for _, key in MILESTONES)


def band_for(ui_pct: float) -> str:
    """Ordnet den Fortschritt einem Erzählband zu (analysieren/arbeiten/abschließen)."""
    if ui_pct < BAND_ANALYSIS_MAX:
        return "analysis"
    if ui_pct < BAND_WORKING_MAX:
        return "working"
    return "finalize"


def variant_key(band: str, index: int) -> str:
    """Deterministische Fassung für die n-te Meldung dieses Bandes.

    ``index % n`` wandert durch alle Fassungen; der ``+ index // n``-Anteil
    versetzt jeden weiteren Durchlauf um eine Stelle, damit zwei aufeinander
    folgende Zyklen nicht dieselbe Reihenfolge haben. §G5 (copilot-instructions.md)
    verbietet Zufall und Uhrzeit als Entscheidungsquelle — hier zählt allein der
    Meldungszähler.
    """
    variants = BAND_VARIANTS.get(band) or BAND_VARIANTS["working"]
    n = len(variants)
    if n <= 1:
        return variants[0]
    _i = max(0, int(index))
    return variants[(_i + _i // n) % n]


def milestone_key(defects_done: int, defects_total: int, seen: set[str]) -> str | None:
    """Nächster noch nicht gemeldeter Meilenstein — sonst ``None``.

    ``seen`` ist der Zustand des Aufrufers (Menge bereits gemeldeter Schlüssel);
    das Modul selbst hält keinen Zustand (§V8 copilot-instructions.md: kein
    verstecktes Modul-Gedächtnis, das einen Song in den nächsten trägt).
    """
    if defects_total <= 0 or defects_done <= 0:
        return None
    pct = 100.0 * float(defects_done) / float(defects_total)
    for threshold, key in MILESTONES:
        if pct + 1e-9 >= threshold and key not in seen:
            return key
    return None


def compose_reassure(
    *,
    band: str,
    index: int,
    defects_done: int = 0,
    defects_total: int = 0,
    eta_text: str = "",
    msg: Callable[..., str] | None = None,
) -> str:
    """Baut die Live-Zeile: was geschieht · was erreicht ist · wie lange noch.

    ``msg`` ist injizierbar (Standard: ``Aurik10.i18n.t``), damit die Funktion
    ohne Qt und ohne Sprachzustand testbar bleibt.
    """
    if msg is None:
        from Aurik10.i18n import t

        msg = t

    parts: list[str] = [msg(variant_key(band, index))]
    if defects_total > 0:
        parts.append(msg("narrative.clause.defects", done=defects_done, total=defects_total))
    if eta_text:
        parts.append(msg("narrative.clause.eta", eta=eta_text))
    return "  ·  ".join(part for part in parts if part)


__all__ = [
    "BAND_ANALYSIS_MAX",
    "BAND_VARIANTS",
    "BAND_WORKING_MAX",
    "MILESTONES",
    "MILESTONE_KEYS",
    "band_for",
    "compose_reassure",
    "milestone_key",
    "variant_key",
]

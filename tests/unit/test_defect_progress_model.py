"""§WP-GUI-2026-10-09: Kanonisches Defekt-Fortschritts-Modell — Vertragstests.

Befund vor dem Fix (Tiefenanalyse 2026-10-09): Die Rückwärtszählung wurde aus
vier Zählwährungen gespeist (Typen, Severity×100-Heuristik, Scan-Ereignisse,
Engine-Behebungen) und von drei Stellen proportional überschrieben — Zähler
sprangen, divergierten zwischen Anzeigen und liefen teils rückwärts.

Diese Tests pinnen das kanonische Modell:
  - Monotonie: `remaining` sinkt nie, nie negativ, nie über `total`.
  - Alias-Gruppen (pops~clipping, noise~noise_level, dropout~dropouts):
    Summen zählen jeden Defekt EINMAL; Behebungen subtrahieren EINMAL.
  - Spärliche Re-Scans verkleinern das Universum nicht (§G9 copilot-instructions.md).
  - Exakte Engine-Zählung (§v10.802) und Marker/Chirurgie-Untergrenzen.
  - Determinismus (§G5 copilot-instructions.md): keine Uhrzeit, kein Zufall.
"""

from __future__ import annotations

from Aurik10.ui.defect_progress_model import DefectProgressModel, kanonischer_schluessel


def test_kanonischer_schluessel_aliasgruppen() -> None:
    assert kanonischer_schluessel("clipping") == "pops"
    assert kanonischer_schluessel("POPS") == "pops"
    assert kanonischer_schluessel("noise_level") == "noise"
    assert kanonischer_schluessel("DROPOUTS") == "dropout"
    assert kanonischer_schluessel("CLICKS") == "clicks"


def test_initial_zaehlt_aliase_einmal() -> None:
    m = DefectProgressModel()
    m.set_initial({"pops": 10, "clipping": 10, "clicks": 5})
    snap = m.snapshot()
    assert snap["total"] == 15, "Alias-Gruppe darf NICHT doppelt zählen"
    assert snap["remaining"] == 15


def test_spaerlicher_rescan_verkleinert_nicht() -> None:
    m = DefectProgressModel()
    m.set_initial({"clicks": 985, "hum": 1})
    m.set_initial({"clicks": 12})  # spärlicher Re-Scan
    snap = m.snapshot()
    assert snap["total"] == 986, "Bestand darf nicht schrumpfen (Befund 2026-10-09)"


def test_exakte_engine_zaehlung_monoton_und_aliasfrei() -> None:
    m = DefectProgressModel()
    m.set_initial({"pops": 10, "clicks": 5})
    m.apply_exact({"CLIPPING": 3})
    assert m.snapshot()["remaining"] == 12
    m.apply_exact({"CLIPPING": 7})
    assert m.snapshot()["remaining"] == 8
    # Kein Rückwärts: kleinere (veraltete) Engine-Zahl hebt Behebungen nicht auf.
    m.apply_exact({"CLIPPING": 2})
    assert m.snapshot()["remaining"] == 8
    # Nie über total / nie negativ:
    m.apply_exact({"CLIPPING": 9999})
    assert m.snapshot()["remaining"] == 5


def test_marker_untergrenze_und_pending_schutz() -> None:
    m = DefectProgressModel()
    m.set_initial({"clicks": 20})
    m.apply_marker_resolved({"clicks": 5})
    assert m.snapshot()["remaining"] == 15
    # Marker zählen nie mehr als initial:
    m.apply_marker_resolved({"clicks": 500})
    assert m.snapshot()["remaining"] == 0


def test_chirurgie_speist_hauptcounter_nicht() -> None:
    """Der Chirurgie-Live-Countdown hat ein eigenes Label — kein zweiter Zähler."""
    m = DefectProgressModel()
    assert not hasattr(m, "apply_chirurgie")


def test_invarianz_remaining_entspricht_typsumme() -> None:
    m = DefectProgressModel()
    m.set_initial({"clicks": 100, "hum": 1})
    m.apply_exact({"clicks": 33})
    m.apply_marker_resolved({"hum": 1})
    snap = m.snapshot()
    assert snap["remaining"] == sum(m.remaining_counts().values())
    assert snap["total"] == snap["remaining"] + snap["resolved"]


def test_snapshot_pct_und_leeres_modell() -> None:
    m = DefectProgressModel()
    assert m.snapshot() == {"total": 0, "remaining": 0, "resolved": 0, "resolved_pct": 0}
    assert not m.has_initial()
    m.set_initial({"clicks": 4})
    m.apply_exact({"clicks": 1})
    snap = m.snapshot()
    assert snap["resolved_pct"] == 25
    assert m.remaining_counts() == {"clicks": 3}


def test_typ_diff_als_untergrenze() -> None:
    m = DefectProgressModel()
    m.set_initial({"clicks": 10, "hum": 1})
    m.apply_resolved_types(["hum", "unbekannt"])
    assert m.remaining_counts()["hum"] == 0
    assert "unbekannt" not in m.remaining_counts(), "unbekannter Typ darf nichts tun"


def test_determinismus_zwei_instanzen_gleich() -> None:
    def _lauf() -> dict[str, int]:
        m = DefectProgressModel()
        m.set_initial({"clicks": 9, "noise_level": 2})
        for k in (1, 3, 5):
            m.apply_exact({"CLICKS": k})
        return m.snapshot()

    assert _lauf() == _lauf()

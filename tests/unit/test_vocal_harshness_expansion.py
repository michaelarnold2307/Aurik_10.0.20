"""Vertragstests: Vocal-Harshness-Datenexpansion + Manifest-Anhänger (2026-11-19).

Modell-frei: die Datenerzeugungs-Kette (SOTA-HANDOVER 37c) wird über
Konventionen und Dedup-Semantik gepinnt — kein Schaden, keine Pipeline, kein
Detektor in den Tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import erstelle_vocal_harshness_expansion as vh
from scripts import haenge_klassen_manifest_an as haenge


def _zeile(klasse: str, titel: str, rezept: str, start: int, end: int) -> dict:
    return {"klasse": klasse, "titel": titel, "rezept": rezept, "start": start, "end": end}


def test_vh_rezepte_und_konventionen() -> None:
    """Genau die zwei Rezepte, aus denen die 4 Bestandsregionen stammen;
    Dateinamen folgen der erstelle_klassen_paare-Konvention `<titel> (<rezept>).wav`."""
    assert vh.REZEPTE == ("clipping", "kassette")
    quelle = Path(vh.__file__).read_text(encoding="utf-8")
    assert 'f"{stamm} ({rezept}).wav"' in quelle
    assert 'f"{stamm} ({rezept})_restauriert.wav"' in quelle
    assert '"*/mixture.wav"' in quelle  # MUSDB-Mixe, Stamm = Titelordner (Namens-Kollision)


def test_vh_schema_und_härtung() -> None:
    quelle = Path(vh.__file__).read_text(encoding="utf-8")
    assert vh.SCHEMA == "aurik.vocal_harshness_expansion/1"
    assert "faulthandler" in quelle  # Crash-Diagnose wie in den übrigen Ketten-Skripten
    assert "AURIK_FORCE_CPU" in quelle  # CPU-Vertrag, GPU-Kette läuft parallel
    assert "bereits fertig, übersprungen (Resume)" in quelle  # Resume-Sicherheit


def test_anhaengen_dedupliziert_idempotent(tmp_path) -> None:
    quelle = tmp_path / "neu.jsonl"
    quelle.write_text(
        json.dumps(_zeile("vocal_harshness", "T1", "clipping", 0, 100))
        + "\n"
        + json.dumps(_zeile("groove_echo", "T1", "clipping", 0, 100))
        + "\n",
        encoding="utf-8",
    )
    ziel = tmp_path / "haupt.jsonl"
    ziel.write_text(json.dumps(_zeile("vocal_harshness", "T1", "clipping", 0, 100)) + "\n", encoding="utf-8")
    assert haenge.main(["--quelle", str(quelle), "--ziel", str(ziel), "--klasse", "vocal_harshness"]) == 0
    zeilen = [json.loads(l) for l in ziel.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(zeilen) == 1, "Duplikat darf nicht erneut angehängt werden"
    # Zweiter Lauf: weiterhin idempotent.
    assert haenge.main(["--quelle", str(quelle), "--ziel", str(ziel), "--klasse", "vocal_harshness"]) == 0
    assert len([l for l in ziel.read_text(encoding="utf-8").splitlines() if l.strip()]) == 1


def test_anhaengen_nur_eigene_klasse_und_fremde_bleiben(tmp_path) -> None:
    quelle = tmp_path / "neu.jsonl"
    quelle.write_text(
        json.dumps(_zeile("vocal_harshness", "T2", "kassette", 10, 20))
        + "\n"
        + json.dumps(_zeile("clicks", "T2", "kassette", 10, 20))
        + "\n",
        encoding="utf-8",
    )
    ziel = tmp_path / "haupt.jsonl"
    ziel.write_text(json.dumps(_zeile("clicks", "T0", "kassette", 1, 2)) + "\n", encoding="utf-8")
    assert haenge.main(["--quelle", str(quelle), "--ziel", str(ziel), "--klasse", "vocal_harshness"]) == 0
    zeilen = [json.loads(l) for l in ziel.read_text(encoding="utf-8").splitlines() if l.strip()]
    klassen = {z["klasse"] for z in zeilen}
    assert "vocal_harshness" in klassen and "clicks" in klassen
    assert len(zeilen) == 2, "nur die vocal_harshness-Zeile wird angehängt, Bestand bleibt"


def test_anhaengen_fehlt_quelle_exit_0(tmp_path) -> None:
    """Fehlendes Vorgänger-Artefakt darf die Kette NICHT hängen lassen (Exit 0)."""
    ziel = tmp_path / "haupt.jsonl"
    ziel.write_text("", encoding="utf-8")
    assert haenge.main(["--quelle", str(tmp_path / "fehlt.jsonl"), "--ziel", str(ziel), "--klasse", "crackle"]) == 0

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


def test_vh_resume_report_enthaelt_bereits_fertige_paare(tmp_path, monkeypatch) -> None:
    import numpy as np

    from scripts import erstelle_schadens_proben as schadens_proben

    root = tmp_path
    quellen = root / "quellen"
    ausgang = root / "ausgang"
    clean_links = root / "clean"
    report_dir = root / "reports"
    paare = {(f"Titel {idx}", rezept) for idx in range(3) for rezept in vh.REZEPTE}
    for titel in {t for t, _ in paare}:
        quelle = quellen / titel / "mixture.wav"
        quelle.parent.mkdir(parents=True, exist_ok=True)
        quelle.write_bytes(b"source")
        for rezept in vh.REZEPTE:
            (ausgang / f"{titel} ({rezept}).wav").parent.mkdir(parents=True, exist_ok=True)
            (ausgang / f"{titel} ({rezept}).wav").write_bytes(b"degraded")
            restauriert = ausgang / f"{titel} ({rezept})_restauriert.wav"
            restauriert.write_bytes(b"" if (titel, rezept) == ("Titel 0", "clipping") else b"restored")

    monkeypatch.setattr(vh, "_ROOT", root)
    monkeypatch.setattr(vh, "REPORT_DIR", report_dir)
    monkeypatch.setattr(vh, "_lade_mono_48k", lambda _quelle: (np.ones(8, dtype=np.float32), vh.SR))
    monkeypatch.setattr(schadens_proben, "schaden_rezept", lambda snippet, _rezept, _rng: snippet.copy())
    monkeypatch.setattr(schadens_proben, "restauriere_schnipsel", lambda audio, *_args: audio)
    report_dir.mkdir()
    assert (
        vh.main(
            [
                "--titel",
                "3",
                "--seed",
                "42",
                "--quellen",
                str(quellen),
                "--ausgang",
                str(ausgang),
                "--clean-links",
                str(clean_links),
            ]
        )
        == 0
    )

    reports = list(report_dir.glob("*_vocal_harshness_expansion.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert {(entry["titel"], entry["rezept"]) for entry in report["eintraege"]} == paare
    assert (ausgang / "Titel 0 (clipping)_restauriert.wav").stat().st_size > 0


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

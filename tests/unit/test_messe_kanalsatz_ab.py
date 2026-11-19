"""Vertragstests: DRT-12-Kanalsatz-A/B (scripts/messe_kanalsatz_ab.py, 2026-11-19).

Modell-frei: Flag-Aktivierung ist mess-only (prozess-lokal, keine Flag-Datei),
die Paar-Auswahl ist deterministisch (Seed 42), die A/B-Zuordnung nutzt die
bewertung.json-Schlüssel (song + rezept), und das Skript initialisiert kein
CUDA (CPU-only-Vertrag, stört die GPU-Kette nicht).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import messe_kanalsatz_ab as ab

_ROOT = Path(__file__).resolve().parents[2]


def test_kanalsatz_flags_vollstaendig() -> None:
    """Alle 12 gate-bestandenen Kanal-Flags sind gelistet (10 Alt + 2 Neu)."""
    assert len(ab.KANALSAETZ_FLAGS) == 12
    assert "use_witness_kopf_clicks" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_groove_echo" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_dynamic_compression_excess" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_dropout_head_contact" in ab.KANALSAETZ_FLAGS


def test_flag_aktivierung_nur_prozess_lokal() -> None:
    """_aktiviere_kanalsatz setzt Attribute, schreibt aber keine Flag-Datei (§G1)."""
    import backend.core.music_model_flags as flags

    ab._aktiviere_kanalsatz()
    assert flags.use_witness_kopf_clicks is True
    assert flags.use_witness_kopf_dropout_head_contact is True
    # Kein Datei-Effekt: die Defaults des Moduls sind Quelltext — es gibt keinen
    # Persistenz-Aufruf im Skript.
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert "flag_freigabe" not in quelle
    assert "open(" not in quelle.split("def main")[1] or ".json" not in quelle.split("def main")[1].split("REPORT_DIR")[0]


def test_paar_auswahl_deterministisch(monkeypatch, tmp_path) -> None:
    for i in range(8):
        (tmp_path / f"song{i} (vinyl).wav").write_bytes(b"x")
    a = ab._paare_auswaehlen(tmp_path, 3, 42)
    b = ab._paare_auswaehlen(tmp_path, 3, 42)
    assert a == b and len(a) == 3
    assert all("_restauriert" not in p.name for _, p in a)


def test_bestand_key_song_plus_rezept() -> None:
    """A-Seite: bewertung.json-Schlüssel = 'song (rezept)' — deckungsgleich mit den Paar-Stems."""
    import json

    roh = {"zeilen": [{"song": "Titel A", "rezept": "vinyl", "delta_musik_mos": 0.1, "delta_defektlast": -0.2}]}
    (Path(ab._ROOT) / "tmp_bewertung_test.json").write_text(json.dumps(roh), encoding="utf-8")
    # Der Schlüssel wird in main() gebaut — hier als Quelltext-Pin:
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert 'key = f"{song} ({rezept})" if rezept else song' in quelle


def test_schema_pin_und_cpu_only() -> None:
    assert ab.SCHEMA == "aurik.kanalsatz_ab/1"
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("AURIK_FORCE_CPU", "1")' in quelle
    assert ".cuda(" not in quelle and "cuda()" not in quelle

"""Vertragstests für den M9-Leak-Schutz (--exclude-pairs + Ausschlussliste, 2026-11-14)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.erstelle_m9_exclude import erstelle_exclude_liste
from scripts.train_tf_gridnet_denoiser import PairedChunkDataset


def test_exclude_liste_aus_manifest_und_gate(tmp_path: Path) -> None:
    """Track-Namen → Paar-Dateien, deterministisch sortiert."""
    manifest = tmp_path / "manifest.json"
    gate = tmp_path / "gate.json"
    manifest.write_text(
        json.dumps(
            {
                "paare": [
                    {"datei": "pair_0000.wav", "quelle": "train/Alpha/mixture.wav"},
                    {"datei": "pair_0001.wav", "quelle": "test/Beta/mixture.wav"},
                    {"datei": "pair_0002.wav", "quelle": "test/Gamma/mixture.wav"},
                ]
            }
        ),
        encoding="utf-8",
    )
    gate.write_text(json.dumps({"tracks": ["Alpha", "Gamma"]}), encoding="utf-8")
    liste = erstelle_exclude_liste(manifest, gate)
    assert liste == ["pair_0000.wav", "pair_0002.wav"]


def test_exclude_liste_fail_closed_ohne_treffer(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    gate = tmp_path / "gate.json"
    manifest.write_text(json.dumps({"paare": [{"datei": "p.wav", "quelle": "train/X/mixture.wav"}]}), encoding="utf-8")
    gate.write_text(json.dumps({"tracks": ["Unbekannt"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        erstelle_exclude_liste(manifest, gate)


def test_dataset_filtert_ausgeschlossene_paare(tmp_path: Path) -> None:
    """PairedChunkDataset schließt gelistete Paar-Dateien aus (Leak-Schutz)."""
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    clean.mkdir()
    noisy.mkdir()
    for name in ("pair_0.wav", "pair_1.wav", "pair_2.wav"):
        (clean / name).write_bytes(b"x")
        (noisy / name).write_bytes(b"x")
    ds = PairedChunkDataset(clean, noisy, exclude_pairs={"pair_1.wav"})
    namen = {p.name for p, _ in ds._pairs}
    assert namen == {"pair_0.wav", "pair_2.wav"}


def test_dataset_ohne_exclude_behaelt_alle(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    clean.mkdir()
    noisy.mkdir()
    (clean / "pair_0.wav").write_bytes(b"x")
    (noisy / "pair_0.wav").write_bytes(b"x")
    ds = PairedChunkDataset(clean, noisy)
    assert len(ds._pairs) == 1


def test_dataset_fail_closed_wenn_alles_ausgeschlossen(tmp_path: Path) -> None:
    """Alles ausgeschlossen ⇒ ehrlicher Fehler statt leerem Datensatz."""
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    clean.mkdir()
    noisy.mkdir()
    (clean / "pair_0.wav").write_bytes(b"x")
    (noisy / "pair_0.wav").write_bytes(b"x")
    with pytest.raises(ValueError):
        PairedChunkDataset(clean, noisy, exclude_pairs={"pair_0.wav"})

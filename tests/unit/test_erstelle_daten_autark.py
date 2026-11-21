#!/usr/bin/env python3
"""Vertragstests der autarken Daten-Erzeuger (R5/R6/D-K0-8, 2026-11-12).

Geprüft wird der DATEN-Vertrag (modell-frei, deterministisch, CPU-only):

- ``erstelle_cantus_real_paare`` (R5, SOTA-M14): Paarung
  ``<stem>_degraded.wav`` <-> ``<stem>_clean.wav`` exakt wie von
  ``train_cantus.py`` Z. 644/645 konsumiert, Manifest-Felder,
  sha256_pcm == Hash der float32-PCM-Daten, Determinismus (zwei Läufe,
  identische Hashes).
- ``erstelle_gender_labels`` (R6): CSV mit exakt der Kopfzeile
  ``file,gender``, alle Labels gültig (Validator des Trainers selbst:
  ``train_gender_head._read_labels``), Pfade relativ zum MUSDB-Root.
- ``erstelle_corpus_vinyl`` (D-K0-8): Bestands-Manifest wird NIE
  zerstört (D-K3-80-Klasse) — vorhandene Einträge bleiben, neue werden
  angehängt, dedupe per ``file``, Wiederholung idempotent.

Fixtures sind synthetische 48-kHz-WAVs (Sinus + Rauschen + Impuls),
keine Modelle, kein GPU, keine Netzwerk-Zugriffe. Determinismus: Seed 42
(§G5 (GEBOTE.md)).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.erstelle_cantus_real_paare as cantus_real
import scripts.erstelle_corpus_vinyl as corpus_vinyl
import scripts.erstelle_gender_labels as gender_labels
from scripts.csv_export_safety import render_spreadsheet_safe_csv

_SR = 48000


def _sinus(freq: float, sek: float, sr: int = _SR) -> np.ndarray:
    _t = np.arange(int(sek * sr), dtype=np.float64) / sr
    return (0.5 * np.sin(2 * np.pi * freq * _t)).astype(np.float32)


def _rauschen(sek: float, seed: int, sr: int = _SR) -> np.ndarray:
    _rng = np.random.default_rng(seed)
    return (0.02 * _rng.standard_normal(int(sek * sr))).astype(np.float32)


def _ir(sek: float, sr: int = _SR) -> np.ndarray:
    """Deterministische Pseudо-RIR (abklingender Impuls)."""
    _n = int(sek * sr)
    _h = np.zeros(_n, dtype=np.float32)
    _h[0] = 1.0
    _h[1:] = 0.3 * np.exp(-np.arange(1, _n) / (0.1 * sr))
    return _h.astype(np.float32)


def _musdb_fixture(tmp_path: Path, tracks: list[str]) -> Path:
    """Mini-MUSDB-Layout: <root>/train/<track>/vocals.wav (2 s, 110/220 Hz)."""
    _root = tmp_path / "musdb"
    for _i, _track in enumerate(tracks):
        _dir = _root / "train" / _track
        _dir.mkdir(parents=True)
        _freq = 110.0 if _i % 2 == 0 else 220.0
        sf.write(_dir / "vocals.wav", _sinus(_freq, 2.0), _SR)
    return _root


def _bank_fixture(tmp_path: Path) -> Path:
    _dir = tmp_path / "bank"
    _dir.mkdir(parents=True)
    sf.write(_dir / "noise_a.wav", _rauschen(3.0, 7), _SR)
    sf.write(_dir / "noise_b.wav", _rauschen(3.0, 8), _SR)
    return _dir


def _rir_fixture(tmp_path: Path) -> Path:
    _dir = tmp_path / "rir"
    _dir.mkdir(parents=True)
    sf.write(_dir / "room_a.wav", _ir(1.5), _SR)
    return _dir


def _pcm_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()


class TestCantusRealPaare:
    def test_paarung_und_manifest(self, tmp_path: Path):
        _musdb = _musdb_fixture(tmp_path, ["Track A", "Track B"])
        _bank = _bank_fixture(tmp_path)
        _rirs = _rir_fixture(tmp_path)
        _out = tmp_path / "cantus_real"
        _rc = cantus_real.main(
            [
                "--musdb-root",
                str(_musdb),
                "--noise-bank",
                str(_bank),
                "--ir-dir",
                str(_rirs),
                "--out",
                str(_out),
                "--tracks",
                "2",
                "--seconds",
                "2",
                "--seed",
                "42",
            ]
        )
        assert _rc == 0
        _deg = sorted(_out.glob("*_degraded.wav"))
        assert len(_deg) == 6  # 2 Titel × 3 Arten
        for _d in _deg:
            # train_cantus-Paarung (Z. 644/645): degradiert -> clean per Name
            _c = _d.with_name(_d.name.replace("_degraded.wav", "_clean.wav"))
            assert _c.is_file()
        _manifest = json.loads((_out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert _manifest["split"] == "train"
        assert _manifest["kind"] in ("vinyl", "rir", "vinyl_rir")
        assert "sha256_pcm" in _manifest and "sha256_pcm_clean" in _manifest
        _deg_audio = sf.read(str(_out / _manifest["degraded"]), dtype="float32")[0]
        assert _manifest["sha256_pcm"] == _pcm_sha256(_deg_audio)

    def test_determinismus(self, tmp_path: Path):
        _musdb = _musdb_fixture(tmp_path, ["Track A"])
        _bank = _bank_fixture(tmp_path)
        _rirs = _rir_fixture(tmp_path)
        _outs: list[str] = []
        for _i in range(2):
            _o = tmp_path / f"out{_i}"
            assert (
                cantus_real.main(
                    [
                        "--musdb-root",
                        str(_musdb),
                        "--noise-bank",
                        str(_bank),
                        "--ir-dir",
                        str(_rirs),
                        "--out",
                        str(_o),
                        "--tracks",
                        "1",
                        "--seconds",
                        "2",
                    ]
                )
                == 0
            )
            _outs.append(str(_o))
        _h1 = sorted(
            p["sha256_pcm"] for p in map(json.loads, (tmp_path / "out0" / "manifest.jsonl").read_text().splitlines())
        )
        _h2 = sorted(
            p["sha256_pcm"] for p in map(json.loads, (tmp_path / "out1" / "manifest.jsonl").read_text().splitlines())
        )
        assert _h1 == _h2


class TestGenderLabels:
    def test_csv_vertrag(self, tmp_path: Path):
        _musdb = _musdb_fixture(tmp_path, ["Track A", "Track B"])
        _csv = tmp_path / "gender_labels.csv"
        assert gender_labels.main(["--musdb-root", str(_musdb), "--out", str(_csv), "--sekunden", "2"]) == 0
        assert _csv.is_file()
        from scripts.train_gender_head import _read_labels

        _pairs = _read_labels(_csv)
        assert _pairs
        for _name, _label in _pairs:
            assert _label in ("male", "female", "child")
            # Pfade sind relativ zum MUSDB-Root auflösbar
            assert (_musdb / _name).is_file()

    def test_csv_export_quotes_values_and_rejects_formula_cells(self) -> None:
        rendered = render_spreadsheet_safe_csv(["file", "gender"], [["track.wav", "female"]])
        assert list(csv.reader(io.StringIO(rendered))) == [["file", "gender"], ["track.wav", "female"]]
        assert '"track.wav"' in rendered

        for value in ("=1+1", "  =1+1", "\t=1+1", "\v=1+1", "+SUM(A1:A2)", "-1+2", "@SUM(A1:A2)"):
            with pytest.raises(ValueError, match="formula-like cell"):
                render_spreadsheet_safe_csv(["file", "gender"], [[value, "female"]])


class TestCorpusVinyl:
    def test_bestand_wird_nie_zerstoert(self, tmp_path: Path):
        _out = tmp_path / "vinyl"
        _out.mkdir()
        (_out / "damaged").mkdir()
        _alt = _out / "damaged" / "alt_probe.wav"
        sf.write(_alt, _sinus(440.0, 1.0), _SR)
        _alt_sha = _pcm_sha256(sf.read(str(_alt), dtype="float32")[0])
        (_out / "manifest.yaml").write_text(
            "corpus_version: 1.0.0\n"
            "material: vinyl\n"
            "entries:\n"
            f"- file: damaged/alt_probe.wav\n  duration_s: 1.0\n  sample_rate: {_SR}\n"
            "  material: vinyl\n  era_year: 1965\n  genre: pop\n"
            f"  checksum_sha256: {_alt_sha}\n",
            encoding="utf-8",
        )
        _bank = _bank_fixture(tmp_path)
        assert corpus_vinyl.main(["--out", str(_out), "--noise-bank", str(_bank), "--count", "2", "--seed", "42"]) == 0
        import yaml

        _manifest = yaml.safe_load((_out / "manifest.yaml").read_text(encoding="utf-8"))
        _files = [str(e["file"]) for e in _manifest["entries"]]
        assert "damaged/alt_probe.wav" in _files  # Bestand erhalten
        assert len(_files) == 3  # 1 alt + 2 neu
        assert len(set(_files)) == 3  # kein Duplikat

    def test_idempotent(self, tmp_path: Path):
        _out = tmp_path / "vinyl"
        _bank = _bank_fixture(tmp_path)
        assert corpus_vinyl.main(["--out", str(_out), "--noise-bank", str(_bank), "--count", "2", "--seed", "42"]) == 0
        import yaml

        _n1 = len(yaml.safe_load((_out / "manifest.yaml").read_text(encoding="utf-8"))["entries"])
        assert corpus_vinyl.main(["--out", str(_out), "--noise-bank", str(_bank), "--count", "2", "--seed", "42"]) == 0
        _n2 = len(yaml.safe_load((_out / "manifest.yaml").read_text(encoding="utf-8"))["entries"])
        assert _n2 == _n1

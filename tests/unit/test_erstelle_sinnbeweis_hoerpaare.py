#!/usr/bin/env python3
"""Vertragstests des Sinn-Beweis-§4-Hörpaket-Erzeugers (2026-11-18, modell-frei)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.erstelle_sinnbeweis_hoerpaare import _match_p25, _p25

ROOT = Path(__file__).resolve().parent.parent.parent


def test_match_p25_gleicht_pegel_an() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(48000).astype(np.float32) * 0.1
    b = a * 2.0  # 6 dB lauter
    bm = _match_p25(b, a)
    assert abs(_p25(bm) - _p25(a)) < 1e-6, "p25 nicht angeglichen"
    assert abs(_p25(bm) - _p25(b)) > 1e-3, "kein Effekt erkennbar"


def test_fail_closed_ueberschreiben() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        verzeichnis = Path(tmp) / "v"
        ausgang = Path(tmp) / "aus"
        verzeichnis.mkdir()
        (verzeichnis / "schadens_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "aurik.schadensproben/1",
                    "songs": [
                        {
                            "stamm": "T1",
                            "rezept": "raum",
                            "schaden": str(verzeichnis / "s.wav"),
                            "restauriert": str(verzeichnis / "r.wav"),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        x = np.zeros(48000, dtype=np.float32)
        sf.write(str(verzeichnis / "s.wav"), x, 48000)
        sf.write(str(verzeichnis / "r.wav"), x, 48000)
        ausgang.mkdir()
        (ausgang / "T1_A.wav").write_bytes(b"block")
        r = subprocess.run(
            [
                str(ROOT / ".venv_aurik" / "bin" / "python"),
                "-B",
                str(ROOT / "scripts" / "erstelle_sinnbeweis_hoerpaare.py"),
                "--verzeichnis",
                str(verzeichnis),
                "--rezepte",
                "raum",
                "--paare",
                "1",
                "--ausgang",
                str(ausgang),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 2, f"fail-closed verletzt: {r.returncode}"
        assert "ABBRUCH" in r.stdout or "ABBRUCH" in r.stderr


def test_schluessel_getrennt_und_vollstaendig() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        verzeichnis = Path(tmp) / "v"
        ausgang = Path(tmp) / "aus"
        verzeichnis.mkdir()
        songs = []
        rng = np.random.default_rng(2)
        for i in range(2):
            x = rng.standard_normal(48000).astype(np.float32) * 0.05
            s = verzeichnis / f"s{i}.wav"
            r = verzeichnis / f"r{i}.wav"
            sf.write(str(s), x, 48000)
            sf.write(str(r), x * 1.1, 48000)
            songs.append({"stamm": f"T{i}", "rezept": "raum", "schaden": str(s), "restauriert": str(r)})
        (verzeichnis / "schadens_manifest.json").write_text(
            json.dumps({"schema": "aurik.schadensproben/1", "songs": songs}), encoding="utf-8"
        )
        r = subprocess.run(
            [
                str(ROOT / ".venv_aurik" / "bin" / "python"),
                "-B",
                str(ROOT / "scripts" / "erstelle_sinnbeweis_hoerpaare.py"),
                "--verzeichnis",
                str(verzeichnis),
                "--rezepte",
                "raum",
                "--paare",
                "2",
                "--ausgang",
                str(ausgang),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        keys = json.loads((ausgang / "schluessel.json").read_text(encoding="utf-8"))
        assert len(keys) == 4
        for name in keys:
            assert keys[name] in ("schaden", "restauriert")
        # Je Paar genau ein A und ein B mit komplementärem Schlüssel
        for i in range(2):
            assert keys[f"T{i}_A.wav"] != keys[f"T{i}_B.wav"]
        manifest = json.loads((ausgang / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema"] == "aurik.hoerpaare/1"

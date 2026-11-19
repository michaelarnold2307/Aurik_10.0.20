#!/usr/bin/env python3
"""Vertragstests fuer scripts/erstelle_bandlimit_paare.py (modell-frei).

Fahrplan B-0(1) (docs/reports/current/2026-11-13_aktivierungs_fahrplan.md):
Der bandwidth_loss-Datensatz ist die Grundlage des Klassenkopfs der ersten
metrik-sichtbaren Q≈0-Klasse. Gepinnt werden Determinismus (§G5 (copilot-instructions.md)),
der Kontrast-Filter (nur defektlast-Delta > JND) und fail-closed-Pfade.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.erstelle_bandlimit_paare as eb


def test_sha256_pcm_deterministisch_und_stabil() -> None:
    x = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    assert eb._sha256_pcm(x) == eb._sha256_pcm(x)
    assert len(eb._sha256_pcm(x)) == 64


def _musikaehnlich(n: int, rng: np.random.Generator) -> np.ndarray:
    """Musik-ähnliches Test-Signal (Rauschen + Töne): Bandlimit erzeugt hier
    defektlast-Kontrast in richtiger Richtung (hf_verlust steigt)."""
    t = np.arange(n, dtype=np.float64) / 48000.0
    x = rng.standard_normal(n).astype(np.float64) * 0.3
    for f in (440.0, 880.0, 3000.0):
        x += 0.4 * np.sin(2 * np.pi * f * t)
    return (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)


def test_manifest_form_nach_generierung(tmp_path) -> None:
    """Kleiner End-to-End: 2 Titel, bandlimit-Variante, Kontrast-Filter aktiv."""
    import numpy as np
    from scipy.io import wavfile

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    rng = np.random.default_rng(7)
    for i in range(2):
        x = _musikaehnlich(eb.WINDOW * 8, rng)
        wavfile.write(clean_dir / f"titel_{i}.wav", 48000, (x * 32767).astype(np.int16))

    rc = eb.main(
        [
            "--clean-dir",
            str(clean_dir),
            "--out",
            str(tmp_path / "paare"),
            "--fenster-pro-titel",
            "4",
        ]
    )
    assert rc == 0
    manifest = tmp_path / "paare" / "manifest.jsonl"
    zeilen = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert zeilen, "Kontrast-Filter hat alle Regionen verworfen (Rauschen + Bandlimit muss Kontrast haben)"
    for z in zeilen:
        assert z["klasse"] == "bandwidth_loss"
        assert z["rezept"] == "bandlimit"
        assert "schaden" in z and "clean" in z
        assert len(z["sha256_pcm_schaden"]) == 64 and len(z["sha256_pcm_clean"]) == 64


def test_zwei_laeufe_identisches_manifest(tmp_path) -> None:
    from scipy.io import wavfile

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    rng = np.random.default_rng(3)
    x = _musikaehnlich(eb.WINDOW * 8, rng)
    wavfile.write(clean_dir / "t.wav", 48000, (x * 32767).astype(np.int16))
    args = ["--clean-dir", str(clean_dir), "--out", str(tmp_path / "p"), "--fenster-pro-titel", "4"]
    assert eb.main(args) == 0
    m1 = (tmp_path / "p" / "manifest.jsonl").read_bytes()
    assert eb.main(args) == 0
    m2 = (tmp_path / "p" / "manifest.jsonl").read_bytes()
    assert m1 == m2, "Determinismus verletzt (§G5 (copilot-instructions.md))"


def test_fail_closed_ohne_clean_titel(tmp_path) -> None:
    assert eb.main(["--clean-dir", str(tmp_path / "leer"), "--out", str(tmp_path / "o")]) == 2


def test_kontrast_filter_schliesst_schwache_regionen_aus(tmp_path, monkeypatch) -> None:
    """Regionen ohne defektlast-Kontrast duerfen nicht ins Manifest (Fahrplan B-0(1))."""
    from scipy.io import wavfile

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    # Gleichanteil statt Rauschen: Bandlimit erzeugt hier keinen Metrik-Kontrast.
    x = np.ones(eb.WINDOW * 4, dtype=np.float32) * 0.5
    wavfile.write(clean_dir / "dc.wav", 48000, (x * 32767).astype(np.int16))
    rc = eb.main(["--clean-dir", str(clean_dir), "--out", str(tmp_path / "p"), "--fenster-pro-titel", "2"])
    # Entweder alle verworfen (Exit 2) oder 0 Zeilen — beides beweist den Filter.
    manifest = tmp_path / "p" / "manifest.jsonl"
    if rc == 0:
        assert manifest.read_text(encoding="utf-8").strip() == ""
    else:
        assert rc == 2

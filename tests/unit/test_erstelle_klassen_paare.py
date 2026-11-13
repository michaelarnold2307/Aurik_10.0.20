#!/usr/bin/env python3
"""Vertragstests fuer scripts/erstelle_klassen_paare.py (modell-frei).

Trainingsstrategie Stufe 3 (Report
`2026-11-13_hoerzentrierte_trainingsstrategie.md`): deterministische
Per-Klassen-Trainingspaare aus den Schadens-Rezepten. Gepinnt werden
Determinismus (§G5 (copilot-instructions.md)), PCM-Hash-Konvention (D-K3-88)
und die Manifest-Form. Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.erstelle_klassen_paare as ek


def _schreibe_wav(pfad: Path, audio: np.ndarray, sr: int = 48000) -> None:
    from scipy.io import wavfile

    pfad.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(pfad, sr, (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


def test_sha256_pcm_deterministisch() -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal(1024).astype(np.float32)
    assert ek._sha256_pcm(a) == ek._sha256_pcm(a.copy())
    assert len(ek._sha256_pcm(a)) == 16


def test_regions_mergen_verschmilzt_und_filtert() -> None:
    regionen = [
        {"start": 0, "end": 100, "typ": "clicks", "staerke": 0.5},
        {"start": 120, "end": 200, "typ": "clicks", "staerke": 0.5},
        {"start": 1000, "end": 1010, "typ": "hum", "staerke": 0.5},
    ]
    gemergt = ek._regions_mergen(regionen, merge_gap=50, min_dauer=150)
    assert gemergt == [{"start": 0, "end": 200, "typ": "clicks", "staerke": 0.5}]
    assert ek._regions_mergen(regionen, merge_gap=50, min_dauer=0)[1]["end"] == 1010


def test_wav_lesen_normalisiert_int16() -> None:
    sr = 48000
    t = np.arange(sr // 2) / sr
    audio = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    pfad = Path("/tmp/test_klassen_int16.wav")
    _schreibe_wav(pfad, audio, sr)
    sr_ist, geladen = ek._wav_lesen(pfad)
    assert sr_ist == sr
    assert geladen.dtype == np.float32
    assert float(np.abs(geladen).max()) <= 1.0
    assert float(np.abs(geladen).max()) > 0.1


def test_main_deterministisch_und_manifest(tmp_path) -> None:
    """Zwei Läufe auf denselben synthetischen Eingaben ⇒ identisches Manifest (§G5 (copilot-instructions.md))."""
    schaden_dir = tmp_path / "schaden"
    clean_dir = tmp_path / "clean"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    sr = 48000
    t = np.arange(sr) / sr
    # Rausch-haftes Material: der Scanner findet Regionen (oder nicht) — der
    # Vertrag ist, dass BEIDE Läufe dasselbe Manifest liefern.
    rng = np.random.default_rng(42)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * rng.standard_normal(sr)).astype(np.float32)
    _schreibe_wav(schaden_dir / "Test (clipping).wav", audio, sr)
    _schreibe_wav(schaden_dir / "Test (clipping)_restauriert.wav", audio * 0.99, sr)
    _schreibe_wav(clean_dir / "Test original.wav", audio, sr)

    assert ek.main(["--schaden-dir", str(schaden_dir), "--clean-dir", str(clean_dir), "--out", str(out1)]) == 0
    assert ek.main(["--schaden-dir", str(schaden_dir), "--clean-dir", str(clean_dir), "--out", str(out2)]) == 0
    m1 = (out1 / "manifest.jsonl").read_text(encoding="utf-8").replace(str(out1.resolve()), "OUT")
    m2 = (out2 / "manifest.jsonl").read_text(encoding="utf-8").replace(str(out2.resolve()), "OUT")
    assert m1 == m2, "Zwei Läufe müssen dasselbe Manifest liefern (Determinismus, §G5 (copilot-instructions.md))"
    if m1.strip():
        erste = __import__("json").loads(m1.splitlines()[0])
        assert "klasse" in erste and "start" in erste and "end" in erste
        assert erste["sha256_pcm_schaden"]

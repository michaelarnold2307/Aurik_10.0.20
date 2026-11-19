"""Unit-Tests für scripts/build_mushra_stimuli.py (P1-4-Vorbereitung).

Geprüft: Quellen-Gate (50-Byte-/Kurzdateien → leer), deterministischer Cut,
Tiefpass, LUFS-Abgleich (±0,5 LUFS, Peak-Deckel), Low-Anchor (−6 LUFS),
End-to-End-Build mit synthetischen Szenarien (Status „aurik_leer“,
Manifest-SHA, Reproduzierbarkeit — §G5 (GEBOTE.md)).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location("mushra_builder", "scripts/build_mushra_stimuli.py")
mushra = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(mushra)


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), x.astype(np.float32), sr)


def _tone(sr: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_load_mono_48k_rejects_tiny_files(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.wav"
    _write_wav(tiny, np.zeros(2, dtype=np.float32), 48000)  # 50-Byte-Export-Äquivalent
    assert mushra._load_mono_48k(tiny) is None
    assert mushra._load_mono_48k(tmp_path / "fehlt.wav") is None


def test_load_mono_48k_resamples(tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    _write_wav(wav, _tone(44100, 2.0), 44100)
    out = mushra._load_mono_48k(wav)
    assert out is not None
    assert abs(len(out) - 2.0 * 48000) < 3


def test_cut_energetic_deterministic_and_bounded() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(30 * 48000).astype(np.float32) * 0.05
    a = mushra._cut_energetic(x, 10.0)
    b = mushra._cut_energetic(x, 10.0)
    assert len(a) == 10 * 48000
    assert np.array_equal(a, b)
    short = _tone(48000, 1.0)
    assert len(mushra._cut_energetic(short, 10.0)) == len(short)


def test_lowpass_reduces_hf_energy() -> None:
    rng = np.random.RandomState(1)
    x = rng.randn(48000).astype(np.float32) * 0.05  # weißes Rauschen
    y = mushra.lowpass_butter(x, 3500.0)
    hf_x = float(np.mean(np.abs(np.diff(x))))
    hf_y = float(np.mean(np.abs(np.diff(y))))
    assert hf_y < hf_x * 0.5


def test_gain_to_lufs_within_tolerance() -> None:
    rng = np.random.RandomState(2)
    x = rng.randn(4 * 48000).astype(np.float32) * 0.02
    out, achieved = mushra._gain_to_lufs(x, -23.0)
    assert abs(achieved - (-23.0)) < 0.5
    assert float(np.max(np.abs(out))) <= 0.99


def test_build_study_end_to_end(tmp_path: Path) -> None:
    """2 synthetische Szenarien: eins ok, eins mit leerem Aurik — deterministisch."""
    import soundfile as sf

    in_a = tmp_path / "in_a.wav"
    in_b = tmp_path / "in_b.wav"
    aur_a = tmp_path / "aur_a.wav"
    aur_b = tmp_path / "aur_b.wav"  # leer (50-Byte-Äquivalent)
    rng = np.random.RandomState(42)
    _write_wav(in_a, (rng.randn(4 * 48000) * 0.02).astype(np.float32), 48000)
    _write_wav(in_b, (rng.randn(4 * 48000) * 0.02).astype(np.float32), 48000)
    _write_wav(aur_a, (rng.randn(4 * 48000) * 0.01).astype(np.float32), 48000)
    _write_wav(aur_b, np.zeros(2, dtype=np.float32), 48000)

    scenarios = [("s_ok", str(in_a), str(aur_a)), ("s_leer", str(in_b), str(aur_b))]
    out = tmp_path / "study"
    m1 = mushra.build_study(out, seed=42, duration_s=2.0, scenarios=scenarios)
    m2 = mushra.build_study(out, seed=42, duration_s=2.0, scenarios=scenarios)

    assert m1["scenarios"][0]["status"] == "ok"
    assert m1["scenarios"][1]["status"] == "aurik_leer"
    assert len(m1["scenarios"][0]["stimuli"]) == 3
    assert len(m1["scenarios"][1]["stimuli"]) == 2

    # LUFS-Abgleich: reference definiert das Ziel; aurik liegt im Toleranzband.
    ref_lufs = m1["scenarios"][0]["stimuli"][0]["integrated_lufs"]
    aur_lufs = m1["scenarios"][0]["stimuli"][2]["integrated_lufs"]
    anchor_lufs = m1["scenarios"][0]["stimuli"][1]["integrated_lufs"]
    assert abs(aur_lufs - ref_lufs) < 0.6
    assert abs(anchor_lufs - (ref_lufs - 6.0)) < 0.8

    # Determinismus: identische SHA-Einträge über zwei Läufe (§G5 (GEBOTE.md)).
    h1 = [s["sha256"] for sc in m1["scenarios"] for s in sc["stimuli"]]
    h2 = [s["sha256"] for sc in m2["scenarios"] for s in sc["stimuli"]]
    assert h1 == h2

    # Manifest liegt als JSON vor und Stimuli-Dateien existieren.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for sc in manifest["scenarios"]:
        for stim in sc["stimuli"]:
            p = out / stim["path"]
            assert p.is_file()
            assert sf.info(str(p)).samplerate == 48000

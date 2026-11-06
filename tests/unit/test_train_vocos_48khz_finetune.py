"""Vertragstests D-K2-3 (scripts/train_vocos_48khz_finetune.py, SOTA-M6).

Ohne Modell-Laden: Daten-/CLI-Pfade testbar, Modell-Teile nur fail-closed
(§G5 (GEBOTE.md): keine umgebungsabhaengigen Kipp-Tests). Zitierdisziplin
AGENTS.md §5.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.io import wavfile

import scripts.train_vocos_48khz_finetune as tvf

_SRC = pathlib.Path(tvf.__file__).read_text(encoding="utf-8")


def _musdb(tmp_path: pathlib.Path, n_tracks: int = 3) -> pathlib.Path:
    root = tmp_path / "musdb18hq"
    for i in range(n_tracks):
        d = root / "train" / f"track{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(i)
        t = np.arange(48000 * 6) / 48000.0
        x = 0.2 * np.sin(2 * np.pi * (80.0 + 30.0 * i) * t) + 0.01 * rng.standard_normal(t.size)
        wavfile.write(d / "vocals.wav", 48000, (x * 32767).astype(np.int16))
    return root


def test_cli_vertrag_r5() -> None:
    """Kette-4-Vertrag R5: --musdb-root/--epochs/--device/--out-dir exakt bedient."""
    args = tvf.parse_args(
        [
            "--musdb-root",
            "data/musdb18hq",
            "--epochs",
            "30",
            "--device",
            "cuda",
            "--out-dir",
            "models/vocos_48khz/finetune",
        ]
    )
    assert args.epochs == 30
    assert args.device == "cuda"
    assert args.seed == 42
    assert args.lr == 2e-5
    assert args.batch == 8


def test_stable_track_list_determinismus(tmp_path: pathlib.Path) -> None:
    root = _musdb(tmp_path)
    a = tvf._stable_track_list(root, 42)
    b = tvf._stable_track_list(root, 42)
    assert [str(p) for p in a] == [str(p) for p in b]
    assert len(a) == 3
    assert all((p / "vocals.wav").is_file() for p in a)


def test_stable_track_list_fail_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        tvf._stable_track_list(tmp_path / "fehlt", 42)


def test_windows_layout_und_determinismus() -> None:
    x_stereo = np.stack([np.ones(96000), np.zeros(96000)], axis=0)  # (C, N)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    w1 = tvf._windows(x_stereo, rng1, 2)
    w2 = tvf._windows(x_stereo, rng2, 2)
    assert len(w1) == len(w2) == 2
    for a, b in zip(w1, w2):
        assert np.array_equal(a, b)
        assert a.shape == (192000,)
        assert np.all(np.isfinite(a))
    # Layout-Invariante: (N, C) liefert denselben Mono-Mix wie (C, N)
    x_nc = x_stereo.T
    rng3 = np.random.default_rng(42)
    w3 = tvf._windows(x_nc, rng3, 2)
    for a, b in zip(w1, w3):
        assert np.array_equal(a, b)


def test_load_vocos_fail_closed_missing_checkpoint(monkeypatch, tmp_path: pathlib.Path) -> None:
    import sys
    import types

    fake = types.ModuleType("vocos")
    fake.pretrained = types.ModuleType("vocos.pretrained")
    fake.pretrained.Vocos = type("Vocos", (), {})  # nur fuer den Import-Pfad noetig
    monkeypatch.setitem(sys.modules, "vocos", fake)
    monkeypatch.setitem(sys.modules, "vocos.pretrained", fake.pretrained)
    monkeypatch.setattr(tvf, "_ROOT", tmp_path)
    (tmp_path / "models" / "vocos_48khz").mkdir(parents=True)
    with pytest.raises(SystemExit, match="unvollstaendig"):
        tvf._load_vocos("cpu")


def test_load_vocos_fail_closed_missing_package(monkeypatch, tmp_path: pathlib.Path) -> None:
    (tmp_path / "models" / "vocos_48khz").mkdir(parents=True)
    monkeypatch.setattr(tvf, "_ROOT", tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "vocos", None)
    with pytest.raises(SystemExit, match="vocos-Paket fehlt"):
        tvf._load_vocos("cpu")


def test_keine_flag_flips() -> None:
    assert "flag_freigabe" not in _SRC
    assert ".flags" not in _SRC

"""Vertragstests F9 (scripts/train_voiceencoder_finetune.py, D-K4-5).

Ohne Modell-Laden; Torch wird nur fuer den NT-Xent-Test benoetigt und dort via
importorskip abgesichert (§G5 (GEBOTE.md): Tests duerfen nicht
umgebungsabhaengig kippen). Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.io import wavfile

import scripts.train_voiceencoder_finetune as tve

_SRC = pathlib.Path(tve.__file__).read_text(encoding="utf-8")


def _track(tmp_path: pathlib.Path, name: str, dur_s: float = 12.0, seed: int = 0) -> pathlib.Path:
    track = tmp_path / name
    (track / "vocals.wav").parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * 16000)) / 16000.0
    x = 0.3 * np.sin(2 * np.pi * (110.0 + seed * 17.0) * t) + 0.02 * rng.standard_normal(t.size)
    wavfile.write(track / "vocals.wav", 16000, (x * 32767).astype(np.int16))
    return track


def test_cli_vertrag_r3() -> None:
    """Kette-4-Vertrag R3: --musdb-root/--epochs/--device/--out-dir exakt bedient."""
    args = tve.build_parser().parse_args(
        [
            "--musdb-root",
            "data/musdb18hq",
            "--epochs",
            "30",
            "--device",
            "cuda",
            "--out-dir",
            "models/resemblyzer/finetune",
        ]
    )
    assert args.epochs == 30
    assert args.device == "cuda"
    assert args.seed == 42
    assert args.lr == 1e-5
    assert args.batch == 16


def test_mono_16k_layout_und_normalisierung() -> None:
    stereo_cn = (np.random.default_rng(1).standard_normal((2, 32000)) * 1000).astype(np.int16)
    y = tve._mono_16k(stereo_cn, 32000)
    assert y.ndim == 1
    assert y.size == 16000
    assert np.max(np.abs(y)) <= 1.0
    stereo_nc = np.ascontiguousarray(stereo_cn.T)
    y2 = tve._mono_16k(stereo_nc, 32000)
    assert np.allclose(y, y2, atol=1e-5)


def test_augment_determinismus_und_finit() -> None:
    x = np.random.default_rng(3).standard_normal(16000).astype(np.float32)
    for kind in tve._AUG_KINDS:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = tve._augment(x, kind, rng1, 16000)
        b = tve._augment(x, kind, rng2, 16000)
        assert np.array_equal(a, b), kind
        assert np.all(np.isfinite(a)), kind
        if kind != "clean":
            rng3 = np.random.default_rng(43)
            assert not np.array_equal(a, tve._augment(x, kind, rng3, 16000)), kind


def test_build_batches_positiv_paare(tmp_path: pathlib.Path) -> None:
    musdb = tmp_path / "musdb"
    tracks = [_track(musdb / "train", f"t{i}", seed=i) for i in range(4)]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    b1 = tve._build_batches(tracks, rng1, batch=3)
    b2 = tve._build_batches(tracks, rng2, batch=3)
    assert b1 and len(b1) == len(b2)
    for ba, bb in zip(b1, b2):
        assert len(ba["views"]) == len(bb["views"])
        assert ba["labels"] == bb["labels"]
        assert len(ba["views"]) == 2 * len(set(ba["labels"])), "je Track 2 Sichten (Anker + Positiv-Partner)"
        assert all(v.shape == ba["views"][0].shape for v in ba["views"])


def test_iter_tracks_fail_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        tve._iter_tracks(tmp_path / "leer")


def test_nt_xent_ausrichtung() -> None:
    torch = pytest.importorskip("torch")
    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
    labels_gut = np.array([0, 0, 1, 1])
    labels_schlecht = np.array([0, 1, 0, 1])
    loss_gut = tve._nt_xent_torch(z, labels_gut, 0.1)
    loss_schlecht = tve._nt_xent_torch(z, labels_schlecht, 0.1)
    assert float(loss_gut) < float(loss_schlecht)
    assert torch.isfinite(loss_gut)


def test_keine_flag_flips() -> None:
    assert "flag_freigabe" not in _SRC
    assert ".flags" not in _SRC

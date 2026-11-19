"""Unit-Tests für scripts/train_gacela_vocal_inpaint.py (F2-Vorbereitung, Datenpfad).

Der Upstream-Trainingspfad (tifresi/ltfatpy) ist der dokumentierte Blocker;
getestet wird der MUSDB→22,05-kHz-WAV-Datenpfad, der davon unabhängig ist.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

_SPEC = importlib.util.spec_from_file_location("gacela_prep", "scripts/train_gacela_vocal_inpaint.py")
gacela = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gacela)


def _fake_track(tmp_path: Path, sr: int, seconds: float) -> Path:
    track = tmp_path / "Fake - Track"
    track.mkdir()
    rng = np.random.RandomState(0)
    x = (0.3 * rng.randn(int(sr * seconds))).astype(np.float32)
    if sr == 44100:
        wavfile.write(str(track / "vocals.wav"), sr, (x * 32767).astype(np.int16))
    else:
        wavfile.write(str(track / "vocals.wav"), sr, x)
    return track


def test_write_vocals_wavs_44k1(tmp_path: Path) -> None:
    import soundfile as sf

    track = _fake_track(tmp_path, 44100, 2.0)
    out = tmp_path / "wavs"
    out.mkdir()
    n = gacela._write_vocals_wavs([track], out)
    assert n == 1
    files = list(out.glob("*.wav"))
    assert len(files) == 1
    data, sr = sf.read(str(files[0]))
    assert sr == 22050
    assert data.ndim == 1
    assert np.all(np.isfinite(data))
    assert 0.0 < float(np.max(np.abs(data))) <= 0.95


def test_write_vocals_wavs_stereo_downmix(tmp_path: Path) -> None:
    import soundfile as sf

    track = tmp_path / "Fake - Stereo"
    track.mkdir()
    rng = np.random.RandomState(1)
    x = rng.randn(22050 * 2, 2).astype(np.float32) * 0.2
    wavfile.write(str(track / "vocals.wav"), 22050, x)
    out = tmp_path / "wavs2"
    out.mkdir()
    n = gacela._write_vocals_wavs([track], out)
    assert n == 1
    data, sr = sf.read(str(list(out.glob("*.wav"))[0]))
    assert sr == 22050 and data.ndim == 1
    assert np.all(np.isfinite(data))

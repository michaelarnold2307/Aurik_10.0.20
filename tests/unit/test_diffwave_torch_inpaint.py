"""Unit-Tests für backend/core/dsp/diffwave_torch_inpaint.py (§SOTA-VOCAL-INPAINT-S3).

Abgedeckt: Aktivierungsvertrag (diffwave_vocal_ready), input-abgeleiteter Seed
(Determinismus §G5 (GEBOTE.md)), §V6 (copilot-instructions.md)-Fallback ohne
Modell, DDIM-Loop mit Fake-Modell (Shape/Länge/Endlichkeit), Resampling
48 kHz ↔ 22,05 kHz, strict load des echten Checkpoints (0/0).
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.diffwave_torch_inpaint as dti


@pytest.fixture
def no_ckpt(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(dti, "_CKPT_FINETUNED", tmp_path / "missing.ckpt")
    monkeypatch.setattr(dti, "_CKPT_BASE", tmp_path / "missing_base.ckpt")
    dti.reset_ready_cache()
    dti._unload()
    yield
    dti.reset_ready_cache()
    dti._unload()


def _tone(sr: int, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_vocal_ready_false_without_checkpoint(no_ckpt) -> None:
    assert dti.diffwave_vocal_ready() is False


def test_seed_for_is_input_derived_and_deterministic() -> None:
    x = _tone(22050)
    s1 = dti._seed_for(x, 1000, 2000)
    s2 = dti._seed_for(x, 1000, 2000)
    s3 = dti._seed_for(x, 1000, 2001)
    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**31


def test_inpaint_gap_none_without_model(no_ckpt) -> None:
    x = _tone(22050)
    assert dti.diffwave_inpaint_gap(x, 5000, 7000, 22050) is None


class _FakeModel:
    """forward() liefert −x: stabiler, deterministischer DDIM-Verlauf."""

    def __call__(self, x, mel, step):
        return -x


def test_inpaint_gap_shape_finite_deterministic(no_ckpt, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(dti, "_load_model", lambda: fake)
    monkeypatch.setattr(dti, "_device", lambda: "cpu")
    x = _tone(22050, seconds=2.0)
    gap_start, gap_end = 22000, 22000 + 6615  # 300 ms
    f1 = dti.diffwave_inpaint_gap(x, gap_start, gap_end, 22050)
    f2 = dti.diffwave_inpaint_gap(x, gap_start, gap_end, 22050)
    assert f1 is not None and f2 is not None
    assert f1.shape == (gap_end - gap_start,)
    assert np.all(np.isfinite(f1))
    assert np.array_equal(f1, f2)  # input-abgeleiteter Seed ⇒ bit-identisch


def test_inpaint_gap_resamples_48k(no_ckpt, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(dti, "_load_model", lambda: fake)
    monkeypatch.setattr(dti, "_device", lambda: "cpu")
    x = _tone(48000, seconds=2.0)
    gap_start, gap_end = 48000, 48000 + 14400  # 300 ms @ 48 kHz
    f = dti.diffwave_inpaint_gap(x, gap_start, gap_end, 48000)
    assert f is not None
    assert f.shape == (14400,)
    assert np.all(np.isfinite(f))


@pytest.mark.skipif(not dti._CKPT_BASE.is_file(), reason="models/diffwave/diffwave.ckpt fehlt")
def test_real_checkpoint_strict_load() -> None:
    import torch

    from backend.core.dsp.diffwave_model import DiffWave

    state = torch.load(str(dti._CKPT_BASE), map_location="cpu", weights_only=True)
    model = DiffWave()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected

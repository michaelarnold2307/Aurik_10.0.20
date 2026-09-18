"""§SOTA-ML-V8 — MuQ-MuLan-Torch-ROCm-Kern: Unit-Tests (CPU, Fake-Core).

Der echte Audio-Turm (2,65-GB-Checkpoint-Load) bleibt im GPU-Smoke/
Produktionslauf; hier wird mit einem Fake-Core (Identity-Turm) geprüft:
Form/NaN-Guard, Fail-closed (§V6 (copilot-instructions.md)), Wrong-Shape.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.core.dsp import muq_mulan_torch_rocm as mmt


def _fake_core():
    """Identity-Turm: Embedding = Zeilenmittel der Eingabe, 768-d gestreckt."""

    class _IdentityTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            mean = x.mean(dim=1, keepdim=True)  # [B, 1]
            return mean.repeat(1, 768)

    return _IdentityTower().eval()


def test_embed_shape_and_nan_guard():
    core = _fake_core()
    rng = np.random.default_rng(3)
    w = rng.standard_normal((1, 240000)).astype(np.float32)
    out = mmt.embed_muq_mulan_torch(core, w)
    assert out.shape == (1, 768)
    assert out.dtype == np.float32
    w_bad = w.copy()
    w_bad[0, 0] = np.nan
    out_bad = mmt.embed_muq_mulan_torch(core, w_bad)
    assert np.isfinite(out_bad).all()


def test_embed_accepts_1d_and_rejects_wrong_shape():
    core = _fake_core()
    out = mmt.embed_muq_mulan_torch(core, np.zeros(240000, dtype=np.float32))
    assert out.shape == (1, 768)
    with pytest.raises(ValueError):
        mmt.embed_muq_mulan_torch(core, np.zeros((1, 48000), dtype=np.float32))


def test_fail_closed_without_cuda(monkeypatch):
    """Ohne CUDA/ROCm muss get_muq_mulan_torch_core() None liefern (ONNX-Pfad bleibt)."""
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mmt, "_core", None)
    monkeypatch.setattr(mmt, "_core_resolved", False)
    assert mmt.get_muq_mulan_torch_core() is None

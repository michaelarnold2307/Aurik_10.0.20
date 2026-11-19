"""§SOTA-ML-V7 — FCPE-Torch-ROCm-Kern: Unit-Tests (CPU, Fake-Core).

Der echte Conformer-Kern bleibt im GPU-Smoke/Produktionslauf; hier wird mit
einem Fake-Core (Identity-Salience) geprüft: Form/NaN-Guard, Wrong-Shape,
Fail-closed (§V6 (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.core.dsp import fcpe_torch_rocm as ftr


def _fake_core():
    """Identity-Salience: [B,T,128] → [B,T,360] (deterministisch, billig)."""

    class _IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, mel):
            return mel.mean(dim=2, keepdim=True).repeat(1, 1, 360)

    return _IdentityModel().eval()


def test_salience_shape_and_nan_guard():
    core = _fake_core()
    rng = np.random.default_rng(3)
    mel = rng.standard_normal((1, 600, 128)).astype(np.float32)
    out = ftr.salience_fcpe_torch(core, mel)
    assert out.shape == (1, 600, 360)
    assert out.dtype == np.float32
    mel_bad = mel.copy()
    mel_bad[0, 0, 0] = np.nan
    out_bad = ftr.salience_fcpe_torch(core, mel_bad)
    assert np.isfinite(out_bad).all()


def test_salience_rejects_wrong_shape():
    core = _fake_core()
    with pytest.raises(ValueError):
        ftr.salience_fcpe_torch(core, np.zeros((1, 600, 64), dtype=np.float32))


def test_fail_closed_without_cuda(monkeypatch):
    """Ohne CUDA/ROCm muss get_fcpe_torch_core() None liefern (ONNX-CPU-Pfad bleibt)."""
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ftr, "_core", None)
    monkeypatch.setattr(ftr, "_core_resolved", False)
    assert ftr.get_fcpe_torch_core() is None

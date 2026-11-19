"""§SOTA-BSR-GPU — BS-RoFormer-317-Torch-Separation: Unit-Tests (CPU, Fake-Core).

Der ROCm-Core wird hier durch eine Identity-Maske ersetzt (nn.Module mit
Parameter — realer Device-Pfad bleibt im GPU-Smoke abgedeckt).
Determinismus (§G5 (copilot-instructions.md)) und Fail-closed
(§V6 (copilot-instructions.md)) sind direkt prüfbar.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.core.dsp.bsr317_torch_rocm import get_bsr317_torch_core, separate_bsr317_torch

SR = 48000


class _IdentityCore(torch.nn.Module):
    """Mask = 1 überall → vocal ≈ Input (STFT-Roundtrip), instrumental ≈ 0."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3], 2), device=x.device)
        m[..., 0] = 1.0  # reale Maske = 1, imaginäre = 0
        return m


def _make_audio(seconds: float = 5.0, seed: int = 3):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def test_identity_mask_vocal_equals_input():
    core = _IdentityCore()
    x = _make_audio(5.0)
    stems = separate_bsr317_torch(x, SR, core, stems=("vocals", "instruments"))
    assert set(stems) == {"vocals", "instruments"}
    # Identity-Maske: vocal ≈ Input — scipy-'even'-Paarung ist exakt invertierbar
    assert np.max(np.abs(stems["vocals"] - x)) < 1e-2
    assert np.max(np.abs(stems["instruments"])) < 1e-2


def test_stems_filter():
    core = _IdentityCore()
    stems = separate_bsr317_torch(_make_audio(3.0), SR, core, stems=("vocals",))
    assert set(stems) == {"vocals"}


def test_determinism():
    core = _IdentityCore()
    x = _make_audio(4.0)
    s1 = separate_bsr317_torch(x, SR, core)
    s2 = separate_bsr317_torch(x, SR, core)
    assert np.array_equal(s1["vocals"], s2["vocals"])
    assert np.array_equal(s1["instruments"], s2["instruments"])


def test_chunking_over_30s():
    core = _IdentityCore()
    x = _make_audio(31.0)
    stems = separate_bsr317_torch(x, SR, core, stems=("vocals",))
    assert stems["vocals"].shape == x.shape
    assert np.max(np.abs(stems["vocals"] - x)) < 1e-2


def test_stereo_nx2_layout():
    core = _IdentityCore()
    x = np.stack([_make_audio(3.0, 1), _make_audio(3.0, 2)], axis=1)  # (N, 2)
    stems = separate_bsr317_torch(x, SR, core, stems=("vocals",))
    assert stems["vocals"].shape == x.shape
    assert np.max(np.abs(stems["vocals"] - x)) < 1e-2


def test_fail_closed_without_cuda(monkeypatch):
    """Ohne CUDA/ROCm muss get_bsr317_torch_core() None liefern (ONNX-Pfad bleibt)."""
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_bsr317_torch_core() is None

"""§SOTA-ML-V5 — BANQUET-Torch-ROCm-Kern: Unit-Tests (CPU, Fake-Core).

Der ROCm-Kern wird hier entweder als CPU-Rekonstruktion (_build_core) oder
als Fake-Core (Identity-Modul) getestet; der reale Device-Pfad bleibt im
GPU-Benchmark/Produktionslauf abgedeckt. Geprüft: Parität vs. ONNX
(max|Δ| ≤ 1e-4), Batch-Form, NaN-Guard, Fail-closed
(§V6 (copilot-instructions.md)), Plugin-OLA-Verdrahtung inkl.
Quarantäne-Fallback und Determinismus (§G5 (copilot-instructions.md)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.core.dsp import banquet_torch_rocm as btr


def _fake_core():
    """Identity-Modul mit Parameter (Device-Präsenz wie echtes Modul)."""

    class _IdentityCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

    return _IdentityCore().eval()


def test_restore_shape_and_batch():
    core = _fake_core()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 128, 128, 128)).astype(np.float32)
    out = btr.restore_banquet_torch(core, x)
    assert out.shape == (2, 128, 128, 128)
    assert out.dtype == np.float32
    assert np.allclose(out, x, atol=1e-6)


def test_restore_rejects_wrong_shape():
    core = _fake_core()
    with pytest.raises(ValueError):
        btr.restore_banquet_torch(core, np.zeros((128, 128, 128), dtype=np.float32))


def test_restore_nan_guard():
    core = _fake_core()
    x = np.zeros((1, 128, 128, 128), dtype=np.float32)
    x[0, 0, 0, 0] = np.nan
    out = btr.restore_banquet_torch(core, x)
    assert np.isfinite(out).all()
    assert out[0, 0, 0, 0] == 0.0


def test_fail_closed_without_cuda(monkeypatch):
    """Ohne CUDA/ROCm muss get_banquet_torch_core() None liefern (ONNX-Pfad bleibt)."""
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(btr, "_core", None)
    monkeypatch.setattr(btr, "_core_resolved", False)
    assert btr.get_banquet_torch_core() is None


def test_cpu_parity_vs_onnx():
    """CPU-Rekonstruktion == ONNX-Referenz (max|Δ| ≤ 1e-4). Dauer ~7 s."""
    onnxruntime = pytest.importorskip("onnxruntime")
    core = btr._build_core()
    model = Path(btr.__file__).resolve().parents[3] / "models" / "banquet" / "banquet_vinyl_final.onnx"
    sess = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 128, 128, 128)).astype(np.float32)
    ref = sess.run(None, {"input_fixed": x})[0]
    out = btr.restore_banquet_torch(core, x)
    assert np.abs(ref - out).max() <= 1e-4


def test_plugin_torch_ola_and_determinism(monkeypatch):
    """Plugin-_process_torch_rocm: OLA-Verdrahtung + Determinismus (§G5 (copilot-instructions.md))."""
    from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

    fake = _fake_core()
    monkeypatch.setattr(btr, "restore_banquet_torch", lambda core, feat: np.asarray(feat, dtype=np.float32))
    plugin = object.__new__(BanquetVinylPlugin)
    plugin._torch_core = fake
    plugin._torch_quarantined = False

    def _prepare(chunk, channels):
        del chunk, channels
        return (
            np.zeros((1, 128, 128, 128), dtype=np.float32),
            np.zeros((128, 128), dtype=np.complex64),
        )

    plugin._prepare_input = _prepare

    def _extract(raw, channels, chunk_len, ctx):
        del raw, ctx
        return np.full((channels, chunk_len), 0.5, dtype=np.float32)

    plugin._extract_output = _extract
    plugin._infer_window = lambda chunk, channels, chunk_len, strength: np.zeros(
        (channels, chunk_len), dtype=np.float32
    )
    sr = 48000
    audio = (np.random.default_rng(5).standard_normal((1, sr * 2)) * 0.01).astype(np.float32)
    window = np.hanning(sr).astype(np.float32)
    a = plugin._process_torch_rocm(audio, 1, sr, sr // 2, window, 1.0)
    b = plugin._process_torch_rocm(audio, 1, sr, sr // 2, window, 1.0)
    assert a.shape == audio.shape
    assert np.isfinite(a).all()
    # Hann-COLA bei 50 % Überlapp: Innenbereich voll gewichtet → 0.5 überall.
    assert np.allclose(a[0, sr // 2 : sr], 0.5, atol=1e-6)
    assert np.array_equal(a, b)


def test_plugin_torch_fallback_on_kernel_error(monkeypatch):
    """Kern-Exception → Quarantäne + ONNX-Einzelfenster-Rest (§V6 (copilot-instructions.md))."""
    from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

    fake = _fake_core()

    def _boom(core, feat):
        raise RuntimeError("simulierter Kern-Fehler")

    monkeypatch.setattr(btr, "restore_banquet_torch", _boom)
    plugin = object.__new__(BanquetVinylPlugin)
    plugin._torch_core = fake
    plugin._torch_quarantined = False

    def _prepare(chunk, channels):
        del chunk, channels
        return (
            np.zeros((1, 128, 128, 128), dtype=np.float32),
            np.zeros((128, 128), dtype=np.complex64),
        )

    plugin._prepare_input = _prepare
    calls = []

    def _infer(chunk, channels, chunk_len, strength):
        del chunk, strength
        calls.append(chunk_len)
        return np.zeros((channels, chunk_len), dtype=np.float32)

    plugin._infer_window = _infer
    sr = 48000
    audio = np.zeros((1, sr * 2), dtype=np.float32)
    window = np.hanning(sr).astype(np.float32)
    out = plugin._process_torch_rocm(audio, 1, sr, sr // 2, window, 1.0)
    assert plugin._torch_quarantined is True
    assert plugin._torch_core is None
    assert len(calls) > 0  # ONNX-Pfad übernahm
    assert out.shape == audio.shape

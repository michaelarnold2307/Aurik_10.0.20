"""Norm-Suite für Symphonia, das Instrumentalgegenstück zu Cantus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _feeds(samples: int = 4096, frames: int = 8) -> dict[str, torch.Tensor]:
    return {
        "x": torch.zeros(1, samples, 1),
        "t": torch.tensor([0.5]),
        "mert": torch.zeros(1, frames, 1024),
        "rhythm": torch.zeros(1, frames, 2),
        "harm": torch.zeros(1, 768),
        "use_cond": torch.zeros(1),
    }


def test_symphonia_model_shape_and_determinism():
    from models.cantus.cantus_model import CantusConfig
    from models.symphonia.symphonia_model import create_symphonia

    torch.manual_seed(20261004)
    model = create_symphonia(CantusConfig(dim=32, depth=2, heads=2, patch_size=256, high_hidden=8, high_depth=1)).eval()
    with torch.no_grad():
        first = model(**_feeds())
        second = model(**_feeds())
    assert first.shape == (1, 4096, 1)
    assert torch.equal(first, second)
    assert bool(torch.isfinite(first).all())


def test_symphonia_dsp_fallback_without_weights():
    from plugins.symphonia_plugin import SymphoniaPlugin

    plugin = SymphoniaPlugin(model_path=Path("/nonexistent/symphonia.onnx"))
    audio = np.sin(np.linspace(0, 100, 4096, dtype=np.float32)).astype(np.float32)
    result = plugin.enhance(audio, 48000, "mp3_low")
    assert result.model_used == "dsp_fallback"
    assert result.audio.shape == audio.shape
    assert bool(np.isfinite(result.audio).all())


def test_canonical_plugin_keeps_unqualified_checkpoint_in_dsp_fallback():
    from plugins.symphonia_plugin import SymphoniaPlugin

    plugin = SymphoniaPlugin()
    assert plugin._inference_backend == "none"
    assert plugin._fallback_active is True
    assert "Status ist nicht active" in plugin._fallback_reason


def test_symphonia_prefers_torch_rocm_before_onnx(monkeypatch, tmp_path):
    from plugins.symphonia_plugin import SymphoniaPlugin

    plugin = SymphoniaPlugin(model_path=tmp_path / "missing.onnx")
    plugin._model_loaded = True
    plugin._fallback_active = False
    plugin._torch_model = object()
    plugin.__dict__["_extract_conditions"] = lambda _audio: {
        "mert": np.zeros((1, 1024), dtype=np.float32),
        "rhythm": np.zeros((1, 2), dtype=np.float32),
        "harm": np.zeros(768, dtype=np.float32),
        "use_cond": np.asarray(0.0, dtype=np.float32),
    }
    monkeypatch.setattr(plugin, "_run_torch_rocm", lambda feeds: np.zeros_like(feeds["x"]))
    monkeypatch.setattr(plugin, "_run_ort_with_cpu_fallback", lambda _feeds: (_ for _ in ()).throw(AssertionError()))
    audio = np.ones(4096, dtype=np.float32)
    assert np.array_equal(plugin._restore_single(audio), audio)

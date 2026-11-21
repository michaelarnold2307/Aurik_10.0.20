"""Unit-Tests Phase 67: Crackle Texture Removal (§SR-CK-Textur).

Vertrag geprueft:
- Metadaten: DEFECT_REMOVAL, RESTORATION_ONLY
- Layout-Normalisierung: channels-first / samples-first / mono
- Passthrough-Semantik ohne Modell (§V6, copilot-instructions.md): bit-identisch, kein stiller Fallback
- Deterministisch: gleicher Input -> gleicher Output
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import backend.core.ml_device_manager as device_manager
import backend.core.phases.phase_67_crackle_texture_removal as phase_module
from backend.core.phases.phase_67_crackle_texture_removal import CrackleTextureRemovalPhase
from backend.core.phases.phase_interface import PhaseCategory, PhaseMode


def test_metadata_contract() -> None:
    p = CrackleTextureRemovalPhase()
    md = p.get_metadata()
    assert md.phase_id == "phase_67_crackle_texture_removal"
    assert md.category == PhaseCategory.DEFECT_REMOVAL
    assert md.phase_mode == PhaseMode.RESTORATION_ONLY
    assert "crackle" in md.defect_types


def test_passthrough_ohne_modell_bit_identisch(monkeypatch: pytest.MonkeyPatch) -> None:
    p = CrackleTextureRemovalPhase()
    monkeypatch.setattr(p, "_load_model", lambda: False)
    rng = np.random.default_rng(0)
    x = rng.random((2, 48000), dtype=np.float32) * 0.2 - 0.1
    res = p.process(x, sample_rate=48000)
    assert res.success
    assert np.array_equal(res.audio, x)  # bit-identisch (§V6-Passthrough, copilot-instructions.md)
    assert "modell_nicht_ladbar" in res.warnings


def test_layout_normalisierung_ohne_modell(monkeypatch: pytest.MonkeyPatch) -> None:
    p = CrackleTextureRemovalPhase()
    monkeypatch.setattr(p, "_load_model", lambda: False)
    rng = np.random.default_rng(1)
    x_cf = rng.random((2, 24000), dtype=np.float32) * 0.2 - 0.1
    res_cf = p.process(x_cf, sample_rate=48000)
    assert res_cf.audio.shape == (2, 24000)
    res_sf = p.process(x_cf.T, sample_rate=48000)
    assert res_sf.audio.shape == (24000, 2)
    res_m = p.process(x_cf[0], sample_rate=48000)
    assert res_m.audio.shape == (24000,)


def test_sample_rate_guard() -> None:
    p = CrackleTextureRemovalPhase()
    x = np.zeros((2, 1000), dtype=np.float32)
    res = p.process(x, sample_rate=44100)
    assert "sample_rate" in res.warnings
    assert np.array_equal(res.audio, x)


def test_force_cpu_prevents_crackle_rocm_initialization(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _FakeNet:
        devices: list[str] = []

        def load_state_dict(self, _state, strict: bool = True) -> None:
            assert strict

        def to(self, device: str) -> _FakeNet:
            self.devices.append(device)
            return self

        def eval(self) -> None:
            pass

    pt_path = tmp_path / "crackle.pt"
    onnx_path = tmp_path / "crackle.onnx"
    monkeypatch.setattr(phase_module, "_BASE_PT", str(pt_path))
    monkeypatch.setattr(phase_module, "_BASE_ONNX", str(onnx_path))
    monkeypatch.setattr(phase_module.os.path, "exists", lambda path: path == str(pt_path))
    monkeypatch.setattr(phase_module, "_CrackleNet", lambda _torch: _FakeNet())
    map_locations: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: pytest.fail("forced CPU must not probe CUDA")),
        load=lambda _path, map_location, weights_only: map_locations.append(map_location) or {"state": {}},
    )
    monkeypatch.setenv("AURIK_FORCE_CPU", "1")
    monkeypatch.setattr(
        device_manager.MLDeviceManager,
        "_detect_backend",
        lambda _self: pytest.fail("AURIK_FORCE_CPU must bypass backend detection"),
    )
    monkeypatch.setattr(device_manager, "_instance", None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    phase = CrackleTextureRemovalPhase()

    assert phase._load_model()
    assert phase._device == "cpu"
    assert map_locations == ["cpu"]
    assert phase._net.devices == ["cpu"]


def test_verarbeitung_mit_modell_falls_vorhanden() -> None:
    p = CrackleTextureRemovalPhase()
    if not p._load_model():
        pytest.skip("Modell nicht vorhanden (CI ohne Modelle) — Passthrough bereits abgedeckt")
    rng = np.random.default_rng(2)
    x = rng.random((2, 48000), dtype=np.float32) * 0.2 - 0.1
    res1 = p.process(x, sample_rate=48000)
    res2 = p.process(x, sample_rate=48000)
    assert res1.ml_used
    assert res1.audio.shape == (2, 48000)
    assert np.isfinite(res1.audio).all()
    assert np.array_equal(res1.audio, res2.audio)  # deterministisch (§G5, copilot-instructions.md)

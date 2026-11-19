import pytest

"""Unit tests for §MCG-1 ModelCapabilityGate."""

import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from backend.core.dsp.model_capability_gate import CapabilityReport, ModelCapabilityGate


@pytest.mark.unit
def test_model_capability_gate_recognizes_miipher_compensation_as_sota_real(monkeypatch, tmp_path):
    fake_miipher = types.SimpleNamespace(_MIIPHER_ONNX_PATH=None, _instance=None)
    fake_sgmse_model = tmp_path / "sgmse_plus.ts"
    fake_sgmse_model.write_bytes(b"stub")
    fake_sgmse = types.SimpleNamespace(
        _TS_PATH=fake_sgmse_model,
        _CKPT_CANDIDATES=(),
        _instance_plus=types.SimpleNamespace(_model_loaded=True),
    )
    fake_bs = types.SimpleNamespace(
        BSRoFormerPlugin=types.SimpleNamespace(_LOCAL_MBR=tmp_path / "missing_mbr.onnx"),
        _instance=None,
    )
    fake_dfn_dir = tmp_path / "dfn"
    fake_dfn_dir.mkdir()
    for name in ("enc.onnx", "dec.onnx", "erb_dec.onnx"):
        (fake_dfn_dir / name).write_bytes(b"stub")
    fake_dfn = types.SimpleNamespace(_DIR=str(fake_dfn_dir), _inst=None)
    fake_demucs = types.SimpleNamespace(_MODEL_PATH=str(tmp_path / "missing_demucs.onnx"), _instance=None)

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(sys_modules, "plugins.miipher_plugin", fake_miipher)
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", fake_sgmse)
    monkeypatch.setitem(sys_modules, "plugins.bs_roformer_plugin", fake_bs)
    monkeypatch.setitem(sys_modules, "plugins.deepfilternet_v3_ii_plugin", fake_dfn)
    monkeypatch.setitem(sys_modules, "plugins.demucs_v4_plugin", fake_demucs)

    report: CapabilityReport = ModelCapabilityGate().build_report()
    capabilities = report["capabilities"]
    miipher_meta = cast(Mapping[str, Any], capabilities["miipher"].get("metadata", {}))
    assert capabilities["miipher"]["status"] == "sota_real"
    assert capabilities["miipher"]["reason"] == "compensation_chain_ready"
    assert miipher_meta["compensation_chain_ready"] is True
    assert capabilities["sgmse_plus"]["status"] == "sota_fallback"
    assert capabilities["deepfilternet_v3_ii"]["status"] == "sota_fallback"
    assert "miipher" not in report["summary"]["degraded_capabilities"]


def test_model_capability_gate_marks_miipher_partial_compensation_as_fallback(monkeypatch, tmp_path):
    fake_miipher = types.SimpleNamespace(_MIIPHER_ONNX_PATH=None, _instance=None)
    fake_sgmse_model = tmp_path / "sgmse_plus.ts"
    fake_sgmse_model.write_bytes(b"stub")
    fake_sgmse = types.SimpleNamespace(
        _TS_PATH=fake_sgmse_model,
        _CKPT_CANDIDATES=(),
        _instance_plus=types.SimpleNamespace(_model_loaded=True),
    )
    fake_bs = types.SimpleNamespace(
        BSRoFormerPlugin=types.SimpleNamespace(_LOCAL_MBR=tmp_path / "missing_mbr.onnx"),
        _instance=None,
    )
    fake_dfn = types.SimpleNamespace(_DIR=str(tmp_path / "missing_dfn"), _inst=None)
    fake_demucs = types.SimpleNamespace(_MODEL_PATH=str(tmp_path / "missing_demucs.onnx"), _instance=None)

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(sys_modules, "plugins.miipher_plugin", fake_miipher)
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", fake_sgmse)
    monkeypatch.setitem(sys_modules, "plugins.bs_roformer_plugin", fake_bs)
    monkeypatch.setitem(sys_modules, "plugins.deepfilternet_v3_ii_plugin", fake_dfn)
    monkeypatch.setitem(sys_modules, "plugins.demucs_v4_plugin", fake_demucs)

    report: CapabilityReport = ModelCapabilityGate().build_report()
    capabilities = report["capabilities"]
    miipher_meta = cast(Mapping[str, Any], capabilities["miipher"].get("metadata", {}))
    assert capabilities["miipher"]["status"] == "sota_fallback"
    assert capabilities["miipher"]["reason"] == "model_not_loaded_partial_compensation"
    assert miipher_meta["compensation_chain_ready"] is False
    assert miipher_meta["compensation_sgmse_ready"] is True
    assert miipher_meta["compensation_deepfilternet_ready"] is False


def test_model_capability_gate_detects_real_melband_and_miipher(monkeypatch, tmp_path):
    melband = tmp_path / "melbandroformer_optimized.onnx"
    melband.write_bytes(b"stub")
    miipher = tmp_path / "miipher.onnx"
    miipher.write_bytes(b"stub")

    fake_bs = types.SimpleNamespace(
        BSRoFormerPlugin=types.SimpleNamespace(_LOCAL_MBR=melband),
        _instance=types.SimpleNamespace(_model_loaded=True),
    )
    fake_miipher = types.SimpleNamespace(
        _MIIPHER_ONNX_PATH=miipher,
        _instance=types.SimpleNamespace(_model_loaded=True),
    )
    fake_sgmse = types.SimpleNamespace(_TS_PATH=tmp_path / "missing.ts", _CKPT_CANDIDATES=(), _instance_plus=None)
    fake_dfn = types.SimpleNamespace(_DIR=str(tmp_path / "missing_dfn"), _inst=None)
    fake_demucs = types.SimpleNamespace(_MODEL_PATH=str(tmp_path / "missing_demucs.onnx"), _instance=None)

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(sys_modules, "plugins.bs_roformer_plugin", fake_bs)
    monkeypatch.setitem(sys_modules, "plugins.miipher_plugin", fake_miipher)
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", fake_sgmse)
    monkeypatch.setitem(sys_modules, "plugins.deepfilternet_v3_ii_plugin", fake_dfn)
    monkeypatch.setitem(sys_modules, "plugins.demucs_v4_plugin", fake_demucs)

    gate = ModelCapabilityGate()
    report: CapabilityReport = gate.build_report()
    assert report["capabilities"]["melbandroformer"]["status"] == "sota_real"
    assert report["capabilities"]["miipher"]["status"] == "sota_real"
    assert report["summary"]["vocal_nr_primary"] == "miipher"
    assert report["summary"]["separation_primary"] == "melbandroformer"
    assert gate.vocal_restoration_status() == "sota_real"


def test_capability_path_helper_handles_none():
    path, exists = ModelCapabilityGate._path_exists(None)  # pylint: disable=protected-access
    assert path == ""
    assert exists is False
    path, exists = ModelCapabilityGate._path_exists(Path("/definitely/missing/model.onnx"))  # pylint: disable=protected-access
    assert path.endswith("model.onnx")
    assert exists is False

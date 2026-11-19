from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
def test_main_aborts_export_when_quality_gate_fails(monkeypatch, tmp_path):
    import _aurik_run_excellence as runner

    audio = np.zeros((48_000, 2), dtype=np.float32)

    result = types.SimpleNamespace(
        material="mp3_low",
        rt_factor=0.1,
        quality_estimate=0.20,
        goals_passed=0,
        musical_goals={},
        phases_executed=[],
        warnings=[],
        stage_notes={},
        audio=audio,
    )

    class _Denker:
        def denke(self, *_args, **_kwargs):
            return result

    exporter_called = {"value": False}

    class _Exporter:
        def export(self, *_args, **_kwargs):
            exporter_called["value"] = True
            return Path(tmp_path / "x.wav")

    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "_load_audio", lambda _path: (audio.T, runner.TARGET_SR))
    monkeypatch.setattr(runner, "_resample_to_48k", lambda _audio, _sr: _audio)
    monkeypatch.setattr(runner, "_to_pipeline_format", lambda _audio: audio)
    monkeypatch.setattr("denker.aurik_denker.get_aurik_denker", lambda: _Denker())
    monkeypatch.setattr("backend.core.audio_exporter.AudioExporter", _Exporter)
    monkeypatch.setattr(runner, "_progress_cb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.sys, "argv", ["_aurik_run_excellence.py", str(tmp_path / "dummy.mp3")])
    _original_path_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda self: True if str(self).endswith("dummy.mp3") else _original_path_exists(self)
    )

    rc = runner.main()

    assert rc == 2
    assert exporter_called["value"] is False


def test_excellence_runner_prefers_bridge_loader_and_guards_fallback_export() -> None:
    src = Path(__file__).resolve().parents[1].joinpath("_aurik_run_excellence.py").read_text(encoding="utf-8")

    load_idx = src.index('_bridge_module: Any = _optional_import("backend.api.bridge")')
    bridge_load_idx = src.index("_load_fn = _get_load_audio_fn()")
    sf_read_idx = src.index("_sf.read(str(path)")
    guard_import_idx = src.index('_export_guard: Any = getattr(_bridge_module, "export_guard", None)')
    fallback_idx = src.index("AudioExporter fehlgeschlagen")
    write_idx = src.index("_sf.write(str(tmp_wav)", fallback_idx)
    guard_idx = src.index("_export_guard(audio_out)", write_idx)
    replace_idx = src.index("tmp_wav.replace(out_wav)", guard_idx)

    assert load_idx < bridge_load_idx < sf_read_idx
    assert guard_import_idx < fallback_idx < write_idx < guard_idx < replace_idx

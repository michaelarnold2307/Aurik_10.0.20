"""GPU-ROCm (2026-09-13): Provider- und Optimierungs-Level-Logik des BANQUET-Plugins.

Produktionsbefund: Die Session registrierte nur CPU, obwohl
ROCMExecutionProvider angefordert war — Ursache: `ORT_DISABLE_ALL`
(kein Partitioning) → ORT fällt bei EINEM nicht-ROCm-fähigen Op komplett
auf CPU zurück. Fix: ENABLE_BASIC bei GPU-Request, DISABLE_ALL nur für
CPU; zusätzlich Warnung bei stillem CPU-Fallback (§V6 (copilot-instructions.md)).
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")  # CI-Minimal-Umgebung (cross-platform)

import onnxruntime as _real_ort


def _fake_session_factory(captured: dict):
    class _In:
        name = "input_fixed"
        shape = [1, 128, 128, 128]

    class _Out:
        name = "output_fixed"

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            captured["providers"] = list(kwargs.get("providers") or [])

        def get_inputs(self):
            return [_In()]

        def get_outputs(self):
            return [_Out()]

        def get_providers(self):
            return captured.get("session_providers_override", captured["providers"])

    return _FakeSession


def _make_plugin(monkeypatch, tmp_path, providers, captured):
    import plugins.banquet_vinyl_plugin as _bp

    (tmp_path / "banquet_vinyl_final.onnx").write_bytes(b"dummy")
    monkeypatch.setattr(_bp.BanquetVinylPlugin, "_patch_onnx", staticmethod(lambda src, dst: src))
    monkeypatch.setattr("onnxruntime.InferenceSession", _fake_session_factory(captured))
    monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda *a, **k: True)
    monkeypatch.setattr("backend.core.ml_device_manager.get_ort_providers", lambda name: list(providers))
    monkeypatch.setattr("backend.core.gpu_model_registry.apply_gpu_policy", lambda p, m: list(p))
    plugin = _bp.BanquetVinylPlugin(model_dir=str(tmp_path))
    return plugin


def test_gpu_request_uses_enable_basic(monkeypatch, tmp_path):
    """GPU-Request → Partitioning aktiv (ENABLE_BASIC) + GPU-Provider durchgereicht."""
    captured: dict = {}
    gpu_providers = [("ROCMExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    plugin = _make_plugin(monkeypatch, tmp_path, gpu_providers, captured)

    assert plugin._model_ok is True
    assert captured["providers"] == gpu_providers
    lvl = captured["kwargs"]["sess_options"].graph_optimization_level
    assert lvl == _real_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC


def test_cpu_request_keeps_disable_all(monkeypatch, tmp_path):
    """CPU-Request → DISABLE_ALL (bekannter Schutz vor dem Slice-Rewrite-Crash)."""
    captured: dict = {}
    plugin = _make_plugin(monkeypatch, tmp_path, ["CPUExecutionProvider"], captured)

    assert plugin._model_ok is True
    lvl = captured["kwargs"]["sess_options"].graph_optimization_level
    assert lvl == _real_ort.GraphOptimizationLevel.ORT_DISABLE_ALL


def test_silent_cpu_fallback_logs_warning(monkeypatch, tmp_path, caplog):
    """§V6 (copilot-instructions.md): GPU angefordert, Session registriert nur CPU → sichtbare Warnung."""
    import logging

    captured: dict = {}
    captured["session_providers_override"] = ["CPUExecutionProvider"]
    with caplog.at_level(logging.WARNING):
        _make_plugin(
            monkeypatch,
            tmp_path,
            [("ROCMExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
            captured,
        )
    assert any("CPU-Fallback" in r.message for r in caplog.records)


def test_reset_for_song_clears_failures_keeps_quarantine(monkeypatch, tmp_path):
    """§V8/§G1 (copilot-instructions.md): Song-Reset isoliert nur den Zähler; die Modell-Quarantäne bleibt fail-closed."""
    captured: dict = {}
    plugin = _make_plugin(monkeypatch, tmp_path, ["CPUExecutionProvider"], captured)

    plugin._chunk_failures = 2
    plugin._runtime_quarantined = True
    plugin.reset_for_song()
    assert plugin._chunk_failures == 0
    assert plugin._runtime_quarantined is True  # Modell-Zustand, kein Song-Zustand


def test_reset_for_song_thread_smoke(monkeypatch, tmp_path):
    """Ein-Prozess-Batch: parallele Song-Resets racerieren nicht (Lock-Smoke)."""
    import threading

    captured: dict = {}
    plugin = _make_plugin(monkeypatch, tmp_path, ["CPUExecutionProvider"], captured)
    plugin._chunk_failures = 5

    errors: list[Exception] = []

    def _worker() -> None:
        try:
            for _ in range(200):
                plugin.reset_for_song()
                plugin._chunk_failures += 1
        except Exception as exc:  # pragma: no cover — nur im Fehlerfall
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert plugin._chunk_failures >= 0


def test_infer_parallel_bit_identical_to_sequential(monkeypatch):
    """INFER_PARALLEL > 1: parallele Fenster-Inferenz ist bit-identisch zum Einzelpfad (OLA-Reihenfolge fix)."""
    import numpy as _np

    import plugins.banquet_vinyl_plugin as _bp

    monkeypatch.setattr(_bp.BanquetVinylPlugin, "_try_load_model", lambda self: None)

    class _EchoSession:
        def run(self, output_names, feed_dict):
            return [_np.asarray(feed_dict["input"], dtype=_np.float32)]

    def _plugin() -> _bp.BanquetVinylPlugin:
        p = _bp.BanquetVinylPlugin()
        p._session = _EchoSession()
        p._input_name = "input"
        p._output_name = "output"
        p._model_ok = True
        p._runtime_quarantined = False
        return p

    rng = _np.random.default_rng(11)
    audio = rng.standard_normal((1, 48_000 * 3)).astype(_np.float32) * 0.05

    seq = _plugin()
    seq.INFER_PARALLEL = 0
    out_seq = seq._process_onnx(audio, 1.0)

    par = _plugin()
    par.INFER_PARALLEL = 4
    out_par = par._process_onnx(audio.copy(), 1.0)

    _np.testing.assert_array_equal(out_seq, out_par)


def test_ensure_model_loaded_reloads_once_after_eviction(monkeypatch):
    """§PERF-R (2026-09-18): Nach PLM-Eviction lädt ensure_model_loaded() das Modell
    EINMAL nach — kein stiller Dauerverbleib im DSP-Fallback (§V6 (copilot-instructions.md))."""
    import plugins.banquet_vinyl_plugin as _bp

    calls: list[int] = []

    def _fake_load(self):
        calls.append(1)
        self._session = object()
        self._model_ok = True

    monkeypatch.setattr(_bp.BanquetVinylPlugin, "_try_load_model", _fake_load)
    p = _bp.BanquetVinylPlugin()
    assert p._model_ok is True
    _init_calls = len(calls)  # __init__-Load

    # PLM-Eviction simulieren (unload_fn-Zustand):
    p._session = None
    p._model_ok = False
    p._load_attempted = False

    assert p.ensure_model_loaded() is True
    assert len(calls) == _init_calls + 1, "Reload muss genau EINMAL erfolgen"
    assert p.ensure_model_loaded() is True
    assert len(calls) == _init_calls + 1, "Kein Retry-Schleifen nach erfolgreichem Reload"


def test_ensure_model_loaded_no_retry_after_failed_load(monkeypatch):
    """§PERF-R (2026-09-18): Fehlgeschlagener Reload wird nicht je Aufruf wiederholt."""
    import plugins.banquet_vinyl_plugin as _bp

    calls: list[int] = []

    def _fake_load(self):
        calls.append(1)
        self._model_ok = False

    monkeypatch.setattr(_bp.BanquetVinylPlugin, "_try_load_model", _fake_load)
    p = _bp.BanquetVinylPlugin()
    p._model_ok = False
    p._load_attempted = False

    _init_calls = len(calls)
    assert p.ensure_model_loaded() is False
    assert p.ensure_model_loaded() is False
    assert len(calls) == _init_calls + 1, "Einmal je Eviction-Zustand versuchen, dann fail-closed"


def test_process_reloads_after_eviction_instead_of_dsp_fallback(monkeypatch):
    """§PERF-R (2026-09-18): process() stellt die Session nach Eviction wieder her
    und nutzt den ML-Pfad statt still auf DSP zu fallen."""
    import numpy as _np

    import plugins.banquet_vinyl_plugin as _bp

    class _EchoSession:
        def run(self, output_names, feed_dict):
            return [_np.asarray(feed_dict["input"], dtype=_np.float32)]

    monkeypatch.setattr(_bp.BanquetVinylPlugin, "_try_load_model", lambda self: None)
    p = _bp.BanquetVinylPlugin()
    p._session = _EchoSession()
    p._input_name = "input"
    p._output_name = "output"
    p._model_ok = True
    p._runtime_quarantined = False

    # Eviction simulieren — die Session-Referenz bleibt für den Reload-Test erhalten:
    _saved_session = p._session
    p._session = None
    p._model_ok = False
    p._load_attempted = False

    def _fake_load():
        p._session = _saved_session
        p._model_ok = True

    monkeypatch.setattr(p, "_try_load_model", _fake_load)

    def _dsp_forbidden(*args, **kwargs):
        raise AssertionError("DSP-Fallback darf nach Reload nicht mehr laufen")

    monkeypatch.setattr(p, "_process_dsp", _dsp_forbidden)

    audio = _np.random.default_rng(5).standard_normal((1, 48_000)).astype(_np.float32) * 0.05
    out = p.process(audio, 48_000)
    assert out is not None
    assert p._model_ok is True and p._session is not None

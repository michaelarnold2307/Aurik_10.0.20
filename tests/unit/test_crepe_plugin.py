from __future__ import annotations

import sys
import types

import numpy as np
import pytest


@pytest.mark.unit
def test_crepe_falls_back_to_yin_when_pyin_fails(monkeypatch):
    from plugins.crepe_plugin import CrepePlugin

    def _pyin_fail(*_args, **_kwargs):
        raise RuntimeError("pyin failed")

    fake_librosa = types.SimpleNamespace(
        note_to_hz=lambda _note: 32.703195,
        pyin=_pyin_fail,
        yin=lambda audio, **_kwargs: np.full(max(1, len(audio) // 512), 220.0, dtype=np.float32),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    # §PERF-R14: pyin_compat scheitern lassen → Plugin fällt auf librosa.yin zurück.
    monkeypatch.setattr("backend.core.dsp.pyin_viterbi_fast.pyin_compat", _pyin_fail)

    plugin = CrepePlugin()
    plugin._session = None
    # §PERF-R14: Reload-Versuch im Fail-closed-Szenario unterbinden —
    # der Test simuliert den Zustand „Session-Fehler + Fallback-Kette".
    monkeypatch.setattr(CrepePlugin, "_load_model", lambda self: None)

    audio = np.random.randn(48_000).astype(np.float32) * 0.01
    result = plugin.analyze(audio, 48_000)

    assert result.model_used == "dsp_yin"
    assert result.f0_hz.size > 0
    assert np.all(result.f0_hz >= 0.0)
    assert np.all(np.isfinite(result.voiced_prob))


def test_crepe_returns_empty_result_when_yin_also_fails(monkeypatch):
    from plugins.crepe_plugin import CrepePlugin

    def _fail(*_args, **_kwargs):
        raise RuntimeError("fallback failed")

    fake_librosa = types.SimpleNamespace(
        note_to_hz=lambda _note: 32.703195,
        pyin=_fail,
        yin=_fail,
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    # §PERF-R14: pyin_compat scheitern lassen → Gesamt-Fallback (leeres Ergebnis).
    monkeypatch.setattr("backend.core.dsp.pyin_viterbi_fast.pyin_compat", _fail)

    plugin = CrepePlugin()
    plugin._session = None
    # §PERF-R14: Reload-Versuch im Fail-closed-Szenario unterbinden.
    monkeypatch.setattr(CrepePlugin, "_load_model", lambda self: None)

    audio = np.random.randn(24_000).astype(np.float32) * 0.01
    result = plugin.analyze(audio, 48_000)

    assert result.model_used == "dsp_yin_failed"
    assert result.f0_hz.shape == (1,)
    assert np.all(result.f0_hz == 0.0)


def test_crepe_self_heals_after_plm_eviction(monkeypatch):
    """§PERF-R14: Nach PLM-Eviction (Session=None) lädt analyze() das Modell
    selbst wieder — statt dauerhaft auf dem langsamen pYIN-Pfad zu bleiben.
    """
    from plugins.crepe_plugin import CrepePlugin

    _loads = {"n": 0}

    def _fake_load(self) -> None:
        _loads["n"] += 1
        self._session = types.SimpleNamespace(run=lambda *a, **k: [np.zeros((1, 8, 360), dtype=np.float32)])

    plugin = CrepePlugin()
    plugin._session = None
    monkeypatch.setattr(CrepePlugin, "_load_model", _fake_load)
    monkeypatch.setattr(plugin, "_analyze_onnx", lambda audio, sr: types.SimpleNamespace(model_used="crepe_onnx"))

    result = plugin.analyze(np.random.randn(48_000).astype(np.float32) * 0.01, 48_000)
    assert _loads["n"] == 1
    assert result.model_used == "crepe_onnx"

"""§PERF-R (2026-09-18) — End-of-Song-Cleanup läuft im Chunked-Pfad nur EINMAL.

Produktionsbefund: Der aggressive Cleanup am Ende jedes `restore()` evakuierte
nach JEDEM Chunk alle warmen Modelle (`force_evict_all`) — Modell-Reloads bzw.
stille ML→DSP-Fallbacks je Chunk (BANQUET-Singleton verlor die Session ohne
Reload-Pfad, §V6 (copilot-instructions.md)). Fix: Der Cleanup ist in
`_run_end_of_song_cleanup()` extrahiert; `restore()` überspringt ihn im
Chunked-Pfad (`_chunked_tail_skip`), `_restore_chunked()` ruft ihn EINMAL
nach der Song-Assembly auf. Der Ganz-Song-Pfad bleibt unverändert.
"""

from __future__ import annotations

import contextlib
import importlib as _real_importlib
from unittest.mock import patch

import numpy as np
import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3
from tests.unit.test_precomputed_phase_plan_determinism import (
    _excellence_mock,
    _fast_restore_kwargs,
    _make_fc_class_mock,
    _make_mgc_class_mock,
    _make_pipeline_mock,
)


@pytest.fixture
def short_audio():
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


@contextlib.contextmanager
def _common_patches(audio):
    """Patch-Set, das restore() ohne schwere Modelle durchlaufen lässt."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(UnifiedRestorerV3, "_select_phases"))
        stack.enter_context(
            patch.object(UnifiedRestorerV3, "_execute_pipeline", return_value=_make_pipeline_mock(audio))
        )
        stack.enter_context(patch.object(UnifiedRestorerV3, "_collect_reporting_analytics", return_value={}))
        stack.enter_context(
            patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock)
        )
        stack.enter_context(patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(audio)))
        stack.enter_context(
            patch(
                "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
                return_value=_make_mgc_class_mock(),
            )
        )
        stack.enter_context(patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0))
        yield


@pytest.mark.unit
def test_restore_skips_cleanup_in_chunked_mode(short_audio):
    """§PERF-R: Chunked-Pfad (_chunked_tail_skip) darf den End-of-Song-Cleanup
    NICHT je Chunk ausführen — sonst Modell-Load-Churn je Chunk."""
    uv3 = UnifiedRestorerV3()
    calls: list[dict] = []

    def _cleanup_spy() -> dict:
        calls.append({})
        return {"unloaded": [], "errors": []}

    with patch.object(UnifiedRestorerV3, "_run_end_of_song_cleanup", side_effect=_cleanup_spy):
        with _common_patches(short_audio):
            result = uv3.restore(
                short_audio, 48000, _chunked_tail_skip=True, _chunked_last=False, **_fast_restore_kwargs()
            )

    assert calls == [], "Chunked-restore() darf den End-of-Song-Cleanup nicht ausführen"
    assert "memory_cleanup" not in (result.metadata or {})


@pytest.mark.unit
def test_restore_runs_cleanup_once_in_whole_song_mode(short_audio):
    """§PERF-R: Ganz-Song-Pfad behält das bisherige Verhalten (Cleanup EINMAL am Ende)."""
    uv3 = UnifiedRestorerV3()
    calls: list[dict] = []

    def _cleanup_spy() -> dict:
        calls.append({})
        return {"unloaded": ["FlashSR"], "errors": []}

    with patch.object(UnifiedRestorerV3, "_run_end_of_song_cleanup", side_effect=_cleanup_spy):
        with _common_patches(short_audio):
            result = uv3.restore(short_audio, 48000, **_fast_restore_kwargs())

    assert len(calls) == 1, "Ganz-Song-restore() muss den Cleanup genau EINMAL ausführen"
    assert result.metadata.get("memory_cleanup") == {"unloaded": ["FlashSR"], "errors": []}


@pytest.mark.unit
def test_run_end_of_song_cleanup_returns_report_and_evicts_plm(short_audio):
    """§PERF-R: Der extrahierte Cleanup entlädt alle spezifizierten Modelle,
    evakuiert den PLM und liefert einen Report — ohne bei Fehlern zu crashen."""
    uv3 = UnifiedRestorerV3()
    unloaded: list[str] = []

    class _FakeUnloadModule:
        def __init__(self, name: str):
            self._name = name

        def __getattr__(self, item: str):
            if item.startswith("unload_"):

                def _unload():
                    unloaded.append(self._name)

                return _unload
            raise AttributeError(item)

    _UNLOAD_MODULES = {
        "plugins.flashsr_plugin": "FlashSR",
        "plugins.utmos_plugin": "UTMOS",
        "plugins.laion_clap_plugin": "LAION-CLAP",
        "plugins.mert_plugin": "MERT",
        "plugins.fcpe_plugin": "FCPE",
        "plugins.basicpitch_plugin": "BasicPitch",
    }

    _original_import_module = _real_importlib.import_module  # vor dem Patch einfrieren!

    def _fake_import_module(name: str):
        if name in _UNLOAD_MODULES:
            return _FakeUnloadModule(_UNLOAD_MODULES[name])
        return _original_import_module(name)

    import backend.core.unified_restorer_v3 as _uv3

    with patch.object(_uv3.importlib, "import_module", side_effect=_fake_import_module):
        with patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=3):
            report = uv3._run_end_of_song_cleanup()

    assert isinstance(report, dict)
    assert report["plm_evicted"] == 3
    assert set(report["unloaded"]) == set(_UNLOAD_MODULES.values())
    assert "errors" in report


# ---------------------------------------------------------------------------
# keep_warm: Fenster-Eviction überspringt teuer ladbare Modelle,
# Druck-Eviction und force_evict_all bleiben aktiv (§PLM-Invariante:
# reines RAM-Scheduling, bit-identisches Audio).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKeepWarm:
    def test_keep_warm_skips_window_eviction(self):
        from unittest.mock import patch

        import backend.core.ml_memory_budget as _mbb
        from backend.core.plugin_lifecycle_manager import PluginLifecycleManager

        plm = PluginLifecycleManager()
        try:
            unloaded = {"n": 0}

            def _unload() -> None:
                unloaded["n"] += 1

            plm.register("TestKeepWarmModel", 0.5, _unload, keep_warm=True)
            plm.register("TestNormalModel", 0.3, _unload)
            # §v10.742: Fenster-Eviction läuft nur bei ≥75 % Budget-Belegung — simulieren.
            with patch.object(_mbb, "_total_gb", 9.0), patch.object(_mbb, "ML_MAX_GB", 10.0):
                plm.evict_for_phase_window(["phase_02_hum_removal"])
            with plm._lock:
                names = set(plm._entries.keys())
            # keep_warm-Modell bleibt, normales Modell wurde entladen
            assert "TestKeepWarmModel" in names
            assert "TestNormalModel" not in names
            assert unloaded["n"] == 1
        finally:
            plm.force_evict_all()

    def test_force_evict_all_evicts_keep_warm(self):
        from backend.core.plugin_lifecycle_manager import PluginLifecycleManager

        plm = PluginLifecycleManager()
        unloaded = {"n": 0}

        def _unload() -> None:
            unloaded["n"] += 1

        plm.register("TestKeepWarmModel", 0.5, _unload, keep_warm=True)
        plm.force_evict_all()
        assert unloaded["n"] == 1
        with plm._lock:
            assert "TestKeepWarmModel" not in plm._entries

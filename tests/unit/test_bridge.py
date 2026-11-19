from __future__ import annotations

"""Tests für backend/api/bridge.py — Aurik 10.0.0 API Bridge (§11 Spec 08).

Prüft:
- Alle __all__-Einträge sind tatsächlich importierbar
- Lazy-Import-Wrapper geben den korrekten Typ zurück (Klasse/Callable/dict)
- export_guard bereinigt NaN, Inf und Clipping korrekt
- Defect-Cache ist Thread-sicher (FIFO, 64 Einträge)
- get_audio_exporter_class() gibt None zurück (kein Hard-Fail)
- get_ml_memory_budget_status() gibt immer ein Dict zurück
- warmup_models_background() hat kein blockierendes time.sleep()
- TYPE_CHECKING-Guards erzeugen keine zirkulären Imports
- __all__ enthält alle Pflicht-Funktionen aus Spec §11
"""


import builtins
import importlib
import inspect
import sys
import threading
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bridge():
    """Importiert die Bridge einmalig pro Modul."""
    return importlib.import_module("backend.api.bridge")


# ---------------------------------------------------------------------------
# 1. Grundlegender Import + __all__
# ---------------------------------------------------------------------------


class TestBridgeImport:
    """Bridge-Modul ist importierbar und hat valides __all__."""

    def test_bridge_imports_cleanly(self, bridge):
        assert bridge is not None, "backend.api.bridge konnte nicht importiert werden"

    def test_bridge_has_all(self, bridge):
        assert hasattr(bridge, "__all__"), "__all__ fehlt — Spec §11 fordert explizite Export-Liste"

    def test_all_is_list_of_strings(self, bridge):
        assert isinstance(bridge.__all__, (list, tuple)), "__all__ muss eine Liste sein"
        for name in bridge.__all__:
            assert isinstance(name, str), f"__all__ enthält Nicht-String: {name!r}"

    def test_all_entries_are_importable(self, bridge):
        """Alle __all__-Einträge müssen tatsächlich im Modul existieren."""
        missing = [name for name in bridge.__all__ if not hasattr(bridge, name)]
        assert not missing, f"In __all__ deklariert, aber nicht im Modul: {missing}"

    def test_all_entries_are_callable_or_data(self, bridge):
        """Alle __all__-Einträge sind Callables oder public data."""
        for name in bridge.__all__:
            obj = getattr(bridge, name)
            # Kann Callable, Klasse, dict, etc. sein — nur None allein ist falsch
            assert obj is not None or name.startswith("_"), (
                f"__all__ enthält None-Eintrag '{name}' — würde Hard-Fail im Frontend verursachen"
            )


# ---------------------------------------------------------------------------
# 2. Pflicht-Funktionen aus Spec §11 (vollständige Liste)
# ---------------------------------------------------------------------------

PFLICHT_FUNKTIONEN = [
    # Defect-Cache
    "cache_defect_result",
    "get_cached_defect_result",
    "clear_defect_cache",
    # Enums
    "get_quality_mode",
    "get_medium_type_enum",
    "get_processing_mode_enum",
    # Kern-Einstiegspunkte
    "get_restorer_classes",
    "get_aurik_denker_class",
    "get_aurik_denker_instance",
    # Analyse
    "get_defect_scanner",
    "get_defect_type",
    "get_medium_classifier_fn",
    "get_era_classifier_fn",
    "get_genre_classifier_fn",
    "get_restorability_estimator_class",
    "get_carrier_forensics_fn",
    "get_audio_file_validator",
    # Qualitätsbewertung
    "get_musical_goals_checker",
    "get_adaptive_goals_fn",
    "get_mushra_evaluator",
    "get_perceptual_quality_scorer",
    # Infrastruktur
    "get_plugin_lifecycle_manager",
    "get_ml_memory_budget_status",
    "get_pipeline_health_state_enum",
    "normalize_pipeline_health_state",
    "resolve_pipeline_fail_reason",
    "get_experience_insights",
    # Audio-Verarbeitung
    "get_audio_exporter_class",
    "get_stem_remix_balancer_fn",
    "get_clipping_classifier",
    "get_lyrics_guided_enhancement_fn",
    "get_cleanup_after_file_fn",
    # Export / Warmup
    "export_guard",
    "warmup_models_background",
]


class TestPflichtFunktionenVorhanden:
    """Alle normativen Bridge-Funktionen aus §11 Spec 08 sind vorhanden."""

    @pytest.mark.parametrize("name", PFLICHT_FUNKTIONEN)
    def test_funktion_im_modul(self, bridge, name):
        assert hasattr(bridge, name), f"Pflicht-Bridge-Funktion '{name}' fehlt — Spec §11 Softwareschichten-Architektur"

    @pytest.mark.parametrize("name", PFLICHT_FUNKTIONEN)
    def test_funktion_in_all(self, bridge, name):
        assert name in bridge.__all__, f"'{name}' fehlt in bridge.__all__ — muss explizit exportiert werden"

    @pytest.mark.parametrize("name", PFLICHT_FUNKTIONEN)
    def test_funktion_ist_callable(self, bridge, name):
        obj = getattr(bridge, name)
        assert callable(obj), f"'{name}' ist nicht callable (Typ: {type(obj).__name__})"


# ---------------------------------------------------------------------------
# 3. export_guard — NaN/Inf-Bereinigung und Clipping (§3.1 Spec 08)
# ---------------------------------------------------------------------------


class TestExportGuard:
    """export_guard bereinigt Audio korrekt (§3.1 Spec 08)."""

    def test_nan_replaced_with_zero(self, bridge):
        audio = np.array([0.5, float("nan"), -0.3], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert np.all(np.isfinite(result)), "export_guard hat NaN nicht entfernt"
        assert result[1] == 0.0

    def test_posinf_clipped(self, bridge):
        audio = np.array([float("inf"), 0.5], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert result[0] == 0.0, "export_guard hat +Inf nicht auf 0.0 gesetzt"

    def test_neginf_clipped(self, bridge):
        audio = np.array([float("-inf"), -0.5], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert result[0] == 0.0, "export_guard hat -Inf nicht auf 0.0 gesetzt"

    def test_values_clipped_to_minus1_plus1(self, bridge):
        audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert np.max(np.abs(result)) <= 1.0, "export_guard clippt nicht auf [-1, 1]"

    def test_output_is_float32(self, bridge):
        audio = np.array([0.3, 0.6], dtype=np.float64)
        result = bridge.export_guard(audio)
        assert result.dtype == np.float32, "export_guard gibt kein float32 zurück"

    def test_valid_audio_unchanged(self, bridge):
        audio = np.array([0.1, -0.2, 0.5, -0.7], dtype=np.float32)
        result = bridge.export_guard(audio)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_stereo_shape_preserved(self, bridge):
        audio = np.zeros((2, 1024), dtype=np.float32)
        audio[0, 0] = float("nan")
        result = bridge.export_guard(audio)
        assert result.shape == (2, 1024), "export_guard verändert Audio-Shape"
        assert np.all(np.isfinite(result))

    def test_empty_array_handled(self, bridge):
        audio = np.array([], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert result.shape == (0,)


# ---------------------------------------------------------------------------
# 4. Defect-Cache — FIFO, Thread-Sicherheit, Grenzwerte
# ---------------------------------------------------------------------------


class TestDefectCache:
    """Defect-Cache ist Thread-sicher und begrenzt auf 64 Einträge (FIFO)."""

    def setup_method(self):
        # Sauberen Zustand sicherstellen
        pass

    def test_cache_round_trip(self, bridge):
        sentinel = object()
        bridge.cache_defect_result("/tmp/test.wav", sentinel)
        assert bridge.get_cached_defect_result("/tmp/test.wav") is sentinel

    def test_cache_miss_returns_none(self, bridge):
        bridge.clear_defect_cache("/tmp/nonexistent.wav")
        assert bridge.get_cached_defect_result("/tmp/nonexistent.wav") is None

    def test_clear_single_entry(self, bridge):
        bridge.cache_defect_result("/tmp/clear_test.wav", {"defects": []})
        bridge.clear_defect_cache("/tmp/clear_test.wav")
        assert bridge.get_cached_defect_result("/tmp/clear_test.wav") is None

    def test_clear_all(self, bridge):
        for i in range(5):
            bridge.cache_defect_result(f"/tmp/clear_all_{i}.wav", i)
        bridge.clear_defect_cache()
        for i in range(5):
            assert bridge.get_cached_defect_result(f"/tmp/clear_all_{i}.wav") is None

    def test_fifo_limit_64(self, bridge):
        """Cache trimmt auf 64 Einträge (FIFO)."""
        bridge.clear_defect_cache()
        for i in range(70):
            bridge.cache_defect_result(f"/tmp/fifo_{i}.wav", i)
        # Die ersten 6 sollten verdrängt worden sein
        [i for i in range(6) if bridge.get_cached_defect_result(f"/tmp/fifo_{i}.wav") is not None]
        # neueste 64 müssen vorhanden sein
        present = [i for i in range(6, 70) if bridge.get_cached_defect_result(f"/tmp/fifo_{i}.wav") is not None]
        assert len(present) == 64, f"FIFO-Limit nicht korrekt: {len(present)} von 64 vorhanden"

    def test_thread_safe_concurrent_writes(self, bridge):
        """Parallele Cache-Schreiboperationen dürfen nicht zu Exceptions führen."""
        bridge.clear_defect_cache()
        errors: list[Exception] = []

        def write_loop(thread_id: int):
            try:
                for i in range(20):
                    bridge.cache_defect_result(f"/tmp/thread_{thread_id}_{i}.wav", (thread_id, i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_loop, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread-Safety-Verletzung im Defect-Cache: {errors}"


# ---------------------------------------------------------------------------
# 5. get_audio_exporter_class — kein Hard-Fail (optional, §11.3)
# ---------------------------------------------------------------------------


class TestAudioExporterClass:
    """get_audio_exporter_class() gibt None zurück statt Exception (§11.3)."""

    def test_returns_type_or_none(self, bridge):
        result = bridge.get_audio_exporter_class()
        assert result is None or isinstance(result, type), (
            f"get_audio_exporter_class() muss type oder None zurückgeben, nicht {type(result)}"
        )

    def test_no_import_error_raised(self, bridge):
        """Kein ImportError bei fehlendem Modul — Fallback-Guard vorhanden."""
        try:
            bridge.get_audio_exporter_class()
        except ImportError as e:
            pytest.fail(f"get_audio_exporter_class() wirft ImportError: {e}")


class TestAudioExporterStatus:
    """AudioExporter-Fallbackpfad liefert explizite Importstatus-Telemetrie."""

    def test_get_audio_exporter_status_has_stable_keys(self, bridge):
        status = bridge.get_audio_exporter_status()
        assert isinstance(status, dict)
        for key in ("available", "failures", "last_error"):
            assert key in status

    def test_audio_exporter_import_failure_updates_status(self, bridge, monkeypatch):
        before = bridge.get_audio_exporter_status()
        before_failures = int(before.get("failures", 0))

        real_import = builtins.__import__

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "backend.core.audio_exporter":
                raise ImportError("simulated audio_exporter import failure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raising_import)

        result = bridge.get_audio_exporter_class()
        assert result is None

        after = bridge.get_audio_exporter_status()
        assert after.get("available") is False
        assert int(after.get("failures", 0)) >= before_failures + 1
        assert "ImportError" in str(after.get("last_error", ""))


# ---------------------------------------------------------------------------
# 5b. MediumDetector-Stub-Policy — fail-closed + Telemetrie
# ---------------------------------------------------------------------------


class TestMediumDetectorStubPolicy:
    """Bridge-MediumDetector-Fallback ist explizit, fail-closed und telemetrierbar."""

    def test_medium_detector_import_failure_returns_fail_closed_stub(self, bridge, monkeypatch):
        fake_mod = types.ModuleType("forensics.medium_detector")
        monkeypatch.setitem(sys.modules, "forensics.medium_detector", fake_mod)

        md = bridge.get_medium_detector()

        assert md is not None, "Bei Importfehler muss ein expliziter Stub statt None zurückkommen"
        assert getattr(md, "is_stub", False) is True
        assert hasattr(md, "detect")

        with pytest.raises(RuntimeError, match="legacy fallback ist deaktiviert"):
            md.detect(np.zeros(128, dtype=np.float32), 48000, file_ext=".wav")

    def test_medium_detector_stub_status_tracks_activation(self, bridge, monkeypatch):
        before = bridge.get_medium_detector_stub_status()
        before_count = int(before.get("activations", 0))

        fake_mod = types.ModuleType("forensics.medium_detector")
        monkeypatch.setitem(sys.modules, "forensics.medium_detector", fake_mod)
        _ = bridge.get_medium_detector()

        after = bridge.get_medium_detector_stub_status()
        assert after.get("active") is True
        assert int(after.get("activations", 0)) >= before_count + 1
        assert isinstance(after.get("last_error", ""), str) and after.get("last_error", "")


# ---------------------------------------------------------------------------
# 6. get_ml_memory_budget_status — immer Dict (§2.37)
# ---------------------------------------------------------------------------


class TestMlMemoryBudgetStatus:
    """get_ml_memory_budget_status() gibt immer ein Dict zurück."""

    def test_returns_dict(self, bridge):
        result = bridge.get_ml_memory_budget_status()
        assert isinstance(result, dict), f"get_ml_memory_budget_status() muss dict zurückgeben, nicht {type(result)}"

    def test_fallback_dict_has_required_keys(self, bridge):
        """Pflicht-Keys aus ml_memory_budget.get_status() sind vorhanden."""
        result = bridge.get_ml_memory_budget_status()
        for key in ("allocated_gb", "free_gb", "max_gb", "models"):
            assert key in result, f"Pflicht-Key '{key}' fehlt in get_ml_memory_budget_status()-Rückgabe"

    def test_values_are_numeric_or_dict(self, bridge):
        result = bridge.get_ml_memory_budget_status()
        assert isinstance(result.get("max_gb", 0), (int, float))
        assert isinstance(result.get("allocated_gb", 0), (int, float))
        assert isinstance(result.get("free_gb", 0), (int, float))
        assert isinstance(result.get("models", {}), dict)


class TestMlMemoryBudgetImportStatus:
    """ml_memory_budget-Fallbackpfad liefert explizite Importstatus-Telemetrie."""

    def test_get_ml_memory_budget_import_status_has_stable_keys(self, bridge):
        status = bridge.get_ml_memory_budget_import_status()
        assert isinstance(status, dict)
        for key in ("available", "failures", "last_error"):
            assert key in status

    def test_ml_memory_budget_import_failure_updates_status(self, bridge, monkeypatch):
        before = bridge.get_ml_memory_budget_import_status()
        before_failures = int(before.get("failures", 0))

        real_import = builtins.__import__

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "backend.core.ml_memory_budget":
                raise ImportError("simulated ml_memory_budget import failure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raising_import)

        result = bridge.get_ml_memory_budget_status()
        assert isinstance(result, dict)
        assert {"max_gb", "allocated_gb", "free_gb", "models"}.issubset(set(result.keys()))

        after = bridge.get_ml_memory_budget_import_status()
        assert after.get("available") is False
        assert int(after.get("failures", 0)) >= before_failures + 1
        assert "ImportError" in str(after.get("last_error", ""))


# ---------------------------------------------------------------------------
# 6b. get_experience_insights — Contract & Robustness (§11.1a Spec 08)
# ---------------------------------------------------------------------------


class _DummyResult:
    def __init__(self, metadata):
        self.metadata = metadata


class TestExperienceInsights:
    def test_defaults_when_metadata_missing(self, bridge):
        res = bridge.get_experience_insights(object())
        assert res["joy_index"] == 0.0
        assert res["fatigue_index"] == 0.0
        assert res["cluster_key"] == ""
        assert res["cluster_policy"] == {}
        assert res["recommendations"] == []
        assert res["recommendation_count"] == 0
        assert "user_guidance" in res
        assert "quality_scale" in res
        assert res["user_guidance"]["tone"] in {"focus", "caution", "confidence"}
        assert res["quality_scale"]["band"] in {"hoch", "mittel", "kritisch"}

    def test_clamps_and_sanitizes_invalid_values(self, bridge):
        r = _DummyResult(
            {
                "joy_runtime_index": {"joy_index": float("nan"), "fatigue_index": 2.5},
                "song_calibration": {"cluster_key": "warm_vocal", "cluster_policy": {"x": 1}},
                "auto_improvement_recommendations": {
                    "count": -3,
                    "recommendations": [
                        {
                            "priority": "high",
                            "focus": "transparenz",
                            "reason": "masking",
                            "action": "reduce phase_03",
                        },
                        "invalid-entry",
                    ],
                },
            }
        )
        res = bridge.get_experience_insights(r)
        assert res["joy_index"] == 0.0
        assert res["fatigue_index"] == 1.0
        assert res["cluster_key"] == "warm_vocal"
        assert res["cluster_policy"] == {"x": 1}
        assert len(res["recommendations"]) == 1
        assert res["recommendations"][0]["focus"] == "transparenz"
        assert res["recommendation_count"] == 1

    def test_count_never_below_recommendation_length(self, bridge):
        r = _DummyResult(
            {
                "auto_improvement_recommendations": {
                    "count": 0,
                    "recommendations": [{"focus": "fatigue", "action": "lower air"}],
                }
            }
        )
        res = bridge.get_experience_insights(r)
        assert res["recommendation_count"] == 1

    def test_guidance_tone_caution_for_degraded_status(self, bridge):
        r = _DummyResult(
            {
                "degradation_status": "degraded",
                "fail_reason": "HPI_FAIL",
            }
        )
        res = bridge.get_experience_insights(r)
        ug = res["user_guidance"]
        assert ug["tone"] == "caution"
        assert isinstance(ug["next_actions"], list)

    def test_recovery_certainty_defaults(self, bridge):
        r = _DummyResult({})
        res = bridge.get_experience_insights(r)
        assert "recovery_certainty" in res
        rc = res["recovery_certainty"]
        assert rc["recoverability_ceiling"] == 0.0
        assert rc["uncertainty_index"] == 1.0
        assert rc["conservative_audio_scalar"] == 1.0
        assert rc["confidence_band"] == ""

    def test_recovery_certainty_sanitized(self, bridge):
        r = _DummyResult(
            {
                "recovery_certainty": {
                    "recoverability_ceiling": 1.4,
                    "uncertainty_index": -1.0,
                    "conservative_audio_scalar": 0.87,
                    "confidence_band": "medium",
                    "restorability_score": 73.2,
                    "transfer_generation_count": 3,
                    "hf_loss_db": -21.5,
                }
            }
        )
        res = bridge.get_experience_insights(r)
        rc = res["recovery_certainty"]
        assert rc["recoverability_ceiling"] == 1.0
        assert rc["uncertainty_index"] == 0.0
        assert rc["conservative_audio_scalar"] == pytest.approx(0.87, abs=1e-6)
        assert rc["confidence_band"] == "medium"
        assert rc["restorability_score"] == pytest.approx(73.2, abs=1e-6)
        assert rc["transfer_generation_count"] == 3
        assert rc["hf_loss_db"] == pytest.approx(-21.5, abs=1e-6)

    def test_spectral_tilt_guard_defaults(self, bridge):
        r = _DummyResult({})
        res = bridge.get_experience_insights(r)
        assert "spectral_tilt_guard" in res
        tg = res["spectral_tilt_guard"]
        assert tg["guard_fired_count"] == 0
        assert tg["phases_guarded"] == []
        assert tg["max_deviation_db_per_oct"] == 0.0
        assert tg["max_wet_cap_applied"] == 0.0

    def test_spectral_tilt_guard_sanitized(self, bridge):
        r = _DummyResult(
            {
                "spectral_tilt_guard": {
                    "guard_fired_count": 3,
                    "phases_guarded": ["phase_06_frequency_restoration", "phase_39_air_band_enhancement"],
                    "max_deviation_db_per_oct": 2.37,
                    "max_wet_cap_applied": 0.72,
                }
            }
        )
        res = bridge.get_experience_insights(r)
        tg = res["spectral_tilt_guard"]
        assert tg["guard_fired_count"] == 3
        assert len(tg["phases_guarded"]) == 2
        assert tg["max_deviation_db_per_oct"] == pytest.approx(2.37, abs=1e-6)
        assert tg["max_wet_cap_applied"] == pytest.approx(0.72, abs=1e-6)

    def test_hf_hallucination_guard_defaults(self, bridge):
        """HF-Guard-Schlüssel ist vorhanden und zeigt Null-Zustand wenn keine Phasen feuerten."""
        r = _DummyResult({})
        res = bridge.get_experience_insights(r)
        assert "hf_hallucination_guard" in res
        hf = res["hf_hallucination_guard"]
        assert hf["guard_fired_count"] == 0
        assert hf["phases_guarded"] == []
        assert hf["max_delta_ratio"] == pytest.approx(0.0, abs=1e-9)
        assert hf["min_cap_hz"] is None

    def test_hf_hallucination_guard_populated(self, bridge):
        """Wenn hf_hallucination_guard in metadata vorhanden, wird es korrekt weitergegeben."""
        r = _DummyResult(
            {
                "hf_hallucination_guard": {
                    "guard_fired_count": 2,
                    "phases_guarded": ["phase_06_frequency_restoration", "phase_55_harmonic_exciter"],
                    "max_delta_ratio": 0.12,
                    "min_cap_hz": 7000.0,
                }
            }
        )
        res = bridge.get_experience_insights(r)
        hf = res["hf_hallucination_guard"]
        assert hf["guard_fired_count"] == 2
        assert "phase_06_frequency_restoration" in hf["phases_guarded"]
        assert hf["max_delta_ratio"] == pytest.approx(0.12, abs=1e-6)
        assert hf["min_cap_hz"] == pytest.approx(7000.0, abs=0.1)


# ---------------------------------------------------------------------------
# 7. warmup_models_background — kein blockierendes sleep() (§9.7.4)
# ---------------------------------------------------------------------------


class TestWarmupModelsBackground:
    """warmup_models_background() blockiert nicht durch time.sleep() (§9.7.4)."""

    def test_no_redundant_sleep_in_source(self, bridge):
        """Quellcode darf kein time.sleep(2) haben — QTimer regelt das Timing."""
        source = inspect.getsource(bridge.warmup_models_background)
        assert "time.sleep(2)" not in source, (
            "warmup_models_background() enthält redundantes time.sleep(2) — "
            "§9.7.4: QTimer.singleShot(2000, ...) steuert Timing; sleep im Thread ist überflüssig"
        )

    def test_is_callable(self, bridge):
        assert callable(bridge.warmup_models_background)

    @pytest.mark.timeout(600)  # Volles Plugin-Warmup ist lastabhängig: Befund 2026-10-05 —
    # im Chunk-Verbund unter CPU-Last lief der Test in das CLI-Timeout von 20–25 s und wurde
    # fälschlich rot (isoliert und im Chunk sofort grün, 339 passed). Der Marker übersteuert
    # das CLI-Default deterministisch (§G5 (GEBOTE.md): Tests dürfen nicht lastabhängig kippen).
    def test_completes_without_exception(self, bridge):
        """Warmup läuft durch ohne Exception (alle Plugins optional)."""
        # Synchroner Aufruf — alle Imports schlagen fehl → kein Absturz
        try:
            bridge.warmup_models_background()
        except Exception as e:
            pytest.fail(f"warmup_models_background() wirft Exception: {e}")


# ---------------------------------------------------------------------------
# 7b. Startup-Check — keine stille Exception, Status-Telemetrie vorhanden
# ---------------------------------------------------------------------------


class TestStartupCheckStatus:
    """Startup-Check-Bridge meldet Fehler transparent über Status-Telemetrie."""

    def test_get_startup_check_status_has_stable_keys(self, bridge):
        status = bridge.get_startup_check_status()
        assert isinstance(status, dict)
        for key in ("available", "failures", "last_error"):
            assert key in status

    def test_startup_check_import_failure_increments_status(self, bridge, monkeypatch):
        before = bridge.get_startup_check_status()
        before_failures = int(before.get("failures", 0))

        real_import = builtins.__import__

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "backend.core.startup_model_check":
                raise ImportError("simulated startup check import failure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raising_import)

        result = bridge.get_startup_check_result()
        assert result is None

        after = bridge.get_startup_check_status()
        assert after.get("available") is False
        assert int(after.get("failures", 0)) >= before_failures + 1
        assert "ImportError" in str(after.get("last_error", ""))


class TestPreAnalysisResultStatus:
    """PreAnalysisResult-Importpfad ist transparent und telemetriert."""

    def test_get_pre_analysis_result_status_has_stable_keys(self, bridge):
        status = bridge.get_pre_analysis_result_status()
        assert isinstance(status, dict)
        for key in ("available", "failures", "last_error"):
            assert key in status

    def test_pre_analysis_result_import_failure_updates_status(self, bridge, monkeypatch):
        before = bridge.get_pre_analysis_result_status()
        before_failures = int(before.get("failures", 0))

        real_import = builtins.__import__

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "backend.core.pre_analysis":
                raise ImportError("simulated pre_analysis import failure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raising_import)

        result_type = bridge.get_pre_analysis_result_type()
        assert result_type is None

        after = bridge.get_pre_analysis_result_status()
        assert after.get("available") is False
        assert int(after.get("failures", 0)) >= before_failures + 1
        assert "ImportError" in str(after.get("last_error", ""))


# ---------------------------------------------------------------------------
# 8. Qualitätsbewertungs-Wrapper (neu hinzugefügt §8.1)
# ---------------------------------------------------------------------------


class TestQualitaetsBewertungsWrapper:
    """Neue Qualitätsbewertungs-Accessor sind vorhanden und aufrufbar."""

    def test_get_musical_goals_checker_returns_type(self, bridge):
        result = bridge.get_musical_goals_checker()
        assert isinstance(result, type), (
            f"get_musical_goals_checker() muss eine Klasse zurückgeben, nicht {type(result)}"
        )

    def test_get_adaptive_goals_fn_returns_callable(self, bridge):
        result = bridge.get_adaptive_goals_fn()
        assert callable(result), f"get_adaptive_goals_fn() muss einen Callable zurückgeben, nicht {type(result)}"

    def test_get_mushra_evaluator_returns_something(self, bridge):
        result = bridge.get_mushra_evaluator()
        assert result is not None, "get_mushra_evaluator() gibt None zurück"

    def test_get_perceptual_quality_scorer_returns_something(self, bridge):
        result = bridge.get_perceptual_quality_scorer()
        assert result is not None, "get_perceptual_quality_scorer() gibt None zurück"

    def test_get_plugin_lifecycle_manager_returns_something(self, bridge):
        result = bridge.get_plugin_lifecycle_manager()
        assert result is not None, "get_plugin_lifecycle_manager() gibt None zurück"


# ---------------------------------------------------------------------------
# 9. TYPE_CHECKING — keine zirkulären Imports (§11 Spec 08)
# ---------------------------------------------------------------------------


class TestTypeCheckingGuards:
    """TYPE_CHECKING-Guards erzeugen keine zirkulären Imports."""

    def test_bridge_importable_in_fresh_interpreter(self, bridge):
        """Bridge-Modul ist ohne Vorwärts-Imports importierbar."""
        # Bereits durch das fixture geladen — Smoke-Test
        assert hasattr(bridge, "__all__")

    def test_no_circular_import_via_typing(self):
        """Erneuter Import ist idempotent."""
        import importlib

        m1 = importlib.import_module("backend.api.bridge")
        m2 = importlib.import_module("backend.api.bridge")
        assert m1 is m2, "Modul wird bei erneutem Import neu geladen (kein Caching)"


# ---------------------------------------------------------------------------
# 10. Lazy-Import-Muster — Rückgabetypen der wichtigsten Wrapper
# ---------------------------------------------------------------------------


class TestLazyImportMuster:
    """Lazy-Import-Wrapper geben den spezifizierten Typ zurück."""

    @pytest.mark.parametrize(
        ("raw_mode", "expected"),
        [
            ("Restoration", "Restoration"),
            ("restoration", "Restoration"),
            ("quality", "Restoration"),
            ("fast", "Restoration"),
            ("balanced", "Restoration"),
            ("Studio 2026", "Studio 2026"),
            ("studio2026", "Studio 2026"),
            ("studio", "Studio 2026"),
            ("maximum", "Studio 2026"),
            ("unknown", "Restoration"),
            (None, "Restoration"),
        ],
    )
    def test_normalize_user_mode_maps_release_and_legacy_aliases(self, bridge, raw_mode, expected):
        assert bridge.normalize_user_mode(raw_mode) == expected

    def test_get_quality_mode_returns_enum_type(self, bridge):
        qm = bridge.get_quality_mode()
        assert isinstance(qm, type), "get_quality_mode() gibt keine Klasse zurück"

    def test_get_restorer_classes_returns_tuple_of_two_types(self, bridge):
        result = bridge.get_restorer_classes()
        assert isinstance(result, tuple) and len(result) == 2, (
            "get_restorer_classes() muss (RestorationConfig, UnifiedRestorerV3) als 2-Tuple zurückgeben"
        )
        assert all(isinstance(t, type) for t in result), "Tuple-Elemente sind keine Klassen"

    def test_get_defect_type_returns_enum_type(self, bridge):
        dt = bridge.get_defect_type()
        assert isinstance(dt, type), "get_defect_type() gibt keine Klasse zurück"

    def test_get_medium_classifier_fn_returns_callable(self, bridge):
        fn = bridge.get_medium_classifier_fn()
        assert callable(fn), "get_medium_classifier_fn() gibt keinen Callable zurück"

    def test_get_era_classifier_fn_returns_callable(self, bridge):
        fn = bridge.get_era_classifier_fn()
        assert callable(fn), "get_era_classifier_fn() gibt keinen Callable zurück"

    def test_get_stem_remix_balancer_fn_returns_callable(self, bridge):
        fn = bridge.get_stem_remix_balancer_fn()
        assert callable(fn), "get_stem_remix_balancer_fn() gibt keinen Callable zurück"

    def test_get_carrier_forensics_fn_returns_callable(self, bridge):
        fn = bridge.get_carrier_forensics_fn()
        assert callable(fn), "get_carrier_forensics_fn() gibt keinen Callable zurück"


# ---------------------------------------------------------------------------
# 11. resolve_pipeline_fail_reason — strukturierter Fail-Reason (§RELEASE_MUST)
# ---------------------------------------------------------------------------


class TestResolvePipelineFailReason:
    """resolve_pipeline_fail_reason gibt immer eine Zeichenkette zurück."""

    def test_returns_string_without_args(self, bridge):
        result = bridge.resolve_pipeline_fail_reason()
        assert isinstance(result, str)

    def test_returns_string_with_metadata(self, bridge):
        result = bridge.resolve_pipeline_fail_reason(
            metadata={"fail_reason": "test_error"},
        )
        assert isinstance(result, str)

    def test_uses_typed_fail_reason(self, bridge):
        result = bridge.resolve_pipeline_fail_reason(
            typed_fail_reason="goal_regression",
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 12. normalize_pipeline_health_state — Typ-Sicherheit
# ---------------------------------------------------------------------------


class TestNormalizePipelineHealthState:
    """normalize_pipeline_health_state ist robust gegen unbekannte Werte."""

    def test_returns_object_for_none(self, bridge):
        result = bridge.normalize_pipeline_health_state(None)
        assert result is not None

    def test_returns_object_for_ok(self, bridge):
        result = bridge.normalize_pipeline_health_state("ok")
        assert result is not None

    def test_property_value_exists(self, bridge):
        result = bridge.normalize_pipeline_health_state("degraded")
        assert hasattr(result, "value"), "normalize_pipeline_health_state() muss Objekt mit .value-Attribut zurückgeben"


# ---------------------------------------------------------------------------
# 13. _AnalysisLruCache — LRU-Eviction + Content-Addressing
# ---------------------------------------------------------------------------


class TestAnalysisLruCache:
    """_AnalysisLruCache: LRU-Eviction, Thread-Safety, Content-Addressing."""

    def _make_cache(self, bridge, maxsize: int = 4):
        return bridge._AnalysisLruCache(maxsize=maxsize)

    def test_put_get_round_trip(self, bridge):
        c = self._make_cache(bridge)
        c.put("k1", {"a": 1})
        assert c.get("k1") == {"a": 1}

    def test_miss_returns_none(self, bridge):
        c = self._make_cache(bridge)
        assert c.get("no_such_key") is None

    def test_lru_evicts_oldest_accessed(self, bridge):
        """LRU evicts the least-recently *accessed* entry, not oldest *inserted*."""
        c = self._make_cache(bridge, maxsize=3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        # Access 'a' so it becomes MRU
        assert c.get("a") == 1
        # Insert 'd' → 'b' should be evicted (LRU), not 'a'
        c.put("d", 4)
        assert c.get("b") is None, "LRU muss den am längsten nicht zugegriffenen Eintrag ('b') verdrängen"
        assert c.get("a") == 1, "'a' wurde nach LRU-Zugriff fälschlicherweise verdrängt"
        assert c.get("c") == 3
        assert c.get("d") == 4

    def test_put_updates_existing_promotes_to_mru(self, bridge):
        c = self._make_cache(bridge, maxsize=2)
        c.put("x", 1)
        c.put("y", 2)
        c.put("x", 99)  # update — should promote x to MRU
        c.put("z", 3)  # evicts y (LRU), not x
        assert c.get("x") == 99
        assert c.get("y") is None
        assert c.get("z") == 3

    def test_path_alias_lookup(self, bridge):
        c = self._make_cache(bridge)
        c.put("content_hash_abc", "result_A", path_alias="/tmp/song.wav")
        assert c.get_by_path("/tmp/song.wav") == "result_A"

    def test_evicted_key_removes_path_alias(self, bridge):
        c = self._make_cache(bridge, maxsize=2)
        c.put("k1", "v1", path_alias="/tmp/one.wav")
        c.put("k2", "v2", path_alias="/tmp/two.wav")
        c.put("k3", "v3", path_alias="/tmp/three.wav")  # evicts k1
        assert c.get_by_path("/tmp/one.wav") is None, "Alias für evicted key muss entfernt werden"

    def test_remove_by_key(self, bridge):
        c = self._make_cache(bridge)
        c.put("del_me", "x")
        c.remove("del_me")
        assert c.get("del_me") is None

    def test_remove_by_path_alias(self, bridge):
        c = self._make_cache(bridge)
        c.put("hash123", "v", path_alias="/tmp/rm.wav")
        c.remove("/tmp/rm.wav")
        assert c.get_by_path("/tmp/rm.wav") is None

    def test_clear_empties_cache(self, bridge):
        c = self._make_cache(bridge, maxsize=10)
        for i in range(5):
            c.put(f"k{i}", i)
        c.clear()
        assert len(c) == 0

    def test_thread_safe_concurrent_operations(self, bridge):
        c = self._make_cache(bridge, maxsize=50)
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(30):
                    c.put(f"t{tid}_k{i}", (tid, i))
                    _ = c.get(f"t{tid}_k{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors, f"Thread-Safety-Verletzung in _AnalysisLruCache: {errors}"

    def test_maxsize_never_exceeded(self, bridge):
        c = self._make_cache(bridge, maxsize=5)
        for i in range(20):
            c.put(f"k{i}", i)
        assert len(c) <= 5, f"Cache überschreitet maxsize: len={len(c)}"


# ---------------------------------------------------------------------------
# 14. content_cache_key — Content-Addressing Utility
# ---------------------------------------------------------------------------


class TestContentCacheKey:
    """content_cache_key liefert stabile, kollisionsarme SHA-256-Schlüssel."""

    def test_returns_64_char_hex_string(self, tmp_path, bridge):
        f = tmp_path / "audio.wav"
        f.write_bytes(b"\x00" * 100)
        key = bridge.content_cache_key(str(f))
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_file_same_key(self, tmp_path, bridge):
        f = tmp_path / "stable.wav"
        f.write_bytes(b"\x01\x02\x03" * 200)
        k1 = bridge.content_cache_key(str(f))
        k2 = bridge.content_cache_key(str(f))
        assert k1 == k2, "content_cache_key muss für identische Datei stabil sein"

    def test_different_content_different_key(self, tmp_path, bridge):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        f1.write_bytes(b"\xaa" * 100)
        f2.write_bytes(b"\xbb" * 100)
        assert bridge.content_cache_key(str(f1)) != bridge.content_cache_key(str(f2))

    def test_same_content_different_path_same_key(self, tmp_path, bridge):
        """Selber Inhalt unter verschiedenem Pfad → gleicher Key (Content-Addressing)."""
        data = b"\xff\x00\x1a" * 512
        f1 = tmp_path / "original.wav"
        f2 = tmp_path / "renamed.wav"
        f1.write_bytes(data)
        f2.write_bytes(data)
        assert bridge.content_cache_key(str(f1)) == bridge.content_cache_key(str(f2)), (
            "Content-Addressing: gleicher Inhalt muss gleichen Key ergeben"
        )

    def test_missing_file_returns_path_fallback(self, bridge):
        key = bridge.content_cache_key("/nonexistent/audio_xyz_12345.wav")
        assert isinstance(key, str) and len(key) > 0, "Fehlende Datei darf kein Exception werfen"


# ---------------------------------------------------------------------------
# 15. LRU-Cache via Defect-Cache-API (Integrations-Test)
# ---------------------------------------------------------------------------


class TestDefectCacheLruIntegration:
    """Defect-Cache-API verwendet jetzt LRU statt FIFO."""

    def test_lru_keeps_recently_accessed_entry(self, bridge):
        """LRU: ein früh eingefügter, aber kürzlich geholter Eintrag bleibt erhalten."""
        bridge.clear_defect_cache()
        # Fülle auf 64 Einträge
        for i in range(64):
            bridge.cache_defect_result(f"/tmp/lru_fill_{i:03d}.wav", {"idx": i})
        # Greife auf den ersten Eintrag zu → wird MRU
        _ = bridge.get_cached_defect_result("/tmp/lru_fill_000.wav")
        # Füge einen weiteren ein → should evict entry #1 (LRU), not #0
        bridge.cache_defect_result("/tmp/lru_new.wav", {"idx": 99})
        assert bridge.get_cached_defect_result("/tmp/lru_fill_000.wav") is not None, (
            "LRU darf den zuletzt zugegriffenen Eintrag nicht verdrängen"
        )
        assert bridge.get_cached_defect_result("/tmp/lru_fill_001.wav") is None, (
            "LRU muss den am längsten nicht zugegriffenen Eintrag verdrängen"
        )

    def test_same_content_different_path_hits_cache(self, tmp_path, bridge):
        """Selber Dateiinhalt unter zwei Pfaden → zweiter Aufruf trifft Cache."""
        data = b"\xde\xad\xbe\xef" * 2048
        p1 = tmp_path / "original.wav"
        p2 = tmp_path / "copy.wav"
        p1.write_bytes(data)
        p2.write_bytes(data)
        sentinel = {"scores": {"HUM": 0.9}}
        bridge.cache_defect_result(str(p1), sentinel)
        result = bridge.get_cached_defect_result(str(p2))
        assert result is sentinel, (
            "Content-Addressed Cache miss: gleicher Dateiinhalt unter anderem Pfad nicht gefunden"
        )


# ---------------------------------------------------------------------------
# 16. Neue Bridge-Getter (§11 erweiterte Core-Module)
# ---------------------------------------------------------------------------

NEW_GETTER_NAMES = [
    "get_german_schlager_classifier_fn",
    "get_harmonic_preservation_guard",
    "get_feedback_chain",
    "get_physical_ceiling_estimator",
    "get_per_phase_musical_goals_gate",
    "get_emotional_arc_metric",
    "get_micro_dynamics_em",
    "get_goal_applicability_filter",
    "get_perceptual_salience_estimator",
]


class TestNeueGetterVorhanden:
    """Alle 9 neuen Core-Modul-Getter sind in bridge.__all__ und aufrufbar."""

    @pytest.mark.parametrize("name", NEW_GETTER_NAMES)
    def test_getter_in_all(self, bridge, name):
        assert name in bridge.__all__, f"'{name}' fehlt in bridge.__all__"

    @pytest.mark.parametrize("name", NEW_GETTER_NAMES)
    def test_getter_is_callable(self, bridge, name):
        fn = getattr(bridge, name, None)
        assert fn is not None and callable(fn), f"'{name}' ist nicht callable"

    @pytest.mark.parametrize("name", NEW_GETTER_NAMES)
    def test_getter_returns_non_none(self, bridge, name):
        fn = getattr(bridge, name)
        result = fn()
        assert result is not None, f"'{name}()' gibt None zurück — Core-Modul nicht verfügbar?"

    def test_get_harmonic_preservation_guard_has_protect_or_guard(self, bridge):
        guard = bridge.get_harmonic_preservation_guard()
        has_method = (
            hasattr(guard, "guard")
            or hasattr(guard, "protect")
            or hasattr(guard, "apply")
            or hasattr(guard, "apply_correction")
        )
        assert has_method, "HarmonicPreservationGuard fehlt .guard/.protect/.apply/.apply_correction-Methode"

    def test_get_feedback_chain_has_run_or_evaluate(self, bridge):
        fc = bridge.get_feedback_chain()
        has_method = hasattr(fc, "run") or hasattr(fc, "evaluate") or hasattr(fc, "optimize")
        assert has_method, "FeedbackChain fehlt .run/.evaluate/.optimize-Methode"

    def test_get_physical_ceiling_estimator_has_estimate(self, bridge):
        pce = bridge.get_physical_ceiling_estimator()
        assert hasattr(pce, "estimate"), "PhysicalCeilingEstimator fehlt .estimate()-Methode"

    def test_get_mdem_has_morph(self, bridge):
        mdem = bridge.get_micro_dynamics_em()
        assert hasattr(mdem, "morph"), "MicroDynamicsEnvelopeMorphing fehlt .morph()-Methode"

    def test_get_goal_filter_has_evaluate_or_filter(self, bridge):
        gaf = bridge.get_goal_applicability_filter()
        has_method = hasattr(gaf, "evaluate") or hasattr(gaf, "filter") or hasattr(gaf, "apply")
        assert has_method, "GoalApplicabilityFilter fehlt .evaluate/.filter/.apply-Methode"

    def test_content_cache_key_in_all(self, bridge):
        assert "content_cache_key" in bridge.__all__

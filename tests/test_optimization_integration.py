from __future__ import annotations

"""Smoke tests for the canonical CLI entrypoint replacing the legacy orchestrator script."""


import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest


@pytest.mark.unit
def test_optimization_integration():
    """Canonical CLI exists and exposes the current AurikDenker-based interface."""
    cli_path = Path("cli/aurik_cli.py")
    assert cli_path.exists(), "Kanonischer CLI-Einstieg cli/aurik_cli.py fehlt"
    assert not Path("orchestrator_and_cli.py").exists(), (
        "Legacy-Skript orchestrator_and_cli.py sollte nicht mehr verwendet werden"
    )

    result = subprocess.run(
        [sys.executable, "-m", "cli.aurik_cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI --help schlug fehl: {result.stderr or result.stdout}"
    help_text = (result.stdout or "") + (result.stderr or "")
    assert "Restoration" in help_text
    assert "Studio 2026" in help_text

    source = cli_path.read_text(encoding="utf-8")
    assert "AurikDenker" in source or "get_aurik_denker" in source
    assert "orchestrator_and_cli.py" not in source
    assert "run_pre_analysis(" in source
    assert "denker.denke(" in source


def test_cli_uses_frontend_export_contract():
    """CLI muss denselben Bridge-Exportvertrag wie das Frontend nutzen."""
    source = Path("cli/aurik_cli.py").read_text(encoding="utf-8")
    assert "export_guard(" in source
    assert "validate_export_quality(" in source
    assert "build_export_quality_gate_payload(" in source
    assert "get_audio_exporter_class(" in source
    assert "os.replace(tmp_path, out_path)" in source
    assert "sf.write(output_path" not in source


def test_export_quality_blocks_structural_silence_lift() -> None:
    """§0h: Export darf in struktureller Stille keine Energie hinzufuegen."""
    from backend.exporter import validate_export_quality

    result = SimpleNamespace(
        quality_estimate=0.9,
        metadata={
            "structural_silence_audit": {
                "failed": True,
                "max_lift_db": 2.4,
            },
            "musical_goals": {"scores": {}, "violations": [], "thresholds": {}},
        },
    )

    passed, warnings = validate_export_quality(result)

    assert passed is False
    assert any("Structural-Silence" in warning or "Stille-Zonen" in warning for warning in warnings)


def test_export_quality_blocks_severe_vocal_naturalness_damage() -> None:
    """§0p: schwere Formant-/Vibrato-Schäden sind Export-Hörschaden, kein kosmetischer Warnfall."""
    from backend.exporter import validate_export_quality

    result = SimpleNamespace(
        quality_estimate=0.9,
        metadata={
            "vocal_quality_check": {
                "formant_integrity": 0.68,
                "vibrato_depth_preservation": 0.76,
            },
            "musical_goals": {"scores": {}, "violations": [], "thresholds": {}},
        },
    )

    passed, warnings = validate_export_quality(result)

    assert passed is False
    assert any("Formant" in warning for warning in warnings)
    assert any("Vibrato" in warning for warning in warnings)


def test_ab_plan_recommendation_prefers_safe_candidate_over_metric_gain() -> None:
    """A/B-Harness darf Metrikgewinn nie ueber AFG/VQI-Sicherheitsveto stellen."""
    from scripts.ab_plan_eval import _recommend_plan

    baseline = {"hpi": 0.05, "artifact_freedom": 0.96, "vqi": 0.78, "n_phases_executed": 3}
    unsafe_candidate = {"hpi": 0.20, "artifact_freedom": 0.90, "vqi": 0.69, "n_phases_executed": 2}

    recommendation = _recommend_plan(baseline, unsafe_candidate)

    assert recommendation["winner"] == "A"
    assert "artifact_freedom<0.95" in recommendation["safety_violations_b"]
    assert "vqi<0.72" in recommendation["safety_violations_b"]


def test_cli_export_helper_uses_audio_exporter_and_quality_payload(monkeypatch, tmp_path):
    """CLI-Export muss AudioExporter, Export-Guard und Gate-Payload wie das Frontend nutzen."""
    from cli import aurik_cli

    calls: dict[str, Any] = {}

    class FakeAudioExporter:
        FORMATS: ClassVar[dict[str, dict[str, Any]]] = {".wav": {}}

        def export(self, audio, sr, output_path, **kwargs):
            calls["audio"] = np.asarray(audio)
            calls["sr"] = sr
            calls["output_path"] = output_path
            calls["kwargs"] = kwargs
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

    def fake_export_guard(audio):
        calls["guard_count"] = int(calls.get("guard_count", 0)) + 1
        return np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32)), -1.0, 1.0)

    monkeypatch.setattr(aurik_cli, "get_audio_exporter_class", lambda: FakeAudioExporter)
    monkeypatch.setattr(aurik_cli, "export_guard", fake_export_guard)
    monkeypatch.setattr(aurik_cli, "validate_export_quality", lambda result: (False, ["gate-warning"]))
    monkeypatch.setattr(
        aurik_cli,
        "build_export_quality_gate_payload",
        lambda result: {
            "passed": False,
            "degradation_status": "degraded",
            "fail_reason": "gate-warning",
            "recovery_attempted": True,
            "best_possible_reached": True,
            "fallback_quality_floor": {"status": "recovered"},
        },
    )

    result = SimpleNamespace(audio=np.zeros((2, 8), dtype=np.float32), metadata={})
    # §0c: Gate-Fehler degradieren statt blockieren — Export MUSS die Datei
    # schreiben (Hardstop ohne Ausgabedatei ist normativ unzulässig).
    ok, warnings, payload = aurik_cli._export_audio_frontend_parity(
        result,
        str(tmp_path / "out.wav"),
        np.array([[np.nan, 2.0, -2.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
        aurik_cli.logging.getLogger("test_cli_export"),
    )

    assert (tmp_path / "out.wav").read_bytes() == b"fake-wav"  # Datei existiert — kein Hardstop
    assert calls["guard_count"] == 2
    assert result.metadata["export_quality_gate_failed"] is True
    assert result.metadata["export_quality_gate_warnings"] == ["gate-warning"]
    assert result.metadata["export_degraded_by_quality_gate"] is True
    assert result.metadata["degradation_status"] == "degraded"


def test_cli_export_helper_applies_mono_guard_and_forwards_musiclover_metadata(monkeypatch, tmp_path):
    """Bei mono_compatibility_warning muss CLI vor Export leichtes Stereo-Softening anwenden."""
    from cli import aurik_cli

    calls: dict[str, Any] = {}

    class FakeAudioExporter:
        FORMATS: ClassVar[dict[str, dict[str, Any]]] = {".wav": {}}

        def export(self, audio, sr, output_path, **kwargs):
            calls["audio"] = np.asarray(audio)
            calls["sr"] = sr
            calls["kwargs"] = kwargs
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

    def fake_export_guard(audio):
        return np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32)), -1.0, 1.0)

    monkeypatch.setattr(aurik_cli, "get_audio_exporter_class", lambda: FakeAudioExporter)
    monkeypatch.setattr(aurik_cli, "export_guard", fake_export_guard)
    monkeypatch.setattr(aurik_cli, "validate_export_quality", lambda result: (True, []))
    monkeypatch.setattr(
        aurik_cli,
        "build_export_quality_gate_payload",
        lambda result: {
            "passed": True,
            "degradation_status": "ok",
            "fail_reason": "",
            "recovery_attempted": False,
            "best_possible_reached": False,
            "fallback_quality_floor": {"status": "passed"},
            "worldclass_composite_gate": {
                "wcs": 0.86,
                "threshold": 0.85,
                "profile": "instrumental",
                "artifact_veto": False,
                "passed": True,
            },
            "threshold_evidence": {
                "worldclass_composite_gate": {
                    "source_class": "C",
                    "revalidate_by": "2026-09-30",
                }
            },
            "musiclover": {
                "vocal_integrity": {"vqi": 0.87, "singer_identity_cosine": 0.95},
                "temporal_risk": {"hotspot_count": 2},
                "stereo_integrity": {"mono_compatibility_warning": True},
                "decision_trace": {
                    "all_sota_real": False,
                    "vocal_restoration_capability_status": "sota_fallback",
                },
            },
        },
    )

    # Anti-phase Stereo, damit MonoGuard wirksam wird.
    restored = np.column_stack(
        [
            np.ones(64, dtype=np.float32),
            -np.ones(64, dtype=np.float32),
        ]
    )
    result = SimpleNamespace(
        audio=restored.copy(),
        metadata={
            "hybrid_engineer_vector": {
                "vocal_identity_preservation": 0.93,
                "formant_integrity": 0.89,
                "vibrato_depth_preservation": 0.88,
                "breath_naturalness": 0.91,
                "micro_dynamic_correlation": 0.92,
                "transient_articulation": 0.90,
                "stereo_scene_stability": 0.86,
                "noise_texture_authenticity": 0.94,
                "spectral_color_preservation": 0.89,
                "emotional_arc_preservation": 0.93,
                "artifact_freedom": 0.98,
                "goal_team_balance": 0.87,
            }
        },
    )
    ok, warnings, payload = aurik_cli._export_audio_frontend_parity(
        result,
        str(tmp_path / "ok.wav"),
        restored,
        restored,
        aurik_cli.logging.getLogger("test_cli_export"),
    )

    assert ok is True
    assert warnings == []
    assert payload["passed"] is True
    out_audio = calls["audio"]
    # Side wurde um 8 % reduziert: ±1.0 → ±0.92
    assert float(out_audio[:, 0].mean()) == pytest.approx(0.92, abs=1e-5)
    assert float(out_audio[:, 1].mean()) == pytest.approx(-0.92, abs=1e-5)
    md = calls["kwargs"]["metadata"]
    assert md["quality_gate_musiclover_vqi"] == "0.87"
    assert md["quality_gate_musiclover_temporal_hotspots"] == "2"
    assert md["quality_gate_musiclover_mono_warning"] == "True"
    assert md["quality_gate_musiclover_all_sota_real"] == "False"
    assert md["quality_gate_musiclover_sota_reason"] == "sota_fallback"
    assert md["quality_gate_worldclass_score"] == "0.86"
    assert md["quality_gate_worldclass_threshold"] == "0.85"
    assert md["quality_gate_worldclass_passed"] == "True"
    assert md["quality_gate_worldclass_profile"] == "instrumental"
    assert '"artifact_freedom": 0.98' in md["quality_gate_hybrid_engineer_vector"]
    assert '"vocal_identity_preservation": 0.93' in md["quality_gate_hybrid_engineer_vector"]
    assert md["quality_gate_evidence_worldclass_source_class"] == "C"
    assert md["quality_gate_evidence_worldclass_revalidate_by"] == "2026-09-30"


def test_batch_processor_forwards_worldclass_and_hybrid_metadata(monkeypatch, tmp_path):
    """Batch-Results sollen den aktuellen Weltklasse-Contract in metadata weiterreichen."""
    import batch_processor as bp

    class _FakeSf:
        @staticmethod
        def write(_path, _audio, _sr):
            return None

    class _FakeDenker:
        def denke(self, _audio, _sr, **_kwargs):
            return SimpleNamespace(
                audio=np.zeros((8, 2), dtype=np.float32),
                total_time_seconds=1.2,
                quality_estimate=0.91,
                musical_goals_scores={"natuerlichkeit": 0.95, "authentizitaet": 0.92, "tonal_center": 0.97},
                metadata={
                    "worldclass_composite_gate": {"wcs": 0.89},
                    "hybrid_engineer_vector": {
                        "artifact_freedom": 0.98,
                        "vocal_identity_preservation": 0.93,
                    },
                },
            )

    monkeypatch.setattr(bp, "sf", _FakeSf)
    monkeypatch.setattr(bp, "_get_aurik_denker", lambda: _FakeDenker())
    monkeypatch.setattr(bp, "_run_pre_analysis", lambda **_kwargs: None)
    monkeypatch.setattr(bp, "_validate_export_quality", lambda _result: (True, []))
    monkeypatch.setattr(
        bp,
        "_build_export_quality_gate_payload",
        lambda _result: {
            "passed": True,
            "degradation_status": "ok",
            "fail_reason": "",
            "recovery_attempted": False,
            "best_possible_reached": False,
            "fallback_quality_floor": {"status": "passed"},
            "thresholds": {"quality_estimate": 0.55, "level_drop_db": 3.0},
            "worldclass_composite_gate": {
                "wcs": 0.89,
                "threshold": 0.85,
                "profile": "instrumental",
                "artifact_veto": False,
                "passed": True,
            },
            "threshold_evidence": {
                "worldclass_composite_gate": {
                    "source_class": "C",
                    "revalidate_by": "2026-09-30",
                }
            },
            "musiclover": {
                "vocal_integrity": {"vqi": 0.87, "singer_identity_cosine": 0.95},
                "temporal_risk": {"hotspot_count": 2},
                "stereo_integrity": {"mono_compatibility_warning": True},
                "goal_attainment": {"remaining_count": 1},
                "decision_trace": {
                    "all_sota_real": False,
                    "vocal_restoration_capability_status": "sota_fallback",
                },
            },
        },
    )
    monkeypatch.setattr(
        bp,
        "_load_audio_file",
        lambda *_args, **_kwargs: {"audio": np.zeros((8, 2), dtype=np.float32), "sr": 48_000},
    )

    processor = bp.BatchProcessor(output_dir=tmp_path, workers=1, resume=False)
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")

    res = processor.process_file(input_file, {"mode": "restoration"})

    assert res["success"] is True
    assert res["metadata"]["quality_gate_worldclass_score"] == "0.89"
    assert res["metadata"]["quality_gate_worldclass_threshold"] == "0.85"
    assert res["metadata"]["quality_gate_worldclass_passed"] == "True"
    assert res["metadata"]["quality_gate_worldclass_profile"] == "instrumental"
    assert '"artifact_freedom": 0.98' in res["metadata"]["quality_gate_hybrid_engineer_vector"]
    assert '"vocal_identity_preservation": 0.93' in res["metadata"]["quality_gate_hybrid_engineer_vector"]
    assert res["metadata"]["quality_gate_evidence_worldclass_source_class"] == "C"
    assert res["metadata"]["quality_gate_evidence_worldclass_revalidate_by"] == "2026-09-30"
    assert res["metadata"]["quality_gate_musiclover_vqi"] == "0.87"
    assert res["metadata"]["quality_gate_musiclover_temporal_hotspots"] == "2"
    assert res["metadata"]["quality_gate_musiclover_mono_warning"] == "True"
    assert res["metadata"]["quality_gate_musiclover_all_sota_real"] == "False"
    assert res["metadata"]["quality_gate_musiclover_sota_reason"] == "sota_fallback"


def test_batch_processor_normalizes_release_alias_mode(monkeypatch, tmp_path):
    """Batch muss Release-Aliase via Bridge normalisieren (maximum -> Studio 2026)."""
    import batch_processor as bp

    mode_calls: dict[str, str] = {}

    class _FakeSf:
        @staticmethod
        def write(_path, _audio, _sr):
            return None

    class _FakeDenker:
        def denke(self, _audio, _sr, **kwargs):
            mode_calls["mode"] = str(kwargs.get("mode", ""))
            return SimpleNamespace(
                audio=np.zeros((8, 2), dtype=np.float32),
                total_time_seconds=1.0,
                metadata={"worldclass_composite_gate": {"wcs": 0.9}, "hybrid_engineer_vector": {}},
            )

    monkeypatch.setattr(bp, "sf", _FakeSf)
    monkeypatch.setattr(bp, "_get_aurik_denker", lambda: _FakeDenker())
    monkeypatch.setattr(bp, "_run_pre_analysis", lambda **_kwargs: None)
    monkeypatch.setattr(bp, "_validate_export_quality", lambda _result: (True, []))
    monkeypatch.setattr(
        bp,
        "_build_export_quality_gate_payload",
        lambda _result: {"passed": True, "degradation_status": "ok", "fail_reason": ""},
    )
    monkeypatch.setattr(
        bp,
        "_load_audio_file",
        lambda *_args, **_kwargs: {"audio": np.zeros((8, 2), dtype=np.float32), "sr": 48_000},
    )

    processor = bp.BatchProcessor(output_dir=tmp_path, workers=1, resume=False)
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")

    res = processor.process_file(input_file, {"mode": "maximum"})

    assert res["success"] is True
    assert mode_calls["mode"] == "Studio 2026"

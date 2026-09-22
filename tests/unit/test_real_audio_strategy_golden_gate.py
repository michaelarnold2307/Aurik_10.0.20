from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import backend.core.real_audio_strategy_golden_gate as _strategy_gate
from backend.core.real_audio_strategy_golden_gate import run_real_audio_strategy_golden_gate


@pytest.mark.unit
def test_resolve_strategy_phase_coalitions_uses_uv3_registry(monkeypatch) -> None:
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    monkeypatch.setattr(
        UnifiedRestorerV3,
        "get_active_phase_coalitions",
        classmethod(
            lambda cls, selected_phases, is_studio_2026=False: {
                "digital_repair_chain": (
                    "phase_23_spectral_repair",
                    "phase_50_spectral_repair",
                )
            }
        ),
    )

    result = _strategy_gate._resolve_strategy_phase_coalitions(
        ["phase_23_spectral_repair", "phase_50_spectral_repair"],
        is_studio_2026=False,
    )
    assert "digital_repair_chain" in result


def test_scan_strategy_case_passes_coalitions_to_mapper(monkeypatch, tmp_path) -> None:
    audio_file = tmp_path / "case.wav"
    audio_file.write_bytes(b"RIFF")

    monkeypatch.setattr(
        _strategy_gate,
        "_audio_from_import",
        lambda path, target_sr: (np.zeros(target_sr, dtype=np.float32), target_sr, 1.0),
    )

    class _DummyScanner:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def scan(self, audio, sr, material_type, file_ext):
            return types.SimpleNamespace(
                scores={"a": types.SimpleNamespace(defect_type=types.SimpleNamespace(value="aliasing"), severity=0.8)},
                duration_seconds=1.0,
                sample_rate=sr,
                get_top_defects=lambda n: [
                    types.SimpleNamespace(defect_type=types.SimpleNamespace(value="aliasing"), severity=0.8)
                ],
            )

    monkeypatch.setattr(_strategy_gate, "DefectScanner", _DummyScanner)
    monkeypatch.setattr(
        _strategy_gate,
        "reason_about_defects",
        lambda defect_scores, material, audio, sample_rate: types.SimpleNamespace(
            primary_cause="aliasing",
            ranked_causes=[("aliasing", 0.8)],
            recommended_phases=["phase_23_spectral_repair", "phase_50_spectral_repair"],
        ),
    )
    monkeypatch.setattr(
        _strategy_gate,
        "_resolve_strategy_phase_coalitions",
        lambda seed_phases, is_studio_2026=False: {
            "digital_repair_chain": ("phase_23_spectral_repair", "phase_50_spectral_repair")
        },
    )

    captured: dict[str, object] = {}

    class _DummyMapper:
        def phases_for_defect_profile(
            self, defects, max_phases=10, mode="restoration", material=None, phase_coalitions=None
        ):
            captured["mode"] = mode
            captured["material"] = material
            captured["phase_coalitions"] = phase_coalitions
            return ["phase_23_spectral_repair", "phase_50_spectral_repair"]

    monkeypatch.setattr(_strategy_gate, "DefectPhaseMapper", lambda: _DummyMapper())

    case = {
        "case_id": "T-1",
        "path": audio_file.name,
        "material_type": "cd_digital",
        "accepted_causes": ["aliasing"],
        "required_phases": ["phase_23_spectral_repair"],
    }
    result = _strategy_gate._scan_strategy_case(case, tmp_path, 48_000)

    assert captured["mode"] == "restoration"
    assert captured["material"] == "cd_digital"
    assert isinstance(captured["phase_coalitions"], dict)
    assert result.metadata["dominant_coalition"] == "digital_repair_chain"
    assert result.metadata["dominant_coalition_ratio"] == 1.0


def test_real_audio_strategy_golden_manifest_passes_gate() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root / "audit" / "real_audio_strategy_golden_manifest.json"

    if not (repo_root / "test_audio").is_dir():
        pytest.skip("test_audio-Korpus nicht vorhanden (gitignored)")

    report = run_real_audio_strategy_golden_gate(manifest_path=manifest, repo_root=repo_root)

    assert report.scanned_cases >= 8
    assert report.skipped_cases == []
    assert report.gate.passed is True
    assert report.gate.cause_topk_accuracy == 1.0
    assert report.gate.phase_recall == 1.0
    assert report.gate.phase_precision == 1.0
    assert report.gate.forbidden_phase_violations == 0
    assert report.gate.order_violations == 0
    assert report.gate.fail_reasons == ()

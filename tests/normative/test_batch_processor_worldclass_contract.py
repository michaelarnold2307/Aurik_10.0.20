from __future__ import annotations

"""[RELEASE_MUST] Batch-Contract fuer Worldclass-/Hybrid-Metadaten.

Sichert, dass der Batch-Pfad die Worldclass-/Hybrid-Metadaten
`quality_gate_worldclass_score` und `quality_gate_hybrid_engineer_vector`
im Ergebnisobjekt fuehrt.
"""


import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_batch_source_declares_worldclass_and_hybrid_metadata_contract() -> None:
    src = Path("batch_processor.py").read_text(encoding="utf-8")
    assert '"quality_gate_worldclass_score"' in src
    assert '"quality_gate_worldclass_threshold"' in src
    assert '"quality_gate_worldclass_passed"' in src
    assert '"quality_gate_hybrid_engineer_vector"' in src
    assert '"quality_gate_evidence_worldclass_source_class"' in src
    assert '"quality_gate_musiclover_vqi"' in src
    assert "json.dumps(" in src


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_batch_process_file_forwards_worldclass_and_hybrid_metadata(monkeypatch, tmp_path: Path) -> None:
    import batch_processor as bp

    class _FakeSf:
        @staticmethod
        def write(_path: str, _audio: np.ndarray, _sr: int) -> None:
            return None

    class _FakeDenker:
        def denke(self, _audio: np.ndarray, _sr: int, **_kwargs):
            return SimpleNamespace(
                audio=np.zeros((8, 2), dtype=np.float32),
                total_time_seconds=1.0,
                quality_estimate=0.93,
                musical_goals_scores={
                    "natuerlichkeit": 0.95,
                    "authentizitaet": 0.92,
                    "tonal_center": 0.97,
                },
                metadata={
                    "worldclass_composite_gate": {"wcs": 0.91},
                    "hybrid_engineer_vector": {
                        "artifact_freedom": 0.98,
                        "vocal_identity_preservation": 0.94,
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
            "worldclass_composite_gate": {
                "wcs": 0.91,
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
                "vocal_integrity": {"vqi": 0.88, "singer_identity_cosine": 0.95},
                "temporal_risk": {"hotspot_count": 1},
                "stereo_integrity": {"mono_compatibility_warning": False},
                "decision_trace": {
                    "all_sota_real": True,
                    "vocal_restoration_capability_status": "all_real",
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
    input_file = tmp_path / "case.wav"
    input_file.write_bytes(b"fake")

    result = processor.process_file(input_file, {"mode": "restoration"})

    assert result["success"] is True
    assert result["metadata"]["quality_gate_worldclass_score"] == "0.91"
    assert result["metadata"]["quality_gate_worldclass_threshold"] == "0.85"
    assert result["metadata"]["quality_gate_worldclass_passed"] == "True"
    assert result["metadata"]["quality_gate_evidence_worldclass_source_class"] == "C"
    assert result["metadata"]["quality_gate_musiclover_vqi"] == "0.88"

    vector_json = result["metadata"]["quality_gate_hybrid_engineer_vector"]
    vector = json.loads(vector_json)
    assert vector["artifact_freedom"] == pytest.approx(0.98)
    assert vector["vocal_identity_preservation"] == pytest.approx(0.94)

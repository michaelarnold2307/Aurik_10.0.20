from __future__ import annotations

"""[RELEASE_MUST] §8.6c Evidenzklassen-Metadatenvertrag.

Prueft, dass Gate-Schwellen eine maschinenlesbare Evidenzklassifikation tragen.
"""


from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_07 = _ROOT / ".github" / "specs" / "07_quality_and_tests.md"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestEvidenceClassMetadataContract:
    def test_spec_declares_evidence_class_gate_requirements(self) -> None:
        content = _SPEC_07.read_text(encoding="utf-8")

        assert "§8.6c Wissenschaftliche Evidenzklassen (Gate-faehig)" in content
        assert "source_class" in content
        assert "source_ref" in content
        assert "validated_on" in content
        assert "revalidate_by" in content

    def test_uv3_exports_threshold_evidence_payload(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"threshold_evidence": {' in content
        assert '"artifact_freedom_gate": {' in content
        assert '"vqi_gate": {' in content
        assert '"hpi_gate": {' in content
        assert '"worldclass_composite_gate": {' in content
        assert '"psychoacoustic_naturalness_gate": {' in content

    def test_uv3_worldclass_gate_uses_class_c_with_revalidation_deadline(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"worldclass_composite_gate": {' in content
        assert '"source_class": "C"' in content
        assert '"revalidate_by": "2026-09-30"' in content

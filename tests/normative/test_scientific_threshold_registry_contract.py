from __future__ import annotations

"""[RELEASE_MUST] Wissenschaftlicher Registry-Vertrag fuer Gate-Schwellen.

Sichert, dass P1-Governance-Luecken (HPI/AFG/VQI/WCS) in Specs und Policy
maschinenlesbar und mit wissenschaftlichen Quellenachsen verankert sind.
"""


from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_07 = _ROOT / ".github" / "specs" / "07_quality_and_tests.md"
_REGISTRY = _ROOT / "policy" / "scientific_threshold_evidence_registry.yaml"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestScientificThresholdRegistryContract:
    def test_spec_declares_registry_contract(self) -> None:
        content = _SPEC_07.read_text(encoding="utf-8")

        assert "§8.6f [RELEASE_MUST] Scientific Threshold Evidence Registry" in content
        assert "policy/scientific_threshold_evidence_registry.yaml" in content

    def test_registry_covers_all_release_gates(self) -> None:
        content = _REGISTRY.read_text(encoding="utf-8")

        assert "artifact_freedom_gate:" in content
        assert "vqi_gate:" in content
        assert "hpi_gate:" in content
        assert "worldclass_composite_gate:" in content
        assert "psychoacoustic_naturalness_gate:" in content

    def test_registry_enforces_source_metadata(self) -> None:
        content = _REGISTRY.read_text(encoding="utf-8")

        assert "source_class:" in content
        assert "source_ids:" in content
        assert "source_ref:" in content
        assert "validated_on:" in content
        assert "revalidate_by:" in content

    def test_runtime_threshold_evidence_uses_scientific_refs(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"threshold_evidence": {' in content
        assert "DOI:10.1109/ICASSP.2017.7952243" in content
        assert "DOI:10.1109/89.365378" in content
        assert "DOI:10.1016/S0892-1997(05)80150-X" in content
        assert "DOI:10.1016/j.jvoice.2003.09.003" in content
        assert "DOI:10.1121/10.0015518" in content
        assert "ITU-R BS.1770-5" in content
        assert "EBU R128" in content

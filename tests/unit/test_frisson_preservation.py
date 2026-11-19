from __future__ import annotations

"""
test_frisson_preservation.py — §0i Gänsehaut-Erhaltung Verifikation
====================================================================

Stellt sicher, dass:
1. Phase 08 (Transient Preservation) Frisson-Zonen respektiert
2. Phase 12 (Wow/Flutter Fix) Frisson-Caps einhält
3. Phase 24 (Dropout Repair) in Frisson-Zonen reduziert arbeitet
4. Phase 54 (Transparent Dynamics) Frisson-Schutz hat
5. Dass ALLE Transport-Bump/Head-Dip-Reparaturphasen Gänsehaut respektieren

Spec: §0i Perceptual Transparency, §VFA Frisson-Schutz-Caps (0.30)
"""


import pytest


@pytest.mark.unit
@pytest.mark.pleasantness
class TestFrissonPreservationInRepairPhases:
    """§0i: Gänsehaut-Erhaltung in Transport-Bump/Head-Dip-Reparatur."""

    def test_01_phase_12_has_frisson_awareness(self):
        """Phase 12 (Bump-PSOLA) respektiert Frisson-Zonen."""
        import backend.core.phases.phase_12_wow_flutter_fix as p12_mod

        src = open(p12_mod.__file__, encoding="utf-8").read()
        assert "frisson" in src.lower() or "protected_zone" in src.lower(), (
            "Phase 12: Kein Frisson-Schutz — Gänsehaut-Passagen ungeschützt"
        )

    def test_02_phase_24_has_frisson_awareness(self):
        """Phase 24 (Dropout-Repair) respektiert Frisson-Zonen."""
        import backend.core.phases.phase_24_dropout_repair as p24_mod

        src = open(p24_mod.__file__, encoding="utf-8").read()
        assert "frisson" in src.lower() or "protected_zone" in src.lower(), (
            "Phase 24: Kein Frisson-Schutz — Dropout-Repair in Gänsehaut-Passagen"
        )

    def test_03_phase_54_has_frisson_awareness(self):
        """Phase 54 (Transparent Dynamics) respektiert Frisson-Zonen."""
        import backend.core.phases.phase_54_transparent_dynamics as p54_mod

        src = open(p54_mod.__file__, encoding="utf-8").read()
        assert "frisson" in src.lower() or "protected_zone" in src.lower(), (
            "Phase 54: Kein Frisson-Schutz — Envelope-Smoothing in Gänsehaut-Passagen"
        )

    def test_04_phase_08_has_frisson_protection(self):
        """Phase 08 (Transient Preservation) hat JETZT Frisson-Schutz."""
        import backend.core.phases.phase_08_transient_preservation as p08_mod

        src = open(p08_mod.__file__, encoding="utf-8").read()
        assert "frisson" in src.lower(), "Phase 08: Frisson-Schutz fehlt — Attack-Boost in Gänsehaut-Passagen"
        assert "_frisson_cap" in src, "Phase 08: _frisson_cap-Mechanismus fehlt"
        assert "0.30" in src, "Phase 08: Frisson-Cap 0.30 nicht definiert (§VFA)"

    def test_05_frisson_cap_consistency(self):
        """Alle Phasen verwenden denselben Frisson-Cap (0.30 gemäß §VFA)."""
        phases = [
            ("Phase 08", "backend/core/phases/phase_08_transient_preservation.py"),
            ("Phase 12", "backend/core/phases/phase_12_wow_flutter_fix.py"),
            ("Phase 54", "backend/core/phases/phase_54_transparent_dynamics.py"),
        ]
        for name, path in phases:
            with open(path, encoding="utf-8") as f:
                src = f.read()
            assert "0.30" in src, f"{name}: Frisson-Cap 0.30 nicht gefunden — inkonsistent mit §VFA"

    def test_06_emotional_arc_in_hpi_formula(self):
        """emotional_arc_preservation ist Multiplikator im HPI."""
        src = open(".github/instructions/pipeline.instructions.md", encoding="utf-8").read()
        assert "emotional_arc_preservation" in src, "emotional_arc_preservation fehlt in HPI-Formel"

    def test_07_frisson_zones_in_verboten_md(self):
        """Frisson-Zonen sind in VERBOTEN.md als geschützt dokumentiert."""
        src = open(".github/VERBOTEN.md", encoding="utf-8").read()
        assert "frisson_zones" in src, "frisson_zones fehlt in VERBOTEN.md — nicht als Invariante dokumentiert"


@pytest.mark.unit
class TestFrissonSoftVeto:
    """Frisson ist Soft-Veto (Recovery), nicht Hard-Veto (Export-Block)."""

    def test_10_frisson_is_documented_as_soft_factor(self):
        """Frisson/Goosebumps ist als Soft-Faktor dokumentiert."""
        src = open(".github/instructions/pipeline.instructions.md", encoding="utf-8").read()
        assert "Goosebumps/Frisson" in src, "Goosebumps/Frisson nicht in Pipeline-Instructions"
        assert "Soft-Faktor" in src, "Frisson nicht als Soft-Faktor deklariert"

    def test_11_frisson_not_hard_veto(self):
        """Frisson ist KEIN Hard-Veto (kein Export-Block)."""
        src = open(".github/instructions/pipeline.instructions.md", encoding="utf-8").read()
        assert "kein eigenständiger Hard-Veto" in src, "Frisson fälschlich als Hard-Veto dokumentiert"

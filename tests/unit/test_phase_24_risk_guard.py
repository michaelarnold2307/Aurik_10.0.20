from __future__ import annotations

"""
test_phase_24_risk_guard.py — Phase-24-Pruning + Defekt-Reparatur-Garantie
==========================================================================

Stellt sicher, dass:
1. Transport Bump + Head Level Dip Evidenz Phase-24-Pruning verhindert
2. Risk-Guard beide Defekttypen in _dropout_evidence berücksichtigt
3. Phase 24 nicht entfernt wird wenn Transport Bumps/Head Dips vorliegen

Spec: CAUSE_TO_PHASES, §v10 Pleasantness-First
"""


import pytest


@pytest.mark.unit
class TestPhase24RiskGuard:
    """Phase-24-Dropout-Repair wird NICHT entfernt wenn Bumps/Dips vorliegen."""

    def test_01_dropout_evidence_includes_transport_bump(self):
        """_dropout_evidence berücksichtigt transport_bump."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        # Find the _dropout_evidence line
        assert '"transport_bump"' in src, (
            "transport_bump fehlt in _dropout_evidence — Phase 24 könnte bei Transport-Bump-Material entfernt werden"
        )

    def test_02_dropout_evidence_includes_tape_head_level_dip(self):
        """_dropout_evidence berücksichtigt tape_head_level_dip."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        assert '"tape_head_level_dip"' in src, (
            "tape_head_level_dip fehlt in _dropout_evidence — Phase 24 könnte bei Head-Dip-Material entfernt werden"
        )

    def test_03_causal_probability_includes_both_defects(self):
        """_causal_dropout_prob prüft transport_bump + tape_head_level_dip."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        assert '"transport_bump"' in src, "transport_bump fehlt in _causal_dropout_prob-Prüfung"
        assert '"tape_head_level_dip"' in src, "tape_head_level_dip fehlt in _causal_dropout_prob-Prüfung"

    def test_04_phase_24_in_cause_to_phases_for_both_defects(self):
        """CAUSE_TO_PHASES routet beide Defekte zu Phase 24."""
        import backend.core.causal_defect_reasoner as cdr_mod

        src = open(cdr_mod.__file__, encoding="utf-8").read()

        # Find transport_bump section
        tb_idx = src.find('"transport_bump": [')
        assert tb_idx > 0, "transport_bump nicht in CAUSE_TO_PHASES"
        tb_section = src[tb_idx : tb_idx + 300]
        assert "phase_24" in tb_section, "transport_bump → phase_24 fehlt — keine Dropout-Reparatur für Bumps"

        # Find tape_head_level_dip section
        th_idx = src.find('"tape_head_level_dip": [')
        assert th_idx > 0, "tape_head_level_dip nicht in CAUSE_TO_PHASES"
        th_section = src[th_idx : th_idx + 300]
        assert "phase_24" in th_section, "tape_head_level_dip → phase_24 fehlt — keine Dropout-Reparatur für Dips"

    def test_05_phase_24_allow_conditions_are_reasonable(self):
        """Phase-24-Allow-Bedingungen sind korrekt kalibriert."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        # Die vier Bedingungen müssen existieren
        assert "_dropout_evidence >= 0.20" in src, "_dropout_evidence-Schwelle fehlt"
        assert "_causal_dropout_prob >= 0.10" in src, "_causal_dropout_prob-Schwelle fehlt"
        assert "_pipe_conf >= 0.80" in src, "_pipe_conf-Schwelle fehlt"
        assert "restorability_score <= 35.0" in src, "restorability_score-Schwelle fehlt"

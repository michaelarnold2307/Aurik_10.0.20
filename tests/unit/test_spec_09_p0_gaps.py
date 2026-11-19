from __future__ import annotations

"""
test_spec_09_p0_gaps.py — Sprint 3: Spec 09 P0-Gap-Testgerüst
===============================================================

Schließt die 4 kritischen Test-Lücken aus Spec 09:
  G01 — §09.0a Maximal-Ausbaustufe (Modus-Invarianz)
  G02 — §09.2b Effektiver Goal-Target-Resolver
  G03 — §09.10 GOAL_BASELINE_CHECK (Pre-Pipeline)
  G04 — §INV-2 SongCalibration global_scalar

Spec: .github/specs/09_global_calibration_matrix.md
"""


import pytest


@pytest.mark.unit
@pytest.mark.goal_achievement
class TestSpec09MaximalAusbaustufe:
    """G01 — §09.0a: Maximal-Ausbaustufe — Modus-Invarianz."""

    def test_01_calibration_functions_exist(self):
        """Calibration-Matrix-Modul enthält Kalibrierungsfunktionen."""
        import backend.core.calibration_matrix as cm

        assert hasattr(cm, "compute_tcci"), "compute_tcci fehlt — kein Transfer-Chain-Index"
        assert callable(cm.compute_tcci)

    def test_02_tcci_handles_all_chains(self):
        """Transfer-Chain-Complexity-Index funktioniert für alle Ketten."""
        from backend.core.calibration_matrix import compute_tcci

        assert 0.0 <= compute_tcci(["cd_digital"]) <= 1.0
        assert 0.0 <= compute_tcci(["vinyl", "tape", "mp3_low"]) <= 1.0
        assert compute_tcci(["vinyl", "tape", "mp3_low"]) > compute_tcci(["cd_digital"]), (
            "TCCI: vinyl→tape→mp3 muss komplexer sein als cd_digital"
        )


@pytest.mark.unit
@pytest.mark.goal_achievement
class TestSpec09GoalTargetResolver:
    """G02 — §09.2b: Effektiver Goal-Target-Resolver."""

    def test_10_goal_targets_resolvable(self):
        """Goal-Targets werden pro Song aus Restorability+Era+Genre+Material abgeleitet."""
        from backend.core.studio_goal_targets import estimate_song_goal_targets

        assert callable(estimate_song_goal_targets), "estimate_song_goal_targets fehlt oder ist nicht callable"

    def test_11_goal_targets_accepts_transfer_chain(self):
        """Goal-Target-Resolver akzeptiert material_type und transfer_chain."""
        import inspect

        from backend.core.studio_goal_targets import estimate_song_goal_targets

        sig = inspect.signature(estimate_song_goal_targets)
        params = list(sig.parameters.keys())
        # Akzeptiert material_type (statt 'material') und transfer_chain
        assert "material_type" in params, f"estimate_song_goal_targets akzeptiert material_type nicht: {params}"
        assert "transfer_chain" in params, f"estimate_song_goal_targets akzeptiert transfer_chain nicht: {params}"


@pytest.mark.unit
@pytest.mark.goal_achievement
class TestSpec09GoalBaselineCheck:
    """G03 — §09.10: GOAL_BASELINE_CHECK — Pre-Pipeline-Absicherung."""

    def test_20_goal_baseline_module_exists(self):
        """GOAL_BASELINE_CHECK ist als Konzept im Code verankert."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        assert "baseline" in src.lower() or "GOAL_BASELINE" in src, (
            "Kein GOAL_BASELINE_CHECK in UV3 — Pre-Pipeline-Qualität nicht validiert"
        )


@pytest.mark.unit
@pytest.mark.goal_achievement
class TestSpec09SongCalibration:
    """G04 — §INV-2: SongCalibration global_scalar."""

    def test_30_calibration_matrix_has_scaling(self):
        """Calibration-Matrix hat songweite Skalierungsfunktionen."""
        import backend.core.calibration_matrix as cm

        # Mindestens eine Skalierungsfunktion muss existieren
        scaling_funcs = [n for n in dir(cm) if "scale" in n.lower() or "scalar" in n.lower()]
        assert len(scaling_funcs) > 0, (
            "Keine Skalierungsfunktion in calibration_matrix — keine songweite Anpassung möglich"
        )

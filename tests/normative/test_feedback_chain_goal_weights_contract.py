from __future__ import annotations

"""Normative §2.56 FeedbackChain Goal-Weights Contract.

Verifies the key invariant from spec §2.56:
  "P1/P2-dominant song NEVER produces looser pruning than a uniform-weight song."

Also verifies:
  - MOS regression tolerance tightens for P1/P2-heavy songs
  - Convergence delta shrinks at high MOS for P1/P2-heavy songs
  - Uniform weights produce identical results to no-weights (backwards compat)
  - Extreme P4/P5 dominance does not push threshold past floor (-0.30)
  - Extreme P1/P2 dominance does not push threshold past ceiling (-0.005)

These are spec-level contractual tests that must pass in CI.
"""


from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feedback_chain(
    goal_weights: dict | None = None,
    material: str = "cd_digital",
    restorability: float = 60.0,
    defect_severity: float = 0.3,
) -> Any:
    """Create a FeedbackChain with the given goal_weights and contextual attributes."""
    from backend.core.feedback_chain import FeedbackChain

    fc = FeedbackChain(
        material=material,
        restorability_score=restorability,
        defect_severity_mean=defect_severity,
    )
    fc.goal_weights = goal_weights
    return fc


def _uniform_weights() -> dict:
    return dict.fromkeys(
        [
            "natuerlichkeit",
            "authentizitaet",
            "tonal_center",
            "timbre_authentizitaet",
            "artikulation",
            "emotionalitaet",
            "mikrodynamik",
            "groove",
            "transparenz",
            "waerme",
            "bassgewalt",
            "separation_fidelity",
            "brillanz",
            "raumtiefe",
        ],
        1.0,
    )


def _p1p2_heavy_weights(multiplier: float = 1.5) -> dict:
    gw = _uniform_weights()
    for k in ("natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation"):
        gw[k] = float(min(2.0, multiplier))
    return gw


def _p4p5_heavy_weights(multiplier: float = 1.6) -> dict:
    gw = _uniform_weights()
    for k in ("brillanz", "raumtiefe", "waerme", "bassgewalt"):
        gw[k] = float(min(2.0, multiplier))
    return gw


# ---------------------------------------------------------------------------
# §2.56 Prune Threshold Invariant
# ---------------------------------------------------------------------------


class TestPruneThresholdContract:
    """P1/P2-dominant song must produce *stricter* (more negative) pruning threshold."""

    @pytest.mark.parametrize(
        "material,restorability",
        [
            ("vinyl", 40.0),
            ("cd_digital", 85.0),
            ("shellac", 20.0),
            ("reel_tape", 55.0),
        ],
    )
    def test_p1p2_tighter_than_uniform(self, material: str, restorability: float) -> None:
        """For any material/restorability, P1/P2-heavy song → threshold ≥ uniform threshold.

        "Tighter" means the threshold value is less negative (closer to zero):
        phases need a smaller negative delta to survive — stricter pruning,
        so that only truly beneficial phases remain active on critical P1/P2 goals.
        """
        fc_uniform = _make_feedback_chain(_uniform_weights(), material=material, restorability=restorability)
        fc_p1p2 = _make_feedback_chain(_p1p2_heavy_weights(1.4), material=material, restorability=restorability)

        t_uniform = fc_uniform._compute_adaptive_prune_threshold(is_restorative=True)
        t_p1p2 = fc_p1p2._compute_adaptive_prune_threshold(is_restorative=True)
        assert t_p1p2 >= t_uniform - 1e-9, (
            f"P1/P2-heavy threshold ({t_p1p2:.4f}) must be ≥ uniform ({t_uniform:.4f}) "
            f"(less negative = stricter) for material={material}, rest={restorability}"
        )

    @pytest.mark.parametrize(
        "material,restorability",
        [
            ("vinyl", 40.0),
            ("cd_digital", 85.0),
            ("shellac", 20.0),
        ],
    )
    def test_p4p5_looser_than_uniform(self, material: str, restorability: float) -> None:
        """P4/P5-heavy song → looser threshold (more negative = more lenient).

        Brillanz/Raumtiefe goals are less critical; phases get more slack.
        """
        fc_uniform = _make_feedback_chain(_uniform_weights(), material=material, restorability=restorability)
        fc_p4p5 = _make_feedback_chain(_p4p5_heavy_weights(1.5), material=material, restorability=restorability)

        t_uniform = fc_uniform._compute_adaptive_prune_threshold(is_restorative=True)
        t_p4p5 = fc_p4p5._compute_adaptive_prune_threshold(is_restorative=True)
        # P4/P5 heavy → more negative (looser) = lower numeric value
        assert t_p4p5 <= t_uniform + 1e-9, (
            f"P4/P5-heavy threshold ({t_p4p5:.4f}) must be ≤ uniform ({t_uniform:.4f}) "
            f"(more negative = looser) for material={material}, rest={restorability}"
        )

    def test_bounds_respected_p1p2_extreme(self) -> None:
        """Extreme P1/P2 weights must not push threshold above -0.005 (ceiling)."""
        for mat in ("cd_digital", "vinyl", "shellac"):
            fc = _make_feedback_chain(_p1p2_heavy_weights(2.0), material=mat, restorability=50.0)
            t = fc._compute_adaptive_prune_threshold(is_restorative=False)
            assert t <= -0.005 + 1e-9, f"Threshold {t:.4f} exceeds ceiling -0.005 for {mat}"

    def test_bounds_respected_p4p5_extreme(self) -> None:
        """Extreme P4/P5 weights must not push threshold below -0.30 (floor)."""
        for mat in ("shellac", "wax_cylinder"):
            fc = _make_feedback_chain(_p4p5_heavy_weights(2.0), material=mat, restorability=10.0)
            t = fc._compute_adaptive_prune_threshold(is_restorative=True)
            assert t >= -0.30 - 1e-9, f"Threshold {t:.4f} breaches floor -0.30 for {mat}"

    def test_uniform_equals_no_weights(self) -> None:
        """Uniform weights (all 1.0) must produce same result as no weights (backwards compat)."""
        for mat in ("vinyl", "cd_digital"):
            fc_none = _make_feedback_chain(None, material=mat, restorability=60.0)
            fc_uniform = _make_feedback_chain(_uniform_weights(), material=mat, restorability=60.0)
            t_none = fc_none._compute_adaptive_prune_threshold(is_restorative=False)
            t_uni = fc_uniform._compute_adaptive_prune_threshold(is_restorative=False)
            assert abs(t_none - t_uni) < 1e-6, (
                f"Uniform weights should produce identity result vs no-weights for {mat}: "
                f"none={t_none:.6f}, uniform={t_uni:.6f}"
            )


# ---------------------------------------------------------------------------
# §2.56 MOS Regression Tolerance Invariant
# ---------------------------------------------------------------------------


class TestMosRegressionToleranceContract:
    """P1/P2-dominant songs must have tighter MOS tolerance (maximum fidelity protection)."""

    @pytest.mark.parametrize(
        "material,restorability",
        [
            ("vinyl", 45.0),
            ("cd_digital", 90.0),
            ("cassette", 60.0),
        ],
    )
    def test_p1p2_tighter_tolerance(self, material: str, restorability: float) -> None:
        """P1/P2-heavy song must have tolerance ≤ uniform tolerance."""
        fc_uniform = _make_feedback_chain(_uniform_weights(), material=material, restorability=restorability)
        fc_p1p2 = _make_feedback_chain(_p1p2_heavy_weights(1.5), material=material, restorability=restorability)

        tol_uniform = fc_uniform._compute_adaptive_mos_regression_tolerance()
        tol_p1p2 = fc_p1p2._compute_adaptive_mos_regression_tolerance()
        assert tol_p1p2 <= tol_uniform + 1e-9, (
            f"P1/P2-heavy MOS tolerance ({tol_p1p2:.4f}) must be ≤ uniform ({tol_uniform:.4f}) "
            f"for material={material}, rest={restorability}"
        )

    def test_tolerance_always_positive(self) -> None:
        """MOS tolerance must always remain > 0 even with extreme weights."""
        for gw, mat, rest in [
            (None, "shellac", 15.0),
            (_uniform_weights(), "shellac", 15.0),
            (_p1p2_heavy_weights(2.0), "shellac", 15.0),
            (_p4p5_heavy_weights(2.0), "shellac", 15.0),
        ]:
            fc = _make_feedback_chain(gw, material=mat, restorability=rest)
            tol = fc._compute_adaptive_mos_regression_tolerance()
            assert tol > 0.0, f"MOS tolerance must be > 0, got {tol} for goal_weights={gw}"

    def test_tolerance_sane_range(self) -> None:
        """MOS tolerance must stay within [0.03, 0.25] — spec sane range."""
        for gw in (None, _p1p2_heavy_weights(2.0), _p4p5_heavy_weights(2.0)):
            for mat in ("cd_digital", "vinyl", "shellac"):
                fc = _make_feedback_chain(gw, material=mat, restorability=50.0)
                tol = fc._compute_adaptive_mos_regression_tolerance()
                assert 0.03 <= tol <= 0.25, (
                    f"MOS tolerance {tol:.4f} out of sane range [0.03, 0.25] for {mat} with weights={gw}"
                )


# ---------------------------------------------------------------------------
# §2.56 Convergence Delta Invariant
# ---------------------------------------------------------------------------


class TestConvergenceDeltaContract:
    """At high MOS, P1/P2-heavy songs must have tighter convergence delta."""

    def test_p1p2_tighter_delta_at_high_mos(self) -> None:
        """At MOS ≥ 4.0, P1/P2 > 1.1 must tighten convergence delta."""
        fc_uniform = _make_feedback_chain(_uniform_weights())
        fc_p1p2 = _make_feedback_chain(_p1p2_heavy_weights(1.5))

        delta_uniform = fc_uniform._adaptive_convergence_delta(current_mos=4.2)
        delta_p1p2 = fc_p1p2._adaptive_convergence_delta(current_mos=4.2)
        assert delta_p1p2 < delta_uniform, (
            f"P1/P2-heavy convergence delta ({delta_p1p2:.6f}) must be < uniform ({delta_uniform:.6f}) at MOS 4.2"
        )

    def test_no_tightening_at_low_mos(self) -> None:
        """At MOS < 4.0, convergence delta must NOT be tightened by goal_weights."""
        fc_uniform = _make_feedback_chain(_uniform_weights())
        fc_p1p2 = _make_feedback_chain(_p1p2_heavy_weights(1.8))

        delta_uniform = fc_uniform._adaptive_convergence_delta(current_mos=3.5)
        delta_p1p2 = fc_p1p2._adaptive_convergence_delta(current_mos=3.5)
        # At low MOS goal_weights do not tighten — allow marginal floating-point diff only
        assert abs(delta_p1p2 - delta_uniform) < 1e-9, (
            f"At MOS 3.5, convergence delta must be unaffected by goal_weights: "
            f"uniform={delta_uniform:.8f}, p1p2={delta_p1p2:.8f}"
        )

    def test_delta_strictly_positive(self) -> None:
        """Convergence delta must always remain > 0 (pipeline must not stall)."""
        for gw in (None, _uniform_weights(), _p1p2_heavy_weights(2.0)):
            fc = _make_feedback_chain(gw)
            for mos in (2.0, 3.0, 4.0, 4.8):
                d = fc._adaptive_convergence_delta(current_mos=mos)
                assert d > 0.0, f"Convergence delta must be > 0 at MOS {mos}, got {d}"

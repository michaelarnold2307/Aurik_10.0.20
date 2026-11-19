from __future__ import annotations

"""
Tests for §2.47b JND Sub-Threshold Filter in PerPhaseMusicalGoalsGate.

Normative reference:
- §2.47b: Phase with all goal-deltas ≥ 0 AND all < JND → "sub_threshold" action.
- No retry, no rollback when sub_threshold.
- Regression (any delta < 0) → normal PMGG-Retry-Logik, NOT sub_threshold.
- "sub_threshold_phases" list in metadata is populated.
"""


import numpy as np
import pytest

from backend.core.per_phase_musical_goals_gate import (
    FAST_GOALS_SUBSET,
    JND_MIN_DELTA,
)

SR = 48000


@pytest.mark.unit
class TestJNDMinDeltaConstants:
    """JND_MIN_DELTA dict must contain normative values for all 15 goals."""

    EXPECTED_GOALS = {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "transient_energie",
        "emotionalitaet",
        "micro_dynamics",
        "groove",
        "transparenz",
        "waerme",
        "bass_kraft",
        "separation_fidelity",
        "brillanz",
        "spatial_depth",
    }

    def test_all_15_goals_present(self):
        for goal in self.EXPECTED_GOALS:
            assert goal in JND_MIN_DELTA, f"Missing JND entry for '{goal}'"

    def test_tonal_center_jnd_is_smallest_or_near_smallest(self):
        """tonal_center is most sensitive — should have one of the lowest JND thresholds."""
        # tonal_center ≤ natuerlichkeit and ≤ waerme (both have higher perceptual JND)
        assert JND_MIN_DELTA["tonal_center"] <= JND_MIN_DELTA["natuerlichkeit"]
        assert JND_MIN_DELTA["tonal_center"] <= JND_MIN_DELTA["waerme"]

    def test_spatial_depth_jnd_is_largest_or_near_largest(self):
        """spatial_depth least JND-sensitive — should have one of the highest thresholds."""
        assert JND_MIN_DELTA["spatial_depth"] >= JND_MIN_DELTA["natuerlichkeit"]
        assert JND_MIN_DELTA["spatial_depth"] >= JND_MIN_DELTA["tonal_center"]

    def test_all_values_positive(self):
        for goal, val in JND_MIN_DELTA.items():
            assert val > 0.0, f"JND for '{goal}' must be > 0"

    def test_all_values_below_one(self):
        """JND values are fractions, not raw scores [0,1]."""
        for goal, val in JND_MIN_DELTA.items():
            assert val < 1.0, f"JND for '{goal}' should be < 1.0, got {val}"

    def test_fast_goals_subset_all_covered(self):
        """All goals in FAST_GOALS_SUBSET must have a JND entry."""
        for goal in FAST_GOALS_SUBSET:
            assert goal in JND_MIN_DELTA, (
                f"FAST_GOALS_SUBSET goal '{goal}' has no JND_MIN_DELTA entry — "
                "§2.47b requires all measured goals to have a JND threshold"
            )


class TestJNDSubThresholdLogic:
    """Tests for the sub_threshold decision logic itself (unit tests of the boolean condition)."""

    def _all_below_jnd(self, deltas: dict[str, float]) -> bool:
        """Mirror of the JND check in _run_with_retry."""
        return all(d >= 0.0 for d in deltas.values()) and all(
            abs(d) < JND_MIN_DELTA.get(g, 0.015) for g, d in deltas.items()
        )

    def test_all_zero_deltas_is_sub_threshold(self):
        deltas = dict.fromkeys(FAST_GOALS_SUBSET, 0.0)
        assert self._all_below_jnd(deltas)

    def test_tiny_positive_deltas_is_sub_threshold(self):
        """Deltas of +0.001 for all goals → sub-threshold."""
        deltas = dict.fromkeys(FAST_GOALS_SUBSET, 0.001)
        assert self._all_below_jnd(deltas)

    def test_one_large_positive_not_sub_threshold(self):
        """If natuerlichkeit delta = +0.05 (> JND 0.012) → not sub-threshold."""
        deltas = dict.fromkeys(FAST_GOALS_SUBSET, 0.001)
        deltas["natuerlichkeit"] = 0.05  # exceeds JND (0.012)
        assert not self._all_below_jnd(deltas)

    def test_one_negative_delta_not_sub_threshold(self):
        """Any negative delta → must NOT be sub-threshold."""
        deltas = dict.fromkeys(FAST_GOALS_SUBSET, 0.001)
        deltas["natuerlichkeit"] = -0.001  # regression
        assert not self._all_below_jnd(deltas)

    def test_exactly_at_jnd_boundary_not_sub_threshold(self):
        """Delta exactly equal to JND is NOT strictly below → not sub-threshold."""
        deltas = dict.fromkeys(FAST_GOALS_SUBSET, 0.001)
        # tonal_center JND = 0.008 (Krumhansl 1990) → delta = 0.008 is NOT < JND
        deltas["tonal_center"] = JND_MIN_DELTA["tonal_center"]
        assert not self._all_below_jnd(deltas)

    def test_just_below_jnd_boundary_is_sub_threshold(self):
        """Delta just below JND for all goals → sub-threshold."""
        deltas = {g: JND_MIN_DELTA[g] - 0.0001 for g in FAST_GOALS_SUBSET}
        assert self._all_below_jnd(deltas)

    def test_mixed_positive_and_negative_not_sub_threshold(self):
        """Any negative delta blocks sub-threshold regardless of other goals."""
        deltas = dict.fromkeys(FAST_GOALS_SUBSET, 0.001)
        deltas["groove"] = -0.005
        deltas["brillanz"] = 0.015  # 0.015 < JND(brillanz)=0.016 → fine alone, but groove blocks it
        assert not self._all_below_jnd(deltas)  # groove regression blocks it


class TestJNDSpecificGoalThresholds:
    """Verify specific normative JND threshold values from §2.47b spec."""

    def test_tonal_center_jnd_is_0_008(self):
        # Krumhansl (1990) + Temperley (2001): key very salient in tonal vocal music
        assert JND_MIN_DELTA["tonal_center"] == pytest.approx(0.008, abs=1e-6)

    def test_spatial_depth_jnd_is_0_018(self):
        # Blauert (1997) + Griesinger (1997): reverb JND in music reproduction
        assert JND_MIN_DELTA["spatial_depth"] == pytest.approx(0.018, abs=1e-6)

    def test_natuerlichkeit_jnd_is_0_012(self):
        # Moore (1977) + Caclin et al. (2005): spectral-complex/timbral JND in music ≈1 %
        assert JND_MIN_DELTA["natuerlichkeit"] == pytest.approx(0.012, abs=1e-6)

    def test_brillanz_jnd_is_0_016(self):
        # Schubert et al. (2004) + Moore (2012): HF brightness JND ≈1 dB above 6 kHz
        assert JND_MIN_DELTA["brillanz"] == pytest.approx(0.016, abs=1e-6)

    def test_artikulation_jnd_is_0_010(self):
        # London (2004) + Repp (2005): rhythmic timing JND ~8–10 ms in music
        assert JND_MIN_DELTA["artikulation"] == pytest.approx(0.010, abs=1e-6)


class TestSubThresholdWrapPhaseIntegration:
    """Integration: wrap_phase propagates sub_threshold into log_entry metadata."""

    def test_log_entry_has_sub_threshold_marker(self, monkeypatch):
        """wrap_phase must put phase_id in metadata['sub_threshold_phases'] when sub_threshold."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        gate = PerPhaseMusicalGoalsGate()
        audio = (np.random.default_rng(0).standard_normal(SR * 2) * 0.3).astype(np.float32)

        class _MockPhase:
            phase_id = "phase_test_jnd"

            def process(self, aud, **kwargs):
                from backend.core.phases.phase_interface import create_phase_result

                return create_phase_result(audio=aud.copy(), modifications={}, warnings=[], metadata={})

        _base = dict.fromkeys(FAST_GOALS_SUBSET, 0.8)
        _tiny = dict.fromkeys(FAST_GOALS_SUBSET, 0.8 + 0.001)  # all +0.001 < JND

        call_count = [0]

        def _mock_measure(sample, sr, reference=None):
            r = _tiny if call_count[0] > 0 else _base
            call_count[0] += 1
            return r

        monkeypatch.setattr("backend.core.per_phase_musical_goals_gate._measure_quick", _mock_measure)
        monkeypatch.setattr(
            "backend.core.per_phase_musical_goals_gate._extract_sample",
            lambda a, sr, **kw: a[:SR],
        )
        # Patch content-integrity to return no penalty
        monkeypatch.setattr(
            "backend.core.per_phase_musical_goals_gate._content_integrity_penalty",
            lambda *a, **kw: (0.0, {}),
        )

        audio_out, scores_after, log_entry = gate.wrap_phase(
            _MockPhase(),
            audio=audio,
            sr=SR,
            scores_before=_base,
            phase_id="phase_test_jnd",
        )
        assert log_entry.action == "sub_threshold", f"Expected 'sub_threshold', got '{log_entry.action}'"
        assert "sub_threshold_phases" in log_entry.metadata
        assert "phase_test_jnd" in log_entry.metadata["sub_threshold_phases"]

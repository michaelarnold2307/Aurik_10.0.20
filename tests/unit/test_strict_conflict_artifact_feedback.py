from __future__ import annotations

"""Unit tests for strict conflict adaptation from artifact-related phase events."""


import types

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


def _profile() -> dict:
    return {
        "family_scalars": {
            "general": 1.0,
            "denoise": 1.0,
            "reconstruction": 1.0,
            "time_pitch_transport": 1.0,
        },
        "strict_conflict_policy": {
            "rollback_decay_per_family": {
                "general": 0.96,
                "denoise": 0.90,
                "reconstruction": 0.93,
                "time_pitch_transport": 0.93,
            },
            "rollback_decay_floor": 0.55,
            "phase_strength_caps": {
                "phase_03_denoise": 0.80,
            },
        },
    }


@pytest.mark.unit
class TestStrictConflictArtifactFeedback:
    """Artifact-related conflicts must tighten future phase intervention online."""

    def test_artifact_rollback_decays_matching_family_scalar(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "artifact_freedom_rollback",
            {"artifact_freedom": 0.83},
        )

        family_scalars = restorer._song_calibration_profile["family_scalars"]
        # artifact_freedom=0.83 -> severity boost (weight 1.2) => decay 0.88
        assert family_scalars["denoise"] == pytest.approx(0.88, abs=1e-9)
        runtime = restorer._phase_goal_conflict_runtime
        assert runtime["by_family"]["denoise"] == 1
        assert runtime["by_phase"]["phase_03_denoise"] == 1
        assert runtime["events"][-1]["reason"] == "artifact_freedom_rollback"

    def test_repeated_artifact_events_tighten_phase_cap(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "artifact_freedom_rollback",
            {"artifact_freedom": 0.84},
        )
        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "noise_texture_rollback",
            {"noise_texture_deviation_db_oct": 16.0},
        )

        profile = restorer._song_calibration_profile
        # 1st rollback severity-weighted, 2nd rollback base decay => 0.7935
        assert profile["family_scalars"]["denoise"] == pytest.approx(0.7935, abs=1e-9)
        assert profile["strict_conflict_policy"]["phase_strength_caps"]["phase_03_denoise"] == pytest.approx(
            0.736,
            abs=1e-9,
        )

    def test_unknown_phase_uses_general_decay_without_crashing(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_27_click_pop_removal",
            "hf_hallucination_rescue",
            {"hf_delta_ratio": 0.21},
        )

        assert restorer._song_calibration_profile["family_scalars"]["general"] == pytest.approx(0.96, abs=1e-9)

    def test_time_pitch_conflict_decays_time_pitch_family(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_12_wow_flutter_fix",
            "vocal_no_harm_rollback",
            {"reason": "formant_shift"},
        )

        profile = restorer._song_calibration_profile
        assert profile["family_scalars"]["time_pitch_transport"] == pytest.approx(0.93, abs=1e-9)
        assert profile["family_scalars"]["denoise"] == pytest.approx(1.0, abs=1e-9)
        assert restorer._phase_goal_conflict_runtime["by_family"]["time_pitch_transport"] == 1

    def test_pmgg_best_effort_uses_milder_family_decay_weight(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()
        policy = restorer._song_calibration_profile["strict_conflict_policy"]
        policy["reason_decay_weight"] = {"pmgg_best_effort": 0.55}

        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r1"},
        )

        # Base denoise decay 0.90 -> attenuation 0.10. With weight 0.55 => decay 0.945.
        assert restorer._song_calibration_profile["family_scalars"]["denoise"] == pytest.approx(0.945, abs=1e-9)

    def test_pmgg_best_effort_uses_milder_cap_tightening_weight(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()
        policy = restorer._song_calibration_profile["strict_conflict_policy"]
        policy["reason_cap_tighten_weight"] = {"pmgg_best_effort": 0.25}

        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r1"},
        )
        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )

        # Base cap multiplier 0.92, weighted with 0.25 => 0.98.
        assert policy["phase_strength_caps"]["phase_03_denoise"] == pytest.approx(0.784, abs=1e-9)

    def test_artifact_freedom_severity_scales_decay_strength(self):
        mild = UnifiedRestorerV3()
        mild._song_calibration_profile = _profile()
        severe = UnifiedRestorerV3()
        severe._song_calibration_profile = _profile()

        mild._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "artifact_freedom_rollback",
            {"artifact_freedom": 0.94},
        )
        severe._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "artifact_freedom_rollback",
            {"artifact_freedom": 0.80},
        )

        mild_scalar = float(mild._song_calibration_profile["family_scalars"]["denoise"])
        severe_scalar = float(severe._song_calibration_profile["family_scalars"]["denoise"])
        assert severe_scalar < mild_scalar

    def test_pmgg_best_effort_emergency_stricter_than_regular_best_effort(self):
        regular = UnifiedRestorerV3()
        regular._song_calibration_profile = _profile()
        emergency = UnifiedRestorerV3()
        emergency._song_calibration_profile = _profile()

        regular._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        emergency._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_emergency"},
        )

        regular_scalar = float(regular._song_calibration_profile["family_scalars"]["denoise"])
        emergency_scalar = float(emergency._song_calibration_profile["family_scalars"]["denoise"])
        assert emergency_scalar < regular_scalar

    def test_goal_priority_p1_regression_stricter_than_p5_at_same_magnitude(self):
        p1 = UnifiedRestorerV3()
        p1._song_calibration_profile = _profile()
        p5 = UnifiedRestorerV3()
        p5._song_calibration_profile = _profile()

        p1._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {
                "action": "best_effort_r2",
                "goal_regressions": {"natuerlichkeit": -0.08},
            },
        )
        p5._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {
                "action": "best_effort_r2",
                "goal_regressions": {"spatial_depth": -0.08},
            },
        )

        p1_scalar = float(p1._song_calibration_profile["family_scalars"]["denoise"])
        p5_scalar = float(p5._song_calibration_profile["family_scalars"]["denoise"])
        assert p1_scalar < p5_scalar

    def test_vocal_presence_strengthens_pmgg_best_effort_penalty_for_vocal_goals(self):
        non_vocal = UnifiedRestorerV3()
        non_vocal._song_calibration_profile = _profile()
        vocal = UnifiedRestorerV3()
        vocal._song_calibration_profile = _profile()
        vocal._restoration_context = {"panns_singing": 0.85}

        details = {
            "action": "best_effort_r2",
            "goal_regressions": {"natuerlichkeit": -0.08},
        }
        non_vocal._register_phase_goal_conflict_event("phase_03_denoise", "pmgg_best_effort", details)
        vocal._register_phase_goal_conflict_event("phase_03_denoise", "pmgg_best_effort", details)

        non_vocal_scalar = float(non_vocal._song_calibration_profile["family_scalars"]["denoise"])
        vocal_scalar = float(vocal._song_calibration_profile["family_scalars"]["denoise"])
        assert vocal_scalar < non_vocal_scalar

    def test_vocal_presence_strengthens_vocal_no_harm_conflict_penalty(self):
        low = UnifiedRestorerV3()
        low._song_calibration_profile = _profile()
        high = UnifiedRestorerV3()
        high._song_calibration_profile = _profile()
        high._restoration_context = {"panns_singing": 0.90}

        details = {"checks": {"singer_identity": False}}
        low._register_phase_goal_conflict_event("phase_42_vocal_enhancement", "vocal_no_harm_rollback", details)
        high._register_phase_goal_conflict_event("phase_42_vocal_enhancement", "vocal_no_harm_rollback", details)

        low_scalar = float(low._song_calibration_profile["family_scalars"]["vocal"])
        high_scalar = float(high._song_calibration_profile["family_scalars"]["vocal"])
        assert high_scalar < low_scalar

        last_event = high._phase_goal_conflict_runtime["events"][-1]
        assert "severity_score" in last_event
        assert "severity_bucket" in last_event
        assert "severity_fingerprint" in last_event
        assert isinstance(last_event["severity_fingerprint"], dict)
        assert "vocal_presence" in last_event["severity_fingerprint"]

    def test_strict_conflict_report_detects_negative_goal_regressions(self):
        entry = types.SimpleNamespace(
            phase_id="phase_03_denoise",
            action="best_effort_r1",
            goal_regressions={"natuerlichkeit": -0.06, "groove": -0.02},
        )

        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        events = report.get("regressive_phase_events", [])
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["phase_id"] == "phase_03_denoise"

    def test_strict_conflict_report_ignores_tolerated_pass_action(self):
        entry = types.SimpleNamespace(
            phase_id="phase_29_tape_hiss_reduction",
            action="passed_p4p5_tolerated",
            goal_regressions={"natuerlichkeit": -0.08},
            metadata={"pmgg_decision_class": "pass", "pmgg_decision_reason": "priority_tolerance_band_accept"},
        )

        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        events = report.get("regressive_phase_events", [])
        assert isinstance(events, list)
        assert len(events) == 0

    def test_strict_conflict_report_legacy_pass_actions_are_ignored(self):
        entry = types.SimpleNamespace(
            phase_id="phase_31_speed_pitch_correction",
            action="passthrough",
            goal_regressions={"groove": -0.09},
        )

        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        events = report.get("regressive_phase_events", [])
        assert isinstance(events, list)
        assert len(events) == 0

    def test_strict_conflict_report_ignores_legacy_best_effort_accepted(self):
        entry = types.SimpleNamespace(
            phase_id="phase_03_denoise",
            action="best_effort_accepted",
            goal_regressions={"natuerlichkeit": -0.08},
            metadata={"pmgg_decision_class": "best_effort", "pmgg_decision_reason": "legacy_best_effort_accepted"},
        )

        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        events = report.get("regressive_phase_events", [])
        assert isinstance(events, list)
        assert len(events) == 0
        assert report.get("best_effort_phases", []) == []

    def test_strict_conflict_report_ignores_reason_based_pass_without_class(self):
        entry = types.SimpleNamespace(
            phase_id="phase_29_tape_hiss_reduction",
            action="legacy_unknown",
            goal_regressions={"natuerlichkeit": -0.09},
            pmgg_decision_reason="priority_tolerance_band_accept",
        )

        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        events = report.get("regressive_phase_events", [])
        assert isinstance(events, list)
        assert len(events) == 0

    def test_strict_conflict_report_ignores_reason_based_pass_from_metadata(self):
        entry = types.SimpleNamespace(
            phase_id="phase_31_speed_pitch_correction",
            action="best_effort_r2",
            goal_regressions={"groove": -0.07},
            metadata={"pmgg_decision_reason": "jnd_sub_threshold_accept"},
        )

        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        events = report.get("regressive_phase_events", [])
        assert isinstance(events, list)
        assert len(events) == 0

    def test_strict_conflict_report_weights_p1_regression_above_p5(self):
        p1_entry = types.SimpleNamespace(
            phase_id="phase_03_denoise",
            action="best_effort_r2",
            goal_regressions={"natuerlichkeit": -0.08},
        )
        p5_entry = types.SimpleNamespace(
            phase_id="phase_03_denoise",
            action="best_effort_r2",
            goal_regressions={"spatial_depth": -0.08},
        )

        p1_report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[p1_entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )
        p5_report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[p5_entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={},
        )

        assert float(p1_report.get("regressive_weight_sum", 0.0)) > float(p5_report.get("regressive_weight_sum", 0.0))
        assert float(p1_report.get("conflict_score", 0.0)) > float(p5_report.get("conflict_score", 0.0))

    def test_strict_conflict_report_weights_vocal_presence_for_vocal_goal_regression(self):
        entry = types.SimpleNamespace(
            phase_id="phase_42_vocal_enhancement",
            action="best_effort_r2",
            goal_regressions={"natuerlichkeit": -0.08},
        )

        low_vocal = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={"vocal_presence": 0.0},
        )
        high_vocal = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[entry],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={},
            musical_goals_passed={},
            song_calibration_profile={"vocal_presence": 0.9},
        )

        assert float(high_vocal.get("regressive_weight_sum", 0.0)) > float(low_vocal.get("regressive_weight_sum", 0.0))
        assert float(high_vocal.get("conflict_score", 0.0)) > float(low_vocal.get("conflict_score", 0.0))

    def test_strict_conflict_report_contains_runtime_severity_aggregation(self):
        report = UnifiedRestorerV3._build_strict_phase_goal_conflict_report(
            phase_gate_entries=[],
            phase_regression_log={},
            interaction_guard_meta={},
            runtime_conflict_state={
                "events": [
                    {"phase_id": "phase_03_denoise", "severity_score": 1.2},
                    {"phase_id": "phase_42_vocal_enhancement", "severity_score": 1.5},
                ]
            },
            musical_goals_passed={},
            song_calibration_profile={},
        )

        assert float(report.get("runtime_severity_sum", 0.0)) == pytest.approx(2.7, abs=1e-9)
        assert float(report.get("runtime_severity_max", 0.0)) == pytest.approx(1.5, abs=1e-9)

    def test_soft_conflict_streak_applies_fatigue_rebound_for_pmgg_best_effort(self):
        baseline = UnifiedRestorerV3()
        baseline._song_calibration_profile = _profile()
        baseline._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        baseline_scalar = float(baseline._song_calibration_profile["family_scalars"]["denoise"])

        rebound = UnifiedRestorerV3()
        rebound._song_calibration_profile = _profile()
        rebound._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "low",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 3},
            "by_family": {"denoise": 3},
            "total": 3,
        }
        rebound._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        rebound_scalar = float(rebound._song_calibration_profile["family_scalars"]["denoise"])
        assert rebound_scalar > baseline_scalar

        ev = rebound._phase_goal_conflict_runtime["events"][-1]
        fp = ev.get("severity_fingerprint", {})
        assert fp.get("fatigue_rebound_applied") is True
        assert int(fp.get("prior_soft_conflict_streak", 0)) >= 3

    def test_recent_hard_conflict_disables_fatigue_rebound(self):
        baseline = UnifiedRestorerV3()
        baseline._song_calibration_profile = _profile()
        baseline._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        baseline_scalar = float(baseline._song_calibration_profile["family_scalars"]["denoise"])

        guarded = UnifiedRestorerV3()
        guarded._song_calibration_profile = _profile()
        guarded._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "low",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "artifact_freedom_rollback",
                    "severity_bucket": "high",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 4},
            "by_family": {"denoise": 4},
            "total": 4,
        }
        guarded._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        guarded_scalar = float(guarded._song_calibration_profile["family_scalars"]["denoise"])
        assert guarded_scalar == pytest.approx(baseline_scalar, abs=1e-9)

        ev = guarded._phase_goal_conflict_runtime["events"][-1]
        fp = ev.get("severity_fingerprint", {})
        assert fp.get("fatigue_rebound_applied") is False

    def test_reason_disagreement_brake_softens_pmgg_best_effort_penalty(self):
        baseline = UnifiedRestorerV3()
        baseline._song_calibration_profile = _profile()
        baseline._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        baseline_scalar = float(baseline._song_calibration_profile["family_scalars"]["denoise"])

        disagree = UnifiedRestorerV3()
        disagree._song_calibration_profile = _profile()
        disagree._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "transport_motion_reduction",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "phase_interaction_rollback",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 3},
            "by_family": {"denoise": 3},
            "total": 3,
        }
        disagree._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        disagree_scalar = float(disagree._song_calibration_profile["family_scalars"]["denoise"])

        assert disagree_scalar > baseline_scalar
        ev = disagree._phase_goal_conflict_runtime["events"][-1]
        fp = ev.get("severity_fingerprint", {})
        assert fp.get("disagreement_brake_applied") is True
        assert int(fp.get("recent_reason_diversity", 0)) >= 3

    def test_reason_disagreement_brake_is_disabled_when_recent_hard_reason_exists(self):
        baseline = UnifiedRestorerV3()
        baseline._song_calibration_profile = _profile()
        baseline._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        baseline_scalar = float(baseline._song_calibration_profile["family_scalars"]["denoise"])

        guarded = UnifiedRestorerV3()
        guarded._song_calibration_profile = _profile()
        guarded._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "transport_motion_reduction",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "artifact_freedom_rollback",
                    "severity_bucket": "high",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "phase_interaction_rollback",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 4},
            "by_family": {"denoise": 4},
            "total": 4,
        }
        guarded._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        guarded_scalar = float(guarded._song_calibration_profile["family_scalars"]["denoise"])

        assert guarded_scalar == pytest.approx(baseline_scalar, abs=1e-9)
        ev = guarded._phase_goal_conflict_runtime["events"][-1]
        fp = ev.get("severity_fingerprint", {})
        assert fp.get("disagreement_brake_applied") is False

    def test_disagreement_guard_uses_material_era_adaptive_thresholds(self):
        baseline = UnifiedRestorerV3()
        baseline._song_calibration_profile = _profile()
        baseline._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "phase_interaction_rollback",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 2},
            "by_family": {"denoise": 2},
            "total": 2,
        }
        baseline._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        baseline_scalar = float(baseline._song_calibration_profile["family_scalars"]["denoise"])

        adaptive = UnifiedRestorerV3()
        adaptive._song_calibration_profile = {
            **_profile(),
            "material": "shellac",
            "era_decade": 1940,
        }
        adaptive._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "phase_interaction_rollback",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 2},
            "by_family": {"denoise": 2},
            "total": 2,
        }
        adaptive._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        adaptive_scalar = float(adaptive._song_calibration_profile["family_scalars"]["denoise"])

        assert adaptive_scalar > baseline_scalar
        ev = adaptive._phase_goal_conflict_runtime["events"][-1]
        fp = ev.get("severity_fingerprint", {})
        assert fp.get("disagreement_brake_applied") is True
        assert int(fp.get("disagreement_threshold", 0)) == 2
        assert int(fp.get("disagreement_window", 0)) >= 5

    def test_disagreement_guard_is_more_conservative_for_vocal_dominant_material(self):
        baseline = UnifiedRestorerV3()
        baseline._song_calibration_profile = {
            **_profile(),
            "material": "shellac",
            "era_decade": 1940,
        }
        baseline._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "phase_interaction_rollback",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "transport_motion_reduction",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 3},
            "by_family": {"denoise": 3},
            "total": 3,
        }
        baseline._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        baseline_scalar = float(baseline._song_calibration_profile["family_scalars"]["denoise"])

        vocal = UnifiedRestorerV3()
        vocal._song_calibration_profile = {
            **_profile(),
            "material": "shellac",
            "era_decade": 1940,
            "vocal_presence": 0.85,
        }
        vocal._phase_goal_conflict_runtime = {
            "events": [
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "pmgg_best_effort",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "phase_interaction_rollback",
                    "severity_bucket": "medium",
                },
                {
                    "phase_id": "phase_03_denoise",
                    "family": "denoise",
                    "reason": "transport_motion_reduction",
                    "severity_bucket": "medium",
                },
            ],
            "by_phase": {"phase_03_denoise": 3},
            "by_family": {"denoise": 3},
            "total": 3,
        }
        vocal._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "pmgg_best_effort",
            {"action": "best_effort_r2"},
        )
        vocal_scalar = float(vocal._song_calibration_profile["family_scalars"]["denoise"])

        assert vocal_scalar < baseline_scalar
        ev = vocal._phase_goal_conflict_runtime["events"][-1]
        fp = ev.get("severity_fingerprint", {})
        assert fp.get("disagreement_brake_applied") is False
        assert int(fp.get("disagreement_threshold", 0)) == 4

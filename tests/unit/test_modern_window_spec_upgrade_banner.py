from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt5")

from Aurik10.ui.modern_window import ModernMainWindow


def _mock_window():
    """Create a mock window with the methods _build_quality_banner_sections needs."""
    m = MagicMock()
    m._build_recovery_diagnostic_line.return_value = (
        "🧾  Recovery-Diagnose: Urteil=UPGRADE  ·  Pipeline=ok  ·  Export=vorhanden  ·  A/B=—"
    )
    return m


def _base_kwargs() -> dict:
    return {
        "is_sota_run": True,
        "sota_warning_reason": "",
        "fail_reason": "",
        "degradation_status": "ok",
        "runtime_fallback_original": False,
        "primary_error_code": "",
        "phases_exec_count": 0,
        "phases_skip_count": 0,
        "total_time_s": 0.0,
        "rt_factor": 0.0,
        "top_causal_cause": "",
        "causal_conf": 0.0,
        "quality_before_score": 0.0,
        "quality_after_score": 0.0,
        "quality_delta": 0.0,
        "delta_snr": 0.0,
        "temporal_coh_score": 0.0,
        "emotional_arc_score": 0.0,
        "feedback_retries": 0,
        "feedback_chain_score": 0.0,
        "excellence_steps": [],
        "genre_is_schlager": False,
        "genre_accordion": 0.0,
        "genre_bpm": 0.0,
        "genre_key": "",
        "pipeline_hint": "",
        "pipeline_tier": "",
        "joy_index": 0.0,
        "fatigue_index": 0.0,
        "frisson_index": 0.0,
        "cluster_key": "",
        "cluster_policy": {},
        "auto_recommendations": [],
        "musical_violations": [],
        "phase_gate_notes": [],
        "ceiling_reached": False,
        "xp_recovery_certainty": {},
        "xp_hf_guard": {},
        "xp_tilt_guard": {},
        "xp_carrier_ratio": 0.0,
        "xp_carrier_ref_shifted": False,
        "xp_fqf": {},
        "xp_quality_gate": {},
        "xp_threshold_evidence": {},
        "xp_user_guidance": {},
        "xp_quality_scale": {},
        "xp_ml_fallbacks": [],
        "xp_team_coord": {},
        "xp_user_confidence": {},
        "preventive_actions": [],
        "spec_upgrade": False,
        "goal_upgrade_decision": {},
        "phase_upgrade_candidate": [],
        "goal_threshold_source": "",
        "decision_authority": "",
    }


@pytest.mark.unit
def test_spec_upgrade_banner_promoted_is_visible() -> None:
    args = _base_kwargs()
    args.update(
        {
            "spec_upgrade": True,
            "goal_upgrade_decision": {
                "reason": "promote_to_spec",
                "improved_goals_count": 2,
                "non_degraded_goals_count": 15,
                "degraded_goals_count": 0,
            },
            "phase_upgrade_candidate": ["phase_03_denoise", "phase_29_tape_hiss_reduction"],
            "goal_threshold_source": "pmgg_ceiling_capped_targets",
            "decision_authority": "uv3_final_gate",
        }
    )

    banner = ModernMainWindow._build_quality_banner_sections(_mock_window(), **args)
    joined = "\n".join(banner)

    assert "Spec-Upgrade: Upgrade freigegeben" in joined
    assert "Kandidat-Phasen: phase_03_denoise, phase_29_tape_hiss_reduction" in joined
    assert "Quelle: pmgg_ceiling_capped_targets" in joined
    assert "Authority: uv3_final_gate" in joined


@pytest.mark.unit
def test_spec_upgrade_banner_rejection_reason_is_visible() -> None:
    args = _base_kwargs()
    args.update(
        {
            "spec_upgrade": False,
            "goal_upgrade_decision": {
                "reason": "vqi_regression_or_missing",
                "improved_goals_count": 1,
                "non_degraded_goals_count": 15,
                "degraded_goals_count": 0,
            },
        }
    )

    banner = ModernMainWindow._build_quality_banner_sections(_mock_window(), **args)
    joined = "\n".join(banner)

    assert "Spec-Upgrade: abgelehnt (VQI fehlt/Regression)" in joined


@pytest.mark.unit
def test_quality_score_header_shows_green_spec_upgrade_ampel() -> None:
    txt = ModernMainWindow._build_quality_score_text(
        _mock_window(),
        is_sota_run=True,
        sota_warning_reason="",
        spec_upgrade=True,
        goal_upgrade_decision={"reason": "promote_to_spec"},
        mos_est=4.3,
        restorability_grade="A",
        restorability_mos_min=4.0,
        restorability_mos_max=4.8,
        mushra_score=82.0,
        mushra_grade="Gut",
        mushra_itu="A",
        era_label_full="1980er",
        era_label="1980",
        era_conf=0.9,
        genre_label="pop",
        genre_bpm=120.0,
        genre_key="C",
        pipeline_confidence=0.91,
        output_path="/tmp/out.wav",
    )
    assert "Spec-Upgrade: 🟢 promoted  ·  promote_to_spec" in txt


@pytest.mark.unit
def test_quality_score_header_hides_ampel_when_not_evaluated() -> None:
    txt = ModernMainWindow._build_quality_score_text(
        _mock_window(),
        is_sota_run=True,
        sota_warning_reason="",
        spec_upgrade=False,
        goal_upgrade_decision={"reason": "not_evaluated"},
        mos_est=3.8,
        restorability_grade="B",
        restorability_mos_min=3.3,
        restorability_mos_max=4.2,
        mushra_score=70.0,
        mushra_grade="Mittel",
        mushra_itu="B",
        era_label_full="1970er",
        era_label="1970",
        era_conf=0.8,
        genre_label="rock",
        genre_bpm=96.0,
        genre_key="Am",
        pipeline_confidence=0.88,
        output_path="/tmp/out.wav",
    )
    assert "Spec-Upgrade:" not in txt

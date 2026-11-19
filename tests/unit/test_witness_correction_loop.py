"""Tests für H3: Witness-Veto → global_scalar/family_scalars (§V7 (copilot-instructions.md), todo t11).

Invarianten: delta-basiert (nur Regressionen über JND zählen),
Clamping [0.50, 1.50] / Familie [0.5, 1.5], deterministisch, clean = no-op.
"""

from __future__ import annotations

from backend.core.dsp.witness_correction_loop import apply_witness_veto, compute_witness_veto
from backend.core.listening_witness import ListeningWitnessResult


def _profile() -> dict:
    return {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "enhancement": 1.0,
            "repair": 1.0,
            "compression": 1.0,
            "click_removal": 1.0,
        },
    }


def test_clean_witness_no_op() -> None:
    wit = ListeningWitnessResult(phase_id="test")
    res = compute_witness_veto(wit, "clean")
    assert res.applied is False
    prof = _profile()
    apply_witness_veto(prof, wit, "clean")
    assert prof["global_scalar"] == 1.0
    assert all(v == 1.0 for v in prof["family_scalars"].values())


def test_roughness_veto_reduces_denoise_family_and_global() -> None:
    wit = ListeningWitnessResult(phase_id="test", roughness_rise_asper=0.9)
    prof = _profile()
    res = apply_witness_veto(prof, wit, "denoise_stage")
    assert res.applied is True
    assert prof["family_scalars"]["denoise"] < 1.0
    assert prof["global_scalar"] < 1.0
    assert res.adjustments["denoise"] == 0.9


def test_pre_echo_veto_reduces_repair_family() -> None:
    wit = ListeningWitnessResult(phase_id="test", pre_echo_db=-6.0)  # > −12 → Befund
    prof = _profile()
    apply_witness_veto(prof, wit, "repair_stage")
    assert prof["family_scalars"]["repair"] == 0.9


def test_stereo_collapse_reduces_spatial_families() -> None:
    wit = ListeningWitnessResult(phase_id="test", iacc_drop=0.4)
    res = compute_witness_veto(wit, "stereo_stage")
    assert res.applied is True
    assert res.adjustments.get("enhancement") == 0.95
    assert res.adjustments.get("compression") == 0.95


def test_stereo_collapse_via_finding() -> None:
    wit = ListeningWitnessResult(phase_id="test", findings=["stereo_collapse"])
    res = compute_witness_veto(wit, "stereo_stage")
    assert res.applied is True


def test_clamping_floor() -> None:
    prof = _profile()
    for _ in range(30):
        wit = ListeningWitnessResult(phase_id="test", roughness_rise_asper=0.9, hnr_drop_db=5.0, pre_echo_db=-3.0)
        apply_witness_veto(prof, wit, "loop")
    assert prof["global_scalar"] >= 0.50
    for _k, v in prof["family_scalars"].items():
        assert v >= 0.5, f"Familie {_k} unter Floor: {v}"


def test_determinism() -> None:
    wit = ListeningWitnessResult(phase_id="test", roughness_rise_asper=0.5, pitch_drift_cents=20.0)
    r1 = compute_witness_veto(wit, "det")
    r2 = compute_witness_veto(wit, "det")
    assert r1.adjustments == r2.adjustments and r1.global_factor == r2.global_factor


def test_external_metric_veto_delta_based() -> None:
    from backend.core.dsp.witness_correction_loop import apply_external_metric_veto

    prof = _profile()
    # Kleine Regression unter Schwelle → kein Veto.
    r0 = apply_external_metric_veto(prof, "utmos", -0.1, 0.5, "utmos_gate")
    assert r0.applied is False and prof["global_scalar"] == 1.0
    # Klare Regression → enhancement-Familie + global_scalar reduziert.
    r1 = apply_external_metric_veto(prof, "utmos", -0.8, 0.5, "utmos_gate")
    assert r1.applied is True
    assert prof["family_scalars"]["enhancement"] == 0.9
    assert prof["global_scalar"] == 0.95
    assert r1.adjustments["enhancement"] == 0.90


def test_preference_veto_recording() -> None:
    from backend.core.dsp.witness_correction_loop import record_veto_to_preferences
    from backend.core.preference_learner import get_preference_learner

    learner = get_preference_learner()
    n_before = len(learner._history)
    ok = record_veto_to_preferences("feedback_chain", ["Rauigkeit", "Stereo-Kollaps"])
    assert ok is True
    assert len(learner._history) == n_before + 1
    assert learner._history[-1]["feedback"] == "sounds_artificial"
    assert learner._history[-1]["source"] == "witness_veto"

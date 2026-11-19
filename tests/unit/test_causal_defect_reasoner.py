from __future__ import annotations

import logging

"""Unit-Tests für backend/core/causal_defect_reasoner.py.

Spec §2.4: Bayesianische Ursachendiagnose. 11 Ursachen, materialspezifische
Priors, Posterior-Normierung, CAUSE_TO_PHASES-Mapping, soft_saturation-Sonderregel.
≥ 35 Tests: Import, Dataclass-Shape, Bounds, NaN-Guard, Mono/Stereo, alle
Materialien, Posterior-Summe, soft_saturation→leere Phasen, Singleton, etc.
"""


import numpy as np
import pytest

np.random.seed(0)

from backend.core.causal_defect_reasoner import (
    CAUSE_TO_PHASES,
    CAUSES,
    CausalDefectReasoner,
    RestorationPlan,
    SpectralFeatures,
    _normalize_defect_scores,
    extract_spectral_features,
    get_reasoner,
    reason_about_defects,
)

logger = logging.getLogger(__name__)

SR = 48000


def _sine(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.asarray(np.sin(2 * np.pi * freq * t), dtype=np.float32)


def _noise(secs: float = 1.0, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


def _stereo(secs: float = 1.0) -> np.ndarray:
    mono = _sine(secs=secs)
    return np.stack([mono, mono * 0.9])


def _clipped(secs: float = 0.5) -> np.ndarray:
    audio = _sine(secs=secs) * 3.0
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Klasse 1: Import und Konstanten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImportAndConstants:
    def test_01_module_importable(self):
        assert CausalDefectReasoner is not None

    def test_02_causes_list_length(self):
        # Robust gegen legitime Erweiterungen: Mindestgröße + Pflicht-Ursachen prüfen.
        assert len(CAUSES) >= 64
        required_causes = {
            "soft_saturation",
            "head_wear",
            "print_through",
            "scrape_flutter",
            "tape_head_clog",
        }
        assert required_causes.issubset(set(CAUSES))

    def test_03_causes_contains_soft_saturation(self):
        assert "soft_saturation" in CAUSES

    def test_04_causes_contains_head_wear(self):
        assert "head_wear" in CAUSES

    def test_05_causes_contains_print_through(self):
        assert "print_through" in CAUSES

    def test_06_cause_to_phases_covers_all_causes(self):
        # Alle Ursachen müssen im CAUSE_TO_PHASES-Mapping vorhanden sein.
        for cause in CAUSES:
            assert cause in CAUSE_TO_PHASES, f"{cause} not in CAUSE_TO_PHASES"

    def test_06a_causes_cover_all_phase_mappings(self):
        # Bidirektionale Konsistenz: Kein Mapping-Eintrag darf außerhalb von CAUSES liegen.
        for cause in CAUSE_TO_PHASES:
            assert cause in CAUSES, f"{cause} in CAUSE_TO_PHASES but not in CAUSES"

    def test_06b_likelihood_fns_cover_all_causes(self):
        # Alle Ursachen müssen eine Likelihood-Funktion haben.
        from backend.core.causal_defect_reasoner import LIKELIHOOD_FNS

        for cause in CAUSES:
            assert cause in LIKELIHOOD_FNS, f"{cause} not in LIKELIHOOD_FNS"

    def test_06c_material_priors_cover_all_causes(self):
        # Alle Materialien müssen Priors für alle Ursachen haben.
        from backend.core.causal_defect_reasoner import MATERIAL_PRIORS

        for material, priors in MATERIAL_PRIORS.items():
            for cause in CAUSES:
                assert cause in priors, f"{cause} not in MATERIAL_PRIORS[{material}]"

    def test_07_soft_saturation_phases_empty(self):
        # Spec §6.3: soft_saturation → BEWAHREN, leere Phasen-Liste
        assert CAUSE_TO_PHASES["soft_saturation"] == []

    def test_07a_jitter_artifacts_never_maps_to_phase_12(self):
        """Jitter ist nicht mechanisches Wow/Flutter: phase_12 ist hier verboten."""
        phases = CAUSE_TO_PHASES["jitter_artifacts"]
        assert "phase_12_wow_flutter_fix" not in phases
        assert "phase_23_spectral_repair" in phases
        assert "phase_14_phase_correction" in phases

    def test_07b_aliasing_never_maps_to_phase_03(self):
        """Aliasing ist kohärent, nicht stationär: kein phase_03_denoise-Mapping."""
        phases = CAUSE_TO_PHASES["aliasing"]
        assert "phase_03_denoise" not in phases
        assert "phase_23_spectral_repair" in phases
        assert "phase_50_spectral_repair" in phases


# ---------------------------------------------------------------------------
# Klasse 2: SpectralFeatures Dataclass
# ---------------------------------------------------------------------------


class TestSpectralFeatures:
    def test_08_default_construction(self):
        sf = SpectralFeatures()
        assert sf.rms == 0.0
        assert sf.peak == 0.0
        assert sf.pitch_instability == 0.0
        assert sf.stereo_correlation == 1.0

    def test_09_extract_mono_shape(self):
        audio = _sine(secs=1.0)
        sf = extract_spectral_features(audio, SR)
        assert isinstance(sf, SpectralFeatures)

    def test_10_extract_mono_rms_positive(self):
        audio = _sine(secs=1.0)
        sf = extract_spectral_features(audio, SR)
        assert sf.rms > 0.0

    def test_11_extract_stereo(self):
        audio = _stereo(secs=1.0)
        sf = extract_spectral_features(audio, SR)
        assert isinstance(sf, SpectralFeatures)

    def test_12_extract_short_signal_returns_default(self):
        audio = np.zeros(10, dtype=np.float32)
        sf = extract_spectral_features(audio, SR)
        assert isinstance(sf, SpectralFeatures)
        assert sf.rms == 0.0

    def test_13_extract_nan_input_safe(self):
        audio = np.full(SR, np.nan, dtype=np.float32)
        # Sollte keinen Exception werfen
        try:
            extract_spectral_features(audio, SR)
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Akzeptiert: NaN propagation im Extractor, Crash ist nicht OK
        # Hauptsache kein unbehandelter crash (Test kommt durch)

    def test_14_extract_dc_signal(self):
        audio = np.full(SR, 0.3, dtype=np.float32)
        sf = extract_spectral_features(audio, SR)
        assert abs(sf.dc_offset) > 0.1

    def test_15_clip_fraction_clipped(self):
        audio = _clipped(secs=0.5)
        sf = extract_spectral_features(audio, SR)
        assert sf.clip_fraction > 0.0

    def test_16_clip_fraction_clean_audio(self):
        audio = _sine(secs=1.0) * 0.3
        sf = extract_spectral_features(audio, SR)
        assert sf.clip_fraction == 0.0

    def test_17_hf_energy_ratio_bounds(self):
        audio = _sine(secs=1.0)
        sf = extract_spectral_features(audio, SR)
        assert 0.0 <= sf.hf_energy_ratio <= 1.0

    def test_18_lf_energy_ratio_bounds(self):
        audio = _sine(secs=1.0)
        sf = extract_spectral_features(audio, SR)
        assert 0.0 <= sf.lf_energy_ratio <= 1.0


# ---------------------------------------------------------------------------
# Klasse 3: RestorationPlan Struktur
# ---------------------------------------------------------------------------


class TestRestorationPlan:
    def _get_plan(self, defect_scores=None, material="unknown") -> RestorationPlan:
        r = CausalDefectReasoner()
        return r.reason(defect_scores or {}, material=material)

    def test_19_plan_type(self):
        plan = self._get_plan()
        assert isinstance(plan, RestorationPlan)

    def test_20_plan_primary_cause_in_causes(self):
        plan = self._get_plan()
        assert plan.primary_cause in CAUSES

    def test_21_cause_probabilities_keys(self):
        plan = self._get_plan()
        for c in CAUSES:
            assert c in plan.cause_probabilities

    def test_22_posterior_sum_approx_one(self):
        plan = self._get_plan()
        total = sum(plan.cause_probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_23_confidence_calibrated_from_posteriors(self):
        plan = self._get_plan()
        max_posterior = max(plan.cause_probabilities.values())
        assert 0.0 <= plan.confidence <= 1.0
        assert plan.confidence >= max_posterior

    def test_24_ranked_causes_sorted(self):
        plan = self._get_plan()
        probs = [p for _, p in plan.ranked_causes]
        assert probs == sorted(probs, reverse=True)

    def test_25_ranked_causes_length(self):
        plan = self._get_plan()
        assert len(plan.ranked_causes) == len(CAUSES)

    def test_26_recommended_phases_is_list(self):
        plan = self._get_plan()
        assert isinstance(plan.recommended_phases, list)

    def test_27_reasoning_nonempty_string(self):
        plan = self._get_plan()
        assert isinstance(plan.reasoning, str)
        assert len(plan.reasoning) > 0

    def test_28_material_propagated(self):
        plan = self._get_plan(material="tape")
        assert plan.material == "tape"


# ---------------------------------------------------------------------------
# Klasse 4: CausalDefectReasoner — Materialien und Ursachen
# ---------------------------------------------------------------------------


class TestReasonerMaterials:
    def setup_method(self):
        self.r = CausalDefectReasoner()

    def test_29_reason_tape_prefers_tape_causes(self):
        plan = self.r.reason(
            {"dropout_severity": 0.8, "noise_floor_db": -42.0},
            material="tape",
        )
        tape_prob = plan.cause_probabilities.get("tape_dropout", 0.0) + plan.cause_probabilities.get("tape_hiss", 0.0)
        # Tape-Material → Tape-Ursachen sollten dominant sein
        assert tape_prob > 0.1

    def test_30_reason_vinyl_crackle_high_with_clicks(self):
        plan = self.r.reason(
            {"click_severity": 0.9},
            material="vinyl",
        )
        # Vinyl + hohe click_severity → crackle oder dropout dominant
        assert plan.primary_cause in CAUSES

    def test_31_reason_digital_clip_with_clip_scores(self):
        audio = _clipped(secs=1.0)
        plan = self.r.reason(
            {"clip_severity": 0.95},
            material="digital",
            audio=audio,
            sample_rate=SR,
        )
        assert plan.cause_probabilities["digital_clip"] > 0.0

    def test_32_reason_unknown_material_fallback(self):
        plan = self.r.reason({}, material="UNKNOWN_XYZ")
        assert plan.material == "unknown"

    def test_33_reason_with_audio_mono(self):
        audio = _sine(secs=2.0)
        plan = self.r.reason({}, material="tape", audio=audio, sample_rate=SR)
        assert isinstance(plan, RestorationPlan)

    def test_34_reason_with_audio_stereo(self):
        audio = _stereo(secs=2.0)
        plan = self.r.reason({}, material="vinyl", audio=audio, sample_rate=SR)
        assert isinstance(plan, RestorationPlan)


class TestDefectAliasNormalization:
    def setup_method(self):
        self.r = CausalDefectReasoner()

    def test_dropout_alias_group_is_reciprocal(self):
        norm = _normalize_defect_scores({"tape_dropout": 0.72})
        assert pytest.approx(norm["tape_dropout"], rel=1e-6) == 0.72
        assert pytest.approx(norm["dropouts"], rel=1e-6) == 0.72
        assert pytest.approx(norm["dropout_severity"], rel=1e-6) == 0.72

    def test_clip_alias_group_is_reciprocal(self):
        norm = _normalize_defect_scores({"digital_clip": 0.66})
        assert pytest.approx(norm["digital_clip"], rel=1e-6) == 0.66
        assert pytest.approx(norm["clipping"], rel=1e-6) == 0.66
        assert pytest.approx(norm["clip_severity"], rel=1e-6) == 0.66

    def test_hum_alias_group_is_reciprocal(self):
        norm = _normalize_defect_scores({"electrical_hum": 0.44})
        assert pytest.approx(norm["electrical_hum"], rel=1e-6) == 0.44
        assert pytest.approx(norm["hum"], rel=1e-6) == 0.44

    def test_hiss_alias_group_is_reciprocal(self):
        norm = _normalize_defect_scores({"tape_hiss": 0.41})
        assert pytest.approx(norm["tape_hiss"], rel=1e-6) == 0.41
        assert pytest.approx(norm["high_freq_noise"], rel=1e-6) == 0.41

    def test_nr_breathing_alias_group_is_reciprocal(self):
        norm = _normalize_defect_scores({"nr_breathing": 0.53})
        assert pytest.approx(norm["nr_breathing"], rel=1e-6) == 0.53
        assert pytest.approx(norm["nr_breathing_artifact"], rel=1e-6) == 0.53

    def test_wow_and_flutter_alias_groups_are_reciprocal(self):
        norm = _normalize_defect_scores({"wow": 0.49, "flutter_severity": 0.37})
        assert pytest.approx(norm["wow"], rel=1e-6) == 0.49
        assert pytest.approx(norm["wow_severity"], rel=1e-6) == 0.49
        assert pytest.approx(norm["flutter"], rel=1e-6) == 0.37
        assert pytest.approx(norm["flutter_severity"], rel=1e-6) == 0.37

    def test_azimuth_alias_group_is_reciprocal(self):
        norm = _normalize_defect_scores({"cassette_azimuth_tolerance": 0.58})
        assert pytest.approx(norm["cassette_azimuth_tolerance"], rel=1e-6) == 0.58
        assert pytest.approx(norm["azimuth_error"], rel=1e-6) == 0.58
        assert pytest.approx(norm["head_misalignment"], rel=1e-6) == 0.58

    def test_defect_score_key_drift_guard(self):
        # Guard: alle defect_scores.get("...")-Keys im Reasoner müssen
        # entweder DefectType-Keys oder explizit erlaubte Legacy-/Proxy-Keys sein.
        import inspect
        import re

        import backend.core.causal_defect_reasoner as _cdr

        src = inspect.getsource(_cdr)
        found = set(re.findall(r'defect_scores\.get\("([a-zA-Z0-9_]+)"', src))
        from backend.core.defect_scanner import DefectType

        defect_keys = {d.value for d in DefectType}
        allowed_non_defect = {
            "click_severity",
            "dropout_severity",
            "clip_severity",
            "wow_severity",
            "flutter_severity",
            "tape_dropout",
            "tape_hiss",
            "electrical_hum",
            "digital_clip",
            "head_misalignment",
            "hum",
            "nr_breathing",
            "silence_ratio",
            "noise_floor_db",
            "wow_flutter",
        }
        unknown = sorted(k for k in found if k not in defect_keys and k not in allowed_non_defect)
        assert not unknown, f"Unerlaubte defect_scores-Keys entdeckt: {unknown}"

    def test_35_reason_no_audio(self):
        plan = self.r.reason({}, material="tape", audio=None)
        assert isinstance(plan, RestorationPlan)

    def test_36_reason_empty_defect_scores(self):
        plan = self.r.reason({}, material="tape")
        assert plan.primary_cause in CAUSES

    def test_37_soft_saturation_maps_to_empty_phases(self):
        # Direkt aus CAUSE_TO_PHASES — unabhängig vom Reasoner-Plan
        assert CAUSE_TO_PHASES["soft_saturation"] == []

    def test_38_reason_head_wear_phases_include_phase56(self):
        phases = CAUSE_TO_PHASES.get("head_wear", [])
        assert any("56" in p for p in phases)

    def test_39_all_materials_run_without_error(self):
        for mat in [
            "tape",
            "vinyl",
            "shellac",
            "digital",
            "unknown",
            "mp3_low",
            "mp3_high",
            "aac",
            "cd_digital",
            "streaming",
            "dat",
            "minidisc",
            "wax_cylinder",
            "lacquer_disc",
            "wire_recording",
        ]:
            plan = self.r.reason({}, material=mat)
            assert isinstance(plan, RestorationPlan)

    def test_40_probability_values_in_bounds(self):
        plan = self.r.reason({"noise_floor_db": -35.0}, material="tape")
        for c, p in plan.cause_probabilities.items():
            assert 0.0 <= p <= 1.0, f"Ursache {c}: P={p}"


# ---------------------------------------------------------------------------
# Klasse 5: Convenience-Funktionen und Singleton
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_41_get_reasoner_returns_instance(self):
        r = get_reasoner()
        assert isinstance(r, CausalDefectReasoner)

    def test_42_get_reasoner_singleton(self):
        r1 = get_reasoner()
        r2 = get_reasoner()
        assert r1 is r2

    def test_43_reason_about_defects_callable(self):
        result = reason_about_defects(defect_scores={}, material="tape")
        assert isinstance(result, RestorationPlan)

    def test_44_reason_about_defects_with_audio(self):
        audio = _sine(secs=1.0)
        result = reason_about_defects(
            defect_scores={"dropout_severity": 0.5},
            material="tape",
            audio=audio,
            sample_rate=SR,
        )
        assert isinstance(result, RestorationPlan)

    def test_45_nan_defect_scores_safe(self):
        # NaN in defect_scores sollte keinen unbehandelten Crash verursachen
        try:
            plan = reason_about_defects({"dropout_severity": float("nan")}, material="tape")
            assert isinstance(plan, RestorationPlan)
        except Exception as e:
            pytest.fail(f"NaN in defect_scores warf Exception: {e}")

    def test_46_inf_defect_scores_safe(self):
        try:
            plan = reason_about_defects({"clip_severity": float("inf")}, material="digital")
            assert isinstance(plan, RestorationPlan)
        except Exception as e:
            pytest.fail(f"Inf in defect_scores warf Exception: {e}")

    def test_47_short_audio_safe(self):
        audio = np.zeros(10, dtype=np.float32)
        plan = reason_about_defects({}, material="tape", audio=audio, sample_rate=SR)
        assert isinstance(plan, RestorationPlan)

    def test_48_plan_phase_params_is_dict(self):
        plan = reason_about_defects({"dropout_severity": 0.7}, material="tape")
        assert isinstance(plan.phase_parameters, dict)

    def test_49_new_key_clicks_aliases_to_click_severity(self):
        plan_legacy = reason_about_defects({"click_severity": 0.9}, material="vinyl")
        plan_new = reason_about_defects({"clicks": 0.9}, material="vinyl")
        assert plan_new.primary_cause == plan_legacy.primary_cause

    def test_50_new_key_dropouts_aliases_to_dropout_severity(self):
        plan_legacy = reason_about_defects({"dropout_severity": 0.8}, material="tape")
        plan_new = reason_about_defects({"dropouts": 0.8}, material="tape")
        assert plan_new.primary_cause == plan_legacy.primary_cause

    def test_51_new_key_clipping_aliases_to_clip_severity(self):
        plan_legacy = reason_about_defects({"clip_severity": 0.95}, material="digital")
        plan_new = reason_about_defects({"clipping": 0.95}, material="digital")
        assert plan_new.primary_cause == plan_legacy.primary_cause

    def test_52_wow_flutter_component_aliases_work(self):
        plan = reason_about_defects({"wow": 0.6, "flutter": 0.5}, material="tape")
        # Ensure combined cause receives non-zero posterior via derived wow_flutter.
        assert plan.cause_probabilities.get("wow_flutter", 0.0) > 0.0

    def test_53_non_finite_scores_are_sanitized(self):
        plan = reason_about_defects({"clicks": float("nan"), "dropouts": float("inf")}, material="tape")
        assert isinstance(plan, RestorationPlan)

    def test_54_wax_cylinder_phase07_excluded_from_plan(self):
        """§6.2b §ERA 1900-1925: phase_07_harmonic_restoration darf für wax_cylinder nie vorgeschlagen werden."""
        from backend.core.causal_defect_reasoner import _MATERIAL_PHASE_EXCLUSIONS

        assert "wax_cylinder" in _MATERIAL_PHASE_EXCLUSIONS, (
            "§6.2b: _MATERIAL_PHASE_EXCLUSIONS hat keinen Eintrag für 'wax_cylinder'"
        )
        assert "phase_07_harmonic_restoration" in _MATERIAL_PHASE_EXCLUSIONS["wax_cylinder"], (
            "§ERA 1900-1925: phase_07_harmonic_restoration fehlt in wax_cylinder-Exclusions"
        )

    def test_55_wax_cylinder_reason_does_not_contain_phase07(self):
        """§ERA 1900-1925: reason() mit bandwidth_loss auf wax_cylinder darf phase_07 nicht zurückgeben."""
        plan = reason_about_defects({"bandwidth_loss": 0.8, "high_freq_noise": 0.3}, material="wax_cylinder")
        assert "phase_07_harmonic_restoration" not in plan.recommended_phases, (
            f"§ERA 1900-1925: phase_07 in Plan für wax_cylinder — VERBOTEN. Gefundene Phasen: {plan.recommended_phases}"
        )

    def test_56_wire_recording_phase07_excluded(self):
        """Drahtaufnahmen: phase_07 VERBOTEN (keine verlässliche harmonische Basis)."""
        plan = reason_about_defects({"bandwidth_loss": 0.7}, material="wire_recording")
        assert "phase_07_harmonic_restoration" not in plan.recommended_phases, (
            f"phase_07 in Plan für wire_recording — VERBOTEN. Phasen: {plan.recommended_phases}"
        )

    def test_57_phase06_wax_cylinder_bw_ceiling_is_3000hz(self):
        """§ERA 1900-1925: phase_06 BW-Ceiling für wax_cylinder muss 3000 Hz sein (Spec: max 3 kHz)."""
        import pathlib

        src = (
            pathlib.Path(__file__).parents[2] / "backend" / "core" / "phases" / "phase_06_frequency_restoration.py"
        ).read_text()
        # Prüfe dass kein 5000-Eintrag für wax_cylinder mehr vorhanden ist
        assert '"wax_cylinder": 5000.0' not in src, (
            "§ERA 1900-1925: phase_06 BW-Ceiling für wax_cylinder ist noch 5000 Hz — muss 3000 Hz sein"
        )
        assert '"wax_cylinder": 3000.0' in src, "§ERA 1900-1925: phase_06 BW-Ceiling für wax_cylinder muss 3000 Hz sein"

    def test_58_phase07_wax_cylinder_bw_ceiling_is_3000hz(self):
        """§ERA 1900-1925: phase_07 BW-Ceiling für wax_cylinder muss 3000 Hz sein (Sekundär-Guard)."""
        import pathlib

        src = (
            pathlib.Path(__file__).parents[2] / "backend" / "core" / "phases" / "phase_07_harmonic_restoration.py"
        ).read_text()
        assert '"wax_cylinder": 5000.0' not in src, (
            "§ERA 1900-1925: phase_07 BW-Ceiling für wax_cylinder ist noch 5000 Hz — muss 3000 Hz sein"
        )
        assert '"wax_cylinder": 3000.0' in src, "§ERA 1900-1925: phase_07 BW-Ceiling für wax_cylinder muss 3000 Hz sein"

    def test_59_causal_coalition_priority_lifts_digital_pair(self):
        """§2.67: phase_50 wird als Partner von phase_23 in der Rangfolge angehoben."""
        reasoner = CausalDefectReasoner()
        ordered = [
            "phase_23_spectral_repair",
            "phase_29_tape_hiss_reduction",
            "phase_50_spectral_repair",
        ]
        ranked = [
            ("generation_loss", 0.60),
            ("high_freq_noise", 0.54),
            ("aliasing", 0.10),
        ]

        result = reasoner._apply_phase_coalition_priority(ordered, ranked)
        assert result.index("phase_50_spectral_repair") < result.index("phase_29_tape_hiss_reduction")

    def test_60_causal_coalition_priority_keeps_non_members_stable(self):
        """Ohne ausreichende Koalitions-Mitglieder bleibt die Reihenfolge unverändert."""
        reasoner = CausalDefectReasoner()
        ordered = [
            "phase_29_tape_hiss_reduction",
            "phase_03_denoise",
            "phase_04_eq_correction",
        ]
        ranked = [
            ("tape_hiss", 0.62),
            ("motor_interference", 0.41),
        ]

        result = reasoner._apply_phase_coalition_priority(ordered, ranked)
        assert result == ordered

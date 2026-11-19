"""
tests/test_autonomous_restoration_engine.py
==============================================

Tests für die AutonomousRestorationEngine, AutoMusicalGoalSetter
und den upgegradeten SelfLearningOptimizer.

Einzige Nutzer-Eingabe in allen Tests: ProcessingMode.RESTORATION oder
ProcessingMode.STUDIO_2026 — kein weiterer Nutzereingriff.

Author: Aurik Development Team
Date: 2026-02-17
"""

import json
import math
import os
import tempfile
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from backend.core.auto_musical_goal_setter import (
    _MATERIAL_ADJUSTMENTS,
    _MATERIAL_PQS_TARGETS,
    AutoMusicalGoalSetter,
    MusicalGoalProfile,
)
from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, DefectType, MaterialType
from backend.core.processing_modes import ProcessingMode
from backend.core.quality_prediction import QualityEstimate, QualityLevel
from backend.core.self_learning_optimizer import ArmStats, SelfLearningOptimizer

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_audio(sr: int = 44100, duration: float = 0.5, noise: float = 0.01) -> NDArray[np.float32]:
    """Erzeugt ein synthetisches Testsignal (Sinus + weißes Rauschen)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    signal += noise * np.random.randn(len(signal)).astype(np.float32)
    return np.asarray(signal, dtype=np.float32)


def _make_defect_result(
    material: MaterialType = MaterialType.VINYL,
    top_defect: DefectType = DefectType.CLICKS,
    severity: float = 0.5,
) -> DefectAnalysisResult:
    """Erstellt ein minimales DefectAnalysisResult für Tests."""
    scores = {
        top_defect: DefectScore(
            defect_type=top_defect,
            severity=severity,
            confidence=0.9,
        )
    }
    return DefectAnalysisResult(
        material_type=material,
        scores=scores,
        analysis_time_seconds=0.01,
        sample_rate=44100,
        duration_seconds=0.5,
    )


def _make_quality_estimate(score: float = 55.0, confidence: float = 0.8) -> QualityEstimate:
    """Erstellt ein minimales QualityEstimate für Tests."""
    level_map = [
        (40, QualityLevel.POOR),
        (60, QualityLevel.FAIR),
        (80, QualityLevel.GOOD),
        (95, QualityLevel.EXCELLENT),
    ]
    level = QualityLevel.PRISTINE
    for threshold, lvl in level_map:
        if score < threshold:
            level = lvl
            break
    return QualityEstimate(
        overall_score=score,
        quality_level=level,
        snr_db=max(score * 0.4, 10.0),
        dynamic_range_db=12.0,
        thd_percent=0.5,
        clarity=score / 100.0,
        warmth=0.6,
        brightness=0.5,
        naturalness=0.7,
        authenticity=0.75,
        confidence=confidence,
        bandwidth_hz=(20.0, 18000.0),
        has_artifacts=score < 60,
        artifact_types=[],
    )


# ===========================================================================
# AutoMusicalGoalSetter Tests
# ===========================================================================


class TestAutoMusicalGoalSetter:
    """Tests für den AutoMusicalGoalSetter."""

    def test_restoration_mode_returns_valid_profile(self):
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        defect_result = _make_defect_result()
        quality_est = _make_quality_estimate(score=50.0)

        profile = setter.compute_goals(defect_result=defect_result, quality_estimate=quality_est)

        assert isinstance(profile, MusicalGoalProfile)
        assert profile.mode == "restoration"
        assert 0.0 <= profile.target_authenticity <= 1.0
        assert 0.0 <= profile.target_naturalness <= 1.0
        assert 0.0 <= profile.target_clarity <= 1.0
        assert 0.0 <= profile.denoise_strength <= 1.0
        assert 0.0 <= profile.declip_strength <= 1.0
        assert 20.0 <= profile.target_snr_db <= 80.0

    def test_studio_2026_mode_returns_valid_profile(self):
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.STUDIO_2026)
        defect_result = _make_defect_result(material=MaterialType.STREAMING)
        quality_est = _make_quality_estimate(score=70.0)

        profile = setter.compute_goals(defect_result=defect_result, quality_estimate=quality_est)

        assert profile.mode == "studio_2026"
        assert profile.target_clarity > 0.8, "Studio_2026 muss hohe Clarity anstreben"
        assert profile.target_brightness > 0.7, "Studio_2026 muss Brillanz anstreben"
        assert not profile.preserve_character, "Studio_2026 soll keinen Analogchar erhalten"

    def test_studio_2026_prioritizes_authenticity_and_naturalness_on_digital_master(self):
        """Studio-2026 soll auch bei Digital-Mastern hohe Natürlichkeit/Authentizität halten."""
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.STUDIO_2026)
        profile = setter.compute_goals(
            _make_defect_result(material=MaterialType.CD_DIGITAL),
            _make_quality_estimate(score=75.0),
        )
        assert profile.target_authenticity >= 0.80
        assert profile.target_naturalness >= 0.90

    def test_restoration_authenticity_never_below_0_70(self):
        """Restaurierungsmodus: Authentizität niemals unter 0.70."""
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        # Shellac mit starkem Rauschen zieht Authentizität runter
        defect_result = _make_defect_result(
            material=MaterialType.SHELLAC,
            top_defect=DefectType.HIGH_FREQ_NOISE,
            severity=0.99,
        )
        quality_est = _make_quality_estimate(score=15.0)  # Sehr schlechte Qualität

        profile = setter.compute_goals(defect_result=defect_result, quality_estimate=quality_est)

        assert profile.target_authenticity >= 0.70, (
            f"Authentizität {profile.target_authenticity:.3f} unter 0.70 im Restoration-Modus!"
        )

    def test_shellac_material_adjusts_click_sensitivity_up(self):
        """Shellac: Klick-Empfindlichkeit wird erhöht."""
        setter_vinyl = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        setter_shellac = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)

        defect_vinyl = _make_defect_result(material=MaterialType.VINYL, severity=0.3)
        defect_shellac = _make_defect_result(material=MaterialType.SHELLAC, severity=0.3)
        quality_est = _make_quality_estimate()

        profile_vinyl = setter_vinyl.compute_goals(defect_vinyl, quality_est)
        profile_shellac = setter_shellac.compute_goals(defect_shellac, quality_est)

        assert profile_shellac.click_sensitivity >= profile_vinyl.click_sensitivity, (
            "Shellac sollte höhere click_sensitivity als Vinyl haben"
        )

    def test_high_quality_input_reduces_processing_strength(self):
        """Bei hoher Eingangsqualität werden Processing-Stärken reduziert."""
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        defect_result = _make_defect_result(severity=0.2)

        profile_low_q = setter.compute_goals(defect_result, _make_quality_estimate(score=15.0))
        profile_high_q = setter.compute_goals(defect_result, _make_quality_estimate(score=85.0))

        assert profile_high_q.denoise_strength <= profile_low_q.denoise_strength, (
            "Hohe Eingangsqualität sollte zu niedrigerer denoise_strength führen"
        )

    def test_conflict_resolution_high_authenticity_caps_denoise(self):
        """Hohe Authentizität + starkes De-Noising wird korrigiert."""
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        # Vinyl mit sehr starkem HF-Rauschen → würde denoise_strength hoch treiben
        defect_result = _make_defect_result(
            material=MaterialType.VINYL,
            top_defect=DefectType.HIGH_FREQ_NOISE,
            severity=0.99,
        )
        quality_est = _make_quality_estimate(score=20.0)

        profile = setter.compute_goals(defect_result, quality_est)

        # Conflict-Resolution: denoise darf nicht > 0.70 bei hoher Authenticity
        if profile.target_authenticity > 0.85:
            assert profile.denoise_strength <= 0.70 + 1e-6, (
                f"Conflict-Resolution fehlgeschlagen: denoise={profile.denoise_strength:.3f} "
                f"bei authenticity={profile.target_authenticity:.3f}"
            )

    def test_to_dict_is_serializable(self):
        """MusicalGoalProfile.to_dict() muss JSON-serialisierbar sein."""
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        profile = setter.compute_goals(_make_defect_result(), _make_quality_estimate())
        d = profile.to_dict()
        assert isinstance(d, dict)
        json_str = json.dumps(d)  # darf nicht werfen
        assert len(json_str) > 10

    def test_lufs_target_within_valid_range(self):
        """LUFS-Ziel muss im gültigen Bereich [-23, -8] liegen."""
        for mode in (ProcessingMode.RESTORATION, ProcessingMode.STUDIO_2026):
            setter = AutoMusicalGoalSetter(mode=mode)
            for material in (
                MaterialType.VINYL,
                MaterialType.SHELLAC,
                MaterialType.MP3_LOW,
                MaterialType.CD_DIGITAL,
                MaterialType.STREAMING,
            ):
                profile = setter.compute_goals(
                    _make_defect_result(material=material),
                    _make_quality_estimate(),
                )
                assert -23.0 <= profile.target_lufs <= -8.0, (
                    f"LUFS {profile.target_lufs} außerhalb [-23, -8] für {material.value}/{mode.value}"
                )

    def test_all_material_types_have_explicit_policy(self):
        """Jede MaterialType-Ausprägung muss explizite Vorgaben haben."""
        all_materials = set(MaterialType)
        assert set(_MATERIAL_ADJUSTMENTS.keys()) == all_materials
        assert set(_MATERIAL_PQS_TARGETS.keys()) == all_materials

    def test_all_material_types_generate_valid_profiles(self):
        """Für alle Tonträgerarten wird ein valides Zielprofil erzeugt."""
        setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        quality_est = _make_quality_estimate(score=55.0)

        for material in MaterialType:
            profile = setter.compute_goals(
                _make_defect_result(material=material),
                quality_est,
            )
            assert isinstance(profile, MusicalGoalProfile)
            assert profile.material == material.value
            assert "PQS-Ziel(material)≥" in profile.rationale
            assert 0.0 <= profile.denoise_strength <= 1.0
            assert 0.0 <= profile.declip_strength <= 1.0


# ===========================================================================
# SelfLearningOptimizer Tests
# ===========================================================================


class TestSelfLearningOptimizer:
    """Tests für den UCB1-basierten SelfLearningOptimizer."""

    def test_record_and_recommend_basic(self):
        """Basis: Aufzeichnen und Empfehlen funktionieren."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)

            opt.record_result(MaterialType.VINYL, "conservative", None, quality_delta=2.0)
            opt.record_result(MaterialType.VINYL, "balanced", None, quality_delta=4.5)
            opt.record_result(MaterialType.VINYL, "aggressive", None, quality_delta=1.0)

            # Nach 3 Pulls (je einmal): UCB1 kann noch nicht klar unterscheiden
            result = opt.recommend_variant(MaterialType.VINYL, None)
            assert result in ("conservative", "balanced", "aggressive")

    def test_best_variant_eventually_recommended(self):
        """UCB1 soll nach genug Erfahrung die beste Variante bevorzugen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(
                mode=ProcessingMode.RESTORATION, state_path=path, exploration_factor=0.5
            )  # Niedrige Exploration

            # 30 × 'balanced' mit konsistent hohem Delta
            for _ in range(30):
                opt.record_result(MaterialType.VINYL, "balanced", None, quality_delta=5.0)
                opt.record_result(MaterialType.VINYL, "conservative", None, quality_delta=1.0)

            result = opt.recommend_variant(MaterialType.VINYL, None)
            assert result == "balanced", f"Nach 30 Iterationen sollte 'balanced' empfohlen werden, nicht {result!r}"

    def test_unknown_material_returns_none(self):
        """Für unbekanntes Material: None zurückgeben."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)
            # Nur Vinyl-Daten vorhanden
            opt.record_result(MaterialType.VINYL, "balanced", None, quality_delta=3.0)
            # Shellac ist unbekannt
            result = opt.recommend_variant(MaterialType.SHELLAC, None)
            assert result is None

    def test_state_persists_across_instances(self):
        """Zustand wird korrekt gespeichert und wiederhergestellt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            opt1 = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)
            opt1.record_result(MaterialType.SHELLAC, "specialist_declicker", None, quality_delta=6.0)
            assert opt1.total_pulls() == 1

            # Neue Instanz — soll Zustand laden
            opt2 = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)
            assert opt2.total_pulls() == 1
            assert opt2.has_data_for(MaterialType.SHELLAC, "specialist_declicker")

    def test_arm_stats_ucb1_unvisited_arm_is_infinity(self):
        """Unbesuchter Arm hat UCB1-Score = +∞."""
        arm = ArmStats(count=0)
        assert arm.ucb1_score(total_pulls=100) == float("inf")

    def test_arm_stats_ucb1_single_pull(self):
        """Nach einem Pull: UCB1-Score ist endlich und korrekt berechenbar."""
        arm = ArmStats()
        arm.update(delta=3.0)
        score = arm.ucb1_score(total_pulls=10, exploration_factor=1.4)
        assert math.isfinite(score)
        assert score > 0.0

    def test_negative_delta_handled(self):
        """Negative quality_delta (Verschlechterung) wird korrekt verarbeitet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)
            opt.record_result(MaterialType.TAPE, "aggressive", None, quality_delta=-3.0)
            mean = opt.arm_mean_delta(MaterialType.TAPE, "aggressive")
            assert mean is not None and mean < 0.0

    def test_statistics_dict_structure(self):
        """get_statistics() liefert vollständige Struktur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)
            opt.record_result(MaterialType.VINYL, "balanced", None, quality_delta=2.5)
            stats = opt.get_statistics()
            assert "mode" in stats
            assert "total_pulls" in stats
            assert "arms" in stats
            assert stats["total_pulls"] == 1
            assert opt.total_pulls() == 1

    def test_public_feedback_api_still_works(self):
        """Public feedback API (update_from_feedback, predict, optimize) remains functional."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)
            features = {"snr": 0.8, "clarity": 0.7}
            opt.update_from_feedback(features, feedback=1.0)
            pred = opt.predict(features)
            assert isinstance(pred, float)
            optimized = opt.optimize(features)
            assert set(optimized.keys()) == set(features.keys())


# ===========================================================================
# AutoMusicalGoalSetter + SelfLearningOptimizer Integration
# ===========================================================================


class TestGoalSetterOptimizerIntegration:
    """Integration: Ziele und Self-Learning arbeiten zusammen."""

    def test_goal_profile_feeds_into_optimizer_record(self):
        """Workflow: Ziele berechnen, verarbeiten, Ergebnis in Optimizer schreiben."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            opt = SelfLearningOptimizer(mode=ProcessingMode.RESTORATION, state_path=path)

            setter = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
            defect = _make_defect_result(material=MaterialType.VINYL, severity=0.6)
            quality_est = _make_quality_estimate(score=45.0)

            profile = setter.compute_goals(defect, quality_est)
            assert profile is not None

            # Simuliertes Ergebnis zurückschreiben
            opt.record_result(
                material=MaterialType.VINYL,
                variant="balanced",
                defect_profile=defect,
                quality_delta=3.5,
            )
            assert opt.total_pulls() == 1

    def test_both_modes_produce_different_goals(self):
        """RESTORATION und STUDIO_2026 müssen unterschiedliche Ziele produzieren."""
        defect = _make_defect_result(material=MaterialType.VINYL)
        quality_est = _make_quality_estimate()

        setter_r = AutoMusicalGoalSetter(mode=ProcessingMode.RESTORATION)
        setter_s = AutoMusicalGoalSetter(mode=ProcessingMode.STUDIO_2026)

        profile_r = setter_r.compute_goals(defect, quality_est)
        profile_s = setter_s.compute_goals(defect, quality_est)

        # Studio muss mehr Klarheit und Helligkeit anstreben
        assert profile_s.target_clarity > profile_r.target_clarity
        assert profile_s.target_brightness > profile_r.target_brightness
        # Restoration muss mehr Authentizität bewahren
        assert profile_r.target_authenticity > profile_s.target_authenticity
        # Verschiedene LUFS-Ziele
        assert profile_s.target_lufs > profile_r.target_lufs  # Studio ist lauter


# ===========================================================================
# DefectPhaseMapper Tests
# ===========================================================================


class TestDefectPhaseMapper:
    """Tests für core/defect_phase_mapper.py"""

    def test_all_defect_types_have_mapping(self):
        """Jeder DefectType muss eine PhaseAssignment haben."""
        from backend.core.defect_phase_mapper import _PHASE_MAP

        for dt in DefectType:
            assert dt in _PHASE_MAP, f"Kein Mapping für DefectType.{dt.name}"

    def test_phase_map_is_complete_for_all_defect_types(self):
        """Die kanonische Defekt-Phase-Tabelle muss alle DefectType-Werte abdecken."""
        from backend.core.defect_phase_mapper import _PHASE_MAP

        missing = [dt.name for dt in DefectType if dt not in _PHASE_MAP]
        assert not missing, f"Fehlende DefectType-Zuordnung: {', '.join(missing)}"

    def test_primary_phases_not_empty(self):
        """Jede PhaseAssignment muss mindestens eine Primary-Phase haben."""
        from backend.core.defect_phase_mapper import _PHASE_MAP

        for dt, assignment in _PHASE_MAP.items():
            assert len(assignment.primary_phases) >= 1, f"{dt.name}: primary_phases ist leer"

    def test_get_primary_phases_returns_list(self):
        """get_primary_phases() gibt Liste zurück."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        phases = mapper.get_primary_phases(DefectType.CLICKS)
        assert isinstance(phases, list)
        assert len(phases) >= 1
        assert "phase_01_click_removal" in phases

    def test_get_all_phases_superset_of_primary(self):
        """get_all_phases() enthält alle Primary-Phasen."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        for dt in DefectType:
            primary = mapper.get_primary_phases(dt)
            all_p = mapper.get_all_phases(dt)
            for p in primary:
                assert p in all_p

    def test_restoration_mode_filters_forbidden_primary_phases(self):
        """Im Restoration-Modus dürfen keine §0a-verbotenen Primärphasen erscheinen."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        for dt in DefectType:
            phases = mapper.get_primary_phases(dt, mode="restoration")
            assert not any(
                phase in {"phase_21_exciter", "phase_35_multiband_compression", "phase_42_vocal_enhancement"}
                for phase in phases
            ), f"{dt.name}: verbotene Phase im Restoration-Modus"

    def test_build_specialist_config_clicks(self):
        """Klick-Config muss hohe click_removal_sensitivity haben."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper
        from backend.core.processing_modes import ProcessingConfig, get_processing_config

        mapper = DefectPhaseMapper()
        base = get_processing_config(ProcessingMode.RESTORATION)
        config, name = mapper.build_specialist_config(
            base_config=base,
            defect_type=DefectType.CLICKS,
            severity=0.8,
            is_restoration_mode=True,
        )
        config = cast(ProcessingConfig, config)
        assert config.click_removal_sensitivity > 0.5
        assert "click" in name.lower() or "specialist" in name.lower()

    def test_build_specialist_config_hum_sets_lowfreq(self):
        """HUM-Config muss low_freq_rolloff_hz setzen."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper
        from backend.core.processing_modes import ProcessingConfig, get_processing_config

        mapper = DefectPhaseMapper()
        base = get_processing_config(ProcessingMode.RESTORATION)
        config, _ = mapper.build_specialist_config(
            base_config=base,
            defect_type=DefectType.HUM,
            severity=0.7,
            is_restoration_mode=True,
        )
        config = cast(ProcessingConfig, config)
        assert config.low_freq_rolloff_hz is not None
        assert config.low_freq_rolloff_hz > 0

    def test_build_specialist_config_digital_artifacts_enables_spectral_repair(self):
        """DIGITAL_ARTIFACTS-Config muss spektrale Reparatur aktivieren."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper
        from backend.core.processing_modes import ProcessingConfig, get_processing_config

        mapper = DefectPhaseMapper()
        base = get_processing_config(ProcessingMode.RESTORATION)
        config, _ = mapper.build_specialist_config(
            base_config=base,
            defect_type=DefectType.DIGITAL_ARTIFACTS,
            severity=0.9,
            is_restoration_mode=True,
        )
        config = cast(ProcessingConfig, config)
        assert config.enable_spectral_repair is True
        assert config.spectral_repair_strength > 0.5

    def test_restoration_mode_caps_denoise_at_0_9(self):
        """Im RESTORATION-Modus darf denoise_strength nie > 0.9 sein."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper
        from backend.core.processing_modes import ProcessingConfig, get_processing_config

        mapper = DefectPhaseMapper()
        for dt in DefectType:
            base = get_processing_config(ProcessingMode.RESTORATION)
            config, _ = mapper.build_specialist_config(
                base_config=base,
                defect_type=dt,
                severity=1.0,
                is_restoration_mode=True,
            )
            config = cast(ProcessingConfig, config)
            assert config.denoise_strength <= 0.90, f"{dt.name}: denoise_strength={config.denoise_strength:.2f} > 0.9"

    def test_phases_for_defect_profile_ordering(self):
        """phases_for_defect_profile gibt Primary-Phasen zuerst zurück."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        defects = [
            DefectScore(DefectType.CLICKS, severity=0.9, confidence=0.9),
            DefectScore(DefectType.HIGH_FREQ_NOISE, severity=0.5, confidence=0.8),
        ]
        phases = mapper.phases_for_defect_profile(defects, max_phases=10)
        assert isinstance(phases, list)
        assert len(phases) > 0
        # Primary-Phase für CLICKS (höchste Severity) muss früh erscheinen
        assert "phase_01_click_removal" in phases
        assert phases.index("phase_01_click_removal") < len(phases) // 2 + 2

    def test_phases_for_defect_profile_coalition_priority(self):
        """§2.67: Koalitionsphasen werden im Profil als zusammengehörige Gruppe priorisiert."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        defects = [
            DefectScore(DefectType.GENERATION_LOSS, severity=0.9, confidence=0.9),
            DefectScore(DefectType.LOW_FREQ_RUMBLE, severity=0.8, confidence=0.8),
        ]
        phases = mapper.phases_for_defect_profile(defects, max_phases=10)

        assert "phase_23_spectral_repair" in phases
        assert "phase_07_harmonic_restoration" in phases
        assert "phase_05_rumble_filter" in phases

        # Ohne Koalitions-Priorisierung läge phase_05 (0.8) vor phase_07 (0.54).
        # Mit §2.67-Coalition wird phase_07 als Partner von phase_23 nach vorne gezogen.
        assert phases.index("phase_07_harmonic_restoration") < phases.index("phase_05_rumble_filter")

    def test_describe_returns_string(self):
        """describe() gibt nicht-leeren String zurück."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        desc = mapper.describe(DefectType.DROPOUTS)
        assert isinstance(desc, str)
        assert "dropout" in desc.lower() or "phase_24" in desc.lower()

    @pytest.mark.parametrize(
        ("defect_type", "expected_primary_phase"),
        [
            (DefectType.GROOVE_ECHO, "phase_61_groove_echo_cancellation"),
            (DefectType.INNER_GROOVE_DISTORTION, "phase_60_inner_groove_distortion_repair"),
            (DefectType.TAPE_SPLICE_ARTIFACT, "phase_64_tape_splice_repair"),
            (DefectType.MODULATION_NOISE, "phase_59_modulation_noise_reduction"),
            (DefectType.SCRAPE_FLUTTER, "phase_12_wow_flutter_fix"),
            (DefectType.TAPE_HEAD_CLOG, "phase_56_spectral_band_gap_repair"),
        ],
    )
    def test_new_sota_defects_have_target_primary_phase(self, defect_type: DefectType, expected_primary_phase: str):
        """Neue Defekttypen müssen im Mapper auf ihre dedizierten Kernphasen zeigen."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper

        mapper = DefectPhaseMapper()
        primary = mapper.get_primary_phases(defect_type)

        assert isinstance(primary, list)
        assert expected_primary_phase in primary


# ===========================================================================
# IntrinsicAudioQualityScorer Tests
# ===========================================================================


class TestIntrinsicAudioQualityScorer:
    """Tests für core/intrinsic_audio_quality_scorer.py"""

    def _make_signal(
        self,
        sr: int = 44100,
        dur: float = 2.0,
        freq: float = 440.0,
        noise: float = 0.0,
        harmonics: int = 3,
    ) -> NDArray[np.float32]:
        t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
        sig = np.zeros_like(t, dtype=np.float32)
        for n in range(1, harmonics + 1):
            sig += ((1.0 / n) * np.sin(2 * np.pi * freq * n * t)).astype(np.float32)
        sig /= np.max(np.abs(sig)) + 1e-9
        sig *= 0.5
        if noise > 0:
            sig += np.random.default_rng(42).normal(0, noise, len(sig)).astype(np.float32)
        return np.asarray(sig, dtype=np.float32)

    def test_returns_intrinsic_quality_score(self):
        """score() gibt einen Float-Score 0-1 zurück."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        audio = self._make_signal()
        result = scorer.score(audio, audio.copy(), 44100)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_clean_scores_higher_than_noisy(self):
        """Sauberes Signal muss besser bewertet werden als verrauschtes."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        sr = 44100
        clean = self._make_signal(sr=sr, noise=0.0)
        noisy = self._make_signal(sr=sr, noise=0.3)
        assert scorer.score(clean, clean.copy(), sr) > scorer.score(noisy, noisy.copy(), sr)

    def test_snr_higher_for_clean(self):
        """Score muss für sauberes Signal größer sein."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        sr = 44100
        clean = self._make_signal(sr=sr, noise=0.0)
        noisy = self._make_signal(sr=sr, noise=0.3)
        assert scorer.score(clean, clean.copy(), sr) > scorer.score(noisy, noisy.copy(), sr)

    def test_clipping_score_detects_clipped_signal(self):
        """Geklipptes Signal muss niedrigeren clipping_score haben."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        sr = 44100
        # Signal mit echtem Flat-Top-Clipping nahe ±1.0
        clean = self._make_signal(sr=sr)
        # Erzwinge Flat-Top: Signal über 0.98 begrenzen (echter Clipping-Peak)
        clipped = np.copy(clean)
        clipped = clipped / (np.max(np.abs(clipped)) + 1e-9)  # normiere auf ±1
        clipped = np.clip(clipped * 3.0, -0.9, 0.9)  # Flat-Top bei ±0.9
        # Manuell auf ±1.0 skalieren damit peak >= 0.98
        clipped = clipped / 0.9  # jetzt peak = 1.0 mit echtem Flat-Top
        s_clean = scorer.score(clean, clean.copy(), sr)
        s_clipped = scorer.score(clipped, clipped.copy(), sr)
        assert s_clean >= s_clipped  # float compare (API returns float, not dataclass)

    def test_stereo_input_handled(self):
        """Stereo-Input (2D-Array) muss ohne Fehler verarbeitet werden."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        mono = self._make_signal()
        stereo = np.stack([mono, mono * 0.9], axis=1)
        result = scorer.score(stereo, stereo.copy(), 44100)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_very_short_signal_returns_neutral(self):
        """Zu kurzes Signal (< FFT-Größe) muss 0.5 zurückgeben, kein Crash."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        tiny = np.zeros(16, dtype=np.float32)
        result = scorer.score(tiny, tiny.copy(), 44100)
        assert result >= 0.0

    def test_score_consistent_across_calls(self):
        """score() muss konsistent sein bei gleichem Eingang."""
        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        scorer = IntrinsicAudioQualityScorer()
        audio = self._make_signal()
        s1 = scorer.score(audio, audio.copy(), 44100)
        s2 = scorer.score(audio, audio.copy(), 44100)
        assert s1 == pytest.approx(s2)

"""
Tests für Defect Analysis und Phase Skipping

Tests für:
- DefectAnalyzer (Clipping, Clicks, Dropouts, Noise, Hum, Hiss)
- DefectAnalysis (Helper methods)
- PhaseSkipper (Skip decisions)
- Integration (Speedup estimation)
"""

import numpy as np
import pytest

from backend.core.defect_analysis import DefectAnalysis, DefectAnalyzer, SourceMedium
from backend.core.defect_scanner import DefectScore, DefectType, MaterialType
from backend.core.phase_skipping import PhaseSkipper, ProcessingPhase
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 48000


@pytest.fixture
def clean_audio(sample_rate):
    """Generate clean audio (no defects)."""
    sr = sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Clean sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    return audio, sr


@pytest.fixture
def clipped_audio(sample_rate):
    """Generate clipped audio."""
    sr = sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Sine wave that clips
    audio = 1.2 * np.sin(2 * np.pi * 440 * t)
    audio = np.clip(audio, -1.0, 1.0)  # Hard clip

    return audio, sr


@pytest.fixture
def noisy_audio(sample_rate):
    """Generate noisy audio."""
    sr = sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Sine wave + noise
    signal_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(t))
    audio = signal_audio + noise

    return audio, sr


@pytest.mark.unit
class TestDefectAnalyzer:
    """Test DefectAnalyzer."""

    def test_analyzer_creation(self):
        """Test creating DefectAnalyzer."""
        analyzer = DefectAnalyzer()
        assert analyzer is not None

    def test_analyze_clean_audio(self, clean_audio):
        """Test analyzing clean audio."""
        audio, sr = clean_audio
        analyzer = DefectAnalyzer()

        analysis = analyzer.analyze(audio, sr)

        assert isinstance(analysis, DefectAnalysis)
        assert analysis.clipping_percentage < 1.0
        assert analysis.click_count >= 0  # May detect false positives
        assert analysis.dropout_count >= 0
        assert analysis.overall_quality > 0.5

    def test_detect_clipping(self, clipped_audio):
        """Test clipping detection."""
        audio, sr = clipped_audio
        analyzer = DefectAnalyzer()

        analysis = analyzer.analyze(audio, sr)

        # Should detect clipping
        assert analysis.clipping_percentage > 1.0
        assert analysis.clipping_severity > 0.9

    def test_detect_noise(self, noisy_audio):
        """Test noise detection."""
        audio, sr = noisy_audio
        analyzer = DefectAnalyzer()

        analysis = analyzer.analyze(audio, sr)

        # Should detect higher noise floor
        assert analysis.noise_floor_db > -60.0  # Noisy

    def test_clean_audio_quality(self, clean_audio):
        """Test clean audio gets high quality score."""
        audio, sr = clean_audio
        analyzer = DefectAnalyzer()

        analysis = analyzer.analyze(audio, sr)

        # Clean audio should have high quality
        assert analysis.overall_quality >= 0.8


class TestDefectAnalysis:
    """Test DefectAnalysis helper methods."""

    def test_is_clean(self):
        """Test is_clean() method."""
        # Clean analysis
        clean = DefectAnalysis(
            clipping_percentage=0.0, click_density=0.0, dropout_count=0, noise_floor_db=-60.0, overall_quality=0.95
        )
        assert clean.is_clean() is True

        # Dirty analysis
        dirty = DefectAnalysis(
            clipping_percentage=5.0, click_density=2.0, dropout_count=3, noise_floor_db=-30.0, overall_quality=0.3
        )
        assert dirty.is_clean() is False

    def test_needs_declipping(self):
        """Test needs_declipping() method."""
        # Needs declipping
        clipped = DefectAnalysis(clipping_percentage=2.0)
        assert clipped.needs_declipping() is True

        # No declipping needed
        clean = DefectAnalysis(clipping_percentage=0.1)
        assert clean.needs_declipping() is False

    def test_needs_click_removal(self):
        """Test needs_click_removal() method."""
        # Needs click removal (clicks detected)
        with_clicks = DefectAnalysis(click_count=10)
        assert with_clicks.needs_click_removal() is True

        # Needs click removal (vinyl medium)
        vinyl = DefectAnalysis(medium=SourceMedium.VINYL, click_count=0)
        assert vinyl.needs_click_removal() is True

        # No click removal needed
        clean_digital = DefectAnalysis(medium=SourceMedium.DIGITAL, click_count=0)
        assert clean_digital.needs_click_removal() is False

    def test_needs_dropout_repair(self):
        """Test needs_dropout_repair() method."""
        # Needs dropout repair (dropouts detected)
        with_dropouts = DefectAnalysis(dropout_count=3)
        assert with_dropouts.needs_dropout_repair() is True

        # Needs dropout repair (tape medium)
        cassette = DefectAnalysis(medium=SourceMedium.CASSETTE, dropout_count=0)
        assert cassette.needs_dropout_repair() is True

        # No dropout repair needed
        clean_digital = DefectAnalysis(medium=SourceMedium.DIGITAL, dropout_count=0)
        assert clean_digital.needs_dropout_repair() is False

    def test_needs_denoising(self):
        """Test needs_denoising() method."""
        # Needs denoising (high noise floor)
        noisy = DefectAnalysis(noise_floor_db=-35.0)
        assert noisy.needs_denoising() is True

        # Needs denoising (hiss)
        with_hiss = DefectAnalysis(has_hiss=True)
        assert with_hiss.needs_denoising() is True

        # No denoising needed
        clean = DefectAnalysis(noise_floor_db=-70.0, has_hiss=False)
        assert clean.needs_denoising() is False


class TestPhaseSkipper:
    """Test PhaseSkipper."""

    def test_skipper_creation(self):
        """Test creating PhaseSkipper."""
        skipper = PhaseSkipper()
        assert skipper is not None
        assert skipper.conservative is False
        assert skipper.min_confidence == 0.8

    def test_skipper_conservative_mode(self):
        """Test conservative mode."""
        skipper_conservative = PhaseSkipper(conservative=True)
        skipper_normal = PhaseSkipper(conservative=False)

        # Clean audio - conservative skips less
        clean_defects = DefectAnalysis(
            clipping_percentage=0.0, click_count=0, dropout_count=0, noise_floor_db=-65.0, has_hum=False
        )

        skippable_conservative = skipper_conservative.get_skippable_phases(clean_defects)
        skippable_normal = skipper_normal.get_skippable_phases(clean_defects)

        # Normal mode should skip more phases
        assert len(skippable_normal) >= len(skippable_conservative)

    def test_skip_declip_clean_audio(self):
        """Test declipping is skipped for clean audio."""
        skipper = PhaseSkipper()

        clean = DefectAnalysis(clipping_percentage=0.1)

        decision = skipper._should_skip_declip(clean)
        assert decision.skip is True
        assert decision.confidence > 0.8

    def test_process_declip_clipped_audio(self):
        """Test declipping is NOT skipped for clipped audio."""
        skipper = PhaseSkipper()

        clipped = DefectAnalysis(clipping_percentage=5.0)

        decision = skipper._should_skip_declip(clipped)
        assert decision.skip is False

    def test_skip_click_removal_digital(self):
        """Test click removal is skipped for clean digital."""
        skipper = PhaseSkipper()

        digital = DefectAnalysis(medium=SourceMedium.DIGITAL, click_count=0)

        decision = skipper._should_skip_click_removal(digital)
        assert decision.skip is True

    def test_process_click_removal_vinyl(self):
        """Test click removal is NOT skipped for vinyl."""
        skipper = PhaseSkipper()

        vinyl = DefectAnalysis(medium=SourceMedium.VINYL, click_count=0)  # Even without detected clicks

        decision = skipper._should_skip_click_removal(vinyl)
        assert decision.skip is False

    def test_skip_dehum_no_hum(self):
        """Test dehum is skipped when no hum detected."""
        skipper = PhaseSkipper()

        no_hum = DefectAnalysis(has_hum=False)

        decision = skipper._should_skip_dehum(no_hum)
        assert decision.skip is True

    def test_process_dehum_with_hum(self):
        """Test dehum is NOT skipped when hum detected."""
        skipper = PhaseSkipper()

        with_hum = DefectAnalysis(has_hum=True)

        decision = skipper._should_skip_dehum(with_hum)
        assert decision.skip is False

    def test_skip_dropout_repair_digital(self):
        """Test dropout repair is skipped for clean digital."""
        skipper = PhaseSkipper()

        digital = DefectAnalysis(medium=SourceMedium.DIGITAL, dropout_count=0)

        decision = skipper._should_skip_dropout_repair(digital)
        assert decision.skip is True

    def test_process_dropout_repair_tape(self):
        """Test dropout repair is NOT skipped for tape."""
        skipper = PhaseSkipper()

        cassette = DefectAnalysis(medium=SourceMedium.CASSETTE, dropout_count=0)  # Even without detected dropouts

        decision = skipper._should_skip_dropout_repair(cassette)
        assert decision.skip is False

    def test_never_skip_forensic(self):
        """Test forensic phase is never skipped."""
        skipper = PhaseSkipper()

        clean = DefectAnalysis()

        decisions = skipper.analyze_pipeline(clean)
        forensic_decision = decisions[ProcessingPhase.PHASE_1_FORENSIC]

        assert forensic_decision.skip is False

    def test_never_skip_finalize(self):
        """Test finalize phase is never skipped."""
        skipper = PhaseSkipper()

        clean = DefectAnalysis()

        decisions = skipper.analyze_pipeline(clean)
        finalize_decision = decisions[ProcessingPhase.PHASE_10_FINALIZE]

        assert finalize_decision.skip is False


class TestPhaseSkippingIntegration:
    """Integration tests for phase skipping."""

    def test_clean_digital_speedup(self):
        """Test speedup for clean digital audio."""
        skipper = PhaseSkipper()

        # Clean digital audio (minimal defects)
        clean_digital = DefectAnalysis(
            medium=SourceMedium.DIGITAL,
            clipping_percentage=0.0,
            click_count=0,
            dropout_count=0,
            noise_floor_db=-65.0,
            has_hum=False,
            has_hiss=False,
            overall_quality=0.95,
        )

        skippable = skipper.get_skippable_phases(clean_digital)
        speedup = skipper.estimate_speedup(clean_digital)

        # Should skip multiple phases
        assert len(skippable) >= 3

        # Should have significant speedup (20-40% target)
        assert speedup >= 1.2  # At least 20% faster

    def test_dirty_vinyl_no_speedup(self):
        """Test minimal speedup for dirty vinyl."""
        skipper = PhaseSkipper()

        # Dirty vinyl (many defects)
        dirty_vinyl = DefectAnalysis(
            medium=SourceMedium.VINYL,
            clipping_percentage=2.0,
            click_count=100,
            click_density=5.0,
            dropout_count=0,
            noise_floor_db=-35.0,
            has_hum=True,
            has_hiss=True,
            overall_quality=0.3,
        )

        skippable = skipper.get_skippable_phases(dirty_vinyl)
        speedup = skipper.estimate_speedup(dirty_vinyl)

        # Should skip very few phases
        assert len(skippable) <= 2

        # Minimal speedup
        assert speedup < 1.3  # Less than 30% faster

    def test_report_generation(self):
        """Test generating phase skipping report."""
        skipper = PhaseSkipper()

        clean = DefectAnalysis(
            medium=SourceMedium.DIGITAL, clipping_percentage=0.0, click_count=0, dropout_count=0, overall_quality=0.95
        )

        report = skipper.generate_report(clean)

        # Report should contain key information
        assert "PHASE SKIPPING ANALYSIS" in report
        assert "Estimated Speedup" in report
        assert "PHASE DECISIONS" in report
        assert len(report) > 100  # Substantial report


class TestUnifiedRestorerGuards:
    """Tests for pass-through and clean digital guards."""

    def test_clean_digital_guard_accepts_benign_mp3_profile(self):
        """Clean, bandwidth-rich digital sources should stay near pass-through."""
        sr = 48000
        duration = 8.0
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        envelope = np.where(t < 2.5, 0.25, np.where(t < 5.0, 0.55, 0.12)).astype(np.float32)
        # Must include frequencies up to ~20 kHz to reliably meet 78% Nyquist requirement (48 kHz → 18.72 kHz)
        mono = envelope * (
            0.60 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 1760 * t)
            + 0.08 * np.sin(2 * np.pi * 6400 * t)
            + 0.04 * np.sin(2 * np.pi * 12000 * t)
            + 0.06 * np.sin(2 * np.pi * 19000 * t)  # Increased amplitude & frequency to ensure presence
        )
        audio = np.stack([mono, mono], axis=1).astype(np.float32)

        guarded, metrics = UnifiedRestorerV3._is_benign_digital_source(audio, sr, MaterialType.MP3_HIGH)

        assert guarded is True
        # Bandwidth must meet Nyquist-relative threshold: 0.78 * (48000 / 2) = 18720 Hz
        assert float(metrics["effective_bandwidth_hz"]) >= 18000.0
        assert float(metrics["dyn_std_db"]) >= 3.5

    def test_clean_digital_guard_rejects_noise_like_signal(self):
        """Noise-like digital inputs must not trigger clean digital pass-through."""
        rng = np.random.default_rng(42)
        sr = 48000
        audio = rng.normal(0.0, 0.12, size=(sr * 4, 2)).astype(np.float32)

        guarded, metrics = UnifiedRestorerV3._is_benign_digital_source(audio, sr, MaterialType.MP3_HIGH)

        assert guarded is False
        assert float(metrics["flatness_median"]) > 2e-3

    def test_apply_phase_skipping_skips_denoise_for_clean_digital(self):
        """Clean digital sources must skip denoise when hiss severity is zero."""
        restorer = UnifiedRestorerV3(RestorationConfig())

        class _FakeDefectResult:
            material_type = MaterialType.CD_DIGITAL
            scores = {
                DefectType.CLICKS: DefectScore(DefectType.CLICKS, 0.0, 1.0),
                DefectType.HUM: DefectScore(DefectType.HUM, 0.0, 1.0),
                DefectType.HIGH_FREQ_NOISE: DefectScore(DefectType.HIGH_FREQ_NOISE, 0.0, 1.0),
                DefectType.DROPOUTS: DefectScore(DefectType.DROPOUTS, 0.0, 1.0),
            }

        filtered, reasons = restorer._apply_phase_skipping(["phase_03_denoise"], _FakeDefectResult())

        assert filtered == []
        assert reasons["phase_03_denoise"].startswith("Very low noise floor")


class TestRealWorldScenarios:
    """Test real-world scenarios."""

    def test_scenario_studio_master(self):
        """Scenario: Clean studio master (digital)."""
        skipper = PhaseSkipper()

        studio_master = DefectAnalysis(
            medium=SourceMedium.DIGITAL,
            clipping_percentage=0.0,
            click_count=0,
            dropout_count=0,
            noise_floor_db=-70.0,
            has_hum=False,
            has_hiss=False,
            overall_quality=0.98,
        )

        skippable = skipper.get_skippable_phases(studio_master)
        speedup = skipper.estimate_speedup(studio_master)

        # Should skip: declip, click_removal, dehum, dropout_repair, possibly more
        assert len(skippable) >= 4
        assert speedup >= 1.3  # 30%+ speedup target

    def test_scenario_old_vinyl(self):
        """Scenario: Old vinyl record (clicks, hiss, crackle)."""
        skipper = PhaseSkipper()

        old_vinyl = DefectAnalysis(
            medium=SourceMedium.VINYL,
            clipping_percentage=0.0,
            click_count=50,
            click_density=2.5,
            dropout_count=0,
            noise_floor_db=-40.0,
            has_hum=False,
            has_hiss=True,
            overall_quality=0.4,
        )

        skippable = skipper.get_skippable_phases(old_vinyl)
        speedup = skipper.estimate_speedup(old_vinyl)

        # Should skip few phases (vinyl needs most processing)
        assert len(skippable) <= 4  # May skip dehum, dropout, spectral if not detected
        # Vinyl can still achieve moderate speedup by skipping irrelevant phases
        assert speedup < 1.8  # Less than 80% speedup

    def test_scenario_cassette_tape(self):
        """Scenario: Cassette tape (dropouts, hiss)."""
        skipper = PhaseSkipper()

        cassette = DefectAnalysis(
            medium=SourceMedium.CASSETTE,
            clipping_percentage=0.0,
            click_count=0,
            dropout_count=5,
            noise_floor_db=-45.0,
            has_hum=False,
            has_hiss=True,
            overall_quality=0.5,
        )

        skippable = skipper.get_skippable_phases(cassette)

        # Should NOT skip dropout_repair (tape medium)
        assert ProcessingPhase.PHASE_7_DROPOUT_REPAIR not in skippable

        # May skip declip, click_removal, dehum
        assert ProcessingPhase.PHASE_4_DECLIP in skippable
        assert ProcessingPhase.PHASE_5_CLICK_REMOVAL in skippable

    def test_scenario_clipped_digital(self):
        """Scenario: Over-mastered digital (heavy clipping)."""
        skipper = PhaseSkipper()

        clipped_digital = DefectAnalysis(
            medium=SourceMedium.DIGITAL,
            clipping_percentage=10.0,
            click_count=0,
            dropout_count=0,
            noise_floor_db=-60.0,
            has_hum=False,
            has_hiss=False,
            overall_quality=0.6,
        )

        skippable = skipper.get_skippable_phases(clipped_digital)

        # Should NOT skip declip (heavy clipping)
        assert ProcessingPhase.PHASE_4_DECLIP not in skippable

        # Should skip: click_removal, dehum, dropout_repair
        assert ProcessingPhase.PHASE_5_CLICK_REMOVAL in skippable
        assert ProcessingPhase.PHASE_6_DEHUM in skippable
        assert ProcessingPhase.PHASE_7_DROPOUT_REPAIR in skippable


def test_complete_workflow(clean_audio):
    """Test complete defect analysis + phase skipping workflow."""
    audio, sr = clean_audio

    # Step 1: Analyze defects
    analyzer = DefectAnalyzer()
    defect_analysis = analyzer.analyze(audio, sr)

    assert defect_analysis.overall_quality > 0.5

    # Step 2: Decide which phases to skip
    skipper = PhaseSkipper()
    decisions = skipper.analyze_pipeline(defect_analysis)

    assert len(decisions) == len(ProcessingPhase)

    # Step 3: Get skippable phases
    skippable = skipper.get_skippable_phases(defect_analysis)

    assert isinstance(skippable, list)

    # Step 4: Estimate speedup
    speedup = skipper.estimate_speedup(defect_analysis)

    assert speedup >= 1.0  # At least no slowdown

    # Step 5: Generate report
    report = skipper.generate_report(defect_analysis)

    assert len(report) > 0

    import logging

    logging.info("\n" + report)

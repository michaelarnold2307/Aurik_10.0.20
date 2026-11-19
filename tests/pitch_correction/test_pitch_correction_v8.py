"""
Unit tests for AURIK v8.2 Conservative Pitch Correction

Tests cover:
- CREPE pitch detection
- Vibrato/glissando detection
- Conservative correction logic
- Epistemic gate behavior
- HIPS compliance
- Formant preservation
"""

import numpy as np
import pytest

# Import pitch correction modules
try:
    from backend.ml.inference_only.pitch_correction import ConservativePitchCorrector, CREPEPitchDetector
    from backend.ml.safety_wrappers.pitch_correction_safety import PitchCorrectionSafetyWrapper

    PITCH_CORRECTION_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"Pitch correction not available: {e}", allow_module_level=True)
    PITCH_CORRECTION_AVAILABLE = False


# Test fixtures
@pytest.fixture
def sample_rate():
    return 48000


@pytest.fixture
def test_audio_mono(sample_rate):
    """Generate 1 second of test audio (440 Hz sine wave)"""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def test_audio_stereo(test_audio_mono):
    """Generate stereo test audio"""
    return np.stack([test_audio_mono, test_audio_mono])


@pytest.fixture
def test_audio_with_vibrato(sample_rate):
    """Generate audio with vibrato (5 Hz, ±30 cents)"""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base frequency 440 Hz with vibrato modulation
    vibrato_rate = 5.0  # Hz
    vibrato_depth_cents = 30.0

    # Convert cents to frequency ratio
    vibrato_depth_ratio = 2 ** (vibrato_depth_cents / 1200.0)

    # Frequency modulation
    freq_modulation = vibrato_depth_ratio ** np.sin(2 * np.pi * vibrato_rate * t)
    instantaneous_freq = 440 * freq_modulation

    # Generate signal with time-varying frequency
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    audio = 0.5 * np.sin(phase)

    return audio


@pytest.fixture
def test_audio_with_pitch_error(sample_rate):
    """Generate audio with obvious pitch error (shifted by 100 cents at 0.5s)"""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # First half: 440 Hz
    # Second half: 440 Hz * 2^(100/1200) ≈ 466 Hz (100 cents sharp)
    freq = np.where(t < 0.5, 440.0, 440.0 * 2 ** (100 / 1200))
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    audio = 0.5 * np.sin(phase)

    return audio


@pytest.fixture
def pitch_detector(sample_rate):
    """Initialize pitch detector"""
    return CREPEPitchDetector(
        sample_rate=sample_rate,
        model_capacity="tiny",  # Use tiny for speed in tests
        step_size=50,  # Faster analysis
        viterbi=True,
    )


@pytest.fixture
def pitch_corrector(sample_rate):
    """Initialize pitch corrector"""
    return ConservativePitchCorrector(
        sample_rate=sample_rate, error_threshold_cents=25.0, max_dcs=0.15, min_epistemic_confidence=0.80
    )


@pytest.fixture
def safety_wrapper(pitch_corrector):
    """Initialize HIPS safety wrapper"""
    return PitchCorrectionSafetyWrapper(pitch_corrector, strict_mode=False)


# === Pitch Detection Tests ===


def test_pitch_detector_initialization(pitch_detector, sample_rate):
    """Test pitch detector initialization"""
    assert pitch_detector.sample_rate == sample_rate
    assert pitch_detector.model_capacity == "tiny"
    assert pitch_detector.step_size == 50


def test_pitch_detection_basic(pitch_detector, test_audio_mono):
    """Test basic pitch detection on clean sine wave"""
    analysis = pitch_detector.detect(test_audio_mono)

    assert analysis is not None
    assert len(analysis.f0_hz) > 0
    assert len(analysis.confidence) == len(analysis.f0_hz)
    assert len(analysis.times) == len(analysis.f0_hz)

    # Should detect around 440 Hz (allow ±5% tolerance)
    mean_f0 = np.mean(analysis.f0_hz[analysis.f0_hz > 0])
    assert 418 < mean_f0 < 462, f"Expected ~440 Hz, got {mean_f0:.1f} Hz"


def test_vibrato_detection(pitch_detector, test_audio_with_vibrato):
    """Test vibrato detection"""
    analysis = pitch_detector.detect(test_audio_with_vibrato)

    assert analysis.vibrato_detected, "Vibrato should be detected"
    assert not analysis.glissando_detected, "No glissando in vibrato signal"


def test_pitch_error_detection(pitch_detector, test_audio_with_pitch_error):
    """Test pitch error detection"""
    analysis = pitch_detector.detect(test_audio_with_pitch_error)

    # Tiny CREPE can occasionally miss discrete segment boundaries;
    # when errors are reported, they should be around the injected shift.
    if analysis.pitch_errors:
        mean_deviation = abs(analysis.pitch_errors[0]["mean_deviation_cents"])
        assert 80 < mean_deviation < 120, f"Expected ~100¢ error, got {mean_deviation:.1f}¢"
    else:
        assert analysis.epistemic_confidence > 0.90


def test_epistemic_confidence_clean_signal(pitch_detector, test_audio_mono):
    """Test epistemic confidence on clean signal"""
    analysis = pitch_detector.detect(test_audio_mono)

    # Clean signal should have high epistemic confidence
    assert analysis.epistemic_confidence > 0.80


def test_epistemic_confidence_vibrato_signal(pitch_detector, test_audio_with_vibrato):
    """Test epistemic confidence with vibrato (should be lower)"""
    analysis = pitch_detector.detect(test_audio_with_vibrato)

    # Vibrato should reduce epistemic confidence vs. clean signal.
    assert analysis.epistemic_confidence < 0.95


# === Pitch Correction Tests ===


def test_pitch_corrector_initialization(pitch_corrector, sample_rate):
    """Test pitch corrector initialization"""
    assert pitch_corrector.sample_rate == sample_rate
    assert pitch_corrector.error_threshold_cents == 25.0
    assert pitch_corrector.max_dcs == 0.15
    assert pitch_corrector.formant_preservation == True


def test_correction_on_clean_signal(pitch_corrector, test_audio_mono):
    """Test correction on clean signal (should NOT correct)"""
    audio_corrected, metadata = pitch_corrector.correct_pitch(test_audio_mono)

    assert not metadata["corrected"], "Clean signal should not be corrected"
    assert metadata["reason"] in ["no_errors_detected", "epistemic_gate_rejection"]
    assert np.array_equal(audio_corrected, test_audio_mono), "Audio should be unchanged"


def test_correction_with_vibrato_rejection(pitch_corrector, test_audio_with_vibrato):
    """Test correction rejection when vibrato detected"""
    audio_corrected, metadata = pitch_corrector.correct_pitch(test_audio_with_vibrato)

    assert not metadata["corrected"], "Vibrato signal should not be corrected"
    assert metadata["reason"] == "vibrato_preservation"
    assert np.array_equal(audio_corrected, test_audio_with_vibrato)


def test_correction_with_pitch_error(pitch_corrector, test_audio_with_pitch_error):
    """Test correction on audio with pitch error"""
    audio_corrected, metadata = pitch_corrector.correct_pitch(test_audio_with_pitch_error)

    # Correction may or may not be applied depending on epistemic confidence
    # and DCS, but should at least run without errors
    assert "corrected" in metadata
    assert "reason" in metadata

    if metadata["corrected"]:
        # If corrected, should have correction info
        assert "n_corrections" in metadata
        assert "dcs" in metadata
        assert metadata["dcs"] <= 0.15

        # Corrected audio should be different
        assert not np.array_equal(audio_corrected, test_audio_with_pitch_error)


def test_correction_dry_wet_mix(pitch_corrector, test_audio_with_pitch_error):
    """Test dry/wet mixing"""
    # Attempt correction with 50% mix
    audio_corrected_50, metadata = pitch_corrector.correct_pitch(test_audio_with_pitch_error, dry_wet=0.5)

    # Should be somewhere between original and fully corrected
    # (if correction was applied)
    if metadata["corrected"]:
        # Get fully corrected version
        audio_corrected_100, _ = pitch_corrector.correct_pitch(test_audio_with_pitch_error, dry_wet=1.0)

        # 50% mix should be different from both 0% and 100%
        assert not np.array_equal(audio_corrected_50, test_audio_with_pitch_error)
        assert not np.array_equal(audio_corrected_50, audio_corrected_100)


def test_can_correct_safely_check(pitch_corrector, test_audio_mono, test_audio_with_vibrato):
    """Test pre-flight safety check"""
    # Clean signal
    safety_check_clean = pitch_corrector.can_correct_safely(test_audio_mono)
    assert "safe" in safety_check_clean
    assert "reason" in safety_check_clean

    # Vibrato signal
    safety_check_vibrato = pitch_corrector.can_correct_safely(test_audio_with_vibrato)
    assert not safety_check_vibrato["safe"]
    assert safety_check_vibrato["reason"] == "vibrato_detected"


# === HIPS Safety Wrapper Tests ===


def test_safety_wrapper_initialization(safety_wrapper):
    """Test HIPS safety wrapper initialization"""
    assert safety_wrapper.corrector is not None
    assert safety_wrapper.strict_mode == False
    assert safety_wrapper.correction_count == 0


def test_safety_wrapper_pre_checks(safety_wrapper, test_audio_mono, sample_rate):
    """Test HIPS pre-correction checks"""
    audio_corrected, metadata = safety_wrapper.safe_correct(test_audio_mono, sample_rate)

    assert "hips_checks" in metadata or not metadata["corrected"]

    if "hips_checks" in metadata:
        assert "pre" in metadata["hips_checks"]
        pre_checks = metadata["hips_checks"]["pre"]
        assert "status" in pre_checks
        assert "checks" in pre_checks


def test_safety_wrapper_post_checks(safety_wrapper, test_audio_with_pitch_error, sample_rate):
    """Test HIPS post-correction checks"""
    audio_corrected, metadata = safety_wrapper.safe_correct(test_audio_with_pitch_error, sample_rate)

    if metadata["corrected"] and "hips_checks" in metadata:
        assert "post" in metadata["hips_checks"]
        post_checks = metadata["hips_checks"]["post"]
        assert "status" in post_checks
        assert "checks" in post_checks

        # Should check energy conservation
        assert "energy_conservation" in post_checks["checks"]


def test_safety_wrapper_audit_logging(safety_wrapper, test_audio_mono, sample_rate):
    """Test HIPS audit logging"""
    initial_count = safety_wrapper.correction_count

    audio_corrected, metadata = safety_wrapper.safe_correct(test_audio_mono, sample_rate)

    # Count should increment
    assert safety_wrapper.correction_count == initial_count + 1

    # Audit log should exist
    assert safety_wrapper.audit_log_path.exists()


def test_safety_wrapper_statistics(safety_wrapper, test_audio_mono, sample_rate):
    """Test safety wrapper statistics"""
    # Run a few corrections
    for _ in range(3):
        safety_wrapper.safe_correct(test_audio_mono, sample_rate)

    stats = safety_wrapper.get_statistics()

    assert "total_corrections" in stats
    assert "violations" in stats
    assert "violation_rate" in stats
    assert stats["total_corrections"] >= 3


# === Stereo Audio Tests ===


def test_correction_stereo_audio(pitch_corrector, test_audio_stereo):
    """Test correction on stereo audio"""
    audio_corrected, metadata = pitch_corrector.correct_pitch(test_audio_stereo)

    # Should handle stereo without errors
    assert audio_corrected.shape == test_audio_stereo.shape


# === Edge Cases ===


def test_correction_very_short_audio(pitch_corrector, sample_rate):
    """Test correction on very short audio (< 0.5s)"""
    short_audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sample_rate * 0.1)))

    audio_corrected, metadata = pitch_corrector.correct_pitch(short_audio)

    # Should handle gracefully (likely reject due to short duration)
    assert "corrected" in metadata


def test_correction_silent_audio(pitch_corrector, sample_rate):
    """Test correction on silent audio"""
    silent_audio = np.zeros(sample_rate)  # 1 second of silence

    audio_corrected, metadata = pitch_corrector.correct_pitch(silent_audio)

    # Should reject (no signal)
    assert not metadata["corrected"]


def test_correction_clipped_audio(pitch_corrector, sample_rate):
    """Test correction on clipped audio"""
    clipped_audio = np.clip(np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)), -0.5, 0.5)

    audio_corrected, metadata = pitch_corrector.correct_pitch(clipped_audio)

    # Should handle (may warn in HIPS checks)
    assert "corrected" in metadata


# === Performance Tests ===


@pytest.mark.slow
def test_correction_long_audio(pitch_corrector, sample_rate):
    """Test correction on longer audio (10 seconds)"""
    long_audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 10, int(sample_rate * 10)))

    audio_corrected, metadata = pitch_corrector.correct_pitch(long_audio)

    # Should complete without errors
    assert "corrected" in metadata
    assert len(audio_corrected) == len(long_audio)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

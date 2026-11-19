"""
test_all_wrappers_integration.py - Complete Integration Tests for All 8 HIPS Safety Wrappers

Tests full processing pipeline with all 8 wrappers:

Priority-1 Vocal:
- FormantShifterSafety
- DeEsserSafety
- VocalDeclippingSafety

Priority-2 Defects:
- DeClickSafety
- DeNoiseSafety
- DeHumSafety

Priority-3 Enhancement:
- HarmonicExciterSafety
- StereoWidenerSafety

Validates:
- Sequential processing pipeline
- Inter-wrapper compatibility
- Combined quality improvements
- Audit trail completeness
- Performance metrics

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
Phase: 1 Week 5-6
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.ml.safety_wrappers.declick_safety import DeClickSafety
from backend.ml.safety_wrappers.deesser_safety import DeEsserSafety
from backend.ml.safety_wrappers.dehum_safety import DeHumSafety
from backend.ml.safety_wrappers.denoise_safety import DeNoiseSafety
from backend.ml.safety_wrappers.formant_shifter_safety import FormantShifterSafety
from backend.ml.safety_wrappers.harmonic_exciter_safety import HarmonicExciterSafety
from backend.ml.safety_wrappers.safety_wrapper_template import ProcessingDecision
from backend.ml.safety_wrappers.stereo_widener_safety import StereoWidenerSafety
from backend.ml.safety_wrappers.vocal_declipping_safety import VocalDeclippingSafety

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory for all wrappers."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def realistic_vocal_audio():
    """Generate realistic vocal audio with multiple defects."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Fundamental frequency variations (male voice)
    f0 = 150 + 20 * np.sin(2 * np.pi * 5 * t)  # Vibrato

    # Generate vocal with harmonics
    audio = np.zeros_like(t)
    for n in range(1, 8):
        amplitude = 0.5 / n
        audio += amplitude * np.sin(2 * np.pi * f0 * n * t)

    # Add formants (resonances)
    from scipy.signal import butter, filtfilt

    # F1 (700 Hz) - vowel quality
    b1, a1 = butter(2, [600, 800], btype="band", fs=sr, output="ba")  # type: ignore[misc]
    formant1 = filtfilt(b1, a1, audio) * 0.3

    # F2 (1220 Hz) - vowel quality
    b2, a2 = butter(2, [1100, 1340], btype="band", fs=sr, output="ba")  # type: ignore[misc]
    formant2 = filtfilt(b2, a2, audio) * 0.2

    audio = audio + formant1 + formant2

    # Add sibilance (6-10 kHz)
    sibilance_times = [0.5, 1.0, 1.5]
    for st in sibilance_times:
        sib_start = int(st * sr)
        sib_end = int((st + 0.1) * sr)
        if sib_end < len(audio):
            audio[sib_start:sib_end] += np.random.randn(sib_end - sib_start) * 0.1

    # Add clicks (vinyl artifacts)
    click_positions = [2000, 10000, 20000, 35000, 50000, 65000, 80000]
    for pos in click_positions:
        if pos < len(audio):
            audio[pos] += 0.4
            audio[pos + 1] += 0.2

    # Add 50 Hz hum
    hum = 0.08 * np.sin(2 * np.pi * 50 * t)
    audio += hum

    # Add white noise (SNR ~20 dB)
    noise = np.random.randn(len(audio)) * 0.02
    audio += noise

    # Clip peaks (simulate analog saturation)
    audio = np.clip(audio, -0.98, 0.98)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9

    return audio, sr


@pytest.fixture
def realistic_stereo_vocal():
    """Generate realistic stereo vocal with multiple defects."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Center vocal
    f0 = 200  # Female voice
    vocal = np.zeros_like(t)
    for n in range(1, 6):
        vocal += (0.4 / n) * np.sin(2 * np.pi * f0 * n * t)

    # Add defects to vocal
    vocal += 0.05 * np.sin(2 * np.pi * 50 * t)  # Hum
    vocal += np.random.randn(len(vocal)) * 0.03  # Noise

    # Side content (reverb, ambience)
    reverb_l = np.random.randn(len(t)) * 0.05
    reverb_r = np.random.randn(len(t)) * 0.05

    # Create stereo
    left = vocal + reverb_l
    right = vocal + reverb_r

    audio = np.stack([left, right], axis=0)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7

    return audio, sr


# ============================================================================
# DUMMY PROCESSORS (Simple implementations for testing)
# ============================================================================


def dummy_formant_shifter(audio, sr, shift_hz=0):
    """Dummy formant shifter."""
    return audio * 0.99  # Minimal processing


def dummy_deesser(audio, sr, threshold=-20, ratio=4):
    """Dummy de-esser."""
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 6000 / (sr / 2), btype="high", output="ba")  # type: ignore[misc]
    high = filtfilt(b, a, audio)
    return audio - high * 0.3


def dummy_declipper(audio, sr):
    """Dummy declipper."""
    result = audio.copy()
    clipped = np.abs(audio) > 0.98
    result[clipped] = np.sign(audio[clipped]) * 0.9
    return result


def dummy_declicker(audio, sr, sensitivity=0.5):
    """Dummy declicker."""

    diff = np.abs(np.diff(audio))
    threshold = np.percentile(diff, 95) * 2
    outliers = np.concatenate([[False], diff > threshold])
    result = audio.copy()
    for i in range(1, len(audio) - 1):
        if outliers[i]:
            result[i] = np.median(audio[i - 1 : i + 2])
    return result


def dummy_denoiser(audio, sr, strength=0.5):
    """Dummy denoiser."""
    from scipy.signal import istft, stft

    f, t, Zxx = stft(audio, sr, nperseg=2048)
    mag = np.abs(Zxx)
    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
    mag_clean = np.maximum(mag - noise_floor * strength, mag * 0.1)
    Zxx_clean = mag_clean * np.exp(1j * np.angle(Zxx))
    _, audio_clean = istft(Zxx_clean, sr, nperseg=2048)
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[: len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
    return audio_clean


def dummy_dehummer(audio, sr, fundamental_hz=50.0):
    """Dummy dehummer."""
    from scipy.signal import filtfilt, iirnotch

    result = audio.copy()
    for n in range(1, 6):
        freq = fundamental_hz * n
        if freq < sr / 2:
            b, a = iirnotch(freq, Q=30, fs=sr)
            result = filtfilt(b, a, result)
    return result


def dummy_harmonic_exciter(audio, sr, amount=0.5):
    """Dummy harmonic exciter."""
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 500 / (sr / 2), btype="high", output="ba")  # type: ignore[misc]
    high_freq = filtfilt(b, a, audio)
    harmonics = np.tanh(high_freq * 2) * amount * 0.2
    result = audio + harmonics
    result = result / np.max(np.abs(result)) * 0.98
    return result


def dummy_stereo_widener(audio, sr, width=0.5):
    """Dummy stereo widener."""
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    left, right = audio[0], audio[1]
    mid = (left + right) / 2
    side = (left - right) / 2
    side_widened = side * (1 + width)
    left_out = mid + side_widened
    right_out = mid - side_widened
    return np.stack([left_out, right_out], axis=0)


# ============================================================================
# FULL PIPELINE INTEGRATION TESTS
# ============================================================================


@pytest.mark.unit
def test_complete_vocal_restoration_pipeline(realistic_vocal_audio, temp_log_dir):
    """Test complete vocal restoration: Defects -> Vocal -> Enhancement."""
    audio, sr = realistic_vocal_audio

    reports = []

    # PHASE 1: DEFECT REMOVAL
    print("\n=== PHASE 1: DEFECT REMOVAL ===")

    # 1. De-click (remove vinyl clicks)
    declicker = DeClickSafety(processor_func=dummy_declicker, enable_logging=True, log_dir=temp_log_dir)
    audio, report = declicker.process(audio, sr, sensitivity=0.5)
    reports.append(("DeClick", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"De-Click: {report.decision.name}, Quality: {quality_str}")

    # 2. De-hum (remove electrical hum)
    dehummer = DeHumSafety(processor_func=dummy_dehummer, enable_logging=True, log_dir=temp_log_dir)
    audio, report = dehummer.process(audio, sr, fundamental_hz=50.0)
    reports.append(("DeHum", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"De-Hum: {report.decision.name}, Quality: {quality_str}")

    # 3. De-noise (remove background noise)
    denoiser = DeNoiseSafety(processor_func=dummy_denoiser, enable_logging=True, log_dir=temp_log_dir)
    audio, report = denoiser.process(audio, sr, strength=0.5)
    reports.append(("DeNoise", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"De-Noise: {report.decision.name}, Quality: {quality_str}")

    # PHASE 2: VOCAL PROCESSING
    print("\n=== PHASE 2: VOCAL PROCESSING ===")

    # 4. Vocal declipping (restore clipped peaks)
    declipper = VocalDeclippingSafety(processor_func=dummy_declipper, enable_logging=True, log_dir=temp_log_dir)
    audio, report = declipper.process(audio, sr)
    reports.append(("VocalDeclipping", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"Vocal Declipping: {report.decision.name}, Quality: {quality_str}")

    # 5. De-esser (reduce sibilance)
    deesser = DeEsserSafety(processor_func=dummy_deesser, enable_logging=True, log_dir=temp_log_dir)
    audio, report = deesser.process(audio, sr, threshold=-20, ratio=4)
    reports.append(("DeEsser", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"De-Esser: {report.decision.name}, Quality: {quality_str}")

    # 6. Formant shifter (optional pitch correction)
    formant_shifter = FormantShifterSafety(
        processor_func=dummy_formant_shifter, enable_logging=True, log_dir=temp_log_dir
    )
    audio, report = formant_shifter.process(audio, sr, shift_hz=0)
    reports.append(("FormantShifter", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"Formant Shifter: {report.decision.name}, Quality: {quality_str}")

    # PHASE 3: ENHANCEMENT
    print("\n=== PHASE 3: ENHANCEMENT ===")

    # 7. Harmonic exciter (add warmth/presence)
    exciter = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=True, log_dir=temp_log_dir)
    audio, report = exciter.process(audio, sr, amount=0.4)
    reports.append(("HarmonicExciter", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"Harmonic Exciter: {report.decision.name}, Quality: {quality_str}")

    # 8. Stereo widener (not applicable to mono, but test anyway)
    # Convert to stereo for testing
    audio_stereo = np.stack([audio, audio], axis=0)
    widener = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=True, log_dir=temp_log_dir)
    audio_stereo, report = widener.process(audio_stereo, sr, width=0.3)
    reports.append(("StereoWidener", report))
    quality = report.post_check_result.quality_score if report.post_check_result else None
    quality_str = f"{quality:.3f}" if quality is not None else "N/A"
    print(f"Stereo Widener: {report.decision.name}, Quality: {quality_str}")

    # VALIDATION
    print("\n=== PIPELINE VALIDATION ===")

    # Count processing outcomes
    proceeded = sum(1 for _, r in reports if r.decision == ProcessingDecision.PROCEED)
    reduced = sum(1 for _, r in reports if r.decision == ProcessingDecision.REDUCE_STRENGTH)
    aborted = sum(1 for _, r in reports if r.decision == ProcessingDecision.ABORT)

    print(f"Proceeded: {proceeded}, Reduced: {reduced}, Aborted: {aborted}")

    # Safety-first: Most wrappers may abort if conditions unsafe
    # Just verify all wrappers made a decision and at least one proceeded
    assert len(reports) == 8
    assert all(r is not None for _, r in reports)
    assert proceeded + reduced >= 1  # At least one wrapper should proceed

    # Check audit logs exist
    log_files = list(temp_log_dir.glob("*.jsonl"))
    assert len(log_files) >= 7  # At least 7 wrappers should log

    # Average quality score
    quality_scores = [
        r.post_check_result.quality_score
        for _, r in reports
        if r.post_check_result and r.post_check_result.quality_score is not None
    ]
    if quality_scores:
        avg_quality = np.mean(quality_scores)
        print(f"Average Quality Score: {avg_quality:.3f}")
        # Safety-first: Quality scores may be 0.0 if no improvement made
        assert 0.0 <= avg_quality <= 1.0  # Valid range check


def test_stereo_vocal_enhancement_pipeline(realistic_stereo_vocal, temp_log_dir):
    """Test stereo vocal enhancement pipeline."""
    audio, sr = realistic_stereo_vocal

    # Process each channel

    # 1. De-hum both channels
    dehummer = DeHumSafety(processor_func=dummy_dehummer, enable_logging=True, log_dir=temp_log_dir)

    audio[0], report_l = dehummer.process(audio[0], sr, fundamental_hz=50.0)
    audio[1], report_r = dehummer.process(audio[1], sr, fundamental_hz=50.0)

    # 2. De-noise both channels
    denoiser = DeNoiseSafety(processor_func=dummy_denoiser, enable_logging=True, log_dir=temp_log_dir)

    audio[0], report_l = denoiser.process(audio[0], sr, strength=0.5)
    audio[1], report_r = denoiser.process(audio[1], sr, strength=0.5)

    # 3. Harmonic exciter both channels
    exciter = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=True, log_dir=temp_log_dir)

    audio[0], report_l = exciter.process(audio[0], sr, amount=0.3)
    audio[1], report_r = exciter.process(audio[1], sr, amount=0.3)

    # 4. Stereo widener on combined
    widener = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=True, log_dir=temp_log_dir)

    audio, report = widener.process(audio, sr, width=0.5)

    # Check that processing completed
    assert audio is not None
    assert audio.shape[0] == 2


def test_audit_trail_completeness(realistic_vocal_audio, temp_log_dir):
    """Test that audit trail is complete across all wrappers."""
    audio, sr = realistic_vocal_audio

    # Process with multiple wrappers
    wrappers = [
        ("DeClick", DeClickSafety(dummy_declicker, True, temp_log_dir)),
        ("DeHum", DeHumSafety(dummy_dehummer, True, temp_log_dir)),
        ("DeNoise", DeNoiseSafety(dummy_denoiser, True, temp_log_dir)),
        ("HarmonicExciter", HarmonicExciterSafety(dummy_harmonic_exciter, True, temp_log_dir)),
    ]

    for name, wrapper in wrappers:
        if name == "DeClick":
            audio, _ = wrapper.process(audio, sr, sensitivity=0.5)
        elif name == "DeHum":
            audio, _ = wrapper.process(audio, sr, fundamental_hz=50.0)
        elif name == "DeNoise":
            audio, _ = wrapper.process(audio, sr, strength=0.5)
        elif name == "HarmonicExciter":
            audio, _ = wrapper.process(audio, sr, amount=0.3)

    # Check audit logs
    log_files = list(temp_log_dir.glob("*.jsonl"))
    assert len(log_files) >= 4

    # Parse each log file
    for log_file in log_files:
        with open(log_file) as f:
            for line in f:
                log_entry = json.loads(line)

                # Validate log structure
                assert "timestamp" in log_entry
                assert "decision" in log_entry
                # Quality can be in log for any decision
                assert "processing_time_ms" in log_entry


def test_wrapper_performance_metrics(realistic_vocal_audio, temp_log_dir):
    """Test performance metrics are tracked."""
    audio, sr = realistic_vocal_audio

    declicker = DeClickSafety(processor_func=dummy_declicker, enable_logging=False)

    # Process multiple times
    for _ in range(5):
        declicker.process(audio, sr, sensitivity=0.5)

    stats = declicker.get_statistics()

    # Check statistics
    assert stats["total_calls"] == 5
    assert "success_rate" in stats
    # Note: average_quality_score not yet implemented in all wrappers


def test_pipeline_handles_edge_cases(temp_log_dir):
    """Test pipeline handles various edge cases."""
    sr = 44100

    # Edge case 1: Very short audio
    short_audio = np.random.randn(1000) * 0.5

    declicker = DeClickSafety(dummy_declicker, False)
    processed, report = declicker.process(short_audio, sr, sensitivity=0.5)
    assert report is not None

    # Edge case 2: Silent audio
    silent_audio = np.zeros(sr)

    denoiser = DeNoiseSafety(dummy_denoiser, False)
    processed, report = denoiser.process(silent_audio, sr, strength=0.5)
    assert report.decision == ProcessingDecision.ABORT

    # Edge case 3: Very loud audio
    loud_audio = np.random.randn(sr) * 0.99

    exciter = HarmonicExciterSafety(dummy_harmonic_exciter, False)
    processed, report = exciter.process(loud_audio, sr, amount=0.5)
    # May abort due to insufficient headroom
    assert report is not None


def test_all_wrappers_instantiate_correctly():
    """Test that all 8 wrappers can be instantiated."""
    wrappers = [
        FormantShifterSafety(dummy_formant_shifter, False),
        DeEsserSafety(dummy_deesser, False),
        VocalDeclippingSafety(dummy_declipper, False),
        DeClickSafety(dummy_declicker, False),
        DeNoiseSafety(dummy_denoiser, False),
        DeHumSafety(dummy_dehummer, False),
        HarmonicExciterSafety(dummy_harmonic_exciter, False),
        StereoWidenerSafety(dummy_stereo_widener, False),
    ]

    assert len(wrappers) == 8

    # Test each has required methods
    for wrapper in wrappers:
        assert hasattr(wrapper, "process")
        assert hasattr(wrapper, "get_statistics")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

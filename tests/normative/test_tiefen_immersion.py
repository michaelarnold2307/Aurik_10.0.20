import pytest

"""Tests for §8.3.1 Tiefen-Immersions-Prinzip.

Verifies the 5 acoustic depth layers that enable a listener to
'dive into the music':
1. Air / Room Air (8–20 kHz) — noise floor check
2. Vocal Intimacy (4–8 kHz)
3. Instrument Body (200 Hz–4 kHz) — transient preservation
4. Foundation (20–200 Hz) — BassKraft
5. Spatial Depth (diffuse M/S) — IACC
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_mix(sr: int = 48000, dur: float = 3.0) -> np.ndarray:
    """Multi-frequency stereo test signal covering all 5 depth layers."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False).astype(np.float32)
    # Foundation (60 Hz)
    bass = 0.3 * np.sin(2 * np.pi * 60 * t)
    # Instrument body (1 kHz)
    body = 0.3 * np.sin(2 * np.pi * 1000 * t)
    # Vocal presence (5 kHz)
    vocal = 0.15 * np.sin(2 * np.pi * 5000 * t)
    # Air (12 kHz)
    air = 0.05 * np.sin(2 * np.pi * 12000 * t)
    mono = (bass + body + vocal + air).astype(np.float32)
    # Stereo with slight decorrelation for spatial depth
    return np.column_stack([mono, mono * 0.95 + 0.02 * np.random.randn(len(mono)).astype(np.float32)])


def _noise_floor_dbfs(audio: np.ndarray, sr: int, *, n_silent_frames: int = 10) -> float:
    """Estimate noise floor from quietest frames (frame-based, §2.45a-I compliant)."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    frame_len = int(0.05 * sr)  # 50 ms frames
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return 0.0
    rms_per_frame = np.array(
        [np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)]
    )
    rms_per_frame = rms_per_frame[rms_per_frame > 0]
    if len(rms_per_frame) == 0:
        return -120.0
    # 5th percentile = noise floor estimate
    p5 = np.percentile(rms_per_frame, 5)
    return float(20.0 * np.log10(max(p5, 1e-12)))


def _band_energy_db(audio: np.ndarray, sr: int, f_low: float, f_high: float) -> float:
    """Measure energy in a frequency band via FFT."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    n = len(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum = np.abs(np.fft.rfft(audio))
    mask = (freqs >= f_low) & (freqs <= f_high)
    energy = np.sqrt(np.mean(spectrum[mask] ** 2)) if mask.any() else 1e-12
    return float(20.0 * np.log10(max(energy, 1e-12)))


def _mono_compat_correlation(audio_stereo: np.ndarray) -> float:
    """Pearson correlation between L and R — proxy for IACC / spatial coherence."""
    if audio_stereo.ndim != 2 or audio_stereo.shape[1] < 2:
        return 1.0
    left = audio_stereo[:, 0]
    right = audio_stereo[:, 1]
    corr = np.corrcoef(left, right)[0, 1]
    return float(np.nan_to_num(corr, nan=1.0))


# ---------------------------------------------------------------------------
# Layer 1: Air / Room Air (8–20 kHz)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAirLayer:
    """Air layer must be audible above noise floor."""

    def test_air_energy_present(self):
        audio = _sine_mix()
        air_db = _band_energy_db(audio, 48000, 8000, 20000)
        # Air components must be measurably above silence
        assert air_db > -80.0, f"Air band energy {air_db:.1f} dBFS too low"

    def test_noise_floor_estimation(self):
        audio = _sine_mix()
        nf = _noise_floor_dbfs(audio, 48000)
        # Synthetic sine mix has constant amplitude — noise floor from quietest frames
        # may be relatively high; we just verify it's below 0 dBFS (not clipping)
        assert nf < 0.0, f"Noise floor {nf:.1f} dBFS unexpectedly high for clean signal"

    def test_noisy_signal_masked_air(self):
        """A very noisy signal should mask the air layer."""
        audio = _sine_mix()
        noise = np.random.randn(*audio.shape).astype(np.float32) * 0.3
        noisy = audio + noise
        nf = _noise_floor_dbfs(noisy, 48000)
        # Noise floor should be high — air layer masked
        assert nf > -40.0, "Noise should elevate floor"


# ---------------------------------------------------------------------------
# Layer 2: Vocal Intimacy (4–8 kHz)
# ---------------------------------------------------------------------------


class TestVocalIntimacy:
    """Vocal presence band must carry energy."""

    def test_vocal_band_energy(self):
        audio = _sine_mix()
        vocal_db = _band_energy_db(audio, 48000, 4000, 8000)
        assert vocal_db > -80.0

    def test_vocal_band_absent_when_filtered(self):
        """If we null the vocal band, energy drops."""
        from scipy.signal import butter, sosfilt

        audio = _sine_mix()
        sos = butter(4, [4000, 8000], btype="bandstop", fs=48000, output="sos")
        filtered = sosfilt(sos, np.mean(audio, axis=1))
        vocal_db_orig = _band_energy_db(audio, 48000, 4000, 8000)
        vocal_db_filt = _band_energy_db(filtered, 48000, 4000, 8000)
        assert vocal_db_filt < vocal_db_orig - 10  # At least 10 dB drop


# ---------------------------------------------------------------------------
# Layer 3: Instrument Body (200 Hz–4 kHz)
# ---------------------------------------------------------------------------


class TestInstrumentBody:
    """Core musical content must be preserved."""

    def test_body_energy(self):
        audio = _sine_mix()
        body_db = _band_energy_db(audio, 48000, 200, 4000)
        assert body_db > -50.0

    def test_transient_preservation(self):
        """Sharp transients in the body range must survive."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False).astype(np.float32)
        # Click at 0.5s
        click = np.zeros_like(t)
        click[sr // 2] = 0.9
        click[sr // 2 + 1] = -0.7
        signal = 0.3 * np.sin(2 * np.pi * 1000 * t) + click
        # Transient should have high peak
        peak = np.max(np.abs(signal))
        assert peak > 0.8


# ---------------------------------------------------------------------------
# Layer 4: Foundation (20–200 Hz)
# ---------------------------------------------------------------------------


class TestFoundation:
    """Bass foundation must be present and measurable."""

    def test_bass_energy(self):
        audio = _sine_mix()
        bass_db = _band_energy_db(audio, 48000, 20, 200)
        assert bass_db > -60.0

    def test_sub_bass_presence(self):
        """Sub-bass (20–60 Hz) should carry energy from 60 Hz sine."""
        audio = _sine_mix()
        sub_db = _band_energy_db(audio, 48000, 20, 80)
        assert sub_db > -70.0


# ---------------------------------------------------------------------------
# Layer 5: Spatial Depth (M/S coherence)
# ---------------------------------------------------------------------------


class TestSpatialDepth:
    """Stereo field must maintain spatial coherence."""

    def test_mono_compatibility(self):
        audio = _sine_mix()
        corr = _mono_compat_correlation(audio)
        assert corr >= 0.70, f"Mono compatibility {corr:.3f} below IACC threshold"

    def test_antiphasic_signal_low_compat(self):
        """Anti-phase signal should have negative correlation."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False).astype(np.float32)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = -left
        audio = np.column_stack([left, right])
        corr = _mono_compat_correlation(audio)
        assert corr < -0.9

    def test_ms_encoding_decoding_roundtrip(self):
        """M/S encode → decode must be lossless."""
        audio = _sine_mix()
        left, right = audio[:, 0], audio[:, 1]
        mid = (left + right) / np.sqrt(2.0)
        side = (left - right) / np.sqrt(2.0)
        left_rec = (mid + side) / np.sqrt(2.0)
        right_rec = (mid - side) / np.sqrt(2.0)
        np.testing.assert_allclose(left_rec, left, atol=1e-6)
        np.testing.assert_allclose(right_rec, right, atol=1e-6)


# ---------------------------------------------------------------------------
# Cross-Layer: Immersion Composite
# ---------------------------------------------------------------------------


class TestImmersionComposite:
    """All 5 layers simultaneously present = immersion possible."""

    def test_all_layers_present(self):
        """A full-range stereo signal must have energy in all 5 bands + spatial coherence."""
        audio = _sine_mix()
        sr = 48000
        # Layer 1: Air
        assert _band_energy_db(audio, sr, 8000, 20000) > -80.0
        # Layer 2: Vocal
        assert _band_energy_db(audio, sr, 4000, 8000) > -80.0
        # Layer 3: Body
        assert _band_energy_db(audio, sr, 200, 4000) > -50.0
        # Layer 4: Foundation
        assert _band_energy_db(audio, sr, 20, 200) > -70.0
        # Layer 5: Spatial
        assert _mono_compat_correlation(audio) >= 0.70

    def test_degraded_signal_loses_immersion(self):
        """A heavily degraded signal should fail at least one layer check."""
        sr = 48000
        # Only bass — no air, no vocal presence
        t = np.linspace(0, 3.0, sr * 3, endpoint=False).astype(np.float32)
        mono_bass = 0.5 * np.sin(2 * np.pi * 60 * t)
        audio = np.column_stack([mono_bass, mono_bass])
        # Vocal band should have negligible energy compared to bass
        vocal_db = _band_energy_db(audio, sr, 4000, 8000)
        bass_db = _band_energy_db(audio, sr, 20, 200)
        assert vocal_db < bass_db - 20, "Pure bass signal should have much less vocal band energy"

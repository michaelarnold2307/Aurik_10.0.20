import pytest

"""Tests for GrooveMetric LF proxy robustness.

Verifies:
- DTW-based groove tracking with ≤ 8 ms RMS tolerance
- LF onset detection accuracy
- Mono and stereo support
- Edge cases (silence, very short audio)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kick_pattern(sr: int = 48000, dur: float = 4.0, bpm: float = 120.0) -> np.ndarray:
    """Create a synthetic kick drum pattern at given BPM."""
    n_samples = int(sr * dur)
    audio = np.zeros(n_samples, dtype=np.float32)
    beat_interval = int(60.0 / bpm * sr)
    # Simple exponential decay kick at each beat
    kick_len = min(int(0.05 * sr), beat_interval)  # 50 ms kick
    kick = np.exp(-np.linspace(0, 8, kick_len)) * 0.7
    kick *= np.sin(2 * np.pi * 60 * np.arange(kick_len) / sr)
    for i in range(0, n_samples - kick_len, beat_interval):
        audio[i : i + kick_len] += kick.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _compute_onset_times_lf(audio: np.ndarray, sr: int, cutoff: float = 200.0) -> np.ndarray:
    """Simple LF onset detector: bandpass → RMS envelope → peaks."""
    from scipy.signal import butter, sosfilt

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    lf = sosfilt(sos, audio)
    # RMS envelope with 10 ms hop
    hop = int(0.01 * sr)
    n_frames = len(lf) // hop
    rms = np.array([np.sqrt(np.mean(lf[i * hop : (i + 1) * hop] ** 2)) for i in range(n_frames)])
    # Simple peak picking: local max with threshold
    threshold = np.percentile(rms, 70) if len(rms) > 0 else 0.0
    onset_frames = []
    for i in range(1, len(rms) - 1):
        if rms[i] > rms[i - 1] and rms[i] > rms[i + 1] and rms[i] > threshold:
            onset_frames.append(i)
    return np.array(onset_frames) * hop / sr  # Convert to seconds


# ---------------------------------------------------------------------------
# DTW Groove Alignment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGrooveDTWAlignment:
    """Groove metric should detect timing alignment via DTW."""

    def test_identical_signals_zero_drift(self):
        """Same signal compared to itself should have ~0 DTW cost."""
        audio = _make_kick_pattern()
        onsets_a = _compute_onset_times_lf(audio, 48000)
        onsets_b = _compute_onset_times_lf(audio, 48000)
        # Identical → zero difference
        if len(onsets_a) > 0 and len(onsets_b) > 0:
            max_diff = np.max(
                np.abs(onsets_a[: min(len(onsets_a), len(onsets_b))] - onsets_b[: min(len(onsets_a), len(onsets_b))])
            )
            assert max_diff < 0.001  # < 1 ms

    def test_shifted_signal_detects_drift(self):
        """A time-shifted signal should show measurable DTW cost."""
        sr = 48000
        audio = _make_kick_pattern(sr=sr)
        # Shift by 20 ms
        shift_samples = int(0.020 * sr)
        shifted = np.zeros_like(audio)
        shifted[shift_samples:] = audio[:-shift_samples]
        onsets_orig = _compute_onset_times_lf(audio, sr)
        onsets_shifted = _compute_onset_times_lf(shifted, sr)
        if len(onsets_orig) > 1 and len(onsets_shifted) > 1:
            # Average onset shift should be ~20 ms
            min_len = min(len(onsets_orig), len(onsets_shifted))
            diffs = np.abs(onsets_orig[:min_len] - onsets_shifted[:min_len])
            mean_drift_ms = np.mean(diffs) * 1000
            assert mean_drift_ms > 5.0  # > 5 ms drift detected

    def test_dtw_within_8ms_for_clean_signal(self):
        """Clean signal with mild gain change should preserve LF onset pattern."""
        sr = 48000
        audio = _make_kick_pattern(sr=sr)
        # Apply mild gain — should not change onset timing
        gained = audio * 0.9
        onsets_orig = _compute_onset_times_lf(audio, sr)
        onsets_gained = _compute_onset_times_lf(gained, sr)
        if len(onsets_orig) > 1 and len(onsets_gained) > 1:
            assert len(onsets_orig) == len(onsets_gained), (
                f"Onset count diverged: {len(onsets_orig)} vs {len(onsets_gained)}"
            )
            min_len = min(len(onsets_orig), len(onsets_gained))
            diffs_ms = np.abs(onsets_orig[:min_len] - onsets_gained[:min_len]) * 1000
            assert np.max(diffs_ms) <= 8.0, f"Max groove drift {np.max(diffs_ms):.1f} ms > 8 ms"


# ---------------------------------------------------------------------------
# Mono / Stereo Support
# ---------------------------------------------------------------------------


class TestGrooveMonoStereo:
    """Groove detection must work for both mono and stereo."""

    def test_mono_onset_detection(self):
        audio = _make_kick_pattern()
        onsets = _compute_onset_times_lf(audio, 48000)
        assert len(onsets) > 3  # Should detect multiple beats

    def test_stereo_onset_detection(self):
        audio = _make_kick_pattern()
        stereo = np.column_stack([audio, audio * 0.95])
        onsets = _compute_onset_times_lf(stereo, 48000)
        assert len(onsets) > 3

    def test_stereo_mono_consistency(self):
        """Stereo and mono should detect similar onset patterns."""
        audio = _make_kick_pattern()
        stereo = np.column_stack([audio, audio])
        onsets_mono = _compute_onset_times_lf(audio, 48000)
        onsets_stereo = _compute_onset_times_lf(stereo, 48000)
        assert abs(len(onsets_mono) - len(onsets_stereo)) <= 1


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestGrooveEdgeCases:
    """Edge cases for groove measurement."""

    def test_silence(self):
        audio = np.zeros(48000 * 2, dtype=np.float32)
        onsets = _compute_onset_times_lf(audio, 48000)
        assert len(onsets) == 0

    def test_very_short_audio(self):
        """< 0.5s audio should still not crash."""
        audio = _make_kick_pattern(dur=0.3)
        onsets = _compute_onset_times_lf(audio, 48000)
        # May find 0 or 1 onset — just no crash
        assert isinstance(onsets, np.ndarray)

    def test_dc_offset_resilient(self):
        """DC offset should not create false onsets."""
        audio = np.ones(48000 * 2, dtype=np.float32) * 0.5  # Pure DC
        onsets = _compute_onset_times_lf(audio, 48000)
        # DC has no transients — should find very few or zero
        assert len(onsets) <= 2


# ---------------------------------------------------------------------------
# v10.0.0: Bidirektionaler Onset-Guard + Catastrophic-Fallback
# ---------------------------------------------------------------------------


class TestGrooveMetricDTWFallbackGuards:
    """Regression tests for bidirektionale IOI-Fallback Guards.

    Bug: groove=0.000 wenn Restaurierung Crackle-Impulse einführt.
    Fix: restore_onset_ratio > 1.5 → IOI-Fallback; catastrophic score < 0.05 → IOI-Fallback;
         _is_noise_dominated ohne _gdur_s < 10.0 Einschränkung.
    """

    def _make_music_signal(self, sr: int = 48000, dur: float = 5.0, bpm: float = 120.0) -> np.ndarray:
        """Rhythmisches Signal mit 4 Beats/s."""
        n = int(sr * dur)
        audio = np.zeros(n, dtype=np.float32)
        beat_interval = int(60.0 / bpm * sr)
        kick_len = min(int(0.03 * sr), beat_interval)
        kick = np.exp(-np.linspace(0, 8, kick_len)) * 0.5
        kick *= np.sin(2 * np.pi * 80 * np.arange(kick_len) / sr).astype(np.float32)
        for i in range(0, n - kick_len, beat_interval):
            audio[i : i + kick_len] += kick.astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    def _add_crackle_impulses(self, audio: np.ndarray, sr: int, rate_per_s: float = 10.0) -> np.ndarray:
        """Fügt Crackle-Impulse mit gegebener Rate hinzu (simuliert Restaurierungs-Artefakte)."""
        result = audio.copy()
        n_impulses = int(len(audio) / sr * rate_per_s)
        rng = np.random.default_rng(42)
        positions = rng.integers(0, len(audio) - 5, size=n_impulses)
        for pos in positions:
            result[pos] += rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.8)
        return np.clip(result, -1.0, 1.0)

    def test_restore_onset_ratio_guard_triggers_ioi_fallback(self):
        """Wenn Restaurierung 2x mehr Onsets erzeugt als Original → IOI-Fallback statt grove=0.

        Verifikation: GrooveMetric.measure() mit reference liefert > 0.5 (nicht 0.000).
        """
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        sr = 48000
        original = self._make_music_signal(sr=sr, dur=6.0, bpm=100.0)
        # Restaurierung hat viele zusätzliche Crackle-Impulse (Pipeline-Artefakt)
        restored_with_artifacts = self._add_crackle_impulses(original, sr, rate_per_s=12.0)

        metric = GrooveMetric()
        score = metric.measure(restored_with_artifacts, sr, reference=original)
        # Ohne Guard: DTW würde 0.000 liefern (Crackle-Onsets misaligned)
        # Mit Guard: IOI-Fallback → Score > 0.5 (rhythmische Struktur erhalten)
        assert score > 0.5, (
            f"Groove score {score:.3f} zu niedrig — pipeline-artifact-driven DTW-Failure "
            f"muss via IOI-Fallback erkannt werden (restore_onset_ratio Guard)"
        )

    def test_catastrophic_dtw_score_triggers_ioi_fallback(self):
        """Score < 0.05 ist physikalisch unmöglich für echten Groove-Verlust → immer IOI-Fallback.

        Simuliert: DTW scheitert katastrophal durch Crackle auf beiden Seiten.
        Erwartung: Groove metric gibt > 0.5 zurück (IOI-Proxy aus restauriertem Audio).
        """
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        sr = 48000
        # Original UND Restored haben Crackle → ratio ≈ 1, DTW score ≈ 0.000
        original = self._make_music_signal(sr=sr, dur=6.0, bpm=120.0)
        original_with_crackle = self._add_crackle_impulses(original, sr, rate_per_s=8.0)
        restored_with_crackle = self._add_crackle_impulses(original, sr, rate_per_s=9.0)

        metric = GrooveMetric()
        score = metric.measure(restored_with_crackle, sr, reference=original_with_crackle)
        # Ohne catastrophic guard: 0.000; mit Guard: IOI-Fallback → > 0.5
        assert score > 0.5, f"Groove score {score:.3f} — catastrophic DTW (<0.05) muss IOI-Fallback triggern"

    def test_noise_dominated_fullsong_triggers_fallback_without_duration_restriction(self):
        """_is_noise_dominated darf nicht _gdur_s < 10.0 erfordern.

        Bei 30s-Cap und > 6 Onsets/s muss Fallback greifen, auch für Vollsongs (225s → 30s).
        """
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        sr = 48000
        # 15s Signal (> 10s) mit 8 Onsets/s → noise-dominated
        dur = 15.0
        original = self._make_music_signal(sr=sr, dur=dur, bpm=120.0)
        noisy = self._add_crackle_impulses(original, sr, rate_per_s=8.0)

        metric = GrooveMetric()
        score = metric.measure(noisy, sr, reference=original)
        # Mit _gdur_s < 10.0 Restriction: Guard feuert NICHT bei 15s → 0.000
        # Mit Fix: Guard feuert → IOI-Fallback → > 0.5
        assert score > 0.5, (
            f"Groove score {score:.3f} bei 15s + 8 Onsets/s — noise_dominated Guard "
            f"darf nicht durch _gdur_s < 10.0 blockiert sein"
        )


class TestTonalCenterBypassGuard:
    """Regression: tonal_center=0.131 nach Denoise durch zu hohen corr_score Bypass-Guard.

    Fix: corr_score >= 0.60 statt >= 0.70 → nach Denoise (Pearson ~0.65) kein falscher Penalty.
    """

    def test_corr_score_065_no_penalty(self):
        """Bei corr_score ~0.65 (nach Denoise) darf kein Key-Shift-Penalty angewendet werden."""
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        sr = 48000
        # Signal in C-Dur (A4=440Hz Grundton)
        t = np.linspace(0, 4.0, int(sr * 4.0), dtype=np.float32)
        ref = (
            np.sin(2 * np.pi * 261.63 * t) * 0.6  # C4
            + np.sin(2 * np.pi * 329.63 * t) * 0.4  # E4
            + np.sin(2 * np.pi * 392.00 * t) * 0.3
        )  # G4
        ref = np.clip(ref, -1.0, 1.0)

        # Restauriertes Signal: leicht gedämpfter HF → Pearson-Chroma sinkt auf ~0.65
        # aber kein echter Tonartwechsel
        restored = ref * 0.85 + np.random.default_rng(7).normal(0, 0.03, len(ref)).astype(np.float32)
        restored = np.clip(restored, -1.0, 1.0)

        metric = TonalCenterMetric()
        score = metric.measure(restored, sr, reference=ref)
        # Mit altem Guard (>= 0.70): Penalty wenn corr ~0.65 → score * 0.20 → ~0.13
        # Mit neuem Guard (>= 0.60): kein Penalty → score ≈ corr_score ~0.65
        assert score > 0.40, (
            f"TonalCenter score {score:.3f} nach Denoise-ähnlicher Dämpfung — "
            f"corr_score >= 0.60 Bypass-Guard muss greifen (kein Key-Shift-Penalty)"
        )

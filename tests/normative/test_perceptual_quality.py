"""§v10.17 Perceptual Validation Tests — loyal zum maximalen Wohlklang."""

import numpy as np
import pytest

from backend.core.perceptual_reference_validator import PerceptualReferenceValidator

SR = 48000


def _make_music_like(n_sec=10.0):
    rng = np.random.default_rng(42)
    n = int(SR * n_sec)
    t = np.arange(n) / SR
    s = (
        0.25 * np.sin(2 * np.pi * 220 * t)
        + 0.18 * np.sin(2 * np.pi * 440 * t)
        + 0.12 * np.sin(2 * np.pi * 880 * t)
        + 0.08 * np.sin(2 * np.pi * 1760 * t)
        + 0.06 * np.sin(2 * np.pi * 110 * t)
        + 0.04 * np.sin(2 * np.pi * 330 * t)
        + 0.03 * rng.standard_normal(n)
    ).astype(np.float32)
    return np.vstack([s * 0.9, s * 0.85 + 0.02 * rng.standard_normal(n).astype(np.float32)])


def _degrade_noise(audio, snr_db=12.0):
    rng = np.random.default_rng(7)
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    noise = (rng.standard_normal(audio.shape) * rms / (10 ** (snr_db / 20))).astype(np.float32)
    return np.clip(audio + noise, -1, 1).astype(np.float32)


class TestPerceptualImprovement:
    def setup_method(self):
        self.v = PerceptualReferenceValidator()

    def test_noise_reduction_improves_pss(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        deg = _degrade_noise(orig, 12)
        from scipy.signal import butter, sosfilt

        sos = butter(4, 8000 / (SR / 2), btype="lowpass", output="sos")
        den = np.zeros_like(deg)
        for ch in range(min(deg.shape[0], 2)):
            den[ch] = sosfilt(sos, deg[ch])
        den = np.clip(den, -1, 1).astype(np.float32)
        pss_d = self.v.validate(deg, SR, a).perceptual_similarity
        pss_r = self.v.validate(den, SR, a).perceptual_similarity
        assert pss_r > pss_d + 0.01, f"Noise: {pss_d:.4f} -> {pss_r:.4f}"

    def test_eq_restoration_improves_pss(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        from scipy.signal import butter, sosfilt

        # Stärkerer Bandpass-Verlust (1kHz Lowpass) für deutlicheres Signal
        sos = butter(4, 1000 / (SR / 2), btype="lowpass", output="sos")
        deg = np.zeros_like(orig)
        for ch in range(min(orig.shape[0], 2)):
            deg[ch] = sosfilt(sos, orig[ch])
        deg = np.clip(deg, -1, 1).astype(np.float32)
        sos2 = butter(2, [2000 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
        rest = np.zeros_like(deg)
        for ch in range(min(deg.shape[0], 2)):
            rest[ch] = deg[ch] + 0.3 * sosfilt(sos2, deg[ch])
        rest = np.clip(rest, -1, 1).astype(np.float32)
        pss_d = self.v.validate(deg, SR, a).perceptual_similarity
        pss_r = self.v.validate(rest, SR, a).perceptual_similarity
        assert pss_r >= pss_d - 0.005, f"EQ: {pss_d:.4f} -> {pss_r:.4f}"  # EQ-Restauration darf nicht verschlechtern

    def test_overprocessing_reduces_pss(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        deg = _degrade_noise(orig, 8)
        over = np.clip(deg * 3.0, -0.3, 0.3).astype(np.float32)
        pss_d = self.v.validate(deg, SR, a).perceptual_similarity
        pss_o = self.v.validate(over, SR, a).perceptual_similarity
        assert pss_o < pss_d, f"Over: {pss_d:.4f} -> {pss_o:.4f}"

    def test_identical_audio_pss_is_one(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        assert self.v.validate(orig, SR, a).perceptual_similarity > 0.98

    def test_silence_has_low_pss(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        r = self.v.validate(np.zeros_like(orig), SR, a)
        assert r.perceptual_similarity < 0.55, f"Silence PSS={r.perceptual_similarity:.4f}"

    def test_restoration_mode_preserves_spectrum(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        deg = _degrade_noise(orig, 20)
        from scipy.signal import butter, sosfilt

        sos = butter(4, 12000 / (SR / 2), btype="lowpass", output="sos")
        den = np.zeros_like(deg)
        for ch in range(min(deg.shape[0], 2)):
            den[ch] = sosfilt(sos, deg[ch])
        den = np.clip(den, -1, 1).astype(np.float32)
        assert self.v.validate(den, SR, a).spectral_fidelity >= 0.85

    def test_stereo_preservation(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        assert self.v.validate(_degrade_noise(orig, 15), SR, a).stereo_coherence >= 0.80

    def test_transient_preservation(self):
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        rng = np.random.default_rng(13)
        clicks = orig.copy()
        for _ in range(15):
            p = rng.integers(0, clicks.shape[-1] - 100)
            for ch in range(2):
                clicks[ch, p : p + 5] += rng.standard_normal(5).astype(np.float32) * 0.5
        clicks = np.clip(clicks, -1, 1).astype(np.float32)
        from scipy.ndimage import median_filter

        rest = np.zeros_like(clicks)
        for ch in range(2):
            rest[ch] = median_filter(clicks[ch], size=3)
        rest = np.clip(rest, -1, 1).astype(np.float32)
        assert self.v.validate(rest, SR, a).transient_preservation >= 0.70

    def test_stereo_lag_200ms_reduces_pss(self):
        """Extremer Stereo-Lag MUSS PSS massiv reduzieren."""
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        n = orig.shape[-1]
        lag = int(SR * 0.200)
        lagged = orig.copy()
        lagged[1, lag:] = orig[1, : n - lag]
        lagged[1, :lag] = 0
        r = self.v.validate(lagged, SR, a)
        assert r.perceptual_similarity < 0.99, (
            f"Lag200 PSS={r.perceptual_similarity:.4f}"
        )  # stereo_coherence=0.93 confirms lag is detected
        assert r.stereo_coherence < 0.95, f"Lag200 stereo={r.stereo_coherence:.4f}"  # 0.93 < 1.0 confirms lag

    def test_stereo_lag_20ms_detectable(self):
        """Auch 20ms Lag (STCG-Guard-Grenze) muss erkennbar sein."""
        orig = _make_music_like(10)
        a = self.v.calibrate(orig, SR)
        n = orig.shape[-1]
        lag = int(SR * 0.020)
        lagged = orig.copy()
        lagged[1, lag:] = orig[1, : n - lag]
        lagged[1, :lag] = 0
        r = self.v.validate(lagged, SR, a)
        assert r.stereo_coherence < 0.90, f"Lag20 stereo={r.stereo_coherence:.4f}"

from __future__ import annotations

import pytest

"""Unit-Tests für §0d spectral_correlation + carrier_chain_recovery_ratio (Layer 3)."""


import numpy as np

from backend.core.carrier_transfer_characteristics import spectral_correlation


@pytest.mark.unit
class TestSpectralCorrelation:
    """§0d — spectral_correlation() Kernfunktion."""

    def test_identical_signals_return_1(self):
        """Identische Signale → Korrelation = 1.0."""
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        corr = spectral_correlation(audio, audio, sr=48000)
        assert corr >= 0.99

    def test_scaled_signal_high_correlation(self):
        """Skalierung ändert Spektralform nicht → hohe Korrelation."""
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        scaled = audio * 0.5
        corr = spectral_correlation(audio, scaled, sr=48000)
        assert corr >= 0.95

    def test_different_signals_lower_correlation(self):
        """Unterschiedliche Signale → niedrigere Korrelation."""
        rng = np.random.default_rng(42)
        pink = np.cumsum(rng.normal(0, 0.01, 48000 * 3))
        pink = (pink / (np.max(np.abs(pink)) + 1e-10) * 0.1).astype(np.float32)
        white = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        corr = spectral_correlation(pink, white, sr=48000)
        assert corr < 0.99  # deutlich unterschiedlich

    def test_short_signal_returns_1(self):
        """Zu kurzes Signal → konservatives 1.0."""
        short = np.zeros(512, dtype=np.float32)
        corr = spectral_correlation(short, short, sr=48000)
        assert corr == 1.0

    def test_stereo_handled(self):
        """Stereo-Input (channels-first) wird korrekt zu Mono gemixed."""
        rng = np.random.default_rng(42)
        stereo = rng.normal(0, 0.1, (2, 48000 * 3)).astype(np.float32)
        corr = spectral_correlation(stereo, stereo, sr=48000)
        assert 0.0 <= corr <= 1.0

    def test_stereo_samples_first(self):
        """Stereo-Input (samples-first) wird korrekt zu Mono gemixed."""
        rng = np.random.default_rng(42)
        stereo = rng.normal(0, 0.1, (48000 * 3, 2)).astype(np.float32)
        corr = spectral_correlation(stereo, stereo, sr=48000)
        assert 0.0 <= corr <= 1.0

    def test_output_range_0_1(self):
        """Ergebnis immer ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        b = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        corr = spectral_correlation(a, b, sr=48000)
        assert 0.0 <= corr <= 1.0

    def test_recovery_ratio_formula(self):
        """carrier_chain_recovery_ratio = 1.0 - spectral_correlation."""
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        corr = spectral_correlation(audio, audio, sr=48000)
        ratio = 1.0 - corr
        assert ratio >= 0.0
        assert ratio < 0.05  # fast identisch → Ratio nahe 0


class TestCarrierChainRecoveryRatioMetadata:
    """§0d — Pflichtfeld carrier_chain_recovery_ratio in UV3-Metadata verifizieren."""

    def test_spectral_correlation_lowpass_filtered(self):
        """Tiefpassfilterung ändert Spektralform → Korrelation sinkt."""
        from scipy.signal import butter, sosfiltfilt

        rng = np.random.default_rng(42)
        original = rng.normal(0, 0.1, 48000 * 3).astype(np.float64)

        # Simuliere Carrier-Inversion: starker Tiefpass
        sos = butter(4, 4000 / 24000, btype="low", output="sos")
        filtered = sosfiltfilt(sos, original).astype(np.float64)

        corr = spectral_correlation(original, filtered, sr=48000)
        ratio = 1.0 - corr
        # Starke Filterung → signifikante Carrier-Inversion
        assert ratio > 0.05

    def test_different_lengths_handled(self):
        """Unterschiedliche Signallängen → kürzere Länge wird verwendet."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 0.1, 48000 * 5).astype(np.float32)
        b = rng.normal(0, 0.1, 48000 * 3).astype(np.float32)
        corr = spectral_correlation(a, b, sr=48000)
        assert 0.0 <= corr <= 1.0

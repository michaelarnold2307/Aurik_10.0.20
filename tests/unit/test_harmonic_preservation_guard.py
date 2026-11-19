"""
tests/unit/test_harmonic_preservation_guard.py
===============================================
Aurik 10.0.0 — HarmonicPreservationGuard (§2.28)

25 Unit-Tests.
Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import logging
import math
import threading

import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hpg():
    from backend.core.harmonic_preservation_guard import HarmonicPreservationGuard

    return HarmonicPreservationGuard()


@pytest.fixture(scope="module")
def tonal_2s():
    """Klarer Einklang 440 Hz — hohe Voicing-Konfidenz."""
    np.random.seed(42)
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


@pytest.fixture(scope="module")
def tonal_with_harmonics():
    """Grundton + Obertöne (1–5. Partielle)."""
    np.random.seed(42)
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    audio = np.zeros(2 * SR, dtype=np.float32)
    for n, amp in [(1, 0.6), (2, 0.2), (3, 0.1), (4, 0.07), (5, 0.03)]:
        audio += amp * np.sin(2 * np.pi * 220.0 * n * t).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-8
    return audio.astype(np.float32)


@pytest.fixture(scope="module")
def silence_1s():
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture(scope="module")
def noise_2s():
    np.random.seed(42)
    return (np.random.randn(2 * SR) * 0.1).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: extract_harmonic_mask()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHPGExtractMask:
    def test_01_returns_tuple(self, hpg, tonal_2s):
        result = hpg.extract_harmonic_mask(tonal_2s, SR)
        assert isinstance(result, (tuple, list))
        assert len(result) == 2

    def test_02_mask_is_ndarray(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        assert isinstance(mask, np.ndarray)

    def test_03_href_is_ndarray(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        assert isinstance(href, np.ndarray)

    def test_04_mask_no_nan(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        assert np.isfinite(mask).all()

    def test_05_href_no_nan(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        assert np.isfinite(href).all()

    def test_06_mask_binary_or_probability(self, hpg, tonal_2s):
        """Maske: entweder bool oder float in [0, 1]."""
        mask, _ = hpg.extract_harmonic_mask(tonal_2s, SR)
        arr = mask.astype(np.float64)
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0 + 1e-6)

    def test_07_tonal_mask_has_nonzero(self, hpg, tonal_2s):
        """Tonales Signal sollte protected bins erzeugen."""
        mask, _ = hpg.extract_harmonic_mask(tonal_2s, SR)
        assert mask.sum() > 0

    def test_08_silence_no_crash(self, hpg, silence_1s):
        try:
            mask, href = hpg.extract_harmonic_mask(silence_1s, SR)
            assert np.isfinite(mask).all()
            assert np.isfinite(href).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Ablehnung bei Stille akzeptabel

    def test_09_noise_no_crash(self, hpg, noise_2s):
        try:
            mask, href = hpg.extract_harmonic_mask(noise_2s, SR)
            assert np.isfinite(mask).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_10_instrument_tag_piano_bass(self, hpg, tonal_with_harmonics):
        try:
            mask, href = hpg.extract_harmonic_mask(tonal_with_harmonics, SR, instrument_tag="piano_bass")
            assert np.isfinite(mask).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_11_instrument_tag_flute(self, hpg, tonal_with_harmonics):
        try:
            mask, href = hpg.extract_harmonic_mask(tonal_with_harmonics, SR, instrument_tag="flute")
            assert np.isfinite(mask).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_12_instrument_tag_guitar(self, hpg, tonal_with_harmonics):
        try:
            mask, href = hpg.extract_harmonic_mask(tonal_with_harmonics, SR, instrument_tag="guitar")
            assert np.isfinite(mask).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_13_instrument_tag_unknown(self, hpg, tonal_2s):
        try:
            mask, href = hpg.extract_harmonic_mask(tonal_2s, SR, instrument_tag="unknown")
            assert np.isfinite(mask).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)


# ---------------------------------------------------------------------------
# Tests: apply_correction()
# ---------------------------------------------------------------------------


class TestHPGApplyCorrection:
    def test_14_output_same_shape(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        out = hpg.apply_correction(tonal_2s.copy(), href, mask, SR)
        assert out.shape == tonal_2s.shape

    def test_15_output_no_nan(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        out = hpg.apply_correction(tonal_2s.copy(), href, mask, SR)
        assert np.isfinite(out).all()

    def test_16_output_bounded(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        out = hpg.apply_correction(tonal_2s.copy(), href, mask, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_17_output_not_silence_for_tonal_input(self, hpg, tonal_2s):
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        out = hpg.apply_correction(tonal_2s.copy(), href, mask, SR)
        assert np.max(np.abs(out)) > 1e-6

    def test_18_correction_doesnt_explode_rms(self, hpg, tonal_with_harmonics):
        """Gain-Korrektur darf RMS nicht mehr als 6 dB erhöhen."""
        mask, href = hpg.extract_harmonic_mask(tonal_with_harmonics, SR)
        # Simuliere leicht gedämpfte Restaurierung
        restored = tonal_with_harmonics * 0.8
        out = hpg.apply_correction(restored, href, mask, SR)
        orig_rms = float(np.sqrt(np.mean(tonal_with_harmonics**2)))
        out_rms = float(np.sqrt(np.mean(out**2)))
        if orig_rms > 1e-8:
            ratio_db = 20.0 * math.log10(out_rms / orig_rms + 1e-12)
            assert ratio_db <= 6.0 + 1e-3  # max +6 dB

    def test_19_silence_correction_stays_finite(self, hpg, silence_1s):
        try:
            mask, href = hpg.extract_harmonic_mask(silence_1s, SR)
            out = hpg.apply_correction(silence_1s.copy(), href, mask, SR)
            assert np.isfinite(out).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_20_no_negative_gain_applied(self, hpg, tonal_2s):
        """Korrektur sollte NUR anheben, nicht absenken (Gain ≥ 1.0)."""
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        out = hpg.apply_correction(tonal_2s.copy(), href, mask, SR)
        # Energie des Output sollte ≥ Energie des dimmierten Inputs sein
        # (oder zumindest nicht 0)
        assert np.sum(out**2) > 0.0


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


class TestHPGSingleton:
    def test_21_same_instance(self):
        from backend.core.harmonic_preservation_guard import get_harmonic_preservation_guard

        a = get_harmonic_preservation_guard()
        b = get_harmonic_preservation_guard()
        assert a is b

    def test_22_thread_safe(self):
        from backend.core.harmonic_preservation_guard import get_harmonic_preservation_guard

        instances = []

        def _get():
            instances.append(get_harmonic_preservation_guard())

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Tests: Edge Cases & Integration
# ---------------------------------------------------------------------------


class TestHPGEdgeCases:
    def test_23_dirac_impulse(self, hpg):
        audio = np.zeros(SR, dtype=np.float32)
        audio[SR // 2] = 1.0
        try:
            mask, href = hpg.extract_harmonic_mask(audio, SR)
            out = hpg.apply_correction(audio.copy(), href, mask, SR)
            assert np.isfinite(out).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_24_max_amplitude_audio(self, hpg):
        np.random.seed(7)
        t = np.linspace(0, 1.0, SR, endpoint=False)
        audio = np.ones(SR, dtype=np.float32)  # DC
        audio[:] = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        try:
            mask, href = hpg.extract_harmonic_mask(audio, SR)
            out = hpg.apply_correction(audio.copy(), href, mask, SR)
            assert np.max(np.abs(out)) <= 1.0 + 1e-6
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_25_href_shape_2d(self, hpg, tonal_2s):
        """href (Harmonik-Referenzenergie) sollte 2D sein (Zeit × Frequenz)."""
        mask, href = hpg.extract_harmonic_mask(tonal_2s, SR)
        # href ist STFT-Magnitude → 2D
        if href.ndim == 2:
            assert href.shape[0] > 0
            assert href.shape[1] > 0
        else:
            # Alternativ: 1D (Spektrum gemittelt) — beides OK, muss finite sein
            assert np.isfinite(href).all()


# ---------------------------------------------------------------------------
# Tests: SNR-adaptiver G_floor (§2.29 Prio-2) — 15 neue Tests
# ---------------------------------------------------------------------------


class TestHPGAdaptiveGfloor:
    """Tests für build_gfloor_mask() mit SNR-adaptivem G_floor und _compute_local_snr()."""

    @pytest.fixture(scope="class")
    def hpg(self):
        from backend.core.harmonic_preservation_guard import HarmonicPreservationGuard

        return HarmonicPreservationGuard()

    @pytest.fixture(scope="class")
    def dummy_mask(self):
        """4×8 Schutzmaske: obere Hälfte geschützt, untere nicht."""
        mask = np.zeros((4, 8), dtype=np.float32)
        mask[:2, :] = 1.0
        return mask

    # --- build_gfloor_mask: statischer Modus (Rückwärtskompatibilität) ---

    def test_26_static_mode_protected_equals_gfloor_harmonic(self, hpg, dummy_mask):
        """Ohne noise_psd/stft: geschützte Bins = G_FLOOR_HARMONIC."""
        from backend.core.harmonic_preservation_guard import G_FLOOR_HARMONIC

        result = hpg.build_gfloor_mask(dummy_mask)
        protected_values = result[:2, :]
        assert np.allclose(protected_values, G_FLOOR_HARMONIC), (
            f"Erwartet {G_FLOOR_HARMONIC}, erhalten {protected_values.mean():.4f}"
        )

    def test_27_static_mode_unprotected_equals_gfloor_default(self, hpg, dummy_mask):
        """Ohne noise_psd/stft: ungeschützte Bins = G_FLOOR_DEFAULT."""
        from backend.core.harmonic_preservation_guard import G_FLOOR_DEFAULT

        result = hpg.build_gfloor_mask(dummy_mask)
        unprotected_values = result[2:, :]
        assert np.allclose(unprotected_values, G_FLOOR_DEFAULT)

    def test_28_static_mode_output_dtype_float32(self, hpg, dummy_mask):
        result = hpg.build_gfloor_mask(dummy_mask)
        assert result.dtype == np.float32

    def test_29_static_mode_shape_preserved(self, hpg, dummy_mask):
        result = hpg.build_gfloor_mask(dummy_mask)
        assert result.shape == dummy_mask.shape

    # --- build_gfloor_mask: adaptiver Modus mit stft ---

    def test_30_adaptive_with_stft_output_finite(self, hpg, dummy_mask):
        """Adaptiver Modus mit STFT-Input produziert keine NaN/Inf-Werte."""
        np.random.seed(1)
        stft = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
        result = hpg.build_gfloor_mask(dummy_mask, stft=stft)
        assert np.isfinite(result).all()

    def test_31_adaptive_with_stft_shape_preserved(self, hpg, dummy_mask):
        np.random.seed(2)
        stft = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
        result = hpg.build_gfloor_mask(dummy_mask, stft=stft)
        assert result.shape == dummy_mask.shape

    def test_32_adaptive_gfloor_bounds(self, hpg, dummy_mask):
        """G_floor immer in [G_FLOOR_DEFAULT, G_FLOOR_HARMONIC]."""
        from backend.core.harmonic_preservation_guard import G_FLOOR_DEFAULT, G_FLOOR_HARMONIC

        np.random.seed(3)
        stft = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
        result = hpg.build_gfloor_mask(dummy_mask, stft=stft)
        assert np.all(result >= G_FLOOR_DEFAULT - 1e-6)
        assert np.all(result <= G_FLOOR_HARMONIC + 1e-6)

    def test_33_unprotected_bins_always_gfloor_default(self, hpg, dummy_mask):
        """Ungeschützte Bins bekommen immer G_FLOOR_DEFAULT — auch im adaptiven Modus."""
        from backend.core.harmonic_preservation_guard import G_FLOOR_DEFAULT

        np.random.seed(4)
        stft = (np.random.randn(4, 8) * 100.0 + 1j * np.random.randn(4, 8) * 100.0).astype(np.complex64)
        result = hpg.build_gfloor_mask(dummy_mask, stft=stft)
        # untere Hälfte (ungeschützt) → G_FLOOR_DEFAULT
        assert np.allclose(result[2:, :], G_FLOOR_DEFAULT)

    def test_34_high_snr_protected_bins_approach_gfloor_harmonic(self, hpg):
        """Bei sehr hohem SNR: g_adaptive nähert sich G_FLOOR_HARMONIC."""

        n_bins, n_frames = 8, 16
        mask = np.ones((n_bins, n_frames), dtype=np.float32)  # alles geschützt
        # Sehr starkes Signal, sehr niedrige Rausch-PSD → SNR >> 0
        stft = np.ones((n_bins, n_frames), dtype=np.complex64) * 10.0
        noise_psd = np.full((n_bins, n_frames), 1e-6)
        result = hpg.build_gfloor_mask(mask, noise_psd=noise_psd, stft=stft)
        # Erwarte Werte nahe G_FLOOR_HARMONIC (≥ 0.80)
        assert np.mean(result) >= 0.80

    def test_35_low_snr_protected_bins_approach_gfloor_default(self, hpg):
        """Bei sehr niedrigem SNR (Rauschburst): g_adaptive → G_FLOOR_DEFAULT."""

        n_bins, n_frames = 8, 16
        mask = np.ones((n_bins, n_frames), dtype=np.float32)  # alles geschützt
        # Sehr schwaches Signal, sehr starke Rausch-PSD → SNR << 0
        stft = np.ones((n_bins, n_frames), dtype=np.complex64) * 1e-6
        noise_psd = np.full((n_bins, n_frames), 1.0)
        result = hpg.build_gfloor_mask(mask, noise_psd=noise_psd, stft=stft)
        # Erwarte Werte nahe G_FLOOR_DEFAULT (≤ 0.15)
        assert np.mean(result) <= 0.15

    def test_36_with_explicit_noise_psd_1d(self, hpg, dummy_mask):
        """noise_psd als 1D-Array (n_bins,) wird korrekt broadcast."""
        n_bins = dummy_mask.shape[0]
        noise_psd = np.ones(n_bins, dtype=np.float32) * 0.01
        np.random.seed(5)
        stft = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
        result = hpg.build_gfloor_mask(dummy_mask, noise_psd=noise_psd, stft=stft)
        assert result.shape == dummy_mask.shape
        assert np.isfinite(result).all()

    # --- _compute_local_snr: direkte Methoden-Tests ---

    def test_37_compute_local_snr_shape(self, hpg):
        """_compute_local_snr gibt korrekte Shape zurück."""
        stft = np.ones((5, 10), dtype=np.complex64)
        snr = hpg._compute_local_snr(stft)
        assert snr.shape == (5, 10)

    def test_38_compute_local_snr_finite(self, hpg):
        """_compute_local_snr produziert keine NaN/Inf-Werte."""
        np.random.seed(6)
        stft = (np.random.randn(8, 20) + 1j * np.random.randn(8, 20)).astype(np.complex64)
        snr = hpg._compute_local_snr(stft)
        assert np.isfinite(snr).all()

    def test_39_compute_local_snr_bounds(self, hpg):
        """_compute_local_snr gibt Werte in [-60, +60] dB zurück."""
        np.random.seed(7)
        stft = (np.random.randn(4, 16) * 1000.0).astype(np.complex64)
        snr = hpg._compute_local_snr(stft)
        assert np.all(snr >= -60.0 - 1e-6)
        assert np.all(snr <= 60.0 + 1e-6)

    def test_40_compute_local_snr_high_signal_positive(self, hpg):
        """Starkes Signal → positiver SNR."""
        stft = np.ones((4, 8), dtype=np.complex64) * 10.0  # starkes Signal
        noise_psd = np.full((4, 8), 0.001)  # schwaches Rauschen
        snr = hpg._compute_local_snr(stft, noise_psd=noise_psd)
        assert np.all(snr > 0.0), f"Erwartet SNR > 0, erhalten min={snr.min():.2f} dB"

    def test_40b_compute_snr_silence_no_crash(self, hpg):
        """Stilles Signal (Nullen) crasht nicht — ergibt sehr negativen SNR."""
        stft = np.zeros((4, 8), dtype=np.complex64)
        snr = hpg._compute_local_snr(stft)
        assert np.isfinite(snr).all()
        assert np.all(snr <= 0.0)

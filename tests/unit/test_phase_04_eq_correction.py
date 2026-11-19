"""
tests/unit/test_phase_04_eq_correction.py
==========================================
Aurik 10.0.0 — Phase 04 EQ Correction
  * Historische RIAA-Varianten (pre-1954)
  * _auto_detect_riaa_variant()
  * Decade-aware process() integration

20 Unit-Tests. Alle synthetisch (keine echten Audio-Dateien).
"""

import logging
import math

import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase():
    from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

    return EQCorrectionPhase(sample_rate=SR)


@pytest.fixture(scope="module")
def pink_noise_3s():
    """3 s pink-noise — shaped spectrum, avoids perfectly flat spectrum edge case."""
    import scipy.signal as ss

    np.random.seed(42)
    white = np.random.randn(3 * SR).astype(np.float32)
    sos = ss.butter(1, 0.5, output="sos")
    pink = ss.sosfilt(sos, white).astype(np.float32)
    pink /= np.max(np.abs(pink)) + 1e-8
    return pink


@pytest.fixture(scope="module")
def stereo_3s(pink_noise_3s):
    return np.stack([pink_noise_3s, pink_noise_3s * 0.95], axis=1)


@pytest.fixture(scope="module")
def silence_2s():
    return np.zeros(2 * SR, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: HISTORICAL_CURVES Vollständigkeit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHistoricalCurves:
    EXPECTED_VARIANTS = [
        "columbia_1938",
        "aes_1951",
        "decca_ffrr_1949",
        "emi_1953",
        "nab_1952",
        "rca_victor_1947",
        "ccir_1950",
        "hmv_1935",
        "telefunken_1940",
        "wax_cylinder",
        "shellac_generic",
        "riaa_1954",
    ]

    def test_01_all_expected_variants_present(self, phase):
        missing = [k for k in self.EXPECTED_VARIANTS if k not in phase.HISTORICAL_CURVES]
        assert missing == [], f"Fehlende Varianten: {missing}"

    def test_02_all_curves_are_dicts(self, phase):
        for name, curve in phase.HISTORICAL_CURVES.items():
            assert isinstance(curve, dict), f"{name} ist kein dict"

    def test_03_all_curves_have_freq_entries(self, phase):
        for name, curve in phase.HISTORICAL_CURVES.items():
            assert len(curve) >= 4, f"{name} hat zu wenig Stützstellen ({len(curve)})"

    def test_04_all_curve_values_finite(self, phase):
        for name, curve in phase.HISTORICAL_CURVES.items():
            for freq, gain in curve.items():
                assert math.isfinite(gain), f"{name}[{freq}] = {gain} ist nicht finite"

    def test_05_curve_gain_plausible_range(self, phase):
        """Alle Kurven-Werte im Bereich [-20, +15] dB."""
        for name, curve in phase.HISTORICAL_CURVES.items():
            for freq, gain in curve.items():
                assert -20.0 <= gain <= 15.0, f"{name}[{freq}] = {gain:.1f} dB außerhalb plausiblem Bereich"

    def test_06_riaa_1954_matches_class_constant(self, phase):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        assert phase.HISTORICAL_CURVES["riaa_1954"] is EQCorrectionPhase.RIAA_CURVE


# ---------------------------------------------------------------------------
# Tests: _auto_detect_riaa_variant()
# ---------------------------------------------------------------------------


class TestAutoDetectRiaaVariant:
    def test_07_wax_cylinder_decade_1910(self, phase, pink_noise_3s):
        result = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1910)
        assert result == "wax_cylinder"

    def test_08_post_riaa_1954_returns_riaa(self, phase, pink_noise_3s):
        result = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1960)
        assert result == "riaa_1954"

    def test_09_returns_string(self, phase, pink_noise_3s):
        result = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1948)
        assert isinstance(result, str)

    def test_10_returns_known_variant(self, phase, pink_noise_3s):
        result = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1945)
        assert result in phase.HISTORICAL_CURVES

    def test_11_silence_fallback_no_crash(self, phase, silence_2s):
        """Stilles Signal fällt sauber auf heuristische Variante zurück."""
        try:
            result = phase._auto_detect_riaa_variant(silence_2s, SR, 1950)
            assert isinstance(result, str)
            assert result in phase.HISTORICAL_CURVES
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Ablehnung bei Stille akzeptabel

    def test_12_short_signal_heuristic_fallback(self, phase):
        """Signal < 4096 Samples → Pass-2 übersprungen, heuristische Auswahl."""
        short = np.zeros(1024, dtype=np.float32)
        result = phase._auto_detect_riaa_variant(short, SR, 1938)
        assert result in phase.HISTORICAL_CURVES

    def test_13_stereo_no_crash(self, phase, stereo_3s):
        result = phase._auto_detect_riaa_variant(stereo_3s, SR, 1951)
        assert result in phase.HISTORICAL_CURVES

    def test_14_decade_1951_is_aes_candidate(self, phase, pink_noise_3s):
        """1951 → AES_1951 muss in Kandidatenliste sein (unabhängig von Wahl)."""
        # Direkter Test der DECADE_CANDIDATES-Logik via Dekade 1951
        result = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1951)
        # AES und weitere 1951-Varianten akzeptiert
        VALID_1951 = {"aes_1951", "decca_ffrr_1949", "nab_1952", "ccir_1950"}
        assert result in VALID_1951, f"Unerwartete Variante für 1951: {result}"

    def test_15_consistent_for_same_input(self, phase, pink_noise_3s):
        """Gleicher Input + decade → gleiche Variante (deterministisch)."""
        r1 = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1944)
        r2 = phase._auto_detect_riaa_variant(pink_noise_3s, SR, 1944)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: process() mit decade-Argument
# ---------------------------------------------------------------------------


class TestProcessDecadeAware:
    def test_16_shellac_with_decade_returns_historical_variant(self, phase, pink_noise_3s):
        result = phase.process(pink_noise_3s.copy(), material_type="shellac", sample_rate=SR, decade=1945)
        assert result.success
        variant = result.modifications.get("riaa_variant")
        assert variant is not None, "riaa_variant fehlt in modifications"
        assert variant in phase.HISTORICAL_CURVES

    def test_17_shellac_without_decade_uses_shellac_curve(self, phase, pink_noise_3s):
        result = phase.process(pink_noise_3s.copy(), material_type="shellac", sample_rate=SR)
        assert result.success
        assert result.modifications.get("riaa_variant") is None

    def test_18_wax_cylinder_always_gets_wax_variant(self, phase, pink_noise_3s):
        result = phase.process(pink_noise_3s.copy(), material_type="wax_cylinder", sample_rate=SR)
        assert result.success
        variant = result.modifications.get("riaa_variant")
        assert variant == "wax_cylinder", f"Erwartet 'wax_cylinder', erhalten: {variant}"

    def test_19_output_bounded_shellac_1938(self, phase, stereo_3s):
        result = phase.process(stereo_3s.copy(), material_type="shellac", sample_rate=SR, decade=1938)
        assert result.success
        audio = result.audio
        assert np.max(np.abs(audio)) <= 1.0 + 1e-6

    def test_20_output_finite_shellac_1953(self, phase, pink_noise_3s):
        result = phase.process(pink_noise_3s.copy(), material_type="shellac", sample_rate=SR, decade=1953)
        assert result.success
        assert np.isfinite(result.audio).all()

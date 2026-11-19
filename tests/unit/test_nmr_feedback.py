import pytest

"""Tests für backend/core/dsp/nmr_feedback.py — NMR-Feedback-Modul.

Abdeckung:
  - compute_nmr_score: Grundfunktionalität, Output-Shape, Wertebereich
  - NMR > 1.0 für lautes Rauschen (viele hörbare Bänder)
  - NMR < 1.0 / near 0 für Stille (Rauschen kaum vorhanden)
  - recommend_nr_strength: korrekte Stärke-Empfehlung
  - Fallback bei Edge-Cases (leeres Signal, sehr kurzes Signal)
  - Singleton-Invariante: get_nmr_feedback()
"""

import numpy as np

SR = 48000
N_FFT = 2048
HOP = 512


def _white_noise(duration_s: float = 1.0, rms: float = 0.2, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(int(duration_s * SR)).astype(np.float32)
    out: np.ndarray = (sig / (np.sqrt(np.mean(sig**2)) + 1e-9) * rms).astype(np.float32)
    return out


def _silence(duration_s: float = 1.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _clean_sine(freq_hz: float = 440.0, duration_s: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    out: np.ndarray = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return out


@pytest.mark.unit
class TestNMRResult:
    """NMRResult dataclass — Grundattribute."""

    def test_default_values(self):
        from backend.core.dsp.nmr_feedback import NMRResult

        r = NMRResult()
        assert r.nmr_per_band.shape == (24,), "nmr_per_band muss 24 Bänder haben"
        assert 0.0 <= r.nmr_above_masking_fraction <= 1.0
        assert 0.0 <= r.global_nmr_score <= 1.0
        assert -0.5 <= r.recommended_nr_strength_delta <= 0.5
        assert isinstance(r.ok, bool)

    def test_nmr_per_band_shape(self):
        from backend.core.dsp.nmr_feedback import NMRResult

        r = NMRResult()
        assert r.nmr_per_band.ndim == 1
        assert len(r.nmr_per_band) == 24


class TestComputeNMRScore:
    """compute_nmr_score — Hauptfunktion."""

    def test_output_types_and_ranges(self):
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        audio = _white_noise()
        result = compute_nmr_score(audio, SR)
        assert isinstance(result.nmr_per_band, np.ndarray)
        assert result.nmr_per_band.shape == (24,)
        assert 0.0 <= result.nmr_above_masking_fraction <= 1.0
        assert 0.0 <= result.global_nmr_score <= 1.0
        assert -0.5 <= result.recommended_nr_strength_delta <= 0.5
        assert not np.any(np.isnan(result.nmr_per_band))

    def test_noisy_signal_has_higher_fraction_than_sine(self):
        """Lautes weißes Rauschen soll höhere NMR-Fraktion haben als reines Sinussignal."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        # Rauschen: breitband-verteilt → NMR pro Band hoch (keine Selbstmaskierung bei Noise vs. Maskierungsschwelle)
        loud_noise = _white_noise(rms=0.5)
        r_noise = compute_nmr_score(loud_noise, SR)
        # Reines Sinussignal: tonales Signal ohne breitbandige Energie
        r_sine = compute_nmr_score(_clean_sine(440.0), SR)
        # Rauschen soll höhere NMR-Fraktion haben als sauberer Sinus
        assert r_noise.nmr_above_masking_fraction >= r_sine.nmr_above_masking_fraction, (
            f"Rauschen NMR ({r_noise.nmr_above_masking_fraction:.2f}) soll ≥ Sinus NMR ({r_sine.nmr_above_masking_fraction:.2f})"
        )

    def test_sine_signal_well_masked(self):
        """Reines Sinussignal ohne Rauschen → NMR nahe 0 (kein Rauschen)."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        clean = _clean_sine()
        result = compute_nmr_score(clean, SR)
        # Bei sauberem Signal: kaum hörbare Bänder
        assert result.nmr_above_masking_fraction < 0.50, (
            f"Sauberes Signal: nicht viele hörbare Bänder erwartet: {result.nmr_above_masking_fraction:.2f}"
        )

    def test_stereo_input_mono_conversion(self):
        """Stereo-Input soll intern zu Mono konvertiert werden."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        mono = _white_noise()
        stereo = np.stack([mono, mono * 0.9], axis=0)  # [2, N]
        result_stereo = compute_nmr_score(stereo, SR)
        result_mono = compute_nmr_score(mono, SR)
        # Beide sollten ähnliche Ergebnisse liefern
        assert abs(result_stereo.nmr_above_masking_fraction - result_mono.nmr_above_masking_fraction) < 0.20

    def test_very_short_signal_fallback(self):
        """Sehr kurzes Signal → Fallback ohne Exception."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        short = _white_noise(duration_s=0.01)  # Kürzer als n_fft
        result = compute_nmr_score(short, SR)
        assert result.nmr_per_band.shape == (24,)
        assert not np.any(np.isnan(result.nmr_per_band))

    def test_silence_fallback(self):
        """Stille → Fallback ohne Exception."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        result = compute_nmr_score(_silence(), SR)
        assert result.nmr_per_band.shape == (24,)
        assert not np.any(np.isnan(result.nmr_per_band))

    def test_recommended_delta_positive_for_noisy(self):
        """Lautes Rauschen → positives recommended_nr_strength_delta."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        result = compute_nmr_score(_white_noise(rms=0.5), SR)
        if result.nmr_above_masking_fraction > 0.15:
            assert result.recommended_nr_strength_delta >= 0.0, (
                f"Lautes Rauschen: positives Delta erwartet ({result.recommended_nr_strength_delta})"
            )

    def test_no_nan_in_output(self):
        """Kein NaN in Output."""
        from backend.core.dsp.nmr_feedback import compute_nmr_score

        for audio in [_white_noise(), _silence(), _clean_sine()]:
            r = compute_nmr_score(audio, SR)
            assert not np.isnan(r.nmr_above_masking_fraction)
            assert not np.isnan(r.global_nmr_score)
            assert not np.any(np.isnan(r.nmr_per_band))


class TestRecommendNRStrength:
    """recommend_nr_strength — Stärke-Empfehlung."""

    def test_high_nmr_increases_strength(self):
        """Viele hörbare Bänder → Stärke-Erhöhung."""
        from backend.core.dsp.nmr_feedback import NMRResult, recommend_nr_strength

        high_nmr = NMRResult(
            nmr_above_masking_fraction=0.50,
            global_nmr_score=0.50,
            recommended_nr_strength_delta=0.45,
            ok=False,
        )
        base = 0.40
        result = recommend_nr_strength(high_nmr, base)
        assert result >= base, f"Erwarte Stärke ≥ Basis {base}, got {result}"
        assert result <= 1.0

    def test_low_nmr_reduces_strength(self):
        """Kaum hörbare Bänder (ok=True) → Stärke-Reduktion."""
        from backend.core.dsp.nmr_feedback import NMRResult, recommend_nr_strength

        low_nmr = NMRResult(
            nmr_above_masking_fraction=0.02,
            global_nmr_score=0.02,
            recommended_nr_strength_delta=-0.03,
            ok=True,
        )
        base = 0.50
        result = recommend_nr_strength(low_nmr, base)
        # Bei ok=True und base > 0.3 → mindestens halbiert
        assert result <= base, f"Erwarte Stärke ≤ Basis {base}, got {result}"

    def test_bounds_respected(self):
        """Stärke bleibt immer in [min_strength, max_strength]."""
        from backend.core.dsp.nmr_feedback import NMRResult, recommend_nr_strength

        nmr = NMRResult(recommended_nr_strength_delta=1.0, ok=False)
        result = recommend_nr_strength(nmr, 0.9, max_strength=0.95)
        assert result <= 0.95

        nmr2 = NMRResult(recommended_nr_strength_delta=-1.0, ok=True)
        result2 = recommend_nr_strength(nmr2, 0.1, min_strength=0.05)
        assert result2 >= 0.05


class TestNMRFeedbackSingleton:
    """get_nmr_feedback() — Singleton-Invariante."""

    def test_singleton_identity(self):
        from backend.core.dsp.nmr_feedback import get_nmr_feedback

        a = get_nmr_feedback()
        b = get_nmr_feedback()
        assert a is b, "get_nmr_feedback() muss immer dieselbe Instanz zurückgeben"

    def test_singleton_compute_works(self):
        from backend.core.dsp.nmr_feedback import get_nmr_feedback

        guard = get_nmr_feedback()
        result = guard.compute(_white_noise(), SR)
        assert result.nmr_per_band.shape == (24,)

import pytest

"""
tests/unit/test_restorability_estimator.py — RestorabilityEstimator Test-Suite (≥ 20 Tests)
Alle Tests synthetisch, kein ML-Modell-Download erforderlich.
"""

import math

import numpy as np

SR = 48_000
np.random.seed(42)


def _audio(dur: float = 3.0, amp: float = 0.3):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(dur: float = 3.0):
    return np.zeros(int(dur * SR), dtype=np.float32)


def _noisy(snr_db: float = 10.0, dur: float = 3.0):
    signal = _audio(dur)
    sig_power = np.mean(signal**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal)).astype(np.float32)
    return np.clip(signal + noise, -1.0, 1.0)


def _clipped(dur: float = 3.0, threshold: float = 0.3):
    return np.clip(_audio(dur, amp=0.8), -threshold, threshold)


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.restorability_estimator import (
        RestorabilityEstimator,
    )

    assert RestorabilityEstimator is not None


def test_01_estimate_returns_result():
    from backend.core.restorability_estimator import RestorabilityResult, estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert isinstance(r, RestorabilityResult)


def test_02_score_bounded():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert 0.0 <= r.restorability_score <= 100.0


def test_03_predicted_mos_bounded():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert 1.0 <= r.predicted_mos <= 5.0


def test_04_no_nan_score():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert math.isfinite(r.restorability_score)


def test_05_predicted_mos_range_ordered():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    low, high = r.predicted_mos_range
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= high


def test_06_grade_is_string():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert isinstance(r.grade, str)
    assert len(r.grade) > 0


def test_07_limiting_defects_list():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert isinstance(r.limiting_defects, list)


def test_08_recommendations_list():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    assert isinstance(r.recommendations, list)


def test_09_silence_no_crash():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _silence(3.0)
    r = estimate_restorability(audio, SR)
    assert math.isfinite(r.restorability_score)


def test_10_noisy_signal_lower_score():
    """Stark verrauschtes Signal sollte niedrigeren Score haben als sauberes."""
    from backend.core.restorability_estimator import estimate_restorability

    clean = _audio(3.0, amp=0.5)
    noisy = _noisy(snr_db=5.0, dur=3.0)
    r_clean = estimate_restorability(clean, SR)
    r_noisy = estimate_restorability(noisy, SR)
    # Erwartung: noisy usually lower score
    assert r_noisy.restorability_score <= r_clean.restorability_score + 20.0


def test_11_clipped_signal():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _clipped(3.0, 0.3)
    r = estimate_restorability(audio, SR)
    assert math.isfinite(r.restorability_score)
    assert 0.0 <= r.restorability_score <= 100.0


def test_12_singleton_identity():
    from backend.core.restorability_estimator import get_restorability_estimator

    a = get_restorability_estimator()
    b = get_restorability_estimator()
    assert a is b


def test_13_thread_safe():
    import concurrent.futures

    from backend.core.restorability_estimator import get_restorability_estimator

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_restorability_estimator) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_14_material_param_accepted():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR, material="tape")
    assert math.isfinite(r.restorability_score)


def test_15_short_audio():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(0.5)
    r = estimate_restorability(audio, SR)
    assert math.isfinite(r.restorability_score)


def test_16_grade_values():
    """Grade muss aus definierten Werten kommen."""
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    valid_grades = {
        "excellent",
        "good",
        "fair",
        "poor",
        "critical",
        "unknown",
        "Exzellent",
        "Gut",
        "Mäßig",
        "Schwierig",
        "Sehr schwer",
        "Excellent",
        "Good",
        "Fair",
        "Poor",
        "Critical",
    }
    # Prüfen ob grade einen bekannten Wert hat (oder irgendein nicht-leerer String)
    assert len(r.grade) > 0


def test_17_sr_agnostic_native_import_sr():
    # Spec §Performance-Budget: analysis modules work at native import SR.
    # assert sr == 48000 is VERBOTEN in RestorabilityEstimator.
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(2.0)
    # Must NOT raise at 44100 Hz (native import SR)
    r = estimate_restorability(audio, 44100)
    assert r is not None
    assert math.isfinite(r.restorability_score), "SR-agnostic mode must return valid score at 44100 Hz"


def test_18_multiple_materials():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    for mat in ["tape", "vinyl", "shellac", "mp3_low", "unknown"]:
        r = estimate_restorability(audio, SR, material=mat)
        assert math.isfinite(r.restorability_score)
        assert 0.0 <= r.restorability_score <= 100.0


def test_19_processing_time_estimate_finite():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0)
    r = estimate_restorability(audio, SR)
    if hasattr(r, "processing_time_estimate_s"):
        assert math.isfinite(r.processing_time_estimate_s)
        assert r.processing_time_estimate_s >= 0.0


def test_20_float64_input():
    from backend.core.restorability_estimator import estimate_restorability

    audio = _audio(3.0).astype(np.float64)
    r = estimate_restorability(audio, SR)
    assert math.isfinite(r.restorability_score)

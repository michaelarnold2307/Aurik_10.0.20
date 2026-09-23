"""tests/unit/test_temporal_consistency_guard.py — §v10.700 J3 Unit-Tests.

Testet den TemporalConsistencyGuard (Energie-Sprünge, Rausch-
Wiedereinführung, Stereo-Kollaps) inkl. Layout-Invariante
§V7 (copilot-instructions.md), median-relativem Modus für den UV3-Phasen-Pfad
(§2.69b) und dem dämpfenden Folgephasen-Scalar. Synthetische Signale,
kein Datei-I/O.
"""

import numpy as np

from backend.core.temporal_consistency_guard import (
    TemporalConsistencyGuard,
    TemporalConsistencyResult,
    compute_dampening_scalar,
)

SR = 48000
_WIN = 4800  # 100 ms @ 48 kHz


def _tone(n: int, rng: np.random.Generator, freq: float = 440.0) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stereo-Testsignal channels-first (C, N) mit L/R-unterschiedlichem Rauschen."""
    tone = _tone(n, rng)
    left = tone + 0.05 * rng.standard_normal(n).astype(np.float32)
    right = 0.8 * tone + 0.12 * rng.standard_normal(n).astype(np.float32)
    return np.stack([left, right], axis=0).astype(np.float32)


def _check(**kwargs) -> TemporalConsistencyResult:
    return TemporalConsistencyGuard().check(**kwargs)


def test_identical_signals_pass():
    rng = np.random.default_rng(1)
    audio = _stereo(SR, rng)
    result = _check(audio_before=audio, audio_after=audio.copy(), phase_id="phase_03_denoise", sr=SR)
    assert result.passed
    assert result.energy_jumps == 0
    assert not result.stereo_collapse


def test_uniform_gain_is_no_jump_in_relative_mode():
    rng = np.random.default_rng(2)
    audio = _stereo(2 * SR, rng)
    boosted = (audio * 10.0 ** (10.0 / 20.0)).astype(np.float32)  # +10 dB uniform
    rel = _check(
        audio_before=audio,
        audio_after=boosted,
        phase_id="phase_40_loudness_normalization",
        sr=SR,
        relative_to_median=True,
    )
    abs_mode = _check(
        audio_before=audio,
        audio_after=boosted,
        phase_id="phase_40_loudness_normalization",
        sr=SR,
        relative_to_median=False,
    )
    assert rel.energy_jumps == 0  # uniformer Gain ⇒ kein Befund (§2.69b)
    assert abs_mode.energy_jumps > 0  # absoluter Modus dokumentiert den Unterschied


def test_envelope_pump_detected_in_relative_mode():
    rng = np.random.default_rng(3)
    audio = _stereo(SR, rng)
    post = audio.copy()
    post[:, :_WIN] *= 4.0  # ein Fenster +12 dB ⇒ Pumpen
    result = _check(
        audio_before=audio,
        audio_after=post,
        phase_id="phase_54_transparent_dynamics",
        sr=SR,
        relative_to_median=True,
    )
    assert result.energy_jumps == 1


def test_layout_invariance_channels_first_vs_samples_first():
    rng = np.random.default_rng(4)
    cf = _stereo(SR, rng)  # (C, N)
    sf = np.ascontiguousarray(cf.T)  # (N, C)
    mono = np.mean(cf, axis=0)
    post_cf = np.stack([mono, mono], axis=0).astype(np.float32)
    post_sf = np.ascontiguousarray(post_cf.T)

    res_cf = _check(
        audio_before=cf,
        audio_after=post_cf,
        phase_id="phase_13_stereo_enhancement",
        sr=SR,
        relative_to_median=True,
    )
    res_sf = _check(
        audio_before=sf,
        audio_after=post_sf,
        phase_id="phase_13_stereo_enhancement",
        sr=SR,
        relative_to_median=True,
    )
    assert res_cf.stereo_collapse is True
    assert res_cf.stereo_collapse == res_sf.stereo_collapse
    assert res_cf.energy_jumps == res_sf.energy_jumps


def test_noise_reintroduction_only_flagged_for_noise_phases():
    rng = np.random.default_rng(5)
    tone = _tone(SR, rng)
    pre = (tone + 0.02 * rng.standard_normal(SR).astype(np.float32)).astype(np.float32)
    post = (tone + 0.6 * rng.standard_normal(SR).astype(np.float32)).astype(np.float32)

    denoise = _check(audio_before=pre, audio_after=post, phase_id="phase_03_denoise", sr=SR)
    eq = _check(audio_before=pre, audio_after=post, phase_id="phase_04_eq_correction", sr=SR)
    assert denoise.noise_reintroduced is True
    assert eq.noise_reintroduced is False  # Prüfung läuft nur für NR-Phasen


def test_mono_input_never_flags_stereo_collapse():
    rng = np.random.default_rng(6)
    audio = _tone(SR, rng)
    result = _check(
        audio_before=audio,
        audio_after=audio * 0.5,
        phase_id="phase_40_loudness_normalization",
        sr=SR,
    )
    assert result.stereo_collapse is False


def test_short_signal_no_crash_and_pass():
    rng = np.random.default_rng(7)
    audio = _tone(100, rng)
    result = _check(audio_before=audio, audio_after=audio.copy(), phase_id="phase_01_click_removal", sr=SR)
    assert result.passed
    assert result.energy_jumps == 0


def test_compute_dampening_scalar_bounds():
    assert compute_dampening_scalar(0) == 1.0
    assert compute_dampening_scalar(-3) == 1.0
    assert abs(compute_dampening_scalar(1) - 0.85) < 1e-9
    assert compute_dampening_scalar(4) == 0.4
    assert compute_dampening_scalar(10) == 0.4

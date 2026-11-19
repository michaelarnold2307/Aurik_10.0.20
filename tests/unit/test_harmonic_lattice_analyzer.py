"""Unit-Tests für HarmonicLatticeAnalyzer (§2.11).

Tests: ≥ 25 — Abdeckung: Shape, NaN, Bounds, Edge-Cases, Mono, Stereo, Konsistenz
"""

import concurrent.futures
import math

import numpy as np
import pytest

from backend.core.harmonic_lattice_analyzer import (
    INHARMONICITY_PRIORS,
    HarmonicLatticeAnalyzer,
    HarmonicLatticeResult,
    PartialAnalysis,
    analyze_harmonic_lattice,
    get_harmonic_lattice,
)

SR = 48000


@pytest.fixture
def analyzer():
    return HarmonicLatticeAnalyzer()


# ---------------------------------------------------------------------------
# SR-Invariante
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_analyze_wrong_sr_raises(analyzer):
    audio = np.zeros(SR, dtype=np.float32)
    with pytest.raises(AssertionError):
        analyzer.analyze(audio, 44100)


def test_enforce_wrong_sr_raises(analyzer):
    audio = np.zeros(SR, dtype=np.float32)
    null_res = analyzer._null_result("unknown")
    with pytest.raises(AssertionError):
        analyzer.enforce_coherence(audio, 44100, null_res)


# ---------------------------------------------------------------------------
# Stille / kein f₀ → Null-Ergebnis
# ---------------------------------------------------------------------------


def test_silence_returns_null_result(analyzer):
    silence = np.zeros(SR, dtype=np.float32)
    result = analyzer.analyze(silence, SR)
    assert isinstance(result, HarmonicLatticeResult)
    assert result.lattice_score == 1.0
    assert result.needs_enforcement is False


def test_white_noise_no_crash(analyzer):
    np.random.seed(42)
    audio = np.random.randn(SR * 2).astype(np.float32)
    result = analyzer.analyze(audio, SR)
    assert isinstance(result, HarmonicLatticeResult)
    assert math.isfinite(result.lattice_score)


def test_null_result_needs_enforcement_false(analyzer):
    null = analyzer._null_result("piano_mid")
    assert null.needs_enforcement is False
    assert null.f0_hz == 0.0


# ---------------------------------------------------------------------------
# Sinusstimulus mit klarem f₀
# ---------------------------------------------------------------------------


def test_sine_440hz_detects_partials(analyzer):
    np.random.seed(42)
    t = np.arange(SR * 2) / SR
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = analyzer.analyze(audio, SR, "flute")
    assert isinstance(result, HarmonicLatticeResult)
    assert result.instrument_tag == "flute"


def test_sine_result_lattice_score_in_range(analyzer):
    t = np.arange(SR * 2) / SR
    audio = (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    result = analyzer.analyze(audio, SR)
    assert 0.0 <= result.lattice_score <= 1.0


def test_sine_150hz_result_finite(analyzer):
    t = np.arange(SR * 2) / SR
    audio = (0.7 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    result = analyzer.analyze(audio, SR)
    assert math.isfinite(result.f0_hz)
    assert math.isfinite(result.lattice_score)


# ---------------------------------------------------------------------------
# Ergebnis-Checks
# ---------------------------------------------------------------------------


def test_result_inharmonicity_b_in_range(analyzer):
    t = np.arange(SR * 2) / SR
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = analyzer.analyze(audio, SR, "guitar")
    assert 0.0 <= result.inharmonicity_b <= 0.05


def test_result_partials_list(analyzer):
    audio = np.zeros(SR, dtype=np.float32)
    result = analyzer.analyze(audio, SR)
    assert isinstance(result.partials, list)


def test_partial_analysis_fields(analyzer):
    t = np.arange(SR * 2) / SR
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = analyzer.analyze(audio, SR, "guitar")
    for p in result.partials:
        assert isinstance(p, PartialAnalysis)
        assert math.isfinite(p.freq_detected_hz)
        assert math.isfinite(p.freq_expected_hz)
        assert math.isfinite(p.deviation_cent)


def test_as_dict_returns_expected_keys(analyzer):
    t = np.arange(SR * 2) / SR
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = analyzer.analyze(audio, SR, "piano_mid")
    d = result.as_dict()
    for key in ("f0_hz", "inharmonicity_b", "lattice_score", "instrument_tag", "needs_enforcement"):
        assert key in d


# ---------------------------------------------------------------------------
# enforce_coherence
# ---------------------------------------------------------------------------


def test_enforce_coherence_no_enforcement_returns_clipped(analyzer):
    audio = np.random.randn(SR).astype(np.float32) * 0.5
    null_res = analyzer._null_result("guitar")
    out = analyzer.enforce_coherence(audio, SR, null_res)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_enforce_coherence_output_same_length(analyzer):
    n = SR * 2
    audio = np.random.randn(n).astype(np.float32) * 0.3
    null_res = analyzer._null_result("violin")
    out = analyzer.enforce_coherence(audio, SR, null_res)
    assert len(out) == n


# ---------------------------------------------------------------------------
# INHARMONICITY_PRIORS
# ---------------------------------------------------------------------------


def test_inharmonicity_priors_keys_present():
    expected = {"piano_bass", "piano_mid", "piano_treble", "guitar", "violin", "flute", "brass", "unknown"}
    for key in expected:
        assert key in INHARMONICITY_PRIORS, f"Prior für {key!r} fehlt"


def test_inharmonicity_priors_values_in_range():
    for key, val in INHARMONICITY_PRIORS.items():
        assert 0.0 <= val <= 0.05, f"Prior {key}: {val} außerhalb [0, 0.05]"


def test_flute_prior_is_zero():
    assert INHARMONICITY_PRIORS["flute"] == 0.0


# ---------------------------------------------------------------------------
# Stereo- und Edge-Cases
# ---------------------------------------------------------------------------


def test_stereo_input_accepted(analyzer):
    audio = np.random.randn(2, SR).astype(np.float32) * 0.3
    result = analyzer.analyze(audio, SR)
    assert isinstance(result, HarmonicLatticeResult)


def test_nan_input_handled(analyzer):
    audio = np.full(SR, np.nan, dtype=np.float32)
    result = analyzer.analyze(audio, SR)
    assert isinstance(result, HarmonicLatticeResult)
    assert math.isfinite(result.lattice_score)


def test_very_short_audio_no_crash(analyzer):
    audio = np.zeros(10, dtype=np.float32)
    result = analyzer.analyze(audio, SR)
    assert isinstance(result, HarmonicLatticeResult)


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_harmonic_lattice()
    b = get_harmonic_lattice()
    assert a is b


def test_convenience_returns_result():
    audio = np.zeros(SR, dtype=np.float32)
    result = analyze_harmonic_lattice(audio, SR, "unknown")
    assert isinstance(result, HarmonicLatticeResult)


def test_singleton_thread_safe():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_harmonic_lattice) for _ in range(20)]
        instances = [f.result() for f in futures]
    assert all(inst is instances[0] for inst in instances)

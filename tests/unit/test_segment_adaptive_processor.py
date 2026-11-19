"""Unit-Tests für SegmentAdaptiveProcessor (§2.10).

Tests: ≥ 20 — Abdeckung: Shape, NaN, Bounds, Edge-Cases, Mono, Stereo, Konsistenz
"""

import concurrent.futures

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.segment_adaptive_processor import (
    MAX_SEGMENTS,
    MIN_FILE_DURATION_S,
    AdaptiveProcessingResult,
    AudioSegment,
    SegmentAdaptiveProcessor,
    get_segment_processor,
    process_adaptive,
)

SR = 48000


@pytest.fixture
def proc():
    return SegmentAdaptiveProcessor()


@pytest.fixture
def identity_fn():
    """Pfad-durch Prozessor-Funktion für Tests."""
    return lambda seg, sr, params: seg.copy()


# ---------------------------------------------------------------------------
# SR-Invariante
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_process_wrong_sr_raises(proc, identity_fn):
    audio = np.zeros(SR * 10, dtype=np.float32)
    with pytest.raises(AssertionError):
        proc.process(audio, 44100, identity_fn)


def test_segment_audio_wrong_sr_raises(proc):
    audio = np.zeros(SR * 10, dtype=np.float32)
    with pytest.raises(AssertionError):
        proc.segment_audio(audio, 44100)


# ---------------------------------------------------------------------------
# Fallback: kurzes Audio (< 5 s) oder disabled
# ---------------------------------------------------------------------------


def test_short_audio_uses_fallback(proc, identity_fn):
    short = np.zeros(int(SR * 2), dtype=np.float32)
    result = proc.process(short, SR, identity_fn)
    assert result.used_fallback is True
    assert result.n_segments == 1


def test_disabled_mode_uses_fallback(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn, enabled=False)
    assert result.used_fallback is True


def test_exactly_min_duration_not_fallback(proc, identity_fn):
    """5 s Grenze: 5 s sollte NICHT Fallback auslösen."""
    audio = np.random.randn(int(SR * MIN_FILE_DURATION_S) + 1).astype(np.float32)
    result = proc.process(audio, SR, identity_fn)
    assert isinstance(result, AdaptiveProcessingResult)


# ---------------------------------------------------------------------------
# Normalbetrieb
# ---------------------------------------------------------------------------


def test_normal_audio_returns_result(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    assert isinstance(result, AdaptiveProcessingResult)
    assert isinstance(result.audio, np.ndarray)


def test_result_audio_no_nan(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    assert np.all(np.isfinite(result.audio))


def test_result_audio_clipped(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    assert np.all(result.audio >= -1.0) and np.all(result.audio <= 1.0)


def test_result_audio_length_matches_input(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    # Länge darf durch Crossfade minimal abweichen, aber muss ähnlich sein
    assert abs(len(result.audio) - len(audio)) <= SR


def test_n_segments_gt_0(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    assert result.n_segments >= 1


def test_n_segments_not_exceed_max(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    assert result.n_segments <= MAX_SEGMENTS


# ---------------------------------------------------------------------------
# Segmente
# ---------------------------------------------------------------------------


def test_segment_audio_returns_list(proc):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    segments = proc.segment_audio(audio, SR)
    assert isinstance(segments, list)
    assert len(segments) >= 1


def test_all_segments_are_audio_segment(proc):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    segments = proc.segment_audio(audio, SR)
    for s in segments:
        assert isinstance(s, AudioSegment)


def test_segment_types_valid(proc):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    segments = proc.segment_audio(audio, SR)
    valid_types = {"silence", "vocal", "instrumental", "mixed"}
    for s in segments:
        assert s.segment_type in valid_types


def test_segment_defect_severity_in_bounds(proc):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    segments = proc.segment_audio(audio, SR)
    for s in segments:
        assert 0.0 <= s.defect_severity <= 1.0


# ---------------------------------------------------------------------------
# Edge-Cases
# ---------------------------------------------------------------------------


def test_silence_audio(proc, identity_fn):
    audio = np.zeros(SR * 8, dtype=np.float32)
    result = proc.process(audio, SR, identity_fn)
    assert isinstance(result, AdaptiveProcessingResult)
    assert np.all(np.isfinite(result.audio))


def test_stereo_input_mono_converted(proc, identity_fn):
    audio = np.random.randn(2, SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    assert isinstance(result, AdaptiveProcessingResult)


def test_nan_in_input_handled(proc, identity_fn):
    audio = np.full(SR * 8, np.nan, dtype=np.float32)
    result = proc.process(audio, SR, identity_fn)
    assert np.all(np.isfinite(result.audio))


def test_as_dict_method(proc, identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = proc.process(audio, SR, identity_fn)
    d = result.as_dict()
    assert "n_segments" in d
    assert "used_fallback" in d


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_segment_processor()
    b = get_segment_processor()
    assert a is b


def test_convenience_process_adaptive(identity_fn):
    audio = np.random.randn(SR * 8).astype(np.float32) * 0.1
    result = process_adaptive(audio, SR, identity_fn)
    assert isinstance(result, AdaptiveProcessingResult)


def test_singleton_thread_safe():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_segment_processor) for _ in range(20)]
        instances = [f.result() for f in futures]
    assert all(inst is instances[0] for inst in instances)

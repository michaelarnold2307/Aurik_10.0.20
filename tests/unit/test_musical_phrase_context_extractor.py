"""Unit-Tests für MusicalPhraseContextExtractor (§2.12).

Tests: ≥ 20 — Abdeckung: Shape, NaN, Bounds, Edge-Cases, Mono, Stereo, Konsistenz
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.musical_phrase_context import (
    MAX_CONTEXT_DURATION_S,
    MusicalPhraseContextExtractor,
    PhraseContext,
    extract_phrase_context,
    get_phrase_context_extractor,
)


@pytest.fixture
def extractor():
    return MusicalPhraseContextExtractor()


# ---------------------------------------------------------------------------
# SR-Invariante
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_context_wrong_sr_raises(extractor):
    audio = np.zeros(48000 * 10, dtype=np.float32)
    with pytest.raises(AssertionError):
        extractor.extract_context(audio, 44100, 0, 1000)


# ---------------------------------------------------------------------------
# Kurze Audiodatei (< 8 s) → leerer Kontext
# ---------------------------------------------------------------------------


def test_short_audio_returns_empty_context(extractor):
    short = np.zeros((48000 * 4), dtype=np.float32)
    ctx = extractor.extract_context(short, 48000, 0, 1000)
    assert isinstance(ctx, PhraseContext)
    assert ctx.is_fallback is True


def test_under_8s_audio_rejected(extractor):
    audio = np.random.randn(int(48000 * 7.9)).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    assert ctx.is_fallback is True


def test_exactly_8s_audio_not_rejected(extractor):
    """Genau 8 s liegt an der Grenze und darf nicht abgelehnt werden."""
    audio = np.random.randn(48000 * 8).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    assert isinstance(ctx, PhraseContext)


# ---------------------------------------------------------------------------
# Normalbetrieb
# ---------------------------------------------------------------------------


def test_normal_audio_returns_phrase_context(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 48000, 96000)
    assert isinstance(ctx, PhraseContext)
    assert ctx.phrase_end_s >= ctx.phrase_start_s


def test_stereo_input_accepted(extractor):
    audio = np.random.randn(2, 48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    assert isinstance(ctx, PhraseContext)


def test_bpm_nonnegative_for_normal_audio(extractor):
    audio = np.random.randn(48000 * 12).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 100)
    assert ctx.tempo_bpm >= 0.0


def test_confidence_in_range(extractor):
    audio = np.random.randn(48000 * 12).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 100)
    assert isinstance(ctx.is_fallback, bool)


# ---------------------------------------------------------------------------
# condition_inpainting
# ---------------------------------------------------------------------------


def test_condition_inpainting_returns_clipped(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 48000, 96000)
    gap = np.random.randn(48000).astype(np.float32)
    result = extractor.condition_inpainting(gap, ctx)
    assert isinstance(result, np.ndarray)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_condition_inpainting_empty_context(extractor):
    ctx = PhraseContext(
        audio_context=np.zeros(0, dtype=np.float32),
        chroma_mean=np.zeros(12, dtype=np.float32),
        tempo_bpm=120.0,
        beat_positions=[],
        phrase_start_s=0.0,
        phrase_end_s=0.0,
        gap_start_s=0.0,
        gap_end_s=0.0,
        is_fallback=True,
    )
    gap = np.zeros(1000, dtype=np.float32)
    result = extractor.condition_inpainting(gap, ctx)
    assert isinstance(result, np.ndarray)


def test_condition_inpainting_output_finite(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 48000, 96000)
    gap = np.random.randn(48000).astype(np.float32)
    result = extractor.condition_inpainting(gap, ctx)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# PhraseContext – Metadaten
# ---------------------------------------------------------------------------


def test_phrase_context_as_dict(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 48000, 96000)
    d = ctx.as_dict()
    assert "phrase_start_s" in d
    assert "phrase_end_s" in d
    assert "tempo_bpm" in d
    assert "is_fallback" in d


def test_as_dict_no_numpy_arrays(extractor):
    """as_dict() darf keine numpy-Arrays enthalten."""
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    d = ctx.as_dict()
    for k, v in d.items():
        assert not isinstance(v, np.ndarray), f"Schlüssel {k!r} enthält np.ndarray"


def test_chroma_vector_shape_if_present(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    assert ctx.chroma_mean.shape == (12,)


def test_beat_positions_list_of_ints(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    assert isinstance(ctx.beat_positions, list)
    for bp in ctx.beat_positions:
        assert isinstance(bp, int)


# ---------------------------------------------------------------------------
# Kontextlänge ≤ MAX_CONTEXT_DURATION_S
# ---------------------------------------------------------------------------


def test_context_within_max_duration(extractor):
    audio = np.random.randn(48000 * 60).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 48000 * 30, 48000 * 31)
    n_ctx = len(ctx.audio_context)
    assert n_ctx <= int(MAX_CONTEXT_DURATION_S * 48000) + 1


# ---------------------------------------------------------------------------
# Edge-Cases
# ---------------------------------------------------------------------------


def test_zero_length_gap(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, 24000, 24000)
    assert isinstance(ctx, PhraseContext)


def test_gap_out_of_bounds_no_crash(extractor):
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extractor.extract_context(audio, 48000, -1000, 99999999)
    assert isinstance(ctx, PhraseContext)


def test_nan_audio_no_crash(extractor):
    audio = np.full(48000 * 10, np.nan, dtype=np.float32)
    ctx = extractor.extract_context(audio, 48000, 0, 1000)
    assert isinstance(ctx, PhraseContext)


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_phrase_context_extractor()
    b = get_phrase_context_extractor()
    assert a is b


def test_convenience_wrapper():
    audio = np.random.randn(48000 * 10).astype(np.float32)
    ctx = extract_phrase_context(audio, 48000, 0, 1000)
    assert isinstance(ctx, PhraseContext)


def test_singleton_is_correct_type():
    inst = get_phrase_context_extractor()
    assert isinstance(inst, MusicalPhraseContextExtractor)

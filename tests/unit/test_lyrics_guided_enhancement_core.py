from __future__ import annotations

import pytest

"""Core tests for §2.36 LyricsGuidedEnhancement production path.

This test file intentionally validates only the authoritative production module:
`backend.core.lyrics_guided_enhancement`.
Legacy modules under `backend.lyrics_guided` are obsolete for production tests.
"""


import numpy as np

from backend.core.lyrics_guided_enhancement import (
    ContentAwareProcessor,
    LyricsGuidedEnhancement,
    LyricsTranscriptionResult,
    WordTimestamp,
    get_lyrics_guided_enhancement,
)


def _make_audio(sr: int = 48_000, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_singleton_returns_same_instance() -> None:
    a = get_lyrics_guided_enhancement()
    b = get_lyrics_guided_enhancement()
    assert a is b


def test_enhance_returns_bounded_audio_and_result() -> None:
    engine = LyricsGuidedEnhancement()
    audio = _make_audio(seconds=0.6)

    out, tr = engine.enhance(audio, 48_000)

    assert isinstance(out, np.ndarray)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    assert isinstance(tr, LyricsTranscriptionResult)


def test_transcribe_result_privacy_invariant() -> None:
    engine = LyricsGuidedEnhancement()
    audio = _make_audio(seconds=0.5)

    tr = engine.transcribe(audio, 48_000)

    assert isinstance(tr.words, list)
    for w in tr.words:
        assert isinstance(w, WordTimestamp)
        assert w.word == ""
        assert isinstance(w.phoneme_type, str)


def test_content_aware_processor_has_required_saliency_keys() -> None:
    required = {
        "fricative_stressed",
        "fricative_unstressed",
        "vowel_stressed",
        "plosive",
        "silence",
    }
    assert required.issubset(set(ContentAwareProcessor.SALIENCY_BOOST.keys()))


def _transcription(vocal_presence: float, confidence: float) -> LyricsTranscriptionResult:
    return LyricsTranscriptionResult(
        words=[
            WordTimestamp(
                word="[vocal]",
                start_s=0.2,
                end_s=0.4,
                confidence=confidence,
                is_stressed=False,
                phoneme_type="fricative_stressed",
            )
        ],
        language="en",
        overall_confidence=confidence,
        duration_s=0.6,
        fallback_used=False,
        vocal_presence=vocal_presence,
    )


def test_vocal_gate_instrumental_unchanged() -> None:
    """§Gesangs-Gate: Instrumental-Passage (vocal_presence < 0.15) bleibt bit-identisch."""
    engine = ContentAwareProcessor()
    audio = _make_audio(seconds=0.6)
    out = engine.apply_phoneme_dsp_to_audio(audio, _transcription(0.05, 0.8), sr=48_000, strength=0.5)
    assert np.array_equal(out, audio)


def test_vocal_gate_weak_word_skipped() -> None:
    """§Gesangs-Gate: Halluzinierte Wörter (confidence < 0.30) werden übersprungen → unverändert."""
    engine = ContentAwareProcessor()
    audio = _make_audio(seconds=0.6)
    out = engine.apply_phoneme_dsp_to_audio(audio, _transcription(1.0, 0.10), sr=48_000, strength=0.5)
    assert np.array_equal(out, audio)


def test_vocal_gate_active_presence_processes() -> None:
    """§Gesangs-Gate: Sichere Präsenz + konfidentes Wort → Phonem-DSP aktiv."""
    engine = ContentAwareProcessor()
    audio = _make_audio(seconds=0.6)
    out = engine.apply_phoneme_dsp_to_audio(audio, _transcription(1.0, 0.8), sr=48_000, strength=0.5)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    _seg = int(0.2 * 48_000)
    assert not np.array_equal(out, audio)  # Segment wurde bearbeitet
    assert np.max(np.abs(out[_seg : _seg + 4800] - audio[_seg : _seg + 4800])) > 1e-6

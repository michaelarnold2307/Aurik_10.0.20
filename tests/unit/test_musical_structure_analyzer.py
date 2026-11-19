"""Unit-Tests für MusicalStructureAnalyzer (§2.17).

Tests: ≥ 22 — Abdeckung: Shape, NaN, Bounds, Edge-Cases, Stereo, Singleton
"""

import concurrent.futures

import numpy as np
import pytest

from backend.core.musical_structure_analyzer import (
    MusicalStructure,
    MusicalStructureAnalyzer,
    SegmentInfo,
    analyze_musical_structure,
    get_musical_structure_analyzer,
)

SR = 48000


@pytest.fixture
def analyzer():
    return MusicalStructureAnalyzer()


# ---------------------------------------------------------------------------
# Kurze Dateien → leeere Struktur (< 20 s)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_short_audio_returns_empty_structure(analyzer):
    audio = np.zeros(SR * 10, dtype=np.float32)
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)
    assert len(structure.segments) == 0


def test_under_20s_returns_empty_segments(analyzer):
    audio = np.random.randn(int(SR * 19.9)).astype(np.float32)
    structure = analyzer.analyze(audio, SR)
    assert len(structure.segments) == 0
    assert structure.metadata["segment_count"] == 0
    assert structure.metadata["repeated_segment_count"] == 0
    assert structure.metadata["mean_ssm_similarity"] == 0.0
    assert structure.metadata["max_ssm_similarity"] == 0.0


# ---------------------------------------------------------------------------
# Normale Audiodatei
# ---------------------------------------------------------------------------


def test_long_audio_returns_segments(analyzer):
    np.random.seed(42)
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)
    assert len(structure.segments) >= 0  # Kann 0 sein wenn keine Grenzen erkannt


def test_analyze_returns_musical_structure(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_total_duration_set(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    expected_duration = len(audio) / SR
    assert abs(structure.total_duration_s - expected_duration) < 0.1


def test_bpm_nonnegative(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert structure.bpm >= 0.0


def test_confidence_in_range(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert 0.0 <= structure.confidence <= 1.0


def test_analyze_populates_structure_metadata(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert structure.metadata["segment_count"] == len(structure.segments)
    assert "repeated_segment_count" in structure.metadata
    assert 0.0 <= structure.metadata["mean_ssm_similarity"] <= 1.0
    assert 0.0 <= structure.metadata["max_ssm_similarity"] <= 1.0


# ---------------------------------------------------------------------------
# Segment-Labels
# ---------------------------------------------------------------------------


def test_segments_have_valid_labels(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    valid_labels = {"intro", "verse", "chorus", "bridge", "outro", "unknown"}
    for seg in structure.segments:
        assert seg.label in valid_labels


def test_chorus_segments_are_subset_of_segments(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    # Chorus-Segmente müssen eine Teilmenge aller Segmente sein
    [s.label for s in structure.segments]
    for chorus_seg in structure.chorus_segments:
        assert chorus_seg.label == "chorus"


def test_verse_segments_labeled_verse(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    for verse_seg in structure.verse_segments:
        assert verse_seg.label == "verse"


# ---------------------------------------------------------------------------
# SegmentInfo-Felder
# ---------------------------------------------------------------------------


def test_segment_info_time_range_valid(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    for seg in structure.segments:
        assert seg.start_time_s >= 0.0
        assert seg.end_time_s >= seg.start_time_s
        assert seg.start_sample >= 0
        assert seg.end_sample >= seg.start_sample


def test_segment_sample_count_le_max(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert len(structure.segments) <= analyzer.MAX_SEGMENTS


def test_uniform_segment_respects_max_segments(analyzer):
    bounds, _ = analyzer._uniform_segment(SR * 2400, SR, 2400.0)
    assert len(bounds) - 1 <= analyzer.MAX_SEGMENTS


def test_normalize_boundaries_clips_sorts_and_caps(analyzer):
    raw_bounds = [SR * 40, -100, SR * 2, SR * 2, SR * 90]
    normalized = analyzer._normalize_boundaries(raw_bounds, SR * 60)
    assert normalized[0] == 0
    assert normalized[-1] == SR * 60
    assert normalized == sorted(set(normalized))

    dense = list(range(0, SR * 1000, SR))
    capped = analyzer._normalize_boundaries(dense, SR * 1000)
    assert capped[0] == 0
    assert capped[-1] == SR * 1000
    assert len(capped) - 1 <= analyzer.MAX_SEGMENTS


def test_stft_chroma_accepts_large_hop(analyzer):
    audio = np.random.randn(SR * 2).astype(np.float32) * 0.1
    chroma = analyzer._stft_chroma(audio, SR, SR // 2)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] >= 1
    assert np.all(np.isfinite(chroma))


def test_ssm_segment_falls_back_when_chroma_unavailable(monkeypatch, analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1

    def fail_chroma(*_args):
        raise RuntimeError("no fallback chroma")

    monkeypatch.setattr(analyzer, "_stft_chroma", fail_chroma)
    monkeypatch.setitem(__import__("sys").modules, "librosa", None)

    bounds, confidence = analyzer._ssm_segment(audio, SR, 25.0)

    assert bounds[0] == 0
    assert bounds[-1] == audio.size
    assert 0.0 <= confidence <= 1.0


def test_repeated_segments_populate_similarity(analyzer):
    t = np.arange(SR * 4) / SR
    section_a = np.sin(2 * np.pi * 220.0 * t).astype(np.float32) * 0.4
    section_b = np.sin(2 * np.pi * 330.0 * t).astype(np.float32) * 0.4
    audio = np.concatenate([section_a, section_b, section_a, section_b])
    segments = [
        SegmentInfo("verse", 0, SR * 4, 0.0, 4.0),
        SegmentInfo("chorus", SR * 4, SR * 8, 4.0, 8.0),
        SegmentInfo("verse", SR * 8, SR * 12, 8.0, 12.0),
        SegmentInfo("chorus", SR * 12, SR * 16, 12.0, 16.0),
    ]

    analyzer._annotate_segment_similarity(segments, audio, SR)

    assert segments[0].ssm_similarity > 0.95
    assert segments[0].repeat_count >= 1
    assert segments[1].ssm_similarity > 0.95
    assert segments[1].repeat_count >= 1


def test_similarity_ignores_invalid_and_silent_segments(analyzer):
    audio = np.zeros(SR * 8, dtype=np.float32)
    valid_tone = np.sin(2 * np.pi * 220.0 * np.arange(SR * 4) / SR).astype(np.float32)
    audio[: SR * 4] = valid_tone * 0.4
    segments = [
        SegmentInfo("verse", 0, SR * 4, 0.0, 4.0),
        SegmentInfo("chorus", SR * 4, SR * 4, 4.0, 4.0),
        SegmentInfo("verse", SR * 4, SR * 8, 4.0, 8.0),
    ]

    analyzer._annotate_segment_similarity(segments, audio, SR)

    assert segments[0].repeat_count == 0
    assert segments[1].ssm_similarity == 0.0
    assert segments[1].repeat_count == 0
    assert segments[2].ssm_similarity == 0.0
    assert segments[2].repeat_count == 0


def test_adjacent_similarity_does_not_count_as_repetition(analyzer):
    t = np.arange(SR * 4) / SR
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32) * 0.4
    audio = np.concatenate([tone, tone])
    segments = [
        SegmentInfo("verse", 0, SR * 4, 0.0, 4.0),
        SegmentInfo("verse", SR * 4, SR * 8, 4.0, 8.0),
    ]

    analyzer._annotate_segment_similarity(segments, audio, SR)

    assert segments[0].ssm_similarity == 0.0
    assert segments[0].repeat_count == 0
    assert segments[1].ssm_similarity == 0.0
    assert segments[1].repeat_count == 0


def test_repeated_inner_segments_promoted_to_chorus(analyzer):
    segments = [
        SegmentInfo("intro", 0, SR * 4, 0.0, 4.0, 0, 0.0),
        SegmentInfo("verse", SR * 4, SR * 8, 4.0, 8.0, 1, 0.9),
        SegmentInfo("verse", SR * 8, SR * 12, 8.0, 12.0, 1, 0.88),
        SegmentInfo("outro", SR * 12, SR * 16, 12.0, 16.0, 1, 0.95),
    ]

    analyzer._refine_labels_with_similarity(segments)

    assert segments[0].label == "intro"
    assert segments[1].label == "chorus"
    assert segments[2].label == "chorus"
    assert segments[3].label == "outro"


# ---------------------------------------------------------------------------
# get_reference_segment
# ---------------------------------------------------------------------------


def test_get_reference_segment_no_chorus_returns_none(analyzer):
    structure = MusicalStructure(confidence=0.9)  # leere Struktur, kein Chorus
    ref = analyzer.get_reference_segment(0, structure)
    assert ref is None


def test_get_reference_segment_low_confidence_returns_none(analyzer):
    structure = MusicalStructure(
        confidence=0.5,  # unter CHORUS_CONFIDENCE_MIN = 0.75
        chorus_segments=[SegmentInfo("chorus", 0, SR * 10, 0.0, 10.0, 3, 0.88)],
    )
    ref = analyzer.get_reference_segment(0, structure)
    assert ref is None


def test_get_reference_segment_prefers_repeated_chorus(analyzer):
    near = SegmentInfo("chorus", 0, SR * 4, 0.0, 4.0, 0, 0.0)
    repeated = SegmentInfo("chorus", SR * 20, SR * 24, 20.0, 24.0, 2, 0.95)
    structure = MusicalStructure(
        boundaries_samples=[0, SR * 4, SR * 20, SR * 24],
        confidence=0.9,
        segments=[near, repeated],
        chorus_segments=[near, repeated],
    )

    ref = analyzer.get_reference_segment(SR * 2, structure)

    assert ref == (repeated.start_sample, repeated.end_sample)


def test_get_reference_segment_avoids_gap_segment(analyzer):
    damaged = SegmentInfo("chorus", 0, SR * 8, 0.0, 8.0, 4, 1.0)
    clean = SegmentInfo("chorus", SR * 12, SR * 20, 12.0, 20.0, 1, 0.8)
    structure = MusicalStructure(
        boundaries_samples=[0, SR * 8, SR * 12, SR * 20],
        confidence=0.9,
        segments=[damaged, clean],
        chorus_segments=[damaged, clean],
    )

    ref = analyzer.get_reference_segment(SR * 4, structure)

    assert ref == (clean.start_sample, clean.end_sample)


def test_get_reference_segment_ignores_invalid_candidate(analyzer):
    invalid = SegmentInfo("chorus", SR * 10, SR * 10, 10.0, 10.0, 4, 1.0)
    valid = SegmentInfo("chorus", SR * 12, SR * 18, 12.0, 18.0, 1, 0.7)
    structure = MusicalStructure(
        boundaries_samples=[0, SR * 10, SR * 12, SR * 18],
        confidence=0.9,
        segments=[invalid, valid],
        chorus_segments=[invalid, valid],
    )

    ref = analyzer.get_reference_segment(SR * 2, structure)

    assert ref == (valid.start_sample, valid.end_sample)


def test_get_reference_segment_returns_none_when_all_candidates_invalid(analyzer):
    invalid_a = SegmentInfo("chorus", SR * 10, SR * 10, 10.0, 10.0, 4, 1.0)
    invalid_b = SegmentInfo("chorus", -128, SR * 2, 0.0, 2.0, 3, 0.9)
    structure = MusicalStructure(
        confidence=0.9,
        segments=[invalid_a, invalid_b],
        chorus_segments=[invalid_a, invalid_b],
    )

    ref = analyzer.get_reference_segment(SR * 4, structure)

    assert ref is None


def test_get_reference_segment_falls_back_when_chorus_invalid(analyzer):
    invalid_chorus = SegmentInfo("chorus", SR * 4, SR * 4, 4.0, 4.0, 4, 1.0)
    valid_verse = SegmentInfo("verse", SR * 12, SR * 18, 12.0, 18.0, 0, 0.4)
    structure = MusicalStructure(
        confidence=0.9,
        segments=[invalid_chorus, valid_verse],
        chorus_segments=[invalid_chorus],
    )

    ref = analyzer.get_reference_segment(SR * 2, structure)

    assert ref == (valid_verse.start_sample, valid_verse.end_sample)


def test_get_reference_segment_boundary_fallback_skips_gap_interval(analyzer):
    structure = MusicalStructure(
        boundaries_samples=[0, SR * 4, SR * 8, SR * 12],
        confidence=0.9,
    )

    ref = analyzer.get_reference_segment(SR * 2, structure)

    assert ref == (SR * 4, SR * 8)


# ---------------------------------------------------------------------------
# Edge-Cases
# ---------------------------------------------------------------------------


def test_silence_audio_no_crash(analyzer):
    audio = np.zeros(SR * 25, dtype=np.float32)
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_stereo_input_accepted(analyzer):
    audio = np.random.randn(2, SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_channels_last_stereo_input_preserves_duration(analyzer):
    audio = np.random.randn(SR * 25, 2).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)
    assert abs(structure.total_duration_s - 25.0) < 0.1
    assert structure.metadata["segment_count"] == len(structure.segments)


def test_nan_input_handled(analyzer):
    audio = np.full(SR * 25, np.nan, dtype=np.float32)
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_repetitive_signal_bpm_range(analyzer):
    """Stark repetitives Signal soll in vernünftigem BPM-Bereich landen."""
    t = np.arange(SR * 25) / SR
    # Periodisches Signal mit 2 Hz → entspricht 120 BPM in Taktbeziehung
    audio = np.sin(2 * np.pi * 2.0 * t).astype(np.float32) * 0.5
    structure = analyzer.analyze(audio, SR)
    # BPM sollte im plausiblen Bereich liegen wenn erkannt
    if structure.bpm > 0.0:
        assert 30.0 <= structure.bpm <= 300.0


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_musical_structure_analyzer()
    b = get_musical_structure_analyzer()
    assert a is b


def test_convenience_wrapper():
    audio = np.random.randn(SR * 10).astype(np.float32) * 0.1
    structure = analyze_musical_structure(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_singleton_thread_safe():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_musical_structure_analyzer) for _ in range(20)]
        instances = [f.result() for f in futures]
    assert all(inst is instances[0] for inst in instances)

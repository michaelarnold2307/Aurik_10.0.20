import pytest

"""Tests für LyricsSentimentAnalyzer (§LSM-1).

Spec: §LSM-1 NLP-Lyrik-Sentiment-Modell
"""

import numpy as np


@pytest.mark.unit
class TestLyricsSentimentAnalyzerBase:
    def test_singleton_returns_same_instance(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        a1 = get_lyrics_sentiment_analyzer()
        a2 = get_lyrics_sentiment_analyzer()
        assert a1 is a2

    def test_neutral_fallback_empty_transcription(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        analyzer = get_lyrics_sentiment_analyzer()
        result = analyzer.analyze(None, 30.0)
        assert result.model_used == "neutral_fallback"
        assert result.dominant_emotion == "neutral"
        assert len(result.segments) >= 1

    def test_neutral_fallback_empty_words(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        class FakeTranscription:
            words = []

        analyzer = get_lyrics_sentiment_analyzer()
        result = analyzer.analyze(FakeTranscription(), 30.0)
        assert result.model_used == "neutral_fallback"


class TestKeywordHeuristic:
    def _make_transcription(self, word_list, times=None):
        """Erstellt ein Fake-Transcription-Objekt mit Wörtern und Timestamps."""

        class Word:
            def __init__(self, word, start, end):
                self.word = word
                self.start = start
                self.end = end

        class Transcription:
            def __init__(self, words):
                self.words = words

        if times is None:
            times = [(i * 1.0, (i + 1) * 1.0) for i in range(len(word_list))]

        words = [Word(w, s, e) for w, (s, e) in zip(word_list, times)]
        return Transcription(words)

    def test_sad_keywords_produce_negative_valence(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        t = self._make_transcription(["sad", "cry", "alone", "broken"])
        result = get_lyrics_sentiment_analyzer().analyze(t, 10.0)
        assert result.valence_mean < 0.0

    def test_joyful_keywords_produce_positive_valence(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        t = self._make_transcription(["love", "happy", "joy", "smile"])
        result = get_lyrics_sentiment_analyzer().analyze(t, 10.0)
        assert result.valence_mean > 0.0

    def test_no_sad_keywords_produces_neutral_fallback_or_neutral(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        t = self._make_transcription(["the", "a", "and", "but"])
        result = get_lyrics_sentiment_analyzer().analyze(t, 10.0)
        # Keine Keyword-Matches → Neutral
        assert result.dominant_emotion in ("neutral", "tender", "longing")

    def test_dominant_emotion_computed(self):
        from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

        t = self._make_transcription(
            ["love", "heart", "together", "hope"],
            times=[(0, 3), (3, 5), (5, 7), (7, 9)],
        )
        result = get_lyrics_sentiment_analyzer().analyze(t, 10.0)
        assert result.dominant_emotion != ""
        assert result.model_used == "keyword_heuristic"


class TestSegmentSentiment:
    def test_dsp_params_auto_populated(self):
        from backend.core.lyrics_sentiment_analyzer import SegmentSentiment

        seg = SegmentSentiment(
            start_s=0.0,
            end_s=8.0,
            emotion="sad",
            valence=-0.7,
            arousal=-0.5,
            dominance=-0.6,
            confidence=0.8,
        )
        assert "dynamics_scale" in seg.dsp_params
        assert seg.dsp_params["dynamics_scale"] < 1.0  # Traurig → gedämpftere Dynamik

    def test_get_emotion_at_returns_correct_segment(self):
        from backend.core.lyrics_sentiment_analyzer import LyricsSentimentResult, SegmentSentiment

        segs = [
            SegmentSentiment(0.0, 8.0, "sad", -0.7, -0.5, -0.6, 0.8),
            SegmentSentiment(8.0, 16.0, "joyful", 0.8, 0.7, 0.6, 0.9),
        ]
        result = LyricsSentimentResult(
            segments=segs,
            dominant_emotion="sad",
            valence_mean=-0.1,
            arousal_mean=0.1,
            dominance_mean=0.0,
            model_used="keyword_heuristic",
        )
        seg_at_4 = result.get_emotion_at(4.0)
        assert seg_at_4 is not None
        assert seg_at_4.emotion == "sad"

        seg_at_12 = result.get_emotion_at(12.0)
        assert seg_at_12 is not None
        assert seg_at_12.emotion == "joyful"

    def test_get_dynamics_scale_at_returns_float(self):
        from backend.core.lyrics_sentiment_analyzer import LyricsSentimentResult, SegmentSentiment

        segs = [SegmentSentiment(0.0, 30.0, "intimate", 0.5, -0.5, -0.1, 0.7)]
        result = LyricsSentimentResult(
            segments=segs,
            dominant_emotion="intimate",
            valence_mean=0.5,
            arousal_mean=-0.5,
            dominance_mean=-0.1,
            model_used="keyword_heuristic",
        )
        scale = result.get_dynamics_scale_at(10.0)
        assert isinstance(scale, float)
        assert scale < 1.0  # Intim → gedämpftere Dynamik

    def test_get_emotion_at_outside_range_returns_none(self):
        from backend.core.lyrics_sentiment_analyzer import LyricsSentimentResult, SegmentSentiment

        segs = [SegmentSentiment(0.0, 10.0, "neutral", 0.0, 0.0, 0.0, 0.0)]
        result = LyricsSentimentResult(
            segments=segs,
            dominant_emotion="neutral",
            valence_mean=0.0,
            arousal_mean=0.0,
            dominance_mean=0.0,
            model_used="neutral_fallback",
        )
        assert result.get_emotion_at(15.0) is None


class TestVADAnchors:
    def test_vad_to_emotion_sad_cluster(self):
        from backend.core.lyrics_sentiment_analyzer import _vad_to_emotion

        vad = np.array([-0.75, -0.50, -0.60], dtype=np.float32)
        emotion = _vad_to_emotion(vad)
        assert emotion == "sad"

    def test_vad_to_emotion_joyful_cluster(self):
        from backend.core.lyrics_sentiment_analyzer import _vad_to_emotion

        vad = np.array([+0.85, +0.70, +0.60], dtype=np.float32)
        emotion = _vad_to_emotion(vad)
        assert emotion == "joyful"

    def test_vad_to_emotion_neutral_near_zero(self):
        from backend.core.lyrics_sentiment_analyzer import _vad_to_emotion

        vad = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        emotion = _vad_to_emotion(vad)
        assert emotion == "neutral"

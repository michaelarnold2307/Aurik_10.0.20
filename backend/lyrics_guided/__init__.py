"""
Lyrics-Guided Vocal Enhancement Module
=======================================

World's first lyrics-synchronized audio processing with semantic understanding.

This module provides:
- Automatic lyrics transcription (Whisper)
- Phoneme-level alignment (MFA)
- Word-level timestamps
- Semantic audio analysis
- Integrated processing timeline
- Phoneme-specific processing recommendations

Components:
-----------
- LyricsAligner: Align lyrics with phoneme-level precision
- ContentAwareProcessor: Detect vocal/instrumental/breath segments
- SemanticAudioAnalyzer: Analyze audio semantically (no genre labels)
- IntegratedVocalProcessor: Combine all analyses for comprehensive processing

Quick Start:
-----------
>>> from backend.lyrics_guided import create_integrated_vocal_timeline
>>>
>>> timeline = create_integrated_vocal_timeline(
...     audio, sr=48000, aurik_mode="restoration", language="en"
... )
>>>
>>> # Get processing at timestamp
>>> seg = timeline.get_processing_at_time(2.5)
>>> print(f"Word: {seg.lyrics_text}")
>>> print(f"Sibilant reduction: {seg.sibilant_reduction:.2f}")

Author: AURIK Development Team
Version: 1.0.0
Date: 11. Februar 2026
"""

# Content-Aware Processor
from backend.lyrics_guided.content_aware_processor import (
    ContentAwareProcessor,
    ContentSegment,
    ContentType,
    LyricsGuidedTimeline,
    LyricsSegment,
    ProcessingIntent,
    create_lyrics_guided_timeline,
)

# Integrated Vocal Processor
from backend.lyrics_guided.integrated_vocal_processor import (
    IntegratedVocalProcessor,
    IntegratedVocalSegment,
    IntegratedVocalTimeline,
    VocalProcessingMode,
    create_integrated_vocal_timeline,
)

# Lyrics Aligner
from backend.lyrics_guided.lyrics_aligner import (
    LyricsAligner,
    LyricsAlignment,
    PhonemeAlignment,
    WordAlignment,
)

__all__ = [
    # Content-Aware Processor
    "ContentAwareProcessor",
    "ContentSegment",
    "ContentType",
    # Integrated Vocal Processor
    "IntegratedVocalProcessor",
    "IntegratedVocalSegment",
    "IntegratedVocalTimeline",
    # Lyrics Aligner
    "LyricsAligner",
    "LyricsAlignment",
    "LyricsGuidedTimeline",
    "LyricsSegment",
    "PhonemeAlignment",
    "ProcessingIntent",
    "VocalProcessingMode",
    "WordAlignment",
    "create_integrated_vocal_timeline",
    "create_lyrics_guided_timeline",
]

__version__ = "1.0.0"
__author__ = "AURIK Development Team"

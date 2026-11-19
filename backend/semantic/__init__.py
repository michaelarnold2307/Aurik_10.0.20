"""
Semantic Audio Understanding Module
====================================

Genre-agnostic semantic audio analysis for AURIK restoration.

This module provides:
- Instrument detection (vocals, drums, bass, guitar, etc.)
- Content characterization (transient vs. sustained)
- Semantic tagging without genre classification
- Processing recommendations based on audio content

Components:
-----------
- SemanticAudioAnalyzer: Main analyzer for semantic understanding
- InstrumentType: Enum for detected instruments
- ContentCharacter: Enum for content characteristics
- ProcessingStrategy: Enum for recommended processing approaches
- SemanticProfile: Complete semantic analysis result

Quick Start:
-----------
>>> from backend.semantic import analyze_semantic_content
>>>
>>> profile = analyze_semantic_content(
...     audio, sr=48000, aurik_mode="restoration"
... )
>>>
>>> print(f"Dominant: {profile.dominant_instrument.value}")
>>> print(f"Character: {profile.content_character.value}")
>>> print(f"Preserve transients: {profile.preserve_transients}")

Author: AURIK Development Team
Version: 1.0.0
Date: 11. Februar 2026
"""

from backend.semantic.semantic_audio_analyzer import (
    ContentCharacter,
    InstrumentPresence,
    InstrumentType,
    ProcessingStrategy,
    SemanticAudioAnalyzer,
    SemanticProfile,
    analyze_semantic_content,
)

__all__ = [
    "ContentCharacter",
    "InstrumentPresence",
    "InstrumentType",
    "ProcessingStrategy",
    "SemanticAudioAnalyzer",
    "SemanticProfile",
    "analyze_semantic_content",
]

__version__ = "1.0.0"
__author__ = "AURIK Development Team"

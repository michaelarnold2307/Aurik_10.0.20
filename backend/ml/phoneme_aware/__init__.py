"""
Phoneme-Aware Processing Module (Phase 2 Week 7-9)

This module provides phoneme detection and classification for intelligent
audio processing. Key features:

- Wav2Vec2-based phoneme detection
- IPA phoneme classification
- Linguistic category categorization
- Multi-language support

Author: Aurik Development Team
Version: 1.0.0
Date: February 2026
"""

from backend.ml.phoneme_aware.phoneme_classifier import PhonemeCategory, PhonemeClassifier, SibilantType
from backend.ml.phoneme_aware.phoneme_detector import DetectionConfig, Language, PhonemeDetector, PhonemeSegment

__all__ = [
    "DetectionConfig",
    "Language",
    "PhonemeCategory",
    "PhonemeClassifier",
    "PhonemeDetector",
    "PhonemeSegment",
    "SibilantType",
]

__version__ = "1.0.0"

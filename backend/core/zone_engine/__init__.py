"""
AURIK v8 Zone Engine Package
=============================

Confidence-based audio processing zones and context analysis.

Components:
- ZoneEngine: Classify confidence into zones A/B/C with adaptive thresholds
- ContextAnalyzer: Medium detection and context analysis
- RegionAnalyzer: Audio-segment classification (silence/music/speech/noise)
- ZoneAwareContextAnalyzer: Integrated zone + context analysis

Version: 8.0.0
"""

from .context_analysis import ContextAnalyzer
from .zone_engine import Zone, ZoneAwareContextAnalyzer, ZoneClassification, ZoneEngine

__all__ = ["ContextAnalyzer", "Zone", "ZoneAwareContextAnalyzer", "ZoneClassification", "ZoneEngine"]

__version__ = "9.15.0"

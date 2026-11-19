"""
AURIK v10.0.0 Musical Goals Package
==================================

Messbare Metriken für alle 9 musikalischen Qualitätsziele:

1.  Bass-Kraft     – Kraftvolle Basswiedergabe (20–250 Hz)     ≥ 0.85
2.  Brillanz       – HF-Klarheit (8–20 kHz)                    ≥ 0.85
3.  Wärme          – Mittentiefe (200–2000 Hz)                  ≥ 0.80
4.  Natürlichkeit  – Gesamtklang ohne Artefakte                 ≥ 0.90
5.  Authentizität  – Klang-Fingerabdruck & Stimme               ≥ 0.88
6.  Emotionalität  – Dynamik & Ausdruck                         ≥ 0.87
7.  Transparenz    – Klarheit & Trennung                        ≥ 0.89
8.  Groove         – Mikro-Timing, Swing, DTW ≤ 8 ms RMS       ≥ 0.88  (v10.0.0)
9.  Spatial Depth  – Räumliche Tiefe & Stereo-Bild              ≥ 0.75  (v10.0.0)

Version: 10.0.0
"""

from .musical_goals_metrics import (
    AuthentizitaetMetric,
    BassKraftMetric,
    BrillanzMetric,
    EmotionalitaetMetric,
    GoalMeasurement,
    GrooveMetric,
    MusicalGoalsChecker,
    NatuerlichkeitMetric,
    SpatialDepthMetric,
    TransparenzMetric,
    WaermeMetric,
    get_checker,
    measure_all,
)
from .musical_goals_monitor import MonitoringCheckpoint, MonitoringReport, MusicalGoalsMonitor, PreValidationResult

__all__ = [
    "AuthentizitaetMetric",
    # Metriken (einzeln)
    "BassKraftMetric",
    "BrillanzMetric",
    "EmotionalitaetMetric",
    "GoalMeasurement",
    "GrooveMetric",  # v10.0.0 — 8. Ziel
    "MonitoringCheckpoint",
    "MonitoringReport",
    # Checker
    "MusicalGoalsChecker",
    # Monitor
    "MusicalGoalsMonitor",
    "NatuerlichkeitMetric",
    "PreValidationResult",
    "SpatialDepthMetric",  # v10.0.0 — 9. Ziel
    "TransparenzMetric",
    "WaermeMetric",
    # Singletons & Convenience
    "get_checker",
    "measure_all",
]

__version__ = "9.15.0"

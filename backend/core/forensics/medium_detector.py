"""
Rückwärtskompatibilitäts-Shim (§9.4 Anti-Parallelwelten-Prinzip).

Die kanonische Implementierung liegt in ``forensics/medium_detector.py``
(Top-Level-Paket, §6.7). Dieser Shim leitet alle Imports transparent
dorthin weiter — bestehende Aufrufer müssen nicht angepasst werden.
"""

from forensics.medium_detector import (
    MediumDetectionResult,
    MediumDetector,
    SpectralFingerprint,
    TransferChain,
    detect_medium_chain,
    get_medium_detector,
)

__all__ = [
    "MediumDetectionResult",
    "MediumDetector",
    "SpectralFingerprint",
    "TransferChain",
    "detect_medium_chain",
    "get_medium_detector",
]

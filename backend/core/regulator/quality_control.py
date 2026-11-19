"""Shim – leitet weiter an backend.quality_control (kanonisch).

Gemäß §9.4 Anti-Parallelwelten: kein eigener Code, nur Re-Export.
"""

from backend.quality_control import QualityControl

__all__ = ["QualityControl"]

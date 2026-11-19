"""
core.stereo_authenticity — VERALTET (Kompatibilitäts-Shim)
===========================================================

Dieses Modul ist ein reiner Re-Export-Shim für
``core.stereo_authenticity_invariant``.
Alle Klassen und Funktionen werden von dort bezogen.

Migrationsanleitung::

    # Alt:
    from backend.core.stereo_authenticity import StereoAuthResult
    # Neu:
    from backend.core.stereo_authenticity_invariant import StereoAuthResult

Dieser Shim wird in einer zukünftigen Version entfernt.
Referenz: §2.18 Aurik-9-Spec (v10.0.0)
"""

import warnings as _warnings

_warnings.warn(
    "core.stereo_authenticity ist veraltet. Verwende stattdessen core.stereo_authenticity_invariant.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.core.stereo_authenticity_invariant import (
    StereoAuthenticitiyInvariant,
    StereoAuthResult,
    _is_mono_source,
    check_stereo_authenticity,
    get_stereo_authenticity_invariant,
)

__all__ = [
    "StereoAuthResult",
    "StereoAuthenticitiyInvariant",
    "_is_mono_source",
    "check_stereo_authenticity",
    "get_stereo_authenticity_invariant",
]

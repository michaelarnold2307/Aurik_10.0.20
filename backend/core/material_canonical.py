"""Canonical material terminology helpers.

Provides a single source of truth for medium aliases and user-facing labels,
so backend and UI use unambiguous names.
"""

from __future__ import annotations

from typing import Any

_MATERIAL_ALIASES: dict[str, str] = {
    "cassette": "tape",
    "cassette_standard": "tape",
    "cassette_chrome": "tape",
    "tape_standard": "tape",
    "tape_studio": "reel_tape",
    "vinyl_standard": "vinyl",
    "vinyl_premium": "vinyl",
    "mp3_standard": "mp3_high",
    "cd": "cd_digital",
    "digital": "cd_digital",
}

_MATERIAL_LABELS_DE: dict[str, str] = {
    "wax_cylinder": "Wachswalze",
    "lacquer_disc": "Lackfolie",
    "shellac": "Schellack",
    "vinyl": "Vinyl-Schallplatte",
    "wire_recording": "Drahtband",
    "reel_tape": "Spulenband (Reel-to-Reel)",
    "tape": "Kassette (Band)",
    "dat": "DAT",
    "cd_digital": "CD / Digital",
    "minidisc": "MiniDisc",
    "mp3_low": "MP3 (niedrige Bitrate)",
    "mp3_high": "MP3",
    "aac": "AAC",
    "streaming": "Streaming-Format",
    "unknown": "Unbekannt",
}


def canonical_material_key(material: Any) -> str:
    """Gibt canonical, unambiguous material key used across modules zurück."""
    if material is None:
        return "unknown"

    raw = getattr(material, "value", material)
    key = str(raw).strip().lower()
    if not key:
        return "unknown"

    return _MATERIAL_ALIASES.get(key, key)


def material_label_de(material: Any) -> str:
    """Gibt stable German label for a material key or enum zurück."""
    key = canonical_material_key(material)
    return _MATERIAL_LABELS_DE.get(key, key)

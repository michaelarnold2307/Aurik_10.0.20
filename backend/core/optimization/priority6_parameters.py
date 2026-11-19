"""
optimization/priority6_parameters.py – Genreoptimierte Parameter und Presets.
==============================================================================
"""

from __future__ import annotations

from typing import Any


class GenreOptimizedParameters:
    """Genre-specific processing parameters."""

    _PARAMS: dict[str, dict[str, Any]] = {
        "jazz": {
            "denoiser_strength": 0.25,
            "bass_boost": 0.0,
            "presence_boost_db": 1.0,
            "warmth_db": 1.5,
            "stereo_width": 1.0,
        },
        "rock": {
            "denoiser_strength": 0.40,
            "bass_boost": 2.0,
            "presence_boost_db": 2.0,
            "warmth_db": 0.5,
            "stereo_width": 1.1,
        },
        "classical": {
            "denoiser_strength": 0.20,
            "bass_boost": 0.5,
            "presence_boost_db": 0.5,
            "warmth_db": 1.0,
            "stereo_width": 1.2,
        },
        "pop": {
            "denoiser_strength": 0.35,
            "bass_boost": 1.5,
            "presence_boost_db": 1.5,
            "warmth_db": 0.5,
            "stereo_width": 1.0,
        },
        "blues": {
            "denoiser_strength": 0.30,
            "bass_boost": 1.0,
            "presence_boost_db": 1.2,
            "warmth_db": 2.0,
            "stereo_width": 0.9,
        },
        "folk": {
            "denoiser_strength": 0.20,
            "bass_boost": 0.5,
            "presence_boost_db": 0.8,
            "warmth_db": 1.5,
            "stereo_width": 0.9,
        },
    }

    @classmethod
    def list_genres(cls) -> list[str]:
        """Gibt the list of supported genre identifiers zurück."""
        return list(cls._PARAMS.keys())

    @classmethod
    def get_parameters(cls, genre: str) -> dict[str, Any]:
        """Gibt processing parameters for *genre* zurück.

        Falls back to ``"rock"`` defaults for unknown genres.
        """
        return dict(cls._PARAMS.get(genre, cls._PARAMS["rock"]))


class OptimizedPresets:
    """Named presets bundling quality and performance expectations."""

    _PRESETS: dict[str, dict[str, Any]] = {
        "gentle": {
            "denoiser_strength": 0.15,
            "bass_boost": 0.5,
            "expected_quality": 0.75,
            "expected_performance": 2.0,  # RT-factor
        },
        "balanced": {
            "denoiser_strength": 0.35,
            "bass_boost": 1.0,
            "expected_quality": 0.82,
            "expected_performance": 3.0,
        },
        "aggressive": {
            "denoiser_strength": 0.65,
            "bass_boost": 2.0,
            "expected_quality": 0.88,
            "expected_performance": 5.0,
        },
    }

    @classmethod
    def list_presets(cls) -> list[str]:
        """Gibt the list of supported preset names zurück."""
        return list(cls._PRESETS.keys())

    @classmethod
    def get_preset(cls, name: str) -> dict[str, Any]:
        """Gibt the parameter dict for *name* zurück."""
        return dict(cls._PRESETS.get(name, cls._PRESETS["balanced"]))

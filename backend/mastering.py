"""Shim — forwards to canonical backend.core.regulator.mastering."""

from backend.core.regulator.mastering import (
    adaptive_eq,
    dither,
    limiter,
    lufs_normalize,
    mastering_chain,
    multiband_compress,
    simple_compressor,
    simple_eq,
    stereo_enhance,
)

__all__ = [
    "adaptive_eq",
    "dither",
    "limiter",
    "lufs_normalize",
    "mastering_chain",
    "multiband_compress",
    "simple_compressor",
    "simple_eq",
    "stereo_enhance",
]

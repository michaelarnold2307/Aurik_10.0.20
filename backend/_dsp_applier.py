"""Shim — forwards to canonical backend.core.regulator._dsp_applier."""

from backend.core.regulator._dsp_applier import (
    _ALWAYS_APPLY,
    _RESTORATION_BLOCKED_MODULES,
    _SKIP_GATE_THRESHOLD_DB,
    _apply_dsp_module,
    _compute_snr_db,
    _current_processing_mode,
    apply_dsp_chain,
    apply_dsp_chain_tuple,
    compressor,
    dsp_effects,
    enhancer,
    eq,
    limiter,
    set_dsp_processing_mode,
)

__all__ = [
    "_ALWAYS_APPLY",
    "_RESTORATION_BLOCKED_MODULES",
    "_SKIP_GATE_THRESHOLD_DB",
    "_apply_dsp_module",
    "_compute_snr_db",
    "_current_processing_mode",
    "apply_dsp_chain",
    "apply_dsp_chain_tuple",
    "compressor",
    "dsp_effects",
    "enhancer",
    "eq",
    "limiter",
    "set_dsp_processing_mode",
]

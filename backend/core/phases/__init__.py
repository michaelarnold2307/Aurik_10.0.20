"""
Aurik 10.0 - Phase Module Exports
==================================

Central import hub for all 66 processing phases.

Usage:
    from backend.core.phases import ClickRemovalPhase, DenoisePhase
    from backend.core.phases import PhaseInterface, PhaseCategory, PhaseMetadata

Author: Aurik 10.0 Development Team
Version: 10.0.15
Date: 2026-07-11
"""

import logging as _logging

# Phase 01-09: Defect Removal & Basic Restoration
from .phase_01_click_removal import ClickRemovalPhase
from .phase_02_hum_removal import HumRemovalPhase
from .phase_03_denoise import DenoisePhase
from .phase_04_eq_correction import EQCorrectionPhase
from .phase_05_rumble_filter import RumbleFilterPhase
from .phase_06_frequency_restoration import FrequencyRestorationPhase
from .phase_07_harmonic_restoration import HarmonicRestorationPhase
from .phase_08_transient_preservation import TransientPreservationPhase
from .phase_09_crackle_removal import CrackleRemovalPhase

# Phase 10-19: Dynamics & Spatial Processing
from .phase_10_compression import CompressionPhase
from .phase_11_limiting import LimitingPhase
from .phase_12_wow_flutter_fix import WowFlutterFix
from .phase_13_stereo_enhancement import StereoEnhancementPhaseV2
from .phase_14_phase_correction import PhaseCorrection
from .phase_15_stereo_balance import StereoBalancePhaseV2
from .phase_16_final_eq import FinalEQ
from .phase_17_mastering_polish import MasteringPolishPhase
from .phase_18_noise_gate import NoiseGate
from .phase_19_de_esser import DeEsserPhase

# Phase 20-29: Advanced Restoration & Material-Specific
from .phase_20_reverb_reduction import ReverbReduction
from .phase_21_exciter import Exciter
from .phase_22_tape_saturation import TapeSaturation
from .phase_23_spectral_repair import SpectralRepair
from .phase_24_dropout_repair import DropoutRepairPhase
from .phase_25_azimuth_correction import AzimuthCorrectionPhaseV2
from .phase_26_dynamic_range_expansion import DynamicRangeExpansion
from .phase_27_click_pop_removal import ClickPopRemoval
from .phase_28_surface_noise_profiling import SurfaceNoiseProfiling
from .phase_29_tape_hiss_reduction import TapeHissReductionPhase

# Phase 30-39: Format & Enhancement
from .phase_30_dc_offset_removal import DCOffsetRemoval
from .phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase
from .phase_32_mono_to_stereo import MonoToStereoPhaseV2
from .phase_33_stereo_width_limiter import StereoWidthLimiterPhaseV2
from .phase_34_mid_side_processing import MidSideProcessing
from .phase_35_multiband_compression import MultibandCompressionPhase
from .phase_36_transient_shaper import TransientShaper
from .phase_37_bass_enhancement import BassEnhancement
from .phase_38_presence_boost import PresenceBoost
from .phase_39_air_band_enhancement import AirBandEnhancement

# Phase 40-42: Final Output Processing
from .phase_40_loudness_normalization import LoudnessNormalizationPhase
from .phase_41_output_format_optimization import OutputFormatOptimization
from .phase_42_vocal_enhancement import VocalEnhancement
from .phase_43_ml_deesser import AdaptiveDeEsserPhase, MLDeEsserPhase
from .phase_44_guitar_enhancement import GuitarEnhancementPhase
from .phase_45_brass_enhancement import BrassEnhancementPhase
from .phase_46_spatial_enhancement import SpatialEnhancementPhase
from .phase_47_truepeak_limiter import TruePeakLimiterPhase
from .phase_48_stereo_width_enhancer import StereoWidthEnhancerPhase
from .phase_49_advanced_dereverb import AdvancedDereverbPhase
from .phase_50_spectral_repair import SpectralRepairPhase

# Phase 50+: Tier 1 ML-Hybrid Enhancement (NEW)
from .phase_51_drums_enhancement import DrumsEnhancementV1
from .phase_52_piano_restoration import PianoRestorationV1
from .phase_54_transparent_dynamics import TransparentDynamicsV1
from .phase_55_diffusion_inpainting import DiffusionInpaintingPhase
from .phase_ambience_polish import AmbiencePolishPhase
from .phase_glue_stage import GlueStagePhase

# Phase Interface & Base Classes
from .phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)

_logger = _logging.getLogger(__name__)

# Phase 53: Semantic Audio Analysis — Metadata-only (BPM, Key, Genre-Hint)
try:
    from .phase_53_semantic_audio import SemanticAudioPhase

    _PHASE53_OK = True
except ImportError as _p53_err:
    _logger.debug("Phase 53 nicht verfügbar: %s", _p53_err)
    _PHASE53_OK = False
    SemanticAudioPhase = None  # type: ignore[assignment,misc]

# Phase 57: Print-Through Reduction — Bidirektionale LMS (§7.x DSP-Pflicht)
try:
    from .phase_57_print_through_reduction import PrintThroughReductionPhase

    _PHASE57_OK = True
except ImportError as _p57_err:
    _logger.debug("Phase 57 nicht verfügbar: %s", _p57_err)
    _PHASE57_OK = False
    PrintThroughReductionPhase = None  # type: ignore[assignment,misc]

# Phase 56: HEAD_WEAR Spectral Band Gap Repair (v10.0.0)
try:
    from .phase_56_spectral_band_gap_repair import SpectralBandGapRepairPhase

    _PHASE56_OK = True
except ImportError as _p56_err:
    _logger.debug("Phase 56 nicht verfügbar: %s", _p56_err)
    _PHASE56_OK = False
    SpectralBandGapRepairPhase = None  # type: ignore[assignment,misc]

# Phase 58: Lyrics-Guided Enhancement (§2.36 PFLICHT, v10.0.0.x)
try:
    from .phase_58_lyrics_guided_enhancement import Phase58LyricsGuidedEnhancement

    _PHASE58_OK = True
except ImportError as _p58_err:
    _logger.debug("Phase 58 nicht verfügbar: %s", _p58_err)
    _PHASE58_OK = False
    Phase58LyricsGuidedEnhancement = None  # type: ignore[assignment,misc]


# Phase 59: Modulation Noise Reduction — Tape modulation artifact removal (§6.7)
try:
    from .phase_59_modulation_noise_reduction import ModulationNoiseReductionPhase

    _PHASE59_OK = True
except ImportError as _p59_err:
    _logger.debug("Phase 59 nicht verfügbar: %s", _p59_err)
    _PHASE59_OK = False
    ModulationNoiseReductionPhase = None  # type: ignore[assignment,misc]

# Phase 60: Inner Groove Distortion Repair — Vinyl-specific (§6.8)
try:
    from .phase_60_inner_groove_distortion_repair import InnerGrooveDistortionRepairPhase

    _PHASE60_OK = True
except ImportError as _p60_err:
    _logger.debug("Phase 60 nicht verfügbar: %s", _p60_err)
    _PHASE60_OK = False
    InnerGrooveDistortionRepairPhase = None  # type: ignore[assignment,misc]

# Phase 61: Groove Echo Cancellation — Pre/post-echo removal (§6.9)
try:
    from .phase_61_groove_echo_cancellation import GrooveEchoCancellationPhase

    _PHASE61_OK = True
except ImportError as _p61_err:
    _logger.debug("Phase 61 nicht verfügbar: %s", _p61_err)
    _PHASE61_OK = False
    GrooveEchoCancellationPhase = None  # type: ignore[assignment,misc]

# Phase 62: Crosstalk Cancellation — Stereo bleed removal (§6.10)
try:
    from .phase_62_crosstalk_cancellation import CrosstalkCancellationPhase

    _PHASE62_OK = True
except ImportError as _p62_err:
    _logger.debug("Phase 62 nicht verfügbar: %s", _p62_err)
    _PHASE62_OK = False
    CrosstalkCancellationPhase = None  # type: ignore[assignment,misc]

# Phase 63: Intermodulation Reduction — IMD artifact cleanup (§6.11)
try:
    from .phase_63_intermodulation_reduction import IntermodulationReductionPhase

    _PHASE63_OK = True
except ImportError as _p63_err:
    _logger.debug("Phase 63 nicht verfügbar: %s", _p63_err)
    _PHASE63_OK = False
    IntermodulationReductionPhase = None  # type: ignore[assignment,misc]

# Phase 64: Tape Splice Repair — Physical tape splice artifact removal (§6.12)
try:
    from .phase_64_tape_splice_repair import TapeSpliceRepairPhase

    _PHASE64_OK = True
except ImportError as _p64_err:
    _logger.debug("Phase 64 nicht verfügbar: %s", _p64_err)
    _PHASE64_OK = False
    TapeSpliceRepairPhase = None  # type: ignore[assignment,misc]

# Phase 65: Vocal Naturalness Restoration — Formant/Vibrato preservation (§6.13)
try:
    from .phase_65_vocal_naturalness_restoration import VocalNaturalnessRestorationPhase

    _PHASE65_OK = True
except ImportError as _p65_err:
    _logger.debug("Phase 65 nicht verfügbar: %s", _p65_err)
    _PHASE65_OK = False
    VocalNaturalnessRestorationPhase = None  # type: ignore[assignment,misc]

# Phase 66: Stem-Targeted Noise Reduction — Source-separation NR (§6.14)
try:
    from .phase_66_stem_targeted_nr import StemTargetedNRPhase

    _PHASE66_OK = True
except ImportError as _p66_err:
    _logger.debug("Phase 66 nicht verfügbar: %s", _p66_err)
    _PHASE66_OK = False
    StemTargetedNRPhase = None  # type: ignore[assignment,misc]

# Exported symbols
__all__ = [
    "AdvancedDereverbPhase",
    "AirBandEnhancement",
    "AdaptiveDeEsserPhase",
    "AzimuthCorrectionPhaseV2",
    "BassEnhancement",
    "BrassEnhancementPhase",
    "ClickPopRemoval",
    # Phase 01-09
    "ClickRemovalPhase",
    # Phase 10-19
    "CompressionPhase",
    "CrackleRemovalPhase",
    # Phase 30-39
    "DCOffsetRemoval",
    "DeEsserPhase",
    "DenoisePhase",
    "DiffusionInpaintingPhase",
    "DropoutRepairPhase",
    "DrumsEnhancementV1",
    "DynamicRangeExpansion",
    "EQCorrectionPhase",
    "Exciter",
    "FinalEQ",
    "FrequencyRestorationPhase",
    "GlueStagePhase",
    "GuitarEnhancementPhase",
    "HarmonicRestorationPhase",
    "HumRemovalPhase",
    "LimitingPhase",
    "AmbiencePolishPhase",
    # Phase 40-48
    "LoudnessNormalizationPhase",
    "MLDeEsserPhase",
    "MasteringPolishPhase",
    "MidSideProcessing",
    "MonoToStereoPhaseV2",
    "MultibandCompressionPhase",
    "NoiseGate",
    "OutputFormatOptimization",
    "PhaseCategory",
    "PhaseCorrection",
    # Base Classes
    "PhaseInterface",
    "PhaseMetadata",
    "PhaseResult",
    "PianoRestorationV1",
    "PresenceBoost",
    # Phase 20-29
    "ReverbReduction",
    "RumbleFilterPhase",
    "SpatialEnhancementPhase",
    "SpectralBandGapRepairPhase",
    "SpectralRepair",
    "SemanticAudioPhase",
    "PrintThroughReductionPhase",
    # Phase 50+ (Tier 1 ML-Hybrid)
    "SpectralRepairPhase",
    "SpeedPitchCorrectionPhase",
    "StereoBalancePhaseV2",
    "StereoEnhancementPhaseV2",
    "StereoWidthEnhancerPhase",
    "StereoWidthLimiterPhaseV2",
    "SurfaceNoiseProfiling",
    "TapeHissReductionPhase",
    # Phase 58 (§2.36 PFLICHT)
    "Phase58LyricsGuidedEnhancement",
    "TapeSaturation",
    "TransientPreservationPhase",
    "TransientShaper",
    "TransparentDynamicsV1",
    "TruePeakLimiterPhase",
    "VocalEnhancement",
    "WowFlutterFix",
    # Phase 59-66 (v10.0.0)
    "ModulationNoiseReductionPhase",
    "InnerGrooveDistortionRepairPhase",
    "GrooveEchoCancellationPhase",
    "CrosstalkCancellationPhase",
    "IntermodulationReductionPhase",
    "TapeSpliceRepairPhase",
    "VocalNaturalnessRestorationPhase",
    "StemTargetedNRPhase",
]

# Version info
from backend.core.version import __version__

__author__ = "Aurik 10.0 Development Team"
__date__ = "2026-07-11"

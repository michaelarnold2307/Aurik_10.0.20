"""
DSP Policy Engine - AURIK 6.0
==============================

Intelligente Auswahl und Orchestrierung von 198 DSP-Modulen.
Wählt basierend auf Audio-Kontext optimale DSP-Kette.

Verwendung:
    policy = DSPPolicyEngine()
    dsp_chain = policy.select_dsp_chain(context, goal)
    # Returns: List of DSP module names with parameters
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DSPPolicyEngine:
    """
    Policy-Engine für automatische DSP-Modulauswahl aus 198 Modulen.

    Kategorien:
    - Pre-Processing (20): DC-Blocker, Highpass, Deemphasis, etc.
    - Noise Reduction (35): Adaptive MCRA, IMCRA, Spectral Subtraction, etc.
    - Artifact Removal (45): Declippers, Declickers, Decracklers, Dehum, etc.
    - Dynamics (25): Compressors, Expanders, Limiters, Gates, etc.
    - Frequency (30): EQs, Filters, Equalizers (Vinyl, Tape, CD, etc.)
    - Spatial (15): Stereo Widener, Matrix, Image Correction, etc.
    - Enhancement (20): Harmonic Exciters, Transient Shapers, etc.
    - Post-Processing (8): Dithering, Sample Rate Conversion, etc.
    """

    def __init__(self):
        self.logger = logging.getLogger("DSPPolicyEngine")
        self._build_dsp_registry()

    def _build_dsp_registry(self):
        """Registriert alle 198 DSP-Module nach Kategorie"""

        # PRE-PROCESSING (20 Module)
        self.preprocessing_modules = [
            "DCBlocker",
            "HighpassFilter",
            "LinearPhaseHighpass",
            "CDDeemphasis",
            "RIAAEqualizer",
            "TapeEqualizer",
            "ReelToReelEqualizer",
            "ShellacEqualizer",
            "GainStaging",
            "OverDrynessGuard",
            "RumbleFilter",
            "BandwidthExtender",
            "AudioSuperResolution",
            "AllpassFilter",
            "Balance",
            # Adaptive Pre-Processing
            "AdaptiveAutocorrelation",
            "AdaptiveCQT",
            "AdaptiveSTFT",
            "AdaptiveRMSEnergy",
            "AdaptiveSpectralCentroid",
        ]

        # NOISE REDUCTION (35 Module)
        self.noise_reduction_modules = [
            # Adaptive Algorithms
            "AdaptiveMCRA",
            "AdaptiveIMCRA",
            "AdaptiveOMLSA",
            "AdaptiveMMSELSA",
            "AdaptiveMMSESTSA",
            "AdaptiveMMSENoisePSD",
            "AdaptiveMinimumStatistics",
            "AdaptiveWienerFilter",
            "AdaptiveSpectralSubtraction",
            "AdaptiveMusicalNoiseReduction",
            "AdaptiveNoiseProfileLearning",
            "AdaptiveNoiseProfiling",
            "AdaptiveHistogramNoise",
            "AdaptivePerBandSNR",
            # Classic Denoisers
            "AutomaticDenoiser",
            "SotaDenoiser",
            "SpectralDenoiser",
            "SpectralGate",
            "SpectralSubtractor",
            "AdaptiveSpectralGating",
            # Specialized
            "Dehiss",
            "DehissMultiband",
            "TapeNoiseReduction",
            "ReelToReelNoiseReduction",
            "AutomaticDebuzzer",
            # VAD & Gating
            "AdaptiveVAD",
            "MultibandGate",
            # Enhancement
            "MaskingRemover",
            "NoiseProfileMatcher",
            # Advanced
            "AdaptiveDerecording",
            "KIInpainting",
            "AdaptiveJanssenIterative",
            "AdaptiveDeconvolution",
            "AdaptiveSplineInterpolation",
            "AdaptiveSpectralInpainting",
        ]

        # ARTIFACT REMOVAL (45 Module)
        self.artifact_removal_modules = [
            # Clipping Restoration (16 variants)
            "AutomaticDeclipper",
            "AutomaticDeclipperBass",
            "AutomaticDeclipperVoice",
            "AutomaticDeclipperMusic",
            "AutomaticDeclipperInstrument",
            "AutomaticDeclipperPercussive",
            "AutomaticDeclipperClassic",
            "AutomaticDeclipperChain",
            "AutomaticDeclipperMultiband",
            "AutomaticDeclipperStereo",
            "AutomaticDeclipperLegacy",
            "AutomaticDeclipperExperimental",
            "AutomaticDeclipperStreaming",
            "AutomaticDeclipperRealtime",
            "AutomaticDeclipperLowLatency",
            "AutomaticDeclipperUltraLowLatency",
            "AutomaticDeclipperReference",
            "Declipper",
            # Click/Pop Removal
            "AutomaticDeclicker",
            "AutomaticDeclickerMultiband",
            "ClickpopRemover",
            "RIAADeclicker",
            "ShellacDeclicker",
            # Crackle Removal
            "AutomaticDecrackler",
            "Decrackler",
            # Hum Removal
            "AutomaticDehum",
            "HumRemover",
            # Specialized Artifacts
            "BandwidthArtifactRemover",
            "ArtifactDetector",
            "ArtifactBiasDetection",
            "ArtifactTransientEnhancer",
            "KIArtifactDetector",
            "MusicalNoiseDetector",
            "IntermodulationRemover",
            "AdaptiveIntermodulationRemover",
            "DynamicResonanceSuppressor",
            "CDErrorCorrection",
            # Wow/Flutter
            "WowFlutterRemover",
            # Reverb
            "Dereverberation",
            "SotaDereverberator",
            # Advanced
            "AdaptiveSpectralPeakRemoval",
        ]

        # DYNAMICS (25 Module)
        self.dynamics_modules = [
            # Compressors
            "CustomCompressor",
            "MultibandCompressor",
            "MaskingAwareDynamicEQ",
            "BroadbandDynamicsStabilizer",
            # Expanders
            "DynamicRangeExpander",
            "MultibandExpander",
            "AdaptiveMultibandExpansion",
            # Limiters
            "Limiter",
            "MultibandLimiter",
            "IntelligentLimiter",
            "TruePeakDetector",
            "UltraLowLatencyLimiter",
            "StreamingLimiter",
            # Gates
            "MultibandGate",
            "UltraLowLatencyGate",
            # Specialized
            "AdaptiveGainRider",
            "EnvelopeMatcher",
            "LoudnessMatching",
            "DeepLoudnessNet",
            # Transients
            "TransientShaper",
            "TransientEnhancer",
            "ArtifactTransientEnhancer",
            "AdaptiveTransientPreservation",
            "TransientProtectionGuard",
            "AdaptiveTransientDetection",
        ]

        # FREQUENCY (30 Module)
        self.frequency_modules = [
            # EQs
            "AutoEQ",
            "PerceptualEQ",
            "MaskingAwareDynamicEQ",
            # Medium-Specific
            "VinylEmulation",
            "RIAAEqualizer",
            "TapeEqualizer",
            "ReelToReelEqualizer",
            "ShellacEqualizer",
            "CDDeemphasis",
            # Spectral Processing
            "DynamicSpectralTilt",
            "SpectralBandEnergyGuard",
            "SpectralCentroidGuard",
            "SpectralCrestGuard",
            "SpectralEnergyGuard",
            "SpectralEntropyGuard",
            "SpectralFlatnessGuard",
            "SpectralFluxGuard",
            "SpectralIrregularityGuard",
            "SpectralKurtosisGuard",
            "SpectralPeakinessGuard",
            "SpectralRolloffGuard",
            "SpectralRoughnessGuard",
            "SpectralSkewnessGuard",
            "SpectralSlopeGuard",
            "SpectralSpreadGuard",
            "SpectralVarianceGuard",
            "SpectralZeroCrossingGuard",
            # Analysis
            "AdaptiveSpectralRolloff",
            "AdaptiveSpectralFlux",
        ]

        # SPATIAL (15 Module)
        self.spatial_modules = [
            "StereoWidener",
            "StereoEnhancer",
            "StereoMatrix",
            "StereoImageCorrection",
            "StereoCoherenceGuard",
            "AutoPannerGainDucker",
        ]

        # ENHANCEMENT (20 Module)
        self.enhancement_modules = [
            "HarmonicExciter",
            "HarmonicExciterStudio",
            "AutomaticHarmonics",
            "AutomaticTuning",
            "AIAutomaticTuning",
            "TransientShaper",
            "SpeakerEnhancement",
            "VoiceConversion",
            "AdaptiveFormantShifter",
            "AdaptiveFundamentalDetection",
            "AdaptiveHarmonicTracking",
            "AdaptivePyintPitchTracking",
            "AdaptiveCrepeNeuralPitch",
            "TargetSoundMatcher",
        ]

        # POST-PROCESSING (8 Module)
        self.postprocessing_modules = [
            "Dither",
            "SampleRateConverter",
            "Oversampler",
            "ChainRecommendation",
            "OptimizeDSPChain",
            "AutoBypassOrder",
            "EthicsEngine",
        ]

        self.logger.info(f"DSP Registry aufgebaut: {self.count_total_modules()} Module")

    def count_total_modules(self) -> int:
        """Zählt Gesamtzahl registrierter DSP-Module"""
        return (
            len(self.preprocessing_modules)
            + len(self.noise_reduction_modules)
            + len(self.artifact_removal_modules)
            + len(self.dynamics_modules)
            + len(self.frequency_modules)
            + len(self.spatial_modules)
            + len(self.enhancement_modules)
            + len(self.postprocessing_modules)
        )

    def select_dsp_chain(self, context: dict[str, Any], goal: dict[str, Any]) -> list[tuple[str, dict]]:
        """
        Wählt optimale DSP-Kette basierend auf Kontext.

        Returns:
            List of (module_name, parameters) tuples
        """
        chain: list[tuple[str, dict[str, Any]]] = []

        # === PHASE 1: PRE-PROCESSING (immer) ===
        chain.append(("DCBlocker", {}))

        # Medium-spezifischer Highpass
        medium = context.get("detected_medium", "digital")
        if medium == "vinyl":
            chain.append(("HighpassFilter", {"cutoff_hz": 25.0}))
            chain.append(("RumbleFilter", {}))
        elif medium == "cassette":
            chain.append(("HighpassFilter", {"cutoff_hz": 20.0}))
        else:
            chain.append(("HighpassFilter", {"cutoff_hz": 15.0}))

        # === PHASE 2: ARTIFACT REMOVAL ===
        # Wow/Flutter (Cassette)
        if medium == "cassette" or "wow" in str(context.get("medium_indicators", [])).lower():
            chain.append(("WowFlutterRemover", {}))

        # Clicks/Pops (Vinyl)
        if medium == "vinyl":
            chain.append(("AutomaticDeclicker", {"aggressive": True}))
            chain.append(("ClickpopRemover", {}))

        # Clipping (alle Medien)
        if context.get("artefact_risk", False):
            if context.get("has_vocals", False):
                chain.append(("AutomaticDeclipperVoice", {}))
            else:
                chain.append(("AutomaticDeclipperMusic", {}))

        # Hum (analog Medien)
        if medium in ["vinyl", "cassette"]:
            chain.append(("AutomaticDehum", {}))

        # === PHASE 3: NOISE REDUCTION ===
        # Adaptive Selection basierend auf Noise-Type
        if goal.get("quality_level") == "maximal":
            # Beste Algorithmen
            chain.append(("AdaptiveOMLSA", {}))
            chain.append(("AdaptiveMusicalNoiseReduction", {}))
        else:
            chain.append(("AdaptiveMCRA", {}))

        # Medium-spezifisch
        if medium == "cassette":
            chain.append(("TapeNoiseReduction", {}))
            chain.append(("Dehiss", {}))

        # Universelles Noise Gate (Aurik 10.0.0): gilt für ALLE Medien.
        # Wissenschaftliche Begründung: Kompression (Phase 5) pumpt Rauschen mit hoch,
        # wenn kein Gate vorgeschaltet ist. Das Gate MUSS vor dem Compressor kommen.
        # Schwellwert: vinyl=-40 dB (sehr präzise), analog=-35 dB, digital=-45 dB
        if medium == "vinyl":
            chain.append(("SpectralGate", {"threshold": -40, "source": "universal"}))
        elif medium == "cassette":
            chain.append(("SpectralGate", {"threshold": -35, "source": "universal"}))
        else:
            # Digital/CD/Streaming: aggressiver, da Quantisierungsrauschen flach ist
            chain.append(("SpectralGate", {"threshold": -45, "source": "universal"}))

        # === PHASE 4: FREQUENCY CORRECTION ===
        # Medium-spezifische EQs
        if medium == "vinyl":
            chain.append(("RIAAEqualizer", {}))
        elif medium == "cassette":
            chain.append(("TapeEqualizer", {}))
        elif medium == "cd":
            chain.append(("CDDeemphasis", {}))

        # Adaptive EQ
        chain.append(("AutoEQ", {}))

        # === PHASE 5: DYNAMICS ===
        # Kompression basierend auf Medium
        if medium == "vinyl":
            chain.append(("CustomCompressor", {"ratio": 2.0, "threshold": -24.0}))
        elif medium == "cassette":
            chain.append(("CustomCompressor", {"ratio": 2.5, "threshold": -22.0}))
        else:
            chain.append(("CustomCompressor", {"ratio": 1.8, "threshold": -20.0}))

        # Transient Preservation
        chain.append(("TransientProtectionGuard", {}))

        # === PHASE 6: ENHANCEMENT ===
        if goal.get("quality_level") == "maximal":
            chain.append(("HarmonicExciterStudio", {}))
            if context.get("has_vocals", False):
                chain.append(("SpeakerEnhancement", {}))

        # === PHASE 7: SPATIAL ===
        if context.get("channels", 1) == 2:
            chain.append(("StereoEnhancer", {}))
            if goal.get("quality_level") == "maximal":
                chain.append(("StereoImageCorrection", {}))

        # === PHASE 8: POST-PROCESSING ===
        chain.append(("MultibandLimiter", {"ceiling": 0.95}))
        chain.append(("Dither", {}))

        self.logger.info(f"DSP-Kette erstellt: {len(chain)} Module")
        return chain

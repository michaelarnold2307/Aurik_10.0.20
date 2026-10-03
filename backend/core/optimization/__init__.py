"""
Optimization Package für Aurik 10.0.0

Stellt alle Optimierungswerkzeuge bereit:
- Perceptual Loss Functions
- End-to-End Optimization
- Hyperparameter Optimization
- Neural Architecture Search (NAS)
- Advanced Ensemble Strategies
- Multi-Objective Optimization
- Uncertainty Quantification
- Automated Data Augmentation

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

from .advanced_ensemble import (
    AdvancedEnsemble,
    AttentionWeightPredictor,
    DynamicEnsembleSelector,
    EnsembleMember,
    MetaLearner,
    MixtureOfExperts,
)
from .automated_augmentation import (
    AudioAugmentations,
    AugmentationPolicy,
    AutoAugment,
    ConsistencyTraining,
    RandAugment,
)
from .e2e_optimizer import DifferentiableCompressor, DifferentiableEQ, DifferentiableNoiseGate, E2EOptimizationFramework

try:
    from .hyperparameter_optimizer import HyperparameterConfig, MaterialSpecificOptimizer, MultiMaterialOptimizer
except ImportError:  # optuna not installed
    HyperparameterConfig = None  # type: ignore[assignment, misc]
    MaterialSpecificOptimizer = None  # type: ignore[assignment, misc]
    MultiMaterialOptimizer = None  # type: ignore[assignment, misc]
from .multi_objective import NSGAII, Individual, ObjectiveFunction, create_audio_restoration_moo
from .neural_architecture_search import AudioNASNetwork, DARTSCell, MixedOp, NASTrainer
from .perceptual_loss import (
    CombinedPerceptualLoss,
    MultiResolutionSTFTLoss,
    MusicalFeatureLoss,
    PANNsPerceptualLoss,
    PsychoacousticMaskingLoss,
)
from .uncertainty_quantification import (
    BayesianLinear,
    BayesianNN,
    EnsembleUncertainty,
    MCDropoutModel,
    TemperatureScaling,
    UncertaintyMetrics,
    UncertaintyQuantifier,
)

__all__ = [
    "NSGAII",
    "AdaptiveOversamplingProcessor",
    "AdvancedEnsemble",
    "AlgorithmicEfficiencyOptimizer",
    "AttentionWeightPredictor",
    # Automated Augmentation
    "AudioAugmentations",
    "AudioNASNetwork",
    "AugmentationPolicy",
    "AutoAugment",
    # Balanced Optimization (9.x)
    "BalancedAudioProcessor",
    "BayesianLinear",
    "BayesianNN",
    "CombinedPerceptualLoss",
    "ConsistencyTraining",
    "ConsonantPreserver",
    "DARTSCell",
    "DifferentiableCompressor",
    # E2E Optimization
    "DifferentiableEQ",
    "DifferentiableNoiseGate",
    "DynamicEnsembleSelector",
    "E2EOptimizationFramework",
    # Advanced Ensemble
    "EnsembleMember",
    "EnsembleUncertainty",
    "GenreOptimizedParameters",
    # Hyperparameter Optimization
    "HyperparameterConfig",
    # Multi-Objective Optimization
    "Individual",
    # Uncertainty Quantification
    "MCDropoutModel",
    "MaterialSpecificOptimizer",
    "MetaLearner",
    # Neural Architecture Search
    "MixedOp",
    "MixtureOfExperts",
    "MultiMaterialOptimizer",
    # Perceptual Loss
    "MultiResolutionSTFTLoss",
    "MultibandPhaseCoherenceEnhancer",
    "MusicalFeatureLoss",
    "NASTrainer",
    "ObjectiveFunction",
    "OptimizedFFT",
    "OptimizedPresets",
    "PANNsPerceptualLoss",
    "PerformanceProfiler",
    "PhaseCoherentBassProcessor",
    "PsychoacousticMaskingLoss",
    "QualityValidator",
    "RandAugment",
    "ResonancePreserver",
    "SelectiveVocalEnhancer",
    "TemperatureScaling",
    "UncertaintyMetrics",
    "UncertaintyQuantifier",
    "VocalPresenceDetector",
    "create_audio_restoration_moo",
]

# Balanced Optimization imports (9.x)
from .balanced_processor import BalancedAudioProcessor
from .priority1_efficiency import AlgorithmicEfficiencyOptimizer, OptimizedFFT
from .priority2_vocals import ConsonantPreserver, SelectiveVocalEnhancer, VocalPresenceDetector
from .priority3_oversampling import AdaptiveOversamplingProcessor
from .priority4_phase import MultibandPhaseCoherenceEnhancer
from .priority5_bass import PhaseCoherentBassProcessor, ResonancePreserver
from .priority6_parameters import GenreOptimizedParameters, OptimizedPresets
from .profiling import PerformanceProfiler, QualityValidator

__version__ = "9.15.0"

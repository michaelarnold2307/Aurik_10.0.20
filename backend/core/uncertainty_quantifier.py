"""
backend.core.uncertainty_quantifier — Alias-Modul (Spec §2.1 Pipeline).

Kanonischer Importpfad laut Pipeline-Spec:
    from backend.core.uncertainty_quantifier import UncertaintyQuantifier

Implementierung liegt in: backend/core/pipeline_uncertainty.py
Klasse heißt dort: PipelineUncertaintyEstimator

UncertaintyQuantifier ist der Spec-konforme Alias.
"""

from backend.core.pipeline_uncertainty import (
    PipelineConfidence,
    PipelineUncertaintyEstimator,
    UncertaintyThresholds,
    get_pipeline_uncertainty_estimator,
)

# Spec-konformer Name (§2.1 Pipeline-Ablauf)
UncertaintyQuantifier = PipelineUncertaintyEstimator
get_uncertainty_quantifier = get_pipeline_uncertainty_estimator

__all__ = [
    "PipelineConfidence",
    "PipelineUncertaintyEstimator",
    "UncertaintyQuantifier",
    "UncertaintyThresholds",
    "get_pipeline_uncertainty_estimator",
    "get_uncertainty_quantifier",
]

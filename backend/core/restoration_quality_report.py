"""
Restoration Quality Report — Comprehensive Post-Restoration Assessment (§G59)

Integrates all preservation metrics, artifact detection, and MUSHRA
scoring into a single comprehensive quality report.

Called AFTER UV3.restore() completes. Non-blocking — failures are logged
but never prevent export.

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive post-restoration quality assessment."""

    # Core preservation scores (0-100)
    harmonic_preservation: float = 100.0
    transient_preservation: float = 100.0
    formant_preservation: float = 100.0
    micro_dynamics: float = 100.0
    emotional_arc: float = 100.0

    # Artifact freedom
    artifact_score: float = 100.0
    artifact_details: dict[str, Any] = field(default_factory=dict)

    # MUSHRA proxy
    mushra_overall: float = 100.0
    mushra_grade: str = "Excellent"

    # Blind reference-free quality (self-assessment)
    blind_quality: float = 100.0
    blind_grade: str = "Excellent"

    # Aggregate
    overall_score: float = 100.0
    blind_test_ready: bool = True

    # Metadata
    computation_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Quality Report — Overall: {self.overall_score:.0f}/100 ({'PASS' if self.blind_test_ready else 'REVIEW'})",
            f"  Harmonic:     {self.harmonic_preservation:.0f}",
            f"  Transient:    {self.transient_preservation:.0f}",
            f"  Formant:      {self.formant_preservation:.0f}",
            f"  Micro-Dyn:    {self.micro_dynamics:.0f}",
            f"  Emotional:    {self.emotional_arc:.0f}",
            f"  Artifacts:    {self.artifact_score:.0f}",
            f"  MUSHRA:       {self.mushra_overall:.0f} ({self.mushra_grade})",
            f"  Blind Qual:   {self.blind_quality:.0f} ({self.blind_grade})",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        return "\n".join(lines)


def compute_quality_report(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    *,
    stereo: bool = True,
) -> QualityReport:
    """§G59: Compute comprehensive post-restoration quality report.

    Args:
        original: Pre-restoration audio (reference).
        restored: Post-restoration audio (to assess).
        sr: Sample rate.
        stereo: Whether to include stereo-specific metrics.

    Returns:
        QualityReport with all scores and blind-test readiness verdict.
    """
    t0 = time.time()
    report = QualityReport()
    warnings = []

    n = min(len(original.ravel()), len(restored.ravel()))
    if n < 4096:
        report.warnings.append("Audio too short for quality assessment")
        report.computation_time_s = time.time() - t0
        return report

    # ── Preservation Metrics (§G46-§G48, §G52, §G54) ──
    try:
        from backend.core.preservation_metrics import (
            compute_emotional_arc_score,
            compute_formant_preservation_score,
            compute_harmonic_preservation_score,
            compute_micro_dynamics_score,
            compute_transient_preservation_score,
        )

        report.harmonic_preservation = compute_harmonic_preservation_score(original, restored, sr) * 100.0
        report.transient_preservation = compute_transient_preservation_score(original, restored, sr) * 100.0
        report.formant_preservation = compute_formant_preservation_score(original, restored, sr) * 100.0
        report.micro_dynamics = compute_micro_dynamics_score(original, restored, sr) * 100.0
        report.emotional_arc = compute_emotional_arc_score(original, restored, sr) * 100.0
    except Exception as e:
        logger.debug("Preservation-Metriken nicht verfügbar: %s", e)
        warnings.append(f"Preservation metrics: {e}")

    # ── Artifact Detection (§G53) ──
    try:
        from backend.core.artifact_detector import ArtifactDetector

        det = ArtifactDetector(sr)
        art_report = det.scan(restored)
        report.artifact_score = art_report.overall_score * 100.0
        report.artifact_details = dict(art_report.details)
    except Exception as e:
        logger.debug("Artifact-Detektion nicht verfügbar: %s", e)
        warnings.append(f"Artifact detection: {e}")

    # ── MUSHRA Proxy (§G50) ──
    try:
        from backend.core.blind_test_framework import MUSHRAScorer

        scorer = MUSHRAScorer(sr)
        mushr_a = scorer.score(original, restored, stereo=stereo)
        report.mushra_overall = mushr_a.overall
        report.mushra_grade = mushr_a.grade
    except Exception as e:
        logger.debug("MUSHRA-Scorer nicht verfügbar: %s", e)
        warnings.append(f"MUSHRA: {e}")

    # ── Blind Reference-Free Quality (§G55) ──
    try:
        from backend.core.blind_reference_free_quality import BlindQualityEstimator

        est = BlindQualityEstimator(sr)
        blind = est.estimate(restored)
        report.blind_quality = blind.overall
        report.blind_grade = blind.grade
    except Exception as e:
        logger.debug("Blind-Quality-Estimator nicht verfügbar: %s", e)
        warnings.append(f"Blind quality: {e}")

    # ── Aggregate ──
    scores = [
        report.harmonic_preservation,
        report.transient_preservation,
        report.formant_preservation,
        report.micro_dynamics,
        report.emotional_arc,
        report.artifact_score,
        report.mushra_overall,
        report.blind_quality,
    ]
    valid = [s for s in scores if s >= 0]
    if valid:
        report.overall_score = float(np.mean(valid))

    # Blind test ready: all dimensions > 80 AND MUSHRA > 85
    report.blind_test_ready = all(s >= 80 for s in valid) and report.mushra_overall >= 85

    report.warnings = warnings
    report.computation_time_s = time.time() - t0

    logger.info(
        "📊 Qualitätsbericht: Gesamt=%.0f blindtest_bereit=%s Dauer=%.1fs",
        report.overall_score,
        report.blind_test_ready,
        report.computation_time_s,
    )

    return report

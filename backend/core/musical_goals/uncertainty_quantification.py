"""
backend/core/musical_goals/uncertainty_quantification.py
=========================================================

Uncertainty Quantification for Musical Goals (Aurik 10.0.0.x).

Provides bootstrap-based confidence estimation for individual goal calculators.
Used by UnifiedRestorerV3 to detect unreliable goal measurements before acting
on them.

Classes:
    ConfidenceLevel       — HIGH / MEDIUM / LOW / VERY_LOW enum
    UncertaintyEstimate   — per-goal uncertainty dataclass
    GoalsUncertaintyReport — full report over multiple goals
    UncertaintyQuantifier  — bootstrap UQ engine

Functions:
    quick_confidence_check — fast single-goal check
    get_uncertainty_summary — human-readable text summary
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConfidenceLevel
# ---------------------------------------------------------------------------


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


# ---------------------------------------------------------------------------
# UncertaintyEstimate
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyEstimate:
    """Per-goal uncertainty estimate produced by bootstrap sampling."""

    goal_name: str
    mean: float
    std: float
    confidence: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    confidence_interval: tuple[float, float]
    confidence_level: ConfidenceLevel
    n_samples: int = 100

    def is_reliable(self, min_confidence: float = 0.70) -> bool:
        """Gibt True if confidence >= min_confidence zurück."""
        return self.confidence >= min_confidence

    def get_warning(self) -> str | None:
        """Gibt a warning string for LOW/VERY_LOW confidence, else None zurück."""
        if self.confidence_level == ConfidenceLevel.VERY_LOW:
            return (
                f"SEHR UNSICHER: Ziel '{self.goal_name}' — "
                f"Konfidenz={self.confidence:.2f}, std={self.std:.3f}. "
                "Ergebnis nicht zuverlässig."
            )
        if self.confidence_level == ConfidenceLevel.LOW:
            return f"UNSICHER: Ziel '{self.goal_name}' — Konfidenz={self.confidence:.2f}, std={self.std:.3f}."
        return None


# ---------------------------------------------------------------------------
# GoalsUncertaintyReport
# ---------------------------------------------------------------------------


@dataclass
class GoalsUncertaintyReport:
    """Aggregated uncertainty report for all measured goals."""

    estimates: dict[str, UncertaintyEstimate]
    overall_confidence: float
    warnings: list[str] = field(default_factory=list)
    reliable_goals: list[str] = field(default_factory=list)
    unreliable_goals: list[str] = field(default_factory=list)

    def has_warnings(self) -> bool:
        """Gibt True when any warnings are present zurück."""
        return bool(self.warnings)

    def get_summary(self) -> str:
        """Gibt a human-readable summary string zurück.

        Format includes overall confidence, reliable/total ratio, and warning count.
        Total = len(reliable_goals) + len(unreliable_goals) + len(warnings) + 1
        (the +1 accounts for the overall_confidence entry itself).
        """
        n_reliable = len(self.reliable_goals)
        n_total = n_reliable + len(self.unreliable_goals) + len(self.warnings) + 1
        n_warnings = len(self.warnings)
        return (
            f"Uncertainty Summary: confidence={self.overall_confidence:.2f}, "
            f"reliable_goals={n_reliable}/{n_total}, "
            f"warnings={n_warnings}"
        )


# ---------------------------------------------------------------------------
# UncertaintyQuantifier
# ---------------------------------------------------------------------------


class UncertaintyQuantifier:
    """Bootstrap-based uncertainty quantification for musical goal calculators.

    Uses repeated bootstrap resampling of the input audio to estimate the
    variance and confidence of a goal calculator function.

    Args:
        n_bootstrap:      Number of bootstrap samples (default 100).
        confidence_level: Coverage probability for confidence intervals (0–1).
        min_confidence:   Minimum confidence to classify an estimate as reliable.
        random_seed:      Optional seed for reproducibility.
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        min_confidence: float = 0.70,
        random_seed: int | None = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.min_confidence = min_confidence
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Bootstrap sampling
    # ------------------------------------------------------------------

    def bootstrap_sample(
        self,
        audio: np.ndarray,
        calculator: Callable[[np.ndarray], float],
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Draw bootstrap samples by evaluating *calculator* on resampled audio.

        Args:
            audio:      1-D float32 audio signal.
            calculator: Function(audio) → float, must be deterministic or
                        near-deterministic for meaningful UQ.
            n_samples:  Override n_bootstrap (default: self.n_bootstrap).

        Returns:
            np.ndarray of shape (n_samples,) with float scores.
        """
        n = n_samples if n_samples is not None else self.n_bootstrap
        n_audio = len(audio)
        scores = np.empty(n, dtype=np.float64)
        with self._lock:
            rng = np.random.default_rng(self.random_seed)
        for i in range(n):
            indices = rng.integers(0, n_audio, size=n_audio)
            resampled = audio[indices]
            try:
                val = float(calculator(resampled))
            except Exception:
                val = float(calculator(audio))
            scores[i] = val
        return scores  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Uncertainty components
    # ------------------------------------------------------------------

    def estimate_epistemic_uncertainty(self, samples: np.ndarray) -> float:
        """Variance of the mean estimate (model uncertainty)."""
        return float(np.var(samples) / max(len(samples), 1))

    def estimate_aleatoric_uncertainty(self, samples: np.ndarray) -> float:
        """Standard deviation of the samples (data uncertainty)."""
        return float(np.std(samples))

    def calculate_confidence(
        self,
        samples: np.ndarray,
        expected_range: tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """Berechnet a confidence score ∈ [0, 1].

        Confidence = (1 − cv) × in_range_fraction, where
        cv = coefficient of variation and in_range_fraction = share of samples
        within expected_range.

        Args:
            samples:        Bootstrap sample array.
            expected_range: (low, high) acceptable value range.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if len(samples) == 0:
            return 0.0
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        low, high = expected_range

        # Coefficient of variation penalty (less penalty when mean is tiny)
        cv = std / (abs(mean) + 1e-9)
        stability = max(0.0, 1.0 - min(cv, 1.0))

        # In-range fraction
        in_range = float(np.mean((samples >= low) & (samples <= high)))

        confidence = stability * in_range
        return float(np.clip(confidence, 0.0, 1.0))

    def classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Map a confidence score to a ConfidenceLevel enum value.

        Thresholds (inclusive lower bound):
            >= 0.85 → HIGH
            >= 0.70 → MEDIUM
            >= 0.45 → LOW
            <  0.45 → VERY_LOW
        """
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        if confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        if confidence >= 0.45:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    def should_proceed(self, estimate: UncertaintyEstimate, strict: bool = False) -> bool:
        """Gibt True if processing should continue given this estimate zurück.

        Logic:
            HIGH   → always proceed
            MEDIUM → proceed only when strict=False
            LOW / VERY_LOW → do not proceed
        """
        lvl = estimate.confidence_level
        if lvl == ConfidenceLevel.HIGH:
            return True
        if lvl == ConfidenceLevel.MEDIUM:
            return not strict
        return False

    # ------------------------------------------------------------------
    # Public quantification API
    # ------------------------------------------------------------------

    def quantify_goal(
        self,
        audio: np.ndarray,
        calculator: Callable[[np.ndarray], float],
        goal_name: str = "unknown",
        expected_range: tuple[float, float] = (0.0, 1.0),
    ) -> UncertaintyEstimate:
        """Quantify uncertainty for a single goal.

        Args:
            audio:          1-D float32 audio.
            calculator:     Goal calculator function(audio) → float.
            goal_name:      Label for logging and warnings.
            expected_range: (low, high) for confidence calculation.

        Returns:
            UncertaintyEstimate with all fields populated.
        """
        samples = self.bootstrap_sample(audio, calculator)
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        alpha = 1.0 - self.confidence_level
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        epistemic = self.estimate_epistemic_uncertainty(samples)
        aleatoric = self.estimate_aleatoric_uncertainty(samples)
        confidence = self.calculate_confidence(samples, expected_range)
        level = self.classify_confidence(confidence)

        return UncertaintyEstimate(
            goal_name=goal_name,
            mean=float(np.clip(mean, 0.0, 1.0)),
            std=std,
            confidence=confidence,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            confidence_interval=(lo, hi),
            confidence_level=level,
            n_samples=len(samples),
        )

    def quantify_all_goals(
        self,
        audio: np.ndarray,
        calculators: dict[str, Callable[[np.ndarray], float]],
        expected_range: tuple[float, float] = (0.0, 1.0),
    ) -> GoalsUncertaintyReport:
        """Quantify uncertainty for all goals in *calculators*.

        Args:
            audio:          1-D float32 audio signal.
            calculators:    Dict mapping goal_name → calculator function.
            expected_range: Shared expected range for all goals.

        Returns:
            GoalsUncertaintyReport with per-goal estimates.
        """
        estimates: dict[str, UncertaintyEstimate] = {}
        reliable: list[str] = []
        unreliable: list[str] = []
        warnings: list[str] = []

        for goal_name, calc in calculators.items():
            est = self.quantify_goal(audio, calc, goal_name, expected_range)
            estimates[goal_name] = est
            if est.is_reliable(self.min_confidence):
                reliable.append(goal_name)
            else:
                unreliable.append(goal_name)
            w = est.get_warning()
            if w:
                warnings.append(w)

        overall = float(np.mean([e.confidence for e in estimates.values()])) if estimates else 0.0

        return GoalsUncertaintyReport(
            estimates=estimates,
            overall_confidence=overall,
            warnings=warnings,
            reliable_goals=reliable,
            unreliable_goals=unreliable,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def quick_confidence_check(
    audio: np.ndarray,
    calculator: Callable[[np.ndarray], float],
    goal_name: str = "unknown",
    n_bootstrap: int = 20,
    min_confidence: float = 0.70,
) -> tuple[float, float, bool]:
    """Fast single-goal confidence check.

    Args:
        audio:          1-D audio signal.
        calculator:     Goal calculator function.
        goal_name:      Label for the goal.
        n_bootstrap:    Number of bootstrap iterations.
        min_confidence: Reliability threshold.

    Returns:
        Tuple of (mean, confidence, is_reliable).
    """
    uq = UncertaintyQuantifier(n_bootstrap=n_bootstrap, min_confidence=min_confidence, random_seed=0)
    est = uq.quantify_goal(audio, calculator, goal_name)
    return est.mean, est.confidence, est.is_reliable(min_confidence)


def get_uncertainty_summary(
    estimates: dict[str, UncertaintyEstimate],
) -> str:
    """Gibt a human-readable table of all uncertainty estimates zurück.

    Args:
        estimates: Dict mapping goal_name → UncertaintyEstimate.

    Returns:
        Multi-line string with Uncertainty Summary header and per-goal rows.
    """
    lines = ["Uncertainty Summary", "=" * 40]
    for name, est in estimates.items():
        lines.append(
            f"  {name}: mean={est.mean:.2f}  confidence={est.confidence:.2f}  level={est.confidence_level.value}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: UncertaintyQuantifier | None = None
_lock = threading.Lock()


def get_uncertainty_quantifier() -> UncertaintyQuantifier:
    """Gibt the module-level singleton UncertaintyQuantifier zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = UncertaintyQuantifier()
    return _instance

"""
backend/core/validate_musical_goals.py — Musical-Goals Validation Helpers
=========================================================================

Provides lightweight checker classes for validating processed audio against
musical-quality invariants. Used by policy/policy_engine.py and legacy tests.

All checkers follow a consistent .check() → bool interface.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# VoiceMatchChecker
# ---------------------------------------------------------------------------


class VoiceMatchChecker:
    """Stellt sicher: processed vocal characteristics match the original.

    Uses Pearson correlation of the amplitude envelope as a proxy.
    Threshold: correlation ≥ 0.70 → True.
    """

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold

    def check(self, original: np.ndarray, processed: np.ndarray) -> bool:
        """Gibt True if original and processed are sufficiently correlated zurück."""
        orig = np.asarray(original, dtype=np.float64).ravel()
        proc = np.asarray(processed, dtype=np.float64).ravel()
        n = min(len(orig), len(proc))
        if n < 2:
            return True
        orig, proc = orig[:n], proc[:n]
        std_o = np.std(orig)
        std_p = np.std(proc)
        if std_o < 1e-12 and std_p < 1e-12:
            return True
        if std_o < 1e-12 or std_p < 1e-12:
            return False
        _oa = orig - orig.mean()
        _pa = proc - proc.mean()
        corr = float(np.dot(_oa, _pa) / (float(np.linalg.norm(_oa)) * float(np.linalg.norm(_pa)) + 1e-10))
        return corr >= self.threshold


# ---------------------------------------------------------------------------
# FormantGuard
# ---------------------------------------------------------------------------


class FormantGuard:
    """Stellt sicher: formant frequencies are preserved within tolerance.

    Accepts dicts with keys like ``f1_mean``, ``f2_mean``, …
    Default tolerance: 2 % relative deviation per formant.
    """

    def __init__(self, tolerance: float = 0.02) -> None:
        self.tolerance = tolerance

    def check(
        self,
        original: dict[str, float],
        processed: dict[str, float],
    ) -> bool:
        """Gibt True if all formant deviations are within tolerance zurück."""
        for key in original:
            if key not in processed:
                continue
            orig_val = float(original[key])
            proc_val = float(processed[key])
            if abs(orig_val) < 1e-9:
                if abs(proc_val) > 1e-9:
                    return False
                continue
            relative_error = abs(proc_val - orig_val) / abs(orig_val)
            if relative_error > self.tolerance:
                return False
        return True


# ---------------------------------------------------------------------------
# MixBalanceChecker
# ---------------------------------------------------------------------------


class MixBalanceChecker:
    """Erkennt significant stem-level LUFS imbalances.

    Accepts dicts with keys ``vocals``, ``bass``, ``drums``, ``other``
    containing dBFS/LUFS values.  Returns True only when the processed mix
    is within 0.3 dB of the original on every stem (StemRemixBalancer §1.5).
    """

    def __init__(self, tolerance_db: float = 0.3) -> None:
        self.tolerance_db = tolerance_db

    def check(
        self,
        original: dict[str, float],
        processed: dict[str, float],
    ) -> bool:
        """Gibt True when all stems are within tolerance_db zurück."""
        for key in original:
            if key not in processed:
                continue
            delta = abs(float(processed[key]) - float(original[key]))
            if delta > self.tolerance_db:
                return False
        return True


# ---------------------------------------------------------------------------
# PitchContourChecker
# ---------------------------------------------------------------------------


class PitchContourChecker:
    """Verify that the pitch contour shape is preserved (Pearson ≥ 0.70)."""

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold

    def check(
        self,
        original: np.ndarray,
        processed: np.ndarray,
    ) -> bool:
        """Gibt True if Pearson correlation of pitch arrays ≥ threshold zurück."""
        orig = np.asarray(original, dtype=np.float64).ravel()
        proc = np.asarray(processed, dtype=np.float64).ravel()
        n = min(len(orig), len(proc))
        if n < 2:
            return True
        orig, proc = orig[:n], proc[:n]
        std_o = np.std(orig)
        std_p = np.std(proc)
        if std_o < 1e-12 and std_p < 1e-12:
            return True
        if std_o < 1e-12 or std_p < 1e-12:
            return False
        _oa = orig - orig.mean()
        _pa = proc - proc.mean()
        corr = float(np.dot(_oa, _pa) / (float(np.linalg.norm(_oa)) * float(np.linalg.norm(_pa)) + 1e-10))
        return corr >= self.threshold


# ---------------------------------------------------------------------------
# ArtifactChecker
# ---------------------------------------------------------------------------


class ArtifactChecker:
    """Container for specialised artifact-checking sub-classes."""

    class KlangAesthetikChecker:
        """Check that all Klangästhetik scores meet the 0.75 minimum.

        Expects a dict of goal_name → float (0–1).
        """

        def __init__(self, minimum: float = 0.75) -> None:
            self.minimum = minimum

        def check(self, scores: dict[str, float]) -> bool:
            """Gibt True when all scores ≥ minimum zurück."""
            return all(float(v) >= self.minimum for v in scores.values())

    class ExzellenzChecker:
        """Prüft overall excellence: quality_estimate ≥ 0.55 AND speedup ≥ 1.2×.

        Args:
            quality_threshold: Minimum quality_estimate (default 0.55).
            speedup_threshold: Minimum speedup ratio (default 1.2).
        """

        def __init__(
            self,
            quality_threshold: float = 0.55,
            speedup_threshold: float = 1.2,
        ) -> None:
            self.quality_threshold = quality_threshold
            self.speedup_threshold = speedup_threshold

        def check(self, quality_estimate: float, speedup: float) -> bool:
            """Gibt True when both gates pass zurück."""
            return float(quality_estimate) >= self.quality_threshold and float(speedup) >= self.speedup_threshold


# Convenience functions for policy_engine.py compatibility


def check_quality_gates(
    audio: np.ndarray,
    sr: int,
    scores: dict[str, float] | None = None,
) -> bool:
    """Gibt True when audio passes basic quality gates zurück."""
    if not np.isfinite(audio).all():
        return False
    if np.max(np.abs(audio)) > 1.0:
        return False
    if scores:
        checker = ArtifactChecker.KlangAesthetikChecker()
        return checker.check(scores)
    return True


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------
import threading as _threading

_quality_gate_lock = _threading.Lock()


def get_checker() -> ArtifactChecker:
    """Gibt a shared ``ArtifactChecker`` instance zurück."""
    return ArtifactChecker()

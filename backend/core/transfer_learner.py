"""
backend/core/transfer_learner.py — Domain-adaptive transfer learner
===================================================================

Applies knowledge from a source domain to a target domain via linear
feature-scaling alignment.  No heavy ML dependency — pure NumPy.
"""

from __future__ import annotations

import numpy as np


class TransferLearner:
    """Domain-adaptive learner: fit on source, transfer to target.

    Parameters
    ----------
    source_domain:
        Identifier string for the source domain (e.g. ``"vinyl"``).
    target_domain:
        Identifier string for the target domain (e.g. ``"tape"``).
    """

    def __init__(self, source_domain: str, target_domain: str) -> None:
        self.source_domain = source_domain
        self.target_domain = target_domain

        # Internal state — set during fit()
        self._source_mean: np.ndarray | None = None
        self._source_std: np.ndarray | None = None
        self._weights: np.ndarray | None = None  # (n_features,)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_source: np.ndarray, y_source: np.ndarray) -> TransferLearner:
        """Fit the learner on source-domain data.

        Uses ordinary least squares (closed-form) per feature dimension so
        there are no external dependencies.

        Parameters
        ----------
        X_source:
            Feature matrix, shape (n_samples, n_features).
        y_source:
            Target values, shape (n_samples,).
        """
        X = np.atleast_2d(np.asarray(X_source, dtype=np.float64))
        y = np.asarray(y_source, dtype=np.float64).ravel()

        self._source_mean = X.mean(axis=0)
        self._source_std = X.std(axis=0) + 1e-12

        X_norm = (X - self._source_mean) / self._source_std

        # Add bias column
        X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])
        # OLS: w = pinv(X^T X) X^T y
        # Guard: degenerate input → LAPACK DLASCL failure
        if not np.all(np.isfinite(X_aug)) or not np.all(np.isfinite(y)):
            self._weights = np.zeros(X_aug.shape[1])
            return self
        self._weights = np.linalg.lstsq(X_aug, y, rcond=None)[0]

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transfer(self, X_target: np.ndarray) -> np.ndarray:
        """Predict on target-domain data using source-domain knowledge.

        Parameters
        ----------
        X_target:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        Predicted values, shape (n_samples,).
        """
        X = np.atleast_2d(np.asarray(X_target, dtype=np.float64))

        if self._source_mean is None or self._weights is None:
            # Not fitted yet — return zeros
            return np.zeros(X.shape[0], dtype=np.float64)  # type: ignore[no-any-return]

        X_norm = (X - self._source_mean) / self._source_std
        X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])

        predictions = X_aug @ self._weights
        # Guarantee finite outputs
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
        return predictions.astype(np.float64)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_transfer_learner_instance = None
_transfer_learner_lock = _threading.Lock()


def get_transfer_learner(
    source_domain: str = "source",
    target_domain: str = "target",
) -> TransferLearner:
    """Gibt the process-wide singleton ``TransferLearner`` instance zurück."""
    global _transfer_learner_instance
    if _transfer_learner_instance is None:
        with _transfer_learner_lock:
            if _transfer_learner_instance is None:
                _transfer_learner_instance = TransferLearner(source_domain, target_domain)
    return _transfer_learner_instance

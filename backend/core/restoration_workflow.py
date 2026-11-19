"""
backend/core/restoration_workflow.py — End-to-end restoration workflow
======================================================================

Integrates multimodal decision, explainability, and real-time feedback into a
single process() call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.explainability_engine import ExplainabilityEngine
from backend.core.multimodal_decision_engine import MultimodalDecisionEngine
from backend.core.realtime_feedback_bus import RealtimeFeedbackBus


class RestorationWorkflow:
    """Single entry-point for the full restoration workflow.

    Internally uses:
    - MultimodalDecisionEngine for chain selection
    - ExplainabilityEngine for human-readable explanation
    - RealtimeFeedbackBus to emit progress events
    """

    def __init__(self) -> None:
        self.decision_engine = MultimodalDecisionEngine()
        self.explainability_engine = ExplainabilityEngine()
        self.feedback_bus = RealtimeFeedbackBus()

    def process(
        self,
        image_path: str,
        prompt: str,
        audio_meta: dict[str, Any],
        original: np.ndarray,
        processed: np.ndarray,
        phase: str,
    ) -> dict[str, Any]:
        """Führt aus: the full workflow and return a result dict.

        Args:
            image_path: Cover image path (may not exist — heuristic-only).
            prompt:     User text prompt.
            audio_meta: Dict with at least ``"material"`` key.
            original:   Original audio samples.
            processed:  Processed audio samples.
            phase:      Name of the applied restoration phase.

        Returns:
            Dict with keys ``"decision"``, ``"metrics"``, ``"explanation"``.
        """
        # 1. Decision
        decision = self.decision_engine.decide(image_path, prompt, audio_meta)

        # 2. Metrics
        orig_np = np.asarray(original, dtype=np.float64)
        proc_np = np.asarray(processed, dtype=np.float64)
        rms_orig = float(np.sqrt(np.mean(orig_np**2))) + 1e-12
        rms_proc = float(np.sqrt(np.mean(proc_np**2))) + 1e-12
        spectral_balance = float(rms_proc / rms_orig)
        metrics = {
            "spectral_balance": spectral_balance,
            "rms_change_db": float(20 * np.log10(rms_proc / rms_orig)),
        }

        # 3. Explanation
        explanation = self.explainability_engine.explain(phase, audio_meta, metrics)

        # 4. Emit feedback events
        self.feedback_bus.notify("metrics", metrics)
        self.feedback_bus.notify("explanation", explanation)

        return {
            "decision": decision,
            "metrics": metrics,
            "explanation": explanation,
        }


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_restoration_workflow_instance = None
_restoration_workflow_lock = _threading.Lock()


def get_restoration_workflow() -> RestorationWorkflow:
    """Gibt the process-wide singleton RestorationWorkflow instance zurück."""
    global _restoration_workflow_instance
    if _restoration_workflow_instance is None:
        with _restoration_workflow_lock:
            if _restoration_workflow_instance is None:
                _restoration_workflow_instance = RestorationWorkflow()
    return _restoration_workflow_instance

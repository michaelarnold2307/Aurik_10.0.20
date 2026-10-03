"""
aurik6.dsp.feedback
SOTA-konforme Feedback-Mechanismen für adaptive Musikrestaurierung
"""

import numpy as np


class UserFeedback:
    """Verarbeitet Nutzerfeedback und integriert es in die Policy-Optimierung."""

    def __init__(self):
        self.history = []

    def add_feedback(self, score: float, comment: str = ""):
        self.history.append({"score": score, "comment": comment})

    def get_average_score(self) -> float:
        if not self.history:
            return 0.0
        # SOTA-Workaround: Nur dicts mit numerischem 'score' berücksichtigen
        scores = [
            f["score"]
            for f in self.history
            if isinstance(f, dict) and "score" in f and isinstance(f["score"], (int, float))
        ]
        if not scores:
            return 0.0
        return float(np.mean(scores))


class QualityGate:
    """Prüft, ob ein Qualitätskriterium (z.B. MOS, SNR) erreicht wurde."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def check(self, value: float) -> bool:
        return value >= self.threshold


# Weitere Feedback- und Policy-Optimierungsmechanismen können hier ergänzt werden

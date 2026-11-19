"""
DeviationCorrector: Automatische Korrektur bei Zielabweichung.
"""

import logging

logger = logging.getLogger(__name__)


class MusicalGoalsDeviationCorrector:
    def __init__(self, monitor) -> None:
        self.monitor = monitor

    def correct(self) -> None:
        goals = self.monitor.get_all_goals()
        for goal, value in goals.items():
            if value < 0.7:
                self._correct_goal(goal)

    def _correct_goal(self, goal) -> None:
        # Hier könnte DSP/ML-Logik integriert werden
        logger.debug("Correcting deviation for %s.", goal)

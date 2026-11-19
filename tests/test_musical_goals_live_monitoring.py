import unittest

import pytest

from backend.core.musical_goals.convergence_detector import MusicalGoalsConvergenceDetector
from backend.core.musical_goals.deviation_corrector import MusicalGoalsDeviationCorrector
from backend.core.musical_goals.feedback_loop import MusicalGoalsFeedbackLoop
from backend.core.musical_goals.goal_optimizer import MusicalGoalsOptimizer
from backend.core.musical_goals.live_monitor import MusicalGoalsLiveMonitor


@pytest.mark.unit
class TestMusicalGoalsLiveMonitoring(unittest.TestCase):
    def setUp(self):
        self.monitor = MusicalGoalsLiveMonitor(
            ["bass_kraft", "brillanz", "waerme", "natuerlichkeit", "authentizitaet", "emotionalitaet", "transparenz"]
        )
        self.adjusted = {}

    def test_live_monitor_update_and_notify(self):
        notified = []
        self.monitor.add_listener(lambda goal, value: notified.append((goal, value)))
        self.monitor.update_goal("bass_kraft", 0.8)
        self.assertEqual(self.monitor.get_goal("bass_kraft"), 0.8)
        self.assertIn(("bass_kraft", 0.8), notified)

    def test_feedback_loop_adjustment(self):
        def adjust_callback(deviations):
            self.adjusted.update(deviations)

        loop = MusicalGoalsFeedbackLoop(self.monitor, adjust_callback)
        self.monitor.update_goal("bass_kraft", 0.5)
        loop.check_and_adjust()
        self.assertIn("bass_kraft", self.adjusted)
        self.assertLess(self.adjusted["bass_kraft"], 0.7)

    def test_goal_optimizer(self):
        optimizer = MusicalGoalsOptimizer(self.monitor)
        self.monitor.update_goal("brillanz", 0.6)
        optimizer.optimize()  # Should print boosting message

    def test_convergence_detector(self):
        detector = MusicalGoalsConvergenceDetector(self.monitor, tolerance=0.1)
        self.monitor.update_goal("waerme", 0.7)
        self.assertFalse(detector.has_converged())
        self.monitor.update_goal("waerme", 0.71)
        self.assertTrue(detector.has_converged())

    def test_deviation_corrector(self):
        corrector = MusicalGoalsDeviationCorrector(self.monitor)
        self.monitor.update_goal("emotionalitaet", 0.6)
        corrector.correct()  # Should print correcting message


if __name__ == "__main__":
    unittest.main()

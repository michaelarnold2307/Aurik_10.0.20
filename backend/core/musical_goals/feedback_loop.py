"""
FeedbackLoop für musikalische Ziele: Adaptive Anpassung der Verarbeitung.
"""

from collections.abc import Callable


class MusicalGoalsFeedbackLoop:
    def __init__(self, monitor, adjust_callback: Callable[[dict[str, float]], None]) -> None:
        self.monitor = monitor
        self.adjust_callback = adjust_callback
        self.thresholds = {
            "bass_kraft": 0.7,
            "brillanz": 0.6,
            "waerme": 0.7,
            "natuerlichkeit": 0.7,
            "authentizitaet": 0.7,
            "emotionalitaet": 0.7,
            "transparenz": 0.7,
        }

    def check_and_adjust(self) -> None:
        values = self.monitor.get_all_goals()
        deviations = {g: v for g, v in values.items() if g in self.thresholds and v < self.thresholds[g]}
        if deviations:
            self.adjust_callback(deviations)

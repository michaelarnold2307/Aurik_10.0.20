"""
LiveMonitor für musikalische Ziele: Echtzeit-Tracking und Feedback.
"""

import threading
from collections.abc import Callable


class MusicalGoalsLiveMonitor:
    def __init__(self, goals: list[str]):
        self.goals = goals
        self.values = dict.fromkeys(goals, 0.0)
        self.lock = threading.RLock()
        self.listeners: list[Callable[[str, float], None]] = []

    def update_goal(self, goal: str, value: float):
        with self.lock:
            if goal in self.values:
                self.values[goal] = value
                self._notify(goal, value)

    def get_goal(self, goal: str) -> float:
        with self.lock:
            return self.values.get(goal, 0.0)

    def get_all_goals(self) -> dict[str, float]:
        with self.lock:
            return self.values.copy()

    def add_listener(self, callback: Callable[[str, float], None]) -> None:
        with self.lock:
            self.listeners.append(callback)

    def _notify(self, goal: str, value: float) -> None:
        for listener in self.listeners:
            listener(goal, value)

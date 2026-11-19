"""
ConvergenceDetector: Erkennt Konvergenz der musikalischen Ziele.
"""


class MusicalGoalsConvergenceDetector:
    def __init__(self, monitor, tolerance: float = 0.05) -> None:
        self.monitor = monitor
        self.tolerance = tolerance
        self.last_values = None

    def has_converged(self) -> bool:
        current = self.monitor.get_all_goals()
        if self.last_values is None:
            self.last_values = current
            return False
        diffs = [abs(current[g] - self.last_values[g]) for g in current]
        self.last_values = current
        return all(d < self.tolerance for d in diffs)

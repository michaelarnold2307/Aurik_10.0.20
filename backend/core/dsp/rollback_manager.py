"""
RollbackManager: Rollback bei Over-Processing oder zu niedriger Confidence.
"""


class RollbackManager:
    """Entscheidet, ob ein Rollback der Processing-Kette nötig ist.

    Rollback-Kriterien (konfigurierbar):
    - Mindestens ein Modul unter critical_threshold  -> sofortiger Rollback
    - Mittlere Confidence unter mean_threshold        -> sanfter Rollback
    - Mehr als max_fail_ratio der Module fehlerhaft   -> Rollback
    """

    def __init__(
        self,
        critical_threshold: float = 0.3,
        mean_threshold: float = 0.6,
        max_fail_ratio: float = 0.5,
    ):
        self.critical_threshold = critical_threshold
        self.mean_threshold = mean_threshold
        self.max_fail_ratio = max_fail_ratio

    def should_rollback(self, confidence_map: dict) -> bool:
        """True, wenn Rollback nötig ist.

        :param confidence_map: {module_name: confidence_float [0, 1]}
        :return: bool
        """
        if not confidence_map:
            return False
        values = list(confidence_map.values())
        # Kritischer Schwellwert: sofortiger Rollback
        if any(float(v) < self.critical_threshold for v in values):
            return True
        # Mittlere Confidence zu niedrig
        mean_conf = sum(float(v) for v in values) / len(values)
        if mean_conf < self.mean_threshold:
            return True
        # Zu viele fehlgeschlagene Module
        n_fail = sum(1 for v in values if float(v) < 0.5)
        return bool(len(values) > 0 and n_fail / len(values) >= self.max_fail_ratio)

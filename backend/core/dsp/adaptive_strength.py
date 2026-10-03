"""
AdaptiveStrength: Passt die Processing-Intensität basierend auf Confidence an.
"""

import math


class AdaptiveStrength:
    """Sigmoid-basierte adaptive Stärkenanpassung pro Modul.

    - confidence < low_threshold  -> Gestärke stark reduziert (min_strength)
    - confidence > high_threshold -> Gestärke leicht erhöht (max_strength)
    - dazwischen            -> Sigmoidale Überblendung
    """

    def __init__(
        self,
        low_threshold: float = 0.6,
        high_threshold: float = 0.9,
        min_strength: float = 0.3,
        max_strength: float = 1.2,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_strength = min_strength
        self.max_strength = max_strength

    def _sigmoid_strength(self, conf: float) -> float:
        """Mappe Confidence [0,1] auf Strength [min_strength, max_strength] via Sigmoid."""
        # Sigmoid zentriert bei (low+high)/2, Steilheit k=20
        center = (self.low_threshold + self.high_threshold) / 2.0
        k = 20.0
        s = 1.0 / (1.0 + math.exp(-k * (conf - center)))
        return self.min_strength + s * (self.max_strength - self.min_strength)

    def adjust_strength(self, confidence_map: dict) -> dict:
        """Gibt für jedes Modul einen Strength-Faktor zurück.

        :param confidence_map: {module_name: confidence_float [0, 1]}
        :return: {module_name: strength_factor}
        """
        return {mod: self._sigmoid_strength(float(conf)) for mod, conf in confidence_map.items()}

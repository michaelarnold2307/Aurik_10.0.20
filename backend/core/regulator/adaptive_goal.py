"""
AdaptiveGoalEngine für AURIK: Dynamische Zielsetzung und Anpassung der Audioverarbeitung basierend auf Kontext, Nutzerpräferenz und Analyse.
SOTA-Architektur, modular und erweiterbar.
"""

from typing import Any

import numpy as np


class AdaptiveGoalEngine:
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.last_goal: dict = {}

    def set_goals(self, context: dict[str, Any], user_prefs: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Setzt dynamisch die Zielparameter für die Verarbeitung.
        Kontext: Analyse-Features (Genre, Medium, Instrumentierung, etc.)
        user_prefs: optionale Nutzerpräferenzen (z.B. Ziel-Lautheit, Brillanz, Natürlichkeit)
        """
        goals = {}
        # Beispiel: Ziel-Lautheit je nach Genre
        genre = context.get("genre", "Unknown")
        if genre == "Electronic/Pop":
            goals["target_lufs"] = -8.0
        elif genre == "Rock/Indie":
            goals["target_lufs"] = -10.0
        else:
            goals["target_lufs"] = -14.0
        # Beispiel: Ziel-Brillanz
        goals["target_brightness"] = 0.7 if genre in ["Electronic/Pop", "Rock/Indie"] else 0.5
        # Nutzerpräferenzen überschreiben Defaults
        if user_prefs:
            goals.update(user_prefs)
        return goals

    def adapt_processing(self, audio: np.ndarray, goals: dict[str, Any]) -> np.ndarray:
        """
        Passt das Audio gemäß der gesetzten Ziele an (Dummy-Implementierung, SOTA: ML/Rule-Engine möglich).
        """
        target_lufs = goals.get("target_lufs", -14.0)
        rms = np.sqrt(np.mean(audio**2))
        gain = 0.1 if rms == 0 else (10 ** ((target_lufs + 23) / 20)) / rms
        return audio * gain  # type: ignore[no-any-return]

    def define_goal(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Leitet aus einem Kontext-Objekt ein Ziel ab (portiert aus backend.adaptive_goal).
        Kompatibilitäts-API für ältere Aufrufer; neue Aufrufer sollten set_goals() verwenden.

        Args:
            context: Kontext-Dict mit Schlüsseln wie genre_hint, user_level, artefact_risk,
                     reference_similarity.

        Returns:
            goal-Dict mit target_brightness, quality_level, artefact_avoidance.
        """
        goal: dict[str, Any] = {}
        genre = context.get("genre_hint", "Unbekannt")
        user_level = context.get("user_level", "default")
        artefact_risk = context.get("artefact_risk", False)

        if genre in ("Pop/Rock", "Electronic/Pop", "Rock/Metal"):
            goal["target_brightness"] = "hoch"
        elif genre in ("Klassik/Jazz", "Classical/Jazz", "Jazz/Blues"):
            goal["target_brightness"] = "moderat"
        else:
            goal["target_brightness"] = "neutral"

        goal["quality_level"] = "maximal" if user_level == "pro" else "standard"
        goal["artefact_avoidance"] = artefact_risk

        if context.get("reference_similarity", 1.0) < 0.8:
            goal["reference_warning"] = "Referenz weicht stark ab!"

        self.last_goal = goal
        return goal

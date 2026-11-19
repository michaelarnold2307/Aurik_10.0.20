"""§GUI-T6 — Live-15-Ziel-Radar während der Restaurierung.

Die Engine sendet je Phase den PMGG-Goal-Snapshot (live_metrics["goals"]);
das Fenster reicht ihn ans Radar-Widget weiter. Testet die reine
Datenweitergabe mit einem Stub-Radar (kein Qt-Init nötig).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")  # CI-Minimal-Umgebung (cross-platform)

from Aurik10.ui.modern_window import ModernMainWindow


class _FakeRadar:
    def __init__(self) -> None:
        self.scores: dict | None = None

    def update_scores(self, **kwargs) -> None:
        self.scores = kwargs.get("scores")


def test_live_goal_radar_forwards_scores() -> None:
    win = ModernMainWindow.__new__(ModernMainWindow)  # ohne Qt-Init
    win.radar_widget = _FakeRadar()
    win._update_live_goal_radar({"brillanz": 0.8, "waerme": 0.5})
    assert win.radar_widget.scores == {"brillanz": 0.8, "waerme": 0.5}


def test_live_goal_radar_no_radar_is_safe() -> None:
    win = ModernMainWindow.__new__(ModernMainWindow)
    # kein radar_widget gesetzt → kein Crash, keine Exception
    win._update_live_goal_radar({"brillanz": 0.8})

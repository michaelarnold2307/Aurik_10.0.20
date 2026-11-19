from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from Aurik10.ui.song_prognose_widget import SongPrognoseWidget


@pytest.mark.unit
def test_set_preventive_actions_renders_multiline_text() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    widget = SongPrognoseWidget()
    widget.set_preventive_actions(
        [
            "Phase 31 Damage-Shield aktiv · L/R-Realignment",
            "Phase 12 Loudness-Makeup: +0.80 dB",
        ]
    )

    txt = widget._preventive_lbl.text()
    assert "🛡 Phase 31 Damage-Shield aktiv · L/R-Realignment" in txt
    assert "🛡 Phase 12 Loudness-Makeup: +0.80 dB" in txt


@pytest.mark.unit
def test_set_preventive_actions_limits_to_three_lines() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    widget = SongPrognoseWidget()
    widget.set_preventive_actions(["A", "B", "C", "D"])

    txt = widget._preventive_lbl.text()
    assert "🛡 A" in txt
    assert "🛡 B" in txt
    assert "🛡 C" in txt
    assert "🛡 D" not in txt


@pytest.mark.unit
def test_set_preventive_actions_empty_clears_label() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    widget = SongPrognoseWidget()
    widget.set_preventive_actions(["A"])
    assert widget._preventive_lbl.text()

    widget.set_preventive_actions([])
    assert widget._preventive_lbl.text() == ""

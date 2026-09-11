from __future__ import annotations

"""Tests for defect visualization story and extended defect mapping in modern_window."""


from dataclasses import dataclass, field

import pytest

pytestmark = pytest.mark.gui


@dataclass
class _DummyDefectScore:
    severity: float
    locations: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def test_defect_analysis_to_display_includes_extended_defect_types() -> None:
    """The display conversion must expose newly added defect keys and locations."""
    pytest.importorskip("PyQt5")

    from Aurik10.ui.modern_window import _defect_analysis_to_display
    from backend.core.defect_scanner import DefectType

    scores = {
        DefectType.SOFT_SATURATION: _DummyDefectScore(0.34, [(1.0, 1.6)]),
        DefectType.AZIMUTH_ERROR: _DummyDefectScore(0.52, [(4.2, 5.0)]),
        DefectType.VOCAL_HARSHNESS: _DummyDefectScore(
            0.61,
            [(11.0, 11.8)],
            metadata={"channel_locations": {"L": [(11.0, 11.4)], "R": [(11.4, 11.8)]}},
        ),
    }

    display = _defect_analysis_to_display(scores, status="detected")

    assert display["soft_saturation"] == pytest.approx(34.0, abs=0.2)
    assert display["azimuth_error"] == pytest.approx(52.0, abs=0.2)
    assert display["vocal_harshness"] == pytest.approx(61.0, abs=0.2)

    assert display["_locations"]["soft_saturation"] == [(1.0, 1.6)]
    assert display["_locations"]["azimuth_error"] == [(4.2, 5.0)]
    assert display["_locations"]["vocal_harshness"] == [(11.0, 11.8)]


def test_result_scores_to_display_includes_extended_defect_types() -> None:
    """Completed-result conversion must also expose the extended defect keys."""
    pytest.importorskip("PyQt5")

    from Aurik10.ui.modern_window import _result_scores_to_display
    from backend.core.defect_scanner import DefectType

    result_scores = {
        DefectType.SOFT_SATURATION: 0.08,
        DefectType.AZIMUTH_ERROR: 0.03,
        DefectType.VOCAL_HARSHNESS: 0.11,
    }

    display = _result_scores_to_display(result_scores, status="completed")
    assert display["soft_saturation"] == pytest.approx(8.0, abs=0.2)
    assert display["azimuth_error"] == pytest.approx(3.0, abs=0.2)
    assert display["vocal_harshness"] == pytest.approx(11.0, abs=0.2)


def test_sibilance_locations_force_min_display_count() -> None:
    """Sibilance with temporal locations must remain visible after UI scaling."""
    pytest.importorskip("PyQt5")

    from Aurik10.ui.modern_window import _defect_analysis_to_display
    from backend.core.defect_scanner import DefectType

    # Very low severity that would round to 0 with int(sev*300), but locations exist.
    scores = {
        DefectType.SIBILANCE: _DummyDefectScore(0.001, [(2.0, 2.1)]),
    }

    display = _defect_analysis_to_display(scores, status="detected")

    assert display["sibilance"] >= 1
    assert display["_locations"]["sibilance"] == [(2.0, 2.1)]


def test_defect_story_widget_renders_layman_story_matrix() -> None:
    """DefectStoryWidget should render WAS/WO/WIE/WANN/WESHALB with full matrix header."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.modern_window import DefectStoryWidget

    app = QApplication.instance() or QApplication([])
    _ = app

    widget = DefectStoryWidget()

    defects = {
        "status": "correcting",
        "_locations": {
            "clicks": [(0.2, 0.3), (1.4, 1.45)],
            "vocal_harshness": [(12.0, 12.7)],
        },
        "_channel_locations": {
            "vocal_harshness": {"L": [(12.0, 12.3)], "R": [(12.3, 12.7)]},
        },
        "clicks": 120,
        "vocal_harshness": 67.0,
        "soft_saturation": 8.0,
    }

    widget.update_story(defects, phase_text="phase_42_vocal_enhancement", active_tool="BSRoFormer")
    html = widget._body.text()

    assert "Erkannte Defekte" in html
    assert "WAS:" in html
    assert "WO:" in html
    assert "WIE:" in html
    assert "WANN:" in html
    assert "WESHALB:" in html
    assert "Impact" in html
    assert "Prio" in html
    assert "Vocal-Härte" in html
    assert "phase_42_vocal_enhancement" in html
    assert "BSRoFormer" in html


def test_defect_story_widget_marks_completed_low_scores_as_fixed() -> None:
    """In completed state, near-zero defects should be shown as fixed."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.modern_window import DefectStoryWidget

    app = QApplication.instance() or QApplication([])
    _ = app

    widget = DefectStoryWidget()
    defects = {
        "status": "completed",
        "clicks": 0.0,
        "vocal_harshness": 0.0,
    }

    widget.update_story(defects)
    html = widget._body.text()

    assert "behoben" in html


def test_defect_story_widget_compact_mode_for_many_active_defects() -> None:
    """During active processing with many defects, compact-mode hint should appear."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.modern_window import DefectStoryWidget

    app = QApplication.instance() or QApplication([])
    _ = app

    widget = DefectStoryWidget()
    defects = {
        "status": "correcting",
        "clicks": 300,
        "crackle": 240,
        "hum": 18,
        "wow": 1.4,
        "flutter": 1.2,
        "stereo_imbalance": 32,
        "digital_artifacts": 41,
        "rumble": 30,
        "noise_level": 55,
        "compression_artifacts": 38,
        "phase_issues": 44,
        "dropout": 21,
        "clipping": 87,
        "dc_offset": 16,
        "bandwidth_loss": 37,
        "pitch_drift": 52,
        "reverb_excess": 28,
        "print_through": 19,
    }
    widget.update_story(defects, phase_text="phase_03_denoise", active_tool="DeepFilterNet")
    html = widget._body.text()

    assert "Kompaktansicht aktiv" in html
    assert "Impact" in html

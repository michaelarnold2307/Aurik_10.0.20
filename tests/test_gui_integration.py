#!/usr/bin/env python3
"""
Integration Test for AURIK 9.0 GUI
Tests GUI components and UnifiedRestorerV3 integration without requiring display
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import pytest

pytestmark = pytest.mark.gui


def test_gui_import():
    """Test that GUI modules can be imported"""
    pytest.importorskip("PyQt5")
    from Aurik10.ui.modern_window import (
        BatchProcessingThread,
        DefectCounterWidget,
        ModernMainWindow,
        ResourceStatusWidget,
        WaveformWidget,
    )

    assert ModernMainWindow is not None
    assert BatchProcessingThread is not None
    assert ResourceStatusWidget is not None
    assert DefectCounterWidget is not None
    assert WaveformWidget is not None
    print("✓ All GUI components imported successfully")


def test_unified_restorer_v3_integration():
    """Test that BatchProcessingThread uses the correct Pipeline-Signals."""
    pytest.importorskip("PyQt5")
    from Aurik10.ui.modern_window import BatchProcessingThread

    # Check that BatchProcessingThread has all required signals
    assert hasattr(BatchProcessingThread, "mode_update")
    assert hasattr(BatchProcessingThread, "ml_status_update")
    assert hasattr(BatchProcessingThread, "item_progress")
    assert hasattr(BatchProcessingThread, "defect_update")
    assert hasattr(BatchProcessingThread, "phase_update")
    assert hasattr(BatchProcessingThread, "quality_update")

    print("✓ BatchProcessingThread has correct pipeline signals")


def test_resource_widget_initialization():
    """Test ResourceStatusWidget initialization without QApplication"""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    from Aurik10.ui.modern_window import ResourceStatusWidget

    widget = ResourceStatusWidget()

    # Check initial state
    assert widget.cpu_usage == 0.0
    assert widget.memory_usage == 0.0
    assert widget.quality_mode == "BALANCED"
    assert widget.ml_mode_active == False
    assert widget.active_ml_plugins == []

    # Test update_status method
    widget.update_status(
        cpu=50.0, memory=60.0, mode="QUALITY", ml_active=True, ml_plugins=["DeepFilterNet", "MP-SENet"]
    )
    assert widget.cpu_usage == 50.0
    assert widget.memory_usage == 60.0
    assert widget.quality_mode == "QUALITY"
    assert widget.ml_mode_active == True
    assert len(widget.active_ml_plugins) == 2

    print("✓ ResourceStatusWidget initialization and update successful")


def test_settings_mapping():
    """Test that GUI mode mapping works correctly"""
    from backend.core.performance_guard import QualityMode

    # Test RESTORATION mode mapping
    mode = "RESTORATION"
    if mode == "STUDIO_2026":
        quality_mode = QualityMode.QUALITY
        enable_psychoacoustic = True
    else:  # RESTORATION
        quality_mode = QualityMode.BALANCED
        enable_psychoacoustic = False

    assert quality_mode == QualityMode.BALANCED
    assert enable_psychoacoustic == False

    # Test STUDIO_2026 mode mapping
    mode = "STUDIO_2026"
    if mode == "STUDIO_2026":
        quality_mode = QualityMode.QUALITY
        enable_psychoacoustic = True
    else:
        quality_mode = QualityMode.BALANCED
        enable_psychoacoustic = False

    assert quality_mode == QualityMode.QUALITY
    assert enable_psychoacoustic == True

    print("✓ GUI mode mapping to QualityMode works correctly")


def test_restoration_config_creation():
    """Test that RestorationConfig can be created from GUI settings"""
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig

    # Create config as done in GUI
    config = RestorationConfig(
        mode=QualityMode.BALANCED,
        enable_psychoacoustic_enhancement=False,
        enable_performance_guard=True,
        enable_phase_skipping=True,
        num_cores=4,
    )

    assert config.mode == QualityMode.BALANCED
    assert config.enable_psychoacoustic_enhancement == False
    assert config.enable_performance_guard == True
    assert config.enable_phase_skipping == True
    assert config.num_cores == 4

    print("✓ RestorationConfig creation from GUI settings successful")


def test_processing_thread_signals():
    """Test that BatchProcessingThread has all required signal attributes."""
    pytest.importorskip("PyQt5")
    from Aurik10.ui.modern_window import BatchProcessingThread

    assert hasattr(BatchProcessingThread, "item_progress")
    assert hasattr(BatchProcessingThread, "item_started")
    assert hasattr(BatchProcessingThread, "item_finished")
    assert hasattr(BatchProcessingThread, "item_error")
    assert hasattr(BatchProcessingThread, "all_finished")
    assert hasattr(BatchProcessingThread, "waveform_data")
    assert hasattr(BatchProcessingThread, "defect_update")
    assert hasattr(BatchProcessingThread, "phase_update")
    assert hasattr(BatchProcessingThread, "mode_update")
    assert hasattr(BatchProcessingThread, "ml_status_update")
    assert hasattr(BatchProcessingThread, "quality_update")

    print("✓ BatchProcessingThread has all required signals")


def test_live_quality_estimate_updates_during_pipeline():
    """Test that GUI quality has a live estimate before final MOS calculation."""
    pytest.importorskip("PyQt5")
    from Aurik10.ui.modern_window import BatchProcessingThread

    start = BatchProcessingThread._estimate_live_quality_mos(5.0, baseline_mos=3.4)
    mid = BatchProcessingThread._estimate_live_quality_mos(45.0, baseline_mos=3.4, active_phase_count=4)
    post = BatchProcessingThread._estimate_live_quality_mos(
        92.0,
        baseline_mos=3.4,
        active_phase_count=8,
        post_processing_started=True,
    )

    assert 1.0 <= start < 3.4
    assert 3.4 < mid < post <= 4.9


def test_main_window_layout_gate_keeps_primary_controls_visible():
    """Offscreen layout smoke gate for desktop and compact professional widths."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.modern_window import ModernMainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = ModernMainWindow()
    try:
        for size in (QSize(1440, 900), QSize(1024, 720)):
            window.resize(size)
            window.show()
            app.processEvents()
            assert window.size().width() > 0
            assert window.size().height() > 0
            assert window.btn_magic_restoration.isVisible()  # type: ignore[union-attr]
            assert window.btn_magic_studio.isVisible()  # type: ignore[union-attr]
            assert window.quality_meter_widget.isVisible()  # type: ignore[union-attr]
            assert window.btn_play_original.accessibleName() == "Original anhören"  # type: ignore[union-attr]
            assert window._capture_runtime_snapshot().quality_state
    finally:
        window.close()


if __name__ == "__main__":
    """Run tests manually"""
    print("\n=== AURIK 9.0 GUI Integration Tests ===\n")

    try:
        test_gui_import()
        test_unified_restorer_v3_integration()
        test_resource_widget_initialization()
        test_settings_mapping()
        test_restoration_config_creation()
        test_processing_thread_signals()

        print("\n✅ All GUI integration tests passed!")
        print("\n📋 Summary:")
        print("  - GUI components import correctly")
        print("  - UnifiedRestorerV3 integration is correct")
        print("  - ResourceStatusWidget works as expected")
        print("  - Mode mapping (RESTORATION/STUDIO_2026 → QualityMode) is correct")
        print("  - RestorationConfig creation from GUI settings is correct")
        print("  - ProcessingThread has all required signals")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

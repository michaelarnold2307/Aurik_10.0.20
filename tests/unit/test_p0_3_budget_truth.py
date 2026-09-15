"""§P0-3 (Budget-Wahrheit) — Ehrliches Messzeit-Reporting + EINE Budget-Norm.

Deckt:
- ``PerformanceGuard.get_budget_truth_report()``: Wand-Zeit, Processing-Zeit und
  Analytics-Anteil getrennt — rt_wall (Anwender-Wartezeit) ≥ rt_processing
  (Budget-Last); Analytics wird NICHT verschleiert, sondern ausgewiesen.
- Norm-Konsistenz: alle Modi teilen 32× RT (§2.38 KMV normativ) — die früheren
  abweichenden Limits (3×-Doku) sind Geschichte.

Autor: Aurik Testing Team
"""

import time

from backend.core.performance_guard import PerformanceGuard


def test_report_separates_wall_processing_analytics():
    guard = PerformanceGuard()
    guard.start_monitoring(10.0)
    time.sleep(0.02)
    guard.add_analytics_overhead(1.5)  # 1,5 s fiktive Analytics (measure_all etc.)
    report = guard.get_budget_truth_report()

    assert report["audio_duration_s"] == 10.0  # echte Audio-Dauer (vor 30-s-Floor)
    assert report["budget_duration_s"] == 30.0  # effektive Budget-Basis des Guards
    assert report["analytics_overhead_s"] == 1.5
    assert report["wall_elapsed_s"] >= report["processing_elapsed_s"]
    assert report["processing_elapsed_s"] == report["wall_elapsed_s"] - 1.5
    assert report["rt_wall"] >= report["rt_processing"]
    assert report["target_rt_factor"] == 32.0  # §2.38 KMV normativ


def test_report_before_monitoring_is_zero_safe():
    guard = PerformanceGuard()
    report = guard.get_budget_truth_report()
    assert report["wall_elapsed_s"] == 0.0
    assert report["rt_wall"] == 0.0
    assert report["rt_processing"] == 0.0


def test_single_norm_all_modes_share_32x():
    # P0-3: EINE Budget-Norm — frühere Divergenz (3×-Doku vs. 32×-Guard) ist
    # bereinigt; alle Modi teilen dasselbe End-to-End-Limit.
    assert PerformanceGuard.LIMIT_BALANCED == 32.0
    assert PerformanceGuard.LIMIT_QUALITY == 32.0
    assert PerformanceGuard.LIMIT_MAXIMUM == 32.0
    assert PerformanceGuard.LIMIT_3X_RT == 32.0


def test_analytics_does_not_inflate_processing_rt():
    guard = PerformanceGuard()
    guard.start_monitoring(5.0)
    guard.add_analytics_overhead(50.0)  # Analytics weit über Audio-Dauer
    report = guard.get_budget_truth_report()
    # Processing-RT bleibt 0 (Analytics abgezogen), Wand-RT weist es aus.
    assert report["processing_elapsed_s"] == 0.0
    assert report["rt_processing"] == 0.0
    assert report["analytics_overhead_s"] == 50.0

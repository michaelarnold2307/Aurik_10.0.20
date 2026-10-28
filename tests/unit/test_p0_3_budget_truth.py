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
    guard.add_analytics_overhead(50.0)  # Analytics weit ueber Audio-Dauer
    report = guard.get_budget_truth_report()
    # Processing-RT bleibt 0 (Analytics abgezogen), Wand-RT weist es aus.
    assert report["processing_elapsed_s"] == 0.0
    assert report["rt_processing"] == 0.0
    assert report["analytics_overhead_s"] == 50.0


def test_plan_budget_capped_to_hard_guard_despite_chain_depth_and_restorability():
    """D-K3-22: Budget-Divergenz im StrategieDenker ist geschlossen.

    ``StrategieDenker.plan()`` skaliert das Budget mit Ketten-Tiefe (<=2.0) und
    Restaurierbarkeits-Faktor (<=1.5) und meldete frueher bis zu 73.6x RT,
    waehrend der harte Ausstieg bei 32x stoppt. Der Plan darf nie mehr Zeit
    zusagen, als der Guard-Ausstieg hergibt: der Wunsch wird auf die harte
    32x-Grenze gedeckelt.
    """
    import numpy as np

    from denker.strategie_denker import StrategieDenker

    audio_dur = 10.0
    audio = np.zeros(int(48000 * audio_dur), dtype=np.float32)
    denker = StrategieDenker()

    # Extremfall: tiefe Kette (9) + minimale Restaurierbarkeit (0) =>
    # Budget-Wunsch = 32 x 2.0 x 1.5 = 96x RT, also weit ueber dem Ausstieg.
    plan = denker.plan(
        audio, 48000, mode="quality", chain_depth=9, restorability_score=0.0
    )

    assert plan.audio_duration_s == audio_dur
    # Deckel: nie mehr als die harte Guard-Grenze (32x) * Audiodauer.
    _hard_cap = PerformanceGuard.LIMIT_3X_RT * plan.audio_duration_s
    assert plan.max_processing_s <= _hard_cap + 1e-6
    # Der Deckel muss bei diesem Wunsch exakt greifen (Wunsch > Grenze).
    assert abs(plan.max_processing_s - _hard_cap) < 1e-6
    # Konsistenz: die harte Grenze ist die 32x-Norm (§2.38 KMV).
    assert PerformanceGuard.LIMIT_3X_RT == 32.0

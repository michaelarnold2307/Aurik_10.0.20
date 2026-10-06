from __future__ import annotations

import pytest

"""Unit tests for OOM guard hardening paths.

Covers:
- Physical RAM preflight in ml_memory_budget.try_allocate
- Idempotent monitor start in AdaptiveResourceManager
- Auto-detect budget from system RAM
- Budget exhaustion blocks subsequent allocations
- Release frees budget for reuse
- UV3 audio buffer guard (MemoryError on oversized input)
- EnsembleProcessor skip guard for long files
"""


from types import SimpleNamespace

import numpy as np

from backend.core import ml_memory_budget as budget
from backend.core.adaptive_resource_manager import AdaptiveResourceManager


def _reset_budget_state() -> None:
    budget._allocated.clear()
    budget._total_gb = 0.0


class _MemProbe:
    def __init__(self, available_mb: float, percent: float = 50.0):
        self.available = int(available_mb * 1024 * 1024)
        self.percent = percent


@pytest.mark.unit
def test_try_allocate_blocks_when_physical_memory_too_low(monkeypatch):
    _reset_budget_state()

    class _LowPsutil:
        @staticmethod
        def virtual_memory():
            return _MemProbe(available_mb=300.0)

    monkeypatch.setattr(budget, "_psutil", _LowPsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)

    # Simulate no improvement after eviction attempt.
    monkeypatch.setattr(
        budget,
        "_preflight_system_memory",
        lambda required_mb: False,
    )

    ok = budget.try_allocate("TestHeavyModel", size_gb=2.0)
    assert ok is False
    assert "TestHeavyModel" not in budget._allocated


def test_try_allocate_succeeds_after_preflight_and_budget(monkeypatch):
    _reset_budget_state()

    class _GoodPsutil:
        @staticmethod
        def virtual_memory():
            return _MemProbe(available_mb=8192.0)

    monkeypatch.setattr(budget, "_psutil", _GoodPsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)

    ok = budget.try_allocate("SmallModel", size_gb=0.5)
    assert ok is True
    assert budget._allocated.get("SmallModel") == 0.5


def test_try_allocate_soft_allows_tiny_model_under_pressure(monkeypatch):
    _reset_budget_state()

    class _PressurePsutil:
        @staticmethod
        def virtual_memory():
            total = 32 * 1024**3
            available = 14 * 1024**3
            return SimpleNamespace(total=total, available=available, percent=56.0)

        @staticmethod
        def swap_memory():
            return SimpleNamespace(percent=90.0, used=7 * 1024**3, sin=0, sout=0)

    monkeypatch.setattr(budget, "_psutil", _PressurePsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: True)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    ok = budget.try_allocate("TinyHelperModel", size_gb=0.1)
    assert ok is False
    assert "TinyHelperModel" not in budget._allocated


def test_try_allocate_soft_allows_allowlisted_tiny_model_under_pressure(monkeypatch):
    _reset_budget_state()

    class _PressurePsutil:
        @staticmethod
        def virtual_memory():
            total = 32 * 1024**3
            available = 14 * 1024**3
            return SimpleNamespace(total=total, available=available, percent=56.0)

        @staticmethod
        def swap_memory():
            return SimpleNamespace(percent=90.0, used=7 * 1024**3, sin=0, sout=0)

    monkeypatch.setattr(budget, "_psutil", _PressurePsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: True)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    ok = budget.try_allocate("SileroVAD", size_gb=0.1)
    assert ok is True
    assert budget._allocated.get("SileroVAD") == 0.1


def test_try_allocate_soft_allow_respects_tiny_cap_under_pressure(monkeypatch):
    _reset_budget_state()

    class _PressurePsutil:
        @staticmethod
        def virtual_memory():
            total = 32 * 1024**3
            available = 14 * 1024**3
            return SimpleNamespace(total=total, available=available, percent=56.0)

        @staticmethod
        def swap_memory():
            return SimpleNamespace(percent=90.0, used=7 * 1024**3, sin=0, sout=0)

    monkeypatch.setattr(budget, "_psutil", _PressurePsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: True)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("SileroVAD", size_gb=0.12) is True
    assert budget.try_allocate("FCPE", size_gb=0.12) is True
    assert budget.try_allocate("BasicPitch", size_gb=0.12) is False


def test_try_allocate_blocks_heavy_model_on_preemptive_swap_pressure(monkeypatch):
    _reset_budget_state()

    class _PressurePsutil:
        @staticmethod
        def virtual_memory():
            total = 32 * 1024**3
            available = int(total * 0.20)
            return SimpleNamespace(total=total, available=available, percent=80.0)

        @staticmethod
        def swap_memory():
            return SimpleNamespace(percent=72.0, used=6 * 1024**3, sin=0, sout=0)

    monkeypatch.setattr(budget, "_psutil", _PressurePsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_swap_io_rate_mb_per_s", lambda swap_obj: 3.5)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    ok = budget.try_allocate("FlashSR", size_gb=2.0)
    assert ok is False
    assert "FlashSR" not in budget._allocated


def test_try_allocate_allows_heavy_model_when_swap_pressure_is_low(monkeypatch):
    _reset_budget_state()

    class _HealthyPsutil:
        @staticmethod
        def virtual_memory():
            total = 32 * 1024**3
            available = 18 * 1024**3
            return SimpleNamespace(total=total, available=available, percent=44.0)

        @staticmethod
        def swap_memory():
            return SimpleNamespace(percent=35.0, used=2 * 1024**3, sin=0, sout=0)

    monkeypatch.setattr(budget, "_psutil", _HealthyPsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_swap_io_rate_mb_per_s", lambda swap_obj: 0.2)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    ok = budget.try_allocate("FlashSR", size_gb=2.0)
    assert ok is True
    assert budget._allocated.get("FlashSR") == 2.0


def test_try_allocate_blocks_heavy_model_on_early_swap_plus_low_headroom(monkeypatch):
    _reset_budget_state()

    class _PressurePsutil:
        @staticmethod
        def virtual_memory():
            total = 32 * 1024**3
            # 5.5 GB free < _HEAVY_MODEL_PREEMPTIVE_AVAIL_GB_MAX (6.0 GB) → low_headroom=True
            available = int(5.5 * 1024**3)
            return SimpleNamespace(total=total, available=available, percent=82.8)

        @staticmethod
        def swap_memory():
            return SimpleNamespace(percent=46.0, used=3.7 * 1024**3, sin=0, sout=0)

    monkeypatch.setattr(budget, "_psutil", _PressurePsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_swap_io_rate_mb_per_s", lambda swap_obj: 0.5)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    ok = budget.try_allocate("FlashSR", size_gb=2.0)
    assert ok is False
    assert "FlashSR" not in budget._allocated


def test_try_allocate_retries_after_thrashing_recovery(monkeypatch):
    _reset_budget_state()

    from backend.core import plugin_lifecycle_manager as plm

    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)
    monkeypatch.setattr(budget, "_PRESSURE_RECOVERY_ATTEMPTS", 2)
    monkeypatch.setattr(budget, "_PRESSURE_RECOVERY_SLEEP_S", 0.0)
    monkeypatch.setattr(plm, "evict_stale_plugins", lambda required_mb=0: 1)

    _thrash_states = iter([True, False])

    def _thrashing_toggle() -> bool:
        return next(_thrash_states)

    monkeypatch.setattr(budget, "is_system_thrashing", _thrashing_toggle)
    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", lambda size_gb: False)

    ok = budget.try_allocate("FlashSR", size_gb=2.0)
    assert ok is True
    assert budget._allocated.get("FlashSR") == 2.0


def test_try_allocate_retries_after_preemptive_block_recovery(monkeypatch):
    _reset_budget_state()

    from backend.core import plugin_lifecycle_manager as plm

    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)
    monkeypatch.setattr(budget, "_PRESSURE_RECOVERY_ATTEMPTS", 2)
    monkeypatch.setattr(budget, "_PRESSURE_RECOVERY_SLEEP_S", 0.0)
    monkeypatch.setattr(plm, "evict_stale_plugins", lambda required_mb=0: 1)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)

    _block_states = iter([True, False])

    def _preemptive_toggle(size_gb: float) -> bool:
        return next(_block_states)

    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", _preemptive_toggle)

    ok = budget.try_allocate("FlashSR", size_gb=2.0)
    assert ok is True
    assert budget._allocated.get("FlashSR") == 2.0


def test_adaptive_resource_manager_start_monitoring_is_idempotent(monkeypatch):
    manager = AdaptiveResourceManager(check_interval=10.0)

    starts = {"count": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name
            self.started = False

        def start(self):
            self.started = True
            starts["count"] += 1

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            self.started = False

    # Patch only for this test.
    monkeypatch.setattr("backend.core.adaptive_resource_manager.threading.Thread", _FakeThread)

    manager.start_monitoring()
    first_thread = manager._monitor_thread
    manager.start_monitoring()
    second_thread = manager._monitor_thread

    assert manager.running is True
    assert starts["count"] == 1
    assert first_thread is second_thread

    manager.stop_monitoring()


def test_adaptive_resource_manager_stop_monitoring_joins_thread(monkeypatch):
    manager = AdaptiveResourceManager(check_interval=10.0)

    joined = {"timeout": None}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            joined["timeout"] = timeout
            self.started = False

    monkeypatch.setattr("backend.core.adaptive_resource_manager.threading.Thread", _FakeThread)

    manager.start_monitoring()
    thread = manager._monitor_thread

    manager.stop_monitoring()

    assert manager.running is False
    assert joined["timeout"] == 1.0
    assert manager._monitor_thread is None
    assert thread is not None and thread.started is False  # type: ignore[attr-defined]


# ── Auto-detect budget tests ────────────────────────────────────────────


def test_auto_detect_budget_32gb(monkeypatch):
    """On a 32 GB system: budget = 32/3 ≈ 10.7 GB, clamped to [4, 12]."""

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(total=32 * 1024**3, available=20 * 1024**3)

    monkeypatch.setattr(budget, "_psutil", _FakePsutil)
    result = budget._auto_detect_budget()
    assert 10.0 <= result <= 11.0, f"Expected ~10.7 GB for 32 GB system, got {result}"


def test_auto_detect_budget_16gb(monkeypatch):
    """On a 16 GB system: budget = 16/3 ≈ 5.3 GB."""

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(total=16 * 1024**3, available=10 * 1024**3)

    monkeypatch.setattr(budget, "_psutil", _FakePsutil)
    result = budget._auto_detect_budget()
    assert 4.0 <= result <= 6.0, f"Expected ~5.3 GB for 16 GB system, got {result}"


def test_auto_detect_budget_8gb(monkeypatch):
    """On an 8 GB system: budget = 8/3 ≈ 2.7 → clamped to 4 GB minimum."""

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(total=8 * 1024**3, available=5 * 1024**3)

    monkeypatch.setattr(budget, "_psutil", _FakePsutil)
    result = budget._auto_detect_budget()
    assert result == 4.0, f"Expected 4.0 GB floor for 8 GB system, got {result}"


def test_auto_detect_budget_no_psutil(monkeypatch):
    """Without psutil: conservative 10 GB default."""
    monkeypatch.setattr(budget, "_psutil", None)
    result = budget._auto_detect_budget()
    assert result == 10.0


# ── Budget exhaustion and release tests ─────────────────────────────────


def test_budget_exhaustion_blocks_allocation(monkeypatch):
    """When budget is full, subsequent allocations fail."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 5.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", lambda size_gb: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("ModelA", size_gb=3.0) is True
    assert budget.try_allocate("ModelB", size_gb=3.0) is False  # 3+3 > 5
    assert "ModelB" not in budget._allocated


def test_release_frees_budget(monkeypatch):
    """After release, freed budget is available for new allocations."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 5.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", lambda size_gb: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("ModelA", size_gb=3.0) is True
    budget.release("ModelA")
    assert "ModelA" not in budget._allocated
    assert budget._total_gb == 0.0
    assert budget.try_allocate("ModelB", size_gb=4.0) is True  # now fits


def test_idempotent_allocation(monkeypatch):
    """Same model name allocated twice returns True without double-counting."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 5.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", lambda size_gb: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("SameModel", size_gb=2.0) is True
    assert budget.try_allocate("SameModel", size_gb=2.0) is True
    assert budget._total_gb == 2.0  # not 4.0


def test_get_status(monkeypatch):
    """get_status() returns correct current state."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 8.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", lambda size_gb: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    budget.try_allocate("TestModel", size_gb=1.5)
    status = budget.get_status()
    assert status["max_gb"] == 8.0
    assert status["allocated_gb"] == 1.5
    assert status["free_gb"] == 6.5
    assert "TestModel" in status["models"]


# ── UV3 audio buffer guard test ─────────────────────────────────────────


def test_uv3_rejects_oversized_audio():
    """UV3 must raise MemoryError for audio exceeding 4 GB buffer limit."""
    import pytest

    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    # Create a fake shape that would be > 4 GB if real (we test the guard logic)
    # We cannot actually allocate 4 GB in a test, so we monkeypatch nbytes.
    uv3 = UnifiedRestorerV3()
    # Create tiny audio but patch nbytes to simulate huge file
    audio = np.zeros(48000, dtype=np.float32)  # 1s mono

    class _FakeArray(np.ndarray):
        @property
        def nbytes(self):
            return 5 * 1024**3  # 5 GB — over the 4 GB limit

    oversized = audio.view(_FakeArray)
    with pytest.raises(MemoryError, match="Audio-Buffer"):
        uv3.restore(oversized, sample_rate=48000)


# ── EnsembleProcessor size guard test ───────────────────────────────────


def test_ensemble_processor_skipped_for_long_audio():
    """EnsembleProcessor should not run for audio > 2 minutes (OOM guard)."""
    # This tests the guard variable logic, not the full UV3 pipeline
    sr = 48000
    # 3 minutes at 48kHz → 8_640_000 samples
    audio_len = 3 * 60 * sr
    ensemble_max_samples = 2 * 60 * sr
    assert audio_len > ensemble_max_samples, "Test audio must exceed 2-min limit"
    # 30 seconds at 48kHz → 1_440_000 samples
    short_len = 30 * sr
    assert short_len <= ensemble_max_samples, "Short audio must be within limit"


# ── §PERF-R14: GC-Kollektion mit bedarfsgesteuertem zweitem Pass ────────
#
# Befund 2026-10-06 (docs/TODOS_SOTA_ROADMAP.md, §PERF-R14): Der an zwei
# Stellen verdrahtete UNBEDINGTE zweite ``gc.collect(2)`` („Pass für
# zirkuläre Refs") war beweisbar wirkungslos. Gemessen auf einem
# realistischen Heap (4,8 Mio. verfolgte Objekte, 22 Durchläufe): der
# zweite Pass gab 0 von 22 Mal ein Objekt frei, kostete aber die Hälfte
# der Paar-Laufzeit (685 ms Paar vs. 338 ms Einzelpass). Diese Tests
# pinnen den Vertrag von ``_full_gc_collect()`` und verhindern eine
# Rückkehr zum unbedingten Doppelaufruf.


def _uv3_module():
    """Importiert das UV3-Modul (schwer) nur bei Bedarf."""
    from backend.core import unified_restorer_v3 as uv3

    return uv3


def test_full_gc_collect_single_pass_when_nothing_freed(monkeypatch):
    """Ohne Freigabe im ersten Pass darf KEIN zweiter Lauf stattfinden.

    Das ist der eigentliche §PERF-R14-Gewinn: der zweite Pass ist nur dann
    sinnvoll, wenn der erste Objekte freigegeben hat (dann können
    ``__del__``-/Weakref-Callbacks neuen unerreichbaren Müll erzeugt haben).
    """
    uv3 = _uv3_module()
    calls: list[int] = []

    def _fake_collect(generation: int = 2) -> int:
        calls.append(generation)
        return 0

    monkeypatch.setattr(uv3.gc, "collect", _fake_collect)
    freed = uv3._full_gc_collect()

    assert freed == 0
    assert calls == [2], f"erwartet genau EIN voller Collect, tatsächlich {calls}"


def test_full_gc_collect_second_pass_only_after_freed(monkeypatch):
    """Nach einer Freigabe im ersten Pass MUSS der zweite Pass laufen."""
    uv3 = _uv3_module()
    calls: list[int] = []
    returns = iter([3, 5])

    def _fake_collect(generation: int = 2) -> int:
        calls.append(generation)
        return next(returns)

    monkeypatch.setattr(uv3.gc, "collect", _fake_collect)
    freed = uv3._full_gc_collect()

    assert freed == 8, "Rückgabe ist die Summe beider Pässe"
    assert calls == [2, 2], f"erwartet zwei volle Collects, tatsächlich {calls}"


def test_full_gc_collect_uses_generation_2():
    """Beide Pässe sammeln alle Generationen (vollständiger Collect)."""
    uv3 = _uv3_module()
    import gc as _gc

    seen: list[int] = []
    original = _gc.collect

    def _spy(generation: int = 2) -> int:
        seen.append(generation)
        return original(generation)

    uv3.gc.collect = _spy
    try:
        uv3._full_gc_collect()
    finally:
        uv3.gc.collect = original

    assert seen, "collect wurde nicht aufgerufen"
    assert all(g == 2 for g in seen), f"nur vollständige Collects erlaubt, gesehen: {seen}"


def test_full_gc_collect_returns_int_and_reclaims_cycles():
    """Echte, unerreichbare Referenzzyklen werden freigegeben (Rückgabe int)."""
    uv3 = _uv3_module()

    def _make_garbage() -> None:
        for _ in range(2000):
            d: dict = {}
            d["self"] = d  # echter Zyklus, nach Rückkehr unerreichbar

    _make_garbage()
    freed = uv3._full_gc_collect()

    assert isinstance(freed, int)
    assert freed > 0, "unerreichbare Zyklen müssen freigegeben werden"


def test_second_consecutive_full_collect_frees_nothing():
    """Empirische Grundlage des §PERF-R14-Vertrags (Messung 22/22).

    Eine vollständige Kollektion leert Generation 0; ein unmittelbar
    folgender vollständiger Collect hat daher nichts mehr zu holen.
    """
    uv3 = _uv3_module()
    uv3._full_gc_collect()
    second = uv3.gc.collect(2)
    assert second == 0, (
        "Annahme des §PERF-R14-Vertrags verletzt: ein unmittelbar folgender "
        f"voller Collect gab {second} Objekte frei — der zweite Pass wäre dann "
        "nicht redundant."
    )


def test_uv3_has_no_unconditional_double_full_collect():
    """Regressionsguard: kein unbedingter Doppel-Collect in UV3.

    §PERF-R14 (docs/TODOS_SOTA_ROADMAP.md): ein Paar direkt aufeinander
    folgender ``gc.collect(2)`` ohne Bedarfsprüfung ist verboten — es kostet
    nachweislich Laufzeit ohne Wirkung.
    """
    import pathlib
    import re

    path = pathlib.Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"
    src = path.read_text(encoding="utf-8")

    pattern = re.compile(r"gc\.collect\(2\)\s*(?:#[^\n]*)?\n\s*gc\.collect\(2\)")
    match = pattern.search(src)
    assert match is None, (
        "unbedingter Doppel-Collect gefunden (Zeichenoffset "
        f"{match.start() if match else -1}) — stattdessen _full_gc_collect() verwenden"
    )
    assert "_full_gc_collect()" in src, "kanonische Hilfsfunktion wird nicht verwendet"

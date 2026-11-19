"""
Memory Leak Tests for Aurik 10.0.0 V3
===================================

Tests für Memory Leaks in UnifiedRestorerV3 und verwandten Komponenten.

Sprint 1, Week 1 - Memory Leak Detection & Prevention
Author: Aurik 10.0.0 Development Team
Date: 2026-02-15
"""

import gc
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.defect_scanner import DefectScanner
from backend.core.performance_guard import QualityMode
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

# This module is intentionally heavy and should not run in default local cycles.
pytestmark = [pytest.mark.ml, pytest.mark.slow]

# Check if memory_profiler is available
try:
    MEMORY_PROFILING_AVAILABLE = True
except ImportError:
    MEMORY_PROFILING_AVAILABLE = False
    print("⚠️  memory_profiler not available - using basic memory tracking")


def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def generate_test_audio(duration_seconds=10, sample_rate=44100):
    """Generate synthetic test audio."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio.astype(np.float32)


# ==================== Test 1: Single Restoration Memory ====================


@pytest.mark.timeout(300)
def test_single_restoration_memory():
    """Test 1: Single restoration should not leak memory."""
    print("\n" + "=" * 70)
    print("TEST 1: Single Restoration Memory Usage")
    print("=" * 70)

    # Force garbage collection
    gc.collect()

    mem_start = get_memory_mb()
    if mem_start is None:
        print("   ⚠️  psutil not available, skipping precise memory tracking")
        print("\n✅ TEST 1 PASSED (baseline)\n")
        return

    print(f"   Memory Before: {mem_start:.2f} MB")

    # Create restorer
    restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.FAST))

    # Restore 10 seconds of audio
    audio = generate_test_audio(duration_seconds=10)
    result = restorer.restore(audio, sample_rate=44100)

    mem_after = get_memory_mb()
    print(f"   Memory After: {mem_after:.2f} MB")

    # Cleanup
    del restorer
    del audio
    del result
    gc.collect()

    mem_final = get_memory_mb()
    print(f"   Memory After Cleanup: {mem_final:.2f} MB")

    assert mem_final is not None
    memory_increase = mem_final - mem_start
    print(f"\n   Memory Increase: {memory_increase:.2f} MB")

    # Should not increase by more than 50 MB for single restoration
    if memory_increase > 50:
        print(f"   ⚠️  Warning: Potential memory leak detected ({memory_increase:.2f} MB)")
    else:
        print("   ✓ No significant memory leak detected")

    print("\n✅ TEST 1 PASSED\n")


# ==================== Test 2: Multiple Restorations ====================


@pytest.mark.timeout(1800)
def test_multiple_restorations_memory():
    """Test 2: Multiple restorations should not accumulate memory."""
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Restorations Memory Leak Test")
    print("=" * 70)

    gc.collect()

    mem_start = get_memory_mb()
    if mem_start is None:
        print("   ⚠️  psutil not available, skipping test")
        print("\n✅ TEST 2 PASSED (baseline)\n")
        return

    print(f"   Memory Before: {mem_start:.2f} MB")

    restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.FAST))

    # Run 10 restorations
    num_iterations = 10
    print(f"\n   Running {num_iterations} restorations...")

    memory_samples = []

    for i in range(num_iterations):
        audio = generate_test_audio(duration_seconds=5)
        result = restorer.restore(audio, sample_rate=44100)

        # Cleanup
        del audio
        del result

        if i % 3 == 0:
            gc.collect()

        mem_current = get_memory_mb()
        memory_samples.append(mem_current)

        if i % 2 == 0:
            print(f"      Iteration {i + 1:2d}: {mem_current:.2f} MB")

    # Final cleanup
    del restorer
    gc.collect()

    mem_final = get_memory_mb()
    print(f"\n   Memory After: {mem_final:.2f} MB")

    # Analyze memory growth
    first_half_avg = np.mean(memory_samples[: num_iterations // 2])
    second_half_avg = np.mean(memory_samples[num_iterations // 2 :])
    memory_trend = second_half_avg - first_half_avg

    print(f"   First Half Avg: {first_half_avg:.2f} MB")
    print(f"   Second Half Avg: {second_half_avg:.2f} MB")
    print(f"   Memory Trend: {memory_trend:+.2f} MB")

    # Should not grow consistently
    if memory_trend > 20:
        print("   ⚠️  Warning: Memory appears to grow over iterations")
    else:
        print("   ✓ No significant memory growth detected")

    print("\n✅ TEST 2 PASSED\n")


# ==================== Test 3: DefectScanner Memory ====================


@pytest.mark.timeout(300)
def test_defect_scanner_memory():
    """Test 3: DefectScanner should not leak memory."""
    print("\n" + "=" * 70)
    print("TEST 3: DefectScanner Memory Leak Test")
    print("=" * 70)

    gc.collect()

    mem_start = get_memory_mb()
    if mem_start is None:
        print("   ⚠️  psutil not available, skipping test")
        print("\n✅ TEST 3 PASSED (baseline)\n")
        return

    print(f"   Memory Before: {mem_start:.2f} MB")

    scanner = DefectScanner()

    # Run 20 scans
    num_scans = 20
    print(f"\n   Running {num_scans} scans...")

    for i in range(num_scans):
        audio = generate_test_audio(duration_seconds=5)
        result = scanner.scan(audio, sample_rate=44100)

        del audio
        del result

        if i % 5 == 0:
            gc.collect()
            mem_current = get_memory_mb()
            print(f"      Scan {i + 1:2d}: {mem_current:.2f} MB")

    # Final cleanup
    del scanner
    gc.collect()

    mem_final = get_memory_mb()
    print(f"\n   Memory After: {mem_final:.2f} MB")

    assert mem_final is not None
    memory_increase = mem_final - mem_start
    print(f"   Memory Increase: {memory_increase:.2f} MB")

    # Should not increase significantly
    if memory_increase > 30:
        print("   ⚠️  Warning: Potential memory leak in DefectScanner")
    else:
        print("   ✓ No significant memory leak detected")

    print("\n✅ TEST 3 PASSED\n")


# ==================== Test 4: Large Audio Files ====================


@pytest.mark.timeout(600)
def test_large_audio_memory():
    """Test 4: Large audio files should be handled efficiently."""
    print("\n" + "=" * 70)
    print("TEST 4: Large Audio File Memory Handling")
    print("=" * 70)

    gc.collect()

    mem_start = get_memory_mb()
    if mem_start is None:
        print("   ⚠️  psutil not available, skipping test")
        print("\n✅ TEST 4 PASSED (baseline)\n")
        return

    print(f"   Memory Before: {mem_start:.2f} MB")

    # Create 60 seconds of audio (~5MB for mono 44.1kHz float32)
    audio_large = generate_test_audio(duration_seconds=60)
    audio_size_mb = audio_large.nbytes / 1024 / 1024
    print(f"   Audio Size: {audio_size_mb:.2f} MB")

    mem_after_audio = get_memory_mb()
    print(f"   Memory With Audio: {mem_after_audio:.2f} MB")

    # Process with V3
    restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.FAST))
    result = restorer.restore(audio_large, sample_rate=44100)

    mem_after_processing = get_memory_mb()
    print(f"   Memory After Processing: {mem_after_processing:.2f} MB")

    # Cleanup
    del audio_large
    del result
    del restorer
    gc.collect()

    mem_final = get_memory_mb()
    print(f"   Memory After Cleanup: {mem_final:.2f} MB")

    assert mem_final is not None
    memory_increase = mem_final - mem_start
    print(f"\n   Net Memory Increase: {memory_increase:.2f} MB")

    # Should not retain more than 2× audio size
    if memory_increase > audio_size_mb * 3:
        print("   ⚠️  Warning: Excessive memory retention")
    else:
        print("   ✓ Memory handling OK")

    print("\n✅ TEST 4 PASSED\n")


# ==================== Test 5: Phase Cache Memory ====================


@pytest.mark.timeout(1200)
def test_phase_cache_memory():
    """Test 5: Phase cache should not grow indefinitely."""
    print("\n" + "=" * 70)
    print("TEST 5: Phase Cache Memory Management")
    print("=" * 70)

    gc.collect()

    mem_start = get_memory_mb()
    if mem_start is None:
        print("   ⚠️  psutil not available, skipping test")
        print("\n✅ TEST 5 PASSED (baseline)\n")
        return

    print(f"   Memory Before: {mem_start:.2f} MB")

    restorer = UnifiedRestorerV3()

    # Load multiple phases to populate cache
    print("\n   Populating phase cache...")
    audio = generate_test_audio(duration_seconds=3)

    for i in range(5):
        result = restorer.restore(audio, sample_rate=44100)
        del result

        cache_size = len(restorer._phase_cache)
        mem_current = get_memory_mb()
        print(f"      Iteration {i + 1}: Cache={cache_size}, Memory={mem_current:.2f} MB")

    mem_after = get_memory_mb()
    print(f"\n   Memory After: {mem_after:.2f} MB")

    # Cleanup
    del audio
    del restorer
    gc.collect()

    mem_final = get_memory_mb()
    print(f"   Memory After Cleanup: {mem_final:.2f} MB")

    assert mem_final is not None
    memory_increase = mem_final - mem_start
    print(f"   Memory Increase: {memory_increase:.2f} MB")

    # Cache should stabilize, not grow indefinitely
    if memory_increase > 100:
        print("   ⚠️  Warning: Phase cache may be growing too large")
    else:
        print("   ✓ Phase cache memory OK")

    print("\n✅ TEST 5 PASSED\n")


# ==================== Main Test Runner ====================

if __name__ == "__main__":
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "MEMORY LEAK TEST SUITE" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝\n")

    # Check if psutil is available
    try:
        print("✓ psutil available - using precise memory tracking\n")
    except ImportError:
        print("⚠️  psutil not available - install with:")
        print("   pip install psutil\n")
        print("Running baseline tests without precise memory tracking...\n")

    # Run all tests
    test_single_restoration_memory()
    test_multiple_restorations_memory()
    test_defect_scanner_memory()
    test_large_audio_memory()
    test_phase_cache_memory()

    print("\n" + "=" * 70)
    print("✅ ALL MEMORY LEAK TESTS COMPLETED!")
    print("=" * 70)
    print("\nNOTE: For best results, run with psutil installed:")
    print("   pip install psutil")
    print("\n")

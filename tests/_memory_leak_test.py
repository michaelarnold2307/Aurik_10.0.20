"""
Memory Leak Test for UnifiedRestorerV3
=======================================

Tests V3 for memory leaks over multiple iterations.

Sprint 1, Week 1 - Memory Leak Detection
Author: AI Development Team
Date: 2026-02-15
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.performance_guard import QualityMode
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available, using gc.get_objects() for memory estimation")


def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # MB
    else:
        # Fallback: count objects
        return len(gc.get_objects()) / 10000  # Rough estimate


def generate_test_audio(duration_seconds, sample_rate=44100):
    """Generate test audio with defects."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))

    # Base: 440 Hz sine
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add defects
    # 1. Clicks (sparse)
    num_clicks = int(duration_seconds * 2)  # 2 clicks/second
    for i in range(num_clicks):
        pos = int(np.random.rand() * len(audio))
        audio[pos : pos + 5] += 0.3 * np.random.randn(5)

    # 2. 60Hz Hum
    audio += 0.04 * np.sin(2 * np.pi * 60 * t)

    # 3. White Noise
    audio += 0.02 * np.random.randn(len(audio))

    return audio


def test_memory_leak(duration_seconds=60, num_iterations=10):
    """
    Test for memory leaks over multiple restoration iterations.

    Args:
        duration_seconds: Audio duration per iteration
        num_iterations: Number of iterations to test
    """
    print("\n" + "=" * 70)
    print("MEMORY LEAK TEST - UnifiedRestorerV3")
    print("=" * 70)
    print(f"Audio Duration: {duration_seconds}s")
    print(f"Iterations: {num_iterations}")
    print(f"Memory Tracking: {'psutil' if PSUTIL_AVAILABLE else 'gc.get_objects()'}")
    print("=" * 70 + "\n")

    # Generate test audio once (reuse for all iterations)
    sr = 44100
    print(f"Generating {duration_seconds}s test audio...")
    audio = generate_test_audio(duration_seconds, sr)
    print(f"✅ Test audio ready: {len(audio)} samples\n")

    # Setup V3
    config = RestorationConfig(mode=QualityMode.FAST, num_cores=2, enforce_3x_rt=True)  # Fast mode for quick testing

    # Baseline memory
    gc.collect()
    time.sleep(0.5)
    baseline_memory = get_memory_usage()

    print(f"Baseline Memory: {baseline_memory:.1f} MB\n")
    print("Starting iterations...")
    print("-" * 70)

    memory_readings = [baseline_memory]
    iteration_times = []

    for i in range(num_iterations):
        iteration_start = time.time()

        # Create new restorer instance
        restorer = UnifiedRestorerV3(config)

        # Restore
        result = restorer.restore(audio, sample_rate=sr)

        # Cleanup
        del restorer
        del result
        gc.collect()

        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)

        # Measure memory
        time.sleep(0.1)  # Let GC settle
        current_memory = get_memory_usage()
        memory_readings.append(current_memory)

        memory_delta = current_memory - baseline_memory

        print(
            f"Iteration {i + 1:2d}/{num_iterations}: "
            f"{iteration_time:5.2f}s, "
            f"Memory: {current_memory:7.1f} MB "
            f"(Δ {memory_delta:+6.1f} MB)"
        )

    print("-" * 70)

    # Analysis
    final_memory = memory_readings[-1]
    total_leak = final_memory - baseline_memory
    avg_leak_per_iteration = total_leak / num_iterations if num_iterations > 0 else 0
    avg_iteration_time = np.mean(iteration_times)

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)
    print(f"Baseline Memory:    {baseline_memory:.1f} MB")
    print(f"Final Memory:       {final_memory:.1f} MB")
    print(f"Total Leak:         {total_leak:+.1f} MB")
    print(f"Leak per Iteration: {avg_leak_per_iteration:+.2f} MB")
    print(f"Avg Iteration Time: {avg_iteration_time:.2f}s")
    print()

    # Memory trend analysis
    if len(memory_readings) > 2:
        # Linear regression to detect trend
        x = np.arange(len(memory_readings))
        y = np.array(memory_readings)
        z = np.polyfit(x, y, 1)
        slope = z[0]

        print(f"Memory Trend (slope): {slope:+.3f} MB/iteration")
        print()

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Thresholds
    LEAK_THRESHOLD_MB = 5.0  # Max 5 MB total leak acceptable
    LEAK_PER_ITERATION_THRESHOLD = 0.5  # Max 0.5 MB per iteration

    if abs(total_leak) < LEAK_THRESHOLD_MB and abs(avg_leak_per_iteration) < LEAK_PER_ITERATION_THRESHOLD:
        print("✅ PASS - No significant memory leak detected")
        print(f"   Total leak {total_leak:+.1f} MB < {LEAK_THRESHOLD_MB} MB threshold")
        print(f"   Per-iteration leak {avg_leak_per_iteration:+.2f} MB < {LEAK_PER_ITERATION_THRESHOLD} MB threshold")
        return 0
    else:
        print("❌ FAIL - Memory leak detected!")
        print(
            f"   Total leak {total_leak:+.1f} MB {'>' if abs(total_leak) >= LEAK_THRESHOLD_MB else '<'} {LEAK_THRESHOLD_MB} MB threshold"
        )
        print(
            f"   Per-iteration leak {avg_leak_per_iteration:+.2f} MB {'>' if abs(avg_leak_per_iteration) >= LEAK_PER_ITERATION_THRESHOLD else '<'} {LEAK_PER_ITERATION_THRESHOLD} MB threshold"
        )
        return 1


def main():
    parser = argparse.ArgumentParser(description="Memory Leak Test for V3")
    parser.add_argument("--duration", type=int, default=60, help="Audio duration in seconds (default: 60)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations (default: 10)")

    args = parser.parse_args()

    if not PSUTIL_AVAILABLE:
        print("⚠️ Warning: psutil not installed, memory measurements may be inaccurate")
        print("   Install with: pip install psutil")
        print()

    exit_code = test_memory_leak(args.duration, args.iterations)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

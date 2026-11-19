import time

import numpy as np
import pytest

from backend.core.core_utils import compute_rms


def benchmark_compute_rms():
    audio = np.random.uniform(-1, 1, 48000 * 60).astype(np.float32)  # 1 Minute Audio
    start = time.perf_counter()
    rms = compute_rms(audio)
    duration = time.perf_counter() - start
    print(f"compute_rms: {duration:.6f} Sekunden für 1min Audio")
    assert np.isfinite(rms)
    assert rms >= 0.0


@pytest.mark.slow
def test_benchmark_compute_rms():
    benchmark_compute_rms()

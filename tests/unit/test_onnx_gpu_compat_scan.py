"""§v10.762-Fortsetzung (2026-09-10): Paritäts-Gate des GPU-Kompatibilitäts-Scans.

Testet _parity_note() aus scripts/onnx_gpu_compat_scan.py mit Fake-Sessions
(kein echtes ONNX nötig): Eine EP-Ausgabe wird nur akzeptiert, wenn sie auf
deterministischen Zufalls-Inputs der CPU-Referenz bis auf rel 1e-3 entspricht.

Motivation: Der SGMSE+-Score-Core lieferte auf ROCm/MIGraphX rel ~3–5 % falsche
Scores, obwohl der Scan ihn zuvor allein nach Geschwindigkeit als "rocm"
eingestuft hatte.
"""

from __future__ import annotations

import numpy as np

from scripts.onnx_gpu_compat_scan import _parity_note


class _FakeSession:
    """Minimaler ORT-Session-Ersatz: gibt vorgefertigte Ausgaben zurück."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = outputs
        self.last_feed: dict[str, np.ndarray] = {}

    def run(self, output_names, input_feed):
        self.last_feed = dict(input_feed)
        return self._outputs


def test_parity_ok_returns_empty() -> None:
    _out = np.full((2, 3), 1.0, dtype=np.float32)
    _cpu = _FakeSession([_out.copy()])
    _gpu = _FakeSession([_out.copy() + 1e-6])  # float32-Kernel-Rauschen
    _inputs = {"x_t": np.zeros((2, 3), dtype=np.float32)}
    assert _parity_note(_cpu, _gpu, _inputs) == ""


def test_divergence_reported_with_rel() -> None:
    _cpu = _FakeSession([np.full((4,), 1.0, dtype=np.float32)])
    _gpu = _FakeSession([np.full((4,), 1.05, dtype=np.float32)])
    _inputs = {"x_t": np.zeros((4,), dtype=np.float32)}
    assert _parity_note(_cpu, _gpu, _inputs).startswith("EP-Numerik weicht ab")


def test_gpu_nan_reported_when_cpu_finite() -> None:
    _cpu = _FakeSession([np.full((4,), 1.0, dtype=np.float32)])
    _gpu = _FakeSession([np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)])
    _inputs = {"x_t": np.zeros((4,), dtype=np.float32)}
    assert _parity_note(_cpu, _gpu, _inputs) == "EP liefert NaN/Inf (CPU endlich)"


def test_cpu_nan_gives_no_verdict() -> None:
    _cpu = _FakeSession([np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)])
    _gpu = _FakeSession([np.zeros((4,), dtype=np.float32)])
    _inputs = {"x_t": np.zeros((4,), dtype=np.float32)}
    assert _parity_note(_cpu, _gpu, _inputs) == ""


def test_random_inputs_deterministic() -> None:
    _cpu = _FakeSession([np.zeros((4,), dtype=np.float32)])
    _gpu = _FakeSession([np.zeros((4,), dtype=np.float32)])
    _inputs = {"x_t": np.zeros((4,), dtype=np.float32)}
    _parity_note(_cpu, _gpu, _inputs)
    _feed1 = _cpu.last_feed["x_t"]
    _parity_note(_cpu, _gpu, _inputs)
    _feed2 = _cpu.last_feed["x_t"]
    assert np.any(_feed1 != 0.0)  # Zufalls-Input statt Null-Dummy
    assert np.array_equal(_feed1, _feed2)  # Seed 0 → deterministisch


def test_int64_only_gives_no_verdict() -> None:
    _cpu = _FakeSession([np.zeros((2,), dtype=np.float32)])
    _gpu = _FakeSession([np.zeros((2,), dtype=np.float32)])
    _inputs = {"audio_length": np.zeros((2,), dtype=np.int64)}
    assert _parity_note(_cpu, _gpu, _inputs) == ""

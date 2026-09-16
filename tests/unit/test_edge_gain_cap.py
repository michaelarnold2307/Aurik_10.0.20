"""Unit-Tests für backend/core/dsp/edge_gain_cap.py (§2.46f Edge-Gain-Cap).

Der Cap skaliert Intro-/Outro-Zonen auf max(original × 10^(+2 dB/20))
zurück — nur Absenkung, 50-ms-Crossfade an der Innenkante, layout-sicher,
deterministisch, §V6-fail-open.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.edge_gain_cap import apply_edge_gain_cap


def _zone_rms(audio: np.ndarray, n: int, side: str) -> float:
    if audio.ndim == 1:
        z = audio[:n] if side == "start" else audio[-n:]
    elif audio.shape[0] <= 2 and audio.shape[1] > 2:  # (C, N)
        z = audio[:, :n] if side == "start" else audio[:, -n:]
    else:
        z = audio[:n, :] if side == "start" else audio[-n:, :]
    return float(np.sqrt(np.mean(np.asarray(z, dtype=np.float64) ** 2) + 1e-15))


@pytest.mark.unit
class TestEdgeGainCap:
    def test_mono_overboost_is_capped(self) -> None:
        sr = 48000
        n = sr * 3
        t = np.linspace(0, 3, n, endpoint=False, dtype=np.float32)
        ref = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        proc = ref.copy()
        proc[: sr // 2] *= 1.6  # +4,1 dB Intro-Boost
        out = apply_edge_gain_cap(ref, proc, sr)
        n_edge = int(0.5 * sr)
        ratio = _zone_rms(out, n_edge, "start") / _zone_rms(ref, n_edge, "start")
        assert ratio <= 1.26 + 1e-3
        # Mittelteil bleibt unangetastet (nur Randzone wird gekappt)
        assert np.array_equal(out[n_edge + 1000 : -n_edge - 1000], proc[n_edge + 1000 : -n_edge - 1000])

    def test_no_overboost_is_passthrough(self) -> None:
        sr = 48000
        n = sr * 3
        ref = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n, dtype=np.float32) / sr)).astype(np.float32)
        proc = ref.copy()
        out = apply_edge_gain_cap(ref, proc, sr)
        assert np.array_equal(out, proc)

    def test_quiet_edges_never_boosted(self) -> None:
        sr = 48000
        n = sr * 3
        ref = np.zeros(n, dtype=np.float32)
        ref[sr : 2 * sr] = 0.2
        proc = ref.copy()
        proc[sr : 2 * sr] *= 1.6  # Boost nur in der Mitte — Ränder bleiben still
        out = apply_edge_gain_cap(ref, proc, sr)
        assert np.array_equal(out, proc)  # nichts zu kappen, NIE Anhebung

    def test_channels_first_layout(self) -> None:
        sr = 48000
        n = sr * 3
        t = np.linspace(0, 3, n, endpoint=False, dtype=np.float32)
        mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        ref = np.stack([mono, mono * 0.98])
        proc = ref.copy()
        proc[:, : sr // 2] *= 1.6
        out = apply_edge_gain_cap(ref, proc, sr)
        n_edge = int(0.5 * sr)
        assert _zone_rms(out, n_edge, "start") <= _zone_rms(ref, n_edge, "start") * 1.26 + 1e-6

    def test_channels_last_layout(self) -> None:
        sr = 48000
        n = sr * 3
        t = np.linspace(0, 3, n, endpoint=False, dtype=np.float32)
        mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        ref = np.stack([mono, mono * 0.98], axis=1)
        proc = ref.copy()
        proc[: sr // 2, :] *= 1.6
        out = apply_edge_gain_cap(ref, proc, sr)
        n_edge = int(0.5 * sr)
        assert _zone_rms(out, n_edge, "start") <= _zone_rms(ref, n_edge, "start") * 1.26 + 1e-6

    def test_outro_overboost_is_capped(self) -> None:
        sr = 48000
        n = sr * 3
        t = np.linspace(0, 3, n, endpoint=False, dtype=np.float32)
        ref = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        proc = ref.copy()
        proc[-sr // 2 :] *= 1.8
        out = apply_edge_gain_cap(ref, proc, sr)
        n_edge = int(0.5 * sr)
        assert _zone_rms(out, n_edge, "end") <= _zone_rms(ref, n_edge, "end") * 1.26 + 1e-6

    def test_determinism(self) -> None:
        sr = 48000
        rng = np.random.default_rng(7)
        ref = rng.standard_normal(sr * 3).astype(np.float32) * 0.1
        proc = ref.copy()
        proc[: sr // 2] *= 1.7
        a = apply_edge_gain_cap(ref, proc, sr)
        b = apply_edge_gain_cap(ref, proc, sr)
        assert np.array_equal(a, b)

    def test_shape_mismatch_is_passthrough(self) -> None:
        ref = np.zeros((4800, 2), dtype=np.float32)
        proc = np.zeros(4800, dtype=np.float32)
        out = apply_edge_gain_cap(ref, proc, 48000)
        assert np.array_equal(out, proc)

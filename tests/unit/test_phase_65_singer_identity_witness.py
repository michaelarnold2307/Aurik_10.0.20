"""Unit-Tests für den Sänger-Identitäts-Witness in phase_65 (§SOTA-P65, S4-Muster).

Der Witness ist ein Per-Phase-Gate: Resemblyzer cos(pre, post) ≥ 0,92
(Hörordnung Ebene 1); darunter wird proportional Richtung Input geblendet.
Zeuge, kein Richter (Hörordnung §8a): ohne Modell/Embedding bleibt das
DSP-Ergebnis unverändert (§V6 (copilot-instructions.md)-non-blocking).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_65_vocal_naturalness_restoration import (
    _SINGER_IDENTITY_MIN_COS_65,
    _apply_singer_identity_witness,
)


class _FakeEmbedding:
    def __init__(self, value: float) -> None:
        self.value = value

    def __add__(self, other: object) -> _FakeEmbedding:
        if not isinstance(other, _FakeEmbedding):
            return NotImplemented
        return _FakeEmbedding(self.value + other.value)

    def __mul__(self, scalar: float) -> _FakeEmbedding:
        return _FakeEmbedding(self.value * scalar)

    def __truediv__(self, scalar: float) -> _FakeEmbedding:
        return _FakeEmbedding(self.value / scalar)

    def dot(self, other: _FakeEmbedding) -> float:
        return self.value * other.value


class _FakePlugin:
    def __init__(self, available: bool, cos: float, emb_fail: bool = False) -> None:
        self._available = available
        self._cos = cos
        self._emb_fail = emb_fail
        self.embedded: list[np.ndarray] = []

    def available(self) -> bool:
        return self._available

    def embed(self, audio: np.ndarray, sr: int) -> _FakeEmbedding | None:
        self.embedded.append(audio)
        if self._emb_fail:
            return None
        return _FakeEmbedding(1.0)

    def cosine_similarity(self, a: _FakeEmbedding, b: _FakeEmbedding) -> float:
        return self._cos


@pytest.mark.unit
class TestSingerIdentityWitness:
    def test_passthrough_above_threshold(self) -> None:
        plugin = _FakePlugin(available=True, cos=0.95)
        pre = np.full((4800, 2), 0.01, dtype=np.float32)
        post = np.full((4800, 2), 0.02, dtype=np.float32)
        out, meta = _apply_singer_identity_witness(pre, post, 48000, _plugin_getter=lambda: plugin)
        assert np.array_equal(out, post)
        assert meta["singer_identity_cosine"] == pytest.approx(0.95, abs=1e-3)
        assert "singer_identity_blend" not in meta

    def test_blend_toward_input_below_threshold(self) -> None:
        cos = 0.80
        plugin = _FakePlugin(available=True, cos=cos)
        pre = np.full((4800, 2), 0.01, dtype=np.float32)
        post = np.full((4800, 2), 0.03, dtype=np.float32)
        out, meta = _apply_singer_identity_witness(pre, post, 48000, _plugin_getter=lambda: plugin)
        expected_blend = float(np.clip(cos / _SINGER_IDENTITY_MIN_COS_65, 0.1, 0.8))
        expected = expected_blend * post + (1.0 - expected_blend) * pre
        assert np.allclose(out, expected, atol=1e-6)
        assert meta["singer_identity_blend"] == pytest.approx(expected_blend, abs=1e-3)
        assert meta["singer_identity_cosine"] == pytest.approx(cos, abs=1e-3)

    def test_unavailable_plugin_is_passthrough(self) -> None:
        plugin = _FakePlugin(available=False, cos=0.1)
        pre = np.zeros((4800,), dtype=np.float32)
        post = np.ones((4800,), dtype=np.float32)
        out, meta = _apply_singer_identity_witness(pre, post, 48000, _plugin_getter=lambda: plugin)
        assert np.array_equal(out, post)
        assert meta == {}

    def test_failed_embedding_is_passthrough(self) -> None:
        plugin = _FakePlugin(available=True, cos=0.1, emb_fail=True)
        pre = np.zeros((4800,), dtype=np.float32)
        post = np.ones((4800,), dtype=np.float32)
        out, meta = _apply_singer_identity_witness(pre, post, 48000, _plugin_getter=lambda: plugin)
        assert np.array_equal(out, post)
        assert meta == {}

    def test_plugin_exception_is_non_blocking(self) -> None:
        class _BoomPlugin:
            def available(self) -> bool:  # pragma: no cover - trivial
                raise RuntimeError("kaputt")

        pre = np.zeros((4800,), dtype=np.float32)
        post = np.ones((4800,), dtype=np.float32)
        out, meta = _apply_singer_identity_witness(pre, post, 48000, _plugin_getter=lambda: _BoomPlugin())
        assert np.array_equal(out, post)
        assert meta == {}

    def test_channels_last_layout(self) -> None:
        """Stereo (N,2) wird layout-sicher zu Mono gemittelt (§Stereo-Layout-Invariante)."""
        plugin = _FakePlugin(available=True, cos=0.98)
        pre = np.zeros((4800, 2), dtype=np.float32)
        post = np.ones((4800, 2), dtype=np.float32)
        out, _ = _apply_singer_identity_witness(pre, post, 48000, _plugin_getter=lambda: plugin)
        assert np.array_equal(out, post)
        assert plugin.embedded[0].ndim == 1

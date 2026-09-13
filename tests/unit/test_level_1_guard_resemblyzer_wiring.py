"""Unit-Tests für die Resemblyzer-Witness-Verdrahtung im Ebene-1-Guard.

§Ebene-1 (Hörordnung): singer_identity_cosine ≥ 0.92. Der Guard maß die
Stimm-Identität bis 2026-09-13 über einen Import des nicht existierenden
Pakets `Resemblyzer` (Großbuchstabe) — der ML-Pfad lief nie, stiller
DSP-Ersatzpfad (§V74 (VERBOTEN.md)-Verstoß). Fix: Plugin-Kaskade
plugins/resemblyzer_plugin.py (Package→ONNX→None) als primäre Methode.

Die restlichen Ebene-1-Invarianten und Schwellwerte deckt die bestehende
Suite tests/unit/test_level_1_invariants_guard.py ab (unverändert).

Abgedeckt hier: Plugin-Witness-Pfad (Kosinus), DSP-Ersatzpfad ohne Plugin,
Blend-Auslösung bei Identitätsverletzung, Determinismus (§G5 (GEBOTE.md)),
echter ONNX-Witness (identisches Audio ⇒ cos=1.0).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import plugins.resemblyzer_plugin as rp
from backend.core.dsp.level_1_invariants_guard import Level1InvariantsGuard, check_level_1_invariants


class _FakeResemblyzerPlugin:
    """Feste Embeddings. mode="diff": pre→ones, post→orthogonal (cos≈0.0625);
    mode="same": beide Aufrufe → ones (cos=1.0)."""

    available = True

    def __init__(self, mode: str = "diff") -> None:
        self._calls = 0
        self._mode = mode
        self._pre: np.ndarray = np.ones(256, dtype=np.float32)
        self._post: np.ndarray = np.eye(256, dtype=np.float32)[0]

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self._calls >= 2:  # Paar-Grenze: jeder check() ruft genau 2× embed
            self._calls = 0
        self._calls += 1
        if self._mode == "same":
            return self._pre.copy()
        return self._pre.copy() if self._calls == 1 else self._post.copy()

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        return float(np.dot(a, b) / (na * nb + 1e-12))


def _guard_with(fake: object | None) -> Level1InvariantsGuard:
    """Guard ohne __init__ (Singleton-Probe übersprungen), Plugin injiziert."""
    g = Level1InvariantsGuard.__new__(Level1InvariantsGuard)
    g._resemblyzer_plugin = fake
    return g


def _noise_1s(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(48000).astype(np.float32) * 0.05


def test_singer_identity_uses_plugin_cosine() -> None:
    """Plugin-Witness: Kosinus der zwei Embeddings wird zurückgegeben (0.0625)."""
    fake = _FakeResemblyzerPlugin()
    g = _guard_with(fake)
    score = g._measure_singer_identity(_noise_1s(1), _noise_1s(2), 48000, None)
    assert score == pytest.approx(1.0 / np.sqrt(256.0), abs=1e-4)
    assert fake._calls == 2


def test_singer_identity_ersatzpfad_ohne_plugin() -> None:
    """Ohne Plugin: DSP-Ersatzpfad (MFCC/Centroid) liefert Wert in [0,1]."""
    g = _guard_with(None)
    score = g._measure_singer_identity(_noise_1s(3), _noise_1s(4), 48000, None)
    assert 0.0 <= score <= 1.0


def test_check_blend_triggers_on_identity_violation() -> None:
    """cos≈0.06 < 0.92 ⇒ violated_invariants enthält singer_identity, blend < 1."""
    g = _guard_with(_FakeResemblyzerPlugin())
    res = g.check(_noise_1s(5), _noise_1s(6), 48000, None)
    assert "singer_identity" in res.violated_invariants
    assert res.blend_factor < 1.0
    assert res.singer_identity < 0.92


def test_check_identical_audio_keeps_blend_one() -> None:
    """Identisches pre/post (cos=1.0) ⇒ keine singer_identity-Verletzung."""
    fake = _FakeResemblyzerPlugin(mode="same")
    x = _noise_1s(7)
    g = _guard_with(fake)
    res = g.check(x, x, 48000, None)
    assert "singer_identity" not in res.violated_invariants
    assert res.singer_identity == pytest.approx(1.0, abs=1e-4)
    assert res.blend_factor == pytest.approx(1.0)


def test_check_deterministic() -> None:
    """Gleicher Input ⇒ identisches Ergebnis (§G5 (GEBOTE.md))."""
    g = _guard_with(_FakeResemblyzerPlugin())
    a, b = _noise_1s(8), _noise_1s(9)
    r1 = g.check(a, b, 48000, None)
    r2 = g.check(a, b, 48000, None)
    assert r1.singer_identity == r2.singer_identity
    assert r1.blend_factor == r2.blend_factor
    assert r1.violated_invariants == r2.violated_invariants


def test_check_level_1_convenience_function() -> None:
    """check_level_1_invariants (Singleton-Wrapper) läuft mit 48-kHz-Eingabe."""
    x = _noise_1s(10)
    res = check_level_1_invariants(x, x, 48000, None)
    assert res.singer_identity is not None
    assert 0.0 <= res.singer_identity <= 1.0


@pytest.mark.skipif(
    rp._ort is None or not os.path.exists(rp._ONNX_MODEL_PATH), reason="resemblyzer_voice_encoder.onnx fehlt"
)
def test_real_onnx_witness_identical_audio_is_one() -> None:
    """Echter ONNX-Witness: identisches Audio ⇒ cosine ≈ 1.0 (Paritäts-Anker)."""
    from plugins.resemblyzer_plugin import get_resemblyzer_plugin

    g = _guard_with(get_resemblyzer_plugin())
    x = _noise_1s(11)
    score = g._measure_singer_identity(x, x, 48000, None)
    assert score == pytest.approx(1.0, abs=1e-3)

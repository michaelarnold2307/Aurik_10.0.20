"""Unit-Tests für plugins/resemblyzer_plugin.py — ONNX-Fallback-Pfad (2026-09-13).

§2.35c (copilot-instructions.md) verlangt singer_identity_cosine VOR/NACH der
Pipeline; das Resemblyzer-Package ist in manchen Umgebungen nicht importierbar
(ModuleNotFoundError ohne webrtcvad) — deshalb der ONNX-Pfad
(models/resemblyzer/resemblyzer_voice_encoder.onnx, opset 17, Parität cos=1.0000).

Abgedeckt: Routing Package→ONNX, Mel-Parameter (400/160/40), VAD-Trim-Verhalten,
L2-Norm, Layout-Sicherheit ((C,N)/(N,C)/mono), Determinismus (§G5 (GEBOTE.md)),
Fehlerfälle ohne Crash (§V6 (copilot-instructions.md)-Fallback = None).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import plugins.resemblyzer_plugin as rp

# ─── Pure Logik (läuft immer, ohne Modell) ───────────────────────────────────


def _make_plugin(onnx_session: object | None = None) -> rp.ResemblyzerPlugin:
    """Plugin-Instanz ohne _load() — encoder leer, ONNX-Session injiziert."""
    p = rp.ResemblyzerPlugin.__new__(rp.ResemblyzerPlugin)
    p._encoder = None
    p._preprocess_wav_fn = None
    p._onnx_session = onnx_session
    return p


class _FakeOnnxSession:
    """Captured den Mel-Input und liefert einen festen 256-dim Output."""

    def __init__(self, out: np.ndarray | None = None) -> None:
        self.last_input: dict[str, np.ndarray] | None = None
        self._out = out if out is not None else np.ones(256, dtype=np.float32)

    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_input = inputs
        return [self._out[np.newaxis, :]]


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((16000,), 16000),
        ((2, 16000), 16000),  # channels-first (C, N)
        ((16000, 2), 16000),  # samples-first (N, C)
        ((1, 16000), 16000),
    ],
)
def test_to_mono_layouts(shape: tuple[int, ...], expected: int) -> None:
    """_to_mono bedient 1-D, (C,N), (N,C) und (1,N) — keine Stereo-Kollaps (§AGENTS Stereo-Layout)."""
    x = np.random.RandomState(0).randn(*shape).astype(np.float32)
    out = rp._to_mono(x)
    assert out.shape == (expected,)


def test_embed_routes_to_onnx_when_package_missing() -> None:
    """Ohne Package-Encoder + mit ONNX-Session läuft _embed_onnx (Routing)."""
    fake = _FakeOnnxSession()
    p = _make_plugin(fake)
    assert p.available is True
    rng = np.random.RandomState(1)
    audio = rng.randn(32000).astype(np.float32) * 0.1  # 2 s @ 16 kHz
    emb = p.embed(audio, 16000)
    assert emb is not None
    assert emb.shape == (256,)
    assert fake.last_input is not None
    assert fake.last_input["mels"].ndim == 3
    assert fake.last_input["mels"].shape[0] == 1
    assert fake.last_input["mels"].shape[2] == 40  # n_mels
    # Zeitachse: 2 s @ 16 kHz → 32000 Samples, hop 160 → ~199 Frames
    assert 190 <= fake.last_input["mels"].shape[1] <= 210


def test_embed_onnx_l2_normalized_and_deterministic() -> None:
    """ONNX-Ausgabe wird L2-normiert; identischer Input ⇒ identischer Output (§G5 (GEBOTE.md))."""
    raw = np.array([3.0, 4.0] + [0.0] * 254, dtype=np.float32)
    fake = _FakeOnnxSession(out=raw)
    p = _make_plugin(fake)
    rng = np.random.RandomState(2)
    audio = rng.randn(16000).astype(np.float32) * 0.1
    e1 = p._embed_onnx(audio, 16000)
    e2 = p._embed_onnx(audio, 16000)
    assert e1 is not None and e2 is not None
    assert float(np.linalg.norm(e1)) == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(e1, e2)
    assert np.all(np.isfinite(e1))


def test_embed_onnx_silence_returns_none() -> None:
    """Stille (kein VAD-Treffer) ⇒ None — Aufrufer nutzt DSP-Fallback (§3.1)."""
    fake = _FakeOnnxSession()
    p = _make_plugin(fake)
    assert p._embed_onnx(np.zeros(48000, dtype=np.float32), 16000) is None


def test_embed_onnx_too_short_returns_none() -> None:
    """Kürzer als ein VAD-Frame (480 Samples) ⇒ None statt Crash."""
    fake = _FakeOnnxSession()
    p = _make_plugin(fake)
    rng = np.random.RandomState(3)
    assert p._embed_onnx(rng.randn(100).astype(np.float32), 16000) is None


def test_embed_without_any_backend_returns_none() -> None:
    """Weder Package noch ONNX ⇒ None, kein Crash (§V6 (copilot-instructions.md))."""
    p = _make_plugin(None)
    assert p.available is False
    rng = np.random.RandomState(4)
    assert p.embed(rng.randn(16000).astype(np.float32), 16000) is None


def test_embed_onnx_nan_inf_cleaned() -> None:
    """NaN/Inf im Eingangssignal werden zu 0.0 bereinigt (§3.1)."""
    fake = _FakeOnnxSession()
    p = _make_plugin(fake)
    rng = np.random.RandomState(5)
    audio = rng.randn(16000).astype(np.float32)
    audio[::100] = np.nan
    audio[1::100] = np.inf
    emb = p._embed_onnx(audio, 16000)
    assert emb is not None
    assert np.all(np.isfinite(emb))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.ones(256, dtype=np.float32), np.ones(256, dtype=np.float32)),  # identisch → 1.0
        (np.eye(256, dtype=np.float32)[0], np.eye(256, dtype=np.float32)[1]),  # orthogonal → 0.0
    ],
)
def test_cosine_similarity_bounds(a: np.ndarray, b: np.ndarray) -> None:
    """Kosinus ∈ [0, 1], NaN-sicher."""
    p = _make_plugin(None)
    sim = p.cosine_similarity(a, b)
    assert 0.0 <= sim <= 1.0


def test_cosine_similarity_nan_safe() -> None:
    p = _make_plugin(None)
    bad = np.full(256, np.nan, dtype=np.float32)
    sim = p.cosine_similarity(bad, np.ones(256, dtype=np.float32))
    assert sim == 0.0  # Denom-Guard + clip


# ─── Echter ONNX-Pfad (nur wenn das Modell lokal liegt) ──────────────────────


@pytest.mark.skipif(
    rp._ort is None or not os.path.exists(rp._ONNX_MODEL_PATH), reason="resemblyzer_voice_encoder.onnx fehlt"
)
def test_real_onnx_smoke_16k() -> None:
    """Echtes ONNX: 256-dim, L2-normiert, deterministisch auf 16-kHz-Noise."""
    rng = np.random.RandomState(6)
    audio = rng.randn(48000).astype(np.float32) * 0.05  # 3 s
    p = _make_plugin(rp._ort.InferenceSession(rp._ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]))
    e1 = p._embed_onnx(audio, 16000)
    e2 = p._embed_onnx(audio, 16000)
    assert e1 is not None and e2 is not None
    assert e1.shape == (256,)
    assert float(np.linalg.norm(e1)) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(e1, e2)


@pytest.mark.skipif(
    rp._ort is None or not os.path.exists(rp._ONNX_MODEL_PATH), reason="resemblyzer_voice_encoder.onnx fehlt"
)
def test_real_onnx_smoke_44k1_resample() -> None:
    """Echtes ONNX mit 44,1-kHz-Eingang (librosa-Resample-Zweig)."""
    rng = np.random.RandomState(7)
    audio = rng.randn(44100 * 2).astype(np.float32) * 0.05
    p = _make_plugin(rp._ort.InferenceSession(rp._ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]))
    emb = p._embed_onnx(audio, 44100)
    assert emb is not None
    assert emb.shape == (256,)
    assert np.all(np.isfinite(emb))

"""Unit-Tests für den musiknativen Vokal-Identitäts-Zeugen (KP-3, Pfad A).

Vertrag (§III.11 (copilot-instructions.md), Hörordnung §8): der MuQ-MuLan-Zeuge
ist report-only — er misst nur den Kosinus der musiktrainierten Embeddings und
übt NIE ein Veto aus, flippt keine Flags und ändert keine Schwelle. Die Tests
prüfen Determinismus (§G5 (copilot-instructions.md)), Fallback-Pfade (§V6
(copilot-instructions.md)), Stereo-Layout-Handling (§11 Spec 02) und die
Kosinus-Schwellen-Logik — ohne `try/except: pass`-Hüllen (Lehre D-K3-20).
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.music_identity_oracle as mio


def _sine(n: int = 4000, freq: float = 440.0, sr: int = 48000) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_downmix_channels_first_and_last() -> None:
    """Stereo-Layout-Invariante: (2, N) und (N, 2) liefern dasselbe Mono."""
    n = 1000
    left = _sine(n)
    right = 0.5 * np.sin(2 * np.pi * 880.0 * np.linspace(0, n / 48000, n, endpoint=False)).astype(np.float32)
    cf = np.stack([left, right])  # (2, N)
    cl = np.stack([left, right], axis=1)  # (N, 2)
    np.testing.assert_allclose(mio._downmix(cf), mio._downmix(cl), rtol=1e-6, atol=1e-6)


def test_downmix_mono_identity() -> None:
    mono = _sine()
    np.testing.assert_allclose(mio._downmix(mono), mono, rtol=1e-7)


def test_downmix_nan_inf_guarded() -> None:
    """§0a NaN/Inf-Schutz: nicht-endliche Samples werden neutralisiert."""
    x = np.zeros(1000, dtype=np.float32)
    x[10] = np.nan
    x[20] = np.inf
    x[30] = -np.inf
    assert np.all(np.isfinite(mio._downmix(x)))


def test_cosine_identical_and_orthogonal() -> None:
    a = np.ones(256, dtype=np.float32)
    b = np.eye(256, dtype=np.float32)[0]
    assert mio._cosine(a, a) == 1.0
    assert mio._cosine(a, b) == pytest.approx(1.0 / np.sqrt(256.0), abs=1e-4)


def test_cosine_zero_norm_returns_none() -> None:
    z = np.zeros(256, dtype=np.float32)
    assert mio._cosine(z, z) is None


def test_music_identity_short_signal_returns_none() -> None:
    """Kurzes Signal ⇒ None + Begründung (§V6 (copilot-instructions.md), kein Absturz)."""
    x = np.zeros(100, dtype=np.float32)
    cosine, reasons = mio.music_identity_cosine(x, x, 48000)
    assert cosine is None
    assert any("kurz" in r for r in reasons)


def test_music_identity_const_signal_returns_none() -> None:
    """Konstantes Signal ⇒ None + Begründung (kein Messwert, kein Eingriff)."""
    x = np.full(1000, 0.5, dtype=np.float32)
    cosine, reasons = mio.music_identity_cosine(x, x, 48000)
    assert cosine is None
    assert any("konstant" in r for r in reasons)


def test_music_identity_missing_model_returns_none(monkeypatch) -> None:
    """Modell/Embedding fehlt ⇒ None + Warnung, keine Exception (§V6 (copilot-instructions.md), Lehre D-K3-20)."""

    def _fake_embedding(_audio, _sr):
        return None

    monkeypatch.setattr(mio, "_muq_embedding", _fake_embedding)
    x = _sine()
    cosine, reasons = mio.music_identity_cosine(x, x, 48000)
    assert cosine is None
    assert any("pre-Embedding" in r for r in reasons)


def test_music_identity_deterministic_with_fake_embeddings(monkeypatch) -> None:
    """§G5 (copilot-instructions.md): gleicher Input ⇒ gleicher Kosinus (deterministische Fake-Embeddings)."""

    calls = {"n": 0}

    def _fake_embedding(audio, _sr):
        # Deterministisch: pre→ones, post→(ones mit leichtem Offset) je Aufruf-Reihenfolge.
        calls["n"] += 1
        v = np.ones(1024, dtype=np.float32)
        if calls["n"] % 2 == 0:  # zweiter Aufruf (post) leicht verschieden
            v = v.copy()
            v[0] = 0.0
        return v

    monkeypatch.setattr(mio, "_muq_embedding", _fake_embedding)
    x = _sine()
    c1, _ = mio.music_identity_cosine(x, x, 48000)
    c2, _ = mio.music_identity_cosine(x, x, 48000)
    assert c1 is not None
    assert c1 == c2


def test_measure_and_log_smoke(monkeypatch) -> None:
    """report-only Convenience läuft ohne Veto/Flag-Mutation durch."""

    def _fake_embedding(_audio, _sr):
        return np.ones(1024, dtype=np.float32)

    monkeypatch.setattr(mio, "_muq_embedding", _fake_embedding)
    x = _sine()
    cosine, _ = mio.measure_and_log_music_identity(x, x, 48000, stage="level1_guard")
    assert cosine == pytest.approx(1.0)

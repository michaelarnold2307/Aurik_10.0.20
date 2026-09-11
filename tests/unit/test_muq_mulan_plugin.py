"""Unit-Tests für plugins/muq_mulan_plugin.py (§MuQ-MuLan-SOTA Voranalyse, 2025).

Ohne ONNX-Datei laufen die Fallback-Tests (None-Rückgaben, §V6 (copilot-instructions.md)).
Mit models/muq_mulan/muq_mulan.onnx wird der echte Session-Pfad geprüft
(Determinismus, 768-d, Witness 0–100).
"""

from __future__ import annotations

import numpy as np
import pytest

import plugins.muq_mulan_plugin as mm

# ─── Fallback-Pfad (läuft immer) ─────────────────────────────────────────────


def test_fallback_without_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ohne ONNX liefern alle Schätzer None — kein stiller Crash (§V6 (copilot-instructions.md))."""
    monkeypatch.setattr(mm, "_ONNX_PATH", mm._PROJECT_ROOT / "models" / "muq_mulan" / "does_not_exist.onnx")
    monkeypatch.setattr(mm, "_get_session", lambda: None)

    assert mm.is_available() is False
    silence = np.zeros(48000, dtype=np.float32)
    assert mm.extract_muq_mulan_embedding(silence, 48000) is None
    assert mm.estimate_muq_mulan_witness(silence, 48000) is None


def test_witness_with_fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Witness-Mapping (Kosinus → [0,100]) deterministisch mit Fake-Embeddings."""
    rng = np.random.RandomState(3)
    emb = rng.randn(768).astype(np.float32)
    refs = rng.randn(4, 768).astype(np.float32)
    monkeypatch.setattr(mm, "extract_muq_mulan_embedding", lambda a, sr: emb)
    monkeypatch.setattr(mm, "_get_ref_embeddings", lambda: refs)

    w1 = mm.estimate_muq_mulan_witness(np.zeros(48000, dtype=np.float32), 48000)
    w2 = mm.estimate_muq_mulan_witness(np.zeros(48000, dtype=np.float32), 48000)

    assert w1 is not None
    assert 0.0 <= w1 <= 100.0
    assert w1 == pytest.approx(w2)


def test_to_mono_24k_deterministic() -> None:
    """_to_mono_24k: deterministisch, 24 kHz, zentriertes 10-s-Fenster."""
    sr = 24000
    x = np.arange(sr * 15, dtype=np.float32)  # 15 s mono
    c1 = mm._to_mono_24k(x, sr)
    c2 = mm._to_mono_24k(x, sr)
    assert len(c1) == sr * 10
    assert np.array_equal(c1, c2)
    # Zentrum: (15 s − 10 s) / 2 = 2.5 s → Start bei 2.5 × 24000 = 60000
    assert c1[0] == 60000.0


# ─── Echter ONNX-Pfad (nur wenn exportierte Datei vorhanden) ─────────────────


@pytest.mark.skipif(
    not mm.is_available(),
    reason="models/muq_mulan/muq_mulan.onnx fehlt — Export-Skript noch nicht gelaufen",
)
def test_onnx_embedding_deterministic_and_shape() -> None:
    """Echtes ONNX: deterministisch (§G5 (GEBOTE.md)) und 768-d Embedding."""
    rng = np.random.RandomState(4)
    audio = rng.randn(2, 48000).astype(np.float32) * 0.1

    e1 = mm.extract_muq_mulan_embedding(audio, 48000)
    e2 = mm.extract_muq_mulan_embedding(audio, 48000)

    assert e1 is not None and e2 is not None
    assert e1.shape == (768,)
    # CPU-Pfad ist bit-identisch (§G5 (GEBOTE.md)); ROCm ist bekannt nicht-deterministisch.
    assert np.array_equal(e1, e2)
    assert np.all(np.isfinite(e1))

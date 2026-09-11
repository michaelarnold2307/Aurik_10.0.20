"""Unit-Tests für plugins/muq_plugin.py (§MuQ-SOTA Voranalyse, 2025).

Ohne Checkpoint laufen die Fallback-Tests (kein Crash, None-Rückgaben, §V6 (copilot-instructions.md)).
Mit lokalem MuQ-Checkpoint (OpenMuQ/MuQ-large-msd-iter im HF-Cache) werden
Determinismus (§G5 (GEBOTE.md)), Shape und Wertebereiche gegen das echte Modell geprüft.
"""

from __future__ import annotations

import numpy as np
import pytest

import plugins.muq_plugin as mq

# ─── Fallback-Pfad (läuft immer, auch in CI ohne Modell) ─────────────────────


def test_fallback_without_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ohne Checkpoint liefern alle Schätzer None — kein stiller Crash (§V6 (copilot-instructions.md))."""
    monkeypatch.setattr(mq, "_find_checkpoint_dir", lambda: None)
    monkeypatch.setattr(mq, "get_muq_model", lambda: None)

    assert mq.is_available() is False
    silence = np.zeros(48000, dtype=np.float32)
    assert mq.extract_embedding(silence, 48000) is None
    assert mq.estimate_quality_witness(silence, 48000) is None
    assert mq.estimate_muq_mos(silence, 48000) is None


@pytest.mark.parametrize(
    ("sim", "expected"),
    [
        (0.0, 0.0),
        (0.25, 25.0),
        (0.5, 50.0),
        (0.75, 75.0),
        (1.0, 100.0),
    ],
)
def test_sim_to_witness_mapping(sim: float, expected: float) -> None:
    """Witness-Mapping ist linear über [0,1] und an den Rändern gesättigt."""
    assert mq._sim_to_witness(sim) == pytest.approx(expected, abs=1e-6)


def test_center_window_deterministic() -> None:
    """Zentrumsfenster ist deterministisch und korrekt positioniert."""
    x = np.arange(100, dtype=np.float32)
    w1 = mq._center_window(x, 2)  # n_max = 20 s × 2 Hz = 40 Samples
    w2 = mq._center_window(x, 2)
    assert len(w1) == 40
    assert np.array_equal(w1, w2)
    assert w1[0] == 30  # Mitte (50) − 20


def test_estimator_blends_muq_mos(monkeypatch: pytest.MonkeyPatch) -> None:
    """RestorabilityEstimator blendet den gelernten MOS 50/50 (§MuQ-SOTA)."""
    from backend.core.restorability_estimator import RestorabilityEstimator

    est = RestorabilityEstimator()
    audio = np.random.RandomState(1).randn(48000).astype(np.float32) * 0.1

    monkeypatch.setattr(mq, "estimate_quality_witness", lambda a, sr: 80.0)
    monkeypatch.setattr(mq, "estimate_muq_mos", lambda a, sr: 4.2)

    res = est.estimate(audio, 48000, material="vinyl")

    dsp_mos = est._score_to_mos(res.restorability_score)
    assert res.muq_mos == pytest.approx(4.2)
    assert res.muq_quality_witness == pytest.approx(80.0)
    assert res.predicted_mos == pytest.approx(round(np.clip(0.5 * dsp_mos + 2.1, 1.0, 5.0), 2))
    assert res.as_dict()["muq_mos"] == pytest.approx(4.2)


def test_estimator_without_muq_keeps_dsp_mos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ohne MuQ bleibt predicted_mos die reine DSP-Schätzung (Rückwärtskompatibilität)."""
    from backend.core.restorability_estimator import RestorabilityEstimator

    est = RestorabilityEstimator()
    audio = np.random.RandomState(2).randn(48000).astype(np.float32) * 0.1

    monkeypatch.setattr(mq, "estimate_quality_witness", lambda a, sr: None)
    monkeypatch.setattr(mq, "estimate_muq_mos", lambda a, sr: None)

    res = est.estimate(audio, 48000, material="vinyl")

    assert res.muq_mos is None
    assert res.muq_quality_witness is None
    assert res.predicted_mos == pytest.approx(est._score_to_mos(res.restorability_score))


# ─── Echter Modellpfad (nur wenn Checkpoint vorhanden) ─────────────────


@pytest.mark.skipif(
    not mq.is_available(),
    reason="MuQ-Checkpoint fehlt (models/muq_mulan/ oder HF-Cache)",
)
def test_embedding_deterministic_and_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Echtes Modell: deterministisch (§G5 (GEBOTE.md)) und 1024-dim Embedding."""
    # Im Test-Kontext meldet der Device-Manager ggf. CPU — CPU-Pfad explizit erlauben.
    monkeypatch.setenv("AURIK_MUQ_CPU", "1")
    rng = np.random.RandomState(0)
    audio = rng.randn(2, 48000).astype(np.float32) * 0.1

    e1 = mq.extract_embedding(audio, 48000)
    e2 = mq.extract_embedding(audio, 48000)

    assert e1 is not None and e2 is not None
    assert e1.shape == (1024,)
    assert np.array_equal(e1, e2)
    assert np.all(np.isfinite(e1))


def test_quality_witness_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Witness-Mapping (Kosinus→[0,100]) deterministisch mit Fake-Embeddings.

    Der echte Modellpfad wird in test_embedding_deterministic_and_shape geprüft;
    hier läuft nur die schnelle Zentroid-/Mapping-Logik.
    """
    rng = np.random.RandomState(1)
    emb = rng.randn(1024).astype(np.float32)
    refs = rng.randn(4, 1024).astype(np.float32)
    monkeypatch.setattr(mq, "extract_embedding", lambda a, sr: emb)
    monkeypatch.setattr(mq, "_get_ref_embeddings", lambda: refs)

    w = mq.estimate_quality_witness(np.zeros(48000, dtype=np.float32), 48000)

    assert w is not None
    assert 0.0 <= w <= 100.0
    assert mq.estimate_quality_witness(np.zeros(48000, dtype=np.float32), 48000) == pytest.approx(w)

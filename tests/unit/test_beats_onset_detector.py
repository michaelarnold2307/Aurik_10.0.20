"""Unit-Tests für backend/core/dsp/beats_onset_detector.py (§SOTA-TP-V1).

Abgedeckt: fbank-Konvention (Kaldi 25/10 ms, 128 Bins), Witness-Fallback ohne
Modell (§V6 (copilot-instructions.md)), Konsens-Statistik (bestätigte Onsets,
Toleranzfenster), Kurven-Alignment auf das Eingangs-Grid, Determinismus,
echter BEATs-Encoder-Smoke (skip ohne Modell).
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.beats_onset_detector as bod


def _clicks(sr: int, seconds: float, at_s: list[float]) -> np.ndarray:
    x = np.zeros(int(sr * seconds), dtype=np.float32)
    for t in at_s:
        i = int(t * sr)
        if 0 < i < len(x) - 32:
            x[i : i + 32] = 0.9 * np.hanning(32)
    return x


def test_fbank_shape_16k() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(16000).astype(np.float32) * 0.05
    fb = bod.fbank_16k(x)
    assert fb.shape[0] == 1
    assert fb.shape[2] == 128
    assert 95 <= fb.shape[1] <= 105  # 1 s @ 10-ms-Hop ≈ 99 Frames
    assert np.all(np.isfinite(fb))


def test_curve_none_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bod, "beats_available", lambda: False)
    curve, cs, ce = bod.beats_onset_curve(_clicks(16000, 1.0, [0.5]), 16000)
    assert curve is None and cs == 0 and ce == 0


def test_witness_stats_agreement() -> None:
    sr = 16000
    curve = np.zeros(sr, dtype=np.float32)
    for t_s in (0.2, 0.4, 0.6):
        i = int(t_s * sr)
        curve[i - 16 : i + 16] = 0.8
    onsets = np.array([0.2, 0.4, 0.6, 0.9])  # 0.9: kein Kurven-Peak
    stats = bod.onset_witness_stats(onsets, np.ones(4), curve, sr, 0, sr)
    assert stats["n_onsets"] == 4
    assert stats["n_confirmed"] == 3
    assert stats["agreement_ratio"] == pytest.approx(0.75)
    assert stats["mean_curve_peak_at_onsets"] > 0.5


def test_witness_stats_covered_region_only() -> None:
    """Onsets außerhalb [covered_start, covered_end) zählen als unbestätigt."""
    sr = 16000
    curve = np.zeros(sr, dtype=np.float32)
    i = int(0.75 * sr)  # Peak klar AUSSERHALB der gedeckten Region [0, 0.5 s)
    curve[i - 16 : i + 16] = 0.8
    stats = bod.onset_witness_stats(np.array([0.75]), np.ones(1), curve, sr, 0, sr // 2)
    assert stats["n_confirmed"] == 0


def test_curve_alignment_and_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake-Tokens: Kurve liegt im gedeckten Bereich und ist reproduzierbar."""
    fake_tokens = np.zeros((20, 768), dtype=np.float32)
    fake_tokens[5:8] = 1.0  # Sprung ⇒ Diff-Peak bei Frame ~6
    monkeypatch.setattr(bod, "beats_available", lambda: True)
    monkeypatch.setattr(bod, "_tokens_for_window", lambda seg: fake_tokens)
    x = _clicks(16000, 2.0, [1.0])
    c1, cs1, ce1 = bod.beats_onset_curve(x, 16000)
    c2, cs2, ce2 = bod.beats_onset_curve(x, 16000)
    assert c1 is not None and c2 is not None
    assert cs1 == cs2 and ce1 == ce2
    assert np.array_equal(c1, c2)
    assert c1.shape == (len(x),)
    assert np.all(np.isfinite(c1))
    assert np.max(c1[cs1:ce1]) > 0.2  # Normalisierung deckelt bei ~1/3 (3×Median)
    assert np.all(c1[:cs1] == 0.0)


@pytest.mark.skipif(not bod.beats_available(), reason="models/beats/beats_iter3.onnx fehlt")
def test_real_beats_encoder_smoke() -> None:
    """Echter BEATs-Encoder: 2-s-Clip liefert eine finite Onset-Kurve (CPU)."""
    x = _clicks(16000, 2.0, [0.8, 1.2])
    curve, cs, ce = bod.beats_onset_curve(x, 16000)
    assert curve is not None
    assert cs >= 0 and ce <= len(x)
    assert np.all(np.isfinite(curve))

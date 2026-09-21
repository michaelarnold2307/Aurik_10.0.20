#!/usr/bin/env python3
"""Unit-Tests für den autonomen Never-worsen-Arbiter (§v10.26).

Schnell und deterministisch: `resolve_never_worsen` wird mit einem
injizierten Fake-Scorer getestet (kein MERT-/ONNX-Load) — der echte
BlindQualityEstimator-Pfad ist über `arbitrate` in der E2E-Evaluierung
abgedeckt. Fälle: identisch akzeptiert, Verschlechterung → Retry übernommen,
Retry schlechter/fehlgeschlagen → Eingabe (bit-identisch), Layout-Sicherheit
(channels-first/channels-last), Scorer-Ausfall propagiert (§V6-fail-open beim
Aufrufer).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.blind_reference_free_quality import BlindQualityScore
from backend.core.dsp.autonomous_quality_arbiter import resolve_never_worsen


def _score(overall: float = 60.0, *, hf: float = 60.0, trans: float = 60.0, dyn: float = 60.0) -> BlindQualityScore:
    return BlindQualityScore(
        overall=overall,
        hf_presence=hf,
        transient_density=trans,
        dynamic_range_health=dyn,
        spectral_naturalness=55.0,
        noise_floor_continuity=55.0,
        stereo_naturalness=100.0,
    )


class FakeScorer:
    """Deterministischer Scorer: Reihenfolge-basiert (Eingabe, Erstlauf, Retry)."""

    def __init__(self, scores: list[BlindQualityScore], record: list | None = None):
        self.scores = scores
        self.calls: list[int] = []
        self.record = record

    def __call__(self, mono: np.ndarray) -> BlindQualityScore:
        idx = len(self.calls)
        self.calls.append(len(mono) if mono is not None else -1)
        if self.record is not None:
            self.record.append(int(mono.shape[0]))
        return self.scores[min(idx, len(self.scores) - 1)]


_GOOD_IN = _score(70.0)
_BAD_OUT = _score(40.0, hf=20.0, trans=20.0, dyn=20.0)  # deutlich schlechter als Eingabe
_GOOD_RETRY = _score(75.0)
_BAD_RETRY = _score(30.0, hf=10.0, trans=10.0, dyn=10.0)


def _audio(n: int = 8000, layout: str = "cl") -> np.ndarray:
    rng = np.random.default_rng(42)
    if layout == "cl":
        return rng.standard_normal((n, 2)).astype(np.float32) * 0.1
    return rng.standard_normal((2, n)).astype(np.float32) * 0.1


def test_identisch_akzeptiert():
    x = _audio()
    res = resolve_never_worsen(x, x.copy(), retry_fn=lambda: None, scorer=FakeScorer([_GOOD_IN, _GOOD_IN]))
    assert res.status == "accepted"
    assert np.array_equal(res.chosen_audio, x)
    assert res.retry_score is None


def test_verschlechterung_retry_besser_wird_uebernommen():
    x = _audio()
    retry = _audio()
    res = resolve_never_worsen(
        x,
        _audio(),  # Erstlauf (Inhalt irrelevant, Scorer entscheidet)
        retry_fn=lambda: retry,
        scorer=FakeScorer([_GOOD_IN, _BAD_OUT, _GOOD_RETRY]),
    )
    assert res.status == "retry_balanced"
    assert np.array_equal(res.chosen_audio, retry)
    assert res.retry_score == 75.0
    assert res.reasons  # Erstlauf-Gründe dokumentiert
    assert res.metadata["status"] == "retry_balanced"
    assert res.metadata["retry_out"] == 75.0


def test_verschlechterung_retry_auch_schlechter_eingabe():
    x = _audio()
    res = resolve_never_worsen(
        x,
        _audio(),
        retry_fn=lambda: _audio(),
        scorer=FakeScorer([_GOOD_IN, _BAD_OUT, _BAD_RETRY]),
    )
    assert res.status == "reverted"
    assert np.array_equal(res.chosen_audio, x)  # bit-identischer Passthrough
    assert res.retry_score == 30.0
    assert "overall" in res.reasons[0]


def test_verschlechterung_retry_none_eingabe():
    x = _audio()
    res = resolve_never_worsen(x, _audio(), retry_fn=lambda: None, scorer=FakeScorer([_GOOD_IN, _BAD_OUT]))
    assert res.status == "reverted"
    assert np.array_equal(res.chosen_audio, x)


def test_verschlechterung_retry_exception_eingabe():
    def _boom() -> np.ndarray:
        raise RuntimeError("retry kaputt")

    x = _audio()
    res = resolve_never_worsen(x, _audio(), retry_fn=_boom, scorer=FakeScorer([_GOOD_IN, _BAD_OUT]))
    assert res.status == "reverted"
    assert np.array_equal(res.chosen_audio, x)


def test_layout_channels_first_und_last_downmix():
    n = 8000
    for layout in ("cl", "cf"):
        recorded: list[int] = []
        x = _audio(n, layout=layout)
        res = resolve_never_worsen(
            x,
            x.copy(),
            retry_fn=lambda: None,
            scorer=FakeScorer([_GOOD_IN, _GOOD_IN], record=recorded),
        )
        assert res.status == "accepted"
        # Der Scorer muss echtes Mono (N Samples) sehen, nicht C Samples (2).
        assert recorded == [n, n], f"layout={layout}: {recorded}"


def test_scorer_ausfall_propagiert_fail_open_aufrufer():
    def _bad_scorer(_mono: np.ndarray) -> BlindQualityScore:
        raise RuntimeError("Scorer nicht verfügbar")

    with pytest.raises(RuntimeError):
        resolve_never_worsen(_audio(), _audio(), retry_fn=lambda: None, scorer=_bad_scorer)

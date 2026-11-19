"""tests/unit/test_pre_analysis_depth_cap_and_clap_sr.py

Tests für die Root-Fixes vom 2026-08-16:
1. _apply_carrier_depth_cap: §v10.19 Depth-Cap-2 greift FRÜH (vor DefectScanner).
2. _compute_clap_score: resampelt vor clap.tag auf exakt 48000 Hz.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.pre_analysis import _apply_carrier_depth_cap


def _md(confidence: float, chain: list[str]) -> SimpleNamespace:
    return SimpleNamespace(confidence=confidence, transfer_chain=list(chain))


# ─── 1. Carrier-Depth-Cap ────────────────────────────────────────────────────


def test_depth_cap_trims_low_confidence_chain() -> None:
    md = _md(0.31, ["vinyl", "reel_tape", "mp3_low"])
    _apply_carrier_depth_cap(md)
    assert md.transfer_chain == ["vinyl", "mp3_low"]


def test_depth_cap_leaves_high_confidence_chain() -> None:
    md = _md(0.80, ["vinyl", "reel_tape", "mp3_low"])
    _apply_carrier_depth_cap(md)
    assert md.transfer_chain == ["vinyl", "reel_tape", "mp3_low"]


def test_depth_cap_leaves_short_chain() -> None:
    md = _md(0.40, ["vinyl", "reel_tape"])
    _apply_carrier_depth_cap(md)
    assert md.transfer_chain == ["vinyl", "reel_tape"]


# ─── 2. CLAP 48-kHz-Resample ─────────────────────────────────────────────────


def test_clap_score_resamples_to_48k() -> None:
    from backend.core.genre_classifier import GermanSchlagerClassifier

    clf = GermanSchlagerClassifier.__new__(GermanSchlagerClassifier)
    clf._clap_score_is_fallback = True

    received_sr: dict = {}

    class _FakeClap:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def _result(values: dict[str, float]):
            """Zero-Shot-Ergebnis: Scores liegen in custom_scores, NICHT in genre_tags.

            Vorher las der Classifier ``genre_tags`` nach Schlüsseln wie
            „schlager"/„volksmusik"/„german" — die es in GENRE_TAGS nicht gibt,
            wodurch die Bewertung immer 0.0 ergab (§G9 copilot-instructions.md, Befund 2026-10-06).
            """
            return SimpleNamespace(
                genre_tags={},
                custom_scores=values,
                query_scores=lambda prompts, _v=values: {p: _v[p] for p in prompts if p in _v},
            )

        def tag(self, audio, sr, text_queries=None):
            self.calls += 1
            received_sr["sr"] = sr
            # Aufruf-Reihenfolge ist deterministisch: positiv zuerst, dann negativ.
            _value = 0.7 if self.calls == 1 else 0.1
            return self._result(dict.fromkeys(text_queries or [], _value))

    audio = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 22050, endpoint=False))).astype(np.float32)
    with (
        patch("backend.core.ml_memory_budget.try_allocate", return_value=True),
        patch("plugins.laion_clap_plugin.get_loaded_laion_clap", return_value=_FakeClap()),
        patch("backend.core.ml_memory_budget.release", MagicMock()),
    ):
        score = clf._compute_clap_score(audio, 22050)

    assert received_sr.get("sr") == 48000
    # result_score = clip(0.7 - 0.5 * 0.1) = 0.65 — aus den Zero-Shot-Scores
    # (custom_scores), nicht aus genre_tags. Kein Prior-Fallback mehr.
    assert score == pytest.approx(0.65)
    assert clf._clap_score_is_fallback is False

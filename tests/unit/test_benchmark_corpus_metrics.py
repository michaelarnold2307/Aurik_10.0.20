"""Pegel-ehrliche Messmetrik des Corpus-Harness (scripts/benchmark_corpus.py).

Pinnt die Ehrlichkeits-Semantik, die am 2026-10-07 drei Artefakt-Verdikte
aufgedeckt hat (Register D-K3-41):

  - ``_snr_pair``: die gain-angepasste SNR trägt das Verdikt; rohe SNR und
    Output-Pegel werden getrennt berichtet (die alte rohe SNR bestrafte die
    Lautheits-Normalisierung fälschlich).
  - ``_align_pair``: Mono↔Stereo-Vergleiche kollabieren deterministisch auf
    das Mono-Mittel beider Seiten (Stereo-Layout-Invariante).
  - ``_pair_files``: Mehrfach-Suffixe (``_hiss_crackle_hum``) werden gepaart —
    der alte Präfix-Abgleich fand nur 4 von 56 Paaren.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_corpus import _PROJECT, _align_pair, _pair_files, _snr_pair

_MEDIA = ("cassette", "digital", "reel_tape", "shellac", "tape", "vinyl")


def _signal(seed: int = 7, n: int = 24000, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, channels)) * 0.1).astype(np.float32)


def test_gain_matched_snr_ignores_level_shift() -> None:
    """6 dB leiser, aber inhaltlich identisch ⇒ gm-SNR praktisch perfekt."""
    ref = _signal()
    raw, gm, gain_db = _snr_pair(ref, ref * 0.5)
    assert gm > 150.0, f"gm-SNR muss den Pegel ignorieren, war {gm:.1f} dB"
    assert abs(gain_db - (-6.0)) < 0.1, f"Pegel −6 dB erwartet, war {gain_db:.2f}"
    assert raw < gm - 100.0, "roh bleibt pegel-sensitiv — dokumentiert die alte Illusion"


def test_gain_matched_snr_detects_content_loss() -> None:
    """Echtes Rauschen ⇒ gm-SNR fällt; ohne Pegel-Shift liegen roh und gm nah beieinander."""
    ref = _signal()
    rng = np.random.default_rng(11)
    noisy = (ref + rng.standard_normal(ref.shape).astype(np.float32) * 0.03).astype(np.float32)
    raw, gm, gain_db = _snr_pair(ref, noisy)
    assert 5.0 < gm < 20.0, f"Rausch-SNR erwartet 5–20 dB, war {gm:.1f}"
    assert abs(raw - gm) < 3.0, "ohne Pegel-Shift müssen roh und gm nahe beieinanderliegen"
    assert abs(gain_db) < 1.0


def test_align_pair_handles_mono_stereo_mismatch() -> None:
    """Mono↔Stereo: beide Seiten deterministisch auf Mono-Mittel reduziert."""
    ref = _signal(channels=2)
    mono = ref.mean(axis=1, keepdims=True)
    a, b = _align_pair(ref, mono)
    assert a.shape == b.shape
    assert a.shape[1] == 1


def test_pair_files_finds_multiple_suffixes() -> None:
    """Mehrfach-Suffixe müssen gepaart werden (alter Bug: 4 von 56 Paaren)."""
    if not _PROJECT.joinpath("corpus").exists():
        pytest.skip("Korpus nicht generiert (scripts/generate_corpus.py) — Paarungstest übersprungen")
    total = 0
    for medium in _MEDIA:
        pairs = _pair_files(medium)
        total += len(pairs)
        for damaged, clean in pairs:
            base = clean.stem[: -len("_clean")]
            assert damaged.stem == base or damaged.stem.startswith(base + "_"), (
                f"Fehlpaarung: {damaged.name} ↔ {clean.name}"
            )
    if total == 0:
        pytest.skip("Keine WAVs im Korpus vorhanden — Paarungstest übersprungen")
    assert total >= 30, f"Korpus-Paarung erwartet ≥ 30 Paare, fand {total}"

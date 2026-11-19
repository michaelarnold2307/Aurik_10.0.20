"""D-K3-45: DFN-Pegel-Neutralität im Phase-0-Wrapper (Quality-Guard-Wurzel).

Produktionsbefund 2026-10-07: `DeepFilterNet Quality-Guard: Degradation →
Rollback. RMS=-5.5dB Crest=0.086 HF=-9.1dB` — das DFN-Modell liefert
systematisch leiser (−5,5 dB RMS), der Guard (rms_delta < −3 dB) verwarf damit
die LEGITIME Entrauschung per Rollback. Gleiches Muster wie D-K3-44: Pegel ≠
Form. Fix: `_level_match` gleicht den Clean-Pass-Output vor den Form-Guards an
den Eingangspegel an (±10 dB begrenzt, 0,995-Headroom, <0,1 dB No-Op).
"""

from __future__ import annotations

import numpy as np

from plugins.apollo_phase0_integration import _level_match, _quality_delta

SR = 48_000


def _music_mock(n: int, seed: int = 42) -> np.ndarray:
    t = np.arange(n) / SR
    x = (
        0.30 * np.sin(2 * np.pi * 110.0 * t)
        + 0.15 * np.sin(2 * np.pi * 440.0 * t)
        + 0.05 * np.sin(2 * np.pi * 3000.0 * t)
    )
    rng = np.random.default_rng(seed)
    return (x + 0.01 * rng.standard_normal(n)).astype(np.float32)


def test_level_match_neutralizes_dfn_level_drop() -> None:
    x = _music_mock(SR)
    y = (x * 10 ** (-5.5 / 20.0)).astype(np.float32)  # DFN-Modellpegel −5,5 dB
    q = _quality_delta(x, y, SR)
    assert q["rms_delta_db"] < -3.0, "Fixture-Vertrag: unkompensiert muss der Guard-Trigger bestehen"

    y2 = _level_match(x, y)
    q2 = _quality_delta(x, y2, SR)
    assert abs(q2["rms_delta_db"]) < 0.1, f"Pegel-Angleich wirkungslos: {q2['rms_delta_db']} dB"
    assert q2["hf_delta_db"] > -8.0, "HF-Trigger muss nach Angleich verschwinden"


def test_level_match_respects_gain_limit() -> None:
    x = _music_mock(SR)
    y = (x * 10 ** (-30.0 / 20.0)).astype(np.float32)  # −30 dB: außerhalb der Grenze
    y2 = _level_match(x, y, max_gain_db=10.0)

    def _rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)) + 1e-12)

    gain_db = 20.0 * np.log10(_rms(y2) / _rms(y))
    assert 9.0 < gain_db < 11.0, f"Gain-Begrenzung verletzt: {gain_db:.1f} dB"


def test_level_match_noop_within_tolerance() -> None:
    x = _music_mock(SR)
    y = (x * 10 ** (0.05 / 20.0)).astype(np.float32)  # +0,05 dB
    y2 = _level_match(x, y)
    assert np.array_equal(np.asarray(y2), np.asarray(y)), "No-Op-Fenster (0,1 dB) verletzt"


def test_level_match_respects_headroom() -> None:
    x = _music_mock(SR)
    y = (x * 1e-4).astype(np.float32)  # extrem leise → Gain begrenzt + Headroom
    y2 = _level_match(x, y)
    assert float(np.max(np.abs(y2))) <= 0.996, "Headroom-Grenze (0,995) verletzt"

"""Unit-Tests für das report-only Musik-MOS-Orakel (KP-2, Pfad A — kein Training).

Vertrag (§III.11 (copilot-instructions.md), Hörordnung §8): das Orakel übt NIE ein
Veto aus, flippt keine Flags und mutiert kein Profil — es misst und protokolliert
nur das Musik-MOS-Delta (MuQ+A1 + SingMOS Pro). Die Tests prüfen Determinismus
(§G5 (copilot-instructions.md)), Fallback-Pfade (§V6 (copilot-instructions.md)),
Stereo-Layout-Handling (§11 Spec 02) und die Schwellen-/Delta-Logik — ohne
`try/except: pass`-Hüllen (Lehre D-K3-20).
"""

from __future__ import annotations

import numpy as np

import backend.core.dsp.music_mos_oracle as mmo


def _sine(n: int = 4000, freq: float = 440.0, sr: int = 44100, seed: int = 42) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_downmix_channels_first_and_last() -> None:
    """Stereo-Layout-Invariante: (2, N) und (N, 2) liefern dasselbe Mono."""
    n = 1000
    left = _sine(n)
    right = 0.5 * np.sin(2 * np.pi * 880.0 * np.linspace(0, n / 44100, n, endpoint=False)).astype(np.float32)
    cf = np.stack([left, right])  # (2, N)
    cl = np.stack([left, right], axis=1)  # (N, 2)
    mono_cf = mmo._downmix(cf)
    mono_cl = mmo._downmix(cl)
    assert mono_cf.shape == (n,)
    assert mono_cl.shape == (n,)
    np.testing.assert_allclose(mono_cf, mono_cl, rtol=1e-6, atol=1e-6)


def test_downmix_mono_identity() -> None:
    mono = _sine()
    out = mmo._downmix(mono)
    assert out.shape == mono.shape
    np.testing.assert_allclose(out, mono, rtol=1e-7)


def test_downmix_nan_inf_guarded() -> None:
    """§0a NaN/Inf-Schutz: nicht-endliche Samples werden neutralisiert."""
    x = np.zeros(1000, dtype=np.float32)
    x[10] = np.nan
    x[20] = np.inf
    x[30] = -np.inf
    out = mmo._downmix(x)
    assert np.all(np.isfinite(out))


def test_clip_mos_bounds() -> None:
    assert mmo._clip_mos(0.0) == 1.0
    assert mmo._clip_mos(3.4) == 3.4
    assert mmo._clip_mos(99.0) == 5.0
    assert mmo._clip_mos(None) is None
    assert mmo._clip_mos(float("nan")) is None


def test_composite_delta_only_available() -> None:
    assert mmo._composite_delta(0.2, 0.4) == 0.3
    assert mmo._composite_delta(0.2, None) == 0.2
    assert mmo._composite_delta(None, 0.4) == 0.4
    assert mmo._composite_delta(None, None) is None


def test_witness_always_false_veto_no_model(monkeypatch) -> None:
    """report-only: ohne Modelle bleibt applied=False UND das Profil/Flags unberührt.

    Lehre D-K3-20: kein `try/except: pass` — die Fehlschläge werden via monkeypatch
    explizit auf None gesetzt, damit der Fallback-Pfad deterministisch läuft.
    """

    monkeypatch.setattr(mmo, "_muq_mos", lambda _a, _sr: None)
    monkeypatch.setattr(mmo, "_singmos_pro", lambda _a, _sr: None)
    w = mmo.music_mos_witness(np.zeros(1600), np.zeros(1600), 44100)
    assert w.applied is False
    assert w.composite_delta is None
    assert w.muq_delta is None
    assert w.singmos_delta is None


def test_witness_computes_delta_and_never_vetos(monkeypatch) -> None:
    """MuQ+A1 und SingMOS messen unterschiedliche Ref/Cand-Scores; Composite-Delta wird gebildet — applied bleibt False."""

    state: dict[str, float] = {"muq": 4.0, "sing": 4.5}

    def _fake_muq(audio: np.ndarray, sr: int) -> float:
        # Kandidat (negatives Signal) wird schlechter bewertet.
        return state["muq"] if float(np.asarray(audio).mean()) >= 0 else state["muq"] - 0.5

    def _fake_sing(audio: np.ndarray, sr: int) -> float:
        return state["sing"] if float(np.asarray(audio).mean()) >= 0 else state["sing"] - 1.0

    monkeypatch.setattr(mmo, "_muq_mos", _fake_muq)
    monkeypatch.setattr(mmo, "_singmos_pro", _fake_sing)
    ref = np.zeros(1600, dtype=np.float32)
    cand = np.full(1600, -1.0, dtype=np.float32)
    w = mmo.music_mos_witness(ref, cand, 44100)
    assert w.applied is False
    assert w.muq_delta == round(-0.5, 4)
    assert w.singmos_delta == round(-1.0, 4)
    assert w.composite_delta == round(-0.75, 4)


def test_witness_deterministic_repeat(monkeypatch) -> None:
    """§G5 (copilot-instructions.md): gleicher Input ⇒ gleiche Scores/Deltas (kein Sampling, keine Zeit)."""

    calls: list[tuple] = []

    def _fake_muq(audio: np.ndarray, sr: int) -> float:
        calls.append(("muq", float(np.asarray(audio).sum()), sr))
        return 4.2

    def _fake_sing(audio: np.ndarray, sr: int) -> float:
        calls.append(("sing", float(np.asarray(audio).sum()), sr))
        return 3.1

    monkeypatch.setattr(mmo, "_muq_mos", _fake_muq)
    monkeypatch.setattr(mmo, "_singmos_pro", _fake_sing)
    ref = np.ones(1600, dtype=np.float32)
    cand = np.full(1600, 0.5, dtype=np.float32)
    w1 = mmo.music_mos_witness(ref, cand, 44100)
    w2 = mmo.music_mos_witness(ref, cand, 44100)
    assert w1.composite_delta == w2.composite_delta
    assert w1.muq_delta == w2.muq_delta
    assert w1.singmos_delta == w2.singmos_delta


def test_measure_and_log_does_not_mutate_flags(monkeypatch) -> None:
    """Flag `use_utmos_music` bleibt False; kein Flag-Flip im Zeugen-Pfad."""
    import backend.core.music_model_flags as mmf

    monkeypatch.setattr(mmo, "_muq_mos", lambda _a, _sr: 4.0)
    monkeypatch.setattr(mmo, "_singmos_pro", lambda _a, _sr: None)
    before = mmf.use_utmos_music
    _w = mmo.measure_and_log_music_mos(np.zeros(1600), np.zeros(1600), 44100)
    assert mmf.use_utmos_music is before


def test_witness_log_format_no_crash() -> None:
    w = mmo.MusicMOSWitness(stage="t", muq_delta=-0.25, muq_mos_ref=4.0, muq_mos_candidate=3.75)
    s = mmo.music_mos_witness_log(w)
    assert "report-only" in s
    assert "composite_delta" in s

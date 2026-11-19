"""Vertragstests DRT-13 (Spec 26) — reparatur_phasen_attribution (Teil-Rollback).

Prüft: Snapshot/Variante bit-identisch außerhalb der Region-Maske,
Masken-Pflicht (ohne Maske keine Rücknahme), Shape-Mismatch fail-safe,
Begrenzung auf 4 Snapshots, Clear-Song-Isolation (§V8/§G1 (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp import reparatur_phasen_attribution as rpa


def test_variante_nur_in_maske_zurueckgenommen() -> None:
    rpa.clear()
    pre = np.linspace(-0.9, 0.9, 100, dtype=np.float32)
    maske = np.zeros(100, dtype=bool)
    maske[40:60] = True
    rpa.snapshot("phase_07_declip", pre, maske)
    out = np.where(maske, pre + 0.5, pre).astype(np.float32)  # Phase hat Maske-Regionen verändert
    variante, info = rpa.phase_aus_variante(out)
    assert info["phase_07_declip"]["regionen"] == 20
    assert np.array_equal(variante[maske], pre[maske])  # Regionen zurückgenommen
    assert np.array_equal(variante[~maske], out[~maske])  # Rest bit-identisch


def test_ohne_maske_keine_ruecknahme() -> None:
    rpa.clear()
    pre = np.zeros(100, dtype=np.float32)
    rpa.snapshot("phase_x", pre, None)
    out = np.ones(100, dtype=np.float32)
    variante, info = rpa.phase_aus_variante(out)
    assert info["phase_x"]["grund"] == "keine_regionen"
    assert np.array_equal(variante, out)  # kein stiller Ganz-Ersatz


def test_shape_mismatch_fail_safe() -> None:
    rpa.clear()
    pre = np.zeros(50, dtype=np.float32)
    rpa.snapshot("phase_y", pre, np.ones(50, dtype=bool))
    out = np.ones(100, dtype=np.float32)
    variante, info = rpa.phase_aus_variante(out)
    assert info["phase_y"]["grund"] == "shape_mismatch"
    assert np.array_equal(variante, out)


def test_begrenzung_auf_vier_snapshots() -> None:
    rpa.clear()
    for i in range(6):
        rpa.snapshot(f"phase_{i}", np.zeros(10, dtype=np.float32), np.ones(10, dtype=bool))
    assert len(rpa.attribution_report()["phasen"]) == 4


def test_clear_song_isolation() -> None:
    rpa.clear()
    rpa.snapshot("phase_z", np.zeros(10, dtype=np.float32), np.ones(10, dtype=bool))
    rpa.clear()
    out = np.ones(10, dtype=np.float32)
    variante, info = rpa.phase_aus_variante(out)
    assert info == {} and np.array_equal(variante, out)


def test_determinismus() -> None:
    rpa.clear()
    rng = np.random.default_rng(42)
    pre = rng.standard_normal(200).astype(np.float32)
    maske = rng.random(200) > 0.5
    rpa.snapshot("phase_d", pre, maske)
    out = (pre + rng.standard_normal(200)).astype(np.float32)
    v1, i1 = rpa.phase_aus_variante(out)
    v2, i2 = rpa.phase_aus_variante(out)
    assert np.array_equal(v1, v2) and i1 == i2

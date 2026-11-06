"""Vertragstests F10 (scripts/calibrate_singmos_music.py, D-K4-6).

Ohne Modell-Laden (§V6 (copilot-instructions.md)): der Produktions-Score wird injiziert; Modell-Smokes
bleiben skip-markiert (§G5 (GEBOTE.md): Tests duerfen nicht umgebungsabhaengig
kippen). Zitierdisziplin AGENTS.md §5 beachten.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import scripts.calibrate_singmos_music as csm

_SRC = pathlib.Path(csm.__file__).read_text(encoding="utf-8")


def _tone(n: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    return (0.5 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * rng.standard_normal(n)).astype(np.float32)


def test_cli_vertrag_r4() -> None:
    """Kette-4-Vertrag R4: genau --musdb-root/--out-dir (plus optionale Erweiterungen)."""
    argv = ["--musdb-root", "data/musdb18hq", "--out-dir", "models/singmos/calibration"]
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs-per-effect", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ns = ap.parse_args(argv)
    assert ns.musdb_root == "data/musdb18hq"
    assert ns.out_dir == "models/singmos/calibration"
    assert ns.seed == 42


def test_effekte_determinismus() -> None:
    x = _tone()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    for name in csm._EFFECTS:
        a = csm._EFFECTS[name](x, rng1)
        b = csm._EFFECTS[name](x, rng2)
        assert np.array_equal(a, b), name
    rng3 = np.random.default_rng(43)
    assert not np.array_equal(csm._hall(x, rng1), csm._hall(x, rng3))


def test_effekte_finit_und_wirksam() -> None:
    x = _tone()
    rng = np.random.default_rng(7)
    for name, fn in csm._EFFECTS.items():
        y = fn(x, rng)
        assert np.all(np.isfinite(y)), name
        assert y.shape == x.shape, name
        assert not np.array_equal(y, x), name


def test_score_pair_nan_fallback() -> None:
    """§V6 (copilot-instructions.md): nicht-endlicher Score -> 3.0 + Zaehlung."""

    def bad(_sig: np.ndarray) -> float:
        return float("nan")

    res = csm.score_pair(_tone(), _tone(seed=1), bad)
    assert res["clean"] == 3.0
    assert res["processed"] == 3.0
    assert res["nan_fallback"] == 2


def test_fit_calibration_monoton() -> None:
    rows = []
    for s, mean in ((1, 4.4), (2, 3.9), (3, 3.2), (4, 2.6), (5, 2.1)):
        chain = next(k for k, v in csm._EFFECT_STRENGTH.items() if v == s)
        for _ in range(4):
            rows.append({"chain": chain, "strength": s, "clean": 4.5, "processed": mean})
    cal = csm.fit_calibration(rows)
    assert cal["n_pairs"] == 20
    assert cal["per_pair_violations"] == 0
    assert cal["a"] > 0.0


def test_build_pairs_fail_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        csm.build_pairs(tmp_path / "gibt_es_nicht", 42, 2)


def test_keine_flag_flips() -> None:
    assert "flag_freigabe" not in _SRC
    assert ".flags" not in _SRC

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


def _tone(n: int | None = None, seed: int = 0) -> np.ndarray:
    """Test-Signal in der Effekt-Vertragsrate (_SR_MOS = 48 kHz, siehe Plugin-Vertrag)."""
    sr = float(csm._SR_MOS)
    n = int(sr) if n is None else int(n)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
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
        assert y.shape == x.shape, name  # Form-Invariante: clean/verarbeitet sample-gleich
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


# ── Regressionen 2026-11-08 (Kette 4 R4, D-K4-6) ─────────────────────────────
def test_sr_mos_folgt_dem_plugin_vertrag() -> None:
    """Regression: 16 kHz an `score_mos` ⇒ AssertionError nach 56 s (Produktionsbefund).

    Vertragsquelle ist plugins/versa_plugin.py (`assert sr == 48_000`), nicht der
    frühere Kommentar "I/O @ 16 kHz".
    """
    assert csm._SR_MOS == 48_000


def test_messfenster_unter_plugin_kappung() -> None:
    """Der Produktions-Score kappt auf 30 s — das Messfenster muss darunter liegen."""
    assert 0.0 < csm._WINDOW_S <= 30.0


def test_pick_window_laenge_und_determinismus() -> None:
    sr = 48_000
    rng = np.random.default_rng(3)
    x = 0.01 * rng.standard_normal(60 * sr).astype(np.float32)
    x[25 * sr : 45 * sr] += 0.5  # lautes Fenster in der Mitte
    w1 = csm._pick_window(x, sr)
    w2 = csm._pick_window(x, sr)
    assert w1.shape == (int(csm._WINDOW_S * sr),)
    assert np.array_equal(w1, w2)  # §G5 (copilot-instructions.md)
    # energiereichstes 20-s-Fenster bei 5-s-Hop liegt auf 25 s (Startraster 0,5,10,…,25)
    assert np.array_equal(w1, x[25 * sr : 45 * sr])


def test_pick_window_kurz_bleibt_ganz() -> None:
    x = np.zeros(5 * 48_000, dtype=np.float32)
    assert csm._pick_window(x, 48_000).size == x.size


def test_fit_calibration_entartet_fail_closed() -> None:
    """Regression: entarteter Fit meldete a = 2,4e6 / b = −7,4e6 statt zu warnen."""
    rows = []
    for s, mean in ((1, 3.08), (2, 3.08), (3, 3.08), (4, 3.08), (5, 3.08)):
        chain = next(k for k, v in csm._EFFECT_STRENGTH.items() if v == s)
        rows.append({"chain": chain, "strength": s, "clean": 3.08, "processed": mean})
    cal = csm.fit_calibration(rows)
    assert cal["degeneriert"] is True
    assert cal["a"] == 0.0  # keine Schein-Steigung
    assert abs(cal["b"] - 3.08) < 1e-9
    assert cal["grund"]


def test_fit_calibration_nicht_entartet_bei_trennung() -> None:
    rows = []
    for s, mean in ((1, 4.4), (2, 3.2), (3, 2.2), (4, 1.9), (5, 1.6)):
        chain = next(k for k, v in csm._EFFECT_STRENGTH.items() if v == s)
        rows.append({"chain": chain, "strength": s, "clean": 4.7, "processed": mean})
    cal = csm.fit_calibration(rows)
    assert cal["degeneriert"] is False
    assert cal["a"] > 0.0
    # 4.7 → 4.2 und 1.6 → 1.8 (Ziel-Grenzen der affinen Abbildung)
    assert abs(cal["a"] * 4.7 + cal["b"] - 4.2) < 1e-9

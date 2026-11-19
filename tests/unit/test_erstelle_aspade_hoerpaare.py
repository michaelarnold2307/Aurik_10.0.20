#!/usr/bin/env python3
"""Vertragstests fuer scripts/erstelle_aspade_hoerpaare.py.

Fahrplan A-1 (`docs/reports/current/2026-11-13_aktivierungs_fahrplan.md`):
§4-Hörpaket für den Declipper — blinde A/B-Paare (PCHIP vs. A-SPADE bzw.
UV3-Produktionspfad Flag-AUS vs. Flag-AN). Gepinnt werden die
fail-closed-Pfade (§V6 (copilot-instructions.md)), die deterministische
Fenster-Wahl und die Schlüssel-Semantik. Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.erstelle_aspade_hoerpaare as eh


def test_fenster_schlimmstes_ist_deterministisch() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(48_000).astype(np.float32)
    x[1000:2000] = 1.0  # hartes Plateau — dort muss das Fenster landen
    fenster = 10 * 4800
    a = eh._fenster_schlimmstes(x, fenster, 0.85, 4800 // 2)
    b = eh._fenster_schlimmstes(x, fenster, 0.85, 4800 // 2)
    assert np.array_equal(a, b)
    assert a.shape == (fenster,)
    # Das Plateau (10k geklippte Samples) liegt im gewählten Fenster.
    assert float((np.abs(a) >= 0.85).sum()) >= 1000


def test_fenster_schlimmstes_leeres_signal() -> None:
    x = np.zeros(48_000, dtype=np.float32)
    a = eh._fenster_schlimmstes(x, 4800, 0.85, 4800 // 2)
    assert a.shape == (4800,)
    assert not np.isnan(a).any()


def test_produktionspfad_in_aspade_ausgang_fail_closed(tmp_path) -> None:
    """Fahrplan-A-1-Proben dürfen die PCHIP-vs.-A-SPADE-Paare nicht überschreiben."""
    ausgang = tmp_path / "aspade"
    assert (
        eh.main(
            [
                "--schaden-dir",
                str(tmp_path / "leer"),
                "--ausgang",
                str(ausgang),
                "--produktionspfad",
            ]
        )
        == 2
    )


def test_produktionspfad_ausgang_getrennt_zulaessig(tmp_path) -> None:
    """Eigener Ausgangsordner ist zulässig — Pfadfehler (keine Proben) bleibt Exit 2, kein Traceback."""
    ausgang = tmp_path / "declip_produktionspfad"
    rc = eh.main(["--schaden-dir", str(tmp_path / "leer"), "--ausgang", str(ausgang), "--produktionspfad"])
    assert rc == 2


def test_pegel_gematcht_gleicht_lautheit_an() -> None:
    """Blindes A/B ohne Lautheits-Bias: p25-Skalierung zieht beide Varianten gleich."""
    ziel = np.random.default_rng(1).standard_normal(48000).astype(np.float32) * 0.2
    quelle = np.random.default_rng(2).standard_normal(48000).astype(np.float32) * 0.6
    gematcht = eh._pegel_gematcht(ziel, quelle)
    assert eh._p25(gematcht) == pytest.approx(eh._p25(ziel), rel=1e-4)
    assert gematcht.dtype == np.float32


def test_pegel_gematcht_stille_quelle_unveraendert() -> None:
    ziel = np.ones(4800, dtype=np.float32)
    stille = np.zeros(4800, dtype=np.float32)
    assert np.array_equal(eh._pegel_gematcht(ziel, stille), stille)


def test_produktionspfad_lauf_stellt_flag_zurueck(monkeypatch) -> None:
    """§V6 (copilot-instructions.md): das Flag muss nach dem Lauf exakt den alten Wert haben (Song-Isolation)."""

    class FakeRestorer:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def restore(self, audio: np.ndarray, sample_rate: int):
            import backend.core.music_model_flags as mmf

            self.calls.append(bool(mmf.use_aspade_declipper))
            return type("R", (), {"audio": np.zeros_like(audio)})()

    import backend.core.music_model_flags as mmf

    alt = mmf.use_aspade_declipper
    try:
        mmf.use_aspade_declipper = False
        fake = FakeRestorer()
        monkeypatch.setattr(
            "backend.core.unified_restorer_v3.UnifiedRestorerV3",
            lambda: fake,
            raising=True,
        )
        x = np.zeros(4800, dtype=np.float32)
        eh._produktionspfad_lauf(x, 48000, flag_an=True)
        assert fake.calls == [True]
        assert mmf.use_aspade_declipper is False  # zurückgesetzt
    finally:
        mmf.use_aspade_declipper = alt

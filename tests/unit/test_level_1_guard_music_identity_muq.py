"""Integrationstests für den MuQ-MuLan-Identitäts-Zeugen im Ebene-1-Guard (KP-3).

Prüft: (1) `Level1Result.singer_identity_cosine_muq` existiert und wird in
`check()` zusätzlich gemessen, (2) bestehende Felder/Schwellen/Veto-Logik bleiben
unverändert, (3) der Zeuge übt nie ein Veto aus (report-only, §III.11
(copilot-instructions.md)). Ohne `try/except: pass` (Lehre D-K3-20).
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.level_1_invariants_guard as lig


def _noise_1s(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(48000).astype(np.float32) * 0.05


def test_level1result_has_muq_field() -> None:
    """Das neue Feld existiert mit Default None (report-only, kein Veto)."""
    r = lig.Level1Result()
    assert hasattr(r, "singer_identity_cosine_muq")
    assert r.singer_identity_cosine_muq is None


def test_check_populates_muq_field_without_perturbing_existing(monkeypatch) -> None:
    """check() misst den MuQ-Kosinus zusätzlich, bestehende Felder bleiben. Der MuQ-Pfad
    wird per monkeypatch deterministisch auf eine konstante Einbettung gesetzt (kein
    Modell geladen) — Lehre D-K3-20: kein stilles except:pass, sondern expliziter Stub."""
    import backend.core.dsp.music_identity_oracle as mio

    monkeypatch.setattr(mio, "_muq_embedding", lambda _a, _sr: np.ones(1024, dtype=np.float32))

    g = lig.Level1InvariantsGuard.__new__(lig.Level1InvariantsGuard)
    g._resemblyzer_plugin = None  # DSP-Proxy trägt die Messung (use_resemblyzer_music=False)

    x = _noise_1s(11)
    res = g.check(x, x, 48000, None)

    # Bestehende Felder bleiben messbar (unverändert logisch).
    assert res.singer_identity is not None
    assert res.blend_factor is not None
    assert res.violated_invariants is not None

    # Der MuQ-Zeuge wurde zusätzlich gemessen (identisches Signal ⇒ cos=1.0).
    assert res.singer_identity_cosine_muq == pytest.approx(1.0)


def test_check_no_model_keeps_muq_none_and_no_veto(monkeypatch) -> None:
    """Ohne MuQ-Modell bleibt singer_identity_cosine_muq None UND die Veto-Logik
    unverändert (bestehende singer_identity-Schwelle greift weiterhin, kein Flag-Flip)."""
    import backend.core.dsp.music_identity_oracle as mio
    import backend.core.music_model_flags as mmf

    monkeypatch.setattr(mio, "_muq_embedding", lambda _a, _sr: None)
    before_flag = mmf.use_resemblyzer_music

    g = lig.Level1InvariantsGuard.__new__(lig.Level1InvariantsGuard)
    g._resemblyzer_plugin = None

    x = _noise_1s(12)
    res = g.check(x, x, 48000, None)

    assert res.singer_identity_cosine_muq is None
    assert mmf.use_resemblyzer_music is before_flag
    # Bestehende Veto-Semantik bleibt (kein singer_identity-Verstoß bei identischem Audio).
    assert res.blend_factor == 1.0

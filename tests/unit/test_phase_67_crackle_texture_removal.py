"""Unit-Tests Phase 67: Crackle Texture Removal (§SR-CK-Textur).

Vertrag geprueft:
- Metadaten: DEFECT_REMOVAL, RESTORATION_ONLY
- Layout-Normalisierung: channels-first / samples-first / mono
- Passthrough-Semantik ohne Modell (§V6, copilot-instructions.md): bit-identisch, kein stiller Fallback
- Deterministisch: gleicher Input -> gleicher Output
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_67_crackle_texture_removal import CrackleTextureRemovalPhase
from backend.core.phases.phase_interface import PhaseCategory, PhaseMode


def test_metadata_contract() -> None:
    p = CrackleTextureRemovalPhase()
    md = p.get_metadata()
    assert md.phase_id == "phase_67_crackle_texture_removal"
    assert md.category == PhaseCategory.DEFECT_REMOVAL
    assert md.phase_mode == PhaseMode.RESTORATION_ONLY
    assert "crackle" in md.defect_types


def test_passthrough_ohne_modell_bit_identisch(monkeypatch: pytest.MonkeyPatch) -> None:
    p = CrackleTextureRemovalPhase()
    monkeypatch.setattr(p, "_load_model", lambda: False)
    rng = np.random.default_rng(0)
    x = rng.random((2, 48000), dtype=np.float32) * 0.2 - 0.1
    res = p.process(x, sample_rate=48000)
    assert res.success
    assert np.array_equal(res.audio, x)  # bit-identisch (§V6-Passthrough, copilot-instructions.md)
    assert "modell_nicht_ladbar" in res.warnings


def test_layout_normalisierung_ohne_modell(monkeypatch: pytest.MonkeyPatch) -> None:
    p = CrackleTextureRemovalPhase()
    monkeypatch.setattr(p, "_load_model", lambda: False)
    rng = np.random.default_rng(1)
    x_cf = rng.random((2, 24000), dtype=np.float32) * 0.2 - 0.1
    res_cf = p.process(x_cf, sample_rate=48000)
    assert res_cf.audio.shape == (2, 24000)
    res_sf = p.process(x_cf.T, sample_rate=48000)
    assert res_sf.audio.shape == (24000, 2)
    res_m = p.process(x_cf[0], sample_rate=48000)
    assert res_m.audio.shape == (24000,)


def test_sample_rate_guard() -> None:
    p = CrackleTextureRemovalPhase()
    x = np.zeros((2, 1000), dtype=np.float32)
    res = p.process(x, sample_rate=44100)
    assert "sample_rate" in res.warnings
    assert np.array_equal(res.audio, x)


def test_verarbeitung_mit_modell_falls_vorhanden() -> None:
    p = CrackleTextureRemovalPhase()
    if not p._load_model():
        pytest.skip("Modell nicht vorhanden (CI ohne Modelle) — Passthrough bereits abgedeckt")
    rng = np.random.default_rng(2)
    x = rng.random((2, 48000), dtype=np.float32) * 0.2 - 0.1
    res1 = p.process(x, sample_rate=48000)
    res2 = p.process(x, sample_rate=48000)
    assert res1.ml_used
    assert res1.audio.shape == (2, 48000)
    assert np.isfinite(res1.audio).all()
    assert np.array_equal(res1.audio, res2.audio)  # deterministisch (§G5, copilot-instructions.md)

"""Unit-Tests Phase 68: VinylNet De-Crackle (ML, chirurgisch) (§Spec 03/06).

Vertrag geprüft:
- Metadaten: DEFECT_REMOVAL, RESTORATION_ONLY, DefectTypes clicks/crackle
- Layout-Normalisierung: channels-first / samples-first / mono — Ausgabe in Input-Orientierung
- Passthrough-Semantik ohne Modell (§V6, copilot-instructions.md): bit-identisch, kein stiller Fallback
- NaN/Inf-Schutz (§0a, copilot-instructions.md) über Plugin-Ergebnis
- Factory get_phase() mit Strength-Passthrough
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_68_vinylnet_decrackle import VinylNetDeCracklePhase, get_phase
from backend.core.phases.phase_interface import PhaseCategory, PhaseMode


class _FakePlugin:
    """Minimal-Stub für den Plugin-Pfad (ohne Torch-Modell)."""

    available = True

    def __init__(self, out_fn) -> None:
        self._out_fn = out_fn

    def process(self, audio, sample_rate=44100, strength=0.9):
        return self._out_fn(audio), {"mask_samples": 7, "device": "cpu", "passthrough": False}


def test_metadata_contract() -> None:
    p = VinylNetDeCracklePhase()
    md = p.get_metadata()
    assert md.phase_id == "phase_68_vinylnet_decrackle"
    assert md.category == PhaseCategory.DEFECT_REMOVAL
    assert md.phase_mode == PhaseMode.RESTORATION_ONLY
    assert set(md.defect_types) >= {"clicks", "crackle"}


def test_passthrough_ohne_modell_bit_identisch(monkeypatch: pytest.MonkeyPatch) -> None:
    """§V6 (copilot-instructions.md): Modell nicht ladbar → bit-identischer Passthrough + Warning."""
    p = VinylNetDeCracklePhase()
    monkeypatch.setattr(p, "_get_plugin", lambda: None)
    rng = np.random.default_rng(0)
    x = (rng.random((2, 48000)) * 0.4 - 0.2).astype(np.float32)
    res = p.process(x, sample_rate=44100)
    assert res.success
    assert np.array_equal(res.audio, x)  # bit-identisch
    assert "modell_nicht_ladbar" in res.warnings

    xm = (rng.random(48000) * 0.4 - 0.2).astype(np.float32)
    res_m = p.process(xm, sample_rate=44100)
    assert np.array_equal(res_m.audio, xm)
    assert "modell_nicht_ladbar" in res_m.warnings


def test_layout_normalisierung_ohne_modell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stereo-Layout-Invariante: beide Layouts bedienen, Ausgabe in Input-Orientierung."""
    p = VinylNetDeCracklePhase()
    monkeypatch.setattr(p, "_get_plugin", lambda: None)
    rng = np.random.default_rng(1)
    x_cf = (rng.random((2, 24000)) * 0.4 - 0.2).astype(np.float32)
    assert p.process(x_cf, sample_rate=44100).audio.shape == (2, 24000)
    assert p.process(x_cf.T, sample_rate=44100).audio.shape == (24000, 2)
    assert p.process(x_cf[0], sample_rate=44100).audio.shape == (24000,)


def test_einkanalige_2d_layouts_bleiben_mono(monkeypatch: pytest.MonkeyPatch) -> None:
    """2D-Mono wird intern dupliziert und in der Eingangsform zurückgegeben."""
    p = VinylNetDeCracklePhase()
    monkeypatch.setattr(p, "_get_plugin", lambda: _FakePlugin(lambda a: a))
    mono = np.linspace(-0.2, 0.2, 48000, dtype=np.float32)

    for audio in (mono[None, :], mono[:, None]):
        result = p.process(audio, sample_rate=44100)
        assert result.audio.shape == audio.shape
        np.testing.assert_array_equal(result.audio, audio)


@pytest.mark.parametrize("shape", [(3, 12), (12, 3), (12, 1, 1)])
def test_unsupported_channel_layout_raises_instead_of_collapsing(shape: tuple[int, ...]) -> None:
    """Nicht-Mono-/Stereo-Layouts dürfen nicht still auf einen Kanal kollabieren."""
    with pytest.raises(ValueError, match="erwartet Mono oder Stereo"):
        VinylNetDeCracklePhase().process(np.zeros(shape, dtype=np.float32), sample_rate=44100)


def test_verarbeitung_mit_stub_deterministisch(monkeypatch: pytest.MonkeyPatch) -> None:
    """§G5 (copilot-instructions.md): gleicher Input → bit-identischer Output; ML-Metadaten gesetzt."""
    p = VinylNetDeCracklePhase()
    monkeypatch.setattr(p, "_get_plugin", lambda: _FakePlugin(lambda a: a * 0.5))
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((2, 48000)) * 0.1).astype(np.float32)
    res1 = p.process(x, sample_rate=44100)
    res2 = p.process(x, sample_rate=44100)
    assert res1.ml_used
    assert np.array_equal(res1.audio, res2.audio)  # deterministisch
    assert res1.metadata["strength"] == pytest.approx(0.9)
    assert res1.metadata["mask_samples"] == 7


def test_nan_inf_schutz(monkeypatch: pytest.MonkeyPatch) -> None:
    """§0a (copilot-instructions.md): nicht-finite Plugin-Ausgabe wird genullt, Ergebnis bleibt endlich."""
    p = VinylNetDeCracklePhase()

    def _noisy_out(a: np.ndarray) -> np.ndarray:
        out = a * 0.5
        out[0, 10] = np.nan
        out[1, 20] = -np.inf
        return out

    monkeypatch.setattr(p, "_get_plugin", lambda: _FakePlugin(_noisy_out))
    x = (np.random.default_rng(3).standard_normal((2, 48000)) * 0.1).astype(np.float32)
    res = p.process(x, sample_rate=44100)
    assert np.isfinite(res.audio).all()


def test_get_phase_factory() -> None:
    p_default = get_phase()
    assert isinstance(p_default, VinylNetDeCracklePhase)
    assert p_default.strength == pytest.approx(0.9)
    p_custom = get_phase(strength=0.7)
    assert p_custom.strength == pytest.approx(0.7)


def test_verarbeitung_mit_modell_falls_vorhanden() -> None:
    """Echter Plugin-Pfad (Torch, eval + inference_mode): deterministisch und endlich."""
    p = VinylNetDeCracklePhase()
    plugin = p._get_plugin()
    if plugin is None or not getattr(plugin, "available", False):
        pytest.skip("Modell nicht vorhanden (CI ohne Modelle) — Passthrough bereits abgedeckt")
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((2, 3 * 44100)) * 0.05).astype(np.float32)
    res1 = p.process(x, sample_rate=44100)
    res2 = p.process(x, sample_rate=44100)
    assert res1.ml_used
    assert res1.audio.shape == x.shape
    assert np.isfinite(res1.audio).all()
    assert np.array_equal(res1.audio, res2.audio)  # deterministisch (§G5, copilot-instructions.md)

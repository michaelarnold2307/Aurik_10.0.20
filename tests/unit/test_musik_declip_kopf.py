"""Vertragstests Musik-Declip-Kopf (backend/core/dsp/musik_declip_kopf.py)."""

from __future__ import annotations

import numpy as np
import pytest

SR = 48000


def _tone_clipped(n: int = SR) -> np.ndarray:
    t = np.arange(n) / SR
    return np.clip(np.sin(2 * np.pi * 440.0 * t) * 1.4, -0.9, 0.9).astype(np.float32)


def test_ohne_modell_fail_closed(tmp_path, monkeypatch):
    """Fehlendes Modell ⇒ None (Ersatzpfad, §V6 (copilot-instructions.md))."""
    from backend.core.dsp import musik_declip_kopf

    monkeypatch.setattr(musik_declip_kopf, "MODELL_PFAD", tmp_path / "fehlt.pt")
    musik_declip_kopf._singleton.clear()
    out = musik_declip_kopf.declip_musik_kopf(_tone_clipped(), SR, 0.9)
    assert out is None


def test_masken_blend_schuetzt_saubere_abschnitte(tmp_path, monkeypatch):
    """Außerhalb geclippter Regionen bleibt der Eingang bit-identisch (Musikschutz)."""
    from backend.core.dsp import musik_declip_kopf

    monkeypatch.setattr(musik_declip_kopf, "MODELL_PFAD", tmp_path / "fehlt.pt")
    musik_declip_kopf._singleton.clear()
    # Identity-Modell simulieren: ohne Modell gibt es None — hier direkt die
    # Masken-Logik über ein Identity-Monkeypatch des Kopfes prüfen.
    x = np.zeros(SR, dtype=np.float32)
    x[SR // 2] = 1.0  # einzelner geclippter Sample
    monkeypatch.setattr(musik_declip_kopf, "_lade_modell", lambda: object())
    monkeypatch.setattr(musik_declip_kopf, "_kopf_ola", lambda model, y: y * 0.5)
    out = musik_declip_kopf.declip_musik_kopf(x, SR, 0.9)
    assert out is not None
    assert float(np.abs(out[0] - x[0])) == 0.0  # sauberer Anfang unverändert
    assert float(np.abs(out[SR // 2])) < 1.0  # geclippte Stelle angefasst


def test_ungueltige_rate_fail_closed(tmp_path, monkeypatch):
    from backend.core.dsp import musik_declip_kopf

    monkeypatch.setattr(musik_declip_kopf, "MODELL_PFAD", tmp_path / "fehlt.pt")
    musik_declip_kopf._singleton.clear()
    assert musik_declip_kopf.declip_musik_kopf(_tone_clipped(1000), 0, 0.9) is None


def test_determinismus_beim_ola(tmp_path, monkeypatch):
    """Overlap-Add ist deterministisch (§G5 (copilot-instructions.md))."""
    from backend.core.dsp import musik_declip_kopf

    class _Dummy:
        def __call__(self, t):  # type: ignore[no-untyped-def]
            import torch

            return torch.zeros_like(t)

    x = _tone_clipped(16384)
    monkeypatch.setattr(musik_declip_kopf, "_lade_modell", lambda: _Dummy())
    a = musik_declip_kopf._kopf_ola(_Dummy(), x)
    b = musik_declip_kopf._kopf_ola(_Dummy(), x)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("theta", [0.05, 0.9, 0.999])
def test_schwelle_clamping(tmp_path, monkeypatch, theta):
    from backend.core.dsp import musik_declip_kopf

    monkeypatch.setattr(musik_declip_kopf, "MODELL_PFAD", tmp_path / "fehlt.pt")
    musik_declip_kopf._singleton.clear()
    monkeypatch.setattr(musik_declip_kopf, "_lade_modell", lambda: object())
    monkeypatch.setattr(musik_declip_kopf, "_kopf_ola", lambda model, y: y)
    out = musik_declip_kopf.declip_musik_kopf(_tone_clipped(), SR, theta)
    # Identity-Kopf + Witness (Harmonik identisch ⇒ nicht strikt besser) ⇒ None.
    assert out is None

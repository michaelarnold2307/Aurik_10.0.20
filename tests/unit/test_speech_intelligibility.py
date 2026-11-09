"""Vertragstests SII-Proxy (Hörordnung §3, L1 — Spec-SOTA-Audit 2026-11-09).

Modell-frei: alle Fälle laufen über synthetische Signale (§G5 (GEBOTE.md):
Tests dürfen nicht umgebungsabhängig kippen). Der Proxy ist ein
deterministischer Band-Energie-SII (kein voller STI — dokumentiert).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.speech_intelligibility import (
    SII_JND_VORBELEGUNG,
    eingang_brauchbar,
    sii_delta,
    sii_index,
)


def _band_limited(sr: int, dur_s: float, lo_hz: float, hi_hz: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(int(sr * dur_s))
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(int(sr * dur_s), 1 / sr)
    spec[(f < lo_hz) | (f > hi_hz)] = 0.0
    out = np.fft.irfft(spec, n=int(sr * dur_s))
    return (out / (np.abs(out).max() + 1e-9) * 0.3).astype(np.float32)


def test_sii_richtung_tiefpass_senkt() -> None:
    """Bandbegrenzung auf 2 kHz senkt die verständlichkeitsrelevante Band-Energie."""
    sr = 48000
    full = _band_limited(sr, 6.0, 100.0, 8000.0)
    tp = _band_limited(sr, 6.0, 100.0, 2000.0)
    s_full = sii_index(full, sr)
    s_tp = sii_index(tp, sr)
    assert s_full > s_tp + 0.15, f"voll {s_full:.3f} !> tiefpass {s_tp:.3f} + 0.15"


def test_sii_delta_identisch_ist_null() -> None:
    sr = 48000
    sig = _band_limited(sr, 6.0, 100.0, 8000.0)
    d = sii_delta(sig, sig, sr)
    assert d["delta"] == 0.0
    assert d["verschlechterung"] is False


def test_sii_delta_tiefpass_ist_verschlechterung() -> None:
    sr = 48000
    full = _band_limited(sr, 6.0, 100.0, 8000.0)
    tp = _band_limited(sr, 6.0, 100.0, 2000.0)
    d = sii_delta(full, tp, sr)
    assert d["delta"] < -SII_JND_VORBELEGUNG
    assert d["verschlechterung"] is True
    assert d["panel_status"] == "vorbelegung_hoerpanel_offen"


def test_sii_determinismus() -> None:
    sr = 48000
    sig = _band_limited(sr, 4.0, 100.0, 6000.0, seed=7)
    assert sii_index(sig, sr) == pytest.approx(sii_index(sig, sr), abs=1e-9)


def test_sii_schweigen_ist_null() -> None:
    sr = 48000
    assert sii_index(np.zeros(sr * 2, dtype=np.float32), sr) == 0.0
    d = sii_delta(np.zeros(sr * 2, dtype=np.float32), np.zeros(sr * 2, dtype=np.float32), sr)
    assert d["verschlechterung"] is False  # Schweigen vs. Schweigen: kein Befund


def test_sii_layout_sicher() -> None:
    """Stereo-Layout-Invariante: (C,N) und (N,C) liefern denselben Wert."""
    sr = 48000
    mono = _band_limited(sr, 4.0, 100.0, 8000.0, seed=3)
    stereo_cn = np.stack([mono, mono])  # (2, N)
    stereo_nc = np.stack([mono, mono], axis=1)  # (N, 2)
    a = sii_index(stereo_cn, sr)
    b = sii_index(stereo_nc, sr)
    assert a == pytest.approx(b, abs=1e-6)


def test_sii_zeitmittelung_nicht_blind_bei_stille_anfang() -> None:
    """Regression: ein Lauf, der mit Stille beginnt, darf nicht SII=0 liefern.

    Die erste Fassung maß EIN 2048-Sample-Fenster (42 ms) — bei einem
    Stille-/Fade-Anfang war der Zeuge blind (beide Seiten 0 ⇒ kein Befund).
    """
    sr = 48000
    stille = np.zeros(sr, dtype=np.float32)
    ton = _band_limited(sr, 3.0, 300.0, 6000.0, seed=11)
    sig = np.concatenate([stille, ton])
    assert sii_index(sig, sr) > 0.2


def test_eingang_brauchbar_weist_none_und_kurzes_ab() -> None:
    """§V6 (copilot-instructions.md): ein ausgefallener Messpfad darf nicht als „verfuegbar“ gelten."""
    sr = 48000
    gut = _band_limited(sr, 1.0, 300.0, 6000.0, seed=5)
    assert eingang_brauchbar(gut) == (True, "")
    assert eingang_brauchbar(None)[0] is False
    assert eingang_brauchbar("kein audio")[0] is False
    assert eingang_brauchbar(np.zeros(64))[0] is False


def test_uv3_verdrahtung_drei_produktionspfade() -> None:
    """Wirkungs-Pin (Muster C): der SII-Befund landet in ALLEN drei Pfaden.

    Ohne diesen Träger wäre die Ebene-1-Norm eine Phantom-Norm (Fehlerklasse
    D-K3-73/D-K3-83): kalibriert, getestet — und im Produktionspfad unsichtbar.
    """
    from pathlib import Path

    quelle = (Path(__file__).resolve().parents[2] / "backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
    assert "def _sii_zeuge_meta(" in quelle
    assert quelle.count("self._sii_zeuge_meta(") == 3
    assert 'result.metadata["sii_zeuge"]' in quelle
    assert '_meta["sii_zeuge"]' in quelle
    # REPORT-ONLY: genau DREI Schreibstellen, keine Lesestelle (kein Gate
    # entscheidet über den Zeugen — die Hör-Instanz entscheidet).
    assert quelle.count('"sii_zeuge"') == 3

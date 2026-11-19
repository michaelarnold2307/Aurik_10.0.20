#!/usr/bin/env python3
"""Vertragstests fuer scripts/messe_air_presence_wirkung.py (modell-frei).

Zitierdisziplin AGENTS.md §5. D-K3-98-Folgemessung: der NOLA-/Hard-Clip-Fix
ist synthetisch verifiziert, die WIRKUNG auf die D-K3-95-Rezepte wird mit dem
Messskript belegt. Diese Tests pinnen die Mess-Mechanik, nicht die Messwerte.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.messe_air_presence_wirkung as mess

_SR = mess.SR


def _schreibe_int16_wav(pfad: Path, audio: np.ndarray) -> None:
    from scipy.io import wavfile

    daten = np.clip(audio, -1.0, 1.0)
    wavfile.write(pfad, _SR, (daten * 32767.0).astype(np.int16))


def test_lade_proben_normalisiert_int16_und_ignoriert_restauriert(tmp_path) -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, (_SR // 4, 2)).astype(np.float64)
    _schreibe_int16_wav(tmp_path / "Song (clipping).wav", x)
    _schreibe_int16_wav(tmp_path / "Song (clipping)_restauriert.wav", x)
    _schreibe_int16_wav(tmp_path / "Song (vinyl).wav", x)

    proben = mess._lade_proben(tmp_path, ("clipping",))
    assert len(proben) == 1, "nur die degradierte Probe, kein _restauriert-Pendant"
    audio = proben[0]["audio"]
    assert audio.dtype == np.float32
    assert float(np.abs(audio).max()) <= 1.0
    assert float(np.abs(audio).max()) > 0.1, "Int16 muss auf [-1, 1] normalisiert sein"


def test_enhancer_kern_haelt_eingangsdecke() -> None:
    """D-K3-98-Vertrag: max|out| <= min(0,999, max|in|) — kein neues Flat-Top.

    Heisses Stereo-Signal mit Luftband-Energie am Peak: die Eingangs-Decke
    muss halten, egal ob die Anhebung greift oder zurueckgenommen wird.
    """
    t = np.arange(_SR) / _SR
    l = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.55 * np.sin(2 * np.pi * 12000 * t)
    x = np.clip(l, -0.95, 0.95)
    x = np.stack([x, 0.9 * x], axis=-1).astype(np.float32)

    befund, out = mess._enhancer_kern(x)
    assert befund["decke_ok"] is True
    assert befund["decke_out"] <= befund["decke_in"] + 1e-7
    assert float(np.abs(out).max()) <= float(np.abs(x).max()) + 1e-6
    assert set(befund) >= {
        "applied",
        "witness_ok",
        "hnr_drop_db",
        "pitch_drift_cents",
        "flat_top_rise",
        "decke_in",
        "decke_out",
        "decke_ok",
        "rms_delta",
    }


def test_delta_metriken_jnd_vertrag() -> None:
    """Never-worsen-Semantik: JND 0,05 (eine Quelle: wohlklang_zeuge)."""
    t = np.arange(_SR // 2) / _SR
    basis = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    identisch = basis.copy()
    delta = mess._delta_metriken(basis, identisch)
    assert delta["musik_mos_delta"] == 0.0
    assert delta["defektlast_delta"] == 0.0
    assert delta["nie_schlechter_als_jnd"] is True


def test_schema_und_zusammenfassung() -> None:
    """Beleg-Schema aurik.air_presence_wirkung/1 mit den Pflichtfeldern."""
    befund = {
        "schema": "aurik.air_presence_wirkung/1",
        "datum": "2026-11-12",
        "quelle": "scripts/messe_air_presence_wirkung.py",
        "proben_anzahl": 0,
        "rezepte": ["clipping", "kassette"],
        "panns_singing": None,
        "zeilen": [],
        "zusammenfassung": {
            "kern_engagiert": 0,
            "kern_witness_ok": 0,
            "kern_decke_ok": 0,
            "kern_nie_schlechter": 0,
        },
    }
    assert json.loads(json.dumps(befund, ensure_ascii=False)) == befund
    assert befund["schema"] == "aurik.air_presence_wirkung/1"
    assert befund["zusammenfassung"]["kern_decke_ok"] == 0

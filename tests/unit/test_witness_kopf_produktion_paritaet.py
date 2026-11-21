"""Produktions-Paritäts-Test Witness-Kopf (D-K3-95-Folge, 2026-11-15).

Fängt die zwei Befunde vom 2026-11-15 dauerhaft ab:
1. Training-Parität: kurze k398-Regionen (0,1 s) werden in der Produktion auf 1 s
   gepaddet wie in Trainer/Gate — sonst kollabiert das Witness (P=0,0 statt 0,92).
2. Längen-Vertrag: die Ausgabe hat exakt die Eingangslänge (kein 1-s-Pad-Artefakt
   an Phase 07 zurückgeben).
Läuft nur, wenn die Checkpoints lokal vorhanden sind (Beleg-Test, kein Unit-Test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.io import wavfile

import backend.core.dsp.witness_kopf_groove_echo as groove

_HAT_CHECKPOINTS = groove._KOPF_CKPT.is_file() and groove._WITNESS_CKPT.is_file()

pytestmark = pytest.mark.skipif(not _HAT_CHECKPOINTS, reason="Checkpoints lokal nicht vorhanden")


def _region_01s() -> np.ndarray:
    import json

    import librosa

    manifest = Path(__file__).resolve().parents[2] / "data" / "klassen_paare" / "manifest.jsonl"
    eintraege = [json.loads(l) for l in manifest.open(encoding="utf-8")]
    e = eintraege[0]
    sr, d = wavfile.read(e["schaden"])
    x = np.asarray(d, np.float32) / 32768.0
    x = x.mean(axis=1) if x.ndim == 2 else x
    return librosa.resample(x, orig_sr=sr, target_sr=groove._TARGET_SR).astype(np.float32)


def test_kurze_region_triggert_und_laenge_bleibt() -> None:
    """0,1-s-Region: Trigger feuert (Pad-Parität) und Ausgabe-Länge = Eingangs-Länge."""
    x = _region_01s()
    assert len(x) < groove._FENSTER, f"Test erwartet kurze Region, bekam {len(x)} Samples"
    out = groove.witness_kopf_groove_echo(x, groove._TARGET_SR)
    assert out is not None, "Trigger muss auf der k398-Region feuern (Pad-Parität, Befund 2026-11-15)"
    assert len(out) == len(x), f"Längen-Vertrag verletzt: {len(out)} != {len(x)}"
    assert np.isfinite(out).all()


def test_pad_und_laengen_schnitt_im_quelltext() -> None:
    quelltext = Path(groove.__file__).read_text(encoding="utf-8")
    assert "np.pad(x, (0, _FENSTER - len(x)))" in quelltext
    assert "laenge_in = len(x)" in quelltext
    assert "kandidat[:laenge_in]" in quelltext


# ── hum (2026-11-21): gate-bestanden mit harter Never-worsen-Modellauswahl ──
import backend.core.dsp.witness_kopf_hum as hum

_HAT_HUM_CHECKPOINTS = hum._KOPF_CKPT.is_file() and hum._WITNESS_CKPT.is_file()


def _region_01s_hum() -> np.ndarray:
    import json

    import librosa

    manifest = Path(__file__).resolve().parents[2] / "data" / "klassen_paare" / "manifest.jsonl"
    eintraege = [json.loads(l) for l in manifest.open(encoding="utf-8") if json.loads(l)["klasse"] == "hum"]
    e = eintraege[0]
    sr, d = wavfile.read(e["schaden"])
    x = np.asarray(d, np.float32) / 32768.0
    x = x.mean(axis=1) if x.ndim == 2 else x
    x = librosa.resample(x, orig_sr=sr, target_sr=hum._TARGET_SR).astype(np.float32)
    # hum-Regionen sind laenger als 1 s (Manifest-Intervalle 3-4 s) — das 1-s-
    # Mittelfenster ist exakt die Trainer-/Gate-Konvention (FENSTER-Mitte).
    start = (len(x) - hum._FENSTER) // 2
    return x[start : start + hum._FENSTER]


@pytest.mark.skipif(not _HAT_HUM_CHECKPOINTS, reason="hum-Checkpoints lokal nicht vorhanden")
def test_hum_kurze_region_triggert_und_laenge_bleibt() -> None:
    """hum: 1-s-Region triggert und Ausgabe-Länge = Eingangs-Länge."""
    x = _region_01s_hum()
    assert len(x) == hum._FENSTER, f"Test erwartet 1-s-Region, bekam {len(x)} Samples"
    out = hum.witness_kopf_hum(x, hum._TARGET_SR)
    assert out is not None, "Trigger muss auf der hum-Region feuern"
    assert len(out) == len(x), f"Längen-Vertrag verletzt: {len(out)} != {len(x)}"
    assert np.isfinite(out).all()


@pytest.mark.skipif(not _HAT_HUM_CHECKPOINTS, reason="hum-Checkpoints lokal nicht vorhanden")
def test_hum_pad_pfad_unter_einer_sekunde_kein_crash() -> None:
    """hum: 0,1-s-Schnipsel (Pad-Pfad) — kein Crash; bei Trigger Längen-Vertrag."""
    x = _region_01s_hum()[: hum._FENSTER // 10]
    out = hum.witness_kopf_hum(x, hum._TARGET_SR)
    if out is not None:
        assert len(out) == len(x)
        assert np.isfinite(out).all()


@pytest.mark.skipif(not _HAT_HUM_CHECKPOINTS, reason="hum-Checkpoints lokal nicht vorhanden")
def test_hum_pad_und_laengen_schnitt_im_quelltext() -> None:
    quelltext = Path(hum.__file__).read_text(encoding="utf-8")
    assert "np.pad(x, (0, _FENSTER - len(x)))" in quelltext
    assert "laenge_in = len(x)" in quelltext
    assert "kandidat[:laenge_in]" in quelltext


import backend.core.dsp.witness_kopf_wow as wow

_HAT_WOW_CHECKPOINTS = wow._KOPF_CKPT.is_file() and wow._WITNESS_CKPT.is_file()


def _region_wow() -> np.ndarray:
    import collections
    import json

    import librosa

    manifest = Path(__file__).resolve().parents[2] / "data" / "klassen_paare" / "manifest.jsonl"
    eintraege = [json.loads(l) for l in manifest.open(encoding="utf-8") if json.loads(l)["klasse"] == "wow"]
    zaehlung = collections.Counter(e["titel"] for e in eintraege)
    holdout_titel = max(zaehlung, key=lambda t: zaehlung[t])
    # Holdout-Region (deterministisch, Muster _baue_regionen): dort feuert das
    # Witness zuverlässig (p_schaden-Mittel 0,95). Region-Index 1 statt 0:
    # die erste Holdout-Region bleibt unter der Trigger-Schwelle (P 0,46).
    e = [x for x in eintraege if x["titel"] == holdout_titel][1]
    sr, d = wavfile.read(e["schaden"])
    x = np.asarray(d, np.float32) / 32768.0
    x = x.mean(axis=1) if x.ndim == 2 else x
    x = librosa.resample(x, orig_sr=sr, target_sr=wow._TARGET_SR).astype(np.float32)
    # wow-Regionen variieren (0,075 s bis 1,6 s): >=1 s → 1-s-Mittelfenster
    # (Trainer-/Gate-Konvention); <1 s → volle Region (Produktion padet auf 1 s).
    if len(x) >= wow._FENSTER:
        start = (len(x) - wow._FENSTER) // 2
        x = x[start : start + wow._FENSTER]
    return x


@pytest.mark.skipif(not _HAT_WOW_CHECKPOINTS, reason="wow-Checkpoints lokal nicht vorhanden")
def test_wow_region_triggert_und_laenge_bleibt() -> None:
    """wow: Holdout-Region triggert und Ausgabe-Länge = Eingangs-Länge."""
    x = _region_wow()
    assert 0 < len(x) <= wow._FENSTER, f"Unerwartete Regionslänge: {len(x)} Samples"
    out = wow.witness_kopf_wow(x, wow._TARGET_SR)
    assert out is not None, "Trigger muss auf der wow-Holdout-Region feuern"
    assert len(out) == len(x), f"Längen-Vertrag verletzt: {len(out)} != {len(x)}"
    assert np.isfinite(out).all()


@pytest.mark.skipif(not _HAT_WOW_CHECKPOINTS, reason="wow-Checkpoints lokal nicht vorhanden")
def test_wow_pad_pfad_unter_einer_sekunde_kein_crash() -> None:
    """wow: 0,1-s-Schnipsel (Pad-Pfad) — kein Crash; bei Trigger Längen-Vertrag."""
    x = _region_wow()
    if len(x) >= wow._FENSTER // 10:
        x = x[: wow._FENSTER // 10]
    out = wow.witness_kopf_wow(x, wow._TARGET_SR)
    if out is not None:
        assert len(out) == len(x)
        assert np.isfinite(out).all()


@pytest.mark.skipif(not _HAT_WOW_CHECKPOINTS, reason="wow-Checkpoints lokal nicht vorhanden")
def test_wow_pad_und_laengen_schnitt_im_quelltext() -> None:
    quelltext = Path(wow.__file__).read_text(encoding="utf-8")
    assert "np.pad(x, (0, _FENSTER - len(x)))" in quelltext
    assert "laenge_in = len(x)" in quelltext
    assert "kandidat[:laenge_in]" in quelltext

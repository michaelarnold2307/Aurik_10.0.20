"""Vertragstests für das D3-2-Gate (`scripts/verify_bigvgan_d3_gate.py`, D-K3-96).

Geprüft wird die **Mess- und Urteilslogik**, nicht das Modell: Pegel-Normierung,
Never-worsen-Toleranzen, Vorteils-Schwelle (≥ 2/3), Fail-closed bei nicht
angewandten Tracks, deterministische Track-Auswahl, CPU-Guard und die
Beleg-Pfad-Hygiene (kein Überschreiben, D-K3-85).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import verify_bigvgan_d3_gate as gate


def _rauschen(n: int = 48000, seed: int = 0, amplitude: float = 0.1) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n).astype(np.float32) * amplitude).astype(np.float32)


def _kandidat(
    *,
    af: float,
    hnr: float,
    mos: float,
    defekt: float,
    applied: bool = True,
    bands: int = 1,
) -> dict:
    return {
        "applied": applied,
        "bands_released": bands,
        "roh_norm": {"af": af, "hnr_db": hnr, "musik_mos": mos, "defektlast": defekt},
        "fenster_ausgang": {"af": af, "hnr_db": hnr, "musik_mos": mos, "defektlast": defekt},
    }


def _basis() -> dict:
    return _kandidat(af=0.60, hnr=0.0, mos=4.50, defekt=0.40)


def test_pegel_normiert_trifft_eingangs_rms_und_kappt_peak() -> None:
    eingang = _rauschen(seed=1, amplitude=0.2)
    ausgang = 0.5 * eingang
    norm = gate._pegel_normiert(ausgang, eingang)
    assert float(np.sqrt(np.mean(norm**2))) == pytest.approx(float(np.sqrt(np.mean(eingang**2))), rel=1e-4)
    laut = gate._pegel_normiert(10.0 * eingang, eingang)
    assert float(np.max(np.abs(laut))) <= 0.95 + 1e-6


def test_vergleich_never_worsen_grenzen() -> None:
    basis = _basis()
    # exakt an der Toleranz: af -0,02, HNR -0,5, mos -0,05, defektlast +0,05
    knapp = _kandidat(af=0.58, hnr=-0.5, mos=4.45, defekt=0.45)
    v = gate._vergleich(knapp, basis)
    assert v["never_worsen"]["af"] is True
    assert v["never_worsen"]["hnr_db"] is True
    assert v["never_worsen"]["musik_mos"] is True
    assert v["never_worsen"]["defektlast"] is True
    assert v["bestanden"] is True


def test_vergleich_erkennt_regression_und_vorteil() -> None:
    basis = _basis()
    schlecht = _kandidat(af=0.50, hnr=-1.0, mos=4.40, defekt=0.50)
    v = gate._vergleich(schlecht, basis)
    assert v["never_worsen"]["af"] is False
    assert v["bestanden"] is False

    besser = _kandidat(af=0.63, hnr=0.6, mos=4.56, defekt=0.34)
    v2 = gate._vergleich(besser, basis)
    assert v2["vorteil"] is True
    assert v2["bestanden"] is True


def test_vergleich_nicht_angewandt_ist_kein_bestehen() -> None:
    basis = _basis()
    k = _kandidat(af=0.90, hnr=2.0, mos=4.80, defekt=0.20, applied=False, bands=0)
    v = gate._vergleich(k, basis)
    assert v["vorteil"] is True  # die Rohsynthese ist besser ...
    assert v["bestanden"] is False  # ... aber ohne Produktionswirkung kein Deploy


def _tracks(*kandidaten: dict) -> list[dict]:
    return [{"track": f"T{i}", "varianten": {"f3_prod": k}} for i, k in enumerate(kandidaten)]


def test_gate_urteil_braucht_zwei_vorteils_tracks() -> None:
    basis = _basis()
    gut = _kandidat(af=0.63, hnr=0.6, mos=4.56, defekt=0.34)
    neutral = _kandidat(af=0.60, hnr=0.0, mos=4.50, defekt=0.40)
    tracks = _tracks(
        dict(gut, **gate._vergleich(gut, basis)),
        dict(gut, **gate._vergleich(gut, basis)),
        dict(neutral, **gate._vergleich(neutral, basis)),
    )
    urteil = gate._gate_urteil(tracks, "f3_prod")
    assert urteil["vorteil_tracks"] == 2
    assert urteil["bestanden"] is True


def test_gate_urteil_ein_vorteil_reicht_nicht() -> None:
    basis = _basis()
    gut = _kandidat(af=0.63, hnr=0.6, mos=4.56, defekt=0.34)
    neutral = _kandidat(af=0.60, hnr=0.0, mos=4.50, defekt=0.40)
    tracks = _tracks(
        dict(gut, **gate._vergleich(gut, basis)),
        dict(neutral, **gate._vergleich(neutral, basis)),
        dict(neutral, **gate._vergleich(neutral, basis)),
    )
    urteil = gate._gate_urteil(tracks, "f3_prod")
    assert urteil["vorteil_tracks"] == 1
    assert urteil["bestanden"] is False
    assert "Vorteil" in urteil["grund"]


def test_gate_urteil_fail_closed_bei_nicht_angewandt() -> None:
    basis = _basis()
    gut = _kandidat(af=0.63, hnr=0.6, mos=4.56, defekt=0.34)
    aus = _kandidat(af=0.63, hnr=0.6, mos=4.56, defekt=0.34, applied=False, bands=0)
    tracks = _tracks(
        dict(gut, **gate._vergleich(gut, basis)),
        dict(gut, **gate._vergleich(gut, basis)),
        dict(aus, **gate._vergleich(aus, basis)),
    )
    urteil = gate._gate_urteil(tracks, "f3_prod")
    assert urteil["angewendet_tracks"] == 2
    assert urteil["bestanden"] is False
    assert "angewandt" in urteil["grund"]


def test_fenster_grenzen_liefert_erstes_hr_v1_fenster() -> None:
    ws, we = gate._fenster_grenzen(10 * gate.SR, gate.SR)
    assert 0 <= ws < we <= 10 * gate.SR
    # 5-%-Deckel: das Fenster ist deutlich kürzer als der Track
    assert (we - ws) < 3 * gate.SR


def test_waehle_tracks_deterministisch() -> None:
    a = gate._waehle_tracks(3, 42)
    b = gate._waehle_tracks(3, 42)
    if not a:
        pytest.skip("MUSDB18HQ-Testset lokal nicht vorhanden")
    assert a == b
    assert a == sorted(a)
    assert len(a) == 3


def test_schema_und_toleranzen() -> None:
    assert gate.SCHEMA == "aurik.bigvgan_d3_gate/1"
    assert gate.AF_TOLERANZ == 0.02
    assert gate.HNR_TOLERANZ_DB == 0.5
    assert gate.JND == 0.05
    assert gate.BENEFIT_MIN_TRACKS == 2


def test_cpu_guard_fail_closed_bei_sichtbarer_gpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    grund = gate._cpu_guard()
    assert grund is not None and "GPU" in grund

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gate._cpu_guard() is None


def test_zielpfad_zerstoert_keinen_beleg(tmp_path) -> None:
    ziel = tmp_path / "beleg.json"
    ziel.write_text("alt", encoding="utf-8")
    assert gate._zielpfad(ziel, "alt") == ziel  # identische Wiederholung: idempotent
    assert gate._zielpfad(ziel, "neu") == tmp_path / "beleg_2.json"
    (tmp_path / "beleg_2.json").write_text("x", encoding="utf-8")
    assert gate._zielpfad(ziel, "neu") == tmp_path / "beleg_3.json"


def test_synth_statistik_zero_safe() -> None:
    s = gate.SynthStatistik()
    d = s.als_dict()
    assert d["aufrufe"] == 0
    assert d["rms_mittel"] == 0.0
    assert d["clip_anteil"] == 0.0


def test_trainer_gate_hint_verweist_auf_d3_gate() -> None:
    """D-K3-96: der Trainer darf kein ΔSDR-Gate mehr versprechen (Mel-Vocoder)."""
    quelle = (PROJECT_ROOT / "scripts" / "train_bigvgan_f3.py").read_text(encoding="utf-8")
    assert "verify_bigvgan_d3_gate.py" in quelle
    assert "ΔSDR ≥ +2 dB" not in quelle

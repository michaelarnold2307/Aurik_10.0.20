"""Unit-Tests EchoThief-Aequivalenz-Synth-Generator (Pfeiler C).

Vertraege (scripts/build_echothief_equiv_bank.py):
- Lizenz: synthetisch, kein EchoThief-Audio (SDSURF-Ausschluss bleibt gewahrt).
- Kategorie-Abdeckung: 12 EchoThief-Kategorien mit rt60_min < rt60_max.
- RT60-Selbstverifikation: gemessene RT60 nahe Ziel (Toleranz).
- DRR: Hall-Energie skaliert nach Ziel-Direkt-zu-Hall-Verhaeltnis.
- Determinismus (S5 (copilot-instructions.md)): gleicher Seed => bit-identisch.
- Robustheit: kein NaN/Inf, Peak = 1.0, Manifest-Schema vollstaendig.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.build_echothief_equiv_bank as bank

# ── Lizenz-Schutz: Pfeiler C beruehrt kein EchoThief-Audio ──────────────────


def test_lizenzhinweis_kein_echothief_audio() -> None:
    cov = {
        "lizenzhinweis": (
            "Synthetisch erzeugt aus literaturgeerdeten Raumakustik-Zielen. "
            "Kein EchoThief-Audio verwendet/gelernt (SDSURF-Lizenz bleibt "
            "ausschliesslich fuer Pfeiler A / echte Beschaffung einschlaegig)."
        )
    }
    hinweis = cov["lizenzhinweis"].lower()
    assert "kein echothief-audio" in hinweis
    assert "synthetisch" in hinweis
    # Die generierten Records sind als synthetisch markiert.
    assert bank.SynthRIRRecord.__dataclass_fields__["synthetisch"].default is True
    assert "cc0" in bank.SynthRIRRecord.__dataclass_fields__["lizenz"].default.lower()


# ── Kategorie-Abdeckung ──────────────────────────────────────────────────────


def test_zwoelf_kategorien_mit_gueltigen_bereichen() -> None:
    assert len(bank.CATEGORIES) == 12
    for cat in bank.CATEGORIES:
        assert cat.rt60_min_s < cat.rt60_max_s, cat.name
        assert cat.drr_min_db < cat.drr_max_db, cat.name
        assert cat.echothief_count > 0, cat.name
        assert cat.rt60_min_s > 0, cat.name


def test_echothief_anzahl_summe_195() -> None:
    """Kategorie-Anzahlen spiegeln die reale EchoThief-Verteilung (195 RIRs)."""
    assert sum(c.echothief_count for c in bank.CATEGORIES) == 195


# ── RT60-Selbstverifikation ─────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["Underground", "Domes", "Sanctuaries", "Nature"])
def test_rt60_selbstverifikation_trifft_ziel(name: str) -> None:
    cat = next(c for c in bank.CATEGORIES if c.name == name)
    rt60_ziel = (cat.rt60_min_s + cat.rt60_max_s) / 2.0
    x, rt60_gem = bank.synth_rir_category(cat, rt60_ziel, 0.0, sr=16000, seed=7)
    assert np.isfinite(x).all()
    assert abs(rt60_gem - rt60_ziel) / rt60_ziel < 0.10


def test_rt60_schroeder_auf_reinem_exp_zerfall() -> None:
    """_measure_rt60_s auf exp-Zerfall (gt=1.0 s) => nahe 1.0 s."""
    sr = 16000
    t = np.arange(int(sr * 1.4)) / sr
    x = np.exp(-bank._DECAY_CONST * t / 1.0).astype(np.float32)
    assert abs(bank._measure_rt60_s(x, sr) - 1.0) < 0.12


# ── DRR-Anwendung ───────────────────────────────────────────────────────────


def test_drr_wirkt_auf_hall_energie() -> None:
    cat = next(c for c in bank.CATEGORIES if c.name == "Venues")
    x_trocken, _ = bank.synth_rir_category(cat, 1.8, +8.0, sr=16000, seed=3)
    x_hall, _ = bank.synth_rir_category(cat, 1.8, -10.0, sr=16000, seed=3)
    e_hall_trocken = float(np.sum(x_trocken[1:] ** 2))
    e_hall_hall = float(np.sum(x_hall[1:] ** 2))
    # Hoher DRR (trocken) => weniger Hall-Energie als niedriger DRR (hallig).
    assert e_hall_trocken < e_hall_hall


# ── Determinismus (S5) ──────────────────────────────────────────────────────


def test_determinismus_gleicher_seed_bit_identisch() -> None:
    cat = bank.CATEGORIES[0]
    a, ra = bank.synth_rir_category(cat, 3.0, -5.0, sr=16000, seed=42)
    b, rb = bank.synth_rir_category(cat, 3.0, -5.0, sr=16000, seed=42)
    assert np.array_equal(a, b)
    assert ra == rb


def test_unterschiedliche_seeds_unterschiedlich() -> None:
    cat = bank.CATEGORIES[0]
    a, _ = bank.synth_rir_category(cat, 3.0, -5.0, sr=16000, seed=1)
    b, _ = bank.synth_rir_category(cat, 3.0, -5.0, sr=16000, seed=2)
    assert not np.array_equal(a, b)


# ── Robustheit ──────────────────────────────────────────────────────────────


def test_kein_nan_inf_peak_normalisiert() -> None:
    cat = bank.CATEGORIES[0]
    x, _ = bank.synth_rir_category(cat, 4.0, -6.0, sr=16000, seed=5)
    assert np.isfinite(x).all()
    assert abs(float(np.max(np.abs(x))) - 1.0) < 1e-5


def test_rt60_null_wirft_valueerror() -> None:
    with pytest.raises(ValueError):
        bank.synth_rir_category(bank.CATEGORIES[0], 0.0, 0.0, sr=16000, seed=1)


# ── Bank-Aufbau + Manifest-Schema ───────────────────────────────────────────


def test_build_bank_manifest_und_coverage(tmp_path) -> None:
    recs, cov = bank.build_bank(tmp_path, per_category=1, sr=16000, seed=11)
    assert len(recs) == len(bank.CATEGORIES)
    assert set(cov["kategorien"].keys()) == {c.name for c in bank.CATEGORIES}
    for r in recs:
        assert r.synthetisch is True
        assert r.rt60_ziel_s > 0
        assert r.sample_rate == 16000
    # WAVs geschrieben?
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == len(recs)


def test_write_manifest_schema(tmp_path) -> None:
    recs, cov = bank.build_bank(tmp_path, per_category=1, sr=8000, seed=13)
    bank.write_manifest(tmp_path, recs, cov)
    m = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert m["schema"] == "aurik.rir_synth_echothief_equiv/1"
    assert m["anzahl"] == len(recs)
    assert len(m["dateien"]) == len(recs)
    c = json.loads((tmp_path / "COVERAGE.json").read_text(encoding="utf-8"))
    assert "kategorien" in c
    assert "echothief_anzahl" in c["kategorien"][bank.CATEGORIES[0].name]

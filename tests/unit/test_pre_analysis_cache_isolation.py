"""Vertragstests der Pre-Analyse-Cache-Isolation (2026-11-25, D-K3-112).

Produktionsbefund: Der content-adressierte Bridge-Cache lieferte MUTIERTE
Analyse-Objekte an Folge-Läufe desselben Inhalts zurück (2. Lauf desselben
Songs = Pass-Through; B1≠B2 in der DRT-12-Kampagne). Fix: `copy.deepcopy`
auf Store- UND Load-Seite — der Cache hält die PRISTINE Analyse, jeder Lauf
erhält eine private Kopie (§G5, §V8-Kontext-Isolation). Diese Tests pinnen
die Isolation ohne Modell/Pipeline (modell-frei).
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

from backend.core.pre_analysis import _load_cached_parts, _store_in_cache


def _fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        medium={"chain": ["vinyl"], "is_multi_generation": False},
        era={"decade": 1970, "confidence": 0.8},
        genre={"genre_label": "rock"},
        defects={"clipping": {"severity": 0.5}},
        restorability={"score": 0.7},
    )


def _pfad(tmp_path: Path, name: str = "test_song.wav") -> str:
    # Datei darf fehlen — content_cache_key fällt auf den Pfad zurück.
    return str(tmp_path / name)


def test_store_haelt_pristine_kopie(tmp_path):
    pfad = _pfad(tmp_path)
    res = _fake_result()
    _store_in_cache(pfad, res)
    # Original NACH dem Store mutieren (wie es die Restauration tut)
    res.medium["chain"].append("tape")
    res.defects["clipping"]["severity"] = 0.0
    teile = _load_cached_parts(pfad)
    assert teile["medium"]["chain"] == ["vinyl"]
    assert teile["defects"]["clipping"]["severity"] == 0.5


def test_load_liefert_private_kopie(tmp_path):
    pfad = _pfad(tmp_path)
    _store_in_cache(pfad, _fake_result())
    erste = _load_cached_parts(pfad)
    erste["defects"]["clipping"]["severity"] = 0.0
    erste["medium"]["chain"].append("mutiert")
    zweite = _load_cached_parts(pfad)
    assert zweite["defects"]["clipping"]["severity"] == 0.5
    assert zweite["medium"]["chain"] == ["vinyl"]


def test_era_genre_isoliert(tmp_path):
    pfad = _pfad(tmp_path)
    _store_in_cache(pfad, _fake_result())
    erste = _load_cached_parts(pfad)
    erste["era"]["decade"] = 1980
    zweite = _load_cached_parts(pfad)
    assert zweite["era"]["decade"] == 1970
    assert zweite["genre"]["genre_label"] == "rock"


def test_unbekannter_pfad_liefert_leer_ohne_crash(tmp_path):
    teile = _load_cached_parts(str(tmp_path / "nie_dagewesen.wav"))
    assert isinstance(teile, dict)

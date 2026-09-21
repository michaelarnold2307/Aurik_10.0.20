"""Disk-Persistenz des Bridge-Analyse-Caches (prozessübergreifend, §G5 (copilot-instructions.md)).

Prüft Read-/Write-Through unter dem In-Memory-LRU: Ergebnis-Objekte
überleben Prozessgrenzen bit-exakt (pickle), Versions-/Schema-Wechsel und
korrupte Dateien werden sichtbar verworfen (§V6 (copilot-instructions.md)),
der Kill-Switch (AURIK_ANALYSIS_CACHE=0) deaktiviert die Platte, Clear
räumt auch die Platte auf.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest


@dataclasses.dataclass
class _FakeResult:
    label: str
    scores: np.ndarray
    meta: dict


@pytest.fixture
def bc(monkeypatch, tmp_path):
    import backend.api.bridge_cache as _bc

    monkeypatch.setattr(_bc, "_disk_cache_root", lambda: tmp_path / "analysis_cache")
    monkeypatch.setattr(_bc, "_DISK_CACHE_DISABLED", False)
    monkeypatch.setattr(_bc, "_analysis_cache_version", lambda: "9.9.9-test")
    _bc._defect_lru.clear()
    _bc._era_genre_lru.clear()
    _bc._medium_lru.clear()
    _bc._restorability_lru.clear()
    return _bc


def _audio_file(tmp_path, name: str = "song.wav") -> str:
    p = tmp_path / name
    p.write_bytes(b"RIFF" + bytes(range(256)) * 64)
    return str(p)


def test_defect_disk_roundtrip_across_lru_generation(bc, tmp_path):
    """Write-Through + Read-Through: Ergebnis überlebt geleerte LRU bit-exakt."""
    path = _audio_file(tmp_path)
    res = _FakeResult("vinyl", np.arange(6, dtype=np.float32).reshape(2, 3), {"a": 1})
    bc.cache_defect_result(path, res)
    bc._defect_lru.clear()  # simuliert einen neuen Prozess (Memory weg)
    got = bc.get_cached_defect_result(path)
    assert got is not None
    assert got.label == "vinyl"
    np.testing.assert_array_equal(got.scores, res.scores)
    assert got.meta == {"a": 1}
    assert any((tmp_path / "analysis_cache" / "defect").glob("*.pkl"))


def test_era_genre_dict_roundtrip(bc, tmp_path):
    path = _audio_file(tmp_path)
    bc.cache_era_genre_result(path, era_result={"era": "70s"}, genre_result={"g": 1})
    bc._era_genre_lru.clear()
    got = bc.get_cached_era_genre_result(path)
    assert got == {"era_result": {"era": "70s"}, "genre_result": {"g": 1}}


def test_version_mismatch_invalidates_and_removes(bc, tmp_path):
    """Versionswechsel im Key (§G5 (copilot-instructions.md)) → Eintrag verworfen und gelöscht."""
    path = _audio_file(tmp_path)
    bc.cache_medium_result(path, _FakeResult("x", np.zeros(2), {}))
    bc._medium_lru.clear()
    bc._analysis_cache_version = lambda: "9.9.9-neu"  # type: ignore[assignment]
    got = bc.get_cached_medium_result(path)
    assert got is None
    assert not any((tmp_path / "analysis_cache" / "medium").glob("*.pkl"))


def test_corrupt_file_returns_none_warns_and_removes(bc, tmp_path, caplog):
    """Korrupter Eintrag → None + sichtbare §V6-Warnung + Löschung."""
    path = _audio_file(tmp_path)
    bc.cache_restorability_result(path, _FakeResult("x", np.zeros(2), {}))
    bc._restorability_lru.clear()
    key = bc.content_cache_key(path)
    f = bc._disk_path("restorability", key)
    f.write_bytes(b"\x00\x01garbage")
    with caplog.at_level(logging.WARNING):
        got = bc.get_cached_restorability_result(path)
    assert got is None
    assert not f.exists()
    assert any(("Platte" in r.message or "fehlgeschlagen" in r.message) for r in caplog.records), (
        "Korrupter Eintrag muss sichtbar warnen (§V6 (copilot-instructions.md))"
    )


def test_disabled_switch_no_disk_io(bc, tmp_path):
    """Kill-Switch: kein Platten-I/O, Memory-LRU unverändert nutzbar."""
    bc._DISK_CACHE_DISABLED = True
    path = _audio_file(tmp_path)
    bc.cache_medium_result(path, _FakeResult("x", np.zeros(2), {}))
    assert not (tmp_path / "analysis_cache").exists()
    bc._medium_lru.clear()
    bc._DISK_CACHE_DISABLED = False
    assert bc.get_cached_medium_result(path) is None  # nichts auf Platte


def test_clear_removes_disk_entries(bc, tmp_path):
    path = _audio_file(tmp_path)
    bc.cache_defect_result(path, _FakeResult("x", np.zeros(2), {}))
    assert any((tmp_path / "analysis_cache" / "defect").glob("*.pkl"))
    bc.clear_defect_cache(path)
    assert not any((tmp_path / "analysis_cache" / "defect").glob("*.pkl"))

    bc.cache_medium_result(path, _FakeResult("x", np.zeros(2), {}))
    bc.clear_medium_cache()
    assert not (tmp_path / "analysis_cache" / "medium").exists()

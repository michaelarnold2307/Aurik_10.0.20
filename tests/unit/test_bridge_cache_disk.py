"""Disk-Persistenz des Bridge-Analyse-Caches (prozessübergreifend, §G5 (copilot-instructions.md)).

Prüft Read-/Write-Through unter dem In-Memory-LRU: Ergebnis-Objekte
überleben Prozessgrenzen bit-exakt (JSON), Versions-/Schema-Wechsel und
korrupte Dateien werden sichtbar verworfen (§V6 (copilot-instructions.md)),
der Kill-Switch (AURIK_ANALYSIS_CACHE=0) deaktiviert die Platte, Clear
räumt auch die Platte auf.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


def _touch_marker(path: str) -> None:
    Path(path).touch()


class _UnpicklePayload:
    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self):
        return _touch_marker, (self.path,)


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


def test_default_cache_root_is_per_user(monkeypatch, tmp_path):
    import backend.api.bridge_cache as bc

    monkeypatch.delenv("AURIK_ANALYSIS_CACHE_DIR", raising=False)
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        expected = tmp_path / "local" / "Aurik" / "Cache" / "analysis_cache"
    elif sys.platform == "darwin":
        expected = Path.home() / "Library" / "Caches" / "Aurik" / "analysis_cache"
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        expected = tmp_path / "xdg" / "aurik" / "analysis_cache"

    assert bc._disk_cache_root() == expected


def test_defect_disk_roundtrip_across_lru_generation(bc, tmp_path):
    """Write-Through + Read-Through: Ergebnis überlebt geleerte LRU bit-exakt."""
    path = _audio_file(tmp_path)
    res = {"label": "vinyl", "scores": np.arange(6, dtype=np.float32).reshape(2, 3), "meta": {"a": 1}}
    bc.cache_defect_result(path, res)
    bc._defect_lru.clear()  # simuliert einen neuen Prozess (Memory weg)
    got = bc.get_cached_defect_result(path)
    assert got is not None
    assert got["label"] == "vinyl"
    np.testing.assert_array_equal(got["scores"], res["scores"])
    assert got["meta"] == {"a": 1}
    assert any((tmp_path / "analysis_cache" / "defect").glob("*.json"))


def test_allowlisted_dataclass_roundtrip(bc, tmp_path):
    from backend.core.restorability_estimator import RestorabilityResult

    path = _audio_file(tmp_path)
    result = RestorabilityResult(75.0, 3.8, (3.4, 4.0), ["hiss"], ["Schonend restaurieren"], 12.0, snr_db=28.0)
    bc.cache_restorability_result(path, result)
    bc._restorability_lru.clear()

    restored = bc.get_cached_restorability_result(path)

    assert isinstance(restored, RestorabilityResult)
    assert restored == result


def test_preanalysis_result_types_roundtrip_without_dynamic_imports(bc, tmp_path):
    from backend.core.defect_scanner import (
        DefectAnalysisResult,
        DefectLocationsView,
        DefectScore,
        DefectScoreView,
        DefectType,
        MaterialType,
    )
    from backend.core.era_classifier import EraResult
    from backend.core.genre_classifier import SchlagerClassificationResult
    from backend.core.medium_classifier import ClassificationResult, MaterialEvidence
    from forensics.medium_detector import MediumDetectionResult, SpectralFingerprint, TransferChain

    path = _audio_file(tmp_path)
    result = {
        "medium": MediumDetectionResult(
            ["vinyl"],
            False,
            "vinyl",
            0.9,
            SpectralFingerprint(),
            classification_result=ClassificationResult(
                MaterialType.VINYL,
                0.9,
                evidence=[MaterialEvidence(MaterialType.VINYL, 0.9)],
            ),
        ),
        "transfer_chain": TransferChain(["vinyl"], False, "vinyl", 0.9),
        "era": EraResult(1970, "1970er", 0.9, "vinyl"),
        "genre": SchlagerClassificationResult(False, 0.1, "rock", "classic", 120.0),
        "defects": DefectAnalysisResult(
            MaterialType.VINYL,
            DefectScoreView({DefectType.CLICKS: DefectScore(DefectType.CLICKS, 0.5, 0.9, locations=[(0.1, 0.2)])}),
            0.1,
            44100,
            2.0,
        ),
        "locations": DefectLocationsView({"clicks": [(0.1, 0.2)]}),
    }
    bc.cache_defect_result(path, result)
    bc._defect_lru.clear()

    restored = bc.get_cached_defect_result(path)

    assert restored["medium"].classification_result.evidence[0].material is MaterialType.VINYL
    assert restored["transfer_chain"].chain == ["vinyl"]
    assert restored["era"].noise_profile.shape == (24,)
    assert restored["genre"].genre_label == "rock"
    assert restored["defects"].scores[DefectType.CLICKS].severity == 0.5
    assert restored["locations"]["clicks"] == [(0.1, 0.2)]


def test_legacy_pickle_is_never_deserialized(bc, tmp_path):
    path = _audio_file(tmp_path)
    marker = tmp_path / "executed"
    target = bc._disk_path("medium", bc.content_cache_key(path)).with_suffix(".pkl")
    bc._prepare_disk_cache_dir("medium", create=True)
    target.write_bytes(pickle.dumps(_UnpicklePayload(str(marker))))

    assert bc.get_cached_medium_result(path) is None
    assert not marker.exists()
    assert not target.exists()


def test_era_genre_dict_roundtrip(bc, tmp_path):
    path = _audio_file(tmp_path)
    bc.cache_era_genre_result(path, era_result={"era": "70s"}, genre_result={"g": 1})
    bc._era_genre_lru.clear()
    got = bc.get_cached_era_genre_result(path)
    assert got == {"era_result": {"era": "70s"}, "genre_result": {"g": 1}}


def test_version_mismatch_invalidates_and_removes(bc, tmp_path):
    """Versionswechsel im Key (§G5 (copilot-instructions.md)) → Eintrag verworfen und gelöscht."""
    path = _audio_file(tmp_path)
    bc.cache_medium_result(path, {"label": "x", "scores": np.zeros(2)})
    bc._medium_lru.clear()
    bc._analysis_cache_version = lambda: "9.9.9-neu"  # type: ignore[assignment]
    got = bc.get_cached_medium_result(path)
    assert got is None
    assert not any((tmp_path / "analysis_cache" / "medium").glob("*.json"))


def test_corrupt_file_returns_none_warns_and_removes(bc, tmp_path, caplog):
    """Korrupter Eintrag → None + sichtbare INFO-Meldung + Löschung."""
    path = _audio_file(tmp_path)
    bc.cache_restorability_result(path, {"label": "x", "scores": np.zeros(2)})
    bc._restorability_lru.clear()
    key = bc.content_cache_key(path)
    f = bc._disk_path("restorability", key)
    f.write_bytes(b"\x00\x01garbage")
    with caplog.at_level(logging.INFO):
        got = bc.get_cached_restorability_result(path)
    assert got is None
    assert not f.exists()
    assert any(("Platte" in r.message or "fehlgeschlagen" in r.message) for r in caplog.records), (
        "Korrupter Eintrag muss sichtbar gemeldet werden (Cache kostet keine Qualität — INFO)"
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX cache permission contract")
def test_world_writable_cache_entry_is_rejected(bc, tmp_path, caplog):
    path = _audio_file(tmp_path)
    bc.cache_medium_result(path, {"label": "x", "scores": np.zeros(2)})
    bc._medium_lru.clear()
    target = bc._disk_path("medium", bc.content_cache_key(path))
    target.chmod(0o666)

    with caplog.at_level(logging.WARNING):
        got = bc.get_cached_medium_result(path)

    assert got is None
    assert any("nicht vertrauenswürdiger Analyse-Zwischenspeicher-Eintrag" in r.message for r in caplog.records)


def test_disabled_switch_no_disk_io(bc, tmp_path):
    """Kill-Switch: kein Platten-I/O, Memory-LRU unverändert nutzbar."""
    bc._DISK_CACHE_DISABLED = True
    path = _audio_file(tmp_path)
    bc.cache_medium_result(path, {"label": "x", "scores": np.zeros(2)})
    assert not (tmp_path / "analysis_cache").exists()
    bc._medium_lru.clear()
    bc._DISK_CACHE_DISABLED = False
    assert bc.get_cached_medium_result(path) is None  # nichts auf Platte


def test_clear_removes_disk_entries(bc, tmp_path):
    path = _audio_file(tmp_path)
    bc.cache_defect_result(path, {"label": "x", "scores": np.zeros(2)})
    assert any((tmp_path / "analysis_cache" / "defect").glob("*.json"))
    bc.clear_defect_cache(path)
    assert not any((tmp_path / "analysis_cache" / "defect").glob("*.json"))

    bc.cache_medium_result(path, {"label": "x", "scores": np.zeros(2)})
    bc.clear_medium_cache()
    assert not (tmp_path / "analysis_cache" / "medium").exists()

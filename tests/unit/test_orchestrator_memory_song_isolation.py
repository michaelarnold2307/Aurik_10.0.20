#!/usr/bin/env python3
"""§V8 (copilot-instructions.md)-Folge-Hygiene (D-K3-100): Song-Isolation des Sitzung-MEMORY (SessionLearner).

Befund (Register D-K3-95, Folge-Hygiene): ``~/.aurik/session_memory.json``
war ein Cross-Song-Lernparameter-Cache — ``recall`` matchte Einträge über
Material/Codec/Restorability, nicht über die Song-Identität. §V8 (copilot-instructions.md)
(copilot-instructions.md) verlangt: alle Stateful-Module (Circuit-Breaker,
Caches, Lernparameter) pro Song zurücksetzen. Fix 2026-11-12: ``recall``/
``record`` sind ``song_id``-gekeyt (Content-Fingerprint,
``compute_audio_fingerprint``); Altbestand ohne ``song_id`` matcht nie
(fail-closed).

Diese Tests pinnen den Isolations-Vertrag (modell-frei, tmp-Memory-Pfad):

1. recall desselben Songs → Eintrag kommt zurück;
2. recall eines FREMDEN Songs → None (kein Cross-Song-Recall);
3. Altbestand ohne song_id → matcht nie (auch bei identischem Material);
4. recall ohne song_id (None/leer) → None + Warnung (fail-closed);
5. record persistiert song_id im Eintrag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.aurik_orchestrator import SessionLearner


@pytest.fixture()
def learner(tmp_path: Path) -> SessionLearner:
    _path = tmp_path / "session_memory.json"
    SessionLearner._MEMORY_PATH = _path
    _learner = SessionLearner()
    yield _learner
    SessionLearner._MEMORY_PATH = Path.home() / ".aurik" / "session_memory.json"


class TestSessionMemorySongIsolation:
    def test_recall_nur_derselbe_song(self, learner: SessionLearner) -> None:
        learner.record(
            material_type="vinyl",
            restorability=60.0,
            terminal_codec="wav",
            verdict="improved",
            phases_run=9,
            time_s=120.0,
            warnings=[],
            song_id="songA",
        )
        learner.persist_session()
        _hit = learner.recall("vinyl", 60.0, "wav", song_id="songA")
        assert _hit is not None
        assert _hit["song_id"] == "songA"
        _miss = learner.recall("vinyl", 60.0, "wav", song_id="songB")
        assert _miss is None

    def test_altbestand_ohne_song_id_matcht_nie(self, learner: SessionLearner) -> None:
        _path = learner._MEMORY_PATH
        _path.write_text(
            json.dumps(
                {
                    "songs": [
                        {
                            "material": "vinyl",
                            "restorability": 60.0,
                            "terminal_codec": "wav",
                            "verdict": "improved",
                            "phases_run": 9,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        _learner2 = SessionLearner()
        assert _learner2.recall("vinyl", 60.0, "wav", song_id="songA") is None

    def test_recall_ohne_song_id_fail_closed(self, learner: SessionLearner) -> None:
        assert learner.recall("vinyl", 60.0, "wav", song_id=None) is None
        assert learner.recall("vinyl", 60.0, "wav", song_id="") is None

    def test_record_schreibt_song_id(self, learner: SessionLearner) -> None:
        learner.record(
            material_type="kassette",
            restorability=40.0,
            terminal_codec="mp3",
            verdict="improved",
            phases_run=5,
            time_s=60.0,
            warnings=[],
            song_id="songC",
        )
        learner.persist_session()
        _raw = json.loads(learner._MEMORY_PATH.read_text(encoding="utf-8"))
        assert _raw["songs"][0]["song_id"] == "songC"


class TestSSC1ContentKeyed:
    def test_kein_cross_song_zugriff(self, tmp_path: Path) -> None:
        from backend.core.song_strategy_cache import PhaseStrategyEntry, SongStrategyCache

        _cache = SongStrategyCache(cache_file=tmp_path / "ssc.json")
        _cache.store(
            PhaseStrategyEntry(
                song_id="songA",
                mode="restoration",
                last_used=0.0,
                use_count=1,
                phase_strength_overrides={},
                hpi_achieved=0.5,
                vqi_achieved=0.5,
                oqs_achieved=0.5,
            )
        )
        assert _cache.get("songA", "restoration") is not None
        assert _cache.get("songB", "restoration") is None  # fremder Song nie
        assert _cache.get("songA", "studio_2026") is None  # anderer Modus nie

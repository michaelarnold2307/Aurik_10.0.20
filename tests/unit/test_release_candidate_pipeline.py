"""Pfeiler 3 — Release-Kandidaten-Pipeline: Struktur-Vertragstests (leicht).

Sichert die Kandidaten-Tabelle (Roadmap Pfeiler 3), die Flag-Auflösung in
`backend.core.music_model_flags` und die Manifest-Bildung ohne schwere
Pipeline-Läufe.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "release_candidate_pipeline", _ROOT / "scripts" / "release_candidate_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kandidaten_tabelle_vollstaendig_eindeutig() -> None:
    mod = _load_pipeline()
    keys = [c["key"] for c in mod.CANDIDATES]
    flags = [c["flag"] for c in mod.CANDIDATES]
    assert len(mod.CANDIDATES) == 9
    assert len(set(keys)) == len(keys)
    assert len(set(flags)) == len(flags)
    for cand in mod.CANDIDATES:
        assert cand["defizit"], f"Defizit-ID fehlt bei {cand['key']}"
        assert cand["stufe"], f"Stufe fehlt bei {cand['key']}"


def test_kandidaten_flags_existieren_und_sind_default_aus() -> None:
    import backend.core.music_model_flags as mmf

    mod = _load_pipeline()
    for cand in mod.CANDIDATES:
        assert hasattr(mmf, cand["flag"]), f"Flag fehlt: {cand['flag']}"
        assert getattr(mmf, cand["flag"]) is False, f"Flag nicht Default AUS: {cand['flag']}"


def test_manifest_builder_liefert_gueltiges_trial_schema() -> None:
    mod = _load_pipeline()
    trial = {
        "trial_id": "x1_001",
        "class": "vinyl",
        "catch": False,
        "reference": "output/release_candidates/_t/a.wav",
        "candidate": "output/release_candidates/_t/b.wav",
        "start": 0.0,
        "seconds": 6.0,
        "expected_in_candidate": False,
    }
    manifest = mod._manifest([trial], seed=42)
    data = json.loads(json.dumps(manifest))
    assert data["task"] == "preference"
    assert data["seed"] == 42
    assert len(data["trials"]) == 1
    got = data["trials"][0]
    for field in ("trial_id", "class", "catch", "reference", "candidate"):
        assert field in got, f"Manifest-Feld fehlt: {field}"


def test_korpusliste_fest_und_kategorisiert() -> None:
    mod = _load_pipeline()
    assert len(mod.CORPUS) >= 4
    cats = {c.get("material") for c in mod.CORPUS}
    assert len(cats) >= 3, f"Korpus nicht materialbreit: {cats}"


# ── Pfeiler 5 — Fallback-Ledger (B1 des Wohlklang-Vertrags) ─────────────────


def test_fallback_ledger_zaehler_und_reset() -> None:
    from backend.core import fallback_ledger as fl

    fl.reset()
    fl.record_fallback("x", "grund")
    fl.record_fallback("x", "grund2")
    fl.record_fallback("y", "g")
    snap = fl.snapshot()
    assert snap["total"] == 3
    assert snap["by_component"] == {"x": 2, "y": 1}
    fl.reset()
    assert fl.snapshot()["total"] == 0


def test_fallback_ledger_log_counter_findet_v6_meldungen() -> None:
    import logging

    from backend.core import fallback_ledger as fl

    fl.reset()
    counter = fl.install_log_counter("tests.fallback_probe")
    try:
        log = logging.getLogger("tests.fallback_probe")
        log.warning("DeepFilterNet Fallback aktiv (§V6 (copilot-instructions.md))")
        log.warning("irgendwas harmloses")
    finally:
        logging.getLogger("tests.fallback_probe").removeHandler(counter)
    assert fl.snapshot()["total"] == 1
    fl.reset()


def test_fallback_ledger_release_gate() -> None:
    from backend.core import fallback_ledger as fl

    fl.reset()
    assert fl.fallback_report()["release_gate"] == "erfuellt"
    fl.record_fallback("z", "g")
    assert fl.fallback_report()["release_gate"] == "gesperrt"
    fl.reset()

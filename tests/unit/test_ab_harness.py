"""Tests für den A/B-Hör-Harness (Pfeiler 7: Verblindung + Statistik)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ab_harness import collect, prepare, sign_test_two_sided, wilson_ci


def _probe_items(tmp_path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for i in range(3):
        a = tmp_path / f"a{i}.wav"
        b = tmp_path / f"b{i}.wav"
        a.write_bytes(b"WAVEA" + bytes([i]))
        b.write_bytes(b"WAVEB" + bytes([i]))
        items.append({"id": f"clip{i}", "a": str(a), "b": str(b)})
    return items


def test_prepare_determinismus(tmp_path: Path) -> None:
    items = _probe_items(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    prepare(items, out1, seed=42, repeats=2)
    prepare(items, out2, seed=42, repeats=2)
    t1 = json.loads((out1 / "trials.json").read_text(encoding="utf-8"))
    t2 = json.loads((out2 / "trials.json").read_text(encoding="utf-8"))
    k1 = json.loads((out1 / "key.json").read_text(encoding="utf-8"))
    k2 = json.loads((out2 / "key.json").read_text(encoding="utf-8"))
    assert t1["trials"] == t2["trials"], "gleicher Seed ⇒ identisches Versuchsblatt (§G5 (copilot-instructions.md))"
    assert k1["mapping"] == k2["mapping"], "gleicher Seed ⇒ identischer Schlüssel (§G5 (copilot-instructions.md))"


def test_prepare_verblindung(tmp_path: Path) -> None:
    items = _probe_items(tmp_path)
    out = tmp_path / "o"
    prepare(items, out, seed=7, repeats=2)
    sheet = json.loads((out / "trials.json").read_text(encoding="utf-8"))
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    assert len(sheet["trials"]) == 6
    for entry in sheet["trials"]:
        raw = json.dumps(entry)
        assert '"A"' not in raw and '"B"' not in raw, "Versuchsblatt zeigt keine A/B-Zuordnung"
        for slot in ("left", "right"):
            rel = entry["stimuli"][slot]
            assert rel.startswith("stimuli/"), "Stimuli liegen unter neutralen Namen"
            assert (out / rel).is_file(), "Stimulus-Datei existiert (Byte-Kopie)"
        assert entry["trial_id"] in key["mapping"], "jeder Trial hat einen Schlüssel-Eintrag"
        assert set(key["mapping"][entry["trial_id"]]) >= {"left", "right", "quelle"}
    src = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert src["sha256"], "Provenienz: Quell-Hashes dokumentiert (§III.13)"


def test_collect_statistik_ties_ausgeschlossen(tmp_path: Path) -> None:
    mapping = {f"t{i:02d}": {"left": "A", "right": "B", "quelle": "Schlüssel"} for i in range(23)}
    key = {"schema": "aurik.ab_key/1", "seed": 1, "mapping": mapping, "quelle": "test"}
    verdikte = [{"trial_id": f"t{i:02d}", "preferred": "left"} for i in range(15)]
    verdikte += [{"trial_id": f"t{i:02d}", "preferred": "right"} for i in range(15, 20)]
    verdikte += [{"trial_id": f"t{i:02d}", "preferred": "tie"} for i in range(20, 23)]
    rep = collect(verdikte, key)
    assert rep["n_entscheidend"] == 20, "Ties zählen nicht mit (dokumentiert)"
    assert rep["gleichstaende"] == 3
    assert rep["siege_a"] == 15 and rep["siege_b"] == 5
    assert abs(rep["p_wert_vorzeichen_test"] - 0.041389) < 1e-6, "exakter Vorzeichen-Test"
    assert rep["signifikant"] is True
    assert "vorzuziehen" in rep["empfehlung"]
    assert "H\u00f6r-Instanz" in rep["hinweis"], "Hörordnung: Statistik ist Zeuge, kein Urteil"
    assert rep["quelle"] == "scripts/ab_harness.py collect", "Beleg-Prinzip: Quelle im Report"


def test_statistik_hilfsfunktionen() -> None:
    assert abs(sign_test_two_sided(15, 20) - 0.04138946533203125) < 1e-12
    assert sign_test_two_sided(5, 10) == 1.0, "ausgewogenes Bild ist nicht signifikant"
    assert sign_test_two_sided(10, 10) < 0.01, "10/10 ist hoch signifikant"
    lo, hi = wilson_ci(15, 20)
    assert 0.0 < lo < 0.75 < hi < 1.0, f"Wilson-CI plausibel: {lo}, {hi}"

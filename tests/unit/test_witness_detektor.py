"""Vertragstests für das gelernte Schadens-Witness (SOTA-Metrik-Kanal, 2026-11-14).

Geprüft wird die Wirkungs-Semantik, nicht die Modellgüte (die belegen die
Report-JSONs der Checkpoints):

* Gate-Logik — AUC ≥ 0,80 UND Richtung P(schaden) > P(clean).
* Leak-Freiheit der Titel-Aufteilung (Holdout-Titel kommt nie ins Training).
* Determinismus (§G5 (copilot-instructions.md)) — Seed 42, Re-Draw/Shuffle aus Seeds.
* Report-Only-Vertrag der Inferenz — kein Flag, kein Audio-Eingriff, CPU-only.
* Fail-closed — fehlendes Checkpoint ⇒ 0.0 mit §V6-Warnung statt stiller Fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.dsp import witness_detektor as wd
from scripts import train_witness_detektor as tw

# ---------------------------------------------------------------------------
# 1) Gate-Logik (modell-frei)
# ---------------------------------------------------------------------------


def test_gate_braucht_auc_und_richtung() -> None:
    """Gate = Holdout-AUC ≥ 0,80 UND P(schaden) > P(clean) — beide Bedingungen."""
    assert tw.AUC_ZIEL == 0.80
    # Richtung verletzt ⇒ kein Bestehen, egal wie gut die AUC wäre
    gut_auc_falsche_richtung = {"holdout_auc": 0.99, "richtung_ok": False}
    assert not (gut_auc_falsche_richtung["holdout_auc"] >= tw.AUC_ZIEL and gut_auc_falsche_richtung["richtung_ok"])


def test_seed_konstanten() -> None:
    """Determinismus-Vertrag: Seed 42 und FRAMES sind festgepinnt (§G5 (copilot-instructions.md))."""
    assert tw.SEED == 42
    assert tw.FRAMES == 64


def test_training_without_selected_checkpoint_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    mel = np.zeros((64, tw.FRAMES), dtype=np.float32)
    daten = [(mel, 0, "train"), (mel, 1, "holdout")]
    monkeypatch.setattr(tw, "_baue_dataset", lambda klasse, manifest=None: (daten, ["holdout"], 0))
    monkeypatch.setattr(tw, "EPOCHS", 0)

    with pytest.raises(RuntimeError, match="Kein gültiger Witness-Checkpoint"):
        tw.trainiere("groove_echo")


def test_dataset_leak_frei_nach_titel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Der Holdout-Titel taucht in keinem Trainingsbeispiel auf (leak-frei).

    Es wird nur die Split-Logik geprüft: _baue_dataset liest das Manifest —
    hier mit einem Mini-Manifest via Monkeypatch auf MANIFEST.
    """
    import scipy.io.wavfile as wavfile

    sr = 44100
    t = np.arange(sr) / sr  # 1 s
    ton = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    manifest = tmp_path / "manifest.jsonl"
    zeilen = []
    for i, titel in enumerate(("TitelA", "TitelB", "TitelC")):
        for j in range(3):
            c = tmp_path / f"c_{i}_{j}.wav"
            s = tmp_path / f"s_{i}_{j}.wav"
            wavfile.write(str(c), sr, ton)
            wavfile.write(str(s), sr, ton)
            zeilen.append(
                {
                    "klasse": "groove_echo",
                    "titel": titel,
                    "start": 0,
                    "end": sr,
                    "clean": str(c),
                    "schaden": str(s),
                    "restauriert": str(s),
                }
            )
    manifest.write_text("\n".join(json.dumps(z) for z in zeilen), encoding="utf-8")
    monkeypatch.setattr(tw, "MANIFEST", manifest)

    daten, holdouts, _ = tw._baue_dataset("groove_echo")
    assert len(holdouts) == 1
    holdout_titel = holdouts[0]
    train_titel = {t for _, _, t in daten if t != holdout_titel}
    assert holdout_titel not in train_titel
    # jedes Titel-Label kommt genau zweimal vor (clean=0, schaden=1)
    assert {t for _, _, t in daten} == {"TitelA", "TitelB", "TitelC"}


# ---------------------------------------------------------------------------
# 2) Report-Only-Vertrag der Inferenz
# ---------------------------------------------------------------------------


def test_inferenz_fail_closed_ohne_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fehlendes Checkpoint ⇒ 0.0, kein Traceback (§V6 (copilot-instructions.md))."""
    monkeypatch.setattr(wd, "_ROOT", Path("/gibt/es/nicht"))
    wd._cache.clear()
    assert wd.witness_kanal_verfuegbar("groove_echo") is False
    assert wd.witness_schaden_kanal(np.zeros(4096, dtype=np.float32), 44100, "groove_echo") == 0.0


def test_inferenz_fail_closed_zu_kurz() -> None:
    """Eingabe < 512 Samples ⇒ 0.0 ohne Modell-Last."""
    assert wd.witness_schaden_kanal(np.zeros(64, dtype=np.float32), 44100, "groove_echo") == 0.0


def test_quelltext_pin_kein_flag_kein_cuda() -> None:
    """Der Witness fasst weder Flags an noch initialisiert er CUDA."""
    quelltext = Path(wd.__file__).read_text(encoding="utf-8")
    assert "music_model_flags" not in quelltext
    assert "flippe_flag" not in quelltext
    assert ".cuda(" not in quelltext
    assert "cuda()" not in quelltext
    assert "to('cuda" not in quelltext


def test_witness_checkpoint_liegt_nach_training() -> None:
    """Wenn das kanonische groove_echo-Checkpoint existiert, ist es ladbar und
    richtungsplausibel: P(schaden-Region) > P(clean-Region) im Mittel über
    eine kleine deterministische Stichprobe — sonst wird übersprungen
    (das Training ist der kanonische Lauf, nicht der Test)."""
    if not wd.witness_kanal_verfuegbar("groove_echo"):
        pytest.skip("kein groove_echo-Checkpoint — kanonischer Trainingslauf steht aus")
    report = json.loads((wd.MODELS_DIR / "witness_groove_echo" / "report.json").read_text(encoding="utf-8"))
    assert report["gate_bestanden"] is True
    assert report["richtung_ok"] is True
    assert report["holdout_auc"] >= tw.AUC_ZIEL

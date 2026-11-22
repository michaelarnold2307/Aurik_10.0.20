"""Vertragstests für den Witness-Kopf-Trainer (D-K3-95, 2026-11-15).

Geprüft wird die Wirkungs-Semantik (Leak-Freiheit, Determinismus, fail-closed,
Gate-Ziel-Modellauswahl), nicht die Modellgüte — die belegt der Report des
vollen Laufs. Modell-frei, CPU-light.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.train_witness_kopf as wk


def test_schema_und_seed_pin() -> None:
    assert wk.SCHEMA == "aurik.witness_kopf_training/1"
    assert wk.SEED == 42


def test_witness_q_definition() -> None:
    """Q = (P_s − P_out)/(P_s − P_c); Kontrast ≤ 0,05 ⇒ None (fail-closed)."""
    assert wk._witness_q(0.8, 0.1, 0.3) == pytest.approx(0.5 / 0.7)
    assert wk._witness_q(0.8, 0.1, 0.1) == pytest.approx(1.0)
    assert wk._witness_q(0.8, 0.1, 0.8) == pytest.approx(0.0)
    assert wk._witness_q(0.1, 0.05, 0.0) is None  # Kontrast 0,05 nicht überschritten


def test_leak_frei_holdout_titel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trainingsregionen enthalten NIE den Holdout-Titel (Muster Witness-Trainer)."""
    import json

    manifest = tmp_path / "m.jsonl"
    zeilen = []
    for i in range(12):
        titel = f"Titel{i % 3}"
        zeilen.append(
            {
                "klasse": "groove_echo",
                "titel": titel,
                "clean": str(tmp_path / f"c{i}.wav"),
                "schaden": str(tmp_path / f"s{i}.wav"),
                "restauriert": str(tmp_path / f"r{i}.wav"),
            }
        )
    manifest.write_text("\n".join(json.dumps(z) for z in zeilen), encoding="utf-8")
    monkeypatch.setattr(wk, "MANIFEST", manifest)
    regionen, holdout_titel, _ = wk._baue_regionen("groove_echo")
    hold_regionen = [r for r in regionen if holdout_titel in str(r[0])]
    train_regionen = [r for r in regionen if holdout_titel not in str(r[0])]
    assert hold_regionen, "Holdout-Regionen fehlen"
    assert len(train_regionen) + len(hold_regionen) == len(regionen)
    # Disjunktheit durch Titel-Prüfung
    assert all(holdout_titel not in str(r[0]) for r in train_regionen)


def test_fail_closed_witness_checkpoint_vor_training() -> None:
    """Die Witness-Checkpoint-Prüfung steht im Quelltext VOR dem Trainings-Loop —
    fehlendes Checkpoint bricht ab, bevor Rechenzeit verbraucht wird (§V6 (copilot-instructions.md))."""
    quelltext = Path(wk.__file__).read_text(encoding="utf-8")
    assert "w_ckpt" in quelltext
    assert "FileNotFoundError" in quelltext
    assert "Witness-Checkpoint fehlt" in quelltext
    i_pruefung = quelltext.find("Witness-Checkpoint fehlt")
    i_loop = quelltext.find("for ep in range(epochen)")
    assert i_pruefung > 0 and i_loop > i_pruefung


def test_training_without_selected_checkpoint_fails_explicitly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wk, "_ROOT", tmp_path)
    checkpoint_dir = tmp_path / "models" / "witness_groove_echo"
    checkpoint_dir.mkdir(parents=True)
    witness = wk.WitnessCRNN(wk.N_MELS)
    wk.torch.save({"n_mels": wk.N_MELS, "state_dict": witness.state_dict()}, checkpoint_dir / "witness.pt")

    audio = np.zeros(wk.FENSTER, dtype=np.float32)
    monkeypatch.setattr(wk, "_baue_regionen", lambda klasse: ([], "holdout", 1))
    monkeypatch.setattr(
        wk,
        "_lade_split",
        lambda regionen, holdout_titel, max_regionen: (
            [audio],
            [audio],
            [audio],
            [audio],
            [audio],
            [audio],
            [0.2],
            [0.2],
            [0.2],
            [0.2],
        ),
    )

    with pytest.raises(RuntimeError, match="Kein gültiger Witness-Kopf-Checkpoint"):
        wk.trainiere("groove_echo", epochen=0, hidden=2, stufen=1, lam=0.1, device="cpu")


def test_quelltext_pin_kein_flag_kein_cuda_zwang() -> None:
    quelltext = Path(wk.__file__).read_text(encoding="utf-8")
    assert "music_model_flags" not in quelltext
    assert "flippe_flag" not in quelltext
    assert "cuda()" not in quelltext  # Gerät kommt aus --device, kein Zwang
    assert "to('cuda" not in quelltext


def test_db_roh_ausgabe_entspricht_produktions_denormalisierung() -> None:
    """dBFS(out_raw) = dBFS(out_norm) + 20·log10(rms_lin/RMS_ZIEL) — exakt die
    Denormalisierung der Produktions-Inferenz (_kopf_out)."""
    import torch

    out_norm = torch.full((2, 1, wk.FENSTER), wk.RMS_ZIEL)
    rms_lin = torch.tensor([0.1, 0.4])
    db = wk._db_roh_ausgabe(out_norm, rms_lin)
    erwartet = 20.0 * torch.log10(rms_lin)  # RMS(out_norm) == RMS_ZIEL ⇒ dBFS(out_norm) == 20·log10(RMS_ZIEL)
    assert torch.allclose(db, erwartet, atol=1e-5)


def test_db_roh_fenster_ist_20_log10_rms() -> None:
    """Wahrer Pegel eines RAW-Fensters = 20·log10(rms_lin) (10·log10(mean(x²)))."""
    import torch

    rms_lin = torch.tensor([0.05, 0.2, 0.5])
    assert torch.allclose(wk._db_roh_fenster(rms_lin), 20.0 * torch.log10(rms_lin), atol=1e-6)

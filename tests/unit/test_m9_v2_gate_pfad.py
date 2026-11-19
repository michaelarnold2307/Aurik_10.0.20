"""Vertragstests für den M9-V2-Gate-Pfad (Export + --modell-Flag, 2026-11-14).

Modell-frei: fail-closed-Pfade, Quelltext-Pins (§V7/§G5 (copilot-instructions.md)), kein Flag-Zugriff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import verify_m9_gate as gate
from scripts.export_tf_gridnet_onnx import exportiere


def test_export_fail_closed_ohne_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        exportiere(tmp_path / "gibt_es_nicht.pt", tmp_path / "out.onnx")


def test_export_skript_fasst_flags_nicht_an() -> None:
    quelltext = Path("scripts/export_tf_gridnet_onnx.py").read_text(encoding="utf-8")
    assert "music_model_flags" not in quelltext
    assert "flippe_flag" not in quelltext


def test_gate_fail_closed_mit_fehlerhaftem_modell_pfad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--modell mit fehlendem ONNX ⇒ Exit 2 (kein stiller Erfolg)."""
    monkeypatch.setattr(gate, "MUSDB_TEST", tmp_path / "keine_daten")
    rc = gate.main(["--modell", str(tmp_path / "fehlt.onnx")])
    assert rc == 2


def test_gate_skript_nennt_modell_feld() -> None:
    quelltext = Path("scripts/verify_m9_gate.py").read_text(encoding="utf-8")
    assert '"modell_sha256"' in quelltext
    assert "--modell" in quelltext

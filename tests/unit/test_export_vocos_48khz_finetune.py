"""Vertragstests vocos-48k-Finetune-ONNX-Export (M5, 2026-11-09).

Modell-frei: Pfad-Auflösung, state_dict-Ladung und Argparse-Vertrag werden ohne
147-MB-Modell-Ladung gepinnt. Der eigentliche Export (torch.onnx.export) ist
der Kette-4-Schritt im freien CPU/GPU-Fenster — ein Modell-ladender Smoke ist
daher bewusst skip-markiert (§G5 (GEBOTE.md): Tests dürfen nicht
umgebungsabhängig kippen).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FINETUNE_DIR = PROJECT_ROOT / "models" / "vocos_48khz" / "finetune"


def test_resolve_paths_finetune_vs_base() -> None:
    """M5: --finetune löst Checkpoint + ONNX-Ausgabe ins finetune/-Verzeichnis auf."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.export_vocos_48khz_onnx import FINETUNE_OUT, FINETUNE_WPATH, OUT_PATH, WPATH, _resolve_paths

    w_base, out_base = _resolve_paths(False)
    w_ft, out_ft = _resolve_paths(True)
    assert (w_base, out_base) == (WPATH, OUT_PATH)
    assert w_ft == FINETUNE_WPATH and w_ft.endswith("vocos_48khz_music_ft.pt")
    assert out_ft == FINETUNE_OUT and out_ft.endswith("vocos_48khz_music_ft.onnx")
    assert w_ft != w_base and out_ft != out_base, "Finetune darf die Base-Pfade nie überschreiben"


def test_load_state_dict_finetune_schluessel() -> None:
    """M5: Das Finetune-Checkpoint (Misch-Dict) wird korrekt unter 'state_dict' gelesen."""
    import tempfile

    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.export_vocos_48khz_onnx import _load_state_dict

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ckpt.pt"
        torch.save({"state_dict": {"w": torch.zeros(2)}, "epochs": 30, "seed": 42}, p)
        sd = _load_state_dict(str(p), state_key="state_dict")
        assert set(sd) == {"w"}

        # Fail-closed: fehlender state_dict-Schlüssel wirft KeyError (§V6 (copilot-instructions.md))
        try:
            _load_state_dict(str(p), state_key="gibt_es_nicht")
            raise AssertionError("KeyError erwartet")
        except KeyError:
            pass


def test_argparse_finetune_flag() -> None:
    """M5: Argparse-Vertrag — --finetune existiert, kein Flag-Flip, kein Deploy."""
    r = subprocess.run(
        [sys.executable, "-B", "scripts/export_vocos_48khz_onnx.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "--finetune" in r.stdout


def test_finetune_checkpoint_und_manifest_existieren() -> None:
    """M5: Das trainierte Checkpoint + Trainings-Manifest liegen vor (status ok)."""
    import json

    assert (FINETUNE_DIR / "vocos_48khz_music_ft.pt").is_file(), "Finetune-Checkpoint fehlt"
    m = json.loads((FINETUNE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert m["schema"] == "aurik.vocos48k_finetune/1"
    assert m["status"] == "ok"


def test_export_smoke_skip_markiert() -> None:
    """M5: Modell-ladender Export-Smoke — bewusst skip (GPU/CPU-Fenster, §5 Handover)."""
    import pytest

    pytest.skip("ONNX-Export ist der Kette-4-Schritt im freien Fenster — kein Dauerlauf im Unit-Test")

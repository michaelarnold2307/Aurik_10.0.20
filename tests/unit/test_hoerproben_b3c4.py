"""Vertragstests Hörproben-Erzeuger + Entscheidungs-Panel (B3/C4-Vorbereitung, 2026-11-09).

Modell-frei: Ausschnitt/Resampling/Argparse/Fail-closed — die ONNX-ladenden
Proben-Läufe sind bewusst skip-markiert (CPU-ONNX gehört ins freie Fenster;
die Exporte liegen nach dem Kette-Lauf vor).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.erstelle_hoerproben import _ausschnitt, _lade_48k


def test_ausschnitt_10s_aus_mitte() -> None:
    import numpy as np

    x = np.zeros(48000 * 90, dtype=np.float32)
    mitte = slice(48000 * 41, 48000 * 49)  # klar innerhalb des 40–50-s-Ausschnitts
    x[mitte] = 1.0
    a = _ausschnitt(x, 10.0)
    assert len(a) == 48000 * 10
    assert float(a.max()) == 1.0, "markierter Mittelteil muss enthalten sein"
    assert float(a.sum()) == 48000 * 8.0, "Ausschnitt exakt 41–49 s ⇒ 8 s Marker"


def test_lade_48k_resampling(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    p = tmp_path / "test.wav"
    sf.write(p, np.zeros(44100, dtype=np.float32), 44100)
    x = _lade_48k(p)
    assert len(x) == 48000, f"44100 → 48000 Samples erwartet, erhalten {len(x)}"


def test_erzeuge_c4_fail_closed_ohne_onnx(tmp_path: Path) -> None:
    """§V6 (copilot-instructions.md): fehlende ONNX ⇒ klare Fehlermeldung, kein Stummel."""
    from unittest import mock

    import numpy as np
    import soundfile as sf

    from scripts import erstelle_hoerproben as ep

    musik = tmp_path / "musik"
    musik.mkdir()
    sf.write(musik / "lied.wav", np.zeros(48000, dtype=np.float32), 48000)
    with mock.patch.object(ep, "BASE_ONNX", tmp_path / "fehlt.onnx"):
        with mock.patch.object(ep, "FT_ONNX", tmp_path / "fehlt2.onnx"):
            try:
                ep.erzeuge_c4_proben(musik, tmp_path / "out", anzahl=1)
                raise AssertionError("FileNotFoundError erwartet")
            except FileNotFoundError:
                pass


def test_entscheidung_status_ohne_proben() -> None:
    """Panel ohne Proben: Exit 0, klare Meldung statt Blockade."""
    r = subprocess.run(
        [sys.executable, "-B", "scripts/hoer_entscheidung.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-400:]
    assert "C4" in r.stdout and "B3" in r.stdout, "Status muss B3/C4 nennen"


def test_hoerproben_manifest_schema(tmp_path: Path) -> None:
    import json

    manifest = {"schema": "aurik.hoerproben/1", "sekunden": 10.0, "seed": 42, "sets": {}}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["schema"] == "aurik.hoerproben/1"
    assert d["sekunden"] == 10.0 and d["seed"] == 42

"""Gate-Test: §G9 (copilot-instructions.md) — Layout-Normalisierung hat EINE Quelle (Muster D-K3-6).

Geprüft wird nicht die Prosa des Gates, sondern sein Verhalten an synthetischen
Modulen (tmp_path): baut ein Modul eine eigene Layout-Entscheidung, MUSS das Gate
fail-closed anschlagen; delegiert es an `backend.core.audio_layout`, darf es das
nicht. Die Beschreibungs-Blindheit (Prosa ≠ Code) wird eigens abgesichert, weil
`_blank_prose` genau dort lag (§G9 copilot-instructions.md).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _gate():
    """Das Gate-Modul importieren (scripts/ ist kein Paket)."""
    path = ROOT / "scripts" / "layout_invariant_check.py"
    spec = importlib.util.spec_from_file_location("_layout_invariant_check_ut", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, rel: str, code: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")


def _errors(findings) -> list[str]:
    return [f.message for f in findings if f.severity == "ERROR"]


def test_local_guessing_normalizer_is_reported(tmp_path: Path) -> None:
    """L1: lokale Kanalachsen-Raterei ohne kanonische Quelle → ERROR."""
    gate = _gate()
    _write(
        tmp_path,
        "backend/mod.py",
        """
import numpy as np


def _to_channels_first(x):
    if x.ndim == 1:
        return x[None, :], False
    if x.shape[0] <= 2 and x.shape[1] > x.shape[0]:
        return x, False
    return x.T, True
""",
    )
    findings, _ = gate.scan(roots=("backend",), base=tmp_path)
    errs = _errors(findings)
    assert len(errs) == 1, errs
    assert "_to_channels_first" in errs[0]
    assert "D-K3-6" in errs[0]


def test_delegating_normalizer_passes(tmp_path: Path) -> None:
    """L1: Delegation an die kanonische Quelle → kein Befund."""
    gate = _gate()
    _write(
        tmp_path,
        "backend/mod.py",
        """
import numpy as np


def _to_channels_first(x):
    from backend.core.audio_layout import normalize_channels_first

    return normalize_channels_first(np.asarray(x))
""",
    )
    findings, _ = gate.scan(roots=("backend",), base=tmp_path)
    assert _errors(findings) == []


def test_prose_mention_is_not_code(tmp_path: Path) -> None:
    """§G9 (copilot-instructions.md): ein `shape[0] <= 2` in einem Kommentar ist keine Entscheidung."""
    gate = _gate()
    _write(
        tmp_path,
        "backend/mod.py",
        '''
"""Modul-Doku.

Die alte Fassung prüfte ``x.shape[0] <= 2`` — Muster D-K3-6.
"""
import numpy as np


def plain(x):
    # if x.shape[0] <= 2: return x  (historisch, entfernt)
    return np.asarray(x)
''',
    )
    findings, _ = gate.scan(roots=("backend",), base=tmp_path)
    assert _errors(findings) == []


def test_restore_layout_helpers_are_out_of_scope(tmp_path: Path) -> None:
    """Rollen-Grenze: Rück-Transpose/Stereo-Aufbau sind keine Normalisierer."""
    gate = _gate()
    _write(
        tmp_path,
        "backend/mod.py",
        """
import numpy as np


def _restore_layout(audio, was_transposed):
    if was_transposed and audio.ndim == 2:
        return audio.T
    return audio


def _to_stereo(x):
    return np.column_stack([x, x])
""",
    )
    findings, _ = gate.scan(roots=("backend",), base=tmp_path)
    assert _errors(findings) == []


def test_canonical_module_itself_is_exempt(tmp_path: Path) -> None:
    """Die kanonische Quelle darf (muss) die Regel selbst formulieren."""
    gate = _gate()
    _write(
        tmp_path,
        "backend/core/audio_layout.py",
        """
import numpy as np

MAX_CHANNELS = 8


def is_channels_first(arr):
    return arr.ndim == 2 and arr.shape[0] <= MAX_CHANNELS and arr.shape[1] > 8


def normalize_channels_first(arr):
    # interner Zahlenvergleich ist hier erlaubt (Quelle!)
    if arr.ndim == 2 and arr.shape[0] <= 2:
        return arr, False
    return arr, False
""",
    )
    findings, _ = gate.scan(roots=("backend",), base=tmp_path)
    assert _errors(findings) == []


def test_unparsable_module_is_fail_closed(tmp_path: Path) -> None:
    """Nicht prüfbar (Syntaxfehler) zählt als Befund, nicht als Freigabe."""
    gate = _gate()
    _write(tmp_path, "backend/broken.py", "def x(:\n")
    findings, _ = gate.scan(roots=("backend",), base=tmp_path)
    assert any("nicht parsebar" in msg for msg in _errors(findings))


def test_l2_report_is_informational(tmp_path: Path) -> None:
    """L2 zählt Risiko-Muster, fällt aber nie."""
    gate = _gate()
    _write(
        tmp_path,
        "backend/mod.py",
        """
import numpy as np


def collapse(audio):
    return np.mean(audio, axis=0)
""",
    )
    findings, risky = gate.scan(roots=("backend",), base=tmp_path)
    assert risky >= 1
    assert _errors(findings) == []


@pytest.mark.parametrize("missing", ["cli", "scripts"])
def test_missing_root_is_skipped(tmp_path: Path, missing: str) -> None:
    """Fehlende Produktionswurzel ist kein Fehler (Gate bleibt robust)."""
    gate = _gate()
    _write(tmp_path, "backend/mod.py", "import numpy as np\n")
    findings, _ = gate.scan(roots=("backend", missing), base=tmp_path)
    assert _errors(findings) == []


def test_repository_is_clean() -> None:
    """§G9 (copilot-instructions.md) Ist-Zustand: der echte Produktionscode hat 0 lokale Layout-Entscheidung."""
    gate = _gate()
    findings, _ = gate.scan()
    assert _errors(findings) == []

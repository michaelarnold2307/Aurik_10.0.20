from __future__ import annotations

"""Verdrahtungstest SongContextMap in UnifiedRestorerV3.restore() (Spec 03 §SCM).

Die Karte muss tatsächlich im Ergebnis ankommen — nicht nur im Phasen-Akkumulator
(der bestehende Schlüssel ``song_structure`` erreicht das Ergebnis nie; Spec 07 §8.7).
Zusätzlich: Song-Isolation (§V8 (copilot-instructions.md)) und ehrlicher Scope.
"""

import ast
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from backend.core.song_context_map import DIMENSIONS, SCHEMA
from backend.core.unified_restorer_v3 import UnifiedRestorerV3
from tests.unit.test_end_of_song_cleanup_hoist import _common_patches
from tests.unit.test_precomputed_phase_plan_determinism import _fast_restore_kwargs

_UV3 = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


@pytest.fixture(autouse=True)
def _deterministic_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host-Zustand neutralisieren (tests.instructions.md, Regel 6).

    restore() verweigert den Pipeline-Start bei Swap > 95 % (§MEM-GATE) und die
    ML-Speichergrenze reagiert auf System-Thrashing. Beides hängt vom Live-Zustand
    des Rechners ab (z. B. parallel laufendes Training) und darf das Ergebnis dieses
    Tests nicht bestimmen.
    """
    import psutil

    sswap = namedtuple("sswap", "total used free percent sin sout")  # Feldnamen wie psutil.swap_memory()
    monkeypatch.setattr(psutil, "swap_memory", lambda: sswap(1 << 33, 0, 1 << 33, 0.0, 0, 0))
    monkeypatch.setattr("backend.core.ml_memory_budget.is_system_thrashing", lambda: False)


@pytest.fixture
def short_audio() -> np.ndarray:
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


def _restore_fn() -> ast.FunctionDef:
    tree = ast.parse(_UV3.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "restore":
            return node
    raise AssertionError("UnifiedRestorerV3.restore nicht gefunden")


def test_wiring_contract_reset_then_build_then_result_metadata() -> None:
    fn = _restore_fn()
    reset_lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_song_context_map_meta" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    ]
    build_lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "build_song_context_map"
    ]
    result_lines = [
        k.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "RestorationResult"
        for kw in n.keywords
        if kw.arg == "metadata" and isinstance(kw.value, ast.Dict)
        for k in kw.value.keys
        if isinstance(k, ast.Constant) and k.value == "song_context_map"
    ]
    assert reset_lines and build_lines and result_lines, (
        "Reset, Aufbau und Ergebnis-Metadatum müssen alle vorhanden sein"
    )
    assert min(reset_lines) < min(build_lines) < min(result_lines)


@pytest.mark.unit
def test_whole_song_restore_returns_song_scope_map(short_audio: np.ndarray) -> None:
    uv3 = UnifiedRestorerV3()
    with _common_patches(short_audio):
        result = uv3.restore(short_audio, 48000, **_fast_restore_kwargs())
    scm = result.metadata["song_context_map"]
    assert scm["schema"] == SCHEMA
    assert scm["report_only"] is True
    assert scm["scope"] == "song"
    assert tuple(scm["dimensions"]) == DIMENSIONS
    assert scm["analysis_window_s"] == pytest.approx(1.0, abs=0.01)
    # Jede Dimension ist entweder gemessen oder ehrlich als nicht verfügbar geführt.
    assert all(d["state"] in {"measured", "unavailable"} for d in scm["dimensions"].values())


@pytest.mark.unit
def test_chunk_restore_is_marked_chunk_scope(short_audio: np.ndarray) -> None:
    uv3 = UnifiedRestorerV3()
    with _common_patches(short_audio):
        result = uv3.restore(
            short_audio,
            48000,
            chunk_start_sample=48000,
            _chunked_tail_skip=True,
            _chunked_last=False,
            **_fast_restore_kwargs(),
        )
    assert result.metadata["song_context_map"]["scope"] == "chunk"


@pytest.mark.unit
def test_stale_map_from_previous_song_never_leaks(short_audio: np.ndarray) -> None:
    uv3 = UnifiedRestorerV3()
    uv3._song_context_map_meta = {"stale": "song-1"}
    with _common_patches(short_audio):
        result = uv3.restore(short_audio, 48000, **_fast_restore_kwargs())
    assert "stale" not in (result.metadata["song_context_map"] or {})

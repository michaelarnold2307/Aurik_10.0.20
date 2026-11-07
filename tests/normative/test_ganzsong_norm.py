"""Ganzsong-Norm (§2.75 pipeline.instructions.md, Register D-K4-3).

Pinned wird der normative Zweipass-Vertrag: song-globale Analyse VOR dem
Chunking, chunked DSP mit P0-1-Deferral, Evaluation EINMAL nach der Assembly.
Der Ist-Zustand ist der dokumentierte Zwischenstand — kein stiller Zustand.
"""

from __future__ import annotations

import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PIPELINE_MD = (_REPO / ".github" / "instructions" / "pipeline.instructions.md").read_text(encoding="utf-8")
_UV3 = (_REPO / "backend" / "core" / "unified_restorer_v3.py").read_text(encoding="utf-8")
_REGISTER = (_REPO / ".github" / "SOTA_DEFICIT_REGISTER.md").read_text(encoding="utf-8")


def test_norm_ist_dokumentiert() -> None:
    assert "## §2.75 Ganzsong-Norm" in _PIPELINE_MD
    assert "Pass 1 — Voranalyse (song-global" in _PIPELINE_MD
    assert "Pass 2 — Chunked DSP" in _PIPELINE_MD
    assert "Pass 3 — Evaluation nach Assembly" in _PIPELINE_MD
    assert "SOTA-M12-Rollout (Per-Stem)" in _PIPELINE_MD


def test_zweipass_mechanik_im_code() -> None:
    assert "_restore_chunked" in _UV3
    assert "_chunked_tail_skip" in _UV3 and "_chunked_last" in _UV3
    # P0-1-Slice-Semantik existiert als Waechter der Chunk-Schleife
    assert "chunked_tail_ueberspringen" in _UV3 or "P0-1" in _UV3


def test_emotionalarc_messung_ist_deferred() -> None:
    # ARC-Bewertung darf NIE auf Nicht-Letzten-Chunks laufen (song-global).
    assert "§P0-1 Slice B" in _UV3
    assert "_chunked_tail_skip and not _chunked_last" in _UV3


def test_hoer_blockierter_rest_ist_dokumentiert() -> None:
    # Goosebumps-Wahl/MDEM-Morphing: Rest unter D-K3-65 (b) — §V7 (copilot-instructions.md), kein Blind-Fix.
    assert "D-K3-65" in _REGISTER
    assert "Goosebumps-Wahl/" in _PIPELINE_MD and "Hör-blockiert" in _PIPELINE_MD

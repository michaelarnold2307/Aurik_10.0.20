"""Vertragstests: Vorher/Nachher-Segmentvorschau (Bedienung/Echtzeit, 2026-11-19).

Modell-frei: Segment-Schnitt (Mono + channels-first, Mitte-Standard), die
Bridge-Funktion (Produktions-Pfad ohne Env-Seiteneffekte, kein Cache-Eintrag)
und die GUI-Verdrahtung (Button + Thread + i18n) werden ohne Audio-Inferenz
gepinnt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def test_segment_mitte_und_kurzsignal() -> None:
    from Aurik10.ui.preview_restorer import PREVIEW_SEGMENT_S, segment_fuer_preview

    sr = 48000
    n = sr * 10  # 10 s
    x = np.arange(n, dtype=np.float32)[None, :]  # channels-first (1, N)
    seg = segment_fuer_preview(x, sr)
    assert seg.shape == (1, int(PREVIEW_SEGMENT_S * sr))
    # Mitte-Standard: Offset = (n - len)//2
    assert seg[0, 0] == x[0, (n - seg.shape[-1]) // 2]
    # Kurzsignal bleibt unverändert (kein Pad-Artefakt).
    kurz = np.zeros((2, 1000), dtype=np.float32)
    assert segment_fuer_preview(kurz, sr).shape == (2, 1000)


def test_segment_offset_bedient_mono_und_stereo() -> None:
    from Aurik10.ui.preview_restorer import segment_fuer_preview

    sr = 8000
    mono = np.zeros(sr * 4, dtype=np.float32)
    seg = segment_fuer_preview(mono, sr, sekunden=2.0, offset_s=1.0)
    assert seg.shape == (2 * sr,)
    stereo = np.zeros((2, sr * 4), dtype=np.float32)
    assert segment_fuer_preview(stereo, sr, sekunden=2.0, offset_s=1.0).shape == (2, 2 * sr)


def test_bridge_funktion_produktionspfad_ohne_cache() -> None:
    """restore_segment_preview ruft den Denker-Pfad und verschmutzt den Cache nicht."""
    quelle = (_ROOT / "backend" / "api" / "bridge.py").read_text(encoding="utf-8")
    assert "def restore_segment_preview" in quelle
    assert "run_pre_analysis(" in quelle
    assert "store_in_bridge_cache=False" in quelle
    assert "normalize_user_mode(mode)" in quelle


def test_gui_verdrahtung_button_und_thread() -> None:
    window = (_ROOT / "Aurik10" / "ui" / "modern_window.py").read_text(encoding="utf-8")
    assert "self._panel_btn_preview" in window
    assert "def _start_restore_preview" in window
    assert "def _on_preview_done" in window
    assert "PreviewRestorerThread(" in window
    assert "segment_fuer_preview" in window


def test_thread_fehler_signal_statt_stillem_fallback() -> None:
    """§V6: Fehler werden als failed-Signal gemeldet, nicht verschluckt."""
    quelle = (_ROOT / "Aurik10" / "ui" / "preview_restorer.py").read_text(encoding="utf-8")
    assert "failed = pyqtSignal(str)" in quelle
    assert "self.failed.emit" in quelle
    assert "logger.warning" in quelle


def test_i18n_schluessel_vorhanden() -> None:
    i18n = (_ROOT / "Aurik10" / "i18n" / "__init__.py").read_text(encoding="utf-8")
    for key in ("preview.button", "preview.tooltip", "preview.keine_datei", "preview.laeuft_schon"):
        assert f'"{key}"' in i18n, key

"""Aurik10/ui/preview_restorer.py — Segment-Vorschau der Restaurierung (Bedienung/Echtzeit).

Vorher/Nachher-Vorschau VOR dem Batch: ein kurzes Segment (Standard 6 s) wird
über den echten Produktions-Pfad restauriert (`backend.api.bridge.restore_segment_preview`,
Pre-Analysis → Denker) und an den Aufrufer zurückgegeben, der Original und
Ergebnis abwechselnd abspielt.

Produktreife-Kontext (2026-11-19, Paritäts-Register „Bedienung/Echtzeit“):
RX bietet interaktives Vorhören; Aurik hatte KEINEN Preview vor dem Batch.
Das hier ist der erste Schritt: ehrliche Segment-Vorschau mit Fortschritt —
kein Echtzeit-Streaming (das bliebe eine eigene Architektur-Entscheidung),
aber der Nutzer hört den Effekt, bevor er Minuten an Batch-Zeit investiert.

Threading-Vertrag: die Restaurierung läuft im QThread (UI bleibt responsiv);
Fehler werden als Signal gemeldet (kein stiller Fallback, §V6 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

PREVIEW_SEGMENT_S = 6.0


class PreviewRestorerThread(QThread):
    """Restauriert ein Segment im Hintergrund und meldet Ergebnis/Fehler."""

    progress = pyqtSignal(str)  # menschlicher Fortschritts-Text (Phase/Status)
    done = pyqtSignal(object)  # np.ndarray (restauriertes Segment, float32)
    failed = pyqtSignal(str)  # Fehlermeldung (Deutsch, für Toast/Statusleiste)

    def __init__(self, audio: np.ndarray, sr: int, input_path: str = "", mode: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._audio = np.asarray(audio, dtype=np.float32)
        self._sr = int(sr)
        self._input_path = input_path
        self._mode = mode

    def run(self) -> None:
        try:
            from backend.api.bridge import restore_segment_preview

            self.progress.emit("Vorschau: Voranalyse läuft …")
            ergebnis = restore_segment_preview(self._audio, self._sr, self._input_path, self._mode)
            self.done.emit(np.asarray(ergebnis, dtype=np.float32))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Preview-Restaurierung fehlgeschlagen (%s) — §V6-Ersatzpfad an die UI.", exc)
            self.failed.emit(f"Vorschau fehlgeschlagen: {exc}")


def segment_fuer_preview(
    audio: np.ndarray, sr: int, sekunden: float = PREVIEW_SEGMENT_S, offset_s: float = 0.0
) -> np.ndarray:
    """Deterministischer Segment-Schnitt: Mitte-Standard, sonst fester Offset.

    Channels-first (C, N) und Mono (N,) werden bedient — das Segment wird auf
    der Zeitachse geschnitten (§Stereo-Layout-Invariante (copilot-instructions.md)).
    """
    x = np.asarray(audio, dtype=np.float32)
    n = int(round(sekunden * sr))
    if x.shape[-1] <= n:
        return x
    mitte = max(0, int(offset_s * sr)) if offset_s > 0 else (x.shape[-1] - n) // 2
    mitte = min(mitte, x.shape[-1] - n)
    return x[..., mitte : mitte + n]

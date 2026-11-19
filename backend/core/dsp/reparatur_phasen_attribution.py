"""DRT-13 (Spec 26, 2026-11-17): Reparatur-Phasen-Snapshot, Attribution, Teil-Rollback.

Warum: DRT-10/M1 schreibt bei Überrestauration den Identitäts-Rollback des
ganzen Titels vor — damit gehen belegte Defekt-Gewinne verloren (Jets-vinyl:
defektlast −0,203 bei musik_mos −0,551). Dieser Baustein erlaubt den
**Teil-Rollback**: nur die Regionen, die eine Reparatur-Phase verändert hat,
werden auf den Phasen-Eingang zurückgenommen; der Ganz-Titel-Rollback bleibt
der Fallback (Spec 26 DRT-13, Stufe 1: Einzel-Pfad).

Vertrag:
- **Chunk-lokal begrenzt** (max. 4 Phasen-Snapshots) — kein Ganz-Titel-Speicher.
- **Deterministisch** (§G5 (copilot-instructions.md)): reine Kopien + Masken,
  keine Zufallsgrößen, keine Zeitabhängigkeit.
- **Report-Only-Infrastruktur**: die Entscheidung (ob der Teil-Rollback das
  Ergebnis wird) misst der UV3-DRT-10-Verdikt über die Phase-AUS-Variante —
  hier wird nur der Snapshot gehalten und die Variante gebaut.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_PHASEN = 4

_registry: dict[str, dict[str, Any]] = {}


def snapshot(phase_id: str, audio: np.ndarray, region_mask: np.ndarray | None = None) -> None:
    """Phasen-Eingang + Region-Maske chunk-lokal merken (DRT-13).

    Args:
        phase_id: stabiler Phasen-Name (z. B. ``phase_07_declip``).
        audio: Phasen-Eingang (beliebiges Layout — die Variante erwartet
            exakt dasselbe Shape wie die spätere Ausgabe).
        region_mask: bool-Maske der Regionen, die die Phase verändert hat.
            Ohne Maske wird nichts zurückgenommen (kein stiller Ganz-Ersatz).
    """
    x = np.asarray(audio)
    if region_mask is None:
        maske = np.zeros(x.shape, dtype=bool)
    else:
        maske = np.asarray(region_mask, dtype=bool)
    _registry[phase_id] = {"audio": x.copy(), "maske": maske.copy()}
    while len(_registry) > _MAX_PHASEN:
        _registry.pop(next(iter(_registry)), None)
    logger.debug("DRT-13-Snapshot: %s (%d Regionen-Samples)", phase_id, int(maske.sum()))


def clear() -> None:
    """Alle Snapshots verwerfen (Song-Isolation §V8/§G1 (copilot-instructions.md))."""
    _registry.clear()


def phase_aus_variante(audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Phase-AUS-Variante: alle gesnapshotteten Reparatur-Regionen zurückgenommen.

    Returns:
        (variante, info) — ``info`` ist leer, wenn keine passenden Snapshots
        vorliegen; die Variante ist dann identisch zum Eingang.
    """
    out = np.asarray(audio, dtype=np.float32).copy()
    info: dict[str, Any] = {}
    for phase_id, eintrag in _registry.items():
        pre = np.asarray(eintrag["audio"])
        maske = np.asarray(eintrag["maske"])
        if pre.shape != out.shape or maske.shape != out.shape:
            info[phase_id] = {"regionen": 0, "grund": "shape_mismatch"}
            continue
        if not maske.any():
            info[phase_id] = {"regionen": 0, "grund": "keine_regionen"}
            continue
        out[maske] = pre[maske]
        info[phase_id] = {"regionen": int(maske.sum())}
    return out, info


def attribution_report() -> dict[str, Any]:
    """Report-Only-Metadaten-Beitrag (Schema ``aurik.phase_attribution/1``)."""
    return {
        "schema": "aurik.phase_attribution/1",
        "phasen": sorted(_registry.keys()),
        "regionen": {k: int(np.asarray(v["maske"]).sum()) for k, v in _registry.items()},
    }

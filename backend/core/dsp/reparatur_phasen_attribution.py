"""DRT-13 (Spec 26, 2026-11-17): Reparatur-Phasen-Snapshot, Attribution, Teil-Rollback.

Warum: DRT-10/M1 schreibt bei Überrestauration den Identitäts-Rollback des
ganzen Titels vor — damit gehen belegte Defekt-Gewinne verloren (Jets-vinyl:
defektlast −0,203 bei musik_mos −0,551). Dieser Baustein erlaubt den
**Teil-Rollback**: nur die Regionen, die eine Reparatur-Phase verändert hat,
werden auf den Phasen-Eingang zurückgenommen; der Ganz-Titel-Rollback bleibt
der Fallback (Spec 26 DRT-13, Stufe 1: Einzel-Pfad).

Vertrag:
- **Ausführungskontext-isoliert** (max. 4 Phasen-Snapshots) — kein global geteilter Zustand.
- **Chunk-lokal begrenzt** — sparse Deltas werden sample-first an die Song-Assembly übertragen.
- **Deterministisch** (§G5 (copilot-instructions.md)): reine Kopien + Masken,
  keine Zufallsgrößen, keine Zeitabhängigkeit.
- **Report-Only-Infrastruktur**: die Entscheidung (ob der Teil-Rollback das
  Ergebnis wird) misst der UV3-DRT-10-Verdikt über die Phase-AUS-Variante —
  hier wird nur der Snapshot gehalten und die Variante gebaut.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

import numpy as np

from backend.core.audio_layout import is_channels_first, is_samples_first, to_samples_first

logger = logging.getLogger(__name__)

_MAX_PHASEN = 4

_registry: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar("drt13_attribution_registry", default=None)


def _current_registry() -> dict[str, dict[str, Any]]:
    return _registry.get() or {}


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
    registry = dict(_current_registry())
    registry[phase_id] = {"audio": x.copy(), "maske": maske.copy()}
    while len(registry) > _MAX_PHASEN:
        registry.pop(next(iter(registry)), None)
    _registry.set(registry)
    logger.debug("DRT-13-Snapshot: %s (%d Regionen-Samples)", phase_id, int(maske.sum()))


def clear() -> None:
    """Alle Snapshots verwerfen (Song-Isolation §V8/§G1 (copilot-instructions.md))."""
    _registry.set({})


def begin_song() -> None:
    """Beginnt einen isolierten Snapshot-Lebenszyklus für einen Song oder Chunk."""
    clear()


def phase_aus_variante(audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Phase-AUS-Variante: alle gesnapshotteten Reparatur-Regionen zurückgenommen.

    Returns:
        (variante, info) — ``info`` ist leer, wenn keine passenden Snapshots
        vorliegen; die Variante ist dann identisch zum Eingang.
    """
    out = np.asarray(audio).copy()
    info: dict[str, Any] = {}
    for phase_id, eintrag in _current_registry().items():
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


def phase_aus_delta(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Liefert sparse Phase-AUS-Deltas mit Sample-first-Indizes."""
    original = np.asarray(audio)
    variant, info = phase_aus_variante(original)
    if not any(int(details.get("regionen", 0)) > 0 for details in info.values()):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), info
    if original.ndim == 2 and not (is_channels_first(original) or is_samples_first(original)):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            {phase_id: {**details, "grund": "layout_ambiguous"} for phase_id, details in info.items()},
        )

    original_sf = to_samples_first(original)
    variant_sf = to_samples_first(variant)
    delta = np.asarray(variant_sf - original_sf)
    if not np.isfinite(delta).all():
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            {phase_id: {**details, "grund": "non_finite_delta"} for phase_id, details in info.items()},
        )
    if delta.ndim == 1:
        indices = np.flatnonzero(delta)
        values = delta[indices]
    else:
        changed = np.any(delta != 0, axis=1)
        indices = np.flatnonzero(changed)
        values = delta[changed]
    return indices.astype(np.int64, copy=False), np.asarray(values), info


def attribution_report() -> dict[str, Any]:
    """Report-Only-Metadaten-Beitrag (Schema ``aurik.phase_attribution/1``)."""
    registry = _current_registry()
    return {
        "schema": "aurik.phase_attribution/1",
        "phasen": sorted(registry.keys()),
        "regionen": {k: int(np.asarray(v["maske"]).sum()) for k, v in registry.items()},
    }

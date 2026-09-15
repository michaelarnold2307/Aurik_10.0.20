"""§SOTA-R8 (Sparse Repair) — Defekt-Masken als Rechen-Masken (PERF-D).

Roadmap 2026-09-14, R8: „Sparse Repair — Reparatur nur in Defekt-Nähe statt
Vollband. Rechenzeit + Artefakt-Risiko sinken gemeinsam." Dieses Modul ist das
generische Muster für den per-Phase-Rollout: Eine Phase übergibt ihre bereits
vorhandene Defekt-Maske und ihre bestehende Reparaturfunktion; die Reparatur
läuft dann nur noch in den Defekt-Regionen (+ Kontext) statt Vollband.

Invarianten:
- leere Maske → bit-identischer Passthrough (kein Reparatur-Aufruf)
- Maske fast voll (coverage ≥ threshold) → EIN Voll-Repair (Sparse wäre teurer)
- sonst: Reparatur je zusammenhängender Region + Kontext, Hann-Crossfade an den
  Rändern (kein Naht-Artefakt)
- deterministisch (§G5 (GEBOTE.md)), NaN/Inf-geschützt (§0a), Layout-sicher (C,N)/(N,C)

Autor: Aurik Testing Team
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from backend.core.audio_layout import is_channels_first, to_channels_first, to_samples_first

logger = logging.getLogger(__name__)


@dataclass
class SparseRepairResult:
    """Ergebnis einer Sparse-Repair-Ausführung."""

    audio: np.ndarray
    regions_repaired: int  # Anzahl reparierter Defekt-Regionen
    coverage: float  # Masken-Anteil ∈ [0, 1]
    full_repair: bool  # True wenn Voll-Repair (coverage ≥ threshold)
    skipped: bool  # True wenn keine Reparatur nötig war (Passthrough)


def defect_regions(mask: np.ndarray, min_gap_samples: int = 8) -> list[tuple[int, int]]:
    """Zusammenhängende True-Regionen (start, end) exklusiv aus einer Sample-Maske.

    Lücken < min_gap_samples werden zu einer Region zusammengezogen
    (keine Mikro-Fenster, weniger Overhead).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError(f"defect_regions: Maske muss 1-D sein, erhalten {m.ndim}-D")
    if m.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))
    starts = idx[0::2]
    ends = idx[1::2]
    regions: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        # Lücken < min_gap_samples zu einer Region zusammenziehen
        if regions and int(s) - regions[-1][1] < int(min_gap_samples):
            regions[-1] = (regions[-1][0], int(e))
        else:
            regions.append((int(s), int(e)))
    return regions


def sparse_windowed_repair(
    audio: np.ndarray,
    sr: int,
    defect_mask: np.ndarray,
    repair_fn: Callable[[np.ndarray], np.ndarray],
    context_ms: float = 25.0,
    crossfade_ms: float = 5.0,
    coverage_threshold: float = 0.85,
    min_gap_samples: int = 8,
) -> SparseRepairResult:
    """Führt ``repair_fn`` nur in Defekt-Nähe aus (§SOTA-R8, PERF-D).

    Args:
        audio: Mono (N,) oder Stereo — (C,N) UND (N,C) werden bedient.
        sr: Abtastrate (für ms→Samples-Umrechnung).
        defect_mask: Bool-/Int-Maske über die Sample-Achse (len == N).
        repair_fn: Reparaturfunktion window→window (gleiche Form, gleicher dtype).
        context_ms: Kontext links/rechts je Region (Default 25 ms).
        crossfade_ms: Hann-Crossfade-Breite an den Fenster-Rändern (Default 5 ms).
        coverage_threshold: Ab diesem Masken-Anteil wird EINMAL voll repariert.
        min_gap_samples: Regions-Merging-Lücke (s. ``defect_regions``).

    Returns:
        SparseRepairResult — ``audio`` hat immer die Eingabe-Form.
    """
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    _was_cf = arr.ndim == 2 and is_channels_first(arr)
    work = to_channels_first(arr) if arr.ndim == 2 else arr[None, :] if arr.ndim == 1 else None
    if work is None:
        raise ValueError(f"sparse_windowed_repair: audio muss 1-D oder 2-D sein, erhalten {arr.ndim}-D")

    n = work.shape[1]
    mask = np.asarray(defect_mask, dtype=bool).ravel()
    if mask.size != n:
        raise ValueError(f"sparse_windowed_repair: Maskenlänge {mask.size} != Sample-Anzahl {n}")

    coverage = float(np.mean(mask)) if n else 0.0
    if not np.any(mask):
        return SparseRepairResult(audio=arr.copy(), regions_repaired=0, coverage=0.0, full_repair=False, skipped=True)

    if coverage >= float(coverage_threshold):
        out = _nan_guard(repair_fn(work))
        out = out.astype(np.float32)
        if out.shape != work.shape:
            raise ValueError(f"sparse_windowed_repair: repair_fn lieferte Form {out.shape}, erwartet {work.shape}")
        return SparseRepairResult(
            audio=_restore_layout(out, arr, _was_cf),
            regions_repaired=1,
            coverage=coverage,
            full_repair=True,
            skipped=False,
        )

    ctx = int(context_ms * sr / 1000.0)
    fade = int(crossfade_ms * sr / 1000.0)
    out = work.copy()
    regions = defect_regions(mask, min_gap_samples=min_gap_samples)
    repaired_count = 0
    for s, e in regions:
        w0 = max(0, s - ctx)
        w1 = min(n, e + ctx)
        window = work[:, w0:w1]
        try:
            fixed = _nan_guard(repair_fn(window))
        except Exception as exc:  # §V6 (copilot-instructions.md): Region unverändert lassen
            logger.warning(
                "sparse_repair: repair_fn für Region [%d:%d] fehlgeschlagen (%s) — Region unverändert", w0, w1, exc
            )
            continue
        if fixed.shape != window.shape:
            logger.warning(
                "sparse_repair: repair_fn für Region [%d:%d] lieferte Form %s statt %s — Region unverändert",
                w0,
                w1,
                fixed.shape,
                window.shape,
            )
            continue
        # Hann-Crossfade an den Fenster-Rändern (nicht an den Signalrändern)
        merged = fixed.astype(np.float32)
        fade_in = min(fade, (w1 - w0) // 2)
        if fade_in > 0 and w0 > 0:
            t = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
            ramp = 0.5 * (1.0 - np.cos(np.pi * t))[None, :]
            merged[:, :fade_in] = window[:, :fade_in] * (1.0 - ramp) + fixed[:, :fade_in] * ramp
        fade_out = min(fade, (w1 - w0) // 2)
        if fade_out > 0 and w1 < n:
            t = np.linspace(0.0, 1.0, fade_out, dtype=np.float32)
            ramp = 0.5 * (1.0 - np.cos(np.pi * t))[None, :]
            merged[:, -fade_out:] = window[:, -fade_out:] * ramp + fixed[:, -fade_out:] * (1.0 - ramp)
        out[:, w0:w1] = merged
        repaired_count += 1

    return SparseRepairResult(
        audio=_restore_layout(out, arr, _was_cf),
        regions_repaired=repaired_count,
        coverage=coverage,
        full_repair=False,
        skipped=False,
    )


def _nan_guard(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]


def _restore_layout(out_cf: np.ndarray, original: np.ndarray, was_channels_first: bool) -> np.ndarray:
    if original.ndim == 1:
        return out_cf[0].astype(np.float32)  # type: ignore[no-any-return]
    if was_channels_first:
        return out_cf.astype(np.float32)  # type: ignore[no-any-return]
    return to_samples_first(out_cf).astype(np.float32)  # type: ignore[no-any-return]

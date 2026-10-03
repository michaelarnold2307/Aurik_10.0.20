"""audibility_targets — Reparatur-Zielabnahme „unter die Hörbarkeitsschwelle".

Hörordnung Ebene 2 (.github/instructions/hoerordnung.instructions.md):
eine Reparatur gilt erst als abgeschlossen, wenn ihr Rest UNTER der
psychoakustischen Maskierungsschwelle liegt. Abgrenzung: audibility_gate
entscheidet VOR der Reparatur („Defekt hörbar?"), dieses Modul nimmt NACH
der Reparatur ab („Rest unter Ziel? Material erhalten?").

Messsemantik (EINE Wahrheit, von tests/test_audibility_targets.py
gegenverankert):
- threshold_db : Maskierungsschwelle des Kontexts (masking_model,
  75. Perzentil der Frame-Schwellen — gate-konform).
- target_db    = threshold_db − margin_db (Sicherheitsziel nach unten).
- audible      = delta_db > threshold_db (ab Maske hörbar).
- objective_met = delta_db <= target_db (sicher unter Ziel).

Delta-Messung IMMER über masking_model._frame_band_energies (dieselbe
Domäne wie die Schwelle — Befund 2026-10-02: Eigenmessungen drifteten
um >100 dB). Das Schwerpunkt-Fenster wird tail-gepadet, damit das
Frame-Raster der Referenzmessung mindestens zwei Fenster sieht
(IndexError-Befund 2026-10-02). Die Defekt-Region wird für die
Rest-Maske ausgenullt (audibility_gate-Muster — kein Selbst-Maskieren
des Rests).

Fail-closed: Messfehler ⇒ „nicht erreicht"/„hörbar" (§V6 (VERBOTEN.md)),
Determinismus §G5 (GEBOTE.md), NaN-sicher §0a (copilot-instructions.md).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_LO, _HI = 800.0, 10000.0
_N_FFT = 512
_GRAV_WIN = 512  # Schwerpunkt-Fenster (Energiefenster für den hörbarsten Moment)
_PAD = 4096  # Tail-Pad = masking_model._N_FFT: garantiert zwei volle Frames
_MISSING = {
    "audible": True,
    "objective_met": False,
    "delta_db": 0.0,
    "threshold_db": 0.0,
    "target_db": 0.0,
    "overshoot_db": 0.0,
    "margin_db": 0.0,
}


def repair_strength_for(margin_db: float) -> float:
    """Stärke ∝ Hörbarkeits-Überschuss, geclippt [0.25, 1.0] (Hörordnung §4)."""
    return float(np.clip(abs(float(margin_db)) / 12.0, 0.25, 1.0))


def _gravity_window(seg: np.ndarray) -> np.ndarray:
    """512er-Schwerpunkt-Fenster, tail-gepadet auf masking_model._N_FFT (4096).

    Tail-Pad um _PAD (= masking_model._N_FFT) am Ende (Befund 2026-10-02):
    _frame_band_energies indiziert das Hop-Raster mit n_fft=4096; ein Fenster
    von exakt 4096 Samples läge zu kurz für einen zweiten Frame-Start —
    das Pad sorgt für zwei volle Frames und verhindert den IndexError,
    der die Messung in den fail-closed-Weg warf.
    """
    if seg.size <= _GRAV_WIN:
        win = np.pad(seg, (0, _GRAV_WIN))
    else:
        power = seg * seg
        center = int(np.argmax(np.convolve(power, np.ones(_GRAV_WIN), mode="same")))
        start = int(np.clip(center - _GRAV_WIN // 2, 0, seg.size - _GRAV_WIN))
        win = seg[start : start + _GRAV_WIN]
    return np.asarray(np.pad(win, (0, _PAD)))  # type: ignore[no-any-return]


def residual_audibility(
    residual: np.ndarray,
    context: np.ndarray,
    sr: int,
    margin_db: float = 6.0,
    lo_hz: float = _LO,
    hi_hz: float = _HI,
) -> dict[str, float | bool]:
    """Hörbarkeit eines Residuals gegen die Maskierung des Kontexts.

    Richtungen: residual = original − repariert (Restdefekt) ODER entferntes
    Material (Materialverlust — gegen den Verbleib gemessen). Der hörbarste
    Moment zählt (Schwerpunkt-Fenster); Delta und Schwelle kommen aus
    derselben masking_model-Domäne.
    """
    res = np.nan_to_num(np.asarray(residual, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    ctx = np.nan_to_num(np.asarray(context, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    m = float(margin_db)
    if res.size == 0 or not np.any(res):
        return {
            "audible": False,
            "objective_met": True,
            "delta_db": -200.0,
            "threshold_db": 0.0,
            "target_db": -m,
            "overshoot_db": -200.0,
            "margin_db": m,
        }
    if ctx.size == 0:
        logger.warning("residual_audibility ohne Kontext — fail-closed 'nicht erreicht' (§V6 (VERBOTEN.md))")
        out = dict(_MISSING)
        out["margin_db"] = m
        return out
    try:
        from backend.core.dsp.masking_model import (
            _frame_band_energies,
            bark_band_edges,
            compute_masking_threshold_db,
        )

        thr_frames, _ = compute_masking_threshold_db(ctx, sr)
        edges = bark_band_edges(sr)
        band_idx = np.where((edges[:-1] < hi_hz) & (edges[1:] > lo_hz))[0]
        if len(band_idx) == 0:
            return {
                "audible": False,
                "objective_met": True,
                "delta_db": -200.0,
                "threshold_db": 0.0,
                "target_db": -m,
                "overshoot_db": -200.0,
                "margin_db": m,
            }
        threshold_db = float(np.max(np.percentile(np.asarray(thr_frames)[:, band_idx], 75, axis=0)))
        # Delta in SELBER Domäne wie die Schwelle: Referenz-Messfunktion.
        band_e_db, _ = _frame_band_energies(_gravity_window(res), sr)
        delta_db = float(np.max(np.asarray(band_e_db)[:, band_idx]))
    except Exception as _exc:  # §V6 (VERBOTEN.md): nie still „alles gut"
        logger.warning("residual_audibility: Messung fehlgeschlagen (%s) — fail-closed", _exc)
        out = dict(_MISSING)
        out["margin_db"] = m
        return out

    target_db = threshold_db - m
    return {
        "audible": bool(delta_db > threshold_db),
        "objective_met": bool(delta_db <= target_db),
        "delta_db": round(delta_db, 2),
        "threshold_db": round(threshold_db, 2),
        "target_db": round(target_db, 2),
        "overshoot_db": round(delta_db - threshold_db, 2),
        "margin_db": m,
    }


def repair_objective_met(
    original: np.ndarray,
    repaired: np.ndarray,
    defect_mask: np.ndarray,
    sr: int,
    margin_db: float = 6.0,
) -> dict[str, bool | float]:
    """Doppelte Zielabnahme: Materialerhalt UND Rest-unter-Ziel.

    Materialerhalt: entferntes Material AUSSERHALB der Defektstellen darf
    gegen den Verbleib nicht hörbar sein (gleichfrequente Pegeländerung ist
    maskiert — anderfrequentes fehlendes Material ist hörbarer Verlust).
    Rest-unter-Ziel: AN den Defektstellen bleibt der Impulsrest unter
    target_db; die Maske kommt aus dem Kontext mit ausgenULLTER Defektstelle
    (sonst maskiert der Rest sich selbst — Befund 2026-10-02).
    """
    orig = np.asarray(original, dtype=np.float64).ravel()
    rep = np.asarray(repaired, dtype=np.float64).ravel()
    n = min(orig.size, rep.size)
    orig, rep = orig[:n], rep[:n]
    m_in = np.asarray(defect_mask, dtype=bool).ravel()
    mask = np.zeros(n, dtype=bool)
    mask[: min(n, m_in.size)] = m_in[: min(n, m_in.size)]
    removed = orig - rep

    outside = removed.copy()
    outside[mask] = 0.0
    mat = residual_audibility(outside, rep, sr, margin_db=margin_db)
    material_preserved = not bool(mat["audible"])

    from scipy.signal import medfilt

    rest = np.abs(rep - medfilt(rep, kernel_size=5)) * mask
    rest_context = rep.copy()
    rest_context[mask] = 0.0
    rest_rep = residual_audibility(rest, rest_context, sr, margin_db=margin_db)
    rest_subaudible = bool(rest_rep["objective_met"])

    return {
        "material_preserved": material_preserved,
        "rest_subaudible": rest_subaudible,
        "objective_met": bool(material_preserved and rest_subaudible),
        "material_overshoot_db": float(mat["overshoot_db"]),
        "rest_overshoot_db": float(rest_rep["overshoot_db"]),
    }


def verify_declick_repair(
    original: np.ndarray, repaired: np.ndarray, defect_mask: np.ndarray, sr: int, margin_db: float = 6.0
) -> dict[str, bool | float]:
    """Zielabnahme für Declick-Kern-Repairs (declick_core)."""
    return repair_objective_met(original, repaired, defect_mask, sr, margin_db=margin_db)


def repair_until_inaudible(
    repair_fn,
    signal: np.ndarray,
    defect_mask: np.ndarray,
    sr: int,
    margin_db: float = 6.0,
    max_iter: int = 5,
) -> tuple[np.ndarray, dict[str, bool | float]]:
    """Pleasantness-First: kleinste Stärke, die das Reparaturziel erreicht.

    ``repair_fn(signal, strength) -> repaired`` läuft deterministisch über
    aufsteigende Stärken (0.3, 0.5, 0.7, dosiert, 1.0 — begrenzt auf
    max_iter); gewinnt die schwächste Stärke mit objective_met. Ohne
    Zielerreichung bleibt der beste Lauf ehrlich bei objective_met=False
    (§0c: bestmöglich statt Hardstop).
    """
    sig = np.asarray(signal, dtype=np.float64)
    strengths = sorted({repair_strength_for(margin_db), 0.3, 0.5, 0.7, 1.0})[: max(1, int(max_iter))]
    best, best_rep = (
        sig,
        {
            "objective_met": False,
            "material_preserved": False,
            "rest_subaudible": False,
            "iterations": 0,
            "strength_used": 0.0,
        },
    )
    for i, s in enumerate(strengths, start=1):
        cand = np.asarray(repair_fn(sig, float(s)), dtype=np.float64)
        rep = repair_objective_met(sig, cand, defect_mask, sr, margin_db=margin_db)
        rep["iterations"] = i
        rep["strength_used"] = float(s)
        best, best_rep = cand, rep
        if bool(rep["objective_met"]):
            break
    return best, best_rep


__all__ = [
    "repair_strength_for",
    "residual_audibility",
    "repair_objective_met",
    "verify_declick_repair",
    "repair_until_inaudible",
]

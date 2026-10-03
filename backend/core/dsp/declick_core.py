"""
declick_core.py — Kanonischer Impuls-Kern der Declicker-Familie (SOTA).

EINE Quelle der Wahrheit für Klick-/Knister-Erkennung, -Dosierung und
-Reparatur: automatic_declicker, automatic_declicker_multiband,
automatic_decrackler, shellac_declicker, riaa_declicker.

Drei Stufen (Punkte 1+2 der Wohlklang-Roadmap 2026-10-02):
  1. Erkennung: robuste MAD-/Residuum-Textur-Ausreißer (lokal adaptiv,
     maskierungsbewusst) statt peak-relativer Schwellen.
  2. Dosierung: Reparatur-Stärke ∝ Hörbarkeits-Marge jedes Impulses
     (``repair_strength_for`` aus
     :mod:`backend.core.dsp.audibility_targets`) — knapp hörbare Impulse
     werden minimal korrigiert (halbierter Klick ≈ −6 dB ⇒ unter Maske),
     störende bekommen volle AR-Reparatur. Zentrale Skalierung statt
     phasenindividueller Schwellen (§V7 (VERBOTEN.md)).
  3. Reparatur: beidseitige AR-Interpolation (Burg für kurze Klicks,
     Levinson-Durbin für dichtes Knister) mit Crossfade — keine harten
     Median-Kanten; Identitäts-Guard: bit-identisch außerhalb messbarer
     Defekte (Contract identity_budget).

Zielabnahme (Hörordnung Ebene 2): ``guard_declick``/``verify_declick_repair``
nehmen jede Reparatur doppelt ab (Materialerhalt + Rest unter Maske+Marge)
und loggen Verfehlungen §V6 (VERBOTEN.md)-sichtbar — nie Abbruch (§0c
(copilot-instructions.md): bestmögliches sicheres Ergebnis).

Stereo-Layout-Invariante (AGENTS.md): kanalweise auf der Zeitachse
(:mod:`backend.core.audio_layout`), nie mean(axis=0) ohne Layout-Check.
Determinismus (§G5 (GEBOTE.md)): keine Zufallszahlen, keine Zeitstempel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.signal import medfilt

from backend.core.audio_layout import is_channels_first
from backend.core.dsp.adaptive_ar_prediction_burg import ar_fill_gap, burg_ar_coefficients
from backend.core.dsp.adaptive_ar_prediction_levinson import levinson_durbin_ar

logger = logging.getLogger(__name__)

_ABS_FLOOR = 1e-6  # unter Quantisierungsgrenze 16 Bit (3.05e-5)
_REPAIR_EPS = 1e-5  # Identitäts-Guard: nur messbare Defekte werden ersetzt
_MAX_MARGIN_DB = 24.0  # Dosierungs-Sättigung (≥ 24 dB Überschuss = volle Stärke)


# ---------------------------------------------------------------------------
# Parameter-Semantik — eine Wahrheit für die gesamte Declicker-Familie
# ---------------------------------------------------------------------------


def strictness_from_threshold(threshold: float) -> float:
    """`threshold` (0.1–1.0, höher = konservativer) → robuste k-Sigma-Strenge.

    Historische Bedeutung „Anteil des Maximal-Residuums" wird bewahrt:
    kleineres threshold ⇒ aggressivere Reparatur (Transienten-Smear-Riskiko,
    DSPContract automatic_declicker), größeres threshold ⇒ zurückhaltender.
    """
    t = float(np.clip(threshold, 0.05, 1.0))
    return 1.5 + 8.5 * t


def strictness_from_sensitivity(sensitivity: float) -> float:
    """`sensitivity` (höher = empfindlicher) → robuste k-Sigma-Strenge.

    Default sensitivity=1.0 ⇒ k≈6 (mittleres Profil). Der historische
    Default reparierte gar nie (Schwelle = exakt das Gipfel-Residuum).
    """
    s = max(float(sensitivity), 1e-3)
    return float(np.clip(6.0 / s, 1.5, 24.0))


# ---------------------------------------------------------------------------
# Robuste Impuls-Erkennung (mit Hörbarkeits-Marge pro Sample)
# ---------------------------------------------------------------------------


def _local_residual_scale(resid: np.ndarray, block: int = 2048) -> np.ndarray:
    """Lokale Residuum-Skala (Block-RMS, linear interpoliert, O(n)).

    Das Medianfilter-Residuum glatter Signale ist nicht impulsartig, sondern
    trägt strukturierte Buckel an den Scheiteln (Curvature-Residuum). Dagegen
    misst das lokale RMS die typische Residuum-TEXTUR: Sinusbuckel liegen
    beim Vielfachen ihres RMS unter der Schwelle, echte Impulse stechen
    heraus. Block-RMS ist bis ~10 % Impulsanteil robust; lineare
    Interpolation der Blockmitten ergibt eine stetige, deterministische
    Schwelle ohne Sprungartefakte an Blockgrenzen.
    """
    n = resid.size
    if n == 0:
        return np.asarray(np.zeros(0))  # type: ignore[no-any-return]
    block = int(max(32, min(block, n)))
    n_blocks = max(1, int(np.ceil(n / block)))
    centers = np.empty(n_blocks, dtype=np.float64)
    scales = np.empty(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        s = b * block
        e = min(n, s + block)
        seg = resid[s:e]
        centers[b] = 0.5 * (s + e - 1)
        scales[b] = float(np.sqrt(np.mean(seg * seg)))
    if n_blocks == 1:
        return np.asarray(np.full(n, scales[0]))  # type: ignore[no-any-return]
    return np.asarray(np.interp(np.arange(n), centers, scales))  # type: ignore[no-any-return]


def local_median_scale(values: Any, block: int = 2048) -> np.ndarray:
    """Gleitender Median (O(n) als Block-Median + lineare Interpolation).

    Konsistente Näherung des gleitenden Medians mit Fensterbreite ≈ block:
    pro Block den Median, an den Blockmitten verankert und interpoliert.
    Ersetzt scipy.ndimage.median_filter(size≈W) in den MAD-Detektoren —
    O(n·W) dort war der Profile-Hotspot des KAS-90-s-Budgets
    (Profilevidenz 2026-10-02: 22 s in vier Rangfilter-Aufrufen).
    Gleiche Statistik für stationäre Anteile; an Blockgrenzen geglättet
    statt springend (keine Detektions-Kanten).
    """
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n = arr.size
    if n == 0:
        return np.asarray(np.zeros(0))  # type: ignore[no-any-return]
    block = int(max(32, min(block, n)))
    n_blocks = max(1, int(np.ceil(n / block)))
    centers = np.empty(n_blocks, dtype=np.float64)
    medians = np.empty(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        s = b * block
        e = min(n, s + block)
        centers[b] = 0.5 * (s + e - 1)
        medians[b] = float(np.median(arr[s:e]))
    if n_blocks == 1:
        return np.asarray(np.full(n, medians[0]))  # type: ignore[no-any-return]
    return np.asarray(np.interp(np.arange(n), centers, medians))  # type: ignore[no-any-return]


def detect_click_mask_with_margins(
    x: Any, strictness_k: float = 6.0, med_kernel: int = 5, merge_gap: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Impuls-Maske PLUS Hörbarkeits-Marge (dB-Überschuss) je Sample.

    Marge = 20·log10(Residuum/Schwelle) — der Detektor weiß ohnehin, wie weit
    jeder Impuls über seiner lokalen Schwelle liegt (Hörbarkeits-Überschuss);
    daraus dosiert ``repair_clicks`` die Reparatur-Stärke (Punkt 2 der
    Wohlklang-Roadmap). Rückgabe: (bool-Maske, margin_db-Array; nur an
    maskierten Samples befüllt, sonst 0).
    """
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if sig.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    kernel = int(min(max(med_kernel, 3) | 1, sig.size if sig.size % 2 == 1 else sig.size - 1))
    kernel = max(kernel, 3)
    resid = np.abs(sig - medfilt(sig, kernel_size=kernel))
    scale = _local_residual_scale(resid)
    thr = np.maximum(max(float(strictness_k), 1.0) * scale, _ABS_FLOOR)
    mask = resid > thr
    margins = np.zeros(sig.size, dtype=np.float64)
    if mask.any():
        margins[mask] = np.clip(20.0 * np.log10(resid[mask] / thr[mask]), 0.0, _MAX_MARGIN_DB)
    if merge_gap > 0 and mask.any():
        idx = np.flatnonzero(mask)
        gaps = np.split(idx, np.flatnonzero(np.diff(idx) > merge_gap) + 1)
        for run in gaps:
            mask[run[0] : run[-1] + 1] = True
    return mask, margins


def detect_click_mask(x: Any, strictness_k: float = 6.0, med_kernel: int = 5, merge_gap: int = 2) -> np.ndarray:
    """Bool-Maske impulsiver Ausreißer (Kompatibilitäts-Schnittstelle).

    Residuum gegen Medianfilter, Schwellwert k · lokale Residuum-Textur
    (Block-RMS) — maskierungsbewusst (Hörordnung Ebene 2): laute/texturierte
    Passagen skalieren die Schwelle hoch, in leisen/tongenauen Passagen fallen
    auch leise Impulse auf. Benachbarte Ausreißer (Abstand ≤ merge_gap)
    verschmelzen zu einer Lücke.
    """
    mask, _ = detect_click_mask_with_margins(x, strictness_k=strictness_k, med_kernel=med_kernel, merge_gap=merge_gap)
    return mask


# ---------------------------------------------------------------------------
# AR-Reparatur mit margin-basierter Dosierung
# ---------------------------------------------------------------------------


def repair_clicks(
    x: Any,
    mask: Any,
    order: int = 8,
    ctx: int = 32,
    method: str = "burg",
    margins_db: Any | None = None,
) -> np.ndarray:
    """Ersetzt maskierte Lücken per beidseitiger AR-Interpolation — dosiert.

    Dosierung (Punkt 2): ``out = (1−w)·orig + w·AR-Füllung`` pro Lücke mit
    ``w = repair_strength_for(margin_db)`` aus der Hörbarkeits-Marge der
    Lücke (Maximum ihrer Samples). Ein knapp hörbarer Impuls wird minimal
    korrigiert (kleines w senkt ihn bereits unter die Maske), ein störender
    bekommt die volle AR-Reparatur — nie mehr Eingriff als nötig
    (Pleasantness-First, Spec v10.306).

    Identitäts-Guard: Proben, bei denen die Füllung < _REPAIR_EPS vom
    Original abweicht, bleiben bit-identisch (§G5 (GEBOTE.md)).
    method="burg" → kurze Klicks; method="levinson" → dichtes Knister.
    ``margins_db=None`` ⇒ w = 1.0 (volle Reparatur, historisches Verhalten).
    """
    sig = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    m = np.asarray(mask, dtype=bool).ravel()
    if sig.size == 0 or not m.any():
        return np.asarray(sig)  # type: ignore[no-any-return]
    margins = None if margins_db is None else np.asarray(margins_db, dtype=np.float64).ravel()
    fit = levinson_durbin_ar if method == "levinson" else burg_ar_coefficients
    out = sig.copy()
    flat_idx = np.flatnonzero(m)
    for run in np.split(flat_idx, np.flatnonzero(np.diff(flat_idx)) + 1):
        if run.size == 0:
            continue
        s, e = int(run[0]), int(run[-1]) + 1
        filled = ar_fill_gap(out, s, e, order=order, ctx=ctx, fit=fit)
        segment = slice(s, e)
        delta = np.abs(filled[segment] - out[segment])
        # Identitäts-Guard: nur messbare Defekte berühren
        apply = delta > _REPAIR_EPS
        if margins is not None:
            margin = float(np.max(margins[s:e])) if e > s else 0.0
        else:
            margin = _MAX_MARGIN_DB
        w = repair_strength_for(margin)
        blended = (1.0 - w) * out[segment] + w * filled[segment]
        out[segment] = np.where(apply, blended, out[segment])
    return np.asarray(out)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Layout-sicherer Einstieg mit Zielabnahme
# ---------------------------------------------------------------------------


def declick_signal(
    audio: Any,
    strictness_k: float = 6.0,
    med_kernel: int = 5,
    merge_gap: int = 2,
    ar_order: int = 8,
    ctx: int = 32,
    method: str = "burg",
    sr: int | None = None,
) -> np.ndarray:
    """Impuls-Reparatur für Mono oder Stereo in beliebigem Layout — dosiert.

    Shape, Layout und dtype bleiben erhalten; jeder Kanal läuft
    unabhängig auf der Zeitachse (Stereo-Layout-Invariante, AGENTS.md).
    Mit ``sr`` wird die Hörordnung-Ebene-2-Zielabnahme am Ende protokolliert
    (Materialerhalt + Rest unter Maske+Marge; Verfehlung ⇒ §V6-Warnung,
    Ergebnis bleibt bestmöglich — §0c).
    """
    arr = np.asarray(audio)
    if arr.ndim == 1:
        channels = [arr]
        axis0_is_channels = True
    elif is_channels_first(arr):
        channels = [arr[c] for c in range(arr.shape[0])]
        axis0_is_channels = True
    else:  # (N, C) — Batch-/GUI-Layout
        channels = [arr[:, c] for c in range(arr.shape[1])]
        axis0_is_channels = False

    repaired: list[np.ndarray] = []
    for ch in channels:
        mask, margins = detect_click_mask_with_margins(
            ch, strictness_k=strictness_k, med_kernel=med_kernel, merge_gap=merge_gap
        )
        if mask.any():
            logger.debug("declick_core: %d/%d Samples als Impuls markiert", int(mask.sum()), mask.size)
        fixed = repair_clicks(ch, mask, order=ar_order, ctx=ctx, method=method, margins_db=margins)
        if sr:
            _log_repair_objective(ch, fixed, mask, int(sr))
        repaired.append(fixed)

    out = np.stack(repaired, axis=0 if axis0_is_channels else 1) if arr.ndim == 2 else repaired[0]
    return np.asarray(out.astype(arr.dtype, copy=False))  # type: ignore[no-any-return]


def repair_strength_for(margin_db: float) -> float:
    """Dosierte Reparaturstärke aus der Hörbarkeits-Marge (Hörordnung §4).

    Dünne Umleitung auf die kanonische Instanz
    :mod:`backend.core.dsp.audibility_targets` — eine Stärke-Logik im Projekt.
    """
    from backend.core.dsp.audibility_targets import repair_strength_for as _for

    return _for(margin_db)


def guard_declick(original: Any, repaired: Any, sr: int | None, label: str = "declick") -> np.ndarray:
    """Never-worsen-Guard: nimmt die Reparatur doppelt ab, ändert nichts.

    (Hörordnung Ebene 2) Materialerhalt + Rest-unter-Ziel werden gemessen;
    Verfehlungen erscheinen als §V6 (VERBOTEN.md)-Warnung — das Ergebnis
    bleibt bestmöglich erhalten (§0c (copilot-instructions.md), nie Hardstop).
    Rückgabe: ``repaired`` unverändert (transparenter End-of-Pipe-Guard).
    """
    rep = np.asarray(repaired)
    if sr:
        orig = np.asarray(original)
        n = min(orig.size, rep.size)
        if n:
            flat_o = np.nan_to_num(orig.ravel()[:n], nan=0.0, posinf=0.0, neginf=0.0)
            flat_r = np.nan_to_num(rep.ravel()[:n], nan=0.0, posinf=0.0, neginf=0.0)
            mask = detect_click_mask(flat_o)
            _log_repair_objective(flat_o, flat_r, mask, int(sr), label=label)
    return np.asarray(rep)  # type: ignore[no-any-return]


def _log_repair_objective(
    original: np.ndarray, repaired: np.ndarray, mask: np.ndarray, sr: int, label: str = "declick"
) -> None:
    """Zielabnahme-Report (Hörordnung Ebene 2) — sichtbar statt still."""
    try:
        from backend.core.dsp.audibility_targets import verify_declick_repair

        report = verify_declick_repair(original, repaired, mask, sr)
        if not bool(report["objective_met"]):
            logger.warning(
                "%s: Zielabnahme verfehlt (material=%s rest=%s) — bestmöglich beibehalten (§0c/§V6 (VERBOTEN.md))",
                label,
                report["material_preserved"],
                report["rest_subaudible"],
            )
        else:
            logger.debug("%s: Zielabnahme bestanden (Materialerhalt + Rest unter Maske+Marge)", label)
    except Exception as _exc:  # Guard darf nie stören (§V6 (VERBOTEN.md))
        logger.warning("%s: Zielabnahme nicht messbar (%s) — Ergebnis unverändert", label, _exc)


# ---------------------------------------------------------------------------
# ML-Modell-Pfad (optional) — ONNX CPU / Torch, §V6-konforme Fallbacks
# ---------------------------------------------------------------------------


def load_declick_model(model_path: str, budget_key: str) -> tuple[Any, str | None]:
    """Lädt ein optionales Impuls-Reparatur-Modell (ONNX-CPU, dann Torch).

    Rückgabe (model, backend); (None, None) ⇒ klassischer DSP-Pfad
    (§V6 (VERBOTEN.md): jeder Rückfall wird mit Begründung geloggt).
    """
    try:
        from backend.core.dsp._memory_budget_guard import check_budget

        if not check_budget(budget_key, 0.1):
            logger.warning("Memory-Grenze für %s erreicht — DSP-Ersatzpfad (§V6 (VERBOTEN.md))", budget_key)
            return None, None
    except Exception as e:  # Guard nicht verfügbar ⇒ Budget unbekannt, Modell-Laden zulässig
        logger.warning("Speicher-Grenze-Guard nicht lesbar (%s) — Modell-Laden ohne Grenz-Prüfung", e)
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return session, "onnx"
    except Exception as e:
        logger.warning("ONNX-Modell konnte nicht geladen werden (%s) — Torch-Pfad wird versucht", e)
    try:
        import torch

        model = torch.jit.load(model_path)
        return model, "torch"
    except Exception as e:
        logger.warning("Torch-Modell konnte nicht geladen werden (%s) — DSP-Ersatzpfad (§V6 (VERBOTEN.md))", e)
    return None, None


def run_declick_model(model: Any, backend: str | None, audio: Any) -> np.ndarray | None:
    """Führt das Modell kanalweise aus; None ⇒ DSP-Ersatzpfad (wird geloggt)."""
    if model is None or backend is None:
        return None
    arr = np.asarray(audio)
    channels = (
        [arr]
        if arr.ndim == 1
        else (
            [arr[c] for c in range(arr.shape[0])]
            if is_channels_first(arr)
            else [arr[:, c] for c in range(arr.shape[1])]
        )
    )
    try:
        outs = []
        for ch in channels:
            if backend == "onnx":
                inp = ch.astype(np.float32)[None, None, :]
                out = model.run(None, {model.get_inputs()[0].name: inp})[0]
                outs.append(np.asarray(out).reshape(-1))
            else:
                import torch

                inp = torch.from_numpy(ch.astype(np.float32))[None, None, :]
                out = model(inp).detach().cpu().numpy().reshape(-1)
                outs.append(np.asarray(out))
        stacked = np.stack(outs, axis=0 if is_channels_first(arr) else 1) if arr.ndim == 2 else outs[0]
        return np.asarray(np.nan_to_num(stacked.astype(arr.dtype, copy=False), nan=0.0, posinf=0.0, neginf=0.0))  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("Modell-Inferenz fehlgeschlagen (%s) — DSP-Ersatzpfad (§V6 (VERBOTEN.md))", e)
        return None


__all__ = [
    "strictness_from_threshold",
    "strictness_from_sensitivity",
    "local_median_scale",
    "detect_click_mask",
    "detect_click_mask_with_margins",
    "repair_clicks",
    "repair_strength_for",
    "declick_signal",
    "guard_declick",
    "load_declick_model",
    "run_declick_model",
]

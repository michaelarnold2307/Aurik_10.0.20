"""§SOTA-PSY-A1: Audibility-Gate — „Ist der Defekt über der Maskierungsschwelle hörbar?“

Hörordnung §4: Reparatur gilt als abgeschlossen, wenn ein Defekt UNTER der
psychoakustischen Maskierungsschwelle liegt — nicht wenn sein Messwert Null
ist. Dieses Modul ist die wiederverwendbare Gate-Funktion für die
reparierenden Phasen (Rollout gemäß Roadmap-Expansionsmatrix PSY-A1):

  - `defect_audibility(...)`: Masker = Kontext um den Defekt (links+rechts),
    Defekt-Energie = Δ gegen die Kontext-Schätzung; Bark-Band-Maskierung
    nach masking_model.band_audibility (ISO 11172-3-Modell).
  - Phasen-Konvention: `audible=False` ⇒ Defekt überspringen (keine
    Reparatur nötig, Never-worsen durch Nichtstun gewahrt) und im
    Metadaten-Zähler `subaudible_defects_skipped` führen.

Deterministisch (§G5 (GEBOTE.md)); rein numpy/scipy.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_LO_HZ = 800.0
_DEFAULT_HI_HZ = 10000.0
_DEFAULT_CONTEXT_MS = 250.0


def defect_audibility(
    x: np.ndarray,
    sr: int,
    defect_start: int,
    defect_end: int,
    lo_hz: float = _DEFAULT_LO_HZ,
    hi_hz: float = _DEFAULT_HI_HZ,
    context_ms: float = _DEFAULT_CONTEXT_MS,
) -> dict[str, float | bool]:
    """Verdikt über die Hörbarkeit eines Defekts oberhalb der Maskierungsschwelle.

    Args:
        x: Mono-Signal (float), Defekt unverändert enthalten.
        sr: Abtastrate.
        defect_start/defect_end: Defektgrenzen in Samples.
        lo_hz/hi_hz: relevante Bandgrenzen (Default 800 Hz–10 kHz).
        context_ms: Kontextbreite links/rechts als Masker-Schätzung.

    Returns:
        {"audible", "delta_db", "threshold_db", "skippable"} — deterministisch.
        skippable = True, wenn der Defekt unter der Schwelle liegt (keine
        Reparatur nötig — §4-Vertrag).
    """
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).ravel()
    n = len(arr)
    d0 = max(0, int(defect_start))
    d1 = min(n, int(defect_end))
    if d1 <= d0:
        return {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True}
    ctx = int(context_ms * sr / 1000.0)
    w0 = max(0, d0 - ctx)
    w1 = min(n, d1 + ctx)
    try:
        from backend.core.dsp.masking_model import bark_band_edges, compute_masking_threshold_db

        # Masker = Kontext (Defekt-Region im „Vorher“ genullt); Schwelle über
        # das 75. Perzentil der Frame-Schwellen je Band (robust gegen
        # einzelne Ausreißer-Frames).
        before = arr[w0:w1].copy()
        before[d0 - w0 : d1 - w0] = 0.0
        thr, _ = compute_masking_threshold_db(before, sr)
        edges = bark_band_edges(sr)
        band_idx = np.where((edges[:-1] < hi_hz) & (edges[1:] > lo_hz))[0]
        if len(band_idx) == 0:
            return {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True}
        threshold_db = float(np.max(np.percentile(thr[:, band_idx], 75, axis=0)))

        # Defekt-Energie: ZENTRIERT auf den Defekt (512-Punkt-Hann), damit das
        # Verdikt nicht von der globalen Frame-Ausrichtung abhängt
        # (Befund 2026-09-14: band_audibility ist mit hop=n_fft
        # ausrichtungsabhängig und für Impuls-Gates ungeeignet).
        n_fft_d = 512
        seg = arr[d0:d1].astype(np.float64)
        if len(seg) < n_fft_d:
            pad_l = (n_fft_d - len(seg)) // 2
            seg = np.pad(seg, (pad_l, n_fft_d - len(seg) - pad_l))
        else:
            seg = seg[:n_fft_d]
        win = np.hanning(n_fft_d)
        spec_d = np.abs(np.fft.rfft(seg * win, n=n_fft_d)) ** 2
        freqs = np.fft.rfftfreq(n_fft_d, d=1.0 / sr)
        e_d = np.array(
            [np.sum(spec_d[(freqs >= edges[b]) & (freqs < edges[b + 1])]) for b in band_idx], dtype=np.float64
        )
        delta_db = float(10.0 * np.log10(np.max(e_d) + 1e-12))
        audible = bool(delta_db > threshold_db)
        return {
            "audible": audible,
            "delta_db": round(delta_db, 2),
            "threshold_db": round(threshold_db, 2),
            "skippable": not audible,
        }
    except Exception as _exc:  # §V6 (copilot-instructions.md): nie blockieren
        logger.warning(
            "Audibility-Gate fehlgeschlagen (%s) — Reparatur freigegeben (§V6 (copilot-instructions.md)).", _exc
        )
        return {"audible": True, "delta_db": 0.0, "threshold_db": 0.0, "skippable": False}

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


class SanitizedSignal:
    """§PERF-R6 (2026-09-19): defekt-unabhängige Signal-Sanitisierung EINMAL je Signal.

    ``defect_audibility`` saniert je Aufruf das KOMPLETTE Signal
    (``nan_to_num`` + float32-Konvertierung + ``ravel``) — bei 7426
    Klick-Verdikten je 30-s-Chunk (phase_27, Test-Material) waren das
    gemessen 30 s ``nan_to_num`` + 15 s Konvertierungs-Kopien von 52 s
    Gesamtlaufzeit. Die Sanitisierung hängt nicht vom Defekt ab ⇒ einmal
    je Signal; die Verdikte bleiben bit-identisch (gleiche Operationen,
    gleiche Werte).
    """

    __slots__ = ("arr", "n")

    def __init__(self, x: np.ndarray) -> None:
        self.arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).ravel()
        self.n = len(self.arr)


def defect_audibility(
    x: np.ndarray,
    sr: int,
    defect_start: int,
    defect_end: int,
    lo_hz: float = _DEFAULT_LO_HZ,
    hi_hz: float = _DEFAULT_HI_HZ,
    context_ms: float = _DEFAULT_CONTEXT_MS,
    model: str = "mpeg1",
) -> dict[str, float | bool]:
    """Verdikt über die Hörbarkeit eines Defekts oberhalb der Maskierungsschwelle.

    Args:
        x: Mono-Signal (float), Defekt unverändert enthalten.
        sr: Abtastrate.
        defect_start/defect_end: Defektgrenzen in Samples.
        lo_hz/hi_hz: relevante Bandgrenzen (Default 800 Hz–10 kHz).
        context_ms: Kontextbreite links/rechts als Masker-Schätzung.
        model: Maskierungsmodell — "mpeg1" (ISO 11172-3, Default) oder
            "zwicker" (§SOTA-PSY-A2/Q9, ISO 532-1). Default "mpeg1" ändert
            das Bestandsverhalten NICHT (§G5 (GEBOTE.md) Determinismus);
            bei Import-/Berechnungsfehlern des Zwicker-Pfades wird auf
            MPEG-1 zurückgefallen (fail-open §V6 (copilot-instructions.md)).

    Returns:
        {"audible", "delta_db", "threshold_db", "skippable"} — deterministisch.
        skippable = True, wenn der Defekt unter der Schwelle liegt (keine
        Reparatur nötig — §4-Vertrag).
    """
    return defect_audibility_from_signal(
        SanitizedSignal(x), sr, defect_start, defect_end, lo_hz, hi_hz, context_ms, model
    )


def defect_audibility_from_signal(
    sig: SanitizedSignal,
    sr: int,
    defect_start: int,
    defect_end: int,
    lo_hz: float = _DEFAULT_LO_HZ,
    hi_hz: float = _DEFAULT_HI_HZ,
    context_ms: float = _DEFAULT_CONTEXT_MS,
    model: str = "mpeg1",
) -> dict[str, float | bool]:
    """§PERF-R6: Verdikt auf vorbereitetem Signal — identische Semantik/Werte zu
    ``defect_audibility`` (bit-identisch), aber ohne erneute Vollsignal-
    Sanitisierung je Aufruf.
    """
    arr = sig.arr
    n = sig.n
    d0 = max(0, int(defect_start))
    d1 = min(n, int(defect_end))
    if d1 <= d0:
        return {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True}
    ctx = int(context_ms * sr / 1000.0)
    w0 = max(0, d0 - ctx)
    w1 = min(n, d1 + ctx)
    before = arr[w0:w1].copy()
    before[d0 - w0 : d1 - w0] = 0.0
    if model == "zwicker":
        try:
            return _defect_audibility_zwicker(arr, sr, d0, d1, before, lo_hz, hi_hz)
        except Exception as _exc:  # §V6 (copilot-instructions.md): Rückfall MPEG-1
            logger.debug(
                "Zwicker-Gate fehlgeschlagen (%s) — Rückfall auf MPEG-1 (§V6 (copilot-instructions.md)).", _exc
            )
            # durchfallen zum MPEG-1-Pfad unten
    try:
        from backend.core.dsp.masking_model import bark_band_edges, compute_masking_threshold_db

        # Masker = Kontext (Defekt-Region im „Vorher“ genullt); Schwelle über
        # das 75. Perzentil der Frame-Schwellen je Band (robust gegen
        # einzelne Ausreißer-Frames).
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
            # ZENTRUM statt Anfang: lange Defekt-Regionen (z. B. gepaddete
            # Splice-Fenster) tragen die Defekt-Energie meist in der Mitte —
            # der Anfang wäre reiner Kontext (Befund 2026-09-14, phase_56).
            off = (len(seg) - n_fft_d) // 2
            seg = seg[off : off + n_fft_d]
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


def _defect_audibility_zwicker(
    arr: np.ndarray,
    sr: int,
    d0: int,
    d1: int,
    before: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> dict[str, float | bool]:
    """Zwicker-Pfad des Audibility-Gates (§SOTA-PSY-A2/Q9, ISO 532-1).

    Maskierungsschwelle aus ``zwicker_masking_threshold_db`` (dB SPL) statt
    des MPEG-1-Bark-Modells. Der Defekt-Delta wird in derselben dB-SPL-Domäne
    gemessen (Kurzform im 1/3-Oktav-Band des relevanten Bereichs), damit
    ``audible = delta_db > threshold_db`` semantisch konsistent bleibt.

    Fehler propagieren nach oben (dort Rückfall auf MPEG-1 gemäß
    §V6 (copilot-instructions.md)). Deterministisch, layout-sicher.
    """
    from backend.core.dsp.zwicker_loudness import zwicker_masking_threshold_db

    thr_spl, freq_hz = zwicker_masking_threshold_db(before, sr)
    # Schwelle auf [lo,hi] beschränkt; 75. Perzentil wie im MPEG-1-Pfad als
    # robuste Zusammenfassung gegen Ausreißer.
    band = (freq_hz >= lo_hz) & (freq_hz <= hi_hz)
    if not band.any():
        return {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True}
    threshold_db = float(np.max(np.percentile(thr_spl[band], 75)))

    # Defekt-Delta: dB SPL des lautesten 1/3-Oktav-Bands im zentrierten Signal
    # (analog zur defekt-zentrierten Messung des MPEG-1-Pfades).
    from backend.core.dsp.zwicker_loudness import _third_octave_levels_db_spl

    seg = arr[d0:d1].astype(np.float64)
    n_fft_d = 512
    if len(seg) < n_fft_d:
        pad_l = (n_fft_d - len(seg)) // 2
        seg = np.pad(seg, (pad_l, n_fft_d - len(seg) - pad_l))
    else:
        off = (len(seg) - n_fft_d) // 2
        seg = seg[off : off + n_fft_d]
    levels = _third_octave_levels_db_spl(seg, sr)
    delta_db = float(np.max(levels))
    delta_db = max(delta_db, -200.0)

    audible = bool(delta_db > threshold_db)
    return {
        "audible": audible,
        "delta_db": round(delta_db, 2),
        "threshold_db": round(threshold_db, 2),
        "skippable": not audible,
    }

#!/usr/bin/env python3
"""MusikMOS — referenzfreie Musik-Qualitätsschätzung (UTMOS-Ersatz, E2).

**Warum ein Ersatz:** UTMOSv2 und PQS-DSP sind auf Musik gemessen **blind**
(Richtungszertifikat 2026-10-28: Spearman −0,8; Sättigung bei 4,5–5).
MusikMOS ist für **Musik** gebaut und muss vor Gebrauch das Richtungs-
Zertifikat bestehen (`scripts/quality_metric_direction_gate.py`,
Wohlklang-Vertrag B3, Bedingung 3).

**Aufbau (alle Komponenten dokumentiert, deterministisch, ohne Lernschritt):**

* ``q_rausch``  = 1 − min(1, 2·Spektral-Flatness)      — Rausch-/Defektlast
* ``q_tonal``   = Harmonizität (Median über 2048er-Rahmen) — Tonalitätserhalt
* ``q_clip``    = 1 − min(1, 10·Clipping-Rate)         — Übersteuerungsfreiheit
* ``q_hf``      = HF-Präsenz, gated bei Flatness        — spektrale Vollständigkeit
* ``q_dyn``     = Crest-Factor-Plausibilität (Glocke um 14 dB) — Dynamik

z = gewichtete Summe ∈ [0, 1]; MOS = 1 + 4·σ(10·(z − 0.5)) ∈ [1, 5],
**höher = besser** (MOS-Konvention). Richtungssinn: per Konstruktion
(Defektanteile senken z) und per Zertifikat bewiesen.

Aufruf::

    from plugins.musik_mos_plugin import get_musik_mos
    res = get_musik_mos().estimate_mos(audio, sr)   # -> MusikMOSResult
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MODEL_USED = "musik_mos_v1"
GEWICHTE = {"rausch": 0.28, "tonal": 0.24, "clip": 0.14, "hf": 0.14, "dyn": 0.20}


@dataclass
class MusikMOSResult:
    """Ergebnis: MOS ∈ [1, 5], höher = besser — mit vollständiger Herleitung."""

    mos: float
    model_used: str = MODEL_USED
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"mos": self.mos, "model_used": self.model_used, **self.details}


class MusikMOSPlugin:
    """Referenzfreie Musik-MOS-Schätzung (deterministisch, ohne Modell-Datei)."""

    def estimate_mos(self, audio: np.ndarray, sr: int) -> MusikMOSResult:
        """Schätzt Musik-Qualität 1–5 ohne Referenzsignal (höher = besser)."""
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            # Stereo-Layout-Invariante (AGENTS.md §3): (C, N) -> (N, C) erkennen
            if x.shape[0] <= 8 and x.shape[-1] > 8:
                x = x.T
            x = x.mean(axis=1)
        ziel_sr = 48000
        if sr != ziel_sr:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(ziel_sr, int(sr))
            x = resample_poly(x, ziel_sr // g, int(sr) // g).astype(np.float32)
            sr = ziel_sr
        seg = x[: min(len(x), 2048 * 16)]
        if seg.size < 2048:
            raise ValueError("Signal zu kurz für MusikMOS (mindestens 2048 Samples)")

        # Spektral-Flatness (Rauschanteil)
        spec = np.abs(np.fft.rfft(seg, n=2048)) ** 2 + 1e-18
        flatness = float(np.clip(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12), 0.0, 1.0))

        # Harmonizität: Median über 2048er-Rahmen (robust gegen Klicks)
        harmonie: list[float] = []
        rahmen = 2048
        for start in range(0, seg.size - rahmen + 1, rahmen):
            r = seg[start : start + rahmen]
            acf = np.correlate(r, r, "full")
            acf = acf[len(acf) // 2 :]
            peak = int(np.argmax(acf[20:])) + 20
            harmonie.append(float(np.clip(acf[peak] / (acf[0] + 1e-12), 0.0, 1.0)))
        q_tonal = float(np.median(harmonie)) if harmonie else 0.5

        # Clipping-Rate (Flat-Top)
        clip_ratio = float(np.mean(np.abs(seg) > 0.98))

        # HF-Präsenz (nur bei nicht-rauschartigem Spektrum)
        freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
        hf = float(np.sum(spec[freqs > 8000]) / (np.sum(spec) + 1e-12))

        # Dynamik (Crest-Factor-Plausibilität)
        rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
        crest_db = 20.0 * math.log10((float(np.max(np.abs(seg))) + 1e-12) / rms)
        q_dyn = float(math.exp(-0.5 * ((crest_db - 14.0) / 6.0) ** 2))

        q_rausch = 1.0 - min(1.0, 2.0 * flatness)
        q_clip = 1.0 - min(1.0, 10.0 * clip_ratio)
        q_hf = min(1.0, hf / 0.02) * (1.0 - flatness)

        z = (
            GEWICHTE["rausch"] * q_rausch
            + GEWICHTE["tonal"] * q_tonal
            + GEWICHTE["clip"] * q_clip
            + GEWICHTE["hf"] * q_hf
            + GEWICHTE["dyn"] * q_dyn
        )
        z = float(np.clip(z, 0.0, 1.0))
        mos = 1.0 + 4.0 / (1.0 + math.exp(-10.0 * (z - 0.5)))
        mos = float(np.clip(mos, 1.0, 5.0))
        return MusikMOSResult(
            mos=round(mos, 4),
            details={
                "z": round(z, 4),
                "flatness": round(flatness, 4),
                "q_rausch": round(q_rausch, 4),
                "q_tonal": round(q_tonal, 4),
                "q_clip": round(q_clip, 4),
                "q_hf": round(q_hf, 4),
                "q_dyn": round(q_dyn, 4),
                "crest_db": round(crest_db, 2),
            },
        )


_instance: MusikMOSPlugin | None = None


def get_musik_mos() -> MusikMOSPlugin:
    """Singleton-Fabrik (Repo-Konvention)."""
    global _instance
    if _instance is None:
        _instance = MusikMOSPlugin()
    return _instance

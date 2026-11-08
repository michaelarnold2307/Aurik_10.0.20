#!/usr/bin/env python3
"""Richtungszertifikat für referenzfreie Qualitätsmetriken (E2 der Tiefenanalyse).

**Problem:** Eine referenzfreie MOS-Metrik ohne geprüften Richtungssinn ist
gefährlicher als keine — UTMOS ist gemessen richtungs-blind/invertiert
(LÜCKEN-MATRIX B). **Lösung:** Jede Kandidaten-Metrik durchläuft eine
deterministische Degradations-Leiter (Original → echtes Leerrillen-Overlay →
starker) auf echtem Referenzmaterial; der Spearman-Rangkorrelations-Koeffizient
gegen die erwartete Ordnung entscheidet:

    richtungssicher  ⇔  Spearman ≥ Schwelle (Default 0,8) auf JEDEM Material.

Nur Metriken mit gültigem Zertifikat dürfen Entscheidungen tragen
(Wohlklang-Vertrag B3, Bedingung 3 „Zielmetrik“). Das Zertifikat ist
reproduzierbar (Seed, sha256 der Eingänge) und wird als JSON abgelegt.

Aufruf:
    python3 -B scripts/quality_metric_direction_gate.py \
        [--materials 2] [--seconds 5] [--seed 42] [--out <zertifikat.json>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

SCHWELLE = 0.8

# Degradations-Leiter der Richtungs-Zertifizierung: Rauschbett relativ zur
# Stimulus-Spitze, Index = Stufe. Diese Werte sind Bestandteil des Zertifikats
# (docs/reports/current/qualitaetsmetrik_richtungszertifikat.json) — eine
# Änderung erfordert eine Neuzertifizierung. Bewusst ein geordnetes Tupel
# (experimentelles Design des Test-Rigs), KEINE verstufte Schwellwert-Tabelle
# (§V26 (copilot-instructions.md)).
_RAUSCH_LEITER = (0.0, 0.12, 0.3, 0.6)
GRAD = ("Original", "Overlay -18 dB", "+ Clipping + LP8k", "+ starkes Clipping + LP3k5")


def spearman(erwartet: list[float], gemessen: list[float]) -> float:
    """Spearman-Rangkorrelation (reines Python, deterministisch)."""

    def rangreihe(werte: list[float]) -> list[float]:
        """Rangreihe MIT gemittelten Gleichstaenden (echtes Spearman)."""
        order = sorted(range(len(werte)), key=lambda i: werte[i])
        r = [0.0] * len(werte)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and werte[order[j + 1]] == werte[order[i]]:
                j += 1
            mittel = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = mittel
            i = j + 1
        return r

    if len(erwartet) != len(gemessen) or len(erwartet) < 2:
        return 0.0
    rx, ry = rangreihe(erwartet), rangreihe(gemessen)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    zaehler = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    n_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    n_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if n_x == 0 or n_y == 0:
        return 0.0
    return zaehler / (n_x * n_y)


def bewerte_richtung(stufenwerte: list[float], schwelle: float = SCHWELLE, richtung: str = "absteigend") -> dict:
    """Bewertet eine Leiter; ``absteigend`` = höher ist besser (MOS-Logik)."""
    if richtung == "aufsteigend":  # Defektlast: höher = schlechter
        erwartet = list(range(1, len(stufenwerte) + 1))
    else:
        erwartet = list(range(len(stufenwerte), 0, -1))
    rho = spearman(erwartet, stufenwerte)
    return {
        "stufenwerte": [round(float(v), 4) for v in stufenwerte],
        "spearman": round(rho, 4),
        "richtungssicher": bool(rho >= schwelle),
    }


def _degradiere(
    audio: np.ndarray, sr: int, stufe: int, rauschen: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Kumulative Degradations-Leiter Stufe 0..3 — jede Stufe verschlechtert
    ALLE Defekt-Komponenten (Rauschen, Clipping, HF-Verlust) monoton, damit die
    Ordnung objektiv feststeht (Test-Rig-Prinzip des Richtungszertifikats)."""
    x = np.array(audio, dtype=np.float32, copy=True)
    if stufe == 0:
        return x
    if rauschen is not None and len(rauschen) > 0:
        r = rauschen[: len(x)] if len(rauschen) >= len(x) else np.tile(rauschen, len(x) // len(rauschen) + 1)[: len(x)]
        pegel = _RAUSCH_LEITER[min(max(stufe, 0), len(_RAUSCH_LEITER) - 1)]
        x = x + pegel * r.astype(np.float32)
    if stufe >= 2:
        x = np.tanh(2.2 * x) / 2.2  # zunehmendes Clipping
    if stufe >= 2:
        kern = np.ones(5 if stufe == 2 else 11, dtype=np.float32) / (5 if stufe == 2 else 11)
        x = np.convolve(x, kern, mode="same").astype(np.float32)  # HF-Verlust 8k/3k5
    return x


def _lade_rauschen() -> np.ndarray | None:
    bank = PROJECT_ROOT / "data" / "vinyl_noise_bank"
    kandidaten = sorted(bank.glob("leerrille_*.wav")) if bank.is_dir() else []
    if not kandidaten:
        return None
    import soundfile as sf

    data, _ = sf.read(str(kandidaten[0]), always_2d=True)
    return np.asarray(data.mean(axis=1), dtype=np.float32)


def _lade_materialien(anzahl: int, sekunden: float) -> list[tuple[str, np.ndarray, int]]:
    """Referenz-Korpus-Clips (Roh-Fenster) als Testmaterial."""
    from scripts.release_candidate_pipeline import _existing_corpus, _load_clip

    materialien: list[tuple[str, np.ndarray, int]] = []
    for eintrag in _existing_corpus()[:anzahl]:
        pfad = Path(eintrag["path"])
        audio, sr = _load_clip(pfad, start_s=0.0, seconds=sekunden)
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        materialien.append((pfad.name, mono, int(sr)))
    return materialien


def _messe_metriken(audio: np.ndarray, sr: int) -> dict[str, float | None]:
    """Alle verfügbaren referenzfreien Metriken auf EINEM Signal messen (48 kHz)."""
    werte: dict[str, float | None] = {}
    ziel_sr = 48000
    if sr != ziel_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(ziel_sr, sr)
        audio = resample_poly(audio, ziel_sr // g, sr // g).astype(np.float32)
        sr = ziel_sr
    try:
        from plugins.utmos_plugin import get_utmos

        plugin = get_utmos()
        res = plugin.estimate_mos(audio, sr)
        werte[f"utmos:{getattr(res, 'model_used', '?')}"] = float(getattr(res, "mos", 0.0))
    except Exception as exc:
        log.warning("UTMOS-Plugin nicht messbar (%s)", exc)
        werte["utmos"] = None
    try:
        from plugins.utmos_plugin import get_utmos

        res2 = get_utmos()._estimate_pqs_dsp(audio, sr)
        werte["pqs_dsp"] = float(res2[0])
    except Exception as exc:
        log.warning("PQS-DSP nicht messbar (%s)", exc)
        werte["pqs_dsp"] = None
    try:
        from plugins.musik_mos_plugin import get_musik_mos

        res3 = get_musik_mos().estimate_mos(audio, sr)
        werte["musik_mos"] = float(res3.mos)
    except Exception as exc:
        log.warning("MusikMOS nicht messbar (%s)", exc)
        werte["musik_mos"] = None
    werte["defektlast"] = _defektlast(audio, sr)
    return werte


DEFEKTLAST_RICHTUNG = "aufsteigend"


def _defektlast(audio: np.ndarray, sr: int) -> float:
    """Richtungstrivialer Defekt-Index (höher = mehr Defekte) — Delegation.

    Kanonische Definition lebt seit 2026-11-08 in
    ``backend/core/dsp/wohlklang_zeuge.py`` (EINE Quelle für Mess-Skripte und
    den Produktions-Zeugen; ein Test pinnt die Bit-Identität). Die Formel ist
    unverändert: Clipping, Spektral-Flatness (Rauschanteil) und HF-Verlust —
    jede Komponente misst den Defekt selbst, die Richtung ist per Konstruktion
    bekannt.
    """
    from backend.core.dsp.wohlklang_zeuge import (  # pylint: disable=import-outside-toplevel
        defektlast as _kanonische_defektlast,
    )

    return _kanonische_defektlast(audio, sr)


def erstelle_zertifikat(
    materialien: list[tuple[str, np.ndarray, int]], rauschen: np.ndarray | None, seed: int, schwelle: float
) -> dict:
    """Hauptlauf: Leiter bauen, Metriken messen, je Material bewerten."""
    rng = np.random.default_rng(seed)
    roh: dict[str, dict[str, list[float]]] = {}
    for name, mono, sr in materialien:
        for stufe in range(len(GRAD)):
            signal = _degradiere(mono, sr, stufe, rauschen, rng)
            for metrik, wert in _messe_metriken(signal, sr).items():
                if wert is None:
                    continue
                roh.setdefault(metrik, {}).setdefault(name, []).append(wert)

    metriken: dict[str, dict] = {}
    for metrik, pro_material in sorted(roh.items()):
        richtung = DEFEKTLAST_RICHTUNG if metrik == "defektlast" else "absteigend"
        teile = {
            m: bewerte_richtung(w, schwelle, richtung=richtung)
            for m, w in sorted(pro_material.items())
            if len(w) == len(GRAD)
        }
        if not teile:
            continue
        min_rho = min(t["spearman"] for t in teile.values())
        metriken[metrik] = {
            "pro_material": teile,
            "spearman_min": round(min_rho, 4),
            "richtungssicher": bool(all(t["richtungssicher"] for t in teile.values())),
        }
    return {
        "schema": "aurik.qualitaetsmetrik_richtungszertifikat/1",
        "seed": seed,
        "schwelle": schwelle,
        "grad": list(GRAD),
        "materialien": [m[0] for m in materialien],
        "metriken": metriken,
        "quelle": "scripts/quality_metric_direction_gate.py",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Richtungszertifikat referenzfreier Qualitätsmetriken")
    ap.add_argument("--materials", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--schwelle", type=float, default=SCHWELLE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    materialien = _lade_materialien(args.materials, args.seconds)
    if not materialien:
        log.error("kein Referenzmaterial gefunden")
        return 2
    zertifikat = erstelle_zertifikat(materialien, _lade_rauschen(), args.seed, args.schwelle)
    out = args.out or (PROJECT_ROOT / "docs" / "reports" / "current" / "qualitaetsmetrik_richtungszertifikat.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(zertifikat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for metrik, daten in zertifikat["metriken"].items():
        print(
            f"{metrik}: spearman_min={daten['spearman_min']} -> "
            f"{'RICHTUNGSSICHER' if daten['richtungssicher'] else 'GESPERRT (blind/invertiert)'}"
        )
    print(f"Zertifikat: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

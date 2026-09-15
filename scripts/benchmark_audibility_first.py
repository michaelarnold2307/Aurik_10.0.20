#!/usr/bin/env python3
"""§SOTA-R4 (Audibility-First-Scheduling-Benchmark) — PSY-A1-Einsparung je Phase.

Roadmap 2026-09-14, R4: „Audibility-First-Scheduling: billige Detektion zuerst,
teure Reparatur nur bei Hörbarkeit — als Benchmark gemessen."

Dieses Skript misst die Hörbarkeits-Filterrate (Skip-Rate) der PSY-A1-
Audibility-Gates auf synthetischen Defekt-Korpora: subaudible vs. hörbare
Klicks werden in ein Basis-Signal injiziert; die Phase läuft mit aktivem Gate
und die übersprungenen Reparaturen werden gezählt (phase_01 exportiert
``subaudible_skipped`` seit 2026-09-15). Berichtet wird pro Fall:
injiziert / erkannt / übersprungen (subaudible) / Wall-Clock.

Deterministisch (Seed 42, §G5 (GEBOTE.md)). Nutzung:

    python3 -B scripts/benchmark_audibility_first.py [--duration-s 3] [--out <pfad>]

Autor: Aurik Testing Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Repo-Konvention (wie scripts/benchmark_bwe_candidates.py): Projekt-Root in
# sys.path, damit ``import backend`` auch bei direktem Skript-Aufruf funktioniert.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SR = 48000

# ─── Korpus ──────────────────────────────────────────────────────────────────


def _base_signal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Leises Hochband-Rauschbett (0,0005 RMS) — minimaler Maskierungs-Kontext.

    Bewusst OHNE lautes Breitband-Bett: das PSY-A1-Gate vergleicht die
    Defekt-Fenster-Energie gegen die Maskierungsschwelle des 250-ms-Kontexts;
    ein lauter Rauschboden maskiert Klicks/Bursts so stark, dass subaudible
    und audible Defekte ununterscheidbar werden (Befund 2026-09-15 beim
    Benchmark-Bau — dokumentiert, nicht weggebügelt).
    """
    return rng.normal(0, 0.0005, n).astype(np.float32)


def _inject_defect(sig: np.ndarray, pos: int, amplitude: float, rng: np.random.Generator) -> None:
    """20-Sample 4-kHz-Burst (Hann-gefenstert) — tonaler Defekt im Gate-Band.

    Tonale Bursts trennen am Gate sauber (Energie konzentriert in einem
    Bark-Band); breitbandige Impulse werden durch die Kontext-Maskierung der
    250-ms-Fenster-Geometrie dominiert.
    """
    dur = 20
    t_b = np.arange(dur, dtype=np.float32) / SR
    env = np.hanning(dur).astype(np.float32)
    burst = (amplitude * np.sin(2 * np.pi * 4000.0 * t_b) * env).astype(np.float32)
    end = min(pos + dur, len(sig))
    sig[pos:end] += burst[: end - pos]


def build_corpus(duration_s: float = 3.0, n_subaudible: int = 30, n_audible: int = 30, seed: int = 42) -> dict:
    """Baut die drei Benchmark-Fälle: clean / subaudible / audible Defekte.

    Defekt-Amplituden: subaudible 0,0005 (−66 dBFS, unter der Schwelle des
    leisen Bettes), audible 0,2 (deutlich darüber). Die tatsächliche
    Entscheidung trifft das Gate — das Skript zählt nur.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    base = _base_signal(n, rng)

    cases: dict[str, dict] = {}
    for name, amp, count in (("clean", 0.0, 0), ("subaudible", 0.0005, n_subaudible), ("audible", 0.2, n_audible)):
        sig = base.copy()
        positions: list[int] = []
        if count > 0:
            # Gleichmäßige Verteilung mit 0,5-s-Rand links/rechts; bei sehr
            # kurzen Korpora schrumpft der Rand proportional (Bugfix 2026-09-15:
            # vorher liefen Positionen bei kurzen Korpora über das Ende).
            margin = min(int(SR * 0.5), n // 4)
            step = max(1, (n - 2 * margin) // count)
            for i in range(count):
                pos = int(margin + i * step)
                positions.append(pos)
                _inject_defect(sig, pos, amp, rng)
        cases[name] = {
            "audio": sig,
            "injected_positions": positions,
            "injected_amplitude": amp,
            "expected_category": name,
        }
    return {"sr": SR, "seed": seed, "cases": cases}


# ─── Messung ─────────────────────────────────────────────────────────────────


def gate_oracle(case: dict, sr: int) -> dict[str, int]:
    """Direkte Gate-Entscheidung je injiziertem Klick (ohne Detektions-Schwellen).

    Zählt, wie viele der injizierten Klicks das PSY-A1-Audibility-Gate als
    skippable (subaudible) bewertet — die eigentliche Hörbarkeits-Filterrate.
    """
    from backend.core.dsp.audibility_gate import defect_audibility

    audio = np.asarray(case["audio"], dtype=np.float32)
    skip = 0
    total = 0
    for pos in case["injected_positions"]:
        d0 = max(0, pos - 4)
        d1 = min(len(audio), pos + 26)
        if d1 <= d0:
            continue
        total += 1
        try:
            res = defect_audibility(audio, sr, d0, d1, lo_hz=1200.0, hi_hz=16000.0)
            if bool(res.get("skippable", False)):
                skip += 1
        except Exception as _gate_exc:  # §V6 (copilot-instructions.md): Gate-Fehler zählen als „nicht bewertbar"
            logger.debug("gate_oracle: defect_audibility fehlgeschlagen (%s) — Klick übersprungen", _gate_exc)
            continue
    return {"gate_evaluated": total, "gate_skippable": skip}


def run_case(phase, case: dict, sr: int) -> dict:
    """Fährt eine Phase über einen Fall und sammelt Hörbarkeits-Statistiken."""
    audio = np.asarray(case["audio"], dtype=np.float32)
    t0 = time.monotonic()
    result = phase.process(audio, sample_rate=sr, material_type="vinyl")
    wall_ms = (time.monotonic() - t0) * 1000.0
    mods = getattr(result, "modifications", {}) or {}
    oracle = gate_oracle(case, sr)
    out = {
        "expected_category": case["expected_category"],
        "injected": int(len(case["injected_positions"])),
        "detected_total": int(mods.get("total_clicks_removed", 0)),
        "subaudible_skipped": int(mods.get("subaudible_skipped", 0)),
        "ml_repaired": int(mods.get("ml_repaired", 0)),
        "wall_time_ms": round(float(wall_ms), 1),
        "gate_evaluated": int(oracle["gate_evaluated"]),
        "gate_skippable": int(oracle["gate_skippable"]),
    }
    out["skip_fraction"] = round(out["subaudible_skipped"] / max(1, out["detected_total"]), 4)
    out["gate_skip_fraction"] = round(out["gate_skippable"] / max(1, out["gate_evaluated"]), 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="§SOTA-R4 Audibility-First-Benchmark")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="output/audibility_first_benchmark/audibility_first_report.json")
    args = parser.parse_args()

    from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

    corpus = build_corpus(duration_s=args.duration_s)
    phase = ClickRemovalPhase()
    report: dict = {
        "title": "§SOTA-R4 Audibility-First-Benchmark (PSY-A1-Einsparung, phase_01)",
        "sr": corpus["sr"],
        "seed": corpus["seed"],
        "duration_s": args.duration_s,
        "cases": {},
    }
    for name, case in corpus["cases"].items():
        report["cases"][name] = run_case(phase, case, corpus["sr"])
        print(
            f"[{name:>10}] injiziert={report['cases'][name]['injected']:>3} "
            f"erkannt={report['cases'][name]['detected_total']:>3} "
            f"subaudible_übersprungen={report['cases'][name]['subaudible_skipped']:>3} "
            f"gate_skip={report['cases'][name]['gate_skip_fraction']:.2f} "
            f"wall={report['cases'][name]['wall_time_ms']:.0f} ms"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

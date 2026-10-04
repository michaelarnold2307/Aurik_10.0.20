"""T6-1 A/B-Harness: A/B-Abnahme der schritt-granularen Core-Guards (Option A,
Sign-off 2026-10-03) gegen den Status quo (vorher) — die letzte offene
T6-1-Stufe laut `docs/TODOS_SOTA_ROADMAP.md` (Never-worsen-Gates +
Hoerordnungs-Abnahme).

Normative Anbindung (per Docstring-Referenz, §G9 (GEBOTE.md)):
- `docs/TODOS_SOTA_ROADMAP.md` **TODO-T6-1** (Core-Guard-Rollback-Verschwendung):
  evidence-instrumentierte A/B-Abnahme ueber die Referenz-Song-Konfigurationen,
  deterministisch.
- `.github/copilot-instructions.md` §G5 (gleicher Input + Version => bit-identisch;
  Seed-Pflicht) und §V7 (copilot-instructions.md) (kein Workaround: die Entscheidung misst die URSACHE —
  ob die schritt-granulare Verwerfung den Wohlklang hebt, ohne Hoer-Ebene-1 zu
  verletzen). Hoerordnung-Ebene 1 (Ueber-Stimmen-Phasen-Invarianten) ist dabei
  harte Schranke: die Variante scheidet aus, wenn eine Verletzung auftritt
  (Hoerordnung §8a-Kalibrierung prueft das separat).

Entscheidungsregel (PRAE-REGISTRIERT, nicht nachtraeglich drehbar):
  Die Variante gewinnt, wenn (a) keine Ebene-1-Verletzung vorliegt, (b) der
  mittlere Score-Gewinn (Variante − Status quo ueber alle Konfigurationen)
  >= `min_delta` betraegt und (c) das 95 %-Bootstrap-CI (Seed-fixiert)
  0 ausschliesst.

Eingabe: Referenzmatrix-CSV (Referenz-Song x Stimmen) mit Spalten
  `config,score_ref,score_var` (Scores aus dem fuer beide Varianten
  IDENTISCHEN Playback-/Mess-Pfad — nur die Schritt-Granularitaet der
  Core-Guards unterscheidet sich).

Beweisschluss: `pytest tests/unit/test_validate_t61_ab.py` plus
`python scripts/validate_t61_ab.py <matrix.csv>` (Exit-Code 0 = Variante
gewinnt, 1 = Variante gewinnt nicht; Report als JSON auf stdout).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 20260919  # §G5 (GEBOTE.md): deterministische Bootstrap-Seeds


@dataclass(frozen=True)
class ABDecision:
    variant_wins: bool
    mean_gain: float
    ci95_low: float
    ci95_high: float
    n_configs: int
    reason: str


def bootstrap_ci95(deltas: Sequence[float], seed: int = SEED, resamples: int = 2000) -> tuple[float, float]:
    """Deterministisches Bootstrap-95%-CI des Mittelwerts (Seed-fixiert, §G5 (GEBOTE.md))."""
    if not deltas:
        raise ValueError("leere Deltas")
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def decide_ab(
    scores_ref: Sequence[float],
    scores_var: Sequence[float],
    min_delta: float = 0.25,
    ebene1_violation: bool = False,
    seed: int = SEED,
) -> ABDecision:
    """Prae-registrierte T6-1-Entscheidung (Variante vs. Status quo)."""
    if len(scores_ref) != len(scores_var) or not scores_ref:
        raise ValueError("gleiche, nicht-leere Konfigurationszahl erforderlich")
    deltas = [v - r for r, v in zip(scores_ref, scores_var)]
    mean_gain = sum(deltas) / len(deltas)
    lo, hi = bootstrap_ci95(deltas, seed=seed)
    if ebene1_violation:
        return ABDecision(
            False,
            mean_gain,
            lo,
            hi,
            len(deltas),
            "Hoerordnung-Ebene-1-Verletzung: Variante disqualifiziert (harte Schranke)",
        )
    if mean_gain < min_delta:
        return ABDecision(
            False, mean_gain, lo, hi, len(deltas), f"mittlerer Gewinn {mean_gain:.3f} < min_delta {min_delta}"
        )
    if lo <= 0.0:
        return ABDecision(False, mean_gain, lo, hi, len(deltas), "95%-CI schliesst 0 nicht aus")
    return ABDecision(
        True, mean_gain, lo, hi, len(deltas), "Variante gewinnt: Gewinn >= min_delta, CI>0, Ebene 1 unverletzt"
    )


def load_matrix(path: Path) -> tuple[list[float], list[float]]:
    ref: list[float] = []
    var: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ref.append(float(row["score_ref"]))
            var.append(float(row["score_var"]))
    return ref, var


def main() -> int:
    ap = argparse.ArgumentParser(description="T6-1 A/B-Abnahme (Variante vs. Status quo)")
    ap.add_argument("matrix", type=Path, help="Referenzmatrix-CSV: config,score_ref,score_var")
    ap.add_argument("--min-delta", type=float, default=0.25)
    ap.add_argument(
        "--ebene1-violation",
        action="store_true",
        help="Ebene-1-Verletzung festgestellt -> Variante hart disqualifiziert",
    )
    args = ap.parse_args()
    ref, var = load_matrix(args.matrix)
    d = decide_ab(ref, var, min_delta=args.min_delta, ebene1_violation=args.ebene1_violation)
    print(json.dumps(asdict(d), indent=2, ensure_ascii=False))
    return 0 if d.variant_wins else 1


if __name__ == "__main__":
    raise SystemExit(main())

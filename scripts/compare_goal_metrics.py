#!/usr/bin/env python3
"""Kanonischer Delta-Vergleich für Analyse-JSONs (nested goals support)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import argparse
import json
from pathlib import Path

DEFAULT_METRICS = [
    "natuerlichkeit",
    "authentizitaet",
    "timbre_authentizitaet",
    "tonal_center",
    "spatial_depth",
    "transient_energie",
    "brillanz",
    "waerme",
    "separation_fidelity",
]


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def _goal_scores(data: dict) -> dict[str, float]:
    goals = data.get("final_musical_goals", {})
    if isinstance(goals, dict) and isinstance(goals.get("scores"), dict):
        goals = goals["scores"]
    if not isinstance(goals, dict):
        return {}

    out: dict[str, float] = {}
    for k, v in goals.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
            continue
    return out


def _metric(data: dict, key: str) -> float:
    goals = _goal_scores(data)
    if key in goals:
        return float(goals[key])

    hpg = data.get("holistic_perceptual_gate") or {}

    if key == "vqi":
        v = (data.get("final_vocal_metrics") or {}).get("vqi", None)
        if v is None:
            v = hpg.get("vqi", 0.0)
        return float(v or 0.0)
    if key == "final_hpi":
        h = (data.get("summary") or {}).get("final_hpi", None)
        if h is None:
            h = hpg.get("hpi", 0.0)
        return float(h or 0.0)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vergleicht Baseline vs. Postpatch Analyse-Metriken.")
    parser.add_argument("baseline", type=Path, help="Pfad zur Baseline-Analyse-JSON")
    parser.add_argument("candidate", type=Path, help="Pfad zur Postpatch/Coreguard-Analyse-JSON")
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join([*DEFAULT_METRICS, "vqi", "final_hpi"]),
        help="Komma-separierte Metrikliste",
    )
    args = parser.parse_args()

    base = _read_json(args.baseline)
    cand = _read_json(args.candidate)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("metric|baseline|postpatch|delta")
    for m in metrics:
        b = _metric(base, m)
        c = _metric(cand, m)
        d = c - b
        print(f"{m}|{b:.6f}|{c:.6f}|{d:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

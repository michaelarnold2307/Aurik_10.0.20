"""Freude-Gate for psychoacoustic release validation.

Evaluates listening-test results (MUSHRA/subjective panel) with hard pass/fail
criteria focused on perceived enjoyment and artifact freedom.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    min_items: int = 20
    min_mean_mushra: float = 80.0
    min_p10_mushra: float = 70.0
    max_share_below_65: float = 0.10
    min_mean_enjoyment: float = 4.20  # 1..5 scale
    max_mean_fatigue: float = 2.20  # 1..5 scale, lower is better
    max_artifact_rate: float = 0.05


@dataclass(frozen=True)
class GateMetrics:
    n_items: int
    mean_mushra: float
    p10_mushra: float
    share_mushra_below_65: float
    mean_enjoyment: float
    mean_fatigue: float
    artifact_rate: float


@dataclass(frozen=True)
class GateCheck:
    name: str
    value: float
    comparator: str
    threshold: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    metrics: GateMetrics
    checks: list[GateCheck]


def _build_recommendations(metrics: GateMetrics, cfg: GateConfig) -> list[str]:
    """Return prioritized, actionable recommendations for failed criteria."""
    recs: list[str] = []

    if metrics.mean_enjoyment < cfg.min_mean_enjoyment:
        recs.append(
            "Enjoyment zu niedrig: Phase-Strength in den Familien transient/dynamics_eq um 5-10% senken, "
            "dann erneut Blindtest fahren."
        )
    if metrics.mean_fatigue > cfg.max_mean_fatigue:
        recs.append(
            "Fatigue zu hoch: Presence-Bereich 3-6 kHz und Air-Anhebung >8 kHz konservativer fahren; "
            "zusätzlich De-Essing-Guard schärfen."
        )
    if metrics.artifact_rate > cfg.max_artifact_rate:
        recs.append(
            "Artefaktrate zu hoch: artefaktfreie Best-Checkpoint-Policy priorisieren und aggressive Rekonstruktion "
            "(phase_23/24/55) bei Unsicherheit früher dämpfen."
        )
    if metrics.p10_mushra < cfg.min_p10_mushra or metrics.share_mushra_below_65 > cfg.max_share_below_65:
        recs.append(
            "Zu viele Ausreißer-Songs: material-/genre-spezifische Profile nachschärfen und nur betroffene Cluster "
            "(Era x Material x Defect) rekalibrieren."
        )
    if metrics.mean_mushra < cfg.min_mean_mushra:
        recs.append(
            "Durchschnitts-MUSHRA zu niedrig: PMGG-/SongCalibration-Scalars in konservative Mitte ziehen und "
            "mit A/B-Hörtest iterieren."
        )

    if not recs:
        recs.append("Keine Nachbesserung erforderlich: Gate-Ziele erreicht.")
    return recs


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    vals = sorted(float(v) for v in values)
    q = max(0.0, min(100.0, float(q)))
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Feld 'items' muss eine Liste sein")
    return [i for i in items if isinstance(i, dict)]


def _as_float(item: dict[str, Any], key: str) -> float:
    if key not in item:
        raise ValueError(f"Item {item.get('item_id', '?')} fehlt Feld '{key}'")
    try:
        return float(item[key])
    except Exception as exc:
        raise ValueError(f"Item {item.get('item_id', '?')} hat ungueltigen Wert fuer '{key}'") from exc


def evaluate(items: list[dict[str, Any]], cfg: GateConfig) -> GateResult:
    if len(items) < cfg.min_items:
        raise ValueError(
            f"Zu wenige Items: {len(items)} < {cfg.min_items}. Das Freude-Gate braucht einen robusten Korpus."
        )

    mushra = [_as_float(i, "mushra") for i in items]
    enjoyment = [_as_float(i, "enjoyment") for i in items]
    fatigue = [_as_float(i, "fatigue") for i in items]
    artifact_flags = [bool(i.get("artifact_flag", False)) for i in items]

    metrics = GateMetrics(
        n_items=len(items),
        mean_mushra=float(mean(mushra)),
        p10_mushra=_percentile(mushra, 10.0),
        share_mushra_below_65=float(sum(1 for v in mushra if v < 65.0) / len(mushra)),
        mean_enjoyment=float(mean(enjoyment)),
        mean_fatigue=float(mean(fatigue)),
        artifact_rate=float(sum(1 for v in artifact_flags if v) / len(artifact_flags)),
    )

    checks = [
        GateCheck(
            name="mean_mushra",
            value=metrics.mean_mushra,
            comparator=">=",
            threshold=cfg.min_mean_mushra,
            passed=metrics.mean_mushra >= cfg.min_mean_mushra,
        ),
        GateCheck(
            name="p10_mushra",
            value=metrics.p10_mushra,
            comparator=">=",
            threshold=cfg.min_p10_mushra,
            passed=metrics.p10_mushra >= cfg.min_p10_mushra,
        ),
        GateCheck(
            name="share_mushra_below_65",
            value=metrics.share_mushra_below_65,
            comparator="<=",
            threshold=cfg.max_share_below_65,
            passed=metrics.share_mushra_below_65 <= cfg.max_share_below_65,
        ),
        GateCheck(
            name="mean_enjoyment",
            value=metrics.mean_enjoyment,
            comparator=">=",
            threshold=cfg.min_mean_enjoyment,
            passed=metrics.mean_enjoyment >= cfg.min_mean_enjoyment,
        ),
        GateCheck(
            name="mean_fatigue",
            value=metrics.mean_fatigue,
            comparator="<=",
            threshold=cfg.max_mean_fatigue,
            passed=metrics.mean_fatigue <= cfg.max_mean_fatigue,
        ),
        GateCheck(
            name="artifact_rate",
            value=metrics.artifact_rate,
            comparator="<=",
            threshold=cfg.max_artifact_rate,
            passed=metrics.artifact_rate <= cfg.max_artifact_rate,
        ),
    ]

    return GateResult(
        passed=all(c.passed for c in checks),
        metrics=metrics,
        checks=checks,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aurik Freude-Gate (psychoakustischer Release-Check)")
    p.add_argument("--input", required=True, help="JSON mit Listening-Panel-Ergebnissen")
    p.add_argument("--output", default="reports/freude_gate_report.json", help="Output-Report (JSON)")
    p.add_argument("--min-items", type=int, default=20)
    p.add_argument("--min-mean-mushra", type=float, default=80.0)
    p.add_argument("--min-p10-mushra", type=float, default=70.0)
    p.add_argument("--max-share-below-65", type=float, default=0.10)
    p.add_argument("--min-mean-enjoyment", type=float, default=4.20)
    p.add_argument("--max-mean-fatigue", type=float, default=2.20)
    p.add_argument("--max-artifact-rate", type=float, default=0.05)
    p.add_argument(
        "--enforce",
        action="store_true",
        help="Wenn gesetzt: FAIL fuehrt zu Exit-Code 1 (blocking mode). Standard: non-blocking Nachbesserungsmodus.",
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    cfg = GateConfig(
        min_items=int(args.min_items),
        min_mean_mushra=float(args.min_mean_mushra),
        min_p10_mushra=float(args.min_p10_mushra),
        max_share_below_65=float(args.max_share_below_65),
        min_mean_enjoyment=float(args.min_mean_enjoyment),
        max_mean_fatigue=float(args.max_mean_fatigue),
        max_artifact_rate=float(args.max_artifact_rate),
    )

    try:
        items = _load_items(in_path)
        result = evaluate(items, cfg)
    except Exception as exc:
        print(f"Freude-Gate FEHLER: {exc}")
        return 2

    recommendations = _build_recommendations(result.metrics, cfg)

    report = {
        "gate": "freude_gate_v1",
        "passed": result.passed,
        "mode": "enforce" if args.enforce else "improve",
        "config": asdict(cfg),
        "metrics": asdict(result.metrics),
        "checks": [asdict(c) for c in result.checks],
        "recommendations": recommendations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    status = "PASS" if result.passed else ("FAIL" if args.enforce else "IMPROVE")
    print(f"Freude-Gate: {status}")
    print(f"Report: {out_path}")
    if result.passed:
        return 0
    return 1 if args.enforce else 0


if __name__ == "__main__":
    raise SystemExit(main())

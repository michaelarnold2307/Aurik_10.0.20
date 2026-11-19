"""Stepwise ONNX export orchestrator for Aurik.

This script orchestrates existing export tooling in small, explicit steps.
It is intentionally non-destructive by default (plan mode).

Usage examples:
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 1 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 2 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 3 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 4 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 5 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 6 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 7 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 8 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 9 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 10 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 11 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 12 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 13 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 14 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 15 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 16 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --step 17 --run
    .venv_aurik/bin/python scripts/onnx_export_stepwise.py --profile core --run --continue-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv_aurik" / "bin" / "python"


@dataclass(frozen=True)
class Step:
    id: int
    title: str
    command: list[str]
    kind: str


def _build_steps(deep: bool) -> list[Step]:
    readiness_cmd = [
        str(PY),
        str(ROOT / "scripts" / "onnx_export_readiness.py"),
    ]
    if deep:
        readiness_cmd.append("--deep")

    return [
        Step(
            id=1,
            title="Readiness-Pruefung (quick/deep)",
            command=readiness_cmd,
            kind="readiness",
        ),
        Step(
            id=2,
            title="Apollo TorchScript Export (stabiler Produktionspfad)",
            command=[str(PY), str(ROOT / "scripts" / "export_apollo_torchscript.py")],
            kind="export",
        ),
        Step(
            id=3,
            title="CQTdiff Score-Network TorchScript Export",
            command=[str(PY), str(ROOT / "scripts" / "export_cqtdiff_onnx.py")],
            kind="export",
        ),
        Step(
            id=4,
            title="UTMOSv2 ONNX Deep-Check (architektur-spezifisch)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "onnx_export_readiness.py"),
                "--deep",
                "--only",
                "utmosv2",
            ],
            kind="readiness-deep",
        ),
        Step(
            id=5,
            title="LAION-CLAP Audio-Encoder ONNX Deep-Check (architektur-spezifisch)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "onnx_export_readiness.py"),
                "--deep",
                "--only",
                "laion_clap",
            ],
            kind="readiness-deep",
        ),
        Step(
            id=6,
            title="UTMOSv2 Audio-Encoder ONNX Export",
            command=[
                str(PY),
                str(ROOT / "scripts" / "export_utmosv2_audio_encoder_onnx.py"),
            ],
            kind="export",
        ),
        Step(
            id=7,
            title="LAION-CLAP Audio-Encoder ONNX Export",
            command=[
                str(PY),
                str(ROOT / "scripts" / "export_laion_clap_audio_encoder_onnx.py"),
            ],
            kind="export",
        ),
        Step(
            id=8,
            title="Manifest-Validierung fuer neue ONNX-Artefakte",
            command=[
                str(PY),
                str(ROOT / "scripts" / "validate_manifest_new_onnx.py"),
            ],
            kind="validate",
        ),
        Step(
            id=9,
            title="Core-Model-Presence-Check (RMVPE/SGMSE+/VERSA/Flow/GACELA)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "validate_core_model_presence.py"),
            ],
            kind="validate",
        ),
        Step(
            id=10,
            title="Core-Model-Layout vorbereiten (nur Verzeichnisse)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "prepare_core_model_layout.py"),
            ],
            kind="prepare",
        ),
        Step(
            id=11,
            title="Core-Model-Source-Check (lokale Kandidaten + Next Actions)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "check_core_model_sources.py"),
            ],
            kind="validate",
        ),
        Step(
            id=12,
            title="Manifest-Sync fuer Core-Modelle (wenn Dateien vorhanden)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "sync_core_models_to_manifest.py"),
            ],
            kind="sync",
        ),
        Step(
            id=13,
            title="Core-Model-Auto-Ingest aus lokalen Drop-In-Verzeichnissen",
            command=[
                str(PY),
                str(ROOT / "scripts" / "auto_ingest_core_models.py"),
            ],
            kind="prepare",
        ),
        Step(
            id=14,
            title="Core-Profile Abschlussreport (Tabelle)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "summarize_core_profile_status.py"),
            ],
            kind="report",
        ),
        Step(
            id=15,
            title="Core-Profile Abschlussreport (JSON)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "summarize_core_profile_status.py"),
                "--json-out",
                "reports/core_profile_status.json",
            ],
            kind="report",
        ),
        Step(
            id=16,
            title="Core-Model-Autofetch aus HuggingFace (best effort, exact filename)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "fetch_core_models_from_hf.py"),
                "--apply",
            ],
            kind="prepare",
        ),
        Step(
            id=17,
            title="Core-Modele aus PyTorch laden und ONNX exportieren (sofern möglich)",
            command=[
                str(PY),
                str(ROOT / "scripts" / "fetch_and_export_core_models.py"),
            ],
            kind="export",
        ),
    ]


def _print_plan(steps: list[Step]) -> None:
    print("=" * 96)
    print("Schrittweiser ONNX/Export-Plan")
    print("=" * 96)
    for step in steps:
        cmd = " ".join(step.command)
        print(f"[{step.id}] {step.title}")
        print(f"    Befehl: {cmd}")


def _run_step(step: Step) -> int:
    print("-" * 96)
    print(f"Starte Schritt {step.id}: {step.title}")
    print("-" * 96)
    proc = subprocess.run(
        step.command,
        cwd=str(ROOT),
        text=True,
    )
    return int(proc.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stepwise ONNX export runner")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        help="Run only one specific step",
    )
    parser.add_argument(
        "--profile",
        choices=["core"],
        help="Run a predefined step chain. 'core' runs: 10 -> 11 -> 16 -> 17 -> 13 -> 9 -> 12 -> 14 -> 15.",
    )
    parser.add_argument("--run", action="store_true", help="Execute steps. Default is plan-only.")
    parser.add_argument("--deep", action="store_true", help="Use deep mode for readiness step")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue with remaining steps even if one step fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not PY.exists():
        print(f"Fehler: Python-Umgebung fehlt: {PY}", file=sys.stderr)
        return 2

    steps = _build_steps(args.deep)

    if args.profile is not None and args.step is not None:
        print("Fehler: --profile und --step können nicht zusammen verwendet werden.", file=sys.stderr)
        return 2

    if args.profile == "core":
        chain = [10, 11, 16, 17, 13, 9, 12, 14, 15]
        steps = [s for s in steps if s.id in chain]
        steps.sort(key=lambda s: chain.index(s.id))

    if args.step is not None:
        steps = [s for s in steps if s.id == args.step]

    _print_plan(steps)

    if not args.run:
        print("\nPlan-Modus aktiv. Fuehre mit --run aus.")
        return 0

    for step in steps:
        rc = _run_step(step)
        if rc != 0:
            if step.id == 9:
                print(
                    "Hinweis: Schritt 9 validiert nur die Präsenz der Core-Modelle.\n"
                    "Bei MISSING ist das erwartbar, bis die Artefakte bereitgestellt sind.\n"
                    "Empfohlener Pfad: Schritt 10 (Layout) -> Schritt 11 (Source-Check) -> Schritt 12 (Manifest-Sync) -> Schritt 13 (Auto-Ingest)."
                )
                if args.continue_on_fail:
                    print("continue-on-fail aktiv: fahre mit den nächsten Schritten fort.")
                    continue
                print("Abbruch nach Schritt 9. Fuer fortgesetzten Lauf: --continue-on-fail verwenden.")
                return rc

            if args.continue_on_fail:
                print(
                    f"Schritt {step.id} fehlgeschlagen (Exit-Code {rc}) — continue-on-fail aktiv, fahre fort.",
                    file=sys.stderr,
                )
                continue
            print(f"Schritt {step.id} fehlgeschlagen (Exit-Code {rc}).", file=sys.stderr)
            return rc

    print("\nAlle ausgewaehlten Schritte abgeschlossen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

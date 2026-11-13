#!/usr/bin/env python3
"""Trainings-Job-Runner — Run-ID + Lock + Auto-Resume (Pfeiler 4 der SOTA-Roadmap).

Roadmap: „Job-Runner mit Auto-Resume statt manuellem Relaunch (Ursache der
Doppel-Läufe); Resume-Befehl steht dann im Manifest." Dazu „Run-ID + Lock für
Trainings-Jobs auf models/<modell>/ (Datei-Lock); Manifest erweitert um
(Befehl, Seed, Phase, Checkpoint-Hash)."

Zwei Lock-Ebenen (beide `flock`, beide auf dem Modell-Verzeichnis):
  * Job-Lock  `.train_job.lock` — dieser Runner; verhindert DOPPEL-LÄUFE.
  * Script-Lock `.train.lock`   — das Trainings-Skript selbst
    (``backend.core.training_artifacts.acquire_run_lock``).
Der Job-Lock hat einen eigenen Namen, damit der Kind-Prozess seinen
Script-Lock ungehindert erwerben kann.

Der Runner startet Prozesse in eigenen Prozessgruppen (``start_new_session``)
und kappt sie bei Timeout sauber (TERM → KILL). Jeder Versuch landet sofort
im Manifest (crash-sicher); der Resume-Befehl steht darin — wie gefordert.

Manifest-Status: ``erfolgreich`` (Ziel erreicht), ``zeitbudget`` (Trainer
stoppte an der Epochen-Grenze, Exit 75 — gesund, aber fortsetzbar) oder
``fehlgeschlagen``. Nur ``erfolgreich`` wird beim Ketten-Neustart uebersprungen.

Aufruf:
    python3 -B scripts/train_job_runner.py --name cantus --phase v2 --seed 42 \\
        --ckpt-dir models/cantus/cantus_training \\
        --cmd "python3 -B scripts/train_cantus.py --phase v2" \\
        --resume-cmd "python3 -B scripts/train_cantus.py --phase v2 --resume ..." \\
        [--timeout-s 21600] [--retries 2] [--manifest pfad.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.training_artifacts import (
    acquire_run_lock,
    make_run_id,
    sha256_file,
)

# Trainer mit Zeitbudget (``--max-seconds``) stoppen SELBST an der Epochen-Grenze
# und melden Exit 75 (``zeitbudget_erreicht``). Das ist kein Modellfehler: der
# Lauf ist gesund und wird am Anker fortgesetzt. Vor D-K3-87 (2026-11-09) stand
# ein solcher Lauf als "fehlgeschlagen" im Manifest — die Fehldiagnose liess
# die D3-Finetunes "gescheitert" aussehen, obwohl sie nur nicht fertig waren.
ZEITBUDGET_EXITCODE = 75

log = logging.getLogger(__name__)

JOB_LOCK_NAME = ".train_job.lock"
CKPT_SUFFIXES = (".pt", ".safetensors", ".onnx", ".ckpt")


def _checkpoint_hashes(ckpt_dir: Path) -> dict[str, str]:
    """Checkpoint-Hashes im Modell-Verzeichnis (Manifest-Provenienz, Pfeiler 4)."""
    out: dict[str, str] = {}
    if not ckpt_dir.is_dir():
        return out
    for f in sorted(ckpt_dir.iterdir()):
        if f.is_file() and f.suffix in CKPT_SUFFIXES:
            out[f.name] = sha256_file(f)
    return out


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_manifest(path: Path) -> dict | None:
    """Letztes Manifest lesen (None, wenn es fehlt oder unlesbar ist)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Manifest %s nicht lesbar (%s) — Job wird neu gestartet.", path, exc)
        return None


def _run_attempt(cmd: list[str], timeout_s: float) -> dict:
    """Ein Versuch in eigener Prozessgruppe; Timeout kappt die ganze Gruppe."""
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except OSError:
                break
            try:
                out, _ = proc.communicate(timeout=20)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            out = ""
        if proc.poll() is None:
            proc.kill()
    return {
        "exit_code": proc.returncode,
        "sekunden": round(time.monotonic() - start, 1),
        "timed_out": timed_out,
        "output_tail": (out or "")[-2000:],
    }


def _trainer_aus_cmd(cmd: list[str]) -> Path | None:
    """Extrahiert den Trainer-Pfad aus einem Job-Befehl (scripts/train_*.py)."""
    for teil in cmd:
        if "scripts/train_" in teil and teil.endswith(".py"):
            p = Path(teil)
            return p if p.is_absolute() else (PROJECT_ROOT / p)
    return None


def run_job(
    *,
    name: str,
    ckpt_dir: Path,
    cmd: list[str],
    resume_cmd: list[str],
    phase: str,
    seed: int | None = None,
    timeout_s: float = 21600.0,
    retries: int = 2,
    manifest_path: Path,
    force: bool = False,
) -> int:
    """Führt den Job aus; liefert Exit-Code (0 = erfolgreich oder übersprungen).

    Idempotente Wiederaufnahme (docs/SOTA_HANDOVER.md §3): Ist das Manifest
    bereits ``ergebnis: erfolgreich`` und stimmen Befehl, Phase und Seed
    überein, wird der Job ohne GPU-Zeit übersprungen. ``force=True`` (CLI:
    ``--force``) erzwingt einen Neulauf, z. B. nach geänderten Trainingsdaten.
    """
    befehl_str = " ".join(cmd)
    # Trainingsstrategie Stufe A (Report
    # docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md):
    # Flugschein-Pflicht für die Mel-Familie — kein GPU-Job ohne
    # Produktions-Domänen-Parität (D-K3-96/§G9 (copilot-instructions.md)).
    trainer = _trainer_aus_cmd(cmd)
    if trainer is not None and trainer.exists():
        from scripts.trainings_flugschein import flugschein_pflicht, flugschein_trainer_pruefen

        if flugschein_pflicht(trainer):
            ok, befund = flugschein_trainer_pruefen(trainer)
            if not ok:
                _write_manifest(
                    manifest_path,
                    {
                        "schema": "aurik.train_job/1",
                        "name": name,
                        "phase": phase,
                        "befehl": befehl_str,
                        "ergebnis": "flugschein_abgelehnt",
                        "begruendung": befund,
                        "quelle": "scripts/trainings_flugschein.py",
                    },
                )
                print(
                    f"[{name}] Flugschein ABGELEHNT: {befund} — kein GPU-Job (Fail-fast, Stufe A).",
                    flush=True,
                )
                return 3
    if not force:
        vorhanden = _load_manifest(manifest_path)
        if vorhanden and vorhanden.get("ergebnis") == "erfolgreich":
            if (
                vorhanden.get("befehl") == befehl_str
                and vorhanden.get("phase") == phase
                and vorhanden.get("seed") == seed
            ):
                print(
                    f"[{name}] bereits erfolgreich (Run {vorhanden.get('run_id')}) — "
                    "übersprungen; --force erzwingt einen Neulauf.",
                    flush=True,
                )
                return 0
            log.warning(
                "Manifest %s ist erfolgreich, aber Befehl/Phase/Seed weichen ab — Neulauf.",
                manifest_path,
            )
    try:
        os.nice(10)  # Host-Schutz (Pfeiler 6): Kinder erben die niedrige Priorität
    except OSError:
        pass
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(phase)
    lock = acquire_run_lock(ckpt_dir, run_id, lock_name=JOB_LOCK_NAME)
    if lock is None:
        log.error("Job-Lock auf %s bereits gehalten — kein Doppel-Lauf (Pfeiler 4).", ckpt_dir)
        return 2

    manifest: dict = {
        "schema": "aurik.train_job/1",
        "run_id": run_id,
        "name": name,
        "phase": phase,
        "seed": seed,
        "befehl": befehl_str,
        "resume_befehl": " ".join(resume_cmd),
        "timeout_s": timeout_s,
        "retries": retries,
        "versuche": [],
        "quelle": "scripts/train_job_runner.py",
    }
    _write_manifest(manifest_path, manifest)

    exit_code = 1
    for versuch in range(1, retries + 2):
        befehl = cmd if versuch == 1 else resume_cmd
        print(f"[{run_id}] Versuch {versuch}/{retries + 1}: {' '.join(befehl)}", flush=True)
        res = _run_attempt(befehl, timeout_s)
        res["versuch"] = versuch
        res["befehl"] = " ".join(befehl)
        manifest["versuche"].append(res)
        exit_code = int(res["exit_code"]) if res["exit_code"] is not None else 1
        _write_manifest(manifest_path, manifest)
        if exit_code == 0:
            break
        if exit_code == ZEITBUDGET_EXITCODE and not res["timed_out"]:
            print(
                f"[{run_id}] Versuch {versuch}: Zeitbudget erreicht (exit=75) — "
                "kein Fehler, Fortsetzung am Epochen-Anker.",
                flush=True,
            )
            continue
        print(
            f"[{run_id}] Versuch {versuch} fehlgeschlagen "
            f"(exit={res['exit_code']}, timeout={res['timed_out']}) — Auto-Resume.",
            flush=True,
        )

    manifest["checkpoint_hashes"] = _checkpoint_hashes(ckpt_dir)
    if exit_code == 0:
        manifest["ergebnis"] = "erfolgreich"
    elif exit_code == ZEITBUDGET_EXITCODE:
        manifest["ergebnis"] = "zeitbudget"
    else:
        manifest["ergebnis"] = "fehlgeschlagen"
    _write_manifest(manifest_path, manifest)
    lock.close()
    print(f"[{run_id}] Ergebnis: {manifest['ergebnis']} — Manifest: {manifest_path}", flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trainings-Job-Runner (Run-ID + Lock + Auto-Resume)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--cmd", required=True, help="Startbefehl (Shell-Syntax, ohne shell auszuführen)")
    ap.add_argument("--resume-cmd", required=True, help="Resume-Befehl (Auto-Resume nach Fehler/Timeout)")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=21600.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="auch bei bereits erfolgreichem Manifest neu trainieren (z. B. neue Daten)",
    )
    args = ap.parse_args(argv)

    manifest_path = args.manifest or (PROJECT_ROOT / "docs" / "reports" / "current" / f"train_job_{args.name}.json")
    return run_job(
        name=args.name,
        ckpt_dir=Path(args.ckpt_dir),
        cmd=shlex.split(args.cmd),
        resume_cmd=shlex.split(args.resume_cmd),
        phase=args.phase,
        seed=args.seed,
        timeout_s=args.timeout_s,
        retries=args.retries,
        manifest_path=Path(manifest_path),
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())

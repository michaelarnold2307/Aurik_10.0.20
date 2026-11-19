"""scripts/ketten_watchdog.py — Überwachung + Auto-Restart der detached Mess-/Trainings-Ketten.

Härtungs-Schicht 4 (2026-11-19, „starke Fortschritte ohne Crashs“): Die Ketten
laufen detached (setsid/nohup, Init-Reparenting) und resume-sicher — aber wenn
eine Kette doch stirbt (SIGSEGV, OOM, Laufzeit-Neustart-Folge), startet sie
bisher niemand neu. Dieser Watchdog schließt die Lücke:

- alle INTERVALL_S wird je Kette geprüft: (a) läuft der Prozess noch?
  (b) steht im Log der END-Marker (sauberer Abschluss)?
- nur wenn KEIN Prozess UND KEIN END-Marker: Auto-Restart detached mit
  derselben Start-Semantik (setsid/nohup, Resume-fähig) + Log-Eintrag.
- Die Muster liegen NICHT in der Kommandozeile des Watchdogs (kein
  pgrep-Self-Match): sie werden aus `KETTEN_CONFIG` (Quelltext) gelesen —
  die cmdline trägt nur „ketten_watchdog.py“.

Der Watchdog selbst startet detached und idempotent (ein zweiter Start endet
sofort, wenn schon einer läuft — Lockfile /tmp/ketten_watchdog.lock mit PID).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

INTERVALL_S = 300
LOCKFILE = Path("/tmp/ketten_watchdog.lock")
LOGFILE = Path("/tmp/ketten_watchdog.log")
_REPO = "/media/michael/Software 4TB/Aurik_Standalone"

# (name, pgrep-muster, start-kommando (bash -lc-String), log-pfad, end-marker)
KETTEN_CONFIG: list[tuple[str, str, str, str, str]] = [
    (
        "signal10-resume",
        "ausgang data/schadens_proben_signal10b",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "\"$PY\" -B scripts/erstelle_schadens_proben.py --verzeichnis data/schadens_quellen "
            "--ausgang data/schadens_proben_signal10b --rezepte clicks,hum,wowflutter --songs 10 --seed 42 && "
            "\"$PY\" -B scripts/bewerte_schadens_proben.py --verzeichnis data/schadens_proben_signal10b && "
            "\"$PY\" -B scripts/erstelle_schadens_proben.py --verzeichnis data/schadens_quellen "
            "--ausgang data/schadens_proben_signal10c --rezepte pitch_drift,dropout,lautheit,amp_drift --songs 10 --seed 42 && "
            "\"$PY\" -B scripts/bewerte_schadens_proben.py --verzeichnis data/schadens_proben_signal10c"
        ),
        "/tmp/signal10_resume.log",
        "SIGNAL10_RESUME_END=",
    ),
    (
        "crackle-kette",
        "erstelle_crackle_expansion.py",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "nice -n 19 \"$PY\" -B scripts/erstelle_crackle_expansion.py --titel 6 --seed 42 && "
            "\"$PY\" -B scripts/erstelle_klassen_paare.py --schaden-dir data/klassen_paare_expand_src "
            "--clean-dir data/klassen_paare_expand_clean --rezepte vinyl --out data/klassen_paare_crackle_expansion; "
            "\"$PY\" -B scripts/haenge_klassen_manifest_an.py --quelle data/klassen_paare_crackle_expansion/manifest.jsonl --klasse crackle"
        ),
        "/tmp/crackle_chain.log",
        "CRACKLE_CHAIN_END=",
    ),
    (
        "vinylnet-retrain",
        "venv_rocm72/bin/python -B train.py",
        (
            "cd '" + _REPO + "/models/de_crackle' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 "
            "PYTHONPATH='" + _REPO + "' && /home/michael/.local/share/aurik/venv_rocm72/bin/python -B train.py "
            "--data ../../data/musdb18hq/train --pattern mixture.wav --preset base "
            "--noise-bank ../../data/vinyl_noise_bank --epochs 100 --gan --gan-start 50000 "
            "--mert ../../models/mert/mert_330m.onnx --mem-fraction 0.6 --resume runs/vinylnet/last.pt"
        ),
        "/tmp/vinylnet_resume.log",
        "VINYLNET_RESUME_END=",
    ),
    (
        "kanalsatz-smoke",
        "messe_kanalsatz_ab.py --verzeichnis data/schadens_proben --paare 4 --seed 42",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "nice -n 19 \"$PY\" -B scripts/messe_kanalsatz_ab.py --verzeichnis data/schadens_proben "
            "--paare 4 --seed 42 --bestand data/schadens_proben/bewertung.json"
        ),
        "/tmp/kanalsatz_ab_smoke.log",
        "KANALSATZ_END=",
    ),
    (
        "witness-runde2",
        "train_witness_kopf.py --klasse",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. AURIK_FORCE_CPU=1 && "
            "PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "for SPEC in dropouts:0.8:0.0:30 dropout_oxide:0.8:0.0:30 tape_splice_artifact:0.8:0.0:30 "
            "wow:0.5:0.0:50 hum:0.5:0.0:50; do K=\"${SPEC%%:*}\"; REST=\"${SPEC#*:}\"; LID=\"${REST%%:*}\"; "
            "REST=\"${REST#*:}\"; LOK=\"${REST%%:*}\"; EP=\"${REST##*:}\"; "
            "nice -n 19 \"$PY\" -B scripts/train_witness_kopf.py --klasse \"$K\" --epochen \"$EP\" "
            "--lambda-mos 0.1 --lambda-id \"$LID\" --lambda-lokal \"$LOK\" --device cpu && "
            "nice -n 19 \"$PY\" -B scripts/verify_witness_kopf_gate.py --klasse \"$K\" --mos; done"
        ),
        "/tmp/wq2_summary.txt",
        "WQ2_END=",
    ),
    (
        "kanalsatz-voll",
        "messe_kanalsatz_ab.py --verzeichnis data/schadens_proben --paare 45",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "while ! grep -q 'KANALSATZ_END=' /tmp/kanalsatz_ab_smoke.log 2>/dev/null; do sleep 120; done; "
            "nice -n 19 \"$PY\" -B scripts/messe_kanalsatz_ab.py --verzeichnis data/schadens_proben "
            "--paare 45 --seed 42 --bestand data/schadens_proben/bewertung.json "
            "--beleg docs/reports/current/$(date +%F)_kanalsatz_ab_voll.json"
        ),
        "/tmp/kanalsatz_ab_voll.log",
        "KANALSATZ_VOLL_END=",
    ),
    (
        "signal10-batch4",
        "ausgang data/schadens_proben_signal10d",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "while ! grep -q 'SIGNAL10_RESUME_END=' /tmp/signal10_resume.log 2>/dev/null; do sleep 120; done; "
            "\"$PY\" -B scripts/erstelle_schadens_proben.py --verzeichnis data/schadens_quellen "
            "--ausgang data/schadens_proben_signal10d --rezepte kompression,verzerrung,vinyl,kassette --songs 10 --seed 42 && "
            "\"$PY\" -B scripts/bewerte_schadens_proben.py --verzeichnis data/schadens_proben_signal10d"
        ),
        "/tmp/signal10_batch4.log",
        "SIGNAL10_B4_END=",
    ),
    (
        "vocal-harshness",
        "erstelle_vocal_harshness_expansion.py",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "while ! grep -q 'CRACKLE_CHAIN_END=' /tmp/crackle_chain.log 2>/dev/null; do sleep 120; done; "
            "\"$PY\" -B scripts/haenge_klassen_manifest_an.py --quelle data/klassen_paare_crackle_expansion/manifest.jsonl --klasse crackle; "
            "nice -n 19 \"$PY\" -B scripts/erstelle_vocal_harshness_expansion.py --titel 20 --seed 42 && "
            "\"$PY\" -B scripts/erstelle_klassen_paare.py --schaden-dir data/vocal_harshness_expand_src "
            "--clean-dir data/vocal_harshness_expand_clean --rezepte clipping,kassette --out data/klassen_paare_vh; "
            "\"$PY\" -B scripts/haenge_klassen_manifest_an.py --quelle data/klassen_paare_vh/manifest.jsonl --klasse vocal_harshness"
        ),
        "/tmp/vh_expansion.log",
        "VH_END=",
    ),
]


def _log(msg: str) -> None:
    with LOGFILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def _prozess_laeuft(muster: str) -> bool:
    try:
        out = subprocess.run(
            ["/usr/bin/pgrep", "-f", muster],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except Exception:
        # Fail-safe: bei unklarer Prozess-Sicht NICHT neu starten (ein
        # Massen-Restart lebender Ketten wäre schädlicher als ein
        # versäumter Restart — nächster Zyklus prüft erneut).
        return True
    if not out:
        return False
    # Self-Match-Schutz: der Watchdog selbst trägt das Muster NICHT in seiner
    # cmdline (Muster kommen aus dem Quelltext) — eigene PID dennoch ausschließen.
    meine = str(os.getpid())
    return any(pid != meine for pid in out.splitlines() if pid.strip())


def _end_marker_da(log_pfad: str, marker: str) -> bool:
    try:
        inhalt = Path(log_pfad).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    return marker in inhalt


def main() -> int:
    # Idempotenz: nur EIN Watchdog.
    if LOCKFILE.is_file():
        try:
            pid = int(LOCKFILE.read_text(encoding="utf-8").strip())
            if Path(f"/proc/{pid}").is_dir():
                _log(f"Watchdog läuft bereits (PID {pid}) — zweiter Start endet.")
                return 0
        except ValueError:
            pass
    LOCKFILE.write_text(str(os.getpid()), encoding="utf-8")

    _log(f"Watchdog gestartet (PID {os.getpid()}, Intervall {INTERVALL_S}s, {len(KETTEN_CONFIG)} Ketten).")
    try:
        while True:
            for name, muster, start_cmd, log_pfad, marker in KETTEN_CONFIG:
                laeuft = _prozess_laeuft(muster)
                fertig = _end_marker_da(log_pfad, marker)
                if not laeuft and not fertig:
                    _log(f"Kette {name} tot (kein Prozess, kein END-Marker) — Auto-Restart detached.")
                    try:
                        # Stdout/Stderr in DAS geprüfte Log, damit der END-Marker
                        # sicher landet (sonst Endlosschleife fertiger Ketten).
                        # STATUS fängt den ECHTEN Exit-Status der Kette ein — der
                        # Marker trägt ihn, damit ein stiller Fehlschlag (train.py
                        # Exit 1, Produktionsbefund 2026-11-19) sichtbar bleibt.
                        cmd = f"{start_cmd} >> {log_pfad} 2>&1; STATUS=$?; echo '{marker}'$STATUS >> {log_pfad}"
                        subprocess.Popen(
                            ["setsid", "nohup", "bash", "-lc", cmd],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL,
                            cwd=_REPO,
                            start_new_session=True,
                        )
                    except Exception as exc:  # pragma: no cover
                        _log(f"Restart von {name} fehlgeschlagen: {exc}")
                elif laeuft and fertig:
                    # End-Marker da, aber Prozess noch da: Abschluss-Phase.
                    pass
            time.sleep(INTERVALL_S)
    finally:
        try:
            LOCKFILE.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

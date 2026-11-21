"""scripts/ketten_watchdog.py — Überwachung + Auto-Restart der detached Mess-/Trainings-Ketten.

Härtungs-Schicht 4 (2026-11-19, „starke Fortschritte ohne Crashs“): Die Ketten
laufen detached (setsid/nohup, Init-Reparenting) und resume-sicher — aber wenn
eine Kette doch stirbt (SIGSEGV, OOM, Laufzeit-Neustart-Folge), startet sie
bisher niemand neu. Dieser Watchdog schließt die Lücke:

- alle INTERVALL_S wird je Kette geprüft: (a) läuft der Prozess noch?
  (b) welcher Exit-Status steht im Log an der letzten Marker-Zeile?
- Status 0 (sauberer Abschluss) ⇒ kein Neustart. Kein Marker (nie gelaufen/
  getötet) ODER Status ≠ 0 (Absturz) ⇒ Auto-Restart detached mit derselben
  Start-Semantik (setsid/nohup, Resume-fähig) + Log-Eintrag.
- Die Muster liegen NICHT in der Kommandozeile des Watchdogs (kein
  pgrep-Self-Match): sie werden aus `KETTEN_CONFIG` (Quelltext) gelesen —
  die cmdline trägt nur „ketten_watchdog.py“.
- Härtungs-Schicht 4b (2026-11-19, Produktionsbefund): der ursprüngliche
  Abschluss-Test prüfte nur die Text-Anwesenheit des Markers — ein Absturz
  (`…_END=139`, SIGSEGV) trug den Marker GENAUSO wie ein sauberer Abschluss
  (`…_END=0`) und wurde deshalb NIE neu gestartet, obwohl der echte Status
  schon geloggt wurde (Ursache statt Symptom, §V7 (copilot-instructions.md)).
  `_end_status` parst den Status; Abstürze werden erneut versucht, begrenzt
  durch `MAX_CRASH_RESTARTS` (keine Endlosschleife bei dauerhaft kaputtem
  Kommando — dann Log-Warnung statt stillem Dauer-Neustart).
- Abhängige Ketten benötigen einen erfolgreichen Abschluss des aktuellen
  Versuchs. Ein Startmarker entwertet ältere Abschlüsse; bei bereits laufenden
  Alt-Prozessen ohne Startmarker wird auf einen neuen Abschluss gewartet.
  Ist ein Abschluss nicht eindeutig zuordenbar, bleibt die Kette mit Warnung
  stehen, statt einen alten Erfolg als Nachweis zu verwenden.

Der Watchdog selbst startet detached und idempotent (ein zweiter Start endet
sofort, wenn schon einer läuft — Lockfile /tmp/ketten_watchdog.lock mit PID).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

INTERVALL_S = 300
LOCKFILE = Path("/tmp/ketten_watchdog.lock")
LOGFILE = Path("/tmp/ketten_watchdog.log")
_REPO = "/media/michael/Software 4TB/Aurik_Standalone"


def _wait_for_success(name: str) -> str:
    python = shlex.quote(sys.executable)
    script = shlex.quote(str(Path(__file__).resolve()))
    return f"{python} -B {script} --wait-for {shlex.quote(name)}"


def _is_watchdog_manager_cmdline(argv: list[str]) -> bool:
    """Unterscheidet den Manager von Waiter-Prozessen und wiederverwendeten PIDs."""
    return str(Path(__file__).resolve()) in argv and "--wait-for" not in argv


def _watchdog_manager_alive() -> bool:
    """Prüft, ob der Lock-PID wirklich diesen Manager ausführt."""
    try:
        pid = int(LOCKFILE.read_text(encoding="utf-8").strip())
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    argv = [os.fsdecode(arg) for arg in raw_cmdline.split(b"\0") if arg]
    return _is_watchdog_manager_cmdline(argv)


# (name, pgrep-muster, start-kommando (bash -lc-String), log-pfad, end-marker)
KETTEN_CONFIG: list[tuple[str, str, str, str, str]] = [
    (
        "signal10-resume",
        "ausgang data/schadens_proben_signal10b",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='" + _REPO + "/.venv_aurik/bin/python' && "
            '"$PY" -B scripts/erstelle_schadens_proben.py --verzeichnis data/schadens_quellen '
            "--ausgang data/schadens_proben_signal10b --rezepte clicks,hum,wowflutter --songs 10 --seed 42 && "
            '"$PY" -B scripts/bewerte_schadens_proben.py --verzeichnis data/schadens_proben_signal10b && '
            '"$PY" -B scripts/erstelle_schadens_proben.py --verzeichnis data/schadens_quellen '
            "--ausgang data/schadens_proben_signal10c --rezepte pitch_drift,dropout,lautheit,amp_drift --songs 10 --seed 42 && "
            '"$PY" -B scripts/bewerte_schadens_proben.py --verzeichnis data/schadens_proben_signal10c'
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
            'nice -n 19 "$PY" -B scripts/erstelle_crackle_expansion.py --titel 6 --seed 42 && '
            '"$PY" -B scripts/erstelle_klassen_paare.py --schaden-dir data/klassen_paare_expand_src '
            "--clean-dir data/klassen_paare_expand_clean --rezepte vinyl --out data/klassen_paare_crackle_expansion && "
            '"$PY" -B scripts/haenge_klassen_manifest_an.py --quelle data/klassen_paare_crackle_expansion/manifest.jsonl --klasse crackle'
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
            'nice -n 19 "$PY" -B scripts/messe_kanalsatz_ab.py --verzeichnis data/schadens_proben '
            "--paare 4 --seed 42 --bestand data/schadens_proben/bewertung.json"
        ),
        "/tmp/kanalsatz_ab_smoke.log",
        "KANALSATZ_END=",
    ),
    (
        "witness-runde2",
        "dropouts:0.8:0.0:30",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. AURIK_FORCE_CPU=1 && "
            "PY='" + _REPO + "/.venv_aurik/bin/python' && "
            "for SPEC in dropouts:0.8:0.0:30 dropout_oxide:0.8:0.0:30 tape_splice_artifact:0.8:0.0:30 "
            'wow:0.5:0.0:50 hum:0.5:0.0:50; do K="${SPEC%%:*}"; REST="${SPEC#*:}"; LID="${REST%%:*}"; '
            'REST="${REST#*:}"; LOK="${REST%%:*}"; EP="${REST##*:}"; '
            'nice -n 19 "$PY" -B scripts/train_witness_kopf.py --klasse "$K" --epochen "$EP" '
            '--lambda-mos 0.1 --lambda-id "$LID" --lambda-lokal "$LOK" --device cpu && '
            '( nice -n 19 "$PY" -B scripts/verify_witness_kopf_gate.py --klasse "$K" --mos || '
            'echo "WQ_GATE_FAILED:$K" ); done'
        ),
        "/tmp/wq2_summary.txt",
        "WQ2_END=",
    ),
    (
        "kanalsatz-voll",
        "messe_kanalsatz_ab.py --verzeichnis data/schadens_proben --paare 45",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='"
            + _REPO
            + "/.venv_aurik/bin/python' && "
            + _wait_for_success("kanalsatz-smoke")
            + " && "
            'nice -n 19 "$PY" -B scripts/messe_kanalsatz_ab.py --verzeichnis data/schadens_proben '
            "--paare 45 --seed 42 --bestand data/schadens_proben/bewertung.json "
            "--resume"
        ),
        "/tmp/kanalsatz_ab_voll.log",
        "KANALSATZ_VOLL_END=",
    ),
    (
        "signal10-batch4",
        "ausgang data/schadens_proben_signal10d",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='"
            + _REPO
            + "/.venv_aurik/bin/python' && "
            + _wait_for_success("signal10-resume")
            + " && "
            '"$PY" -B scripts/erstelle_schadens_proben.py --verzeichnis data/schadens_quellen '
            "--ausgang data/schadens_proben_signal10d --rezepte kompression,verzerrung,vinyl,kassette --songs 10 --seed 42 && "
            '"$PY" -B scripts/bewerte_schadens_proben.py --verzeichnis data/schadens_proben_signal10d'
        ),
        "/tmp/signal10_batch4.log",
        "SIGNAL10_B4_END=",
    ),
    (
        "vocal-harshness",
        "erstelle_vocal_harshness_expansion.py",
        (
            "cd '" + _REPO + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. "
            "AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 && PY='"
            + _REPO
            + "/.venv_aurik/bin/python' && "
            + _wait_for_success("crackle-kette")
            + " && "
            '"$PY" -B scripts/haenge_klassen_manifest_an.py --quelle data/klassen_paare_crackle_expansion/manifest.jsonl --klasse crackle && '
            'nice -n 19 "$PY" -B scripts/erstelle_vocal_harshness_expansion.py --titel 20 --seed 42 && '
            '"$PY" -B scripts/erstelle_klassen_paare.py --schaden-dir data/vocal_harshness_expand_src '
            "--clean-dir data/vocal_harshness_expand_clean --rezepte clipping,kassette --out data/klassen_paare_vh && "
            '"$PY" -B scripts/haenge_klassen_manifest_an.py --quelle data/klassen_paare_vh/manifest.jsonl --klasse vocal_harshness'
        ),
        "/tmp/vh_expansion.log",
        "VH_END=",
    ),
    (
        "witness-runde3",
        "dropouts:0.8:30",
        (
            "cd '"
            + _REPO
            + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. AURIK_FORCE_CPU=1 && PY='"
            + _REPO
            + "/.venv_aurik/bin/python' && "
            + _wait_for_success("kanalsatz-voll")
            + " && "
            'for SPEC in dropouts:0.8:30 dropout_oxide:0.8:30 tape_splice_artifact:0.8:30 wow:0.5:50 hum:0.5:50; do K="${SPEC%%:*}"; REST="${SPEC#*:}"; LID="${REST%%:*}"; REST="${REST#*:}"; EP="${REST##*:}"; '
            'nice -n 19 "$PY" -B scripts/train_witness_kopf.py --klasse "$K" --epochen "$EP" '
            '--lambda-mos 0.1 --lambda-id "$LID" --nw-min 0.9 --device cpu && '
            '( nice -n 19 "$PY" -B scripts/verify_witness_kopf_gate.py --klasse "$K" --mos || '
            'echo "WQ_GATE_FAILED:$K" ); done'
        ),
        "/tmp/wq3_summary.txt",
        "WQ3_END=",
    ),
    (
        "witness-runde4",
        "dropouts:0.9:30",
        (
            "cd '"
            + _REPO
            + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. AURIK_FORCE_CPU=1 && PY='"
            + _REPO
            + "/.venv_aurik/bin/python' && "
            + _wait_for_success("witness-runde3")
            + " && "
            'for SPEC in dropouts:0.9:30 wow:0.9:50 hum:0.8:50; do K="${SPEC%%:*}"; REST="${SPEC#*:}"; LID="${REST%%:*}"; REST="${REST#*:}"; EP="${REST##*:}"; '
            'nice -n 19 "$PY" -B scripts/train_witness_kopf.py --klasse "$K" --epochen "$EP" '
            '--lambda-mos 0.1 --lambda-id "$LID" --nw-min 0.9 --device cpu && '
            '( nice -n 19 "$PY" -B scripts/verify_witness_kopf_gate.py --klasse "$K" --mos || '
            'echo "WQ_GATE_FAILED:$K" ); done'
        ),
        "/tmp/wq4_summary.txt",
        "WQ4_END=",
    ),
    (
        "genre-labels",
        "erstelle_genre_labels.py",
        (
            "cd '"
            + _REPO
            + "' && export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. AURIK_FORCE_CPU=1 && PY='"
            + _REPO
            + "/.venv_aurik/bin/python' && "
            + _wait_for_success("kanalsatz-voll")
            + " && "
            'nice -n 19 "$PY" -B scripts/erstelle_genre_labels.py'
        ),
        "/tmp/genre_labels.log",
        "GENRE_LABELS_END=",
    ),
]


def _log(msg: str) -> None:
    with LOGFILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def _prozess_laeuft(muster: str) -> bool:
    try:
        result = subprocess.run(
            ["/usr/bin/pgrep", "-f", muster],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Fail-safe: bei unklarer Prozess-Sicht NICHT neu starten (ein
        # Massen-Restart lebender Ketten wäre schädlicher als ein
        # versäumter Restart — nächster Zyklus prüft erneut).
        logger.warning("Prozessprüfung für %s fehlgeschlagen: %s", muster, exc)
        return True
    if result.returncode not in (0, 1):
        logger.warning("Prozessprüfung für %s fehlgeschlagen: %s", muster, result.stderr.strip())
        return True
    out = result.stdout.strip()
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


# Produktionsbefund 2026-11-19 (D-K3-108-Nachfolge): `_end_marker_da` prüft nur
# die Text-Anwesenheit des Markers — ein Absturz (z. B. `SIGNAL10_RESUME_END=139`,
# SIGSEGV) trägt den Marker GENAUSO wie ein sauberer Abschluss (`…_END=0`) und
# wurde deshalb NIE neu gestartet (§V7 (copilot-instructions.md): Ursache statt
# Symptom — der Status wurde schon geloggt, aber nie ausgewertet). `_end_status`
# liefert den geparsten Exit-Status der ZULETZT geloggten Markerzeile; `main()`
# startet bei Status ≠ 0 erneut, begrenzt durch `MAX_CRASH_RESTARTS` (keine
# Endlosschleife bei einem dauerhaft kaputten Kommando).
_END_STATUS_RE = re.compile(r"(-?\d+)")
MAX_CRASH_RESTARTS = 5


def _start_marker(marker: str) -> str:
    return f"AURIK_WATCHDOG_START:{marker}"


def _end_verlauf(log_pfad: str, marker: str) -> tuple[int | None, int]:
    try:
        inhalt = Path(log_pfad).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, 0
    status = None
    abschluesse = 0
    muster = rf"{re.escape(_start_marker(marker))}|{re.escape(marker)}(-?\d+)"
    for ereignis in re.finditer(muster, inhalt):
        if ereignis.group(1) is None:
            status = None
        else:
            status = int(ereignis.group(1))
            abschluesse += 1
    return status, abschluesse


def _end_status(log_pfad: str, marker: str) -> int | None:
    """Status des letzten Versuchs; ein neuer Start verwirft ältere Abschlüsse."""
    return _end_verlauf(log_pfad, marker)[0]


def _warte_auf_vorgaenger(name: str) -> None:
    vorgaenger = next((kette for kette in KETTEN_CONFIG if kette[0] == name), None)
    if vorgaenger is None:
        raise ValueError(f"Unbekannte Vorgängerkette: {name}")
    _, muster, _, log_pfad, marker = vorgaenger
    abschluesse_bei_laufbeginn = None
    while True:
        status, abschluesse = _end_verlauf(log_pfad, marker)
        laeuft = _prozess_laeuft(muster)
        if laeuft and abschluesse_bei_laufbeginn is None:
            abschluesse_bei_laufbeginn = abschluesse
            if status == 0:
                logger.warning(
                    "Kette %s läuft trotz vorhandenem Erfolgsmarker; warte auf einen neuen Abschluss.",
                    name,
                )
        if (
            not laeuft
            and status == 0
            and (abschluesse_bei_laufbeginn is None or abschluesse > abschluesse_bei_laufbeginn)
        ):
            return
        if not laeuft and status is not None and status != 0:
            versuche = _anzahl_abgestuerzter_versuche(log_pfad, marker)
            if versuche >= MAX_CRASH_RESTARTS:
                logger.error(
                    "Vorgängerkette %s nach %d Fehlversuchen endgültig fehlgeschlagen; "
                    "abhängige Kette wird nicht endlos blockiert.",
                    name,
                    versuche,
                )
                raise SystemExit(1)
        if not laeuft and not _watchdog_manager_alive():
            logger.error(
                "Vorgängerkette %s läuft nicht und der Watchdog-Manager ist nicht aktiv; "
                "abhängige Kette wird abgebrochen.",
                name,
            )
            raise SystemExit(1)
        time.sleep(120)


def _anzahl_abgestuerzter_versuche(log_pfad: str, marker: str) -> int:
    """Zählt Fehlversuche seit dem zuletzt erfolgreich abgeschlossenen Lauf."""
    try:
        inhalt = Path(log_pfad).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return 0
    n = 0
    for teil in inhalt.split(marker)[1:]:
        m = _END_STATUS_RE.match(teil)
        if m:
            if int(m.group(1)) == 0:
                n = 0
            else:
                n += 1
    return n


def main() -> int:
    # Idempotenz: nur EIN Watchdog.
    if LOCKFILE.is_file() and _watchdog_manager_alive():
        _log(f"Watchdog läuft bereits (PID {LOCKFILE.read_text(encoding='utf-8').strip()}) — zweiter Start endet.")
        return 0
    LOCKFILE.write_text(str(os.getpid()), encoding="utf-8")

    _log(f"Watchdog gestartet (PID {os.getpid()}, Intervall {INTERVALL_S}s, {len(KETTEN_CONFIG)} Ketten).")
    try:
        while True:
            for name, muster, start_cmd, log_pfad, marker in KETTEN_CONFIG:
                laeuft = _prozess_laeuft(muster)
                status = _end_status(log_pfad, marker)
                abgestuerzt = status is not None and status != 0
                soll_neustarten = not laeuft and (status is None or abgestuerzt)
                if soll_neustarten and abgestuerzt:
                    versuche = _anzahl_abgestuerzter_versuche(log_pfad, marker)
                    if versuche >= MAX_CRASH_RESTARTS:
                        _log(
                            f"Kette {name} wiederholt abgestürzt (zuletzt Status {status}, "
                            f"{versuche}× seit letztem Erfolg) — Auto-Restart GESTOPPT "
                            f"(MAX_CRASH_RESTARTS={MAX_CRASH_RESTARTS} erreicht, menschliche Prüfung nötig)."
                        )
                        soll_neustarten = False
                if soll_neustarten:
                    grund = "kein Prozess, kein END-Marker" if status is None else f"abgestürzt, Status {status}"
                    _log(f"Kette {name} tot ({grund}) — Auto-Restart detached.")
                    try:
                        with Path(log_pfad).open("a", encoding="utf-8") as log:
                            log.write(f"\n{_start_marker(marker)}\n")
                        # Stdout/Stderr in DAS geprüfte Log, damit der END-Marker
                        # sicher landet (sonst Endlosschleife fertiger Ketten).
                        # STATUS fängt den ECHTEN Exit-Status der Kette ein — der
                        # Marker trägt ihn, damit ein stiller Fehlschlag (train.py
                        # Exit 1, Produktionsbefund 2026-11-19) sichtbar bleibt und
                        # ein Absturz (Status ≠ 0, z. B. SIGSEGV=139) automatisch
                        # erneut versucht wird, statt dauerhaft als „fertig“ zu gelten.
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
                elif laeuft and status is not None:
                    # End-Marker da, aber Prozess noch da: Abschluss-Phase.
                    pass
            time.sleep(INTERVALL_S)
    finally:
        try:
            LOCKFILE.unlink()
        except FileNotFoundError as exc:
            logger.debug("Lock-Datei war beim Aufräumen bereits entfernt: %s", exc)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", metavar="KETTE")
    args = parser.parse_args()
    if args.wait_for:
        _warte_auf_vorgaenger(args.wait_for)
        raise SystemExit(0)
    raise SystemExit(main())

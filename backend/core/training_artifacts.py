"""backend/core/training_artifacts.py — Schutz für Trainingsartefakte.

Befund 2026-10-06: Ein `--smoke`-Validierungslauf überschrieb die
Cantus-Pretrain-Checkpoints (2,45 GB → 4,7 MB) samt Train-Report. GPU-
Trainingsstunden gingen verloren, weil `torch.save` Produktionsdateien still
überschreibt. Dieses Modul verhindert jeden stillen Datenverlust:

  1. **Backup vor Überschreiben** — rotierend, zeitgestempelt, mit Retention.
  2. **Atomares Schreiben** (tmp + ``os.replace``) — keine partiellen Dateien
     bei Absturz/Stromausfall (Muster wie ``recovery_checkpoint.py``).
  3. **Schrumpf-Warnung** (§V6 (copilot-instructions.md)) — eine starke Verkleinerung
     (Preset-/Modellwechsel, Smoke auf Vollmodell) wird sichtbar statt lautlos
     zu passieren.

Adoption in Trainingsskripten: ``torch.save(payload, path)`` ersetzen durch
``save_guarded(path, payload)``; ``path.write_text(...)`` durch
``write_text_guarded(path, text)``.

Beispiel::

    from backend.core.training_artifacts import save_guarded, write_text_guarded

    save_guarded(ckpt_dir / "checkpoint_latest.pt", payload)
    write_text_guarded(report_path, json.dumps(report, indent=2))
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_SUBDIR: str = ".checkpoint_backups"
DEFAULT_KEEP: int = 3
DEFAULT_SHRINK_WARN_RATIO: float = 0.5


def backup_existing(
    path: Path | str,
    *,
    backup_dir: Path | str | None = None,
    keep: int = DEFAULT_KEEP,
) -> Path | None:
    """Sichert eine vorhandene Datei in ein rotierendes Backup.

    Args:
        path: Zu sichernde Datei (Original bleibt unverändert erhalten).
        backup_dir: Zielverzeichnis; Standard: ``<parent>/.checkpoint_backups``.
        keep: Anzahl der aufzubewahrenden Backups (0 = keine Rotation/Löschung).

    Returns:
        Pfad des Backups oder ``None``, wenn keine Quelldatei existierte.
    """
    src = Path(path)
    if not src.exists():
        return None
    target_dir = Path(backup_dir) if backup_dir is not None else src.parent / DEFAULT_BACKUP_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = target_dir / f"{src.name}.{stamp}.bak"
    counter = 0
    while backup.exists():
        counter += 1
        backup = target_dir / f"{src.name}.{stamp}.{counter}.bak"

    shutil.copy2(src, backup)
    _rotate(target_dir, src.name, keep)
    return backup


def _rotate(backup_dir: Path, stem: str, keep: int) -> None:
    """Entfernt ältere Backups, sodass höchstens ``keep`` verbleiben."""
    backups = sorted(backup_dir.glob(f"{stem}.*.bak"))
    if keep <= 0:
        return
    for old in backups[:-keep]:
        try:
            old.unlink()
        except OSError as exc:  # pragma: no cover - nur bei exotischen FS-Rechten
            logger.warning("Backup-Rotation: %s konnte nicht entfernt werden (%s)", old, exc)


def _warn_on_shrink(path: Path, prev_size: int, new_size: int, backup: Path | None, ratio: float) -> None:
    if prev_size <= 0:
        return
    if new_size < prev_size * ratio:
        logger.warning(
            "Trainingsartefakt schrumpft stark: %s %.2f MB → %.2f MB (Backup: %s). "
            "Preset-/Modellwechsel oder Smoke-Lauf prüfen (§V6 (copilot-instructions.md)).",
            path.name,
            prev_size / 1e6,
            new_size / 1e6,
            backup,
        )


def save_guarded(
    path: Path | str,
    payload: Any,
    *,
    backup_dir: Path | str | None = None,
    keep: int = DEFAULT_KEEP,
    shrink_warn_ratio: float = DEFAULT_SHRINK_WARN_RATIO,
) -> Path | None:
    """Speichert ``payload`` atomar und sichert eine vorhandene Datei vorher.

    Vor jedem Überschreiben wird ein rotierendes Backup angelegt; geschrieben
    wird in eine ``.tmp``-Datei, die erst nach Erfolg atomar ersetzt wird.

    Returns:
        Pfad des angelegten Backups oder ``None`` (keine Vorversion vorhanden).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev_size = dest.stat().st_size if dest.exists() else 0

    backup = backup_existing(dest, backup_dir=backup_dir, keep=keep)

    tmp = dest.with_name(dest.name + ".tmp")
    try:
        torch.save(payload, tmp)
        _warn_on_shrink(dest, prev_size, tmp.stat().st_size, backup, shrink_warn_ratio)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return backup


def write_text_guarded(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    backup_dir: Path | str | None = None,
    keep: int = DEFAULT_KEEP,
) -> Path | None:
    """Schreibt Text atomar und sichert eine vorhandene Datei vorher.

    Für Train-Reports/Manifeste, die nicht über ``torch.save`` laufen.

    Returns:
        Pfad des angelegten Backups oder ``None`` (keine Vorversion vorhanden).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    backup = backup_existing(dest, backup_dir=backup_dir, keep=keep)

    tmp = dest.with_name(dest.name + ".tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return backup


# ── Run-ID + Single-Writer-Lock (Pfeiler 4) ─────────────────────────────────
# Kanonische Heimat für alle Trainings-Skripte und den Job-Runner
# (scripts/train_job_runner.py). Ehemals in scripts/train_cantus.py — als
# gemeinsame Quelle hierher verschoben, damit jeder Trainings-Job dieselbe
# Provenienz- und Lock-Disziplin bekommt (Layout-EINE-Quelle).
# Zitiert: §G9 (copilot-instructions.md).


def make_run_id(phase: str) -> str:
    """Eindeutige Run-ID (Provenienz-Metadaten; kein Entscheidungs-Datum)."""
    import random as _random
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{phase}-{stamp}-{_random.randrange(16**6):06x}"


def sha256_file(path: str | Path) -> str:
    """SHA-256 einer Datei (Checkpoint-Identität, Manifest-Provenienz)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire_run_lock(ckpt_dir: Path, run_id: str, lock_name: str = ".train.lock"):
    """Single-Writer-Lock auf dem Checkpoint-Verzeichnis (Pfeiler 4).

    Ein paralleler Lauf ist ausgeschlossen (``flock``). Der Lock-Inhalt
    dokumentiert den Halter (run_id, pid, Befehl). Bei Konflikt: Fehler-Log
    mit Halter + ``None`` — der Aufrufer MUSS ``None`` behandeln (vertraglich
    gepinnt in ``tests/unit/test_training_artifacts.py``).
    """
    import fcntl
    import json as _json
    import sys as _sys

    lock_path = ckpt_dir / lock_name
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        holder = fh.read().strip()[:300]
        fh.close()
        logger.error(
            "Trainings-Lock bereits gehalten von: %s — kein paralleler Lauf (Pfeiler 4, Single-Writer).",
            holder or "unbekannt",
        )
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(
        _json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "command": " ".join(_sys.argv),
            },
            ensure_ascii=False,
        )
    )
    fh.flush()
    return fh


def system_mem_available_gb() -> float:
    """Verfügbarer Systemspeicher (Linux, /proc/meminfo) in GiB (Host-Schutz)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0**2)
    except OSError:
        return 0.0
    return 0.0


# ── Zeitbudget + Resume-feste Report-Historie (D-K3-87, 2026-11-09) ─────────
# Befund: 30-Epochen-Finetunes (BigVGAN ~74 min/Epoche, FlashSR ~80 min/Epoche)
# passen nicht in 2×6 h. Der harte Runner-Timeout killte mitten in der Epoche
# (bis zu 80 min Fortschritt verloren) und das Manifest meldete den gesunden
# Teilfortschritt als „fehlgeschlagen". Diese Helfer beenden den Lauf an der
# Epochen-Grenze und führen die Beleg-Historie über Resume-Grenzen zusammen.


def zeitbudget_erreicht(
    *,
    verstrichen_s: float,
    letzte_epoche_s: float,
    max_seconds: float,
) -> bool:
    """True, wenn die nächste Epoche das Zeitbudget überschreiten würde.

    Die Schätzung nutzt die zuletzt gemessene Epochen-Dauer. Ohne Messung
    (erste Epoche) oder bei ``max_seconds <= 0`` (unbegrenzt) wird nie
    abgebrochen — der Lauf endet dann nur am Ziel oder per Early-Stop.
    """
    if max_seconds <= 0.0 or letzte_epoche_s <= 0.0:
        return False
    return (verstrichen_s + letzte_epoche_s) > max_seconds


def report_historie_laden(out_dir: Path | str, *, seed: int) -> list[dict[str, Any]]:
    """Liest die Epochen-Historie eines früheren Versuchs (Resume-Verkettung).

    Der Report ist das Beleg-Artefakt des Laufs. Ohne diese Zusammenführung
    überschreibt jeder Resume die Historie; der finale Report zeigt dann nur
    die letzte Versuchs-Spanne (Befund 2026-11-09: der FlashSR-Report enthielt
    nur die Epochen 8–10 von 30). Fremde Seeds werden verworfen (§G5 (copilot-instructions.md)).
    """
    pfad = Path(out_dir) / "report.json"
    if not pfad.is_file():
        return []
    try:
        daten = json.loads(pfad.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Trainings-Report unlesbar (%s) — Historie wird neu begonnen (§V6 (copilot-instructions.md)).", exc
        )
        return []
    try:
        report_seed = int(daten.get("seed", -1))
    except (TypeError, ValueError):
        report_seed = -1
    if report_seed != int(seed):
        logger.warning(
            "Trainings-Report gehört zu Seed %s (erwartet %s) — Historie wird nicht übernommen.",
            daten.get("seed"),
            seed,
        )
        return []
    historie = daten.get("history")
    if not isinstance(historie, list):
        return []
    return [eintrag for eintrag in historie if isinstance(eintrag, dict)]


def historie_zusammenfuehren(*historien: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Führt Epochen-Historien zusammen: eine Zeile je Epoche, aufsteigend.

    Bei doppelten Epochen gewinnt der später übergebene Eintrag (der jüngste
    Versuch), damit ein erneut ausgeführter Resume-Stand sichtbar bleibt.
    """
    pro_epoche: dict[int, dict[str, Any]] = {}
    for historie in historien:
        for eintrag in historie:
            if not isinstance(eintrag, dict) or "epoch" not in eintrag:
                continue
            try:
                epoche = int(eintrag["epoch"])
            except (TypeError, ValueError):
                continue
            pro_epoche[epoche] = eintrag
    return [pro_epoche[epoche] for epoche in sorted(pro_epoche)]


def akkumulations_schritt_faellig(*, mikro_index: int, grad_accum: int, n_mikro: int) -> bool:
    """True, wenn nach diesem Mikro-Batch ein Optimizer-Schritt fällig ist.

    Gradient-Akkumulation (D-K3-88): Die Mikro-Batches eines Fensters werden
    aufsummiert; gesteppt wird am Fensterende UND am Epochenende (Rest-Fenster
    geht nie verloren). ``grad_accum <= 1`` bedeutet: jeder Mikro-Batch ist ein
    Schritt (Bestand).
    """
    _accum = max(1, int(grad_accum))
    if _accum <= 1:
        return True
    if mikro_index < 0 or n_mikro <= 0:
        return False
    return (mikro_index + 1) % _accum == 0 or (mikro_index + 1) >= int(n_mikro)

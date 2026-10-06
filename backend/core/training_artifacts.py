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
